"""Orchestration utilities for training probes and exporting attribution maps."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import sys

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "model"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from gradcam_1d import compute_gradcam
from saliency_ig import SaliencyOutputs, compute_all_attributions

# ---------------------------------------------------------------------------
# Optional imports from sibling modules.  The legacy codebase does not ship a
# proper Python package, therefore we dynamically extend ``sys.path`` to reuse
# the existing model and dataset definitions.
# ---------------------------------------------------------------------------

import config  # type: ignore  # noqa: E402
import models  # type: ignore  # noqa: E402
from dataset import EventDatasetSupervised  # type: ignore  # noqa: E402


@dataclass
class XAIConfig:
    """Configuration parameters for attribution export."""

    checkpoint_path: Path
    probe_path: Optional[Path]
    modality: str = "sleep_stages"
    batch_size: int = 16
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ig_steps: int = 50
    smoothgrad_samples: int = 25
    smoothgrad_noise: float = 0.1
    deletion_insertion_steps: int = 20
    baseline_value: float = 0.0
    output_dir: Path = Path("xai_outputs")


@dataclass
class EpochSummary:
    epoch: int
    num_samples: int
    class_distribution: Dict[str, int]
    loss: float
    accuracy: float
    qc_statistics: Dict[str, Dict[str, float]]
    deletion_insertion: Dict[str, Dict[str, float]]


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _load_encoder(modality: str, checkpoint_path: Path, device: torch.device) -> Tuple[nn.Module, int]:
    modality_key = {
        "respiratory": "respiratory_state_dict",
        "sleep_stages": "sleep_stages_state_dict",
        "ekg": "ekg_state_dict",
    }[modality]

    in_channels = len(config.CHANNEL_DATA_IDS["Respiratory"])
    if modality == "sleep_stages":
        in_channels = len(config.CHANNEL_DATA_IDS["Sleep_Stages"])
    elif modality == "ekg":
        in_channels = len(config.CHANNEL_DATA_IDS["EKG"])

    encoder = models.EffNet(in_channel=in_channels, stride=2, dilation=1)
    feature_dim = encoder.fc.in_features
    encoder.fc = nn.Identity()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _strip_module_prefix(checkpoint[modality_key])
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading encoder: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading encoder: %s", unexpected)

    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    return encoder, feature_dim


def _load_probe(
    probe_path: Optional[Path], feature_dim: int, num_classes: int, device: torch.device
) -> nn.Module:
    probe = nn.Linear(feature_dim, num_classes)
    if probe_path is not None and probe_path.exists():
        loaded = torch.load(probe_path, map_location=device)
        if isinstance(loaded, nn.Module):
            probe = loaded
        elif isinstance(loaded, dict):
            if "state_dict" in loaded:
                probe.load_state_dict(loaded["state_dict"])
            elif {"weight", "bias"}.issubset(loaded.keys()):
                probe.load_state_dict(loaded)
            else:
                raise ValueError(
                    "Unsupported probe checkpoint format. Expected state_dict or nn.Module."
                )
        else:
            raise ValueError("Unsupported probe checkpoint type: %s" % type(loaded))

    probe.to(device)
    probe.eval()
    for param in probe.parameters():
        param.requires_grad_(False)
    return probe


class SleepFMProbe(nn.Module):
    """Wrapper combining the frozen encoder with a linear probe."""

    def __init__(self, encoder: nn.Module, probe: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.probe = probe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.probe(features)
        return logits


def _compute_qc_statistics(tensor: torch.Tensor) -> Dict[str, float]:
    arr = tensor.detach().cpu().numpy()
    batch_stats = []
    for sample in arr:
        flat = sample.reshape(-1)
        abs_flat = np.abs(flat)
        total_mass = abs_flat.sum() + 1e-8
        k = max(1, int(math.ceil(0.01 * flat.size)))
        top_mass = abs_flat[np.argpartition(-abs_flat, k - 1)[:k]].sum() / total_mass
        non_zero_fraction = float(np.count_nonzero(flat)) / float(flat.size)
        batch_stats.append(
            {
                "min": float(flat.min()),
                "max": float(flat.max()),
                "top_1_percent_mass": float(top_mass),
                "non_zero_fraction": float(non_zero_fraction),
            }
        )

    keys = batch_stats[0].keys()
    return {k: float(np.mean([sample[k] for sample in batch_stats])) for k in keys}


def _evaluate_masking_auc(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    attribution: torch.Tensor,
    steps: int,
    baseline_value: float,
    mode: str,
) -> float:
    if steps < 1:
        raise ValueError("steps must be >= 1")

    device = inputs.device
    baseline = torch.full_like(inputs, baseline_value)
    flat_attr = attribution.view(attribution.size(0), -1)
    sorted_idx = torch.argsort(flat_attr, dim=1, descending=True)
    total_features = flat_attr.size(1)
    step_size = max(total_features // steps, 1)

    if mode == "deletion":
        mask = torch.ones_like(flat_attr, device=device)
    elif mode == "insertion":
        mask = torch.zeros_like(flat_attr, device=device)
    else:
        raise ValueError("mode must be 'deletion' or 'insertion'")

    probs: List[torch.Tensor] = []
    for step in range(steps + 1):
        mask_view = mask.view_as(inputs)
        perturbed = inputs * mask_view + baseline * (1 - mask_view)
        with torch.no_grad():
            logits = model(perturbed)
            prob = F.softmax(logits, dim=1).gather(1, targets.view(-1, 1)).squeeze(1)
        probs.append(prob.detach())

        if step == steps:
            break

        start = step * step_size
        end = min(total_features, (step + 1) * step_size)
        idx = sorted_idx[:, start:end]
        fill_value = 0.0 if mode == "deletion" else 1.0
        mask.scatter_(1, idx, fill_value)

    curve = torch.stack(probs, dim=1)
    auc = torch.trapz(curve, dx=1.0 / steps, dim=1)
    return float(auc.mean().item())


def _save_heatmap(arr: np.ndarray, output_path: Path, title: str) -> None:
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    plt.figure(figsize=(8, 3))
    plt.imshow(arr, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _collate_class_distribution(labels: Sequence[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {label: 0 for label in config.CLASS_LABELS}
    for label_idx in labels:
        if 0 <= label_idx < len(config.CLASS_LABELS):
            counts[config.CLASS_LABELS[label_idx]] += 1
    return counts


def run_epoch(
    model: SleepFMProbe,
    dataloader: DataLoader,
    epoch: int,
    cfg: XAIConfig,
) -> EpochSummary:
    device = torch.device(cfg.device)
    model.eval()

    losses = []
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    saliency_maps: List[np.ndarray] = []
    ig_maps: List[np.ndarray] = []
    smoothgrad_maps: List[np.ndarray] = []
    gradcam_maps: List[np.ndarray] = []

    saliency_qc: List[Dict[str, float]] = []
    ig_qc: List[Dict[str, float]] = []
    smoothgrad_qc: List[Dict[str, float]] = []
    gradcam_qc: List[Dict[str, float]] = []

    deletion_scores: Dict[str, List[float]] = {"saliency": [], "integrated_gradients": [], "smoothgrad": [], "gradcam": []}
    insertion_scores: Dict[str, List[float]] = {"saliency": [], "integrated_gradients": [], "smoothgrad": [], "gradcam": []}

    all_targets: List[int] = []

    epoch_dir = cfg.output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_targets.extend(labels.cpu().tolist())

        attributions: SaliencyOutputs = compute_all_attributions(
            model,
            inputs,
            target=labels,
            baseline=torch.full_like(inputs, cfg.baseline_value),
            ig_steps=cfg.ig_steps,
            smoothgrad_samples=cfg.smoothgrad_samples,
            smoothgrad_noise=cfg.smoothgrad_noise,
        )
        gradcam = compute_gradcam(
            model,
            model.encoder.stage8 if hasattr(model.encoder, "stage8") else model.encoder,
            inputs,
            target=labels,
            upsample_to=inputs.size(-1),
        )

        saliency_maps.append(attributions.saliency.detach().cpu().numpy())
        ig_maps.append(attributions.integrated_gradients.detach().cpu().numpy())
        smoothgrad_maps.append(attributions.smoothgrad.detach().cpu().numpy())
        gradcam_maps.append(gradcam.detach().cpu().numpy())

        saliency_qc.append(_compute_qc_statistics(attributions.saliency))
        ig_qc.append(_compute_qc_statistics(attributions.integrated_gradients))
        smoothgrad_qc.append(_compute_qc_statistics(attributions.smoothgrad))
        gradcam_qc.append(_compute_qc_statistics(gradcam))

        deletion_scores["saliency"].append(
            _evaluate_masking_auc(
                model,
                inputs,
                labels,
                attributions.saliency,
                cfg.deletion_insertion_steps,
                cfg.baseline_value,
                mode="deletion",
            )
        )
        deletion_scores["integrated_gradients"].append(
            _evaluate_masking_auc(
                model,
                inputs,
                labels,
                attributions.integrated_gradients,
                cfg.deletion_insertion_steps,
                cfg.baseline_value,
                mode="deletion",
            )
        )
        deletion_scores["smoothgrad"].append(
            _evaluate_masking_auc(
                model,
                inputs,
                labels,
                attributions.smoothgrad,
                cfg.deletion_insertion_steps,
                cfg.baseline_value,
                mode="deletion",
            )
        )
        deletion_scores["gradcam"].append(
            _evaluate_masking_auc(
                model,
                inputs,
                labels,
                gradcam,
                cfg.deletion_insertion_steps,
                cfg.baseline_value,
                mode="deletion",
            )
        )

        insertion_scores["saliency"].append(
            _evaluate_masking_auc(
                model,
                inputs,
                labels,
                attributions.saliency,
                cfg.deletion_insertion_steps,
                cfg.baseline_value,
                mode="insertion",
            )
        )
        insertion_scores["integrated_gradients"].append(
            _evaluate_masking_auc(
                model,
                inputs,
                labels,
                attributions.integrated_gradients,
                cfg.deletion_insertion_steps,
                cfg.baseline_value,
                mode="insertion",
            )
        )
        insertion_scores["smoothgrad"].append(
            _evaluate_masking_auc(
                model,
                inputs,
                labels,
                attributions.smoothgrad,
                cfg.deletion_insertion_steps,
                cfg.baseline_value,
                mode="insertion",
            )
        )
        insertion_scores["gradcam"].append(
            _evaluate_masking_auc(
                model,
                inputs,
                labels,
                gradcam,
                cfg.deletion_insertion_steps,
                cfg.baseline_value,
                mode="insertion",
            )
        )

    if not saliency_maps:
        logger.warning("No samples processed for epoch %s", epoch)
        zero_stats = {
            "min": 0.0,
            "max": 0.0,
            "top_1_percent_mass": 0.0,
            "non_zero_fraction": 0.0,
        }
        qc_statistics = {
            "saliency": zero_stats,
            "integrated_gradients": zero_stats,
            "smoothgrad": zero_stats,
            "gradcam": zero_stats,
        }
        deletion_summary = {k: 0.0 for k in deletion_scores}
        insertion_summary = {k: 0.0 for k in insertion_scores}
        return EpochSummary(
            epoch=epoch,
            num_samples=0,
            class_distribution={label: 0 for label in config.CLASS_LABELS},
            loss=0.0,
            accuracy=0.0,
            qc_statistics=qc_statistics,
            deletion_insertion={
                "deletion_auc": deletion_summary,
                "insertion_auc": insertion_summary,
            },
        )

    saliency_arr = np.concatenate(saliency_maps, axis=0)
    ig_arr = np.concatenate(ig_maps, axis=0)
    smoothgrad_arr = np.concatenate(smoothgrad_maps, axis=0)
    gradcam_arr = np.concatenate(gradcam_maps, axis=0)

    np.save(epoch_dir / "saliency.npy", saliency_arr)
    np.save(epoch_dir / "integrated_gradients.npy", ig_arr)
    np.save(epoch_dir / "smoothgrad.npy", smoothgrad_arr)
    np.save(epoch_dir / "gradcam.npy", gradcam_arr)

    _save_heatmap(saliency_arr.mean(axis=0), epoch_dir / "saliency.png", "Saliency (mean)")
    _save_heatmap(ig_arr.mean(axis=0), epoch_dir / "integrated_gradients.png", "Integrated Gradients (mean)")
    _save_heatmap(smoothgrad_arr.mean(axis=0), epoch_dir / "smoothgrad.png", "SmoothGrad (mean)")
    _save_heatmap(gradcam_arr.mean(axis=0), epoch_dir / "gradcam.png", "Grad-CAM (mean)")

    qc_statistics = {
        "saliency": {k: float(np.mean([item[k] for item in saliency_qc])) for k in saliency_qc[0]},
        "integrated_gradients": {k: float(np.mean([item[k] for item in ig_qc])) for k in ig_qc[0]},
        "smoothgrad": {k: float(np.mean([item[k] for item in smoothgrad_qc])) for k in smoothgrad_qc[0]},
        "gradcam": {k: float(np.mean([item[k] for item in gradcam_qc])) for k in gradcam_qc[0]},
    }

    for name, stats in qc_statistics.items():
        with open(epoch_dir / f"{name}_qc.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    deletion_summary = {
        k: float(np.mean(v)) for k, v in deletion_scores.items()
    }
    insertion_summary = {
        k: float(np.mean(v)) for k, v in insertion_scores.items()
    }

    with open(epoch_dir / "deletion_insertion.json", "w", encoding="utf-8") as f:
        json.dump(
            {"deletion_auc": deletion_summary, "insertion_auc": insertion_summary},
            f,
            indent=2,
        )

    loss = float(np.mean(losses)) if losses else 0.0
    accuracy = float(correct / total) if total else 0.0

    class_distribution = _collate_class_distribution(all_targets)

    return EpochSummary(
        epoch=epoch,
        num_samples=total,
        class_distribution=class_distribution,
        loss=loss,
        accuracy=accuracy,
        qc_statistics=qc_statistics,
        deletion_insertion={
            "deletion_auc": deletion_summary,
            "insertion_auc": insertion_summary,
        },
    )


def _write_summary(epochs: List[EpochSummary], cfg: XAIConfig) -> None:
    qc_path = cfg.output_dir / "xai_qc.json"
    qc_payload = {
        "epochs": [
            {
                "epoch": e.epoch,
                "num_samples": e.num_samples,
                "loss": e.loss,
                "accuracy": e.accuracy,
                "qc_statistics": e.qc_statistics,
                "deletion_insertion": e.deletion_insertion,
            }
            for e in epochs
        ]
    }
    with open(qc_path, "w", encoding="utf-8") as f:
        json.dump(qc_payload, f, indent=2)

    report_path = cfg.output_dir / "report.md"
    lines = ["# XAI Quality Report", ""]
    lines.append(f"**Modality:** {cfg.modality}")
    lines.append(f"**Checkpoint:** {cfg.checkpoint_path}")
    if cfg.probe_path:
        lines.append(f"**Probe:** {cfg.probe_path}")
    lines.append("")

    for summary in epochs:
        lines.append(f"## Epoch {summary.epoch}")
        lines.append(f"*Samples:* {summary.num_samples}")
        lines.append(f"*Loss:* {summary.loss:.4f}")
        lines.append(f"*Accuracy:* {summary.accuracy:.4f}")
        lines.append("*Class distribution:*")
        for label, count in summary.class_distribution.items():
            lines.append(f"  - {label}: {count}")
        lines.append("")
        lines.append("### QC Metrics")
        for method, stats in summary.qc_statistics.items():
            lines.append(f"- **{method}**")
            for key, value in stats.items():
                lines.append(f"  - {key}: {value:.6f}")
        lines.append("")
        lines.append("### Deletion / Insertion AUC")
        for mode, stats in summary.deletion_insertion.items():
            lines.append(f"- **{mode}**")
            for method, value in stats.items():
                lines.append(f"  - {method}: {value:.6f}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_xai(cfg: XAIConfig, dataset_path: Path, split: str = "valid", epochs: Iterable[int] = (0,)) -> None:
    device = torch.device(cfg.device)
    encoder, feature_dim = _load_encoder(cfg.modality, cfg.checkpoint_path, device)
    probe = _load_probe(cfg.probe_path, feature_dim, config.NUM_CLASSES, device)
    model = SleepFMProbe(encoder, probe)

    dataset = EventDatasetSupervised(str(dataset_path), split=split, modality_type=cfg.modality)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        drop_last=False,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for epoch in epochs:
        summaries.append(run_epoch(model, dataloader, epoch, cfg))

    _write_summary(summaries, cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate attribution maps for SleepFM probes")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to the encoder checkpoint (best.pt)")
    parser.add_argument("--probe", type=Path, default=None, help="Path to the trained linear probe checkpoint")
    parser.add_argument("--dataset", required=True, type=Path, help="Pickle dataset file for EventDatasetSupervised")
    parser.add_argument("--output", type=Path, default=Path("xai_outputs"), help="Output directory for attribution artifacts")
    parser.add_argument("--modality", choices=["respiratory", "sleep_stages", "ekg"], default="sleep_stages")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--smoothgrad-samples", type=int, default=25)
    parser.add_argument("--smoothgrad-noise", type=float, default=0.1)
    parser.add_argument("--deletion-insertion-steps", type=int, default=20)
    parser.add_argument("--baseline", type=float, default=0.0)
    parser.add_argument("--split", type=str, default="valid")
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="*",
        default=[0],
        help="Epoch indices for which to export maps (used for bookkeeping)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = XAIConfig(
        checkpoint_path=args.checkpoint,
        probe_path=args.probe,
        modality=args.modality,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        ig_steps=args.ig_steps,
        smoothgrad_samples=args.smoothgrad_samples,
        smoothgrad_noise=args.smoothgrad_noise,
        deletion_insertion_steps=args.deletion_insertion_steps,
        baseline_value=args.baseline,
        output_dir=args.output,
    )
    run_xai(cfg, dataset_path=args.dataset, split=args.split, epochs=args.epochs)


if __name__ == "__main__":
    main()

