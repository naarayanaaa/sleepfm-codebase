from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from loguru import logger

import sys
sys.path.append("../model")
import models
from config import CHANNEL_DATA_IDS, PATH_TO_PROCESSED_DATA
from xai import grad_cam_1d, integrated_gradients, saliency_map, smooth_grad


MODALITY_TO_KEY = {
    "respiratory": "respiratory_state_dict",
    "sleep_stages": "sleep_stages_state_dict",
    "ekg": "ekg_state_dict",
}


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def _build_encoder(modality: str, checkpoint: Dict[str, torch.Tensor], embedding_dim: int) -> models.EffNet:
    in_channels = len(CHANNEL_DATA_IDS["Respiratory" if modality == "respiratory" else "Sleep_Stages" if modality == "sleep_stages" else "EKG"])
    encoder = models.EffNet(in_channel=in_channels, stride=2, dilation=1)
    encoder.fc = torch.nn.Linear(encoder.fc.in_features, embedding_dim)
    state_dict = _strip_module_prefix(checkpoint[MODALITY_TO_KEY[modality]])
    encoder.load_state_dict(state_dict)
    return encoder


def _load_linear_head(path: Path, device: torch.device) -> torch.nn.Module:
    artifact = torch.load(path, map_location=device)
    head = torch.nn.Linear(artifact["input_dim"], artifact["num_classes"])
    head.load_state_dict(artifact["state_dict"])
    head.eval()
    return head, artifact


def _default_dataset_event_file(label_key: str) -> str:
    if label_key == "sleep_stage":
        return "dataset_events_-1.pickle"
    safe_key = label_key.replace(" ", "_").lower()
    return f"dataset_events_{safe_key}_-1.pickle"


def main(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(PATH_TO_PROCESSED_DATA)
    best_checkpoint = Path(args.best_checkpoint) if args.best_checkpoint else dataset_dir / args.output_file / "best.pt"
    linear_checkpoint = Path(args.linear_checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / args.output_file / "xai_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Using device: %s", device)

    dataset_event_file = (
        Path(args.dataset_event_file)
        if args.dataset_event_file
        else Path(_default_dataset_event_file(args.label_key))
    )
    with open(dataset_dir / dataset_event_file, "rb") as handle:
        dataset_events = (
            json.load(handle) if dataset_event_file.suffix == ".json" else pickle.load(handle)
        )

    split_data: List = dataset_events[args.split]
    if args.index < 0 or args.index >= len(split_data):
        raise IndexError(f"Index {args.index} is out of range for split {args.split}")

    event_path, event_label = split_data[args.index]
    logger.info("Selected event %s with label %s", event_path, event_label)

    signal = np.load(event_path)
    modality_key = (
        "Respiratory" if args.modality_type == "respiratory" else "Sleep_Stages" if args.modality_type == "sleep_stages" else "EKG"
    )
    indices = CHANNEL_DATA_IDS[modality_key]
    signal = signal[indices]
    signal_tensor = torch.tensor(signal, dtype=torch.float32, device=device).unsqueeze(0)

    checkpoint = _load_checkpoint(best_checkpoint, device)
    linear_head, artifact = _load_linear_head(linear_checkpoint, device)
    encoder = _build_encoder(args.modality_type, checkpoint, artifact["input_dim"])
    encoder.to(device)
    encoder.eval()
    linear_head.to(device)

    class CombinedModel(torch.nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, x):
            embedding = self.encoder(x)
            embedding = torch.nn.functional.normalize(embedding, dim=1)
            return self.head(embedding)

    combined = CombinedModel(encoder, linear_head)
    combined.eval()

    with torch.no_grad():
        logits = combined(signal_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted = int(torch.argmax(probs, dim=1))
    target_index = args.target_index if args.target_index is not None else predicted
    logger.info("Target class index: %d", target_index)

    methods = [method.strip() for method in args.methods.split(",")]
    results: Dict[str, np.ndarray] = {}

    if "saliency" in methods:
        saliency = saliency_map(lambda x: combined(x), signal_tensor, target_index)
        results["saliency"] = saliency.squeeze(0).cpu().numpy()

    if "integrated_gradients" in methods:
        baseline = torch.zeros_like(signal_tensor) if args.baseline == "zero" else torch.tensor(np.load(args.baseline), dtype=torch.float32, device=device).unsqueeze(0)
        ig = integrated_gradients(lambda x: combined(x), signal_tensor, target_index, baseline=baseline, steps=args.steps)
        results["integrated_gradients"] = ig.squeeze(0).cpu().numpy()

    if "smoothgrad" in methods:
        sg = smooth_grad(lambda x: combined(x), signal_tensor, target_index, noise_std=args.noise_std, num_samples=args.num_samples)
        results["smoothgrad"] = sg.squeeze(0).cpu().numpy()

    if "gradcam" in methods:
        cam = grad_cam_1d(combined, signal_tensor, target_index, layer=combined.encoder.stage8)
        results["gradcam"] = cam.squeeze(0).cpu().numpy()

    for name, array in results.items():
        output_path = output_dir / f"{args.split}_{args.index}_{args.modality_type}_{name}.npy"
        np.save(output_path, array)
        logger.info("Saved %s attribution to %s", name, output_path)

    metadata = {
        "event_path": event_path,
        "event_label": event_label,
        "target_index": target_index,
        "methods": methods,
        "probabilities": probs.squeeze(0).cpu().tolist(),
        "class_labels": artifact.get("class_labels"),
    }
    with open(output_dir / f"{args.split}_{args.index}_{args.modality_type}_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    logger.info("Saved metadata for attributions")


if __name__ == "__main__":
    import pickle

    parser = argparse.ArgumentParser(description="Generate attribution maps for SleepFM downstream models")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Processed dataset directory")
    parser.add_argument("--output_file", type=str, required=True, help="Embedding experiment name")
    parser.add_argument("--linear_checkpoint", type=str, required=True, help="Path to the trained linear head checkpoint")
    parser.add_argument("--best_checkpoint", type=str, default=None, help="Path to the SleepFM best.pt checkpoint")
    parser.add_argument("--dataset_event_file", type=str, default=None, help="Dataset event pickle filename")
    parser.add_argument("--label_key", type=str, default="sleep_stage", help="Label key for dataset events")
    parser.add_argument("--modality_type", type=str, default="sleep_stages", choices=["respiratory", "sleep_stages", "ekg"])
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--index", type=int, default=0, help="Index within the split for which to compute attributions")
    parser.add_argument("--methods", type=str, default="saliency,integrated_gradients,smoothgrad,gradcam", help="Comma-separated attribution methods")
    parser.add_argument("--target_index", type=int, default=None, help="Target class index (defaults to predicted class)")
    parser.add_argument("--baseline", type=str, default="zero", help="Baseline for integrated gradients (path to .npy or 'zero')")
    parser.add_argument("--steps", type=int, default=50, help="Steps for integrated gradients")
    parser.add_argument("--num_samples", type=int, default=25, help="Samples for SmoothGrad")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Noise std for SmoothGrad")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to store attribution outputs")
    parser.add_argument("--device", type=str, default=None, help="Explicit device override (e.g., 'cpu')")

    args = parser.parse_args()
    main(args)
