from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import (
    CLASS_LABELS,
    LABELS_DICT,
    MODALITY_TYPES,
    PATH_TO_PROCESSED_DATA,
)
from utils import train_model


class EmbeddingDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def _tensor_to_numpy(array) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _prepare_embeddings(embeddings, modality_type: str) -> np.ndarray:
    if modality_type == "combined":
        arrays = [_tensor_to_numpy(item) for item in embeddings]
        return np.concatenate(arrays, axis=1)
    idx = MODALITY_TYPES.index(modality_type)
    return _tensor_to_numpy(embeddings[idx])


def _split_paths_labels(split_data: Sequence[Sequence[str]]) -> Tuple[List[str], List[str]]:
    paths: List[str] = []
    labels: List[str] = []
    for item in split_data:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(
                "Dataset event entries must be tuples of (path, label); received %r" % (item,)
            )
        paths.append(item[0])
        labels.append(item[1])
    return paths, labels


def _filter_valid_indices(labels: List[str], label_lookup: Dict[str, int]) -> List[int]:
    return [idx for idx, label in enumerate(labels) if label in label_lookup]


def _build_label_lookup(label_key: str, all_labels: Sequence[str]) -> Dict[str, int]:
    if label_key == "sleep_stage":
        return LABELS_DICT
    unique = sorted({label for label in all_labels if label != ""})
    return {label: idx for idx, label in enumerate(unique)}


def _log_split_counts(name: str, labels: Sequence[str]) -> None:
    counter = Counter(labels)
    logger.info("%s label distribution: %s", name, dict(counter))


def _default_dataset_event_file(label_key: str) -> str:
    if label_key == "sleep_stage":
        return "dataset_events_-1.pickle"
    safe_key = label_key.replace(" ", "_").lower()
    return f"dataset_events_{safe_key}_-1.pickle"


def train_torch_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    input_dim: int,
    num_classes: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> nn.Module:
    model = nn.Linear(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(EmbeddingDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                valid_loss += loss.item() * batch_x.size(0)

        epoch_loss /= max(1, len(train_loader.dataset))
        valid_loss /= max(1, len(valid_loader.dataset))
        logger.info("Epoch %d - train loss: %.4f, valid loss: %.4f", epoch + 1, epoch_loss, valid_loss)

        if valid_loss + 1e-6 < best_val_loss:
            best_val_loss = valid_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                logger.info("Early stopping after %d epochs", epoch + 1)
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def _evaluate_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_labels: List[str],
) -> Dict[str, Dict[str, float]]:
    y_pred = y_probs.argmax(axis=1)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion matrix:\n%s", cm)
    return report


def main(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(PATH_TO_PROCESSED_DATA)
    output_dir = dataset_dir / args.output_file
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    probs_dir = output_dir / "probs"
    for path in (figures_dir, models_dir, probs_dir):
        path.mkdir(parents=True, exist_ok=True)

    dataset_event_file = args.dataset_event_file or _default_dataset_event_file(args.label_key)
    dataset_event_path = dataset_dir / dataset_event_file
    if not dataset_event_path.exists():
        raise FileNotFoundError(f"Dataset event file not found: {dataset_event_path}")

    with open(dataset_event_path, "rb") as handle:
        dataset_events = pickle.load(handle)

    eval_dir = output_dir / "eval_data"
    with open(eval_dir / f"{Path(dataset_event_file).stem}_train_emb.pickle", "rb") as handle:
        emb_train = pickle.load(handle)
    with open(eval_dir / f"{Path(dataset_event_file).stem}_valid_emb.pickle", "rb") as handle:
        emb_valid = pickle.load(handle)
    with open(eval_dir / f"{Path(dataset_event_file).stem}_test_emb.pickle", "rb") as handle:
        emb_test = pickle.load(handle)

    X_train_full = _prepare_embeddings(emb_train, args.modality_type)
    X_valid_full = _prepare_embeddings(emb_valid, args.modality_type)
    X_test_full = _prepare_embeddings(emb_test, args.modality_type)

    train_paths, labels_train = _split_paths_labels(dataset_events["train"])
    valid_paths, labels_valid = _split_paths_labels(dataset_events["valid"])
    test_paths, labels_test = _split_paths_labels(dataset_events["test"])

    _log_split_counts("Train", labels_train)
    _log_split_counts("Valid", labels_valid)
    _log_split_counts("Test", labels_test)

    label_lookup = _build_label_lookup(args.label_key, labels_train + labels_valid + labels_test)
    class_labels = (
        CLASS_LABELS if args.label_key == "sleep_stage" else list(label_lookup.keys())
    )

    indices_train = _filter_valid_indices(labels_train, label_lookup)
    indices_valid = _filter_valid_indices(labels_valid, label_lookup)
    indices_test = _filter_valid_indices(labels_test, label_lookup)

    if not indices_train or not indices_test:
        raise ValueError(
            "No samples found for the requested label key; check dataset preparation outputs."
        )

    X_train = X_train_full[indices_train]
    X_valid = X_valid_full[indices_valid]
    X_test = X_test_full[indices_test]

    y_train = np.array([label_lookup[labels_train[i]] for i in indices_train])
    y_valid = np.array([label_lookup[labels_valid[i]] for i in indices_valid])
    y_test = np.array([label_lookup[labels_test[i]] for i in indices_test])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    if args.trainer == "sklearn":
        model, y_probs, class_report = train_model(
            X_train,
            X_test,
            y_train,
            y_test,
            path_to_save=figures_dir / f"{args.modality_type}_{args.label_key}",
            class_labels=class_labels,
            model_name=args.model_name,
            max_iter=args.max_iter,
        )
        model_artifact_path = models_dir / f"{args.modality_type}_{args.label_key}_sklearn.pickle"
        with open(model_artifact_path, "wb") as handle:
            pickle.dump(
                {
                    "model": model,
                    "label_lookup": label_lookup,
                    "class_labels": class_labels,
                    "modality_type": args.modality_type,
                },
                handle,
            )
    else:
        linear = train_torch_linear(
            X_train,
            y_train,
            X_valid,
            y_valid,
            input_dim=X_train.shape[1],
            num_classes=len(label_lookup),
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_epochs=args.max_epochs,
            patience=args.patience,
            device=device,
        )
        linear.eval()
        with torch.no_grad():
            logits = linear(torch.tensor(X_test, dtype=torch.float32, device=device))
            y_probs = torch.softmax(logits, dim=1).cpu().numpy()
        class_report = _evaluate_predictions(y_test, y_probs, class_labels)
        model_artifact_path = models_dir / f"{args.modality_type}_{args.label_key}_linear.pt"
        torch.save(
            {
                "state_dict": linear.state_dict(),
                "input_dim": X_train.shape[1],
                "num_classes": len(label_lookup),
                "label_lookup": label_lookup,
                "class_labels": class_labels,
                "modality_type": args.modality_type,
                "label_key": args.label_key,
            },
            model_artifact_path,
        )

    probs_path = probs_dir / f"{args.modality_type}_{args.label_key}_y_probs.npy"
    np.save(probs_path, y_probs)

    report_path = probs_dir / f"{args.modality_type}_{args.label_key}_class_report.json"
    with open(report_path, "w") as handle:
        json.dump(class_report, handle, indent=2)

    logger.info("Saved probabilities to %s", probs_path)
    logger.info("Saved classification report to %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train downstream classifiers on SleepFM embeddings")
    parser.add_argument("--output_file", type=str, required=True, help="Experiment name (matches embedding directory)")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to the processed dataset")
    parser.add_argument(
        "--dataset_event_file",
        type=str,
        default=None,
        help="Optional override for the dataset event pickle filename",
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default="sleep_stage",
        help="Label key stored inside the per-event dictionaries",
    )
    parser.add_argument(
        "--modality_type",
        type=str,
        default="combined",
        choices=["respiratory", "sleep_stages", "ekg", "combined"],
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="torch",
        choices=["torch", "sklearn"],
        help="Backend used to fit the downstream classifier",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="logistic",
        choices=["logistic", "xgb"],
        help="Model used when trainer=sklearn",
    )
    parser.add_argument("--max_iter", type=int, default=100, help="Max iterations for sklearn logistic")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for torch trainer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for torch trainer")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for torch trainer"
    )
    parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs for torch trainer")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience for torch trainer")

    args = parser.parse_args()
    main(args)
