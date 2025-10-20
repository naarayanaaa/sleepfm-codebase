"""Validation helpers for SleepFM configuration files and CLI metadata."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, MutableMapping

import numpy as np

CHANNEL_SYNONYMS: Mapping[str, str] = {
    "f3-m2": "F3-M2",
    "f3m2": "F3-M2",
    "eegf3-m2": "F3-M2",
    "f4-m1": "F4-M1",
    "f4m1": "F4-M1",
    "eegf4-m1": "F4-M1",
    "c3-m2": "C3-M2",
    "c3m2": "C3-M2",
    "eegc3-m2": "C3-M2",
    "c4-m1": "C4-M1",
    "c4m1": "C4-M1",
    "eegc4-m1": "C4-M1",
    "o1-m2": "O1-M2",
    "o1m2": "O1-M2",
    "eego1-m2": "O1-M2",
    "o2-m1": "O2-M1",
    "o2m1": "O2-M1",
    "eego2-m1": "O2-M1",
    "e1-m2": "E1-M2",
    "e1m2": "E1-M2",
    "lefteye": "E1-M2",
    "chin1-chin2": "Chin1-Chin2",
    "chin1chin2": "Chin1-Chin2",
    "submental": "Chin1-Chin2",
    "abd": "ABD",
    "abdomen": "ABD",
    "abdominal": "ABD",
    "chest": "CHEST",
    "thorax": "CHEST",
    "thoracic": "CHEST",
    "airflow": "AIRFLOW",
    "airflowpressure": "AIRFLOW",
    "nasalairflow": "AIRFLOW",
    "sao2": "SaO2",
    "spo2": "SaO2",
    "spo2%": "SaO2",
    "oxygen": "SaO2",
    "ecg": "ECG",
    "ekg": "ECG",
}
"""Mapping of lower-cased, punctuation-stripped channel aliases to canonical names."""


def _normalise_channel_name(name: str) -> str:
    """Return a normalised key for channel lookup."""
    cleaned = name.strip().lower().replace("−", "-").replace("–", "-").replace("—", "-")
    cleaned = cleaned.replace("ref", "").replace("reference", "")
    for token in (" ", "_", "\t"):
        cleaned = cleaned.replace(token, "")
    return cleaned


def canon_name(channel_name: str) -> str:
    """Return the canonical representation of a channel name."""
    if not channel_name:
        raise ValueError("Channel name must be a non-empty string")
    key = _normalise_channel_name(channel_name)
    return CHANNEL_SYNONYMS.get(key, channel_name.strip())


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    try:
        json.dumps(obj)
    except TypeError:
        return str(obj)
    return obj


def validate_schema(output_dir: str | Path | None = None) -> None:
    """Validate the channel schema declared in :mod:`config`."""
    import config

    channels = getattr(config, "ALL_CHANNELS", None)
    if not channels:
        raise ValueError("config.ALL_CHANNELS must be defined and non-empty")

    canonical_channels = [canon_name(channel) for channel in channels]
    duplicates = [name for name, count in Counter(canonical_channels).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate channel definitions detected: {duplicates}")

    channel_lookup = {canon_name(channel): index for index, channel in enumerate(channels)}

    channel_data = getattr(config, "CHANNEL_DATA", {})
    unresolved: dict[str, list[str]] = {}
    for group, group_channels in channel_data.items():
        for channel in group_channels:
            canonical = canon_name(channel)
            if canonical not in channel_lookup:
                unresolved.setdefault(group, []).append(channel)
    if unresolved:
        raise ValueError(f"CHANNEL_DATA references unknown channels: {unresolved}")

    channel_data_ids = getattr(config, "CHANNEL_DATA_IDS", {})
    for group, indices in channel_data_ids.items():
        if group not in channel_data:
            raise ValueError(f"CHANNEL_DATA_IDS contains unknown group '{group}'")
        expected = [channel_lookup[canon_name(channel)] for channel in channel_data[group]]
        if list(indices) != expected:
            raise ValueError(
                f"CHANNEL_DATA_IDS mismatch for '{group}': expected {expected}, found {list(indices)}"
            )

    target_dir = Path(output_dir) if output_dir else Path.cwd()
    timestamp = datetime.now(timezone.utc).isoformat()
    _atomic_write_text(target_dir / "schema_ok.txt", f"validated\t{timestamp}\n")


def write_run_metadata(
    args: argparse.Namespace | Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> Path:
    """Persist the CLI invocation to ``run_config.json``."""
    payload: MutableMapping[str, Any]
    if isinstance(args, argparse.Namespace):
        payload = dict(vars(args))
    else:
        payload = dict(args)

    from sys import argv

    timestamp = datetime.now(timezone.utc)
    data = {
        "argv": list(argv),
        "timestamp_utc": timestamp.isoformat(),
        "parameters": _jsonify(payload),
    }

    target_dir = Path(output_dir) if output_dir else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "run_config.json"
    with NamedTemporaryFile("w", delete=False, dir=str(target_dir), encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    return path


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "CHANNEL_SYNONYMS",
    "canon_name",
    "validate_schema",
    "write_run_metadata",
    "seed_everything",
]
