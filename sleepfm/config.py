"""Global configuration shared across the SleepFM CAP pipeline.

The defaults align with the CAP sleep dataset and can be overridden via
environment variables to match local directory structures or custom
channel layouts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


def _env_path(name: str, default: Path) -> str:
    value = os.environ.get(name)
    if value:
        return str(Path(value).expanduser())
    return str(default.expanduser())


def _env_json(name: str, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        raise ValueError(
            f"Unable to decode JSON from environment variable {name!r}: {raw}"
        ) from exc


PATH_TO_RAW_DATA = _env_path("SLEEPFM_CAP_RAW_DIR", Path("data/cap/raw"))
PATH_TO_PROCESSED_DATA = _env_path(
    "SLEEPFM_CAP_PROCESSED_DIR", Path("data/cap/processed")
)

# ---------------------------------------------------------------------------
# Sleep stage labels
# ---------------------------------------------------------------------------

LABELS_DICT = {
    "Wake": 0,
    "Stage 1": 1,
    "Stage 2": 2,
    "Stage 3": 3,
    "REM": 4,
}

MODALITY_TYPES = ["respiratory", "sleep_stages", "ekg"]
CLASS_LABELS = list(LABELS_DICT.keys())
NUM_CLASSES = len(CLASS_LABELS)


def _make_event_to_id(labels_dict: Dict[str, int]) -> Dict[str, int]:
    return {name: idx + 1 for name, idx in labels_dict.items()}


EVENT_TO_ID = _make_event_to_id(LABELS_DICT)


LABEL_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage N1": "Stage 1",
    "Sleep stage N2": "Stage 2",
    "Sleep stage N3": "Stage 3",
    "Sleep stage N4": "Stage 3",
    "Sleep stage R": "REM",
    "W": "Wake",
    "N1": "Stage 1",
    "N2": "Stage 2",
    "N3": "Stage 3",
    "N4": "Stage 3",
    "REM": "REM",
    "wake": "Wake",
    "nonrem1": "Stage 1",
    "nonrem2": "Stage 2",
    "nonrem3": "Stage 3",
    "nonrem4": "Stage 3",
    "rem": "REM",
}


CAP_CHANNEL_GROUPS = {
    "Respiratory": [
        "Resp Abdomen",
        "Resp Thorax",
        "Airflow",
        "SaO2",
    ],
    "Sleep_Stages": [
        "EEG Fp2-F4",
        "EEG F4-C4",
        "EEG C4-M1",
        "EEG Fp1-F3",
        "EEG F3-C3",
        "EEG C3-M2",
        "EEG O2-M1",
        "EEG O1-M2",
        "EOG E1-M2",
        "EMG Chin1-Chin2",
    ],
    "EKG": ["ECG"],
}


CAP_CHANNEL_ALIASES = {
    "Resp Abdomen": ["Abdomen", "ABD", "Resp ABD", "Resp abdomen"],
    "Resp Thorax": ["CHEST", "Thorax", "Resp CHEST", "RESP Thorax"],
    "Airflow": ["AIRFLOW", "Flow", "Resp Airflow"],
    "SaO2": ["SpO2", "SaO2"],
    "EEG Fp2-F4": ["Fp2-F4", "EEG FP2-F4"],
    "EEG F4-C4": ["F4-C4", "EEG F4-C4"],
    "EEG C4-M1": ["C4-M1", "EEG C4-M1"],
    "EEG Fp1-F3": ["Fp1-F3", "EEG FP1-F3"],
    "EEG F3-C3": ["F3-C3", "EEG F3-C3"],
    "EEG C3-M2": ["C3-M2", "EEG C3-M2"],
    "EEG O2-M1": ["O2-M1", "EEG O2-M1"],
    "EEG O1-M2": ["O1-M2", "EEG O1-M2"],
    "EOG E1-M2": ["E1-M2", "EOG horizontal"],
    "EMG Chin1-Chin2": ["Chin1-Chin2", "EMG Chin"],
    "ECG": ["ECG1-ECG2", "ECG", "ECG-Lead1"],
}


ALL_CHANNELS = _env_json(
    "SLEEPFM_CAP_CHANNELS",
    [
        *CAP_CHANNEL_GROUPS["Respiratory"],
        *CAP_CHANNEL_GROUPS["Sleep_Stages"],
        *CAP_CHANNEL_GROUPS["EKG"],
    ],
)


CHANNEL_DATA = _env_json("SLEEPFM_CAP_CHANNEL_GROUPS", CAP_CHANNEL_GROUPS)
CHANNEL_ALIASES = _env_json("SLEEPFM_CAP_CHANNEL_ALIASES", CAP_CHANNEL_ALIASES)


def _channel_indices(
    channels: List[str], channel_groups: Dict[str, Iterable[str]]
) -> Dict[str, List[int]]:
    indices: Dict[str, List[int]] = {}
    for key, group in channel_groups.items():
        indices[key] = [channels.index(item) for item in group]
    return indices


CHANNEL_DATA_IDS = _channel_indices(ALL_CHANNELS, CHANNEL_DATA)


ADDITIONAL_LABEL_MAPS = _env_json("SLEEPFM_CAP_ADDITIONAL_LABEL_MAPS", {})


def canonical_stage(label: str) -> str:
    """Return the canonical stage name for a label string."""

    if label in LABEL_MAP:
        return LABEL_MAP[label]
    normalized = label.strip().upper()
    return LABEL_MAP.get(normalized, label.strip())


def resolve_additional_label(key: str, value: str) -> str:
    mapping = ADDITIONAL_LABEL_MAPS.get(key, {})
    return mapping.get(value, value)


__all__ = [
    "ADDITIONAL_LABEL_MAPS",
    "ALL_CHANNELS",
    "CHANNEL_ALIASES",
    "CHANNEL_DATA",
    "CHANNEL_DATA_IDS",
    "CLASS_LABELS",
    "EVENT_TO_ID",
    "LABELS_DICT",
    "LABEL_MAP",
    "MODALITY_TYPES",
    "NUM_CLASSES",
    "PATH_TO_PROCESSED_DATA",
    "PATH_TO_RAW_DATA",
    "canonical_stage",
    "resolve_additional_label",
]
