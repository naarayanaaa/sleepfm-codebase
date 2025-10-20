"""Constants and helper mappings for CAP dataset integration."""
from __future__ import annotations

from typing import Dict, List

# Mapping from SleepFM canonical channel names to the CAP dataset aliases that
# frequently appear in the EDF headers.  The mapping is intentionally verbose so
# that the preprocessing script can automatically match the majority of channel
# label variants encountered in the PhysioNet release.
CAP_CHANNEL_ALIASES: Dict[str, List[str]] = {
    "F3-M2": [
        "F3-M2",
        "F3-A2",
        "EEG F3-A2",
        "EEG F3-M2",
        "F3-LE",
        "EEG F3-LE",
    ],
    "F4-M1": [
        "F4-M1",
        "F4-A1",
        "EEG F4-A1",
        "EEG F4-M1",
        "F4-RE",
        "EEG F4-RE",
    ],
    "C3-M2": [
        "C3-M2",
        "C3-A2",
        "EEG C3-A2",
        "C3-LE",
        "EEG C3-LE",
    ],
    "C4-M1": [
        "C4-M1",
        "C4-A1",
        "EEG C4-A1",
        "C4-RE",
        "EEG C4-RE",
    ],
    "O1-M2": [
        "O1-M2",
        "O1-A2",
        "EEG O1-A2",
        "O1-LE",
        "EEG O1-LE",
    ],
    "O2-M1": [
        "O2-M1",
        "O2-A1",
        "EEG O2-A1",
        "O2-RE",
        "EEG O2-RE",
    ],
    "E1-M2": [
        "E1-M2",
        "ROC-M2",
        "ROC-A2",
        "EOG ROC-M2",
        "EOG ROC-A2",
        "ROC-LE",
        "ROC",
    ],
    "Chin1-Chin2": [
        "Chin1-Chin2",
        "Chin EMG",
        "EMG SUBMENTAL",
        "EMG Chin",
        "EMG",
    ],
    "ABD": [
        "ABD",
        "Abdomen",
        "Abdomen RIP",
        "RESP ABDOMINAL",
        "RESP ABD",
    ],
    "CHEST": [
        "CHEST",
        "Thorax",
        "Thoracic RIP",
        "RESP THORACIC",
        "RESP CHEST",
    ],
    "AIRFLOW": [
        "AIRFLOW",
        "Flow",
        "Nasal Pressure",
        "NASAL PRESSURE",
        "Thermistor",
        "RESP AIRFLOW",
    ],
    "SaO2": [
        "SaO2",
        "SpO2",
        "Oximeter",
        "OXYGEN SATURATION",
    ],
    "ECG": [
        "ECG",
        "ECG Lead 1",
        "EKG",
        "ECG EKG",
        "ECG I",
    ],
}

# Additional label synonyms that appear in the CAP staging files. These are
# merged with the base LABEL_MAP from :mod:`sleepfm.config` during preprocessing.
CAP_STAGE_ALIASES: Dict[str, str] = {
    "N4": "Stage 3",
    "Stage 4": "Stage 3",
    "S4": "Stage 3",
    "S3": "Stage 3",
    "S2": "Stage 2",
    "S1": "Stage 1",
    "S0": "Wake",
    "MT": "Wake",
    "Movement": "Wake",
    "Movement Time": "Wake",
    "ART": "Wake",
    "REM sleep": "REM",
    "REM Sleep": "REM",
    "NonREM1": "Stage 1",
    "NonREM2": "Stage 2",
    "NonREM3": "Stage 3",
    "Stage W": "Wake",
    "Stage N1": "Stage 1",
    "Stage N2": "Stage 2",
    "Stage N3": "Stage 3",
    "Stage R": "REM",
    "Wake": "Wake",
    "Awake": "Wake",
    "Wakefulness": "Wake",
}

# CAP event labels that should be treated as metadata only during preprocessing.
CAP_METADATA_EVENTS = {"A-phase", "B-phase", "CAP", "AROUSAL", "Arousal"}

__all__ = [
    "CAP_CHANNEL_ALIASES",
    "CAP_STAGE_ALIASES",
    "CAP_METADATA_EVENTS",
]
