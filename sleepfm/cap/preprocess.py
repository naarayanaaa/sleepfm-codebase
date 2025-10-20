"""Preprocessing utilities to convert CAP data into the SleepFM format."""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import mne
import numpy as np
import pandas as pd
from loguru import logger

from sleepfm.cap.constants import CAP_CHANNEL_ALIASES, CAP_STAGE_ALIASES
from sleepfm.cap.eda import SubjectRecord, discover_subjects
from sleepfm.cap.stage_parser import annotations_to_epochs, parse_stage_file
from sleepfm.config import ALL_CHANNELS, LABEL_MAP

DEFAULT_CHUNK_DURATION = 30.0
DEFAULT_TARGET_SFREQ = 100.0


@dataclasses.dataclass
class SubjectSummary:
    subject_id: str
    n_epochs_written: int
    n_epochs_skipped: int
    missing_channels: List[str]
    reason: Optional[str] = None


def _normalise_stage_label(label: str) -> Optional[str]:
    if label is None:
        return None
    label = label.strip()
    if not label:
        return None
    if label in CAP_STAGE_ALIASES:
        return CAP_STAGE_ALIASES[label]
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    title_case = label.title()
    if title_case in LABEL_MAP:
        return LABEL_MAP[title_case]
    upper = label.upper()
    if upper in LABEL_MAP:
        return LABEL_MAP[upper]
    return label


def _match_channel(target: str, available: Dict[str, str]) -> Optional[str]:
    candidates = [target] + CAP_CHANNEL_ALIASES.get(target, [])
    for candidate in candidates:
        key = candidate.upper()
        if key in available:
            return available[key]
    return None


def _prepare_channels(raw: mne.io.BaseRaw) -> Dict[str, str]:
    return {ch_name.upper(): ch_name for ch_name in raw.ch_names}


def process_subject(
    subject: SubjectRecord,
    output_dir: Path,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    target_sfreq: float = DEFAULT_TARGET_SFREQ,
) -> SubjectSummary:
    subject_output = output_dir / "X" / subject.subject_id
    subject_output.mkdir(parents=True, exist_ok=True)
    label_output = output_dir / "Y"
    label_output.mkdir(parents=True, exist_ok=True)

    try:
        raw = mne.io.read_raw_edf(subject.edf_path, preload=True, verbose="ERROR")
    except Exception as exc:
        reason = f"Failed to open EDF: {exc}"
        logger.error(f"{subject.subject_id}: {reason}")
        return SubjectSummary(subject.subject_id, 0, 0, ALL_CHANNELS, reason=reason)

    available = _prepare_channels(raw)
    rename_map: Dict[str, str] = {}
    missing_channels: List[str] = []
    for canonical in ALL_CHANNELS:
        matched = _match_channel(canonical, available)
        if matched is None:
            missing_channels.append(canonical)
        else:
            rename_map[matched] = canonical

    if missing_channels:
        logger.warning(f"{subject.subject_id}: missing channels {missing_channels}")

    if not rename_map:
        raw.close()
        reason = "No matching channels"
        return SubjectSummary(subject.subject_id, 0, 0, missing_channels, reason=reason)

    raw.pick(list(rename_map.keys()))
    raw.rename_channels(rename_map)
    ordered_channels = [ch for ch in ALL_CHANNELS if ch in raw.ch_names]
    raw.reorder_channels(ordered_channels)

    if raw.info["sfreq"] != target_sfreq:
        raw.resample(target_sfreq)

    total_duration = float(raw.n_times) / float(raw.info["sfreq"])

    if subject.stage_path is None:
        raw.close()
        reason = "Missing stage file"
        return SubjectSummary(subject.subject_id, 0, 0, missing_channels, reason=reason)

    try:
        annotations = parse_stage_file(subject.stage_path, chunk_duration=chunk_duration)
    except Exception as exc:
        raw.close()
        reason = f"Failed to parse stage file: {exc}"
        logger.error(f"{subject.subject_id}: {reason}")
        return SubjectSummary(subject.subject_id, 0, 0, missing_channels, reason=reason)

    if not annotations:
        raw.close()
        reason = "No stage annotations"
        logger.warning(f"{subject.subject_id}: {reason}")
        return SubjectSummary(subject.subject_id, 0, 0, missing_channels, reason=reason)

    epochs = annotations_to_epochs(annotations, total_duration=total_duration, chunk_duration=chunk_duration)
    if epochs.empty:
        raw.close()
        reason = "Unable to align epochs"
        return SubjectSummary(subject.subject_id, 0, 0, missing_channels, reason=reason)

    raw_data = raw.get_data()
    full_data = np.zeros((len(ALL_CHANNELS), raw_data.shape[1]), dtype=np.float32)
    available_indices = {name: idx for idx, name in enumerate(raw.ch_names)}
    for target_idx, canonical in enumerate(ALL_CHANNELS):
        if canonical in available_indices:
            full_data[target_idx] = raw_data[available_indices[canonical]].astype(np.float32)
        else:
            full_data[target_idx] = 0.0
    samples_per_epoch = int(round(chunk_duration * target_sfreq))
    labels: Dict[str, str] = {}
    n_written = 0
    n_skipped = 0

    for _, row in epochs.iterrows():
        start_sample = int(round(row["start_sec"] * target_sfreq))
        end_sample = start_sample + samples_per_epoch
        if end_sample > full_data.shape[1]:
            n_skipped += 1
            continue
        label = _normalise_stage_label(row["label"])
        if label not in LABEL_MAP.values():
            n_skipped += 1
            continue
        epoch_filename = f"{subject.subject_id}_{int(row['epoch']):05d}.npy"
        epoch_path = subject_output / epoch_filename
        np.save(epoch_path, full_data[:, start_sample:end_sample])
        labels[epoch_filename] = label
        n_written += 1

    raw.close()

    labels_path = label_output / f"{subject.subject_id}.pickle"
    with labels_path.open("wb") as handle:
        pickle.dump(labels, handle)

    reason = None if n_written > 0 else "No epochs written"
    return SubjectSummary(subject.subject_id, n_written, n_skipped, missing_channels, reason=reason)


def preprocess_cap_dataset(
    data_root: Path,
    output_dir: Path,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
    target_sfreq: float = DEFAULT_TARGET_SFREQ,
    max_subjects: Optional[int] = None,
) -> pd.DataFrame:
    subjects = discover_subjects(data_root)
    if not subjects:
        raise FileNotFoundError(f"No CAP EDF files found in {data_root}")

    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[SubjectSummary] = []

    for subject in subjects:
        summary = process_subject(
            subject,
            output_dir=output_dir,
            chunk_duration=chunk_duration,
            target_sfreq=target_sfreq,
        )
        summaries.append(summary)

    df = pd.DataFrame([dataclasses.asdict(summary) for summary in summaries])
    report_path = output_dir / "cap_preprocessing_report.csv"
    df.to_csv(report_path, index=False)

    metadata = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "chunk_duration": chunk_duration,
        "target_sfreq": target_sfreq,
        "n_subjects_processed": len(summaries),
        "report": str(report_path),
    }
    (output_dir / "cap_preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2))
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CAP data into SleepFM format")
    parser.add_argument("data_root", type=Path, help="Directory containing rbd*.edf files")
    parser.add_argument("output_dir", type=Path, help="Output directory for SleepFM formatted data")
    parser.add_argument("--chunk-duration", type=float, default=DEFAULT_CHUNK_DURATION, help="Epoch duration in seconds")
    parser.add_argument("--target-sfreq", type=float, default=DEFAULT_TARGET_SFREQ, help="Resample rate in Hz")
    parser.add_argument("--max-subjects", type=int, default=None, help="Process only the first N subjects")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    preprocess_cap_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        chunk_duration=args.chunk_duration,
        target_sfreq=args.target_sfreq,
        max_subjects=args.max_subjects,
    )


if __name__ == "__main__":
    main()
