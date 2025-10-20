#!/usr/bin/env python3
"""Packet 1 conversion pipeline for the CAP dataset.

This script converts CAP dataset EDF recordings and their stage annotations into the
SleepFM on-disk layout. It standardises channel naming, resamples signals to 256 Hz
with anti-aliasing, slices 30 second epochs with robust per-epoch z-scoring, and
writes atomic artefacts compatible with the downstream SleepFM tooling.

Usage example::

    python sleepfm/cap_to_sleepfm.py /path/to/cap /path/to/output

The CLI discovers EDF files recursively, matches annotation files (``.st`` or
``.txt``), and writes ``X/<subject>/<epoch>.npy`` alongside ``Y/<subject>.pickle``
label dictionaries, per-subject channel audit logs, and ``.done`` sentinel files.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np

from config import ALL_CHANNELS, CHANNEL_DATA, LABEL_MAP

LOGGER = logging.getLogger("sleepfm.cap_to_sleepfm")

# ---------------------------------------------------------------------------
# Channel naming utilities
# ---------------------------------------------------------------------------

def canon_name(raw_name: str) -> Optional[str]:
    """Normalise a raw channel label to the SleepFM canonical name.

    The CAP dataset exposes heterogeneous channel names across labs. We make a
    best-effort canonicalisation by inspecting sanitised channel identifiers
    (uppercase, stripped of separators) and mapping them into the SleepFM
    ``ALL_CHANNELS`` space. Channels that cannot be mapped return ``None``.
    """

    if not raw_name:
        return None

    cleaned = raw_name.strip().upper()
    cleaned = cleaned.replace("EEG", "").replace("EOG", "").replace("EMG", "")
    cleaned = cleaned.replace("POL", "").replace("CHAN", "")
    cleaned = cleaned.replace("REF", "")
    cleaned = cleaned.replace("-", "").replace("_", "").replace(" ", "")

    alias_map = {
        "F3M2": "F3-M2",
        "F3A2": "F3-M2",
        "F3M1": "F3-M2",
        "F3A1": "F3-M2",
        "F4M1": "F4-M1",
        "F4A1": "F4-M1",
        "F4M2": "F4-M1",
        "F4A2": "F4-M1",
        "C3M2": "C3-M2",
        "C3A2": "C3-M2",
        "C3M1": "C3-M2",
        "C4M1": "C4-M1",
        "C4A1": "C4-M1",
        "C4M2": "C4-M1",
        "C4A2": "C4-M1",
        "O1M2": "O1-M2",
        "O1A2": "O1-M2",
        "O2M1": "O2-M1",
        "O2A1": "O2-M1",
        "O2A2": "O2-M1",
        "E1M2": "E1-M2",
        "E1A2": "E1-M2",
        "E1A1": "E1-M2",
        "E1M1": "E1-M2",
        "CHIN1CHIN2": "Chin1-Chin2",
        "CHIN": "Chin1-Chin2",
        "EMG1EMG2": "Chin1-Chin2",
        "EMGCHIN": "Chin1-Chin2",
        "SUBMENT": "Chin1-Chin2",
        "SUBMENTAL": "Chin1-Chin2",
        "CHEST": "CHEST",
        "THORAX": "CHEST",
        "THORACIC": "CHEST",
        "RESPCHEST": "CHEST",
        "RESPTHORAX": "CHEST",
        "ABD": "ABD",
        "ABDOMEN": "ABD",
        "RESPABD": "ABD",
        "RESPABDOMEN": "ABD",
        "AIRFLOW": "AIRFLOW",
        "NASAL": "AIRFLOW",
        "NASALPRESSURE": "AIRFLOW",
        "ORAL": "AIRFLOW",
        "FLOW": "AIRFLOW",
        "RESPFLOW": "AIRFLOW",
        "SAO2": "SaO2",
        "SPO2": "SaO2",
        "OXYGEN": "SaO2",
        "OXIMETRY": "SaO2",
        "O2SAT": "SaO2",
        "ECG": "ECG",
        "EKG": "ECG",
    }

    if cleaned in alias_map:
        return alias_map[cleaned]

    # Fallback heuristics for labels containing additional characters
    if "F3" in cleaned and ("A1" in cleaned or "A2" in cleaned or "M1" in cleaned or "M2" in cleaned):
        return "F3-M2"
    if "F4" in cleaned and ("A1" in cleaned or "A2" in cleaned or "M1" in cleaned or "M2" in cleaned):
        return "F4-M1"
    if "C3" in cleaned and ("A1" in cleaned or "A2" in cleaned or "M1" in cleaned or "M2" in cleaned):
        return "C3-M2"
    if "C4" in cleaned and ("A1" in cleaned or "A2" in cleaned or "M1" in cleaned or "M2" in cleaned):
        return "C4-M1"
    if "O1" in cleaned and ("A1" in cleaned or "A2" in cleaned or "M1" in cleaned or "M2" in cleaned):
        return "O1-M2"
    if "O2" in cleaned and ("A1" in cleaned or "A2" in cleaned or "M1" in cleaned or "M2" in cleaned):
        return "O2-M1"
    if cleaned.startswith("E1"):
        return "E1-M2"
    if "CHIN" in cleaned or "SUBMENT" in cleaned or "EMG" in cleaned:
        return "Chin1-Chin2"
    if "THOR" in cleaned or "CHEST" in cleaned:
        return "CHEST"
    if "ABD" in cleaned:
        return "ABD"
    if "FLOW" in cleaned or "NASAL" in cleaned or "AIR" in cleaned:
        return "AIRFLOW"
    if "SPO" in cleaned or "SAO" in cleaned or "OXY" in cleaned or "SAT" in cleaned:
        return "SaO2"
    if "ECG" in cleaned or "EKG" in cleaned or "CARD" in cleaned:
        return "ECG"

    return None


# ---------------------------------------------------------------------------
# Annotation utilities
# ---------------------------------------------------------------------------

_STAGE_ALIASES = {
    "0": "Wake",
    "W": "Wake",
    "WAKE": "Wake",
    "1": "Stage 1",
    "S1": "Stage 1",
    "STAGE1": "Stage 1",
    "STAGE 1": "Stage 1",
    "SLEEPSTAGE1": "Stage 1",
    "SLEEP STAGE 1": "Stage 1",
    "N1": "Stage 1",
    "NREM1": "Stage 1",
    "2": "Stage 2",
    "S2": "Stage 2",
    "STAGE2": "Stage 2",
    "STAGE 2": "Stage 2",
    "SLEEPSTAGE2": "Stage 2",
    "SLEEP STAGE 2": "Stage 2",
    "N2": "Stage 2",
    "NREM2": "Stage 2",
    "3": "Stage 3",
    "4": "Stage 3",
    "S3": "Stage 3",
    "S4": "Stage 3",
    "STAGE3": "Stage 3",
    "STAGE 3": "Stage 3",
    "STAGE4": "Stage 3",
    "STAGE 4": "Stage 3",
    "SLEEPSTAGE3": "Stage 3",
    "SLEEP STAGE 3": "Stage 3",
    "SLEEPSTAGE4": "Stage 3",
    "SLEEP STAGE 4": "Stage 3",
    "N3": "Stage 3",
    "N4": "Stage 3",
    "NREM3": "Stage 3",
    "NREM4": "Stage 3",
    "R": "REM",
    "REM": "REM",
    "5": "REM",
}


@dataclass
class StageSegment:
    start: float
    end: float
    label: Optional[str]

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def _try_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_time_token(token: str) -> Optional[float]:
    token = token.strip()
    if not token:
        return None
    if ":" in token:
        parts = token.replace(",", ":").split(":")
        parts = [p for p in parts if p]
        if 2 <= len(parts) <= 4:
            try:
                hours = int(parts[0])
                minutes = int(parts[1]) if len(parts) > 1 else 0
                seconds = float(parts[2]) if len(parts) > 2 else 0.0
            except ValueError:
                return None
            # Some files include milliseconds as an extra field
            if len(parts) == 4:
                seconds += float(f"0.{parts[3]}")
            return hours * 3600 + minutes * 60 + seconds
    return _try_float(token)


def _normalise_stage_label(label: str) -> Optional[str]:
    if label is None:
        return None
    cleaned = label.strip().upper().replace("_", " ")
    cleaned = re.sub(r"\s+", "", cleaned)
    if cleaned in _STAGE_ALIASES:
        return _STAGE_ALIASES[cleaned]
    spaced = re.sub(r"\s+", " ", label.strip())
    if spaced in LABEL_MAP:
        return LABEL_MAP[spaced]
    upper_spaced = spaced.upper()
    if upper_spaced in LABEL_MAP:
        return LABEL_MAP[upper_spaced]
    lower_spaced = spaced.lower()
    if lower_spaced in LABEL_MAP:
        return LABEL_MAP[lower_spaced]
    return None


def parse_annotation_file(path: Path, epoch_seconds: float) -> List[Tuple[float, Optional[float], Optional[str]]]:
    """Parse CAP stage annotations.

    Returns a list of ``(start_seconds, duration_seconds or None, label)`` tuples.
    Duration may be ``None`` when the source encoding only lists per-epoch stage
    identifiers; it is later resolved once we know the EDF duration.
    """

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    records: List[Tuple[float, Optional[float], Optional[str]]] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Drop header rows
        lowered = line.lower()
        if any(keyword in lowered for keyword in ("stage", "sleep", "epoch")) and not any(ch.isdigit() for ch in line):
            continue

        parts = re.split(r"[\t,; ]+", line)
        parts = [p for p in parts if p]
        if not parts:
            continue

        stage_label: Optional[str] = None
        start_seconds: Optional[float] = None
        duration_seconds: Optional[float] = None

        numeric_parts = [p for p in parts if _try_float(p) is not None]

        # Case 1: explicit start + duration + label
        if len(parts) >= 3 and _try_float(parts[0]) is not None and _try_float(parts[1]) is not None:
            start_seconds = float(parts[0])
            duration_seconds = float(parts[1])
            stage_label = _normalise_stage_label(parts[2])
        # Case 2: timestamp + label
        elif _parse_time_token(parts[0]) is not None and len(parts) >= 2:
            start_seconds = _parse_time_token(parts[0])
            if len(parts) >= 3 and _try_float(parts[1]) is not None:
                duration_seconds = float(parts[1])
                stage_label = _normalise_stage_label(parts[2])
            else:
                stage_label = _normalise_stage_label(parts[1])
        # Case 3: epoch index + stage
        elif _try_float(parts[0]) is not None and len(parts) >= 2:
            epoch_index = int(float(parts[0]))
            if epoch_index > 0:
                start_seconds = (epoch_index - 1) * epoch_seconds
            else:
                start_seconds = 0.0
            duration_seconds = epoch_seconds
            stage_label = _normalise_stage_label(parts[1])
        else:
            # Attempt to locate stage label anywhere in the line
            for token in parts:
                maybe_label = _normalise_stage_label(token)
                if maybe_label:
                    stage_label = maybe_label
                    break
            if stage_label and numeric_parts:
                start_seconds = float(numeric_parts[0])

        if start_seconds is None:
            continue

        records.append((float(start_seconds), duration_seconds, stage_label))

    records.sort(key=lambda item: item[0])
    return records


def build_segments(
    records: List[Tuple[float, Optional[float], Optional[str]]],
    recording_duration: float,
    epoch_seconds: float,
) -> List[StageSegment]:
    """Resolve durations and construct ordered stage segments."""

    if not records:
        return []

    segments: List[StageSegment] = []
    for idx, (start, duration, label) in enumerate(records):
        if duration is None:
            if idx + 1 < len(records):
                next_start = records[idx + 1][0]
                duration = max(0.0, next_start - start)
            else:
                duration = max(0.0, recording_duration - start)
        if duration is None or math.isinf(duration) or math.isnan(duration):
            duration = epoch_seconds
        end = start + duration
        segments.append(StageSegment(start=start, end=end, label=label))

    # Ensure final segment extends to EDF end if annotated with zero duration
    if segments:
        last = segments[-1]
        if last.end <= last.start or last.end < recording_duration:
            segments[-1] = StageSegment(start=last.start, end=recording_duration, label=last.label)

    return segments


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def atomic_write(path: Path, writer) -> None:
    """Write to *path* atomically using *writer* callback."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        writer(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def atomic_save_numpy(array: np.ndarray, path: Path) -> None:
    atomic_write(path, lambda tmp: np.save(tmp, array, allow_pickle=False))


def atomic_save_pickle(obj, path: Path) -> None:
    def _write(tmp: Path) -> None:
        with open(tmp, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    atomic_write(path, _write)


def atomic_save_json(obj, path: Path) -> None:
    def _write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, sort_keys=True)

    atomic_write(path, _write)


def atomic_touch(path: Path, content: str = "") -> None:
    def _write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(content)

    atomic_write(path, _write)


# ---------------------------------------------------------------------------
# Signal processing utilities
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def robust_zscore(epoch: np.ndarray) -> np.ndarray:
    """Apply per-channel robust z-scoring (median / MAD).

    ``epoch`` is ``(channels, samples)``. Missing values (NaN/Inf) are ignored
    when computing statistics, and replaced with zeros post normalisation.
    """

    normalised = np.zeros_like(epoch, dtype=np.float32)
    for ch_idx in range(epoch.shape[0]):
        channel = epoch[ch_idx].astype(np.float32)
        mask = np.isfinite(channel)
        if not mask.any():
            normalised[ch_idx] = 0.0
            continue
        valid = channel[mask]
        median = float(np.median(valid))
        mad = float(np.median(np.abs(valid - median)))
        scale = mad * 1.4826
        if not np.isfinite(scale) or scale < 1e-6:
            std = float(np.std(valid))
            if not np.isfinite(std) or std < 1e-6:
                scale = 1.0
            else:
                scale = std
        centred = channel.copy()
        centred[mask] = (centred[mask] - median) / scale
        centred[~mask] = 0.0
        centred[~np.isfinite(centred)] = 0.0
        normalised[ch_idx] = centred.astype(np.float32)
    return normalised


# ---------------------------------------------------------------------------
# Validator scaffolding
# ---------------------------------------------------------------------------


class PacketValidator:
    """Light-weight validator for SleepFM Packet 1 outputs."""

    def validate_subject(self, subject_id: str, output_root: Path) -> None:
        x_dir = output_root / "X" / subject_id
        y_path = output_root / "Y" / f"{subject_id}.pickle"
        done_path = output_root / ".done" / f"{subject_id}.done"

        if not x_dir.exists() or not x_dir.is_dir():
            raise FileNotFoundError(f"Missing X directory for {subject_id}: {x_dir}")
        if not y_path.exists():
            raise FileNotFoundError(f"Missing Y pickle for {subject_id}: {y_path}")
        if not done_path.exists():
            raise FileNotFoundError(f"Missing .done sentinel for {subject_id}: {done_path}")

        with open(y_path, "rb") as handle:
            labels = pickle.load(handle)

        for epoch_file, label in labels.items():
            epoch_path = x_dir / epoch_file
            if not epoch_path.exists():
                raise FileNotFoundError(f"Label references missing epoch file: {epoch_path}")
            if label is not None and label not in LABEL_MAP.values():
                raise ValueError(f"Invalid label '{label}' in {y_path}")


# ---------------------------------------------------------------------------
# Subject processing
# ---------------------------------------------------------------------------


def discover_subjects(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("*.edf"))


def match_annotation(edf_path: Path) -> Optional[Path]:
    stem = edf_path.stem.lower()
    parent = edf_path.parent
    candidates: List[Path] = []
    for suffix in (".st", ".ST", ".txt", ".TXT"):
        candidate = edf_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    for candidate in parent.iterdir():
        if candidate.suffix.lower() in {".st", ".txt"} and stem in candidate.stem.lower():
            candidates.append(candidate)
    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def compute_epoch_label(start: float, end: float, segments: Sequence[StageSegment]) -> Optional[str]:
    best_label: Optional[str] = None
    best_overlap = 0.0
    for segment in segments:
        overlap = min(end, segment.end) - max(start, segment.start)
        if overlap <= 0:
            continue
        if segment.label is None:
            continue
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = segment.label
    return best_label


def create_epoch(
    channel_data: Dict[str, np.ndarray],
    start_idx: int,
    stop_idx: int,
    epoch_samples: int,
) -> np.ndarray:
    pad = epoch_samples - (stop_idx - start_idx)
    epoch_channels: List[np.ndarray] = []
    for channel_name in ALL_CHANNELS:
        if channel_name in channel_data:
            segment = channel_data[channel_name][start_idx:stop_idx]
        else:
            segment = np.zeros(max(stop_idx - start_idx, 0), dtype=np.float32)
        if pad > 0:
            if segment.size:
                segment = np.pad(segment, (0, pad), mode="constant", constant_values=np.nan)
            else:
                segment = np.full(stop_idx - start_idx + pad, np.nan, dtype=np.float32)
        epoch_channels.append(segment.astype(np.float32))
    epoch = np.stack(epoch_channels, axis=0)
    return robust_zscore(epoch)


def process_subject(
    edf_path: Path,
    annotation_path: Path,
    output_root: Path,
    epoch_seconds: float,
    target_sfreq: float,
    min_tail_seconds: float,
    overwrite: bool,
    validator: Optional[PacketValidator],
) -> None:
    subject_id = edf_path.stem
    LOGGER.info("Processing %s", subject_id)

    subject_x_dir = output_root / "X" / subject_id
    subject_y_path = output_root / "Y" / f"{subject_id}.pickle"
    log_path = output_root / "logs" / f"{subject_id}_channels.json"
    done_path = output_root / ".done" / f"{subject_id}.done"

    if done_path.exists() and not overwrite:
        LOGGER.info("Skipping %s (already processed)", subject_id)
        return

    if overwrite:
        for path in (subject_x_dir, subject_y_path, log_path, done_path):
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    original_sfreq = float(raw.info["sfreq"])

    channel_mapping: Dict[str, str] = {}
    for raw_name in raw.ch_names:
        canonical = canon_name(raw_name)
        if canonical and canonical not in channel_mapping:
            channel_mapping[canonical] = raw_name

    missing_channels = [ch for ch in ALL_CHANNELS if ch not in channel_mapping]
    critical_missing = [ch for ch in CHANNEL_DATA["Sleep_Stages"] if ch not in channel_mapping]

    channel_log = {
        "subject_id": subject_id,
        "edf_path": str(edf_path),
        "annotation_path": str(annotation_path),
        "raw_channels": raw.ch_names,
        "canonical_mapping": channel_mapping,
        "missing_channels": missing_channels,
        "critical_missing": critical_missing,
    }
    atomic_save_json(channel_log, log_path)

    if critical_missing:
        LOGGER.warning("Skipping %s due to missing critical Sleep_Stages channels: %s", subject_id, critical_missing)
        return

    picked_names = [channel_mapping[name] for name in channel_mapping]
    raw_selected = raw.copy().pick(picked_names, verbose="ERROR")
    raw_selected.load_data()
    raw_selected.resample(sfreq=target_sfreq, npad="auto")

    resampled_data = raw_selected.get_data()
    resampled_sfreq = float(raw_selected.info["sfreq"])
    total_samples = resampled_data.shape[1]
    recording_duration = total_samples / resampled_sfreq

    channel_data: Dict[str, np.ndarray] = {}
    for idx, canonical in enumerate(channel_mapping):
        channel_data[canonical] = resampled_data[idx].astype(np.float32)

    for channel_name in missing_channels:
        channel_data[channel_name] = np.zeros(total_samples, dtype=np.float32)

    annotation_records = parse_annotation_file(annotation_path, epoch_seconds)
    segments = build_segments(annotation_records, recording_duration, epoch_seconds)

    epoch_samples = int(epoch_seconds * resampled_sfreq)
    min_tail_samples = int(min_tail_seconds * resampled_sfreq)

    labels: Dict[str, Optional[str]] = {}

    n_full_epochs = total_samples // epoch_samples
    remainder = total_samples % epoch_samples

    subject_x_dir.mkdir(parents=True, exist_ok=True)

    for epoch_idx in range(int(n_full_epochs)):
        start_idx = epoch_idx * epoch_samples
        stop_idx = start_idx + epoch_samples
        epoch = create_epoch(channel_data, start_idx, stop_idx, epoch_samples)
        start_sec = start_idx / resampled_sfreq
        end_sec = stop_idx / resampled_sfreq
        label = compute_epoch_label(start_sec, end_sec, segments)
        epoch_filename = f"{epoch_idx:05d}.npy"
        atomic_save_numpy(epoch, subject_x_dir / epoch_filename)
        labels[epoch_filename] = label

    if remainder >= min_tail_samples:
        start_idx = total_samples - remainder
        stop_idx = total_samples
        epoch = create_epoch(channel_data, start_idx, stop_idx, epoch_samples)
        epoch_filename = f"{n_full_epochs:05d}.npy"
        atomic_save_numpy(epoch, subject_x_dir / epoch_filename)
        labels[epoch_filename] = None

    atomic_save_pickle(labels, subject_y_path)
    atomic_touch(done_path, "ok\n")

    if validator is not None:
        validator.validate_subject(subject_id, output_root)

    LOGGER.info("Finished %s", subject_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CAP Packet 1 to SleepFM format")
    parser.add_argument("input_root", type=Path, help="Root directory containing EDF files")
    parser.add_argument("output_root", type=Path, help="Destination root directory")
    parser.add_argument("--epoch-seconds", type=float, default=30.0, help="Epoch duration in seconds")
    parser.add_argument("--target-sfreq", type=float, default=256.0, help="Target sampling frequency")
    parser.add_argument("--min-tail-seconds", type=float, default=15.0, help="Minimum tail duration to keep")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess subjects even if .done exists")
    parser.add_argument("--run-validator", action="store_true", help="Run the Packet 1 validator after writing outputs")
    parser.add_argument("--subjects", nargs="*", help="Optional subset of subject identifiers to process")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")
    mne.set_log_level("ERROR")

    set_global_seed(args.seed)

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "X").mkdir(exist_ok=True)
    (output_root / "Y").mkdir(exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    (output_root / ".done").mkdir(exist_ok=True)

    validator = PacketValidator() if args.run_validator else None

    edf_files = list(discover_subjects(input_root))
    if not edf_files:
        LOGGER.warning("No EDF files discovered under %s", input_root)
        return

    subjects_filter = {s.lower() for s in args.subjects} if args.subjects else None

    for edf_path in edf_files:
        subject_id = edf_path.stem
        if subjects_filter and subject_id.lower() not in subjects_filter:
            continue
        annotation_path = match_annotation(edf_path)
        if not annotation_path:
            LOGGER.warning("No annotation file found for %s", edf_path)
            continue
        try:
            process_subject(
                edf_path=edf_path,
                annotation_path=annotation_path,
                output_root=output_root,
                epoch_seconds=args.epoch_seconds,
                target_sfreq=args.target_sfreq,
                min_tail_seconds=args.min_tail_seconds,
                overwrite=args.overwrite,
                validator=validator,
            )
        except Exception:  # pragma: no cover - surfaced in CLI log
            LOGGER.exception("Failed to process %s", edf_path)


if __name__ == "__main__":  # pragma: no cover
    main()
