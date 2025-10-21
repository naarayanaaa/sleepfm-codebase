from __future__ import annotations

import argparse
import csv
import multiprocessing
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mne
import numpy as np
from loguru import logger
from tqdm import tqdm

from config import (
    ALL_CHANNELS,
    CHANNEL_ALIASES,
    EVENT_TO_ID,
    PATH_TO_PROCESSED_DATA,
    PATH_TO_RAW_DATA,
    canonical_stage,
    resolve_additional_label,
)


ID_TO_STAGE = {value: key for key, value in EVENT_TO_ID.items()}


@dataclass
class CAPRecord:
    identifier: str
    psg_path: Path
    hypnogram_path: Path
    cap_annotations_path: Optional[Path] = None


def discover_cap_records(root: Path) -> List[CAPRecord]:
    """Discover PSG/Hypnogram pairs under ``root``.

    The CAP dataset stores files as ``<subject>-PSG.edf`` and
    ``<subject>-Hypnogram.edf`` (or ``.txt``).  We conservatively look for
    both prefixes and only keep subjects that provide a PSG recording and a
    hypnogram file.
    """

    psg_files: Dict[str, Path] = {}
    hypnogram_files: Dict[str, Path] = {}
    cap_annotation_files: Dict[str, Path] = {}

    for path in root.rglob("*.edf"):
        stem = path.stem
        if stem.endswith("-PSG"):
            base = stem[: -len("-PSG")]
            psg_files[base] = path
        elif stem.endswith("-Hypnogram") or stem.endswith("-HypnogramAASM"):
            base = stem.split("-Hypnogram")[0]
            hypnogram_files[base] = path
        elif stem.endswith("-CAP"):
            base = stem[: -len("-CAP")]
            cap_annotation_files[base] = path

    for path in root.rglob("*.txt"):
        stem = path.stem
        if stem.endswith("-Hypnogram"):
            base = stem[: -len("-Hypnogram")]
            # Prefer EDF hypnograms when available.
            hypnogram_files.setdefault(base, path)
        elif stem.endswith("-CAP"):
            base = stem[: -len("-CAP")]
            cap_annotation_files.setdefault(base, path)

    records: List[CAPRecord] = []
    for base, psg_path in sorted(psg_files.items()):
        hypnogram_path = hypnogram_files.get(base)
        if not hypnogram_path:
            logger.warning("Skipping %s because no hypnogram file was found", base)
            continue
        records.append(
            CAPRecord(
                identifier=base,
                psg_path=psg_path,
                hypnogram_path=hypnogram_path,
                cap_annotations_path=cap_annotation_files.get(base),
            )
        )

    return records


def load_metadata(
    metadata_csv: Optional[Path],
    id_column: str,
    label_column: str,
    label_key: str,
) -> Dict[str, Dict[str, str]]:
    if not metadata_csv:
        return {}

    mapping: Dict[str, Dict[str, str]] = {}
    with metadata_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            identifier = row.get(id_column)
            if not identifier:
                continue
            identifier = identifier.strip()
            mapping.setdefault(identifier, {})[label_key] = row.get(label_column, "").strip()
    logger.info(
        "Loaded %d metadata rows with label column '%s'",
        len(mapping),
        label_column,
    )
    return mapping


def _match_channel(raw: mne.io.BaseRaw, canonical_name: str) -> Optional[int]:
    candidates = [canonical_name] + CHANNEL_ALIASES.get(canonical_name, [])
    name_to_index = {name.lower(): idx for idx, name in enumerate(raw.ch_names)}
    simplified_target = canonical_name.replace(" ", "").replace("-", "").lower()

    for candidate in candidates:
        if candidate in raw.ch_names:
            return raw.ch_names.index(candidate)
        key = candidate.lower()
        if key in name_to_index:
            return name_to_index[key]
        simplified = candidate.replace(" ", "").replace("-", "").lower()
        if simplified in name_to_index:
            return name_to_index[simplified]

    for idx, name in enumerate(raw.ch_names):
        simplified = name.replace(" ", "").replace("-", "").lower()
        if simplified == simplified_target:
            return idx
    return None


def _prepare_annotations(
    hypnogram_path: Path,
    chunk_duration: float,
    stage_key: str = "sleep_stage",
) -> mne.Annotations:
    try:
        annotations = mne.read_annotations(str(hypnogram_path))
    except Exception:  # pragma: no cover - relies on external files
        annotations = _parse_hypnogram_text(hypnogram_path, chunk_duration)

    canonical_descriptions: List[str] = []
    for desc in annotations.description:
        canonical = canonical_stage(desc)
        canonical_descriptions.append(canonical)

    return mne.Annotations(
        onset=annotations.onset,
        duration=annotations.duration,
        description=canonical_descriptions,
    )


def _parse_hypnogram_text(path: Path, chunk_duration: float) -> mne.Annotations:
    onsets: List[float] = []
    durations: List[float] = []
    descriptions: List[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = [item for item in stripped.replace(",", " ").split() if item]
            if len(parts) < 2:
                continue
            try:
                epoch_index = int(parts[0])
                onset = epoch_index * chunk_duration
                duration = chunk_duration
                label = parts[1]
            except ValueError:
                try:
                    onset = float(parts[0])
                    duration = float(parts[1]) if len(parts) > 2 else chunk_duration
                    label = parts[2] if len(parts) > 2 else parts[1]
                except ValueError:
                    continue
            onsets.append(onset)
            durations.append(duration)
            descriptions.append(label)

    if not onsets:
        raise RuntimeError(f"Unable to parse hypnogram file: {path}")

    return mne.Annotations(onset=np.array(onsets), duration=np.array(durations), description=descriptions)


def _build_epoch_events(
    annotations: mne.Annotations, sfreq: float, chunk_duration: float
) -> Tuple[np.ndarray, List[str]]:
    events: List[List[int]] = []
    labels: List[str] = []

    for onset, duration, desc in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        stage = canonical_stage(desc)
        if stage not in EVENT_TO_ID:
            continue
        start = max(0.0, onset)
        end = max(start, onset + duration)
        num_epochs = int(np.floor((end - start) / chunk_duration))
        if num_epochs <= 0:
            continue
        for idx in range(num_epochs):
            epoch_start = start + idx * chunk_duration
            sample_index = int(round(epoch_start * sfreq))
            events.append([sample_index, 0, EVENT_TO_ID[stage]])
            labels.append(stage)

    if not events:
        return np.empty((0, 3), dtype=int), []

    return np.asarray(events, dtype=int), labels


def _extract_epochs(
    record: CAPRecord,
    output_root: Path,
    chunk_duration: float,
    target_sampling_rate: int,
    metadata: Dict[str, Dict[str, str]],
    label_key: str,
) -> int:
    raw = mne.io.read_raw_edf(record.psg_path, preload=True, verbose="ERROR")
    annotations = _prepare_annotations(record.hypnogram_path, chunk_duration)

    raw.set_annotations(annotations, emit_warning=True)
    raw.crop(tmax=annotations.onset[-1] + annotations.duration[-1], include_tmax=False)

    if target_sampling_rate and not np.isclose(raw.info["sfreq"], target_sampling_rate):
        raw.resample(target_sampling_rate, npad="auto")
    sfreq = raw.info["sfreq"]

    events, epoch_labels = _build_epoch_events(annotations, sfreq, chunk_duration)
    if len(events) == 0:
        logger.warning("No valid epochs extracted for %s", record.identifier)
        return 0

    try:
        epochs = mne.Epochs(
            raw=raw,
            events=events,
            event_id=EVENT_TO_ID,
            tmin=0.0,
            tmax=chunk_duration - 1.0 / sfreq,
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            on_missing="ignore",
        )
    except Exception as exc:  # pragma: no cover - depends on EDF quirks
        logger.exception("Failed to create epochs for %s: %s", record.identifier, exc)
        return 0

    channel_indices: List[int] = []
    for channel in ALL_CHANNELS:
        idx = _match_channel(raw, channel)
        if idx is None:
            logger.warning(
                "Skipping %s because channel '%s' could not be matched", record.identifier, channel
            )
            return 0
        channel_indices.append(idx)

    patient_x = output_root / "X" / record.identifier
    patient_y = output_root / "Y" / f"{record.identifier}.pickle"
    patient_x.mkdir(parents=True, exist_ok=True)
    patient_y.parent.mkdir(parents=True, exist_ok=True)

    metadata_label = metadata.get(record.identifier, {})

    labels: Dict[str, Dict[str, str]] = {}
    saved_epochs = 0

    for idx, epoch in enumerate(epochs):
        data = epoch[channel_indices, :]
        if data.shape[0] != len(ALL_CHANNELS):
            continue
        file_name = f"{record.identifier}_{idx}.npy"
        np.save(patient_x / file_name, data.astype(np.float32))
        stage_id = int(epochs.events[idx, 2])
        stage = ID_TO_STAGE.get(stage_id)
        if stage is None:
            stage = epoch_labels[idx] if idx < len(epoch_labels) else "Unknown"
        entry = {
            "sleep_stage": stage,
            "subject_id": record.identifier,
        }
        if metadata_label:
            for key, value in metadata_label.items():
                entry[key] = resolve_additional_label(key, value)
        labels[file_name] = entry
        saved_epochs += 1

    if saved_epochs == 0:
        logger.warning("No epochs were saved for %s", record.identifier)
        return 0

    np.save(patient_x / "sampling_rate.npy", np.array([sfreq], dtype=np.float32))

    with patient_y.open("wb") as handle:
        import pickle

        pickle.dump(labels, handle)

    return saved_epochs


def _worker(
    records: Sequence[CAPRecord],
    output_root: Path,
    chunk_duration: float,
    target_sampling_rate: int,
    metadata: Dict[str, Dict[str, str]],
    label_key: str,
) -> int:
    count = 0
    for record in tqdm(records, desc="records", leave=False):
        count += _extract_epochs(
            record=record,
            output_root=output_root,
            chunk_duration=chunk_duration,
            target_sampling_rate=target_sampling_rate,
            metadata=metadata,
            label_key=label_key,
        )
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess CAP EDF files for SleepFM")
    parser.add_argument("--data_path", type=str, default=None, help="Path to raw CAP files")
    parser.add_argument("--save_path", type=str, default=None, help="Path for processed output")
    parser.add_argument("--num_files", type=int, default=-1, help="Number of subjects to process")
    parser.add_argument(
        "--chunk_duration", type=float, default=30.0, help="Duration of each epoch in seconds"
    )
    parser.add_argument(
        "--num_threads", type=int, default=4, help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--target_sampling_rate",
        type=int,
        default=256,
        help="Sampling rate (Hz) after resampling",
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
        help="Optional CSV file containing subject-level labels",
    )
    parser.add_argument(
        "--metadata_id_column",
        type=str,
        default="record_id",
        help="Column in metadata CSV that matches the subject identifier",
    )
    parser.add_argument(
        "--metadata_label_column",
        type=str,
        default="target",
        help="Column in metadata CSV that stores the downstream label",
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default="disorder",
        help="Dictionary key under which metadata labels are stored",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_path) if args.data_path else Path(PATH_TO_RAW_DATA)
    output_root = Path(args.save_path) if args.save_path else Path(PATH_TO_PROCESSED_DATA)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Looking for CAP recordings in %s", data_root)
    records = discover_cap_records(data_root)

    if args.num_files != -1:
        records = records[: args.num_files]

    if not records:
        logger.error("No CAP records were found under %s", data_root)
        return

    metadata = load_metadata(
        Path(args.metadata_csv) if args.metadata_csv else None,
        id_column=args.metadata_id_column,
        label_column=args.metadata_label_column,
        label_key=args.label_key,
    )

    logger.info("Processing %d records", len(records))

    num_threads = max(1, args.num_threads)
    record_splits = np.array_split(records, num_threads)  # type: ignore[name-defined]

    tasks = [
        (
            split.tolist(),
            output_root,
            args.chunk_duration,
            args.target_sampling_rate,
            metadata,
            args.label_key,
        )
        for split in record_splits
        if len(split) > 0
    ]

    total_epochs = 0
    if num_threads == 1 or len(tasks) == 1:
        for task in tasks:
            total_epochs += _worker(*task)
    else:
        with multiprocessing.Pool(processes=len(tasks)) as pool:
            for result in pool.starmap(_worker, tasks):
                total_epochs += result

    logger.info("Saved %d epochs to %s", total_epochs, output_root)


if __name__ == "__main__":
    main()
