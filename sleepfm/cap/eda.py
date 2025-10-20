"""Exploratory analysis utilities for the CAP Sleep Database."""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import pandas as pd
from loguru import logger

from sleepfm.cap.constants import CAP_CHANNEL_ALIASES, CAP_STAGE_ALIASES
from sleepfm.cap.stage_parser import (
    annotations_to_epochs,
    parse_cap_event_file,
    parse_stage_file,
)
from sleepfm.config import ALL_CHANNELS, LABEL_MAP

DEFAULT_CHUNK_DURATION = 30.0


@dataclasses.dataclass
class SubjectRecord:
    subject_id: str
    edf_path: Path
    stage_path: Optional[Path]
    event_path: Optional[Path]


def discover_subjects(data_root: Path) -> List[SubjectRecord]:
    """Discover CAP subject files following the rbd*.edf naming convention."""
    edf_files = sorted(data_root.glob("rbd*.edf"))
    subjects: List[SubjectRecord] = []
    for edf_path in edf_files:
        subject_id = edf_path.stem
        stage_path = edf_path.with_suffix(".edf.st")
        if not stage_path.exists():
            # Some releases use .st without the intermediate suffix
            alt_stage = edf_path.with_suffix(".st")
            stage_path = alt_stage if alt_stage.exists() else None
        event_path = edf_path.with_suffix(".txt")
        if not event_path.exists():
            event_path = None
        subjects.append(
            SubjectRecord(
                subject_id=subject_id,
                edf_path=edf_path,
                stage_path=stage_path,
                event_path=event_path,
            )
        )
    return subjects


def _build_channel_lookup(raw: mne.io.BaseRaw) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for ch_name in raw.ch_names:
        lookup[ch_name.upper()] = ch_name
    return lookup


def _match_channel(target: str, lookup: Dict[str, str]) -> Optional[str]:
    candidates = [target] + CAP_CHANNEL_ALIASES.get(target, [])
    for candidate in candidates:
        key = candidate.upper()
        if key in lookup:
            return lookup[key]
    return None


from typing import Tuple


def summarise_channels(subject: SubjectRecord) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        raw = mne.io.read_raw_edf(subject.edf_path, preload=False, verbose="ERROR")
    except Exception as exc:
        logger.warning(f"Failed to read {subject.edf_path}: {exc}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    lookup = _build_channel_lookup(raw)
    sfreq = float(raw.info.get("sfreq", 0.0))
    channel_types = raw.get_channel_types()
    rows = []
    for ch_name, ch_type in zip(raw.ch_names, channel_types):
        rows.append(
            {
                "subject_id": subject.subject_id,
                "channel": ch_name,
                "type": ch_type,
                "sfreq": sfreq,
            }
        )

    matched = []
    missing = []
    for canonical in ALL_CHANNELS:
        matched_name = _match_channel(canonical, lookup)
        if matched_name:
            matched.append({"subject_id": subject.subject_id, "canonical": canonical, "edf_channel": matched_name})
        else:
            missing.append({"subject_id": subject.subject_id, "canonical": canonical})

    raw.close()
    df_channels = pd.DataFrame(rows)
    df_matched = pd.DataFrame(matched)
    df_missing = pd.DataFrame(missing)
    if not df_missing.empty:
        logger.debug(f"{subject.subject_id}: missing channels {df_missing['canonical'].tolist()}")
    return df_channels, df_matched, df_missing


def _normalise_stage_label(label: str) -> str:
    label = label.strip()
    if label in CAP_STAGE_ALIASES:
        return CAP_STAGE_ALIASES[label]
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    if label.title() in LABEL_MAP:
        return LABEL_MAP[label.title()]
    return label


def summarise_stages(subject: SubjectRecord, total_duration: float, chunk_duration: float = DEFAULT_CHUNK_DURATION) -> pd.DataFrame:
    if subject.stage_path is None:
        return pd.DataFrame()
    try:
        annotations = parse_stage_file(subject.stage_path, chunk_duration=chunk_duration)
    except Exception as exc:
        logger.warning(f"Failed to parse stage file for {subject.subject_id}: {exc}")
        return pd.DataFrame()

    if not annotations:
        return pd.DataFrame()

    epochs = annotations_to_epochs(annotations, total_duration=total_duration, chunk_duration=chunk_duration)
    epochs["label"] = epochs["label"].map(_normalise_stage_label)
    counts = epochs["label"].value_counts().rename_axis("label").reset_index(name="count")
    counts.insert(0, "subject_id", subject.subject_id)
    return counts


def summarise_cap_events(subject: SubjectRecord) -> pd.DataFrame:
    if subject.event_path is None:
        return pd.DataFrame()
    try:
        df = parse_cap_event_file(subject.event_path)
    except Exception as exc:
        logger.warning(f"Failed to parse CAP events for {subject.subject_id}: {exc}")
        return pd.DataFrame()
    if df.empty:
        return df

    # Heuristic: the last column usually contains the textual label.
    label_col = df.columns[-1]
    label_counts = df[label_col].value_counts().rename_axis("event").reset_index(name="count")
    label_counts.insert(0, "subject_id", subject.subject_id)
    return label_counts


def run_eda(data_root: Path, output_dir: Path, chunk_duration: float = DEFAULT_CHUNK_DURATION) -> Dict[str, Path]:
    """Run the exploratory analysis workflow and persist summary artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    subjects = discover_subjects(data_root)
    if not subjects:
        raise FileNotFoundError(f"No CAP EDF files found in {data_root}")

    inventory_rows = []
    channel_rows: List[pd.DataFrame] = []
    matched_rows: List[pd.DataFrame] = []
    missing_rows: List[pd.DataFrame] = []
    stage_rows: List[pd.DataFrame] = []
    event_rows: List[pd.DataFrame] = []

    for subject in subjects:
        record = {
            "subject_id": subject.subject_id,
            "edf_path": str(subject.edf_path),
            "stage_path": str(subject.stage_path) if subject.stage_path else None,
            "event_path": str(subject.event_path) if subject.event_path else None,
        }
        inventory_rows.append(record)

        try:
            raw = mne.io.read_raw_edf(subject.edf_path, preload=False, verbose="ERROR")
        except Exception as exc:
            logger.error(f"Failed to open EDF for {subject.subject_id}: {exc}")
            continue

        duration = float(raw.n_times) / float(raw.info.get("sfreq", 1.0))
        record.update({"duration_sec": duration, "sfreq": float(raw.info.get("sfreq", 0.0))})

        channels, matched, missing = summarise_channels(subject)
        if isinstance(channels, pd.DataFrame) and not channels.empty:
            channel_rows.append(channels)
        if isinstance(matched, pd.DataFrame) and not matched.empty:
            matched_rows.append(matched)
        if isinstance(missing, pd.DataFrame) and not missing.empty:
            missing_rows.append(missing)

        stages = summarise_stages(subject, total_duration=duration, chunk_duration=chunk_duration)
        if not stages.empty:
            stage_rows.append(stages)

        events = summarise_cap_events(subject)
        if not events.empty:
            event_rows.append(events)

        raw.close()

    inventory_df = pd.DataFrame(inventory_rows)
    if channel_rows:
        channel_df = pd.concat(channel_rows, ignore_index=True)
    else:
        channel_df = pd.DataFrame()

    if matched_rows:
        matched_df = pd.concat(matched_rows, ignore_index=True)
    else:
        matched_df = pd.DataFrame()

    if missing_rows:
        missing_df = pd.concat(missing_rows, ignore_index=True)
    else:
        missing_df = pd.DataFrame()

    if stage_rows:
        stage_df = pd.concat(stage_rows, ignore_index=True)
    else:
        stage_df = pd.DataFrame()

    if event_rows:
        event_df = pd.concat(event_rows, ignore_index=True)
    else:
        event_df = pd.DataFrame()

    # Aggregated summaries
    channel_counts = (
        channel_df.groupby("channel")["subject_id"].nunique().rename("n_subjects").reset_index()
        if not channel_df.empty
        else pd.DataFrame()
    )
    sampling_rates = (
        channel_df.groupby(["subject_id"])["sfreq"].median().reset_index().rename(columns={"sfreq": "median_sfreq"})
        if not channel_df.empty
        else pd.DataFrame()
    )
    matched_counts = (
        matched_df.groupby(["canonical", "edf_channel"]).size().reset_index(name="count")
        if not matched_df.empty
        else pd.DataFrame()
    )
    missing_counts = (
        missing_df.groupby("canonical").size().reset_index(name="n_missing_subjects")
        if not missing_df.empty
        else pd.DataFrame()
    )
    stage_distribution = (
        stage_df.pivot_table(index="subject_id", columns="label", values="count", fill_value=0)
        if not stage_df.empty
        else pd.DataFrame()
    )

    artifacts = {}
    artifacts["inventory"] = _write_artifact(inventory_df, output_dir / "inventory.csv")
    artifacts["channels"] = _write_artifact(channel_df, output_dir / "channel_audit.csv")
    artifacts["channel_counts"] = _write_artifact(channel_counts, output_dir / "channel_counts.csv")
    artifacts["sampling_rates"] = _write_artifact(sampling_rates, output_dir / "sampling_rates.csv")
    artifacts["matched_channels"] = _write_artifact(matched_df, output_dir / "matched_channels.csv")
    artifacts["missing_channels"] = _write_artifact(missing_counts, output_dir / "missing_channels.csv")
    artifacts["stage_counts"] = _write_artifact(stage_df, output_dir / "stage_counts.csv")
    artifacts["stage_distribution"] = _write_artifact(stage_distribution, output_dir / "stage_distribution.csv")
    artifacts["cap_event_counts"] = _write_artifact(event_df, output_dir / "cap_event_counts.csv")

    summary = {
        "n_subjects": len(inventory_df),
        "n_stage_files": inventory_df["stage_path"].notna().sum(),
        "n_event_files": inventory_df["event_path"].notna().sum(),
        "artifacts": {key: str(path) for key, path in artifacts.items()},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    artifacts["summary"] = summary_path
    logger.info(f"EDA artifacts written to {output_dir}")
    return artifacts


def _write_artifact(df: pd.DataFrame, path: Path) -> Path:
    if df.empty:
        path.write_text("")
    else:
        df.to_csv(path, index=False)
    return path


__all__ = [
    "SubjectRecord",
    "discover_subjects",
    "run_eda",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CAP exploratory analysis")
    parser.add_argument("data_root", type=Path, help="Directory containing rbd*.edf files")
    parser.add_argument("output_dir", type=Path, help="Directory to store EDA artifacts")
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=DEFAULT_CHUNK_DURATION,
        help="Epoch duration in seconds",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_eda(args.data_root, args.output_dir, chunk_duration=args.chunk_duration)


if __name__ == "__main__":
    main()
