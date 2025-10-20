"""Robust parsers for CAP staging and event annotation files."""
from __future__ import annotations

import csv
import dataclasses
import math
import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import pandas as pd

DEFAULT_CHUNK_DURATION = 30.0


@dataclasses.dataclass
class StageAnnotation:
    """Representation of a single sleep stage annotation."""

    onset: float
    duration: float
    label: str

    @property
    def end(self) -> float:
        return self.onset + self.duration


def _clean_line(raw_line: str) -> str:
    return raw_line.strip().replace(";", " ").replace("\t", " ")


_TIME_RE = re.compile(r"^(?:(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d+))?)$")


def _parse_time_token(token: str) -> Optional[float]:
    match = _TIME_RE.match(token)
    if match:
        hours, minutes, seconds, fraction = match.groups()
        total = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
        if fraction:
            total += float(f"0.{fraction}")
        return float(total)
    try:
        return float(token)
    except ValueError:
        return None


def _iter_stage_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = _clean_line(raw_line)
            if not line or line.startswith(("#", "*")):
                continue
            yield line


def parse_stage_file(path: Path, chunk_duration: float = DEFAULT_CHUNK_DURATION) -> List[StageAnnotation]:
    """Parse a CAP stage file into :class:`StageAnnotation` objects.

    The CAP release mixes several formatting styles. This parser tries a series
    of increasingly permissive heuristics so that most variants are captured
    without manual editing.
    """

    annotations: List[StageAnnotation] = []
    for line in _iter_stage_lines(path):
        if "," in line:
            tokens = [token.strip() for token in line.split(",") if token.strip()]
        else:
            tokens = line.split()

        annotation = _parse_tokens(tokens, chunk_duration)
        if annotation is not None:
            annotations.append(annotation)

    annotations.sort(key=lambda ann: ann.onset)
    return annotations


def _parse_tokens(tokens: Sequence[str], chunk_duration: float) -> Optional[StageAnnotation]:
    if not tokens:
        return None

    # Case 1: explicit onset, duration, label columns.
    if len(tokens) >= 3 and _is_number(tokens[0]) and _is_number(tokens[1]):
        onset = float(tokens[0])
        duration = float(tokens[1])
        label = " ".join(tokens[2:])
        return StageAnnotation(onset=onset, duration=duration, label=label)

    # Case 2: HH:MM:SS timestamp followed by label and optional duration.
    time_value = _parse_time_token(tokens[0])
    if time_value is not None:
        if len(tokens) == 2:
            label = tokens[1]
            duration = chunk_duration
        elif len(tokens) >= 3 and _is_number(tokens[1]):
            duration = float(tokens[1])
            label = " ".join(tokens[2:])
        else:
            label = " ".join(tokens[1:])
            duration = chunk_duration
        return StageAnnotation(onset=float(time_value), duration=duration, label=label)

    # Case 3: Epoch index followed by label.
    if tokens[0].isdigit():
        epoch_index = int(tokens[0]) - 1 if int(tokens[0]) > 0 else int(tokens[0])
        label = " ".join(tokens[1:]) if len(tokens) > 1 else ""
        onset = float(epoch_index) * chunk_duration
        return StageAnnotation(onset=onset, duration=chunk_duration, label=label)

    return None


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    else:
        return True


def merge_annotations(annotations: Sequence[StageAnnotation]) -> List[StageAnnotation]:
    """Merge sequential annotations with the same label."""
    if not annotations:
        return []

    merged: List[StageAnnotation] = []
    current = annotations[0]
    for annotation in annotations[1:]:
        if annotation.label == current.label and math.isclose(
            annotation.onset, current.end, rel_tol=0, abs_tol=1e-6
        ):
            current = StageAnnotation(
                onset=current.onset,
                duration=current.duration + annotation.duration,
                label=current.label,
            )
        else:
            merged.append(current)
            current = annotation
    merged.append(current)
    return merged


def annotations_to_epochs(
    annotations: Sequence[StageAnnotation],
    total_duration: float,
    chunk_duration: float = DEFAULT_CHUNK_DURATION,
) -> pd.DataFrame:
    """Convert annotations to a per-epoch DataFrame.

    Parameters
    ----------
    annotations:
        Sequence of stage annotations sorted by onset.
    total_duration:
        Total duration of the recording in seconds.
    chunk_duration:
        Duration of each epoch.
    """

    annotations = merge_annotations(sorted(annotations, key=lambda ann: ann.onset))
    n_epochs = int(total_duration // chunk_duration)
    epochs = []

    pointer = 0
    for epoch_idx in range(n_epochs):
        start = epoch_idx * chunk_duration
        end = start + chunk_duration

        while pointer < len(annotations) - 1 and annotations[pointer].end <= start:
            pointer += 1

        label = annotations[pointer].label if annotations else ""
        # If the current annotation starts after the epoch, fall back to the
        # previous label (data gap). Otherwise, ensure the annotation spans the
        # epoch midpoint.
        if annotations:
            ann = annotations[pointer]
            if ann.onset > start and pointer > 0:
                ann = annotations[pointer - 1]
            if ann.onset <= start < ann.end:
                label = ann.label
        epochs.append({
            "epoch": epoch_idx,
            "start_sec": start,
            "end_sec": end,
            "label": label,
        })

    return pd.DataFrame(epochs)


def parse_cap_event_file(path: Path) -> pd.DataFrame:
    """Parse CAP event annotation text files into a DataFrame."""
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
        if df.shape[1] <= 1:
            df = pd.read_csv(path, sep=",", engine="python")
    except Exception:
        # Fallback to CSV reader for irregular separators.
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle, delimiter="\t")
            rows = [row for row in reader if row]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)

    df.columns = [f"col_{idx}" for idx in range(df.shape[1])]
    return df


__all__ = [
    "StageAnnotation",
    "parse_stage_file",
    "merge_annotations",
    "annotations_to_epochs",
    "parse_cap_event_file",
]
