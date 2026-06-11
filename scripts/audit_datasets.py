"""Audit ECG/EDA biosignal artifacts for shape, labels, and split readiness."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _as_array(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    return np.asarray(value)


def _counts(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(k): int(v) for k, v in zip(unique, counts)}


def _source_dataset_name(source: str | None) -> str | None:
    if not source:
        return None
    stem = Path(source).stem
    return stem.split("_")[0] if stem else None


def summarize_artifact(artifact: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    """Return a JSON-safe summary for one processed artifact.

    The expected artifact schema is:
    data, labels, participants, dataset, channels, fs, window_sec, overlap, task,
    label_map. Older local files without all metadata are accepted and reported
    with best-effort defaults.
    """
    data = _as_array(artifact.get("data"))
    labels = _as_array(artifact.get("labels"))
    participants = _as_array(artifact.get("participants", artifact.get("groups")))
    channels = list(artifact.get("channels", []))

    if data.size:
        finite = bool(np.isfinite(data).all()) if np.issubdtype(data.dtype, np.number) else False
        data_shape = tuple(int(x) for x in data.shape)
        n_windows = int(data.shape[0])
        window_samples = int(data.shape[1]) if data.ndim >= 2 else None
        n_channels = int(data.shape[2]) if data.ndim >= 3 else (len(channels) or None)
    else:
        finite = None
        data_shape = None
        n_windows = int(labels.size)
        window_samples = None
        n_channels = len(channels) or None

    full_label_set = set(np.unique(labels).tolist()) if labels.size else set()
    missing_classes: list[str] = []
    if participants.size == labels.size and labels.size:
        for participant in np.unique(participants):
            participant_labels = set(np.unique(labels[participants == participant]).tolist())
            if participant_labels != full_label_set:
                missing_classes.append(str(participant))

    return {
        "source": source,
        "dataset": artifact.get("dataset") or _source_dataset_name(source),
        "task": artifact.get("task"),
        "channels": channels,
        "fs": artifact.get("fs"),
        "window_sec": artifact.get("window_sec"),
        "overlap": artifact.get("overlap"),
        "data_shape": data_shape,
        "n_windows": n_windows,
        "window_samples": window_samples,
        "n_channels": n_channels,
        "finite": finite,
        "label_counts": _counts(labels),
        "n_participants": int(len(np.unique(participants))) if participants.size else 0,
        "participant_counts": _counts(participants),
        "participants_missing_classes": missing_classes,
        "schema_keys": sorted(str(k) for k in artifact.keys()),
    }


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} did not contain a dictionary artifact")
    return value


def audit_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        summaries.append(summarize_artifact(load_pickle(path), source=str(path)))
    return summaries


def _write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "dataset",
        "task",
        "channels",
        "fs",
        "window_sec",
        "overlap",
        "data_shape",
        "n_windows",
        "n_participants",
        "label_counts",
        "finite",
        "participants_missing_classes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: json.dumps(summary.get(field)) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="Pickle artifacts to audit")
    parser.add_argument("--glob", default="data/processed/*.pkl", help="Glob used when inputs are omitted")
    parser.add_argument("--json-output", type=Path, default=Path("results/dataset_audit.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("results/dataset_audit.csv"))
    args = parser.parse_args()

    paths = args.inputs or sorted(Path(".").glob(args.glob))
    summaries = audit_paths(paths)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    _write_csv(args.csv_output, summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
