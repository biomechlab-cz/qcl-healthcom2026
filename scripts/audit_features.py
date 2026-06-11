"""Audit ECG/EDA feature tables for split readiness and numeric validity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def expand_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        text = str(value)
        if any(token in text for token in ("*", "?", "[")):
            parent = value.parent if str(value.parent) not in ("", ".") else Path(".")
            paths.extend(sorted(parent.glob(value.name)))
        else:
            paths.append(value)
    if not paths:
        raise FileNotFoundError("No feature files matched")
    return paths


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in values.value_counts(dropna=False).sort_index().items()}


def summarize_feature_file(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path)
    numeric = frame.select_dtypes(include=[np.number])
    feature_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "dataset",
            "task",
            "signal_set",
            "subject_id",
            "participant",
            "label",
            "window_idx",
        }
    ]
    finite = bool(np.isfinite(numeric.to_numpy(dtype=float)).all()) if not numeric.empty else True
    labels = frame["label"] if "label" in frame else pd.Series(dtype=int)
    subjects = frame["subject_id"].astype(str) if "subject_id" in frame else pd.Series(dtype=str)

    full_label_set = set(labels.unique().tolist()) if not labels.empty else set()
    subjects_missing_classes: list[str] = []
    if not labels.empty and not subjects.empty:
        for subject in sorted(subjects.unique()):
            subject_labels = set(labels[subjects == subject].unique().tolist())
            if subject_labels != full_label_set:
                subjects_missing_classes.append(str(subject))

    return {
        "source": str(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "feature_columns": feature_columns,
        "n_features": int(len(feature_columns)),
        "dataset": str(frame["dataset"].iloc[0]) if "dataset" in frame and len(frame) else None,
        "task": str(frame["task"].iloc[0]) if "task" in frame and len(frame) else None,
        "signal_set": str(frame["signal_set"].iloc[0]) if "signal_set" in frame and len(frame) else None,
        "label_counts": _counts(labels) if not labels.empty else {},
        "n_subjects": int(subjects.nunique()) if not subjects.empty else 0,
        "finite_numeric": finite,
        "nan_counts": {column: int(count) for column, count in frame.isna().sum().items() if count},
        "subjects_missing_classes": subjects_missing_classes,
    }


def audit(paths: Iterable[Path]) -> list[dict[str, object]]:
    return [summarize_feature_file(path) for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path, default=Path("results/feature_audit.json"))
    args = parser.parse_args()

    summaries = audit(expand_inputs(args.inputs))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
