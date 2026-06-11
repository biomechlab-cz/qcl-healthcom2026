"""Create harmonized ECG-only and ECG+EDA artifacts for stress/load datasets."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATASET_TASKS: dict[str, dict[str, dict[str, Any]]] = {
    "WESAD": {
        "binary": {
            "keep_labels": [1, 2, 8],
            "label_map": {1: 0, 2: 1, 8: 1},
        },
        "three_class": {
            "keep_labels": [1, 2, 8, 3],
            "label_map": {1: 0, 2: 1, 8: 1, 3: 2},
        },
    },
    "CLAS": {
        "binary": {
            "keep_labels": [0, 1, 3, 6, 8],
            "label_map": {0: 0, 1: 0, 3: 1, 6: 1, 8: 1},
        }
    },
    "StressID": {
        "binary": {
            "label_type": "binary",
            "label_map": {0: 0, 1: 1},
            "task_label_filter": None,
        }
    },
    "SWELL": {
        "binary": {
            "keep_conditions": ["R", "N", "T", "I"],
            "label_map": {"R": 0, "N": 0, "T": 1, "I": 1},
        }
    },
    "CLARE": {
        "binary": {
            "keep_labels": [0, 1],
            "label_map": {0: 0, 1: 1},
            "cl_threshold": 5,
        }
    },
}


RAW_SUBDIRS = {
    "WESAD": Path("WESAD/data"),
    "CLAS": Path("CLAS"),
    "StressID": Path("StressID"),
    "SWELL": Path("SWELL"),
    "CLARE": Path("CLARE"),
}


def _normalize_artifact(
    artifact: dict[str, Any],
    dataset: str,
    task: str,
    channels: list[str],
    fs: int,
    window_sec: float,
    overlap: float,
    label_map: dict[Any, int],
) -> dict[str, Any]:
    data = np.asarray(artifact["data"], dtype=np.float32)
    labels = np.asarray(artifact["labels"], dtype=int)
    participants = np.asarray(artifact["participants"]).astype(str)
    actual_channels = list(artifact.get("channels") or channels)
    return {
        "data": data,
        "labels": labels,
        "participants": participants,
        "dataset": dataset,
        "channels": actual_channels,
        "fs": int(fs),
        "window_sec": float(window_sec),
        "overlap": float(overlap),
        "task": task,
        "label_map": {str(k): int(v) for k, v in label_map.items()},
    }


def preprocess_one(
    dataset: str,
    signal_set: str,
    task: str = "binary",
    fs: int = 250,
    window_sec: float = 30.0,
    overlap: float = 0.5,
    raw_dir: Path = Path("data/raw"),
) -> dict[str, Any]:
    dataset = dataset.strip()
    if dataset not in DATASET_TASKS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if task not in DATASET_TASKS[dataset]:
        raise ValueError(f"Unsupported task for {dataset}: {task}")
    if signal_set not in {"ecg", "ecg_eda"}:
        raise ValueError("signal_set must be 'ecg' or 'ecg_eda'")

    channels = ["ECG"] if signal_set == "ecg" else ["ECG", "EDA"]
    config = DATASET_TASKS[dataset][task]
    src_dir = raw_dir / RAW_SUBDIRS[dataset]

    if dataset == "WESAD":
        from preprocess_WESAD import preprocess_wesad

        raw_artifact = preprocess_wesad(
            str(src_dir),
            "",
            fs,
            window_sec,
            overlap,
            config["keep_labels"],
            config["label_map"],
            signals=channels,
        )
    elif dataset == "CLAS":
        from preprocess_CLAS import preprocess_clas

        raw_artifact = preprocess_clas(
            str(src_dir),
            "",
            fs,
            window_sec,
            overlap,
            config["keep_labels"],
            config["label_map"],
            signals=channels,
        )
    elif dataset == "StressID":
        from preprocess_StressID import TASK_NAMES, preprocess_stressid

        raw_artifact = preprocess_stressid(
            str(src_dir),
            "",
            fs,
            window_sec,
            overlap,
            TASK_NAMES,
            config["label_map"],
            config["label_type"],
            config["task_label_filter"],
            signals=channels,
        )
    elif dataset == "SWELL":
        from preprocess_SWELL import preprocess_swell

        raw_artifact = preprocess_swell(
            str(src_dir),
            "",
            fs,
            window_sec,
            overlap,
            config["keep_conditions"],
            config["label_map"],
            signals=channels,
        )
    elif dataset == "CLARE":
        from preprocess_CLARE import preprocess_clare

        raw_artifact = preprocess_clare(
            str(src_dir),
            "",
            fs,
            window_sec,
            overlap,
            config["keep_labels"],
            config["label_map"],
            config["cl_threshold"],
            signals=channels,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return _normalize_artifact(
        raw_artifact,
        dataset=dataset,
        task=task,
        channels=channels,
        fs=fs,
        window_sec=window_sec,
        overlap=overlap,
        label_map=config["label_map"],
    )


def output_path(processed_dir: Path, dataset: str, task: str, signal_set: str, window_sec: float, overlap: float) -> Path:
    win = f"{window_sec:g}s"
    ovl = int(overlap * 100)
    return processed_dir / f"{dataset}_{task}_{signal_set}_{win}_{ovl}ovl.pkl"


def save_artifact(artifact: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(artifact, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["WESAD", "CLAS", "StressID", "SWELL", "CLARE"])
    parser.add_argument("--signal-sets", nargs="+", default=["ecg", "ecg_eda"], choices=["ecg", "ecg_eda"])
    parser.add_argument("--tasks", nargs="+", default=["binary"])
    parser.add_argument("--fs", type=int, default=250)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--skip-existing", action="store_true", help="Do not recompute artifacts that already exist")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed dataset/task/signal combination")
    args = parser.parse_args()

    failures: list[tuple[str, str, str, str]] = []
    for dataset in args.datasets:
        for signal_set in args.signal_sets:
            for task in args.tasks:
                if task not in DATASET_TASKS.get(dataset, {}):
                    continue
                path = output_path(args.processed_dir, dataset, task, signal_set, args.window_sec, args.overlap)
                if args.skip_existing and path.exists():
                    print(f"Skipping existing artifact: {path}")
                    continue
                try:
                    artifact = preprocess_one(
                        dataset=dataset,
                        signal_set=signal_set,
                        task=task,
                        fs=args.fs,
                        window_sec=args.window_sec,
                        overlap=args.overlap,
                        raw_dir=args.raw_dir,
                    )
                    save_artifact(artifact, path)
                    print(f"Saved {artifact['data'].shape[0]} windows to {path}")
                except Exception as exc:
                    message = f"{dataset}/{task}/{signal_set} failed: {exc}"
                    print(message, file=sys.stderr)
                    failures.append((dataset, task, signal_set, str(exc)))
                    if args.fail_fast:
                        raise

    if failures:
        print("Failed combinations:", file=sys.stderr)
        for dataset, task, signal_set, error in failures:
            print(f"  {dataset}/{task}/{signal_set}: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
