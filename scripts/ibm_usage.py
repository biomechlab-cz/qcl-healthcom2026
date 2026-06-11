"""Append and summarize IBM Quantum usage metadata for reproducibility."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDS = [
    "date",
    "backend",
    "job_id",
    "dataset",
    "signal_set",
    "model",
    "shots",
    "resilience",
    "circuits",
    "qpu_seconds",
    "notes",
]


def append_usage(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    normalized = {field: row.get(field, "") for field in FIELDS}
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


def summarize_usage(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"jobs": 0, "total_qpu_seconds": 0.0, "total_qpu_minutes": 0.0}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    total_seconds = 0.0
    for row in rows:
        try:
            total_seconds += float(row.get("qpu_seconds") or 0.0)
        except ValueError:
            pass
    return {
        "jobs": len(rows),
        "total_qpu_seconds": total_seconds,
        "total_qpu_minutes": total_seconds / 60.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("results/ibm_usage.csv"))
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--date")
    parser.add_argument("--backend")
    parser.add_argument("--job-id")
    parser.add_argument("--dataset")
    parser.add_argument("--signal-set")
    parser.add_argument("--model")
    parser.add_argument("--shots", type=int)
    parser.add_argument("--resilience")
    parser.add_argument("--circuits", type=int)
    parser.add_argument("--qpu-seconds", type=float)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    if args.summary:
        print(summarize_usage(args.path))
        return

    append_usage(
        args.path,
        {
            "date": args.date,
            "backend": args.backend,
            "job_id": args.job_id,
            "dataset": args.dataset,
            "signal_set": args.signal_set,
            "model": args.model,
            "shots": args.shots,
            "resilience": args.resilience,
            "circuits": args.circuits,
            "qpu_seconds": args.qpu_seconds,
            "notes": args.notes,
        },
    )
    print(summarize_usage(args.path))


if __name__ == "__main__":
    main()
