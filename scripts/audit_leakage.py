"""Phase 2 leakage diagnostic on Track B feature parquets.

For each dataset:
- subject count, class balance, per-subject class balance
- per-class mean ± std of physiological features (HR, HRV, EDA tonic)
- subject-identifiability: train classifier on features to predict subject_id
  (5-fold stratified-by-subject CV). High accuracy here means windows carry
  strong subject fingerprint that any LOSO classifier would memorize and reverse.
- per-subject within-class window count (looking for subject:label coupling
  where one subject has ~all-stress or ~all-rest windows, which makes their
  hold-out fold a 100%-of-one-class oracle).

Writes a markdown summary to reports/phase2_leakage.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


METADATA_COLUMNS = {"dataset", "task", "signal_set", "subject_id", "participant", "label", "window_idx"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in METADATA_COLUMNS]


def per_subject_class_balance(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["subject_id", "label"]).size().unstack(fill_value=0)
    grouped.columns = [f"label={c}" for c in grouped.columns]
    grouped["total"] = grouped.sum(axis=1)
    label_cols = [c for c in grouped.columns if c.startswith("label=")]
    for col in label_cols:
        grouped[f"{col}_pct"] = (grouped[col] / grouped["total"] * 100).round(1)
    return grouped


def feature_means_by_class(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df.groupby("label")[features].agg(["mean", "std"])


def subject_identifiability(df: pd.DataFrame, features: list[str], seed: int = 0) -> float:
    """Train RF to predict subject_id from features. Returns mean accuracy.

    High value = features fingerprint subjects strongly = LOSO classifier
    will struggle to generalize, NOT a source of label leakage but a
    diagnostic for how subject-coupled the features are.
    """
    if df["subject_id"].nunique() < 2:
        return float("nan")
    X = df[features].fillna(0.0).to_numpy(dtype=np.float64)
    y = LabelEncoder().fit_transform(df["subject_id"].astype(str))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in cv.split(X, y):
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        accs.append(clf.score(X[test_idx], y[test_idx]))
    return float(np.mean(accs))


def constant_subjects(per_subject: pd.DataFrame, threshold: float = 95.0) -> pd.DataFrame:
    """Find subjects where >=threshold% of windows are one class — they're
    held-out-as-100%-of-one-class in LOSO, which inflates per-fold accuracy.
    """
    label_cols = [c for c in per_subject.columns if c.startswith("label=") and c.endswith("_pct")]
    flags = per_subject[label_cols].max(axis=1) >= threshold
    return per_subject[flags]


def audit_one(path: Path, seed: int = 0) -> dict:
    df = pd.read_parquet(path)
    feats = feature_columns(df)
    n_windows = len(df)
    n_subjects = df["subject_id"].nunique()
    overall_balance = df["label"].value_counts().sort_index()

    per_subj = per_subject_class_balance(df)
    constants = constant_subjects(per_subj, threshold=95.0)
    means = feature_means_by_class(df, feats)
    subj_id_acc = subject_identifiability(df, feats, seed=seed)

    return {
        "path": path,
        "n_windows": n_windows,
        "n_subjects": n_subjects,
        "overall_balance": overall_balance,
        "per_subject": per_subj,
        "constant_subjects": constants,
        "feature_means_by_class": means,
        "subject_identifiability_acc": subj_id_acc,
        "features": feats,
    }


def write_markdown(audits: list[dict], output: Path) -> None:
    lines: list[str] = []
    lines.append("# Phase 2 leakage / red-flag audit")
    lines.append("")
    lines.append("**Generated:** 2026-05-14 (Phase 2.0c)")
    lines.append("")
    lines.append("Context: per-dataset LOSO classical baselines flagged two suspicious numbers:")
    lines.append("- **CLAS** best LOSO accuracy ~96.2% (LightGBM)")
    lines.append("- **WESAD** best LOSO accuracy ~92.5% (RF)")
    lines.append("")
    lines.append("Both trip the plan's red-flag heuristic. This audit examines whether the high accuracy")
    lines.append("is structurally explainable (label design, subject-class coupling, feature fingerprinting)")
    lines.append("or whether it implies a methodological leak that must be fixed before any pooled run.")
    lines.append("")

    for audit in audits:
        path = audit["path"]
        lines.append(f"## {path.name}")
        lines.append("")
        lines.append(f"- windows: **{audit['n_windows']}**, subjects: **{audit['n_subjects']}**")
        bal = audit["overall_balance"].to_dict()
        bal_pct = {k: round(v / audit["n_windows"] * 100, 1) for k, v in bal.items()}
        lines.append(f"- overall class balance (label -> windows): {bal} ({bal_pct} %)")
        lines.append(f"- subject identifiability (RF, 5-fold; baseline = 1/N_subjects = "
                     f"{1.0 / audit['n_subjects']:.3f}): **{audit['subject_identifiability_acc']:.3f}**")
        lines.append("")
        lines.append("### Per-subject class balance")
        lines.append("")
        lines.append("```")
        lines.append(audit["per_subject"].to_string())
        lines.append("```")
        lines.append("")
        if len(audit["constant_subjects"]) > 0:
            lines.append(f"### Subjects with ≥95 % windows in a single class "
                         f"({len(audit['constant_subjects'])} of {audit['n_subjects']})")
            lines.append("")
            lines.append("These are essentially single-class folds in LOSO, which means the held-out subject's")
            lines.append("predictions are dominated by the class-prior — high per-fold accuracy not from true")
            lines.append("class discrimination.")
            lines.append("")
            lines.append("```")
            lines.append(audit["constant_subjects"].to_string())
            lines.append("```")
            lines.append("")
        lines.append("### Per-class feature means (mean ± std across windows)")
        lines.append("")
        lines.append("```")
        means = audit["feature_means_by_class"]
        # show a compact subset: HR, HRV, EDA
        key_cols = [c for c in audit["features"] if c in {
            "ecg_mean_hr_bpm", "ecg_sdnn_ms", "ecg_rmssd_ms", "ecg_pnn50",
            "ecg_sampen", "eda_tonic_mean", "eda_scr_count", "eda_slope",
        }]
        if key_cols:
            mean_view = means.loc[:, pd.IndexSlice[key_cols, :]]
            lines.append(mean_view.to_string(float_format=lambda x: f"{x:.3f}"))
        else:
            lines.append("(no key columns found)")
        lines.append("```")
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--pattern", default="*_binary_ecg_eda_30s_50ovl.parquet")
    parser.add_argument("--output", type=Path, default=Path("reports/phase2_leakage.md"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    inputs = sorted(args.features_dir.glob(args.pattern))
    if not inputs:
        raise FileNotFoundError(f"No parquet feature files at {args.features_dir}/{args.pattern}")

    audits = []
    for path in inputs:
        print(f"Auditing {path.name} ...", flush=True)
        audits.append(audit_one(path, seed=args.seed))

    write_markdown(audits, args.output)
    print(f"\nWrote {args.output}")
    for audit in audits:
        print(f"  {audit['path'].name}: "
              f"subj_id_acc={audit['subject_identifiability_acc']:.3f}, "
              f"constants={len(audit['constant_subjects'])}/{audit['n_subjects']}")


if __name__ == "__main__":
    main()
