"""Revision P0.1 + P0.4 - matched-feature classical control and paired stats.

The quantum feature map encodes only 6 of the 27 HRV+EDA features
(ECG: mean HR, ApEn, DFA-alpha1; EDA: SCR count, SCR amp mean, variability).
Table I's classical baselines use all 27, so 'competitive' conflates the
quantum substrate with a 6-vs-27 information bottleneck. This script:

  1. Trains the identical classical pipelines (RBF-SVM, RandomForest, logreg)
     on EXACTLY those 6 features, same 5-fold subject-grouped protocol,
     5 seeds -> a 'Classical, 6-feature matched' row block for Table I.
  2. On the canonical seed-42 split, computes per-fold accuracy for
     quantum-6 (fusion readout + RBF-SVM), classical-6, and classical-27,
     then reports paired per-fold deltas + Wilcoxon p (P0.4).

Outputs:
  results/v1/WESAD_matched6_kfold5.csv      (5-seed summary, Table I rows)
  results/v1/matched_paired_analysis.csv    (per-fold + paired deltas)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from quantum_models import build_fusion_circuit, z_zz_expectations  # noqa: E402
from run_classical_baselines import (  # noqa: E402
    _model, feature_columns, make_group_labels, run_grouped_cv_baselines,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

ENCODED_6 = [
    "ecg_mean_hr_bpm", "ecg_apen", "ecg_dfa_alpha1",
    "eda_scr_count", "eda_scr_amplitude_mean", "eda_variability",
]
META = ["dataset", "task", "signal_set", "subject_id", "participant", "label", "window_idx"]
SEEDS = [0, 1, 2, 3, 4]


def per_fold_acc(X, y, groups, model_name, split_seed=42, head_seed=0):
    """Per-fold accuracy on a fixed StratifiedGroupKFold split."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
    accs = []
    for tr, te in splitter.split(X, y, groups):
        clf = _model(model_name, head_seed)
        clf.fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(X[te])))
    return np.array(accs)


def quantum_per_fold_acc(df, split_seed=42, head_seed=0):
    """Locked fusion config: 3 q/ch, reps=1, full entanglement, mixer 0.4, RBF-SVM."""
    X_raw = df[ENCODED_6].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X_raw)
    readout = np.array([
        z_zz_expectations(build_fusion_circuit(
            x, channels=["ECG", "EDA"], qubits_per_channel=3,
            reps=1, entanglement="full", mixer_angle=0.4))
        for x in X_scaled
    ])
    y = df["label"].to_numpy(dtype=int)
    groups = make_group_labels(df).to_numpy()
    return per_fold_acc(readout, y, groups, "rbf_svm", split_seed, head_seed), readout


def main() -> None:
    feat = ROOT / "data" / "features" / "WESAD_binary_ecg_eda_30s_50ovl.parquet"
    df = pd.read_parquet(feat)
    missing = [c for c in ENCODED_6 if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing encoded features: {missing}")
    print(f"WESAD: {len(df)} windows, {df['subject_id'].nunique()} subjects")
    print(f"Encoded-6: {ENCODED_6}")

    out_dir = ROOT / "results" / "v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- P0.1: 5-seed summary on the matched 6 features (Table I rows) ----
    meta_present = [c for c in META if c in df.columns]
    df6 = df[meta_present + ENCODED_6].copy()
    models = ["rbf_svm", "random_forest", "logreg"]
    seed_frames = []
    for seed in SEEDS:
        res = run_grouped_cv_baselines(df6, models=models, n_splits=5, random_state=seed)
        res.insert(0, "random_state", seed)
        seed_frames.append(res)
    summary = pd.concat(seed_frames, ignore_index=True)
    summary.to_csv(out_dir / "WESAD_matched6_kfold5.csv", index=False)

    print("\n=== Matched 6-feature classical (mean over 5 seeds) ===")
    for m in models:
        sub = summary[summary["model"] == m]
        print(f"  {m:>14s}: acc {sub['accuracy_mean'].mean():.3f}"
              f" +- {sub['accuracy_std'].mean():.3f}"
              f" | F1 {sub['macro_f1_mean'].mean():.3f}"
              f" | MCC {sub['mcc_mean'].mean():.3f}"
              f" | AUC {sub['roc_auc_mean'].mean():.3f}")

    # ---- P0.4: paired per-fold deltas on the canonical seed-42 split ----
    y = df["label"].to_numpy(dtype=int)
    groups = make_group_labels(df).to_numpy()
    X27 = df[feature_columns(df)].to_numpy(dtype=float)
    X6 = df[ENCODED_6].to_numpy(dtype=float)

    q_acc, _ = quantum_per_fold_acc(df)
    c6_svm = per_fold_acc(X6, y, groups, "rbf_svm")
    c6_rf = per_fold_acc(X6, y, groups, "random_forest")
    c27_svm = per_fold_acc(X27, y, groups, "rbf_svm")
    c27_rf = per_fold_acc(X27, y, groups, "random_forest")

    rows = []
    for name, arr in [("quantum_6_svm", q_acc), ("classical_6_svm", c6_svm),
                      ("classical_6_rf", c6_rf), ("classical_27_svm", c27_svm),
                      ("classical_27_rf", c27_rf)]:
        rows.append({"method": name, "mean_acc": arr.mean(), "std_acc": arr.std(ddof=0),
                     **{f"fold{i}": v for i, v in enumerate(arr)}})
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(out_dir / "matched_paired_analysis.csv", index=False)

    print("\n=== Per-fold accuracy (canonical seed-42 split) ===")
    print(per_fold[["method", "mean_acc", "std_acc"]].to_string(index=False))

    def paired(a, b, la, lb):
        d = a - b
        try:
            p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = float("nan")
        print(f"  {la} vs {lb}: mean delta {d.mean():+.3f} "
              f"(per-fold {np.round(d,3)}), Wilcoxon p={p:.3f}")

    print("\n=== Paired per-fold comparisons (n=5 folds; low power) ===")
    paired(q_acc, c6_svm, "quantum-6", "classical-6 (SVM)")
    paired(q_acc, c6_rf, "quantum-6", "classical-6 (RF)")
    paired(q_acc, c27_rf, "quantum-6", "classical-27 (RF)")
    paired(c27_rf, c6_rf, "classical-27 (RF)", "classical-6 (RF)")


if __name__ == "__main__":
    main()
