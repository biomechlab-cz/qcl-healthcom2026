"""Phase 4.2 — Fusion VQC sweep on WESAD + CLAS HRV features.

For each (dataset × qubits_per_channel × fusion_reps × entanglement × mixer_angle)
the Fusion circuit's readout feature is computed once on globally-scaled
amp-invariant HRV features, then LOSO is run with logreg/ridge/rbf_svm heads
across 5 seeds.

Why global scaling here (vs fold-scaling): the HRV features are per-window
summaries — global scaling rescales the coordinate axes without leaking any
label information from test to train (unlike the per-subject raw-signal
z-scoring that drove WESAD's earlier 92.5% number). The fusion-vs-classical
comparison in this sweep is on equal footing because the linear-HRV baseline
also uses global scaling implicitly inside the sklearn Pipeline.

Outputs:
- results/quantum_fusion_sweep.csv (one row per (config, seed, head))
- results/quantum_fusion_sweep_costs.csv (one row per (config))
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantum_models import (  # noqa: E402
    build_fusion_circuit, circuit_summary, z_zz_expectations,
)
from run_quantum_experiments import make_readout  # noqa: E402

AMPLITUDE_TAINTED = {
    "ecg_mean", "ecg_std", "ecg_min", "ecg_max",
    "eda_tonic_mean", "eda_min", "eda_max", "eda_std",
}

METADATA_COLUMNS = {
    "dataset", "task", "signal_set", "subject_id", "participant", "label", "window_idx",
}

# Order matters: the first qubits_per_channel of each modality are the
# encoding rotations, so the most discriminative features should come first.
# Priority reflects effect-size analysis on WESAD/CLAS after Phase 2 B2
# (2026-05-15): DFA-α1 and ApEn separate classes by ~0.6-0.7σ; mean HR by 1.2σ.
ECG_ORDER = [
    "ecg_mean_hr_bpm",      # dominant: ~1.2σ separation
    "ecg_apen",             # B2: ~0.7σ
    "ecg_dfa_alpha1",       # B2: ~0.6σ
    "ecg_sd1_sd2_ratio",    # B2: ~0.5σ
    "ecg_rmssd_ms",         # ~0.3σ
    "ecg_sd1",              # B2: ~0.3σ
    "ecg_sdnn_ms",          # ~0.3σ
    "ecg_sd2",              # B2
    "ecg_pnn50",
    "ecg_sampen",
    "ecg_lf_hf_ratio",      # B2 — noisy at 30s
    "ecg_peak_count",
    "ecg_slope",
]
EDA_ORDER = [
    "eda_scr_count",
    "eda_scr_amplitude_mean",
    "eda_variability",
    "eda_slope",
]


def select_modality_columns(df: pd.DataFrame, qubits_per_channel: int) -> list[str]:
    ecg = [c for c in ECG_ORDER if c in df.columns and c not in AMPLITUDE_TAINTED][:qubits_per_channel]
    eda = [c for c in EDA_ORDER if c in df.columns and c not in AMPLITUDE_TAINTED][:qubits_per_channel]
    cols = ecg + eda
    if len(cols) < 2 * qubits_per_channel:
        raise ValueError(
            f"Need {2 * qubits_per_channel} features (≥{qubits_per_channel} ECG, ≥{qubits_per_channel} EDA), "
            f"got ECG={ecg} EDA={eda}"
        )
    return cols


def compute_fusion_features(
    X_scaled: np.ndarray,
    qubits_per_channel: int,
    fusion_reps: int,
    entanglement: str,
    mixer_angle: float,
) -> np.ndarray:
    channels = ["ECG", "EDA"]
    rows = []
    for x in X_scaled:
        circuit = build_fusion_circuit(
            x, channels=channels,
            qubits_per_channel=qubits_per_channel,
            reps=fusion_reps,
            entanglement=entanglement,
            mixer_angle=mixer_angle,
        )
        rows.append(z_zz_expectations(circuit))
    return np.asarray(rows, dtype=float)


def score_fold(y_true, y_pred, proba):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "roc_auc": float("nan"),
    }
    if proba is not None and len(np.unique(y_true)) == 2 and proba.shape[1] == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        except ValueError:
            pass
    return out


def loso_eval(X, y, groups, head, seed):
    # 5-fold StratifiedGroupKFold per project-wide CV switch 2026-05-15.
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, te_idx in splitter.split(X, y, groups):
        if len(np.unique(y[tr_idx])) < 2:
            continue
        clf = make_readout(head, seed)
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[te_idx])
        proba = clf.predict_proba(X[te_idx]) if hasattr(clf, "predict_proba") else None
        scores.append(score_fold(y[te_idx], pred, proba))
    df = pd.DataFrame(scores)
    return {
        "n_folds": int(len(df)),
        "accuracy_mean": float(df["accuracy"].mean()),
        "accuracy_std": float(df["accuracy"].std(ddof=0)),
        "macro_f1_mean": float(df["macro_f1"].mean()),
        "macro_f1_std": float(df["macro_f1"].std(ddof=0)),
        "mcc_mean": float(df["mcc"].mean()),
        "mcc_std": float(df["mcc"].std(ddof=0)),
        "roc_auc_mean": float(df["roc_auc"].mean()),
        "roc_auc_std": float(df["roc_auc"].std(ddof=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["WESAD", "CLAS"])
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--qubits-per-channel", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--fusion-reps", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--entanglement", nargs="+", default=["minimal", "full"])
    parser.add_argument("--mixer-angle", nargs="+", type=float, default=[0.0, 0.4])
    parser.add_argument("--heads", nargs="+", default=["ridge", "logreg", "rbf_svm"])
    parser.add_argument("--random-states", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path, default=Path("results/quantum_fusion_sweep.csv"))
    parser.add_argument("--cost-output", type=Path, default=Path("results/quantum_fusion_sweep_costs.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cost_rows: list[dict] = []

    for dataset in args.datasets:
        path = args.features_dir / f"{dataset}_binary_ecg_eda_30s_50ovl.parquet"
        if not path.exists():
            print(f"Skipping {dataset}: {path} not found")
            continue
        df = pd.read_parquet(path)
        y = df["label"].to_numpy(dtype=int)
        groups = np.array([f"{dataset}::{s}" for s in df["subject_id"].astype(str).to_numpy()])
        print(f"\n=== {dataset} === windows={len(df)}, subjects={df['subject_id'].nunique()}")

        for q_per_ch, reps, ent, mixer in product(
            args.qubits_per_channel, args.fusion_reps, args.entanglement, args.mixer_angle
        ):
            cols = select_modality_columns(df, q_per_ch)
            X_raw = df[cols].to_numpy(dtype=float)
            X_scaled = StandardScaler().fit_transform(X_raw)
            t0 = time.time()
            print(f"  q/ch={q_per_ch} reps={reps} ent={ent} mixer={mixer} "
                  f"cols={cols} ... ", end="", flush=True)

            X_fusion = compute_fusion_features(X_scaled, q_per_ch, reps, ent, mixer)
            feat_time = time.time() - t0

            demo = build_fusion_circuit(
                np.zeros(2 * q_per_ch), channels=["ECG", "EDA"],
                qubits_per_channel=q_per_ch, reps=reps,
                entanglement=ent, mixer_angle=mixer,
            )
            cs = circuit_summary(demo)
            cost_rows.append({
                "dataset": dataset, "qubits_per_channel": q_per_ch,
                "fusion_reps": reps, "entanglement": ent, "mixer_angle": mixer,
                "n_input_features": len(cols), "n_quantum_features": int(X_fusion.shape[1]),
                "feat_time_sec": round(feat_time, 2), **cs,
            })
            print(f"feat_time={feat_time:.1f}s n_q_features={X_fusion.shape[1]}", flush=True)

            for head, seed in product(args.heads, args.random_states):
                try:
                    m = loso_eval(X_fusion, y, groups, head, seed)
                except Exception as exc:
                    print(f"    head={head} seed={seed} FAILED: {exc}")
                    continue
                rows.append({
                    "dataset": dataset, "qubits_per_channel": q_per_ch,
                    "fusion_reps": reps, "entanglement": ent, "mixer_angle": mixer,
                    "head": head, "random_state": seed,
                    "n_features": int(X_fusion.shape[1]),
                    **m,
                })
            pd.DataFrame(rows).to_csv(args.output, index=False)
            pd.DataFrame(cost_rows).to_csv(args.cost_output, index=False)

    print(f"\nWrote {args.output} ({len(rows)} rows)")
    print(f"Wrote {args.cost_output} ({len(cost_rows)} configs)")


if __name__ == "__main__":
    main()
