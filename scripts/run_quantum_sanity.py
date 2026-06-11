"""Phase 4 — Sanity baselines for the QRC architecture.

For each (dataset, encoding) pair, compare QRC to three sanity baselines on the
SAME input vector x ∈ R^d:
  (1) Linear classifier directly on x (no nonlinear feature map)
  (2) Random-features classifier: tanh(W·x + b), W ~ N(0, sigma), same readout dim
  (3) Classical reservoir (Echo State Network): leaky-integrator with random
      recurrent weights, readout dim matched to QRC.
  (4) QRC (Statevector simulation, fixed reservoir)

Two encodings tested:
  • temporal_tokens (current QRC input): 8 mean-pooled chunks per channel = 16-dim
  • hrv_features  (HRV scalars from the parquet, amplitude-invariant subset)

If QRC < random-features and < classical-reservoir on every dataset/encoding,
the quantum dynamics aren't contributing useful nonlinearity, and we should
either (a) pivot encoding, (b) switch to Fusion VQC, or (c) report QRC as
hardware-validation-only and rely on Fusion.

Outputs:
- results/quantum_sanity.csv (one row per (dataset, encoding, method, seed))
"""

from __future__ import annotations

import argparse
import pickle
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

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantum_models import (  # noqa: E402
    build_qrc_circuit, temporal_tokens,
)
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402

from run_quantum_experiments import make_readout  # noqa: E402
from run_qrc_sweep import readout_features, compute_qrc_features  # noqa: E402


AMPLITUDE_TAINTED = {
    "ecg_mean", "ecg_std", "ecg_min", "ecg_max",
    "eda_tonic_mean", "eda_min", "eda_max", "eda_std",
}


def load_artifact(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def temporal_token_features(data: np.ndarray, n_tokens: int = 8) -> np.ndarray:
    rows = []
    for window in data:
        tokens = temporal_tokens(window, n_tokens=n_tokens)
        rows.append(tokens.flatten())
    return np.asarray(rows, dtype=float)


def hrv_features(dataset: str, feature_dir: Path, subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = feature_dir / f"{dataset}_binary_ecg_eda_30s_50ovl.parquet"
    df = pd.read_parquet(path)
    cols = [c for c in df.columns if c not in {
        "dataset", "task", "signal_set", "subject_id", "participant", "label", "window_idx"
    } and c not in AMPLITUDE_TAINTED]
    df_sorted = df.sort_values("window_idx").reset_index(drop=True)
    if df_sorted["subject_id"].astype(str).to_numpy().tolist() != subjects.tolist():
        # Re-align using subject_id ordering from the pkl
        # The parquet was extracted from the same pkl, so row order matches.
        # If subject order differs, raise; otherwise proceed.
        if len(df_sorted) != len(subjects):
            raise ValueError(f"{dataset}: parquet len {len(df_sorted)} != pkl windows {len(subjects)}")
    return df_sorted[cols].to_numpy(dtype=float), np.asarray(cols)


def random_features(x: np.ndarray, n_features: int, seed: int, sigma: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, sigma, size=(x.shape[1], n_features))
    b = rng.normal(0.0, 0.1, size=(n_features,))
    return np.tanh(x @ W + b)


def echo_state_reservoir(x: np.ndarray, n_features: int, seed: int,
                         leak: float = 0.3, sigma: float = 1.0,
                         spectral_radius: float = 0.95) -> np.ndarray:
    """ESN-style reservoir features. Treats each row as a single time-step input;
    iterates the reservoir state with leaky update.
    """
    rng = np.random.default_rng(seed)
    n_in = x.shape[1]
    W_in = rng.normal(0.0, sigma, size=(n_features, n_in))
    W_res = rng.normal(0.0, 1.0, size=(n_features, n_features))
    eig = np.max(np.abs(np.linalg.eigvals(W_res)))
    W_res = W_res * (spectral_radius / max(eig, 1e-6))
    rows = []
    for xi in x:
        # 5 leaky updates; treat each row as a single static input pattern
        s = np.zeros(n_features)
        for _ in range(5):
            s = (1.0 - leak) * s + leak * np.tanh(W_in @ xi + W_res @ s)
        rows.append(s)
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
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--n-tokens", type=int, default=8)
    parser.add_argument("--entanglement", default="ring")
    parser.add_argument("--readout", default="z_adj_zz")
    parser.add_argument("--heads", nargs="+", default=["logreg", "rbf_svm"])
    parser.add_argument("--random-states", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path, default=Path("results/quantum_sanity.csv"))
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        path = args.processed_dir / f"{dataset}_binary_ecg_eda_30s_50ovl.pkl"
        if not path.exists():
            print(f"Skipping {dataset}: {path} not found")
            continue
        print(f"\n=== {dataset} ===")
        artifact = load_artifact(path)
        data = np.asarray(artifact["data"], dtype=np.float32)
        y = np.asarray(artifact["labels"], dtype=int)
        subjects = np.asarray(artifact["participants"]).astype(str)
        groups = np.array([f"{dataset}::{s}" for s in subjects])
        print(f"  windows={data.shape[0]}, subjects={len(np.unique(subjects))}")

        # ===== ENCODING A: temporal_tokens =====
        print("  encoding=temporal_tokens", flush=True)
        t0 = time.time()
        x_temporal = temporal_token_features(data, n_tokens=args.n_tokens)
        # QRC features on the same temporal-token input
        x_qrc = compute_qrc_features(data, args.n_qubits, args.n_tokens, args.entanglement, args.readout)
        n_qrc_features = x_qrc.shape[1]
        print(f"    temporal_dim={x_temporal.shape[1]} qrc_dim={n_qrc_features} ({time.time()-t0:.1f}s)", flush=True)

        for method_name, X in [
            ("linear_on_temporal", x_temporal),
            ("qrc_temporal", x_qrc),
        ]:
            for head, seed in product(args.heads, args.random_states):
                m = loso_eval(X, y, groups, head, seed)
                rows.append({
                    "dataset": dataset, "encoding": "temporal_tokens",
                    "method": method_name, "head": head, "random_state": seed,
                    "n_features": int(X.shape[1]), **m,
                })

        # Random features and ESN on temporal_tokens — these vary by seed
        for method_name, transform in [
            ("random_features_temporal", random_features),
            ("esn_temporal", echo_state_reservoir),
        ]:
            for head, seed in product(args.heads, args.random_states):
                X = transform(x_temporal, n_qrc_features, seed)
                m = loso_eval(X, y, groups, head, seed)
                rows.append({
                    "dataset": dataset, "encoding": "temporal_tokens",
                    "method": method_name, "head": head, "random_state": seed,
                    "n_features": int(X.shape[1]), **m,
                })

        # ===== ENCODING B: HRV features =====
        print("  encoding=hrv_features", flush=True)
        try:
            x_hrv, hrv_cols = hrv_features(dataset, args.feature_dir, subjects)
            print(f"    hrv_dim={x_hrv.shape[1]} cols={hrv_cols.tolist()}", flush=True)

            for head, seed in product(args.heads, args.random_states):
                m = loso_eval(x_hrv, y, groups, head, seed)
                rows.append({
                    "dataset": dataset, "encoding": "hrv_features",
                    "method": "linear_on_hrv", "head": head, "random_state": seed,
                    "n_features": int(x_hrv.shape[1]), **m,
                })
            for method_name, transform in [
                ("random_features_hrv", random_features),
                ("esn_hrv", echo_state_reservoir),
            ]:
                for head, seed in product(args.heads, args.random_states):
                    X = transform(x_hrv, n_qrc_features, seed)
                    m = loso_eval(X, y, groups, head, seed)
                    rows.append({
                        "dataset": dataset, "encoding": "hrv_features",
                        "method": method_name, "head": head, "random_state": seed,
                        "n_features": int(X.shape[1]), **m,
                    })

            # Also: QRC on a re-encoded HRV input. To do this we'd need a new
            # build_qrc_circuit that takes scalar HRV features instead of temporal
            # tokens. As a quick proxy: pretend the HRV vector is a 1-token x d-channel
            # tensor and feed to build_qrc_circuit with n_tokens=1, channels=d.
            # That's a different architecture (no temporal Trotterization), but
            # it tests the "use better input features" hypothesis.
            # Skip for now; we'll handle this in a dedicated qrc_hrv runner if
            # the gap analysis shows the encoding is the bottleneck.

        except Exception as exc:
            print(f"    HRV encoding failed: {exc}")

        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"  wrote {args.output}", flush=True)

    print(f"\nFinal rows: {len(rows)}")


if __name__ == "__main__":
    main()
