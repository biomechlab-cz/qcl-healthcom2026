"""Phase 5 - Hardware pilot for the locked Fusion VQC configurations.

Per memory/feedback_hardware_cv_strategy.md, uses 5-fold StratifiedGroupKFold
with stratified per-fold test-window subsampling. Same CV protocol as simulator
headline; this script validates the trained-readout pipeline on real hardware.

**Readout primitive: EstimatorV2 (not SamplerV2).** EstimatorV2 supports
resilience_level=2 (ZNE) natively; SamplerV2 only supports L0/L1. Observables
are Z_0..Z_{n-1} + adjacent Z_i Z_{i+1}, matching quantum_models.z_zz_expectations.

For each fold:
1. Pre-flight budget check via ibm_usage.summarize_usage; abort if cumulative
   + planned would exceed 150 QPU min cap and ALLOW_OVER_150 is unset.
2. Compute training Fusion features on Statevector (exact, no QPU).
3. Fit classical readout on training features.
4. Build per-window Fusion circuits (no measure_all -- Estimator measures its
   own observables). Generate preset pass manager for the backend at the chosen
   optimization level; transpile circuits and remap observables to ISA layout.
5. Run via EstimatorV2 with resilience_level matching the requested mitigation
   level (0=raw, 1=T-REx+DD, 2=+ZNE). Estimator returns 11 expectation values
   per circuit directly.
6. Compare hardware expectations to Statevector-exact expectations for the same
   windows -> divergence diagnostic.
7. Predict via readout; score; write to results/hardware_runs/.
8. If --hardware, append a row to results/ibm_usage.csv via ibm_usage.append_usage.

Locked configs (from reports/fusion_vqc_config.md):
- WESAD: qubits_per_channel=3, fusion_reps=1, entanglement=full, mixer=0.4, head=rbf_svm
- CLAS:  qubits_per_channel=2, fusion_reps=2, entanglement=minimal, mixer=0.4, head=rbf_svm
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ibm_usage import append_usage, summarize_usage  # noqa: E402
from quantum_models import build_fusion_circuit, z_zz_expectations  # noqa: E402
from run_quantum_experiments import make_readout  # noqa: E402
from run_fusion_sweep import select_modality_columns  # noqa: E402

from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402


# Locked Fusion VQC configs from reports/fusion_vqc_config.md
# Updated 2026-05-15 after R2 (no signal scaling) + B2 (expanded HRV) + 5-fold
# StratifiedGroupKFold. Both winners use RBF-SVM head with mixer_angle 0.4.
LOCKED_CONFIGS = {
    "WESAD": {
        # 6 qubits (3 ECG + 3 EDA), depth 4 transpiled, 7 two-qubit gates.
        # Inputs: ecg_mean_hr_bpm, ecg_apen, ecg_dfa_alpha1, eda_scr_count,
        # eda_scr_amplitude_mean, eda_variability. Simulator 5-fold acc 0.830.
        "qubits_per_channel": 3, "fusion_reps": 1,
        "entanglement": "full", "mixer_angle": 0.4, "head": "rbf_svm",
    },
    "CLAS": {
        # 4 qubits (2 ECG + 2 EDA), depth 6 transpiled, 6 two-qubit gates.
        # Inputs: ecg_mean_hr_bpm, ecg_apen, eda_scr_count,
        # eda_scr_amplitude_mean. Simulator 5-fold acc 0.724.
        "qubits_per_channel": 2, "fusion_reps": 2,
        "entanglement": "minimal", "mixer_angle": 0.4, "head": "rbf_svm",
    },
}

BUDGET_HARD_STOP_MIN = 150.0


def preflight_budget_check(usage_log: Path, planned_qpu_min: float = 0.0) -> None:
    """Abort if cumulative QPU + planned would exceed 150 min and ALLOW_OVER_150 unset."""
    summary = summarize_usage(usage_log)
    used = summary["total_qpu_minutes"]
    if used + planned_qpu_min > BUDGET_HARD_STOP_MIN:
        if not os.environ.get("ALLOW_OVER_150"):
            raise RuntimeError(
                f"Pre-flight budget check FAILED: cumulative {used:.1f} min + planned {planned_qpu_min:.1f} min "
                f"= {used + planned_qpu_min:.1f} > {BUDGET_HARD_STOP_MIN:.0f} min cap. "
                f"Set ALLOW_OVER_150=1 to proceed."
            )
        print(f"WARNING: cumulative QPU at {used:.1f} min, planned {planned_qpu_min:.1f} min — proceeding under ALLOW_OVER_150=1")
    else:
        print(f"Pre-flight OK: cumulative {used:.1f} min + planned {planned_qpu_min:.1f} min = "
              f"{used + planned_qpu_min:.1f} of {BUDGET_HARD_STOP_MIN:.0f} cap")


def make_fold_split(df: pd.DataFrame, n_splits: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    y = df["label"].to_numpy()
    groups = df["subject_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(np.zeros(len(df)), y, groups))


def stratified_subsample(df: pd.DataFrame, indices: np.ndarray, max_per_class: int, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    selected: list[int] = []
    sub = df.iloc[indices]
    for label, idx_for_label in sub.groupby("label").groups.items():
        idx_array = np.array(idx_for_label.tolist())
        if len(idx_array) > max_per_class:
            idx_array = rng.choice(idx_array, size=max_per_class, replace=False)
        selected.extend(idx_array.tolist())
    return np.array(sorted(selected))


def compute_fusion_features_exact(
    X_scaled: np.ndarray, q_per_ch: int, reps: int, ent: str, mixer: float,
) -> np.ndarray:
    """Statevector-exact Z + adjacent-ZZ features (training side)."""
    rows = []
    for x in X_scaled:
        c = build_fusion_circuit(x, channels=["ECG", "EDA"], qubits_per_channel=q_per_ch,
                                 reps=reps, entanglement=ent, mixer_angle=mixer)
        rows.append(z_zz_expectations(c))
    return np.asarray(rows, dtype=float)


def build_observables(n_qubits: int) -> list[SparsePauliOp]:
    """Z_0..Z_{n-1} then Z_i Z_{i+1} for i in 0..n-2.

    Order MUST match quantum_models.z_zz_expectations so the readout trained
    on Statevector features maps element-wise onto Estimator output.
    Qiskit SparsePauliOp labels are read right-to-left (rightmost char = qubit 0).
    """
    obs: list[SparsePauliOp] = []
    for q in range(n_qubits):
        label = ["I"] * n_qubits
        label[n_qubits - q - 1] = "Z"
        obs.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    for q in range(n_qubits - 1):
        label = ["I"] * n_qubits
        label[n_qubits - q - 1] = "Z"
        label[n_qubits - q - 2] = "Z"
        obs.append(SparsePauliOp.from_list([("".join(label), 1.0)]))
    return obs


def run_estimator(
    circuits, observables, backend_name: str, shots: int, hardware: bool,
    optimization_level: int, resilience_level: int,
):
    """Run circuits + observables on Aer or IBM via EstimatorV2.

    Returns (n_circuits, n_observables) array of expectation values per circuit.
    Resilience semantics (matching qiskit-ibm-runtime EstimatorV2):
    - L0: no mitigation
    - L1: T-REx readout twirling + dynamical decoupling
    - L2: L1 + zero-noise extrapolation (default noise factors and extrapolator)
    """
    from qiskit_ibm_runtime import EstimatorV2, EstimatorOptions

    if hardware:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
    else:
        backend = AerSimulator(seed_simulator=42)

    # Backend-aware transpilation: produces ISA circuits with backend.layout
    # against which observables must be remapped via apply_layout.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
    isa_circuits = pm.run(circuits)

    # Build PUBs: (isa_circuit, isa_observables) for each test window.
    pubs = []
    for isa_c in isa_circuits:
        layout = isa_c.layout
        if layout is None:
            isa_obs = observables
        else:
            isa_obs = [obs.apply_layout(layout) for obs in observables]
        pubs.append((isa_c, isa_obs))

    options = EstimatorOptions()
    options.default_shots = shots
    if hardware:
        # On hardware the resilience_level shortcut activates the documented
        # IBM Runtime defaults: L1 = T-REx + DD; L2 = + ZNE with default noise
        # factors and exponential extrapolator.
        options.resilience_level = int(resilience_level)
    else:
        # Aer doesn't honor every mitigation toggle; keep at L0 for the dry run.
        options.resilience_level = 0

    estimator = EstimatorV2(mode=backend, options=options)
    t0 = time.time()
    job = estimator.run(pubs)
    job_id = job.job_id() if hasattr(job, "job_id") else f"aer-{int(t0)}"

    try:
        result = job.result()
        # Each pub_result.data.evs is shape (n_observables,). Stack into matrix.
        evs = np.vstack([np.asarray(pr.data.evs, dtype=float).reshape(-1) for pr in result])
        job_error = ""
        job_status = "DONE"
    except Exception as exc:
        evs = None
        job_error = f"{type(exc).__name__}: {exc}"
        try:
            job_status = str(job.status())
        except Exception:
            job_status = "unknown"

    qpu_seconds = 0.0
    try:
        usage = getattr(job, "usage_estimation", None)
        if callable(usage):
            usage = usage()
        if isinstance(usage, dict):
            qpu_seconds = float(usage.get("quantum_seconds") or usage.get("qpu_seconds") or 0.0)
    except Exception:
        pass

    wall_seconds = time.time() - t0
    return {
        "evs": evs, "job_id": str(job_id),
        "qpu_seconds": qpu_seconds, "wall_seconds": wall_seconds,
        "backend_name": backend.name, "job_status": job_status, "job_error": job_error,
        "transpiled_depth": int(np.mean([c.depth() for c in isa_circuits])),
        "two_qubit_gates": int(np.mean([
            sum(c.count_ops().get(g, 0) for g in ("cx", "cz", "rzz", "ecr")) for c in isa_circuits
        ])),
    }


def score(y_true, y_pred, proba):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["WESAD", "CLAS"])
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--fold-idx", type=int, default=0, help="Which of the 5 folds to run")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv-random-state", type=int, default=42)
    parser.add_argument("--max-test-per-class", type=int, default=25,
                        help="Subsample test windows to ≤ 2*this per fold (≤ 50 total).")
    parser.add_argument("--backend", default="aer", help="aer | ibm_boston | ...")
    parser.add_argument("--hardware", action="store_true", help="Actually hit a real backend")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--resilience-level", type=int, default=0,
                        help="0=raw, 1=T-REx+DD, 2=+ZNE (passed through as DD/twirling toggles for SamplerV2)")
    parser.add_argument("--optimization-level", type=int, default=2)
    parser.add_argument("--planned-qpu-min", type=float, default=10.0,
                        help="Estimate fed to pre-flight budget check.")
    parser.add_argument("--usage-log", type=Path, default=Path("results/ibm_usage.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/hardware_runs"))
    args = parser.parse_args()

    cfg = LOCKED_CONFIGS[args.dataset]
    q_per_ch, reps = cfg["qubits_per_channel"], cfg["fusion_reps"]
    ent, mixer, head = cfg["entanglement"], cfg["mixer_angle"], cfg["head"]
    n_qubits = 2 * q_per_ch  # ECG + EDA registers
    enable_dd = args.resilience_level >= 1
    enable_twirling = args.resilience_level >= 1  # T-REx == measurement twirling
    enable_zne = args.resilience_level >= 2

    # Pre-flight budget check (only meaningful for hardware, but print either way)
    if args.hardware:
        preflight_budget_check(args.usage_log, planned_qpu_min=args.planned_qpu_min)
    else:
        print(f"[dry run] backend={args.backend} (Aer); skipping budget check.")

    # ---- load & feature select ----
    parquet = args.features_dir / f"{args.dataset}_binary_ecg_eda_30s_50ovl.parquet"
    df = pd.read_parquet(parquet).reset_index(drop=True)
    cols = select_modality_columns(df, q_per_ch)
    print(f"Dataset {args.dataset}: windows={len(df)}, modality_cols={cols}")
    print(f"Config: qubits_per_channel={q_per_ch} reps={reps} ent={ent} mixer={mixer} head={head}")
    print(f"Resilience level={args.resilience_level} (DD={enable_dd}, twirling={enable_twirling}, ZNE={enable_zne})")

    # ---- 5-fold split & subsample ----
    folds = make_fold_split(df, args.n_splits, args.cv_random_state)
    train_idx, test_idx_full = folds[args.fold_idx]
    test_idx = stratified_subsample(df, test_idx_full, args.max_test_per_class, args.cv_random_state)
    print(f"Fold {args.fold_idx}: train={len(train_idx)}, test_full={len(test_idx_full)}, test_subsample={len(test_idx)}")

    # ---- global scaling (matches Phase 4.2 simulator pipeline) ----
    X_raw = df[cols].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X_raw)

    # ---- training features (Statevector, simulator) ----
    print("Computing training Fusion features on Statevector ...")
    t0 = time.time()
    X_train_q = compute_fusion_features_exact(X_scaled[train_idx], q_per_ch, reps, ent, mixer)
    print(f"  done in {time.time()-t0:.1f}s, shape={X_train_q.shape}")

    # ---- fit readout ----
    y = df["label"].to_numpy()
    readout = make_readout(head, random_state=args.cv_random_state)
    readout.fit(X_train_q, y[train_idx])

    # ---- test circuits (no measure_all; EstimatorV2 handles its own measurements) ----
    print(f"Building {len(test_idx)} test circuits ...")
    test_circuits = []
    for i in test_idx:
        c = build_fusion_circuit(X_scaled[i], channels=["ECG", "EDA"],
                                 qubits_per_channel=q_per_ch, reps=reps,
                                 entanglement=ent, mixer_angle=mixer)
        test_circuits.append(c)
    observables = build_observables(n_qubits)
    assert len(observables) == n_qubits + (n_qubits - 1), \
        f"Expected {n_qubits + (n_qubits - 1)} observables, built {len(observables)}"

    # ---- run estimator ----
    print(f"Running {len(test_circuits)} circuits with {len(observables)} observables on "
          f"backend={args.backend} (hardware={args.hardware}, resilience_level={args.resilience_level}) ...")
    run = run_estimator(
        test_circuits, observables, args.backend, args.shots, args.hardware,
        args.optimization_level, args.resilience_level,
    )
    if run["evs"] is None:
        raise RuntimeError(f"Estimator failed: {run['job_error']}")

    # ---- predict from hardware-evaluated features ----
    X_test_q_hw = run["evs"]  # shape (n_test, 11)
    y_pred_hw = readout.predict(X_test_q_hw)
    proba_hw = readout.predict_proba(X_test_q_hw) if hasattr(readout, "predict_proba") else None
    metrics_hw = score(y[test_idx], y_pred_hw, proba_hw)

    # ---- exact-simulator comparison ----
    X_test_q_sv = compute_fusion_features_exact(X_scaled[test_idx], q_per_ch, reps, ent, mixer)
    y_pred_sv = readout.predict(X_test_q_sv)
    proba_sv = readout.predict_proba(X_test_q_sv) if hasattr(readout, "predict_proba") else None
    metrics_sv = score(y[test_idx], y_pred_sv, proba_sv)

    # divergence: per-feature mean absolute difference
    feature_mad = float(np.mean(np.abs(X_test_q_hw - X_test_q_sv)))
    feature_mad_max = float(np.max(np.abs(X_test_q_hw - X_test_q_sv)))
    pred_agreement = float(np.mean(y_pred_hw == y_pred_sv))

    print("\n=== Results ===")
    print(f"  Statevector (exact): acc={metrics_sv['accuracy']:.3f}, f1={metrics_sv['macro_f1']:.3f}, mcc={metrics_sv['mcc']:.3f}")
    print(f"  {run['backend_name']:>20}: acc={metrics_hw['accuracy']:.3f}, f1={metrics_hw['macro_f1']:.3f}, mcc={metrics_hw['mcc']:.3f}")
    print(f"  Feature MAD (per-feature mean abs diff): {feature_mad:.4f} (max {feature_mad_max:.4f})")
    print(f"  Prediction agreement (HW vs SV): {pred_agreement:.3f}")
    print(f"  Transpiled depth: {run['transpiled_depth']}, 2q-gates: {run['two_qubit_gates']}")
    print(f"  QPU seconds: {run['qpu_seconds']:.2f}, wall: {run['wall_seconds']:.2f}s")

    # ---- write summary ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "date": date.today().isoformat(),
        "dataset": args.dataset, "fold_idx": args.fold_idx,
        "backend": run["backend_name"], "hardware": bool(args.hardware),
        "shots": args.shots, "resilience_level": args.resilience_level,
        "enable_dd": enable_dd, "enable_twirling": enable_twirling, "enable_zne": enable_zne,
        "n_qubits": n_qubits, "fusion_reps": reps, "entanglement": ent,
        "mixer_angle": mixer, "head": head,
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "transpiled_depth": run["transpiled_depth"],
        "two_qubit_gates": run["two_qubit_gates"],
        "qpu_seconds": run["qpu_seconds"], "wall_seconds": run["wall_seconds"],
        "job_id": run["job_id"], "job_status": run["job_status"], "job_error": run["job_error"],
        "feature_mad_mean": feature_mad, "feature_mad_max": feature_mad_max,
        "pred_agreement": pred_agreement,
        "acc_sv": metrics_sv["accuracy"], "f1_sv": metrics_sv["macro_f1"], "mcc_sv": metrics_sv["mcc"],
        "auc_sv": metrics_sv["roc_auc"],
        "acc_hw": metrics_hw["accuracy"], "f1_hw": metrics_hw["macro_f1"], "mcc_hw": metrics_hw["mcc"],
        "auc_hw": metrics_hw["roc_auc"],
    }
    out_csv = args.output_dir / "hardware_pilot_summary.csv"
    new_row = pd.DataFrame([summary])
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        out = pd.concat([existing, new_row], ignore_index=True)
    else:
        out = new_row
    out.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(out)} rows total)")

    if args.hardware:
        append_usage(args.usage_log, {
            "date": summary["date"], "backend": run["backend_name"],
            "job_id": run["job_id"], "dataset": args.dataset, "signal_set": "ecg_eda",
            "model": f"fusion_{head}_r{args.resilience_level}",
            "shots": args.shots,
            "resilience": f"L{args.resilience_level}(dd={enable_dd};twirl={enable_twirling};zne={enable_zne})",
            "circuits": len(test_circuits),
            "qpu_seconds": run["qpu_seconds"],
            "notes": f"fold={args.fold_idx}; n_test={len(test_idx)}; status={run['job_status']}; "
                     f"err={run['job_error'][:60]}",
        })
        used = summarize_usage(args.usage_log)
        print(f"Cumulative QPU: {used['total_qpu_minutes']:.1f} min / {BUDGET_HARD_STOP_MIN:.0f} min cap")


if __name__ == "__main__":
    main()
