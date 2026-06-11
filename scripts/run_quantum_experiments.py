"""Run simulator-based quantum feature experiments with grouped splits."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantum_models import (  # noqa: E402
    build_fusion_circuit,
    build_qrc_circuit,
    circuit_summary,
    temporal_tokens,
    z_zz_expectations,
)


ECG_COLUMNS = [
    "ecg_mean_hr_bpm",
    "ecg_sdnn_ms",
    "ecg_rmssd_ms",
    "ecg_pnn50",
    "ecg_peak_count",
    "ecg_sampen",
    "ecg_std",
    "ecg_slope",
]
EDA_COLUMNS = [
    "eda_tonic_mean",
    "eda_std",
    "eda_slope",
    "eda_scr_count",
    "eda_scr_amplitude_mean",
    "eda_variability",
]
NON_FEATURE_COLUMNS = {
    "dataset",
    "task",
    "signal_set",
    "subject_id",
    "participant",
    "label",
    "window_idx",
}


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
        raise FileNotFoundError("No input files matched")
    return paths


def load_features(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in expand_inputs(paths):
        frames.append(pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def load_artifacts(paths: list[Path]) -> tuple[np.ndarray, pd.DataFrame]:
    data_parts = []
    meta_parts = []
    offset = 0
    for path in expand_inputs(paths):
        with path.open("rb") as handle:
            artifact = pickle.load(handle)
        data = np.asarray(artifact["data"], dtype=np.float32)
        labels = np.asarray(artifact["labels"], dtype=int)
        participants = np.asarray(artifact["participants"]).astype(str)
        dataset = str(artifact.get("dataset") or path.stem.split("_")[0])
        task = str(artifact.get("task") or "unknown")
        signal_set = "ecg_eda" if "EDA" in [str(ch).upper() for ch in artifact.get("channels", [])] else "ecg"
        meta_parts.append(
            pd.DataFrame(
                {
                    "dataset": dataset,
                    "task": task,
                    "signal_set": signal_set,
                    "subject_id": participants,
                    "label": labels,
                    "window_idx": np.arange(labels.size),
                    "row_idx": np.arange(offset, offset + labels.size),
                }
            )
        )
        data_parts.append(data)
        offset += labels.size
    return np.concatenate(data_parts, axis=0), pd.concat(meta_parts, ignore_index=True)


def make_group_labels(frame: pd.DataFrame) -> np.ndarray:
    return (frame["dataset"].astype(str) + "::" + frame["subject_id"].astype(str)).to_numpy()


def select_feature_columns(frame: pd.DataFrame, signal_set: str, n_circuit_inputs: int) -> list[str]:
    if signal_set == "ecg_eda":
        per_modality = max(1, n_circuit_inputs // 2)
        ecg_columns = [column for column in ECG_COLUMNS if column in frame.columns][:per_modality]
        eda_columns = [column for column in EDA_COLUMNS if column in frame.columns][:per_modality]
        columns = ecg_columns + eda_columns
    else:
        columns = [column for column in ECG_COLUMNS if column in frame.columns][:n_circuit_inputs]
    numeric = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in NON_FEATURE_COLUMNS and column not in columns
    ]
    columns.extend(numeric)
    if len(columns) < n_circuit_inputs:
        raise ValueError(f"Need at least {n_circuit_inputs} feature columns, found {len(columns)}")
    return columns[:n_circuit_inputs]


def make_readout(name: str, random_state: int) -> Pipeline:
    if name == "ridge":
        estimator = RidgeClassifier(class_weight="balanced")
    elif name == "logreg":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    elif name == "rbf_svm":
        estimator = SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", probability=True)
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=1)
    else:
        raise ValueError(f"Unknown readout model: {name}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)])


def score_fold(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None) -> dict[str, float]:
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0,
        "roc_auc": np.nan,
    }
    if proba is not None and len(np.unique(y_true)) == 2 and proba.shape[1] == 2:
        try:
            row["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
        except ValueError:
            row["roc_auc"] = np.nan
    return row


def summarize_scores(rows: list[dict[str, float]], model: str, metadata: dict[str, object]) -> dict[str, object]:
    scores = pd.DataFrame(rows)
    summary: dict[str, object] = {**metadata, "model": model, "n_folds": int(len(scores))}
    for metric in ["accuracy", "macro_f1", "mcc", "roc_auc"]:
        values = scores[metric].to_numpy(dtype=float)
        if np.isnan(values).all():
            summary[f"{metric}_mean"] = np.nan
            summary[f"{metric}_std"] = np.nan
        else:
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = float(np.nanstd(values))
    return summary


def split_indices(frame: pd.DataFrame, cv: str, n_splits: int, random_state: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    y = frame["label"].to_numpy()
    groups = make_group_labels(frame)
    if cv == "loso":
        yield from LeaveOneGroupOut().split(np.zeros(len(frame)), y, groups)
    elif cv == "group-kfold":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        yield from splitter.split(np.zeros(len(frame)), y, groups)
    else:
        datasets = frame["dataset"].astype(str).to_numpy()
        for held_out in sorted(np.unique(datasets)):
            train_idx = np.where(datasets != held_out)[0]
            test_idx = np.where(datasets == held_out)[0]
            if train_idx.size and test_idx.size:
                yield train_idx, test_idx


def sample_frame(frame: pd.DataFrame, max_samples_per_class: int | None, random_state: int) -> pd.DataFrame:
    if not max_samples_per_class:
        return frame.reset_index(drop=True)
    pieces = []
    for _, group in frame.groupby("label"):
        pieces.append(group.sample(min(len(group), max_samples_per_class), random_state=random_state))
    sampled = pd.concat(pieces).sort_index()
    return sampled.reset_index(drop=True)


def fusion_quantum_features(
    values: np.ndarray,
    channels: list[str],
    qubits_per_channel: int,
    fusion_reps: int,
    fusion_entanglement: str,
    mixer_angle: float,
) -> np.ndarray:
    rows = []
    for row in values:
        circuit = build_fusion_circuit(
            row,
            channels=channels,
            qubits_per_channel=qubits_per_channel,
            reps=fusion_reps,
            entanglement=fusion_entanglement,
            mixer_angle=mixer_angle,
        )
        rows.append(z_zz_expectations(circuit))
    return np.asarray(rows, dtype=float)


def qrc_quantum_features(data: np.ndarray, n_qubits: int, n_tokens: int) -> np.ndarray:
    rows = []
    for window in data:
        tokens = temporal_tokens(window, n_tokens=n_tokens)
        circuit = build_qrc_circuit(tokens, n_qubits=n_qubits)
        rows.append(z_zz_expectations(circuit))
    return np.asarray(rows, dtype=float)


def run_fusion(
    frame: pd.DataFrame,
    models: list[str],
    cv: str,
    n_splits: int,
    random_state: int,
    scale_mode: str,
    qubits_per_channel: int,
    fusion_reps: int,
    fusion_entanglement: str,
    mixer_angle: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    signal_set = str(frame["signal_set"].iloc[0])
    channels = ["ECG", "EDA"] if signal_set == "ecg_eda" else ["ECG"]
    n_inputs = len(channels) * qubits_per_channel
    columns = select_feature_columns(frame, signal_set, n_inputs)
    y = frame["label"].to_numpy()
    splits = list(split_indices(frame, cv, n_splits, random_state))
    metadata = {
        "quantum_mode": "fusion",
        "cv": cv,
        "signal_set": signal_set,
        "n_input_features": len(columns),
        "n_quantum_features": len(channels) * qubits_per_channel * 2 - 1,
        "fusion_reps": fusion_reps,
        "fusion_entanglement": fusion_entanglement,
        "mixer_angle": mixer_angle,
    }
    cost = circuit_summary(
        build_fusion_circuit(
            np.zeros(n_inputs),
            channels=channels,
            qubits_per_channel=qubits_per_channel,
            reps=fusion_reps,
            entanglement=fusion_entanglement,
            mixer_angle=mixer_angle,
        )
    )
    results = []

    if scale_mode == "global":
        scaler = StandardScaler()
        all_scaled = scaler.fit_transform(frame[columns].to_numpy(dtype=float))
        all_quantum = fusion_quantum_features(
            all_scaled,
            channels,
            qubits_per_channel,
            fusion_reps,
            fusion_entanglement,
            mixer_angle,
        )

    for model_name in models:
        fold_scores = []
        for train_idx, test_idx in splits:
            if len(np.unique(y[train_idx])) < 2:
                continue
            if scale_mode == "global":
                q_train = all_quantum[train_idx]
                q_test = all_quantum[test_idx]
            else:
                scaler = StandardScaler()
                q_train = fusion_quantum_features(
                    scaler.fit_transform(frame.iloc[train_idx][columns].to_numpy(dtype=float)),
                    channels,
                    qubits_per_channel,
                    fusion_reps,
                    fusion_entanglement,
                    mixer_angle,
                )
                q_test = fusion_quantum_features(
                    scaler.transform(frame.iloc[test_idx][columns].to_numpy(dtype=float)),
                    channels,
                    qubits_per_channel,
                    fusion_reps,
                    fusion_entanglement,
                    mixer_angle,
                )
            clf = make_readout(model_name, random_state)
            clf.fit(q_train, y[train_idx])
            pred = clf.predict(q_test)
            proba = clf.predict_proba(q_test) if hasattr(clf, "predict_proba") else None
            fold_scores.append(score_fold(y[test_idx], pred, proba))
        if fold_scores:
            results.append(summarize_scores(fold_scores, model_name, metadata))
    return pd.DataFrame(results).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True), cost


def run_qrc(
    data: np.ndarray,
    frame: pd.DataFrame,
    models: list[str],
    cv: str,
    n_splits: int,
    random_state: int,
    qrc_qubits: int,
    qrc_tokens: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    y = frame["label"].to_numpy()
    splits = list(split_indices(frame, cv, n_splits, random_state))
    quantum = qrc_quantum_features(data, qrc_qubits, qrc_tokens)
    metadata = {
        "quantum_mode": "qrc",
        "cv": cv,
        "signal_set": str(frame["signal_set"].iloc[0]),
        "n_input_features": int(qrc_tokens * data.shape[2]),
        "n_quantum_features": int(quantum.shape[1]),
    }
    cost = circuit_summary(build_qrc_circuit(temporal_tokens(data[0], n_tokens=qrc_tokens), n_qubits=qrc_qubits))
    results = []
    for model_name in models:
        fold_scores = []
        for train_idx, test_idx in splits:
            if len(np.unique(y[train_idx])) < 2:
                continue
            clf = make_readout(model_name, random_state)
            clf.fit(quantum[train_idx], y[train_idx])
            pred = clf.predict(quantum[test_idx])
            proba = clf.predict_proba(quantum[test_idx]) if hasattr(clf, "predict_proba") else None
            fold_scores.append(score_fold(y[test_idx], pred, proba))
        if fold_scores:
            results.append(summarize_scores(fold_scores, model_name, metadata))
    return pd.DataFrame(results).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True), cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", type=Path, help="Feature parquet/csv inputs for fusion mode")
    parser.add_argument("--artifacts", nargs="+", type=Path, help="Harmonized pickle artifacts for QRC mode")
    parser.add_argument("--mode", choices=["fusion", "qrc"], required=True)
    parser.add_argument("--cv", choices=["loso", "group-kfold", "lodo"], default="loso")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["ridge", "logreg", "rbf_svm"])
    parser.add_argument("--scale-mode", choices=["fold", "global"], default="fold")
    parser.add_argument("--max-samples-per-class", type=int)
    parser.add_argument("--qubits-per-channel", type=int, default=3)
    parser.add_argument("--fusion-reps", type=int, default=1)
    parser.add_argument("--fusion-entanglement", choices=["minimal", "full"], default="minimal")
    parser.add_argument("--mixer-angle", type=float, default=0.0)
    parser.add_argument("--qrc-qubits", type=int, default=6)
    parser.add_argument("--qrc-tokens", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--random-states", nargs="+", type=int, help="Run multiple random states and concatenate summaries")
    parser.add_argument("--output", type=Path, default=Path("results/quantum_results.csv"))
    parser.add_argument("--circuit-output", type=Path, default=Path("results/quantum_circuit_costs.csv"))
    args = parser.parse_args()

    random_states = args.random_states or [args.random_state]
    result_frames = []
    cost_rows = []
    for random_state in random_states:
        if args.mode == "fusion":
            if not args.features:
                raise ValueError("--features is required for fusion mode")
            frame = sample_frame(load_features(args.features), args.max_samples_per_class, random_state)
            results, cost = run_fusion(
                frame=frame,
                models=args.models,
                cv=args.cv,
                n_splits=args.n_splits,
                random_state=random_state,
                scale_mode=args.scale_mode,
                qubits_per_channel=args.qubits_per_channel,
                fusion_reps=args.fusion_reps,
                fusion_entanglement=args.fusion_entanglement,
                mixer_angle=args.mixer_angle,
            )
        else:
            if not args.artifacts:
                raise ValueError("--artifacts is required for qrc mode")
            data, frame = load_artifacts(args.artifacts)
            if args.max_samples_per_class:
                sampled = sample_frame(frame, args.max_samples_per_class, random_state)
                data = data[sampled["row_idx"].to_numpy(dtype=int)]
                frame = sampled
            results, cost = run_qrc(
                data=data,
                frame=frame,
                models=args.models,
                cv=args.cv,
                n_splits=args.n_splits,
                random_state=random_state,
                qrc_qubits=args.qrc_qubits,
                qrc_tokens=args.qrc_tokens,
            )
        results.insert(0, "random_state", random_state)
        result_frames.append(results)
        cost_rows.append(
            {
                "output": str(args.output),
                "random_state": random_state,
                "mode": args.mode,
                "cv": args.cv,
                "scale_mode": args.scale_mode if args.mode == "fusion" else "window",
                **cost,
            }
        )

    results = pd.concat(result_frames, ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    cost_frame = pd.DataFrame(cost_rows)
    if args.circuit_output.exists():
        cost_frame = pd.concat([pd.read_csv(args.circuit_output), cost_frame], ignore_index=True)
    cost_frame.to_csv(args.circuit_output, index=False)
    print(results.to_string(index=False))
    print("\nCircuit cost:")
    print(pd.DataFrame(cost_rows).to_string(index=False))


if __name__ == "__main__":
    main()
