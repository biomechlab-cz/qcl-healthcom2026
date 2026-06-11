"""Run subject-independent classical baselines on ECG/EDA feature tables."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
        raise FileNotFoundError("No feature files matched")
    return paths


def make_group_labels(frame: pd.DataFrame) -> pd.Series:
    return frame["dataset"].astype(str) + "::" + frame["subject_id"].astype(str)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric = frame.select_dtypes(include=[np.number]).columns
    return [column for column in numeric if column not in NON_FEATURE_COLUMNS]


def _model(name: str, random_state: int) -> Pipeline:
    if name == "logreg":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    elif name == "rbf_svm":
        estimator = SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", probability=True)
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )
    elif name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        )
    elif name == "xgboost" and importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
    elif name == "lightgbm" and importlib.util.find_spec("lightgbm"):
        from lightgbm import LGBMClassifier

        estimator = LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown or unavailable model: {name}")

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _score_fold(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None) -> dict[str, float]:
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


def _mean_std(values: pd.Series) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    if np.isnan(array).all():
        return np.nan, np.nan
    return float(np.nanmean(array)), float(np.nanstd(array))


def _summarize_scores(scores: pd.DataFrame, count_name: str, count_value: int, model_name: str) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"model": model_name, count_name: int(count_value)}
    for metric in ["accuracy", "macro_f1", "mcc", "roc_auc"]:
        mean, std = _mean_std(scores[metric])
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = std
    return row


def run_loso_baselines(
    frame: pd.DataFrame,
    models: Iterable[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    models = list(models or ["logreg", "rbf_svm", "random_forest", "extra_trees"])
    columns = feature_columns(frame)
    if not columns:
        raise ValueError("No numeric feature columns found")

    X = frame[columns].to_numpy(dtype=float)
    y = frame["label"].to_numpy()
    groups = make_group_labels(frame).to_numpy()
    splitter = LeaveOneGroupOut()

    rows = []
    for model_name in models:
        fold_scores = []
        for train_idx, test_idx in splitter.split(X, y, groups):
            if len(np.unique(y[train_idx])) < 2:
                continue
            clf = _model(model_name, random_state)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            proba = clf.predict_proba(X[test_idx]) if hasattr(clf, "predict_proba") else None
            fold_scores.append(_score_fold(y[test_idx], pred, proba))

        if not fold_scores:
            continue

        rows.append(_summarize_scores(pd.DataFrame(fold_scores), "n_folds", len(fold_scores), model_name))

    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def run_grouped_cv_baselines(
    frame: pd.DataFrame,
    models: Iterable[str] | None = None,
    random_state: int = 42,
    n_splits: int = 5,
) -> pd.DataFrame:
    models = list(models or ["logreg", "rbf_svm", "random_forest", "extra_trees"])
    columns = feature_columns(frame)
    if not columns:
        raise ValueError("No numeric feature columns found")

    X = frame[columns].to_numpy(dtype=float)
    y = frame["label"].to_numpy()
    groups = make_group_labels(frame).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for model_name in models:
        fold_scores = []
        for train_idx, test_idx in splitter.split(X, y, groups):
            if len(np.unique(y[train_idx])) < 2:
                continue
            clf = _model(model_name, random_state)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            proba = clf.predict_proba(X[test_idx]) if hasattr(clf, "predict_proba") else None
            fold_scores.append(_score_fold(y[test_idx], pred, proba))

        if not fold_scores:
            continue

        rows.append(_summarize_scores(pd.DataFrame(fold_scores), "n_folds", len(fold_scores), model_name))

    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def run_leave_one_dataset_out(
    frame: pd.DataFrame,
    models: Iterable[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    models = list(models or ["logreg", "rbf_svm", "random_forest", "extra_trees"])
    columns = feature_columns(frame)
    if not columns:
        raise ValueError("No numeric feature columns found")

    X = frame[columns].to_numpy(dtype=float)
    y = frame["label"].to_numpy()
    datasets = frame["dataset"].astype(str).to_numpy()

    rows = []
    for model_name in models:
        dataset_scores = []
        for held_out in sorted(np.unique(datasets)):
            train_idx = np.where(datasets != held_out)[0]
            test_idx = np.where(datasets == held_out)[0]
            if train_idx.size == 0 or test_idx.size == 0 or len(np.unique(y[train_idx])) < 2:
                continue
            clf = _model(model_name, random_state)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            proba = clf.predict_proba(X[test_idx]) if hasattr(clf, "predict_proba") else None
            score = _score_fold(y[test_idx], pred, proba)
            score["held_out_dataset"] = held_out
            dataset_scores.append(score)

        if not dataset_scores:
            continue

        rows.append(_summarize_scores(pd.DataFrame(dataset_scores), "n_datasets", len(dataset_scores), model_name))

    return pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)


def load_features(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.suffix.lower() == ".csv":
            frames.append(pd.read_csv(path))
        else:
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", nargs="+", type=Path)
    parser.add_argument("--models", nargs="+", default=["logreg", "rbf_svm", "random_forest", "extra_trees"])
    parser.add_argument("--output", type=Path, default=Path("results/classical_loso.csv"))
    parser.add_argument("--output-dir", type=Path, help="Run LOSO separately for each input feature file")
    parser.add_argument("--cv", choices=["loso", "group-kfold"], default="loso")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--random-states", nargs="+", type=int, help="Run multiple random states and concatenate summaries")
    parser.add_argument("--lodo-output", type=Path, help="Optional leave-one-dataset-out summary CSV")
    args = parser.parse_args()

    paths = expand_inputs(args.features)
    random_states = args.random_states or [args.random_state]
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            frame = load_features([path])
            result_frames = []
            for random_state in random_states:
                if args.cv == "group-kfold":
                    results = run_grouped_cv_baselines(
                        frame, models=args.models, n_splits=args.n_splits, random_state=random_state
                    )
                else:
                    results = run_loso_baselines(frame, models=args.models, random_state=random_state)
                results.insert(0, "random_state", random_state)
                result_frames.append(results)
            results = pd.concat(result_frames, ignore_index=True)
            output = args.output_dir / f"{path.stem}_{args.cv}.csv"
            results.to_csv(output, index=False)
            print(f"\n{path}:")
            print(results.to_string(index=False))
        return

    frame = load_features(paths)
    result_frames = []
    for random_state in random_states:
        if args.cv == "group-kfold":
            results = run_grouped_cv_baselines(
                frame, models=args.models, n_splits=args.n_splits, random_state=random_state
            )
        else:
            results = run_loso_baselines(frame, models=args.models, random_state=random_state)
        results.insert(0, "random_state", random_state)
        result_frames.append(results)
    results = pd.concat(result_frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))

    if args.lodo_output:
        lodo_frames = []
        for random_state in random_states:
            lodo = run_leave_one_dataset_out(frame, models=args.models, random_state=random_state)
            lodo.insert(0, "random_state", random_state)
            lodo_frames.append(lodo)
        lodo = pd.concat(lodo_frames, ignore_index=True)
        args.lodo_output.parent.mkdir(parents=True, exist_ok=True)
        lodo.to_csv(args.lodo_output, index=False)
        print("\nLeave-one-dataset-out:")
        print(lodo.to_string(index=False))


if __name__ == "__main__":
    main()
