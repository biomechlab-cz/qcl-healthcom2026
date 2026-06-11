"""Phase 2 (post-R2+B2) - Classical baselines on the v1 feature set.

Project-wide CV per memory/feedback_hardware_cv_strategy.md (2026-05-15):
**5-fold StratifiedGroupKFold with groups = dataset::subject_id is the
only CV protocol.** LOSO is dropped everywhere.

With R2 (deferred scaling) and B2 (expanded HRV: ApEn, Poincare SD1/SD2/ratio,
DFA-alpha1, LF/HF) landed, the amplitude features no longer leak label info via
per-subject preprocessing. We can therefore use the full 27-feature set.

Outputs:
- results/v1/{dataset}_kfold5.csv
- results/v1/pooled_kfold5.csv
- results/v1/pooled_lodo.csv (cross-dataset stress test)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_classical_baselines import (  # noqa: E402
    run_grouped_cv_baselines,
    run_leave_one_dataset_out,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["WESAD", "CLAS"])
    parser.add_argument("--features-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--models", nargs="+",
                        default=["logreg", "rbf_svm", "random_forest", "extra_trees", "xgboost", "lightgbm"])
    parser.add_argument("--random-states", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v1"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for ds in args.datasets:
        p = args.features_dir / f"{ds}_binary_ecg_eda_30s_50ovl.parquet"
        df = pd.read_parquet(p)
        n_feats = len([c for c in df.columns if c not in {
            "dataset", "task", "signal_set", "subject_id", "participant", "label", "window_idx",
        }])
        print(f"Loaded {ds}: {len(df)} rows, {n_feats} features, "
              f"subjects={df['subject_id'].nunique()}")

        kf_frames = []
        for seed in args.random_states:
            res = run_grouped_cv_baselines(df, models=args.models, n_splits=args.n_splits,
                                           random_state=seed)
            res.insert(0, "random_state", seed)
            kf_frames.append(res)
        kf = pd.concat(kf_frames, ignore_index=True)
        kf.to_csv(args.output_dir / f"{ds}_kfold5.csv", index=False)
        best = kf.loc[kf["macro_f1_mean"].idxmax()]
        print(f"  {ds} 5-fold best: {best['model']} @ seed {int(best['random_state'])} -> "
              f"acc {best['accuracy_mean']:.3f}+-{best['accuracy_std']:.3f}, "
              f"F1 {best['macro_f1_mean']:.3f}, ROC-AUC {best['roc_auc_mean']:.3f}")

        frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)
    print(f"\nPooled rows: {len(pooled)}, "
          f"subjects: {pooled.assign(g=pooled['dataset']+'::'+pooled['subject_id']).g.nunique()}")

    kf_frames, lodo_frames = [], []
    for seed in args.random_states:
        kf_frames.append(run_grouped_cv_baselines(pooled, models=args.models, n_splits=args.n_splits,
                                                  random_state=seed).assign(random_state=seed))
        lodo_frames.append(run_leave_one_dataset_out(pooled, models=args.models,
                                                     random_state=seed).assign(random_state=seed))
    pd.concat(kf_frames, ignore_index=True).to_csv(args.output_dir / "pooled_kfold5.csv", index=False)
    pd.concat(lodo_frames, ignore_index=True).to_csv(args.output_dir / "pooled_lodo.csv", index=False)

    for label, fr in [("pooled 5-fold", pd.concat(kf_frames)),
                      ("pooled LODO (cross-dataset)", pd.concat(lodo_frames))]:
        best = fr.loc[fr["macro_f1_mean"].idxmax()]
        print(f"  {label} best: {best['model']} -> "
              f"acc {best['accuracy_mean']:.3f}+-{best['accuracy_std']:.3f}, "
              f"F1 {best['macro_f1_mean']:.3f}, ROC-AUC {best['roc_auc_mean']:.3f}")

    print(f"\nAll outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
