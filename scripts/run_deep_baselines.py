"""Phase 2 B3 — Deep classical baselines on Track B windows.

LOSO subject-independent CV (groups = dataset::subject_id) on the 30s × 250Hz
ECG+EDA windows from data/processed/*_binary_ecg_eda_30s_50ovl.pkl. Trains a
small 1D-CNN per fold on GPU (RTX 5090, cu128). Optionally also trains a
chunked 1D-Transformer (Behinaein-style) when --models transformer is requested.

Outputs:
- results/per_dataset_deep_v0/<dataset>_<model>_loso.csv
- results/deep_baselines.csv (combined summary across datasets/models/seeds)

Same amplitude-leakage caveat as the classical baselines (per-subject scaling
in preprocess_*.py): apply fold-aware global re-normalization in the dataloader
to partially mitigate. Full R2 refactor is the proper fix.
"""

from __future__ import annotations

import argparse
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------- utility -----------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_artifact(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def fold_normalize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re-normalize per-channel using ONLY training-fold statistics. Partially
    mitigates the per-subject z-scoring leakage in preprocessing.
    """
    # X shape: (N, T, C). Compute mean/std per-channel over (N, T) of training data.
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sd = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return (X_train - mu) / sd, (X_test - mu) / sd


# ----------------------------- models -----------------------------------

class CNN1D(nn.Module):
    """Small 1D-CNN: 4 conv stages → global avg pool → linear head."""

    def __init__(self, in_channels: int = 2, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → (B, C, T)
        x = x.transpose(1, 2)
        return self.head(self.features(x))


class ChunkedTransformer(nn.Module):
    """Chunked 1D Transformer (Behinaein-style). The 7500-sample window is
    chunked into N tokens, each linearly embedded, then passed through encoder.
    """

    def __init__(self, in_channels: int = 2, n_classes: int = 2,
                 chunk_size: int = 250, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.chunk_size = chunk_size
        self.embed = nn.Linear(in_channels * chunk_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). Chunk T into T/chunk_size tokens of (chunk_size*C) features.
        B, T, C = x.shape
        n_chunks = T // self.chunk_size
        x = x[:, : n_chunks * self.chunk_size]
        x = x.reshape(B, n_chunks, self.chunk_size * C)
        tokens = self.embed(x)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out = self.encoder(tokens)
        return self.head(out[:, 0])


def build_model(name: str) -> nn.Module:
    if name == "cnn1d":
        return CNN1D()
    if name == "transformer":
        return ChunkedTransformer()
    raise ValueError(f"Unknown model: {name}")


# ----------------------------- training -----------------------------------

@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    val_frac: float = 0.15
    device: str = "cuda"


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=False)


def train_one_fold(
    model: nn.Module,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    cfg: TrainConfig,
    class_weights: np.ndarray,
) -> dict[str, float]:
    device = torch.device(cfg.device)
    model = model.to(device)

    # validation split from training (random)
    n_train = len(X_train)
    idx = np.arange(n_train)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    n_val = max(1, int(cfg.val_frac * n_train))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    train_loader = _make_loader(X_train[tr_idx], y_train[tr_idx], cfg.batch_size, shuffle=True)
    val_loader = _make_loader(X_train[val_idx], y_train[val_idx], cfg.batch_size, shuffle=False)
    test_loader = _make_loader(X_test, y_test, cfg.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    scaler = torch.amp.GradScaler("cuda")

    best_val, patience, best_state = -1.0, 0, None
    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        # val
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(xb).argmax(dim=1)
                val_correct += (pred == yb).sum().item()
                val_total += yb.numel()
        val_acc = val_correct / max(val_total, 1)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test
    model.eval()
    all_pred, all_prob, all_y = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(xb)
                prob = F.softmax(logits.float(), dim=1)
            all_pred.append(prob.argmax(dim=1).cpu().numpy())
            all_prob.append(prob.cpu().numpy())
            all_y.append(yb.numpy())

    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    y_true = np.concatenate(all_y)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "roc_auc": float("nan"),
    }
    if len(np.unique(y_true)) == 2 and y_prob.shape[1] == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except ValueError:
            pass
    return out


# ----------------------------- LOSO driver --------------------------------

def run_dataset_cv(
    path: Path,
    model_name: str,
    seeds: list[int],
    cfg: TrainConfig,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Subject-grouped 5-fold (StratifiedGroupKFold) CV per project-wide CV switch 2026-05-15."""
    artifact = load_artifact(path)
    X = np.asarray(artifact["data"], dtype=np.float32)
    y = np.asarray(artifact["labels"], dtype=np.int64)
    subjects = np.asarray(artifact["participants"]).astype(str)
    unique_subj = np.unique(subjects)
    print(f"  {path.name}: X={X.shape}, y bal={np.bincount(y).tolist()}, "
          f"subjects={len(unique_subj)}, n_splits={n_splits}")

    rows = []
    for seed in seeds:
        set_seed(seed)
        # Use seed=42 for splitter shuffle (fixed across model seeds) to keep
        # the same folds for direct comparison; the seed here perturbs model
        # initialisation / dropout / batch shuffling.
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []
        for tr_idx, te_idx in splitter.split(np.zeros(len(y)), y, subjects):
            if len(np.unique(y[tr_idx])) < 2:
                continue
            X_tr, X_te = fold_normalize(X[tr_idx], X[te_idx])
            y_tr, y_te = y[tr_idx], y[te_idx]
            counts = np.bincount(y_tr, minlength=2)
            class_w = counts.sum() / (2 * np.maximum(counts, 1))
            model = build_model(model_name)
            m = train_one_fold(model, X_tr, y_tr, X_te, y_te, cfg, class_w)
            fold_metrics.append(m)
        if not fold_metrics:
            continue
        df = pd.DataFrame(fold_metrics)
        rows.append({
            "random_state": seed,
            "model": model_name,
            "n_folds": int(len(df)),
            "accuracy_mean": float(df["accuracy"].mean()),
            "accuracy_std": float(df["accuracy"].std(ddof=0)),
            "macro_f1_mean": float(df["macro_f1"].mean()),
            "macro_f1_std": float(df["macro_f1"].std(ddof=0)),
            "mcc_mean": float(df["mcc"].mean()),
            "mcc_std": float(df["mcc"].std(ddof=0)),
            "roc_auc_mean": float(df["roc_auc"].mean()),
            "roc_auc_std": float(df["roc_auc"].std(ddof=0)),
        })
    return pd.DataFrame(rows)


# Back-compat alias
run_dataset_loso = run_dataset_cv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--pattern", default="*_binary_ecg_eda_30s_50ovl.pkl")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Subset of datasets (e.g. WESAD CLAS). Default: all.")
    parser.add_argument("--models", nargs="+", default=["cnn1d"],
                        choices=["cnn1d", "transformer"])
    parser.add_argument("--random-states", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results/per_dataset_deep_v0"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/deep_baselines.csv"))
    args = parser.parse_args()

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      patience=args.patience)

    paths = sorted(args.processed_dir.glob(args.pattern))
    if args.datasets:
        paths = [p for p in paths if any(d in p.name for d in args.datasets)]
    if not paths:
        raise FileNotFoundError(f"No matching pkls in {args.processed_dir}")

    print(f"Device: {cfg.device}, models={args.models}, seeds={args.random_states}, "
          f"epochs={cfg.epochs}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for path in paths:
        dataset = path.name.split("_")[0]
        for model_name in args.models:
            t0 = time.time()
            print(f"\n=== {dataset} / {model_name} ===")
            df = run_dataset_cv(path, model_name, args.random_states, cfg)
            df.insert(0, "dataset", dataset)
            out = args.output_dir / f"{path.stem}_{model_name}_kfold5.csv"
            df.to_csv(out, index=False)
            print(f"  Wrote {out} ({time.time() - t0:.1f}s)")
            print(df.to_string(index=False))
            all_rows.append(df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(args.summary_output, index=False)
        print(f"\nWrote combined summary: {args.summary_output}")


if __name__ == "__main__":
    main()
