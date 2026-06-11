import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neurokit2 as nk
from glob import glob
import numpy as np
import logging
import pickle

from utils import utils
from sklearn.preprocessing import StandardScaler

# WESAD label mapping:
# 0 = undefined, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation
# 5/6/7 = other conditions
# Original code relabels part of stress (label=2) segment as label=8 after 5.5min offset
WESAD_LABELS = {
    "baseline": 1,
    "stress": 2,
    "amusement": 3,
    "meditation": 4,
}

ORIGINAL_FS = 700

# Signals available in WESAD
AVAILABLE_SIGNALS = {"ECG", "RSP", "EDA"}


def load_subject_data(file):
    with open(file, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    ecg = data["signal"]["chest"]["ECG"].flatten()
    rsp = data["signal"]["chest"]["Resp"].flatten()
    eda = data["signal"]["chest"]["EDA"].flatten()
    labels = data["label"]
    subject = data["subject"]

    # Apply stress onset offset: relabel stress segment after 5.5 minutes
    # to account for stress response delay (TSST protocol)
    stress_indices = np.where(labels == WESAD_LABELS["stress"])[0]
    if len(stress_indices) > 0:
        first = stress_indices[0]
        last = stress_indices[-1]
        offset = int(5.5 * 60 * ORIGINAL_FS)
        if first + offset < last:
            labels[first + offset : last + 1] = 8  # relabel as "active stress"

    return {"ECG": ecg, "RSP": rsp, "EDA": eda}, labels, subject


def preprocess_subject(raw_signals, labels, fs, win, overlap, keep_labels, use_signals,
                       scale_signals=False):
    """Resample, clean, and (optionally) z-score each signal per subject.

    Phase 2 R2 (2026-05-15): scale_signals defaults to False. Per-subject
    z-scoring leaks label info into amplitude features (see
    reports/phase2_leakage.md). Downstream pipelines do per-fold feature-level
    scaling inside their sklearn Pipeline, which is the proper LOSO-safe step.
    """
    processed = {}
    for sig_name in use_signals:
        raw = raw_signals[sig_name]
        resampled = nk.signal_resample(raw, sampling_rate=ORIGINAL_FS, desired_sampling_rate=fs)

        if sig_name == "ECG":
            cleaned = nk.ecg_clean(resampled, sampling_rate=fs)
        elif sig_name == "RSP":
            cleaned = nk.rsp_clean(resampled, sampling_rate=fs)
        elif sig_name == "EDA":
            cleaned = nk.eda_clean(resampled, sampling_rate=fs)

        if scale_signals:
            scaler = StandardScaler()
            processed[sig_name] = scaler.fit_transform(cleaned.reshape(-1, 1)).flatten()
        else:
            processed[sig_name] = cleaned.astype(np.float32)

    # Resample labels (nearest-neighbor)
    ref_len = len(next(iter(processed.values())))
    label_indices = np.linspace(0, len(labels) - 1, ref_len)
    labels_r = labels[np.round(label_indices).astype(int)]

    # Segment by contiguous label regions
    all_windows = []
    all_labels = []

    changes = np.where(np.diff(labels_r) != 0)[0] + 1
    segments = np.split(np.arange(len(labels_r)), changes)

    for seg_idx in segments:
        if len(seg_idx) == 0:
            continue
        label = int(labels_r[seg_idx[0]])
        if label not in keep_labels:
            continue

        start, end = seg_idx[0], seg_idx[-1] + 1

        wins_per_sig = []
        for sig_name in use_signals:
            seg = processed[sig_name][start:end]
            w = utils.create_windows(seg, fs, win, overlap)
            wins_per_sig.append(w)

        n = min(len(w) for w in wins_per_sig)
        if n == 0:
            continue

        stacked = np.stack([w[:n] for w in wins_per_sig], axis=-1)
        all_windows.append(stacked)
        all_labels.extend([label] * n)

    if len(all_windows) == 0:
        return np.empty((0, int(fs * win), len(use_signals))), np.array([])

    return np.concatenate(all_windows, axis=0), np.array(all_labels)


def preprocess_wesad(src_dir, out_dir, fs, win, overlap, keep_labels, label_map, signals=None,
                     scale_signals=False):
    logging.info(f"Preprocessing WESAD dataset (scale_signals={scale_signals})")
    files = sorted(glob(f"{src_dir}/S*/S*.pkl"))

    # Determine which signals to use
    requested = signals if signals else list(AVAILABLE_SIGNALS)
    use_signals = [s for s in requested if s in AVAILABLE_SIGNALS]
    logging.info(f"  Signals: requested={requested}, using={use_signals}")

    all_data = []
    all_labels = []
    all_participants = []

    for file in files:
        raw_signals, labels, subject = load_subject_data(file)
        logging.info(f"Processing subject {subject}")

        data, seg_labels = preprocess_subject(
            raw_signals, labels, fs, win, overlap, keep_labels, use_signals,
            scale_signals=scale_signals,
        )

        if len(data) == 0:
            logging.warning(f"No valid windows for subject {subject}")
            continue

        model_labels = np.array([label_map.get(int(l), -1) for l in seg_labels])
        valid = model_labels >= 0
        data = data[valid]
        model_labels = model_labels[valid]

        all_data.append(data)
        all_labels.append(model_labels)
        all_participants.extend([subject] * len(model_labels))

    result = {
        "data": np.concatenate(all_data, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "participants": np.array(all_participants),
        "channels": use_signals,
    }
    logging.info(f"WESAD: {result['data'].shape[0]} windows, shape {result['data'].shape}, "
                 f"channels={use_signals}")
    return result


if __name__ == "__main__":
    utils.setup_logging()

    fs = 1024
    win = 1
    overlap = 0.0
    src_dir = "../data/raw/WESAD/data"
    out_dir = "../data/processed/"

    keep_labels = [1, 8]
    label_map = {1: 0, 8: 1}

    result = preprocess_wesad(src_dir, out_dir, fs, win, overlap, keep_labels, label_map)
    with open(out_dir + "WESAD_preprocessed.pkl", "wb") as f:
        pickle.dump(result, f)
