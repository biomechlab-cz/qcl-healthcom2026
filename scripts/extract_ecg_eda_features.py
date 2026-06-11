"""Extract ECG-first tabular features from harmonized ECG/EDA artifacts.

Phase 2 B2 (2026-05-15): expanded the ECG side to add Approximate Entropy
(ApEn), Poincaré SD1/SD2/ratio, DFA-α1, and frequency-domain LF/HF where
window length permits (≥30 s of RR data). All additions are scaling-invariant
(derived from R-R intervals, not raw amplitude), so they continue to be safe
under per-fold modeling-time scaling (see reports/phase2_leakage.md R2).
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d


METADATA_COLUMNS = {
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
            matches = sorted(parent.glob(value.name))
            paths.extend(matches)
        else:
            paths.append(value)
    if not paths:
        raise FileNotFoundError("No input artifacts matched")
    return paths


def _safe(value: float, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def _slope(x: np.ndarray, fs: float) -> float:
    if x.size < 2:
        return 0.0
    seconds = np.arange(x.size, dtype=float) / max(fs, 1.0)
    return _safe(np.polyfit(seconds, x.astype(float), 1)[0])


def _sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2, max_points: int = 96) -> float:
    x = np.asarray(x, dtype=float)
    if x.size > max_points:
        indices = np.linspace(0, x.size - 1, max_points).astype(int)
        x = x[indices]
    if x.size < 30:
        return 0.0
    r = r_ratio * np.std(x)
    if r <= 0:
        return 0.0

    def count_matches(order: int) -> int:
        count = 0
        for i in range(x.size - order):
            template = x[i : i + order]
            for j in range(i + 1, x.size - order + 1):
                if np.max(np.abs(template - x[j : j + order])) <= r:
                    count += 1
        return count

    a = count_matches(m + 1)
    b = count_matches(m)
    if a == 0 or b == 0:
        return 0.0
    return _safe(-np.log(a / b))


def _approximate_entropy(rr: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """ApEn(m, r=r_ratio*std, N) per Pincus 1991. r_ratio*std avoids
    pathological sensitivity to the absolute scale of RR intervals."""
    rr = np.asarray(rr, dtype=float)
    N = rr.size
    if N < m + 2:
        return 0.0
    r = r_ratio * np.std(rr)
    if r <= 0:
        return 0.0

    def _phi(mm: int) -> float:
        if N - mm + 1 <= 0:
            return 0.0
        templates = np.array([rr[i : i + mm] for i in range(N - mm + 1)])
        # max-norm distances pairwise
        sums = 0.0
        for i in range(templates.shape[0]):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            sums += np.log(np.mean(d <= r))
        return sums / (N - mm + 1)

    return _safe(_phi(m) - _phi(m + 1))


def _dfa_alpha1(rr: np.ndarray, scales: tuple[int, ...] = (4, 5, 6, 7, 8, 10, 12, 16)) -> float:
    """Short-term DFA scaling exponent α1 over RR intervals.

    Returns 0.0 when the RR series is too short for any scale (< 4 beats per scale).
    For 30 s windows this typically yields ~30-40 RR points, so the upper scales
    (12, 16) often drop out. We require ≥ 2 valid scales for a fit; else 0.
    """
    rr = np.asarray(rr, dtype=float)
    if rr.size < 8:
        return 0.0
    x = np.cumsum(rr - rr.mean())
    valid_scales = [s for s in scales if s * 2 <= x.size]
    if len(valid_scales) < 2:
        return 0.0
    fluct = []
    used = []
    for s in valid_scales:
        n_segments = x.size // s
        rms_chunks = []
        for k in range(n_segments):
            seg = x[k * s : (k + 1) * s]
            t = np.arange(s)
            coeffs = np.polyfit(t, seg, 1)
            trend = np.polyval(coeffs, t)
            rms_chunks.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if not rms_chunks:
            continue
        f = float(np.mean(rms_chunks))
        if f > 0:
            fluct.append(f)
            used.append(s)
    if len(fluct) < 2:
        return 0.0
    log_s = np.log(used)
    log_f = np.log(fluct)
    alpha = np.polyfit(log_s, log_f, 1)[0]
    return _safe(float(alpha))


def _hrv_frequency(rr: np.ndarray, min_seconds: float = 20.0) -> tuple[float, float, float]:
    """LF / HF / LF-HF ratio via Welch on interpolated RR series.

    Returns zeros if the RR series spans less than min_seconds (defaults to 20 s,
    below which HF estimation is unreliable). LF (0.04-0.15 Hz) is noisy at
    sub-minute windows; we report it anyway as a noisy feature.
    """
    rr = np.asarray(rr, dtype=float)
    if rr.size < 6:
        return 0.0, 0.0, 0.0
    t = np.cumsum(rr) / 1000.0  # seconds
    if t[-1] - t[0] < min_seconds:
        return 0.0, 0.0, 0.0
    fs_interp = 4.0  # 4 Hz interpolation grid
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs_interp)
    if t_uniform.size < 16:
        return 0.0, 0.0, 0.0
    interp = interp1d(t, rr, kind="cubic", bounds_error=False, fill_value="extrapolate")
    rr_uniform = interp(t_uniform)
    rr_uniform = rr_uniform - rr_uniform.mean()
    nperseg = min(256, rr_uniform.size)
    freqs, psd = signal.welch(rr_uniform, fs=fs_interp, nperseg=nperseg, scaling="density")
    lf_band = (freqs >= 0.04) & (freqs < 0.15)
    hf_band = (freqs >= 0.15) & (freqs < 0.4)
    lf = float(np.trapezoid(psd[lf_band], freqs[lf_band])) if lf_band.any() else 0.0
    hf = float(np.trapezoid(psd[hf_band], freqs[hf_band])) if hf_band.any() else 0.0
    ratio = lf / hf if hf > 0 else 0.0
    return _safe(lf), _safe(hf), _safe(ratio)


def _ecg_features(ecg: np.ndarray, fs: float) -> dict[str, float]:
    ecg = np.asarray(ecg, dtype=float)
    distance = max(1, int(0.3 * fs))
    prominence = max(np.std(ecg) * 0.2, 1e-6)
    peaks, _ = signal.find_peaks(ecg, distance=distance, prominence=prominence)
    rr_ms = np.diff(peaks) / max(fs, 1.0) * 1000.0

    if rr_ms.size:
        hr = 60000.0 / np.mean(rr_ms)
        sdnn = np.std(rr_ms, ddof=1) if rr_ms.size > 1 else 0.0
        rr_diff = np.diff(rr_ms)
        rmssd = np.sqrt(np.mean(rr_diff**2)) if rr_diff.size else 0.0
        pnn50 = float(np.mean(np.abs(rr_diff) > 50.0)) if rr_diff.size else 0.0
        # Poincaré: SD1 = std(RR_diff)/sqrt(2); SD2 = sqrt(2*SDNN^2 - SD1^2)
        sd1 = np.std(rr_diff, ddof=1) / math.sqrt(2.0) if rr_diff.size > 1 else 0.0
        sd2_sq = 2.0 * sdnn**2 - sd1**2
        sd2 = math.sqrt(sd2_sq) if sd2_sq > 0 else 0.0
        sd1_sd2 = sd1 / sd2 if sd2 > 0 else 0.0
    else:
        hr = sdnn = rmssd = pnn50 = sd1 = sd2 = sd1_sd2 = 0.0

    apen = _approximate_entropy(rr_ms) if rr_ms.size >= 6 else 0.0
    dfa_alpha1 = _dfa_alpha1(rr_ms) if rr_ms.size >= 8 else 0.0
    lf, hf, lf_hf = _hrv_frequency(rr_ms) if rr_ms.size >= 6 else (0.0, 0.0, 0.0)

    return {
        "ecg_mean": _safe(np.mean(ecg)),
        "ecg_std": _safe(np.std(ecg)),
        "ecg_min": _safe(np.min(ecg)) if ecg.size else 0.0,
        "ecg_max": _safe(np.max(ecg)) if ecg.size else 0.0,
        "ecg_slope": _slope(ecg, fs),
        "ecg_peak_count": float(peaks.size),
        "ecg_mean_hr_bpm": _safe(hr),
        "ecg_sdnn_ms": _safe(sdnn),
        "ecg_rmssd_ms": _safe(rmssd),
        "ecg_pnn50": _safe(pnn50),
        "ecg_sampen": _sample_entropy(ecg[:: max(1, int(fs // 20))], max_points=96),
        # B2 additions (2026-05-15):
        "ecg_apen": _safe(apen),
        "ecg_sd1": _safe(sd1),
        "ecg_sd2": _safe(sd2),
        "ecg_sd1_sd2_ratio": _safe(sd1_sd2),
        "ecg_dfa_alpha1": _safe(dfa_alpha1),
        "ecg_lf_power": _safe(lf),
        "ecg_hf_power": _safe(hf),
        "ecg_lf_hf_ratio": _safe(lf_hf),
    }


def _eda_features(eda: np.ndarray, fs: float) -> dict[str, float]:
    eda = np.asarray(eda, dtype=float)
    if eda.size == 0:
        return {
            "eda_tonic_mean": 0.0,
            "eda_std": 0.0,
            "eda_slope": 0.0,
            "eda_scr_count": 0.0,
            "eda_scr_amplitude_mean": 0.0,
            "eda_variability": 0.0,
        }

    detrended = signal.detrend(eda)
    distance = max(1, int(fs))
    prominence = max(np.std(detrended) * 0.25, 1e-6)
    peaks, properties = signal.find_peaks(detrended, distance=distance, prominence=prominence)
    prominences = properties.get("prominences", np.array([]))

    return {
        "eda_tonic_mean": _safe(np.mean(eda)),
        "eda_std": _safe(np.std(eda)),
        "eda_min": _safe(np.min(eda)),
        "eda_max": _safe(np.max(eda)),
        "eda_slope": _slope(eda, fs),
        "eda_scr_count": float(peaks.size),
        "eda_scr_amplitude_mean": _safe(np.mean(prominences)) if prominences.size else 0.0,
        "eda_variability": _safe(np.mean(np.abs(np.diff(eda)))) if eda.size > 1 else 0.0,
    }


def extract_features_from_artifact(artifact: dict[str, Any]) -> pd.DataFrame:
    data = np.asarray(artifact["data"])
    labels = np.asarray(artifact["labels"])
    participants = np.asarray(artifact["participants"])
    channels = [str(ch).upper() for ch in artifact.get("channels", [])]
    fs = float(artifact.get("fs") or 250)
    dataset = str(artifact.get("dataset") or "unknown")
    task = str(artifact.get("task") or "unknown")
    signal_set = "ecg_eda" if "EDA" in channels else "ecg"

    if data.ndim != 3:
        raise ValueError(f"Expected data with shape (windows, samples, channels), got {data.shape}")
    if labels.size != data.shape[0] or participants.size != data.shape[0]:
        raise ValueError("labels and participants must have one value per window")
    if "ECG" not in channels:
        raise ValueError("ECG channel is required")

    ecg_idx = channels.index("ECG")
    eda_idx = channels.index("EDA") if "EDA" in channels else None

    rows = []
    for idx in range(data.shape[0]):
        row = {
            "dataset": dataset,
            "task": task,
            "signal_set": signal_set,
            "subject_id": str(participants[idx]),
            "label": int(labels[idx]),
            "window_idx": int(idx),
        }
        row.update(_ecg_features(data[idx, :, ecg_idx], fs))
        if eda_idx is not None:
            row.update(_eda_features(data[idx, :, eda_idx], fs))
        rows.append(row)

    frame = pd.DataFrame(rows)
    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return frame


def load_artifact(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict):
        raise TypeError(f"{path} did not contain a dictionary artifact")
    return artifact


def save_features(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        frame.to_csv(output, index=False)
    else:
        frame.to_parquet(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/features/ecg_eda_features.parquet"))
    parser.add_argument("--output-dir", type=Path, help="Write one parquet feature file per input artifact")
    parser.add_argument("--skip-existing", action="store_true", help="Skip per-input outputs that already exist")
    args = parser.parse_args()
    input_paths = expand_inputs(args.inputs)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path in input_paths:
            output = args.output_dir / f"{path.stem}.parquet"
            if args.skip_existing and output.exists():
                print(f"Skipping existing feature file: {output}")
                continue
            frame = extract_features_from_artifact(load_artifact(path))
            save_features(frame, output)
            print(f"Saved {len(frame)} feature rows to {output}")
        return

    frames = [extract_features_from_artifact(load_artifact(path)) for path in input_paths]
    combined = pd.concat(frames, ignore_index=True)
    save_features(combined, args.output)
    print(f"Saved {len(combined)} feature rows to {args.output}")


if __name__ == "__main__":
    main()
