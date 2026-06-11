import numpy as np
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_windows(signal, fs, win_sec, overlap=0.0):
    """Create non-overlapping or overlapping windows from a 1D signal.

    Args:
        signal: 1D numpy array.
        fs: Sampling rate in Hz.
        win_sec: Window length in seconds.
        overlap: Overlap fraction between 0.0 and 1.0.

    Returns:
        2D numpy array of shape (n_windows, win_samples).
    """
    win_samples = int(fs * win_sec)
    step = max(1, int(win_samples * (1 - overlap)))
    n_windows = max(0, (len(signal) - win_samples) // step + 1)
    if n_windows == 0:
        return np.empty((0, win_samples))
    indices = np.arange(n_windows)[:, None] * step + np.arange(win_samples)
    return signal[indices]
