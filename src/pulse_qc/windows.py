"""Fixed-length windowing of a uniformly sampled signal."""

from __future__ import annotations

import numpy as np


def make_windows(
    n_samples: int, fs: float, length_s: float, step_s: float | None = None
) -> np.ndarray:
    """Return an ``(n_windows, 2)`` array of ``[start, stop)`` sample indices.

    The final partial window is dropped: QC statistics on a short tail are not
    comparable with the rest and the evaluation treats uncovered beats as
    "unwindowed", which is reported rather than hidden.
    """
    if length_s <= 0:
        raise ValueError("window length must be positive")
    step_s = length_s if step_s is None else step_s
    if step_s <= 0:
        raise ValueError("window step must be positive")
    length = int(round(length_s * fs))
    step = int(round(step_s * fs))
    starts = np.arange(0, n_samples - length + 1, step, dtype=np.int64)
    return (
        np.stack([starts, starts + length], axis=1) if len(starts) else np.empty((0, 2), np.int64)
    )


def window_index_of(samples: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """For each sample index, the index of the (non-overlapping) window that contains it,
    or -1 if none does. Windows must be sorted and non-overlapping."""
    if len(windows) == 0:
        return np.full(len(samples), -1, dtype=np.int64)
    starts = windows[:, 0]
    idx = np.searchsorted(starts, samples, side="right") - 1
    idx = np.clip(idx, 0, len(windows) - 1)
    inside = (samples >= windows[idx, 0]) & (samples < windows[idx, 1])
    return np.where(inside, idx, -1)
