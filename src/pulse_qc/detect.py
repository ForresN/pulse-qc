"""Beat detectors, all run on the same preprocessed signal.

Three detectors are registered:

* ``elgendi`` – NeuroKit2's implementation of Elgendi et al. (2013) systolic
  peak detection (two moving averages over the squared, bandpassed signal).
* ``msptd`` – NeuroKit2's implementation of Charlton et al. (2022), the
  multi-scale peak and trough detection (MSPTD) algorithm.
* ``prominence`` – a deliberately simple baseline: ``scipy.signal.find_peaks``
  with a minimum inter-beat distance and a prominence threshold relative to
  the window's spread. It exists so the published methods can be compared with
  something that has no tuned internals.

Preprocessing is shared and configurable: optional polarity inversion (raw
reflectance PPG dips at systole), NaN interpolation, and a zero-phase Butterworth
band-pass. Bump :data:`DETECT_VERSION` if a detector's behaviour changes; cached
detections are keyed on it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import signal as ss

DETECT_VERSION = 1


def preprocess(
    x: np.ndarray,
    fs: float,
    *,
    invert: bool = False,
    band_hz: tuple[float, float] = (0.5, 8.0),
    order: int = 3,
) -> np.ndarray:
    """Interpolate NaNs linearly, band-pass, optionally flip polarity."""
    y = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(y)
    if bad.all():
        return np.zeros_like(y)
    if bad.any():
        idx = np.arange(len(y))
        y[bad] = np.interp(idx[bad], idx[~bad], y[~bad])
    y = y - np.mean(y)
    nyq = fs / 2
    hi = min(band_hz[1], 0.95 * nyq)
    sos = ss.butter(order, [band_hz[0], hi], btype="band", fs=fs, output="sos")
    y = ss.sosfiltfilt(sos, y)
    return -y if invert else y


def detect_elgendi(y: np.ndarray, fs: float, **_: Any) -> np.ndarray:
    import neurokit2 as nk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = nk.ppg_findpeaks(y, sampling_rate=int(fs), method="elgendi")
    return np.asarray(info["PPG_Peaks"], dtype=np.int64)


def detect_msptd(y: np.ndarray, fs: float, **_: Any) -> np.ndarray:
    import neurokit2 as nk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = nk.ppg_findpeaks(y, sampling_rate=int(fs), method="charlton")
    return np.asarray(info["PPG_Peaks"], dtype=np.int64)


def detect_prominence(
    y: np.ndarray, fs: float, *, min_rr_s: float = 0.3, prominence_sd: float = 0.5, **_: Any
) -> np.ndarray:
    sd = float(np.std(y))
    if sd == 0:
        return np.empty(0, dtype=np.int64)
    peaks, _ = ss.find_peaks(y, distance=max(1, int(min_rr_s * fs)), prominence=prominence_sd * sd)
    return np.asarray(peaks, dtype=np.int64)


DETECTORS: dict[str, Callable[..., np.ndarray]] = {
    "elgendi": detect_elgendi,
    "msptd": detect_msptd,
    "prominence": detect_prominence,
}


def run_detector(
    name: str, y: np.ndarray, fs: float, params: dict[str, Any] | None = None
) -> np.ndarray:
    if name not in DETECTORS:
        raise KeyError(f"unknown detector {name!r}; known: {sorted(DETECTORS)}")
    peaks = DETECTORS[name](y, fs, **(params or {}))
    return np.unique(peaks)
