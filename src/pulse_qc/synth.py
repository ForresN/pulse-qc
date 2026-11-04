"""Synthetic pulsatile signals with known beat times, for tests and CI.

Nothing here is a physiological model. Each beat is a fixed-shape pulse (a
sharp systolic rise and slower decay) placed at a known time; the generator
exists so that QC rules and detectors can be tested without any download.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Synthetic:
    fs: float
    signal: np.ndarray
    beat_samples: np.ndarray  # ground-truth "systolic peak" sample indices


def synthetic_ppg(
    duration_s: float = 60.0,
    fs: float = 125.0,
    hr_bpm: float = 70.0,
    hr_jitter: float = 0.03,
    noise_sd: float = 0.02,
    baseline: float = 1.0,
    seed: int = 0,
) -> Synthetic:
    """Return a pulse train with a known peak at each beat.

    Beats are spaced at ``60/hr_bpm`` seconds with multiplicative jitter. Each
    pulse is an asymmetric Gaussian: fast upstroke, slower downstroke, with a
    small dicrotic bump. Amplitude is 1 around ``baseline``.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    period = 60.0 / hr_bpm
    beat_times = []
    tb = 0.3
    while tb < duration_s - 0.3:
        beat_times.append(tb)
        tb += period * (1 + rng.normal(0, hr_jitter))
    beat_times = np.asarray(beat_times)

    sig = np.zeros(n)
    for b in beat_times:
        d = t - b
        up = np.exp(-(d**2) / (2 * 0.04**2)) * (d <= 0)
        down = np.exp(-(d**2) / (2 * 0.12**2)) * (d > 0)
        dicrotic = 0.25 * np.exp(-((d - 0.3) ** 2) / (2 * 0.05**2))
        sig += up + down + dicrotic
    sig = baseline + sig + rng.normal(0, noise_sd, n)
    beat_samples = np.round(beat_times * fs).astype(np.int64)
    return Synthetic(fs=fs, signal=sig, beat_samples=beat_samples)


def inject_nan(signal: np.ndarray, start: int, stop: int) -> np.ndarray:
    out = signal.copy()
    out[start:stop] = np.nan
    return out


def inject_flatline(
    signal: np.ndarray, start: int, stop: int, value: float | None = None
) -> np.ndarray:
    out = signal.copy()
    out[start:stop] = signal[start] if value is None else value
    return out


def inject_clipping(signal: np.ndarray, ceiling: float) -> np.ndarray:
    return np.minimum(signal, ceiling)


def inject_spike(signal: np.ndarray, at: int, value: float) -> np.ndarray:
    out = signal.copy()
    out[at] = value
    return out
