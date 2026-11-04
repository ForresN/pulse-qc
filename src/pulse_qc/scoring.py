"""Score detected beats against reference annotations (the metric code).

The user-facing entry point is :func:`pulse_qc.evaluate`, which runs a whole config.

The reference beats in the PTT-PPG set are ECG R-peaks, and a PPG systolic
peak arrives a pulse-arrival time later (roughly 150–400 ms at the finger). So
before matching, each record's detections are shifted by a single per-record
lag: the median offset from each reference beat to the nearest detection in a
search range. That lag is reported. Matching is then greedy one-to-one within a
tolerance window.

Per-beat outcomes are attributed to the window they fall in, so the same
detections can be scored over all windows, over QC-passing windows only, or
split by activity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pulse_qc.windows import window_index_of


@dataclass(frozen=True)
class Matching:
    lag_s: float
    tp_ref: np.ndarray  # indices into reference beats that were matched
    tp_det: np.ndarray  # indices into detections that were matched
    fn_ref: np.ndarray  # reference indices with no match
    fp_det: np.ndarray  # detection indices with no match
    errors_s: np.ndarray  # signed timing error (det - ref - lag) for matched pairs


def estimate_lag(
    ref: np.ndarray, det: np.ndarray, fs: float, search_s: tuple[float, float]
) -> float:
    """Median delay from reference beats to the nearest following detection within the
    search range. Returns 0.0 if nothing is found (then no shift is applied)."""
    if len(ref) == 0 or len(det) == 0:
        return 0.0
    lo, hi = int(search_s[0] * fs), int(search_s[1] * fs)
    pos = np.searchsorted(det, ref + lo, side="left")
    pos = np.clip(pos, 0, len(det) - 1)
    delta = det[pos] - ref
    valid = (delta >= lo) & (delta <= hi)
    if valid.sum() < 3:
        return 0.0
    return float(np.median(delta[valid]) / fs)


def match_beats(
    ref: np.ndarray,
    det: np.ndarray,
    fs: float,
    tolerance_s: float,
    lag_s: float | None = None,
    lag_search_s: tuple[float, float] = (0.0, 0.6),
) -> Matching:
    """Greedy one-to-one matching of ``det`` to ``ref`` after shifting ``det`` by ``lag_s``.

    Pairs are formed in order of increasing |error|, each beat used at most once."""
    ref = np.asarray(ref, dtype=np.int64)
    det = np.asarray(det, dtype=np.int64)
    if lag_s is None:
        lag_s = estimate_lag(ref, det, fs, lag_search_s)
    shifted = det - int(round(lag_s * fs))
    tol = int(round(tolerance_s * fs))

    candidates: list[tuple[int, int, int]] = []  # (|err|, ref_idx, det_idx)
    for j, d in enumerate(shifted):
        k = np.searchsorted(ref, d)
        for i in (k - 1, k):
            if 0 <= i < len(ref):
                err = int(d - ref[i])
                if abs(err) <= tol:
                    candidates.append((abs(err), i, j))
    candidates.sort()
    used_ref: set[int] = set()
    used_det: set[int] = set()
    tp_ref, tp_det, errors = [], [], []
    for _, i, j in candidates:
        if i in used_ref or j in used_det:
            continue
        used_ref.add(i)
        used_det.add(j)
        tp_ref.append(i)
        tp_det.append(j)
        errors.append((shifted[j] - ref[i]) / fs)
    order = np.argsort(tp_ref) if tp_ref else np.empty(0, dtype=np.int64)
    tp_ref_a = np.asarray(tp_ref, dtype=np.int64)[order]
    tp_det_a = np.asarray(tp_det, dtype=np.int64)[order]
    errors = [errors[k] for k in order]
    fn_ref = np.setdiff1d(np.arange(len(ref)), tp_ref_a)
    fp_det = np.setdiff1d(np.arange(len(det)), tp_det_a)
    return Matching(lag_s, tp_ref_a, tp_det_a, fn_ref, fp_det, np.asarray(errors, dtype=float))


def per_window_counts(
    m: Matching, ref: np.ndarray, det: np.ndarray, fs: float, windows: np.ndarray
) -> pd.DataFrame:
    """TP/FN/FP counts per window. A beat is attributed to the window containing the
    reference beat (for TP/FN) or the lag-shifted detection (for FP). Window -1 holds
    beats outside every window (the dropped partial tail)."""
    shifted = det - int(round(m.lag_s * fs))
    tp_w = window_index_of(ref[m.tp_ref], windows)
    fn_w = window_index_of(ref[m.fn_ref], windows)
    fp_w = window_index_of(shifted[m.fp_det], windows)
    n = len(windows)
    idx = np.arange(-1, n)
    df = pd.DataFrame({"window": idx})
    df["tp"] = np.bincount(tp_w + 1, minlength=n + 1)
    df["fn"] = np.bincount(fn_w + 1, minlength=n + 1)
    df["fp"] = np.bincount(fp_w + 1, minlength=n + 1)
    return df


def scores(tp: int, fn: int, fp: int) -> dict[str, float]:
    sens = tp / (tp + fn) if tp + fn else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fn + fp) if 2 * tp + fn + fp else float("nan")
    return {"sensitivity": sens, "ppv": ppv, "f1": f1}


def summarise(counts: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Aggregate TP/FN/FP over ``by`` columns and attach sensitivity/PPV/F1."""
    g = counts.groupby(by, dropna=False)[["tp", "fn", "fp"]].sum().reset_index()
    sc = g.apply(lambda r: pd.Series(scores(int(r.tp), int(r.fn), int(r.fp))), axis=1)
    out = pd.concat([g, sc], axis=1)
    out["n_ref"] = out["tp"] + out["fn"]
    return out
