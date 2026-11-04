"""Declarative, versioned data-quality rules.

Each rule is a pure function of one window of samples and the sampling rate.
It returns a :class:`RuleResult` with a pass/fail flag, the measured value the
decision was based on, and a human-readable reason when it fails. Rules are
registered by name in :data:`RULES` so a YAML config can pick and parametrise
them without any code change. Bump :data:`RULES_VERSION` whenever the semantics
of an existing rule change, because cached QC results are keyed on it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

RULES_VERSION = 1


@dataclass(frozen=True)
class RuleResult:
    passed: bool
    value: float
    reason: str = ""


RuleFn = Callable[..., RuleResult]


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of ``True`` in a boolean array."""
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return int((edges[1::2] - edges[::2]).max())


def nan_dropout(x: np.ndarray, fs: float, *, max_frac: float = 0.01) -> RuleResult:
    """Fail when more than ``max_frac`` of samples are missing (NaN)."""
    frac = float(np.isnan(x).mean()) if x.size else 1.0
    ok = frac <= max_frac
    return RuleResult(ok, frac, "" if ok else f"{frac:.1%} of samples are NaN (max {max_frac:.1%})")


def sampling_gap(x: np.ndarray, fs: float, *, max_gap_s: float = 0.2) -> RuleResult:
    """Fail when any single contiguous run of missing samples is longer than ``max_gap_s``.

    WFDB stores uniform sampling, so a gap in acquisition shows up as a run of
    NaN. This differs from :func:`nan_dropout`: a few scattered missing samples
    are tolerable, one long hole breaks beat timing."""
    run_s = _longest_run(np.isnan(x)) / fs
    ok = run_s <= max_gap_s
    return RuleResult(ok, run_s, "" if ok else f"gap of {run_s:.2f}s (max {max_gap_s:.2f}s)")


def flatline(x: np.ndarray, fs: float, *, max_flat_s: float = 1.0, eps: float = 0.0) -> RuleResult:
    """Fail when the signal is constant (|diff| <= eps) for longer than ``max_flat_s``."""
    d = np.abs(np.diff(x))
    flat = d <= eps  # NaN diffs compare False, so NaN runs are not flat (they are gaps)
    run_s = (_longest_run(flat) + 1) / fs if flat.any() else 0.0
    ok = run_s <= max_flat_s
    return RuleResult(ok, run_s, "" if ok else f"flat for {run_s:.2f}s (max {max_flat_s:.2f}s)")


def clipping(
    x: np.ndarray,
    fs: float,
    *,
    max_frac: float = 0.02,
    min_run: int = 3,
    rel_tol: float = 0.001,
    lo: float | None = None,
    hi: float | None = None,
) -> RuleResult:
    """Fail when more than ``max_frac`` of samples sit at a saturation rail.

    A rail is ``lo``/``hi`` if given (e.g. ADC limits), otherwise the window's own
    min/max. A sample counts as clipped only if it is within ``rel_tol`` of the
    window's range from a rail *and* part of a run of at least ``min_run`` such
    samples, so that a lone true peak touching the max is not called clipping."""
    finite = x[np.isfinite(x)]
    if finite.size < 2:
        return RuleResult(True, 0.0)
    rng = float(finite.max() - finite.min())
    if rng == 0:
        return RuleResult(True, 0.0)  # flatline rule reports this case
    tol = rel_tol * rng
    lo_rail = finite.min() if lo is None else lo
    hi_rail = finite.max() if hi is None else hi
    at_rail = (x >= hi_rail - tol) | (x <= lo_rail + tol)
    at_rail &= np.isfinite(x)
    # keep only runs >= min_run
    padded = np.concatenate([[False], at_rail, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    clipped = 0
    for s, e in zip(edges[::2], edges[1::2], strict=True):
        if e - s >= min_run:
            clipped += e - s
    frac = clipped / x.size
    ok = frac <= max_frac
    return RuleResult(ok, frac, "" if ok else f"{frac:.1%} of samples clipped (max {max_frac:.1%})")


def plausible_range(
    x: np.ndarray, fs: float, *, lo: float, hi: float, max_frac: float = 0.0
) -> RuleResult:
    """Fail when more than ``max_frac`` of finite samples fall outside ``[lo, hi]``."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return RuleResult(True, 0.0)
    frac = float(((finite < lo) | (finite > hi)).mean())
    ok = frac <= max_frac
    return RuleResult(ok, frac, "" if ok else f"{frac:.1%} of samples outside [{lo}, {hi}]")


RULES: dict[str, RuleFn] = {
    "nan_dropout": nan_dropout,
    "sampling_gap": sampling_gap,
    "flatline": flatline,
    "clipping": clipping,
    "plausible_range": plausible_range,
}
