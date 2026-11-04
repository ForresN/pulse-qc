"""Add a detector and a rule without touching pulse-qc internals, then compare.

Run from the repo root (no download; uses the synthetic fixtures):

    uv run python examples/custom_detector.py

Or point it at real data once configs/ptt-ppg.yaml has been run:

    uv run python examples/custom_detector.py configs/ptt-ppg.yaml
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import signal as ss

import pulse_qc


@pulse_qc.register_detector("derivative_zero_cross")
def derivative_zero_cross(y: np.ndarray, fs: float, *, min_rr_s: float = 0.3) -> np.ndarray:
    """Peaks = downward zero-crossings of the smoothed first derivative."""
    d = np.gradient(ss.savgol_filter(y, max(5, int(0.05 * fs) | 1), 2))
    cross = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0))
    # keep the largest candidate in every refractory period
    keep, last = [], -np.inf
    for c in cross[np.argsort(-y[cross])]:
        if abs(c - last) >= min_rr_s * fs and all(abs(c - k) >= min_rr_s * fs for k in keep):
            keep.append(c)
    return np.sort(np.asarray(keep, dtype=np.int64))


@pulse_qc.register_rule("amplitude_floor", threshold=("min_p2p", "below"))
def amplitude_floor(x: np.ndarray, fs: float, *, min_p2p: float = 0.0) -> pulse_qc.RuleResult:
    """Fail when the window's peak-to-peak range is below ``min_p2p`` (signal units)."""
    finite = x[np.isfinite(x)]
    p2p = float(finite.max() - finite.min()) if finite.size else 0.0
    ok = p2p >= min_p2p
    return pulse_qc.RuleResult(ok, p2p, "" if ok else f"peak-to-peak {p2p:.3g} < {min_p2p:.3g}")


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/synthetic.yaml"
    result = pulse_qc.evaluate(
        config,
        name="custom-detector-demo",
        detectors=["elgendi", "derivative_zero_cross"],
        report=False,
    )
    print(result.gating_table().to_string(index=False))
    print("run dir:", result.run_dir)
    print("git:", result.provenance["git"])
