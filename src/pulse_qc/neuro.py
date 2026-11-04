"""Neuro-ICU waveform features: per-beat pulse amplitude and a naive moving ICP–ABP
correlation, with QC-failed windows masked.

**Non-clinical.** The moving correlation here is a *PRx-style* index computed the
textbook way (Pearson correlation of consecutive 10 s means of ICP and ABP over a
5-minute moving window). It is not the validated PRx used in clinical research
(which is computed on specific monitors with specific artefact handling and on
mean arterial pressure), and nothing here has been checked against any clinical
reference. It exists to show the engineering pattern: a slow index derived from
two waveforms, with the quality gate applied to both before anything is averaged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pulse_qc.cache import Cache, code_hash, config_hash, make_key

NEURO_VERSION = 1


def pulse_amplitude(
    signal: np.ndarray, fs: float, peaks: np.ndarray, max_rr_s: float = 2.0
) -> pd.DataFrame:
    """Per-beat pulse amplitude: peak value minus the minimum in the preceding inter-beat
    interval (capped at ``max_rr_s``). Returns ``t_s`` (peak time), ``peak``, ``trough``,
    ``amplitude`` in the signal's own units. The first peak has no preceding interval and
    is skipped; beats whose interval contains NaN get NaN amplitude."""
    rows = []
    peaks = np.asarray(peaks, dtype=np.int64)
    for i in range(1, len(peaks)):
        p = peaks[i]
        start = max(peaks[i - 1], p - int(max_rr_s * fs))
        seg = signal[start:p]
        if seg.size == 0 or not np.isfinite(seg).all() or not np.isfinite(signal[p]):
            amp, trough = np.nan, np.nan
        else:
            trough = float(seg.min())
            amp = float(signal[p] - trough)
        rows.append({"t_s": p / fs, "peak": float(signal[p]), "trough": trough, "amplitude": amp})
    return pd.DataFrame(rows, columns=["t_s", "peak", "trough", "amplitude"])


def masked_means(
    signal: np.ndarray, fs: float, windows: np.ndarray, passed: np.ndarray
) -> np.ndarray:
    """Mean of each window; NaN where the window failed QC or contains any NaN."""
    out = np.full(len(windows), np.nan)
    for i, (a, b) in enumerate(windows):
        if passed[i]:
            seg = signal[a:b]
            if np.isfinite(seg).all():
                out[i] = seg.mean()
    return out


def moving_correlation(
    x_means: np.ndarray,
    y_means: np.ndarray,
    corr_n: int = 30,
    step_n: int = 6,
    min_valid_frac: float = 0.8,
) -> pd.DataFrame:
    """Pearson correlation of ``x_means`` and ``y_means`` over a moving block of ``corr_n``
    consecutive means, advanced by ``step_n``. A block needs at least ``min_valid_frac`` of
    its pairs valid (both non-NaN) or its correlation is NaN. Returns one row per block with
    the index of its first mean, ``n_valid`` and ``r``."""
    rows = []
    n = len(x_means)
    for start in range(0, max(n - corr_n + 1, 0), step_n):
        xs = x_means[start : start + corr_n]
        ys = y_means[start : start + corr_n]
        ok = np.isfinite(xs) & np.isfinite(ys)
        r = np.nan
        if ok.sum() >= max(3, int(np.ceil(min_valid_frac * corr_n))):
            xv, yv = xs[ok], ys[ok]
            if xv.std() > 0 and yv.std() > 0:
                r = float(np.corrcoef(xv, yv)[0, 1])
        rows.append({"block_start": start, "n_valid": int(ok.sum()), "r": r})
    return pd.DataFrame(rows, columns=["block_start", "n_valid", "r"])


def pair_features(
    icp: np.ndarray,
    abp: np.ndarray,
    fs: float,
    windows: np.ndarray,
    icp_passed: np.ndarray,
    abp_passed: np.ndarray,
    icp_peaks: np.ndarray,
    abp_peaks: np.ndarray,
    corr_n: int = 30,
    step_n: int = 6,
) -> dict[str, pd.DataFrame]:
    """All derived tables for one ICP/ABP pair sharing one window grid."""
    both = np.asarray(icp_passed, bool) & np.asarray(abp_passed, bool)
    icp_m = masked_means(icp, fs, windows, both)
    abp_m = masked_means(abp, fs, windows, both)
    means = pd.DataFrame(
        {
            "t_s": windows[:, 0] / fs,
            "icp_mean": icp_m,
            "abp_mean": abp_m,
            "icp_passed": np.asarray(icp_passed, bool),
            "abp_passed": np.asarray(abp_passed, bool),
        }
    )
    corr = moving_correlation(icp_m, abp_m, corr_n=corr_n, step_n=step_n)
    corr["t_s"] = windows[corr["block_start"].values, 0] / fs if len(corr) else []
    corr["t_end_s"] = (
        windows[np.minimum(corr["block_start"].values + corr_n - 1, len(windows) - 1), 1] / fs
        if len(corr)
        else []
    )
    icp_amp = pulse_amplitude(icp, fs, icp_peaks)
    abp_amp = pulse_amplitude(abp, fs, abp_peaks)
    # mark each beat with whether its window passed (both channels)
    from pulse_qc.windows import window_index_of

    for amp in (icp_amp, abp_amp):
        w = window_index_of((amp["t_s"].values * fs).astype(np.int64), windows)
        amp["window"] = w
        amp["passed"] = np.where(w >= 0, both[np.clip(w, 0, len(both) - 1)], False)
    return {"means": means, "corr": corr, "icp_amplitude": icp_amp, "abp_amplitude": abp_amp}


def summarise_pair(feat: dict[str, pd.DataFrame]) -> dict[str, Any]:
    means, corr = feat["means"], feat["corr"]
    icp_ok = feat["icp_amplitude"][feat["icp_amplitude"]["passed"]]
    abp_ok = feat["abp_amplitude"][feat["abp_amplitude"]["passed"]]
    return {
        "n_windows": int(len(means)),
        "windows_both_passed": int((means["icp_passed"] & means["abp_passed"]).sum()),
        "icp_mean_mmHg_median": float(np.nanmedian(means["icp_mean"]))
        if means["icp_mean"].notna().any()
        else np.nan,
        "abp_mean_mmHg_median": float(np.nanmedian(means["abp_mean"]))
        if means["abp_mean"].notna().any()
        else np.nan,
        "icp_pulse_amp_median": float(icp_ok["amplitude"].median()) if len(icp_ok) else np.nan,
        "abp_pulse_amp_median": float(abp_ok["amplitude"].median()) if len(abp_ok) else np.nan,
        "n_corr_blocks": int(len(corr)),
        "corr_blocks_valid": int(corr["r"].notna().sum()),
        "corr_median": float(corr["r"].median()) if corr["r"].notna().any() else np.nan,
        "corr_frac_above_0.3": float((corr["r"] > 0.3).mean())
        if corr["r"].notna().any()
        else np.nan,
    }


def run_stage(
    cfg: dict[str, Any], loaded: dict[str, dict[str, Any]], cache: Cache
) -> dict[str, Any]:
    """Pair ICP and ABP records by WFDB record name and compute features (cached)."""
    ncfg = cfg["neuro"]
    icp_ch, abp_ch = ncfg.get("icp_channel", "ICP"), ncfg.get("abp_channel", "ABP")
    by_name: dict[tuple[str, str], dict[str, Any]] = {}
    for item in loaded.values():
        spec = item["spec"]
        by_name.setdefault((spec.db, spec.name), {})[spec.channel] = item
    out: dict[str, Any] = {"pairs": {}, "summary": []}
    for (db, name), chans in by_name.items():
        if icp_ch not in chans or abp_ch not in chans:
            continue
        a, b = chans[icp_ch], chans[abp_ch]
        if a["rec"].fs != b["rec"].fs or len(a["windows"]) != len(b["windows"]):
            raise ValueError(f"{db}/{name}: ICP and ABP must share fs and window grid")
        det = ncfg.get("detector", cfg["detectors"][0]["name"])
        key = make_key(
            "neuro",
            icp=a["load_key"],
            abp=b["load_key"],
            icp_qc=a["qc_key"],
            abp_qc=b["qc_key"],
            config=config_hash({**ncfg, "detector": det}),
            version=str(NEURO_VERSION),
            code=code_hash(pair_features, pulse_amplitude, masked_means, moving_correlation),
        )
        feat = cache.get_or_compute(
            "neuro",
            key,
            lambda a=a, b=b, det=det: pair_features(
                a["rec"].signal,
                b["rec"].signal,
                a["rec"].fs,
                a["windows"],
                a["qc_df"]["passed"].values,
                b["qc_df"]["passed"].values,
                a["detections"][det],
                b["detections"][det],
                corr_n=int(ncfg.get("corr_n", 30)),
                step_n=int(ncfg.get("step_n", 6)),
            ),
        )
        pair_id = f"{db}/{name}"
        out["pairs"][pair_id] = {
            **feat,
            "fs": a["rec"].fs,
            "icp": a["rec"].signal,
            "abp": b["rec"].signal,
            "windows": a["windows"],
            "detector": det,
        }
        out["summary"].append({"record": pair_id, "detector": det, **summarise_pair(feat)})
    return out


def save(neuro: dict[str, Any], run_dir: Path) -> None:
    d = Path(run_dir) / "neuro"
    d.mkdir(exist_ok=True)
    pd.DataFrame(neuro["summary"]).to_csv(d / "summary.csv", index=False)
    for pair_id, feat in neuro["pairs"].items():
        tag = pair_id.replace("/", "__")
        feat["means"].to_csv(d / f"{tag}_means.csv", index=False)
        feat["corr"].to_csv(d / f"{tag}_corr.csv", index=False)
        feat["icp_amplitude"].to_csv(d / f"{tag}_icp_amplitude.csv", index=False)
        feat["abp_amplitude"].to_csv(d / f"{tag}_abp_amplitude.csv", index=False)
        np.savez_compressed(
            d / f"{tag}_signals.npz",
            icp=feat["icp"].astype(np.float32),
            abp=feat["abp"].astype(np.float32),
            fs=feat["fs"],
            windows=feat["windows"],
        )


def load(run_dir: Path) -> dict[str, Any] | None:
    d = Path(run_dir) / "neuro"
    if not (d / "summary.csv").exists():
        return None
    summary = pd.read_csv(d / "summary.csv")
    pairs: dict[str, Any] = {}
    for tag_file in sorted(d.glob("*_means.csv")):
        tag = tag_file.name[: -len("_means.csv")]
        sig = np.load(d / f"{tag}_signals.npz")
        pairs[tag.replace("__", "/")] = {
            "means": pd.read_csv(tag_file),
            "corr": pd.read_csv(d / f"{tag}_corr.csv"),
            "icp_amplitude": pd.read_csv(d / f"{tag}_icp_amplitude.csv"),
            "abp_amplitude": pd.read_csv(d / f"{tag}_abp_amplitude.csv"),
            "icp": sig["icp"],
            "abp": sig["abp"],
            "fs": float(sig["fs"]),
            "windows": sig["windows"],
        }
    return {"summary": summary, "pairs": pairs}


def figure(pair_id: str, feat: dict[str, Any], path: Path) -> None:
    """Four panels: ICP, ABP (QC-failed windows shaded), ICP pulse amplitude per beat, and the
    moving ICP–ABP correlation. Signals are drawn as 1 s min/max envelopes so that a 2-hour
    segment stays readable."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = feat["fs"]
    means, corr = feat["means"], feat["corr"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)

    def envelope(ax, sig, label, color):
        n = int(fs)
        m = len(sig) // n
        s = sig[: m * n].reshape(m, n)
        with np.errstate(all="ignore"):
            lo, hi = np.nanmin(s, axis=1), np.nanmax(s, axis=1)
        t = np.arange(m) + 0.5
        ax.fill_between(t, lo, hi, color=color, lw=0, alpha=0.8, label=label)
        ax.set_ylabel(f"{label} (mmHg)")

    envelope(axes[0], feat["icp"], "ICP", "#2f6f9f")
    envelope(axes[1], feat["abp"], "ABP", "#8e44ad")
    for ax, col in ((axes[0], "icp_passed"), (axes[1], "abp_passed")):
        for _, row in means[~means[col]].iterrows():
            ax.axvspan(
                row["t_s"],
                row["t_s"] + (feat["windows"][0, 1] - feat["windows"][0, 0]) / fs,
                color="#c0392b",
                alpha=0.15,
                lw=0,
            )
    amp = feat["icp_amplitude"]
    ok = amp[amp["passed"]]
    bad = amp[~amp["passed"]]
    axes[2].plot(ok["t_s"], ok["amplitude"], ".", ms=2, color="#2f6f9f", label="QC-passed beats")
    axes[2].plot(
        bad["t_s"], bad["amplitude"], ".", ms=2, color="#c0392b", alpha=0.5, label="QC-failed beats"
    )
    axes[2].set_ylabel("ICP pulse amp (mmHg)")
    axes[2].set_ylim(
        0,
        max(1.0, float(np.nanpercentile(amp["amplitude"], 99)) * 1.2)
        if amp["amplitude"].notna().any()
        else 1,
    )
    axes[2].legend(loc="upper right", fontsize=7)
    mid = (corr["t_s"] + corr["t_end_s"]) / 2
    axes[3].plot(
        mid,
        corr["r"],
        "-o",
        ms=3,
        color="#333",
        label="moving ICP–ABP correlation (PRx-style, non-clinical)",
    )
    axes[3].axhline(0, color="#999", lw=0.5)
    axes[3].axhline(0.3, color="#c0392b", lw=0.5, ls="--")
    axes[3].set_ylim(-1, 1)
    axes[3].set_ylabel("r")
    axes[3].set_xlabel("time in segment (s)")
    axes[3].legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"{pair_id}: 1 s min/max envelopes; red = QC-failed window; "
        "correlation blocks need both channels to pass",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
