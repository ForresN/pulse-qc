"""Markdown report, figures and comparison for saved runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_run(run_dir: Path) -> dict[str, Any]:
    d = Path(run_dir)
    out: dict[str, Any] = {
        "dir": d,
        "config": yaml.safe_load((d / "config.yaml").read_text()),
        "window_qc": pd.read_csv(d / "window_qc.csv"),
        "beat_counts": pd.read_csv(d / "beat_counts.csv"),
        "lags": pd.read_csv(d / "lags.csv"),
        "summary": pd.read_csv(d / "summary.csv"),
        "provenance": json.loads((d / "provenance.json").read_text()),
        "cache_stages": json.loads((d / "cache_stages.json").read_text()),
    }
    if (d / "example.npz").exists():
        out["example"] = dict(np.load(d / "example.npz"))
        out["example"].update(json.loads((d / "example.json").read_text()))
    return out


def _fmt(v: float) -> str:
    return "–" if pd.isna(v) else f"{v:.3f}"


def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict[str, Any] | None = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(
                fmt[c](v) if c in fmt else (str(v) if not isinstance(v, float) else _fmt(v))
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def gating_table(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per detector x activity: F1 over all windows, over passing windows, over
    failing windows, and the difference."""
    if summary.empty:
        return pd.DataFrame()
    p = summary.pivot_table(
        index=["detector", "activity"],
        columns="gating",
        values=["f1", "sensitivity", "ppv", "n_ref"],
        aggfunc="first",
    )
    out = pd.DataFrame(
        {
            "detector": [i[0] for i in p.index],
            "activity": [i[1] for i in p.index],
            "beats (all)": p[("n_ref", "all")].astype(int).values,
            "beats (passed)": p[("n_ref", "passed")].fillna(0).astype(int).values
            if ("n_ref", "passed") in p
            else 0,
            "sens (all)": p[("sensitivity", "all")].values,
            "PPV (all)": p[("ppv", "all")].values,
            "F1 (all)": p[("f1", "all")].values,
            "F1 (passed)": p[("f1", "passed")].values if ("f1", "passed") in p else np.nan,
            "F1 (failed)": p[("f1", "failed")].values if ("f1", "failed") in p else np.nan,
        }
    )
    out["ΔF1 (passed − all)"] = out["F1 (passed)"] - out["F1 (all)"]
    order = {"all": 0, "sit": 1, "walk": 2, "run": 3}
    out["_o"] = out["activity"].map(lambda a: order.get(a, 9))
    return out.sort_values(["detector", "_o"]).drop(columns="_o").reset_index(drop=True)


def qc_reasons(window_qc: pd.DataFrame) -> pd.DataFrame:
    failed = window_qc[~window_qc["passed"].astype(bool)]
    rows: dict[str, int] = {}
    for reasons in failed["reasons"].fillna(""):
        for part in str(reasons).split("; "):
            if part:
                rule = part.split(":")[0]
                rows[rule] = rows.get(rule, 0) + 1
    return pd.DataFrame({"rule": list(rows), "windows failed": list(rows.values())}).sort_values(
        "windows failed", ascending=False
    )


def write_report(run_dir: Path, figures: bool = True) -> Path:
    r = load_run(run_dir)
    cfg, prov = r["config"], r["provenance"]
    wq = r["window_qc"]
    md: list[str] = [f"# pulse-qc run: `{cfg['name']}`", ""]
    md.append(
        f"Config hash `{prov['config_hash'][:12]}`, git `{(prov['git'].get('sha') or 'n/a')[:10]}`"
        f"{' (dirty)' if prov['git'].get('dirty') else ''}, generated {prov['generated_at']}."
    )
    md.append("")
    md.append("## Records")
    md.append("")
    recs = pd.DataFrame(prov["records"])
    md.append(
        _md_table(
            recs,
            [
                "record",
                "activity",
                "fs",
                "duration_s",
                "n_annotated_beats",
                "n_windows",
                "n_windows_passed",
            ],
            {
                "duration_s": lambda v: f"{v:.0f}",
                "fs": lambda v: f"{v:g}",
                "n_annotated_beats": lambda v: "–" if pd.isna(v) else f"{int(v)}",
            },
        )
    )
    md.append("")
    md.append("## Quality gating")
    md.append("")
    n_w, n_p = len(wq), int(wq["passed"].astype(bool).sum())
    md.append(
        f"{n_p} of {n_w} windows of {cfg['windows']['length_s']} s passed all rules "
        f"({n_p / n_w:.1%}). Rules (version {cfg['qc'].get('version', 1)}):"
    )
    md.append("")
    for rule in cfg["qc"]["rules"]:
        params = ", ".join(f"{k}={v}" for k, v in rule.items() if k != "name")
        md.append(f"- `{rule['name']}` ({params})")
    md.append("")
    reasons = qc_reasons(wq)
    if reasons.empty:
        md.append("No window failed any rule.")
    else:
        md.append(_md_table(reasons, ["rule", "windows failed"]))
    md.append("")
    summary = r["summary"]
    if not summary.empty:
        ecfg = cfg["evaluation"]
        md.append("## Beat detection vs annotations")
        md.append("")
        md.append(
            f"Tolerance ±{ecfg['tolerance_ms']} ms after removing one per-record lag "
            f"(median reference-to-detection delay searched in "
            f"{ecfg['lag_search_ms'][0]}–{ecfg['lag_search_ms'][1]} ms). "
            "Beats in the dropped partial tail window are excluded from every column."
        )
        md.append("")
        gt = gating_table(summary)
        md.append(_md_table(gt, list(gt.columns), {"beats (all)": str, "beats (passed)": str}))
        md.append("")
        lags = r["lags"]
        md.append("Per-record lag (s), median over records by activity and detector:")
        md.append("")
        lt = lags.groupby(["detector", "activity"])["lag_s"].median().reset_index()
        md.append(_md_table(lt, ["detector", "activity", "lag_s"]))
        md.append("")
        md.append("Median absolute timing error of matched beats (ms), after lag removal:")
        md.append("")
        et = lags.groupby("detector")["abs_err_ms_median"].median().reset_index()
        md.append(
            _md_table(
                et, ["detector", "abs_err_ms_median"], {"abs_err_ms_median": lambda v: f"{v:.1f}"}
            )
        )
        md.append("")
    md.append("## Cache")
    md.append("")
    cs = r["cache_stages"]
    md.append("Stage executions in the run that produced this directory:")
    md.append("")
    md.append("| stage | hits | misses |\n|---|---|---|")
    for stage, c in cs.items():
        md.append(f"| {stage} | {c['hits']} | {c['misses']} |")
    md.append("")
    if figures:
        figs = write_figures(r)
        if figs:
            md.append("## Figures")
            md.append("")
            for f in figs:
                md.append(f"![{f.stem}]({f.name})")
                md.append("")
    md.append("## Environment")
    md.append("")
    md.append(
        "Python "
        + prov["python"]
        + "; "
        + ", ".join(f"{k} {v}" for k, v in prov["packages"].items() if v)
    )
    md.append("")
    out = Path(run_dir) / "report.md"
    out.write_text("\n".join(md))
    return out


def write_figures(r: dict[str, Any]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d: Path = r["dir"]
    out: list[Path] = []
    summary = r["summary"]
    if not summary.empty:
        gt = gating_table(summary)
        gt = gt[gt["activity"] != "all"] if (gt["activity"] != "all").any() else gt
        dets = list(dict.fromkeys(gt["detector"]))
        acts = list(dict.fromkeys(gt["activity"]))
        fig, axes = plt.subplots(
            1, len(dets), figsize=(3.2 * len(dets) + 1, 3.4), sharey=True, squeeze=False
        )
        for ax, det in zip(axes[0], dets, strict=True):
            sub = gt[gt["detector"] == det].set_index("activity").reindex(acts)
            xs = np.arange(len(acts))
            ax.bar(xs - 0.2, sub["F1 (all)"], 0.4, label="all windows", color="#9aa5b1")
            ax.bar(xs + 0.2, sub["F1 (passed)"], 0.4, label="QC-passed windows", color="#2f6f9f")
            ax.set_xticks(xs, acts)
            ax.set_title(det)
            ax.set_ylim(0, 1.02)
            ax.grid(axis="y", alpha=0.3)
        axes[0][0].set_ylabel("F1 vs annotated beats")
        axes[0][-1].legend(loc="lower right", fontsize=8)
        fig.suptitle("Beat detection F1 by activity, with and without quality gating", fontsize=10)
        fig.tight_layout()
        p = d / "fig_f1_by_activity.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out.append(p)
    ex = r.get("example")
    if ex:
        fs = float(ex["fs"])
        x = ex["preprocessed"]
        windows = ex["windows"]
        qc = r["window_qc"]
        qc = qc[qc["record"] == ex["record"]].reset_index(drop=True)
        t0, t1 = 0.0, min(60.0, len(x) / fs)
        sl = slice(int(t0 * fs), int(t1 * fs))
        t = np.arange(sl.start, sl.stop) / fs
        fig, ax = plt.subplots(figsize=(11, 3.2))
        ax.plot(t, x[sl], lw=0.6, color="#333", label="preprocessed signal")
        for i, (a, b) in enumerate(windows):
            if a / fs >= t1:
                break
            passed = bool(qc.loc[i, "passed"]) if i < len(qc) else True
            ax.axvspan(a / fs, b / fs, color="#2f6f9f" if passed else "#c0392b", alpha=0.08)
        ann = ex["annotations"]
        if len(ann):
            m = (ann >= sl.start) & (ann < sl.stop)
            ax.vlines(
                ann[m] / fs,
                *ax.get_ylim(),
                color="#c0392b",
                lw=0.5,
                alpha=0.6,
                label="reference beats",
            )
        dets = {k[4:]: v for k, v in ex.items() if k.startswith("det_")}
        if dets:
            name, pk = next(iter(dets.items()))
            m = (pk >= sl.start) & (pk < sl.stop)
            ax.plot(pk[m] / fs, x[pk[m]], "o", ms=3, color="#2f6f9f", label=f"{name} peaks")
        ax.set_xlabel("time (s)")
        ax.set_title(
            f"{ex['record']}: first {t1:.0f} s, windows shaded blue = QC pass, red = QC fail",
            fontsize=9,
        )
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        fig.tight_layout()
        p = d / "fig_example_qc.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out.append(p)
    return out


def compare_runs(a: Path, b: Path) -> str:
    """Markdown comparing two runs: config differences, cache stages that reran for b,
    and the F1 table side by side."""
    ra, rb = load_run(a), load_run(b)
    md = [f"# Compare `{ra['config']['name']}` vs `{rb['config']['name']}`", ""]
    md.append(f"A: `{a}` (config {ra['provenance']['config_hash'][:12]})")
    md.append(f"B: `{b}` (config {rb['provenance']['config_hash'][:12]})")
    md.append("")
    md.append("## Config differences")
    md.append("")
    diffs = _diff_dict(ra["config"], rb["config"])
    md.extend(diffs or ["(identical)"])
    md.append("")
    md.append("## Stages executed for B")
    md.append("")
    md.append("| stage | hits | misses |\n|---|---|---|")
    for stage, c in rb["cache_stages"].items():
        md.append(f"| {stage} | {c['hits']} | {c['misses']} |")
    md.append("")
    ga, gb = gating_table(ra["summary"]), gating_table(rb["summary"])
    if not ga.empty and not gb.empty:
        md.append("## F1 (all windows / QC-passed windows)")
        md.append("")
        m = ga.merge(gb, on=["detector", "activity"], suffixes=(" A", " B"), how="outer")
        m["ΔF1 (passed) B−A"] = m["F1 (passed) B"] - m["F1 (passed) A"]
        cols = [
            "detector",
            "activity",
            "F1 (all) A",
            "F1 (passed) A",
            "F1 (all) B",
            "F1 (passed) B",
            "ΔF1 (passed) B−A",
        ]
        md.append(_md_table(m, cols))
        md.append("")
    wa, wb = ra["window_qc"], rb["window_qc"]
    md.append(
        f"Windows passed: A {int(wa['passed'].astype(bool).sum())}/{len(wa)}, "
        f"B {int(wb['passed'].astype(bool).sum())}/{len(wb)}."
    )
    return "\n".join(md)


def _diff_dict(a: Any, b: Any, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out.extend(_diff_dict(a.get(k), b.get(k), f"{prefix}{k}."))
    elif a != b:
        out.append(f"- `{prefix.rstrip('.')}`: `{a}` → `{b}`")
    return out
