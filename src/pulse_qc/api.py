"""The small public API: ``evaluate``, ``compare`` and the two plugin decorators.

    import pulse_qc

    result = pulse_qc.evaluate("configs/ptt-ppg.yaml")
    print(result.gating_table())       # F1 over all / QC-passed / QC-failed windows
    result.provenance["git"]           # sha, dirty flag, package versions, record hashes

    @pulse_qc.register_detector("my_peaks")
    def my_peaks(y, fs, *, min_rr_s=0.3):
        ...
        return peak_sample_indices

    result = pulse_qc.evaluate("configs/ptt-ppg.yaml", detectors=["elgendi", "my_peaks"])

Every ``Result`` carries its provenance and is written to ``runs/<name>-<confighash>/``
exactly as the CLI does; the CLI is a thin layer over these functions.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pulse_qc import detect as detect_mod
from pulse_qc.pipeline import RunResult, load_config, load_plugins
from pulse_qc.pipeline import run as _run
from pulse_qc.qc import rules as rules_mod

ConfigLike = str | Path | dict[str, Any]


def _as_config(config: ConfigLike) -> dict[str, Any]:
    if isinstance(config, dict):
        cfg = copy.deepcopy(config)
        if "name" not in cfg:
            raise ValueError("config dict needs a 'name'")
        # apply the same defaults as a file would get
        for k, v in {
            "preprocess": {},
            "windows": {"length_s": 10.0},
            "qc": {"version": 1, "rules": []},
            "detectors": [{"name": "elgendi"}],
            "evaluation": {"tolerance_ms": 100, "lag_search_ms": [0, 600]},
            "output_dir": "runs",
            "cache_dir": "cache",
            "plugins": [],
        }.items():
            cfg.setdefault(k, v)
        return cfg
    return load_config(Path(config))


@dataclass
class Result:
    """What :func:`evaluate` returns. Thin, picklable view over a saved run directory."""

    run_dir: Path
    config: dict[str, Any]
    config_hash: str
    summary: pd.DataFrame
    window_qc: pd.DataFrame
    beat_counts: pd.DataFrame
    detections: pd.DataFrame
    records: list[dict[str, Any]]
    cache_summary: dict[str, dict[str, int]]
    neuro_summary: pd.DataFrame | None = None

    @property
    def provenance(self) -> dict[str, Any]:
        return json.loads((self.run_dir / "provenance.json").read_text())

    @property
    def detectors(self) -> list[str]:
        return [d["name"] for d in self.config["detectors"]]

    def gating_table(self) -> pd.DataFrame:
        """F1 / sensitivity / PPV per detector and activity over all, QC-passed and QC-failed
        windows (empty when the data had no annotations)."""
        from pulse_qc.report import gating_table

        return gating_table(self.summary)

    def scores(self, detector: str, gating: str = "all", activity: str = "all") -> dict[str, float]:
        """One row of the summary as a dict: ``{"sensitivity", "ppv", "f1", "tp", "fn", "fp"}``."""
        s = self.summary
        row = s[(s["detector"] == detector) & (s["gating"] == gating) & (s["activity"] == activity)]
        if row.empty:
            raise KeyError(
                f"no scores for detector={detector!r} gating={gating!r} activity={activity!r}"
            )
        r = row.iloc[0]
        return {k: float(r[k]) for k in ("sensitivity", "ppv", "f1", "tp", "fn", "fp", "n_ref")}

    def report(self, figures: bool = True) -> Path:
        from pulse_qc.report import write_report

        return write_report(self.run_dir, figures=figures)

    def holdout(self, rules: list[str] | None = None, bad_f1: float = 0.9) -> Path:
        """Leave-one-subject-out threshold selection for single-threshold rules; writes
        ``holdout.md`` into the run directory and returns its path."""
        from pulse_qc.holdout import THRESHOLD_RULES, write_holdout

        if rules is None:
            present = {r["name"] for r in self.config["qc"]["rules"]}
            rules = [r for r in THRESHOLD_RULES if r in present]
        return write_holdout(self.run_dir, rules, bad_f1=bad_f1)

    @classmethod
    def from_run(cls, rr: RunResult) -> Result:
        return cls(
            run_dir=rr.run_dir,
            config=rr.config,
            config_hash=rr.config_hash,
            summary=rr.summary,
            window_qc=rr.window_qc,
            beat_counts=rr.beat_counts,
            detections=rr.lags,
            records=rr.records,
            cache_summary=rr.cache_summary,
            neuro_summary=pd.DataFrame(rr.neuro["summary"]) if rr.neuro else None,
        )

    @classmethod
    def load(cls, run_dir: str | Path) -> Result:
        """Reopen a saved run directory without recomputing anything."""
        from pulse_qc.report import load_run

        r = load_run(Path(run_dir))
        prov = r["provenance"]
        return cls(
            run_dir=Path(run_dir),
            config=r["config"],
            config_hash=prov["config_hash"],
            summary=r["summary"],
            window_qc=r["window_qc"],
            beat_counts=r["beat_counts"],
            detections=r["lags"],
            records=prov["records"],
            cache_summary=r["cache_stages"],
            neuro_summary=r["neuro"]["summary"] if r.get("neuro") else None,
        )


def evaluate(
    config: ConfigLike,
    *,
    root: str | Path = ".",
    detectors: list[str | dict[str, Any]] | None = None,
    rules: list[dict[str, Any]] | None = None,
    name: str | None = None,
    report: bool = True,
) -> Result:
    """Run a config end to end (cached) and return a :class:`Result`.

    ``detectors`` and ``rules`` override the config's lists without editing the file, so a
    comparison is one call: ``evaluate(cfg, detectors=["elgendi", "mine"])``. A detector
    may be a name or a ``{"name": ..., **params}`` dict. ``name`` renames the run so that
    its directory does not collide with the base config's.
    """
    cfg = _as_config(config)
    if detectors is not None:
        cfg["detectors"] = [{"name": d} if isinstance(d, str) else dict(d) for d in detectors]
    if rules is not None:
        cfg["qc"] = {**cfg["qc"], "rules": [dict(r) for r in rules]}
    if name is not None:
        cfg["name"] = name
    load_plugins(cfg)
    unknown = [d["name"] for d in cfg["detectors"] if d["name"] not in detect_mod.DETECTORS]
    if unknown:
        raise KeyError(
            f"unknown detector(s) {unknown}; registered: {sorted(detect_mod.DETECTORS)}. "
            "Register with @pulse_qc.register_detector or list the module under plugins:."
        )
    rr = _run(cfg, root=Path(root))
    result = Result.from_run(rr)
    if report:
        result.report(figures=True)
    return result


def compare(config_a: ConfigLike, config_b: ConfigLike, *, root: str | Path = ".") -> str:
    """Run both configs (sharing the cache) and return the comparison as Markdown."""
    from pulse_qc.report import compare_runs

    ra = evaluate(config_a, root=root, report=False)
    rb = evaluate(config_b, root=root, report=False)
    return compare_runs(ra.run_dir, rb.run_dir)


def register_detector(
    name: str,
) -> Callable[[Callable[..., np.ndarray]], Callable[..., np.ndarray]]:
    """Register ``fn(y, fs, **params) -> peak sample indices`` under ``name``.

    ``y`` is the preprocessed signal (NaN-free, band-passed, peaks pointing up). The
    detector is then usable from YAML (``detectors: [{name: <name>, ...}]``) and from
    :func:`evaluate`. Its cache key includes the function's own source, so editing it
    reruns it."""

    def deco(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        if name in detect_mod.DETECTORS and detect_mod.DETECTORS[name] is not fn:
            raise ValueError(f"detector {name!r} is already registered")
        detect_mod.DETECTORS[name] = fn
        return fn

    return deco


def register_rule(
    name: str, *, threshold: tuple[str, str] | None = None, candidates: np.ndarray | None = None
) -> Callable[[Callable[..., rules_mod.RuleResult]], Callable[..., rules_mod.RuleResult]]:
    """Register ``fn(x, fs, *, **params) -> RuleResult`` as a QC rule.

    Declare ``aux`` as a keyword parameter to receive auxiliary channels. If the rule is a
    single threshold on its value, pass ``threshold=("param_name", "below"|"above")`` (the
    direction in which it *fails*) and optionally ``candidates`` so that
    ``pulse-qc holdout`` can tune it on held-out subjects."""

    def deco(fn: Callable[..., rules_mod.RuleResult]) -> Callable[..., rules_mod.RuleResult]:
        if name in rules_mod.RULES and rules_mod.RULES[name] is not fn:
            raise ValueError(f"rule {name!r} is already registered")
        rules_mod.RULES[name] = fn
        if threshold is not None:
            from pulse_qc import holdout

            holdout.THRESHOLD_RULES[name] = threshold
            if candidates is not None:
                holdout.DEFAULT_CANDIDATES[name] = np.asarray(candidates)
        return fn

    return deco
