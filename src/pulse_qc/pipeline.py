"""Stage orchestration: load -> preprocess -> qc -> detect -> evaluate.

Every stage result goes through :class:`~pulse_qc.cache.Cache`, keyed on the
hashes of its inputs, the relevant config section and its own source code. So a
change to the QC rules reruns ``qc`` and ``evaluate`` but not ``preprocess`` or
``detect``; a new detector reruns only that detector and its evaluation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pulse_qc import detect as detect_mod
from pulse_qc import ingest, synth
from pulse_qc import scoring as eval_mod
from pulse_qc.cache import Cache, code_hash, config_hash, make_key
from pulse_qc.ingest import Record
from pulse_qc.qc import RuleSet, apply_rules
from pulse_qc.qc import rules as rules_mod
from pulse_qc.windows import make_windows


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or "name" not in cfg:
        raise ValueError(f"{path}: config must be a mapping with a 'name'")
    cfg.setdefault("preprocess", {})
    cfg.setdefault("windows", {"length_s": 10.0})
    cfg.setdefault("qc", {"version": 1, "rules": []})
    cfg.setdefault("detectors", [{"name": "elgendi"}])
    cfg.setdefault("evaluation", {"tolerance_ms": 100, "lag_search_ms": [0, 600]})
    cfg.setdefault("output_dir", "runs")
    cfg.setdefault("cache_dir", "cache")
    cfg.setdefault("plugins", [])
    return cfg


def config_fingerprint(cfg: dict[str, Any]) -> str:
    """Hash of everything that affects results (not output/cache paths)."""
    keep = {k: v for k, v in cfg.items() if k not in {"output_dir", "cache_dir"}}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class RecordSpec:
    db: str
    name: str
    channel: str
    annotator: str | None
    activity: str | None = None
    subject: str | None = None
    aux_channels: tuple[str, ...] = ()
    segment: dict[str, float] | None = None
    qc: dict[str, Any] | None = None  # per-group override of the top-level qc section

    @property
    def record_id(self) -> str:
        return f"{self.db}/{self.name}:{self.channel}"


def record_specs(cfg: dict[str, Any]) -> list[RecordSpec]:
    specs = []
    for group in cfg["data"]["records"]:
        pattern = re.compile(group.get("subject_pattern", r"^([^_]+)"))
        for name in group["names"]:
            m = pattern.search(name)
            specs.append(
                RecordSpec(
                    group["db"],
                    name,
                    group.get("channel", "signal"),
                    group.get("annotator"),
                    group.get("activity"),
                    m.group(1) if m else name,
                    tuple(group.get("aux_channels", [])),
                    group.get("segment"),
                    group.get("qc"),
                )
            )
    return specs


# --------------------------------------------------------------------------- stages


def _synthetic_record(name: str, channel: str = "signal", aux: tuple[str, ...] = ()) -> Record:
    """Deterministic synthetic fixtures by name, for tests and CI.

    ``channel`` "ICP"/"ABP" produce a slow-drifting pressure-like pair that share beats
    (ABP = 80 + 20*pulse, ICP = 12 + 3*pulse plus a slow component correlated with ABP),
    so the neuro stage can be tested. ``aux`` names ending in ``_x/_y/_z`` get a
    synthetic accelerometer: quiet for the first half, shaking for the second."""
    seed = sum(ord(c) for c in name) % 1000
    base = synth.synthetic_ppg(duration_s=60, fs=125, hr_bpm=72, seed=seed)
    x = base.signal
    fs = base.fs
    if channel in {"ICP", "ABP"}:
        t = np.arange(len(x)) / fs
        slow = np.sin(2 * np.pi * t / 20)  # shared slow wave -> positive ICP/ABP correlation
        pulse = x - 1.0
        x = (80 + 20 * pulse + 8 * slow) if channel == "ABP" else (12 + 3 * pulse + 2 * slow)
    aux_data: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed + 1)
    for a in aux:
        quiet = rng.normal(0, 0.005, len(x))
        shaking = rng.normal(0, 0.3, len(x))
        half = len(x) // 2 if name.startswith("motion") else len(x)
        aux_data[a] = np.where(np.arange(len(x)) < half, quiet, shaking) + (
            1.0 if a.endswith("_z") else 0.0
        )
    if name.startswith("dropout"):
        x = synth.inject_nan(x, int(12 * fs), int(15 * fs))
    elif name.startswith("flat"):
        x = synth.inject_flatline(x, int(31 * fs), int(34 * fs))
    elif name.startswith("clipped"):
        x = synth.inject_clipping(x, float(np.percentile(x, 85)))
    return Record(
        db="synthetic",
        name=name,
        channel=channel,
        fs=fs,
        signal=x,
        ann_samples=base.beat_samples,
        meta={"activity": "synthetic"},
        source_sha256={name: hashlib.sha256(f"{name}:{channel}".encode()).hexdigest()},
        aux=aux_data,
        units="mmHg" if channel in {"ICP", "ABP"} else "",
    )


def _load(spec: RecordSpec, cfg: dict[str, Any], cache: Cache) -> tuple[Record, str]:
    """Return the record and its load key (which encodes the raw file hashes)."""
    if spec.db == "synthetic":
        rec = _synthetic_record(spec.name, spec.channel, spec.aux_channels)
        key = make_key(
            "load",
            source=json.dumps(rec.source_sha256, sort_keys=True),
            channel=spec.channel,
            aux=",".join(spec.aux_channels),
            code=code_hash(synth, _synthetic_record),
        )
        return rec, key
    raw_dir = Path(cfg["data"].get("raw_dir", "data/raw"))
    verify = bool(cfg["data"].get("verify_checksums", True))
    if spec.segment:
        start_s, duration_s = float(spec.segment["start_s"]), float(spec.segment["duration_s"])
        rec = ingest.read_segment(
            spec.db,
            spec.name,
            spec.channel,
            raw_dir,
            start_s,
            duration_s,
            aux_channels=list(spec.aux_channels),
        )
        key = make_key(
            "load",
            source=json.dumps(rec.source_sha256, sort_keys=True),
            channel=spec.channel,
            aux=",".join(spec.aux_channels),
            code=code_hash(ingest),
        )
        return rec, key
    files = [f"{spec.name}.hea", f"{spec.name}.dat"] + (
        [f"{spec.name}.{spec.annotator}"] if spec.annotator else []
    )
    digests = ingest.ensure_files(spec.db, files, raw_dir, verify=verify)
    key = make_key(
        "load",
        source=json.dumps(digests, sort_keys=True),
        channel=spec.channel,
        aux=",".join(spec.aux_channels),
        annotator=spec.annotator or "",
        code=code_hash(ingest),
    )
    rec = cache.get_or_compute(
        "load",
        key,
        lambda: ingest.read_record(
            spec.db,
            spec.name,
            spec.channel,
            raw_dir,
            annotator=spec.annotator,
            verify=verify,
            aux_channels=list(spec.aux_channels),
        ),
        meta={"record": f"{spec.db}/{spec.name}:{spec.channel}"},
    )
    return rec, key


def _preprocess(
    rec: Record, load_key: str, cfg: dict[str, Any], cache: Cache
) -> tuple[np.ndarray, str]:
    section = cfg["preprocess"]
    key = make_key(
        "preprocess",
        input=load_key,
        config=config_hash(section),
        code=code_hash(detect_mod.preprocess),
    )
    params = dict(section)
    if "band_hz" in params:
        params["band_hz"] = tuple(params["band_hz"])
    y = cache.get_or_compute(
        "preprocess", key, lambda: detect_mod.preprocess(rec.signal, rec.fs, **params)
    )
    return y, key


def _qc(
    rec: Record, load_key: str, cfg: dict[str, Any], cache: Cache, qc_cfg: dict | None = None
) -> tuple[pd.DataFrame, np.ndarray, str]:
    ruleset = RuleSet.from_config(qc_cfg or cfg["qc"])
    wcfg = cfg["windows"]
    key = make_key(
        "qc",
        input=load_key,
        windows=config_hash(wcfg),
        rules=ruleset.fingerprint,
        code=code_hash(rules_mod, apply_rules),
    )
    windows = make_windows(len(rec.signal), rec.fs, float(wcfg["length_s"]), wcfg.get("step_s"))
    df = cache.get_or_compute(
        "qc", key, lambda: apply_rules(rec.signal, rec.fs, windows, ruleset, aux=rec.aux)
    )
    return df, windows, key


def _detect(
    y: np.ndarray, fs: float, pre_key: str, det_cfg: dict[str, Any], cache: Cache
) -> tuple[np.ndarray, str]:
    name = det_cfg["name"]
    params = {k: v for k, v in det_cfg.items() if k != "name"}
    key = make_key(
        f"detect:{name}",
        input=pre_key,
        config=config_hash(det_cfg),
        version=str(detect_mod.DETECT_VERSION),
        code=code_hash(detect_mod.DETECTORS[name], detect_mod.run_detector),
    )
    peaks = cache.get_or_compute(
        f"detect:{name}", key, lambda: detect_mod.run_detector(name, y, fs, params)
    )
    return peaks, key


def _evaluate(
    rec: Record,
    peaks: np.ndarray,
    windows: np.ndarray,
    det_key: str,
    qc_key: str,
    cfg: dict[str, Any],
    cache: Cache,
    det_name: str,
) -> tuple[pd.DataFrame, float]:
    ecfg = cfg["evaluation"]
    key = make_key(
        f"evaluate:{det_name}",
        detections=det_key,
        qc=qc_key,
        config=config_hash(ecfg),
        code=code_hash(eval_mod),
    )

    def compute():
        tol = float(ecfg.get("tolerance_ms", 100)) / 1000
        lo, hi = ecfg.get("lag_search_ms", [0, 600])
        m = eval_mod.match_beats(
            rec.ann_samples, peaks, rec.fs, tol, lag_search_s=(lo / 1000, hi / 1000)
        )
        counts = eval_mod.per_window_counts(m, rec.ann_samples, peaks, rec.fs, windows)
        counts["abs_err_ms_median"] = (
            float(np.median(np.abs(m.errors_s)) * 1000) if len(m.errors_s) else np.nan
        )
        return counts, m.lag_s

    return cache.get_or_compute(f"evaluate:{det_name}", key, compute)


# --------------------------------------------------------------------------- run


@dataclass
class RunResult:
    config: dict[str, Any]
    config_hash: str
    run_dir: Path
    window_qc: pd.DataFrame
    beat_counts: pd.DataFrame
    lags: pd.DataFrame
    summary: pd.DataFrame
    records: list[dict[str, Any]]
    cache_summary: dict[str, dict[str, int]]
    examples: dict[str, Any] = field(default_factory=dict)
    neuro: dict[str, Any] = field(default_factory=dict)


def load_plugins(cfg: dict[str, Any]) -> list[str]:
    """Import every module named under ``plugins:`` so that its ``@register_detector`` /
    ``@register_rule`` decorators run. Returns the imported module names."""
    import importlib

    names = list(cfg.get("plugins", []) or [])
    for name in names:
        importlib.import_module(name)
    return names


def run(cfg: dict[str, Any], *, cache: Cache | None = None, root: Path | None = None) -> RunResult:
    root = Path(root or ".")
    load_plugins(cfg)
    cache = cache or Cache(root / cfg["cache_dir"])
    cache.reset_events()
    cfg_hash = config_fingerprint(cfg)
    run_dir = root / cfg["output_dir"] / f"{cfg['name']}-{cfg_hash[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    qc_frames, count_frames, lag_rows, rec_rows = [], [], [], []
    examples: dict[str, Any] = {}
    loaded: dict[str, dict[str, Any]] = {}  # record_id -> pieces the neuro stage needs
    for spec in record_specs(cfg):
        rec, load_key = _load(spec, cfg, cache)
        activity = spec.activity or rec.meta.get("activity", "unknown")
        rid = rec.record_id
        y, pre_key = _preprocess(rec, load_key, cfg, cache)
        qc_df, windows, qc_key = _qc(rec, load_key, cfg, cache, spec.qc)
        qc_df = qc_df.assign(record=rid, subject=spec.subject, activity=activity)
        qc_frames.append(qc_df)
        rec_rows.append(
            {
                "record": rid,
                "db": rec.db,
                "name": rec.name,
                "channel": rec.channel,
                "fs": rec.fs,
                "duration_s": rec.duration_s,
                "subject": spec.subject,
                "activity": activity,
                "n_annotated_beats": int(len(rec.ann_samples))
                if rec.ann_samples is not None
                else None,
                "n_windows": int(len(windows)),
                "n_windows_passed": int(qc_df["passed"].sum()),
                "units": rec.units,
                "segment": spec.segment,
                "source_sha256": rec.source_sha256,
            }
        )
        detections: dict[str, np.ndarray] = {}
        for det_cfg in cfg["detectors"]:
            peaks, det_key = _detect(y, rec.fs, pre_key, det_cfg, cache)
            detections[det_cfg["name"]] = peaks
            det_row = {
                "record": rid,
                "activity": activity,
                "detector": det_cfg["name"],
                "n_detected": int(len(peaks)),
                "detected_per_min": float(len(peaks) / rec.duration_s * 60),
                "lag_s": np.nan,
                "abs_err_ms_median": np.nan,
            }
            if rec.ann_samples is None:
                lag_rows.append(det_row)
                continue
            counts, lag = _evaluate(
                rec, peaks, windows, det_key, qc_key, cfg, cache, det_cfg["name"]
            )
            counts = counts.merge(qc_df[["window", "passed", "reasons"]], on="window", how="left")
            counts["passed"] = counts["passed"].astype("boolean")
            counts = counts.assign(
                record=rid, subject=spec.subject, activity=activity, detector=det_cfg["name"]
            )
            count_frames.append(counts)
            det_row["lag_s"] = lag
            det_row["abs_err_ms_median"] = float(counts["abs_err_ms_median"].iloc[0])
            lag_rows.append(det_row)
        loaded[rid] = {
            "spec": spec,
            "rec": rec,
            "load_key": load_key,
            "qc_df": qc_df,
            "qc_key": qc_key,
            "windows": windows,
            "detections": detections,
        }
        has_fail = not bool(qc_df["passed"].all())
        if not examples or (has_fail and not examples.get("has_fail")):
            examples = {
                "has_fail": has_fail,
                "record": rid,
                "fs": rec.fs,
                "signal": rec.signal,
                "preprocessed": y,
                "windows": windows,
                "qc": qc_df,
                "annotations": rec.ann_samples,
                "detections": detections,
            }

    window_qc = pd.concat(qc_frames, ignore_index=True)
    beat_counts = (
        pd.concat(count_frames, ignore_index=True)
        if count_frames
        else pd.DataFrame(
            columns=[
                "record",
                "subject",
                "activity",
                "detector",
                "window",
                "passed",
                "tp",
                "fn",
                "fp",
            ]
        )
    )
    lags = pd.DataFrame(
        lag_rows,
        columns=[
            "record",
            "activity",
            "detector",
            "n_detected",
            "detected_per_min",
            "lag_s",
            "abs_err_ms_median",
        ],
    )
    summary = summarise_counts(beat_counts)
    neuro_out: dict[str, Any] = {}
    if cfg.get("neuro"):
        from pulse_qc import neuro as neuro_mod

        neuro_out = neuro_mod.run_stage(cfg, loaded, cache)

    result = RunResult(
        cfg,
        cfg_hash,
        run_dir,
        window_qc,
        beat_counts,
        lags,
        summary,
        rec_rows,
        cache.stage_summary(),
        examples,
        neuro_out,
    )
    save_run(result)
    return result


def summarise_counts(beat_counts: pd.DataFrame) -> pd.DataFrame:
    """Scores per detector x activity x gating, where gating is one of
    ``all`` (every windowed beat), ``passed`` (QC-passing windows only) and ``failed``."""
    if beat_counts.empty:
        return pd.DataFrame(
            columns=[
                "detector",
                "activity",
                "gating",
                "tp",
                "fn",
                "fp",
                "sensitivity",
                "ppv",
                "f1",
                "n_ref",
            ]
        )
    windowed = beat_counts[beat_counts["window"] >= 0]
    passed_mask = windowed["passed"].fillna(False).astype(bool)
    frames = []
    for gating, sub in (
        ("all", windowed),
        ("passed", windowed[passed_mask]),
        ("failed", windowed[~passed_mask]),
    ):
        if sub.empty:
            continue
        for by in (["detector", "activity"], ["detector"]):
            s = eval_mod.summarise(sub, by)
            if "activity" not in by:
                s["activity"] = "all"
            s["gating"] = gating
            frames.append(s)
    out = pd.concat(frames, ignore_index=True)
    cols = ["detector", "activity", "gating", "tp", "fn", "fp", "sensitivity", "ppv", "f1", "n_ref"]
    return out[cols].sort_values(["detector", "activity", "gating"]).reset_index(drop=True)


def save_run(result: RunResult) -> None:
    d = result.run_dir
    (d / "config.yaml").write_text(yaml.safe_dump(result.config, sort_keys=False))
    result.window_qc.to_csv(d / "window_qc.csv", index=False)
    result.beat_counts.to_csv(d / "beat_counts.csv", index=False)
    result.lags.to_csv(d / "lags.csv", index=False)
    result.summary.to_csv(d / "summary.csv", index=False)
    (d / "cache_stages.json").write_text(json.dumps(result.cache_summary, indent=2))
    from pulse_qc.provenance import provenance

    (d / "provenance.json").write_text(json.dumps(provenance(result), indent=2, default=str))
    if result.neuro:
        from pulse_qc import neuro as neuro_mod

        neuro_mod.save(result.neuro, d)
    if result.examples:
        ex = result.examples
        np.savez_compressed(
            d / "example.npz",
            signal=ex["signal"],
            preprocessed=ex["preprocessed"],
            windows=ex["windows"],
            fs=ex["fs"],
            annotations=ex["annotations"] if ex["annotations"] is not None else np.empty(0),
            **{f"det_{k}": v for k, v in ex["detections"].items()},
        )
        (d / "example.json").write_text(json.dumps({"record": ex["record"]}))
