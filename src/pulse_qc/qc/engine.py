"""Apply a configured set of rules to every window of a signal."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pulse_qc.qc.rules import RULES, RULES_VERSION, RuleResult


@dataclass(frozen=True)
class RuleSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def run(self, x: np.ndarray, fs: float) -> RuleResult:
        return RULES[self.name](x, fs, **self.params)


@dataclass(frozen=True)
class RuleSet:
    """An ordered list of rules plus a config version. ``fingerprint`` is what the
    cache is keyed on: it changes if any rule, parameter, order, or the rules
    code version changes."""

    rules: tuple[RuleSpec, ...]
    version: int = 1

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> RuleSet:
        specs = []
        for item in cfg.get("rules", []):
            item = dict(item)
            name = item.pop("name")
            if name not in RULES:
                raise KeyError(f"unknown QC rule {name!r}; known: {sorted(RULES)}")
            specs.append(RuleSpec(name, item))
        return cls(tuple(specs), int(cfg.get("version", 1)))

    def to_config(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "rules": [{"name": r.name, **r.params} for r in self.rules],
        }

    @property
    def fingerprint(self) -> str:
        payload = {"rules_code_version": RULES_VERSION, **self.to_config()}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def apply_rules(
    signal: np.ndarray, fs: float, windows: np.ndarray, ruleset: RuleSet
) -> pd.DataFrame:
    """Evaluate every rule on every window.

    Returns one row per window with ``passed`` (all rules passed), ``reasons``
    (semicolon-joined failure reasons, empty when passed) and one ``<rule>``
    column holding the measured value for each rule.
    """
    rows = []
    for i, (start, stop) in enumerate(windows):
        x = signal[start:stop]
        row: dict[str, Any] = {
            "window": i,
            "start_s": start / fs,
            "end_s": stop / fs,
        }
        reasons = []
        for spec in ruleset.rules:
            res = spec.run(x, fs)
            row[spec.name] = res.value
            if not res.passed:
                reasons.append(f"{spec.name}: {res.reason}")
        row["passed"] = not reasons
        row["reasons"] = "; ".join(reasons)
        rows.append(row)
    cols = ["window", "start_s", "end_s", "passed", "reasons"] + [r.name for r in ruleset.rules]
    return pd.DataFrame(rows, columns=cols)
