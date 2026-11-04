import numpy as np
import pytest

from pulse_qc.qc import RuleSet, apply_rules
from pulse_qc.qc.engine import RuleSpec
from pulse_qc.synth import inject_flatline, inject_nan
from pulse_qc.windows import make_windows, window_index_of

CFG = {
    "version": 1,
    "rules": [
        {"name": "nan_dropout", "max_frac": 0.01},
        {"name": "flatline", "max_flat_s": 1.0},
        {"name": "clipping", "max_frac": 0.02},
    ],
}


def test_from_config_roundtrip():
    rs = RuleSet.from_config(CFG)
    assert rs.to_config() == CFG
    assert rs.rules[0] == RuleSpec("nan_dropout", {"max_frac": 0.01})


def test_unknown_rule_rejected():
    with pytest.raises(KeyError):
        RuleSet.from_config({"rules": [{"name": "vibes"}]})


def test_fingerprint_changes_with_params_and_version():
    a = RuleSet.from_config(CFG).fingerprint
    b = RuleSet.from_config({**CFG, "version": 2}).fingerprint
    c = RuleSet.from_config(
        {**CFG, "rules": [{"name": "nan_dropout", "max_frac": 0.02}, *CFG["rules"][1:]]}
    ).fingerprint
    assert len({a, b, c}) == 3
    assert RuleSet.from_config(CFG).fingerprint == a  # deterministic


def test_apply_rules_reports_per_window_reasons(clean):
    fs = clean.fs
    y = clean.signal.copy()
    y = inject_nan(y, int(12 * fs), int(14 * fs))  # inside window 1 (10–20 s)
    y = inject_flatline(y, int(31 * fs), int(34 * fs))  # inside window 3
    windows = make_windows(len(y), fs, 10.0)
    df = apply_rules(y, fs, windows, RuleSet.from_config(CFG))
    assert list(df["window"]) == list(range(len(windows)))
    assert df["passed"].tolist() == [True, False, True, False, True, True]
    assert df.loc[1, "reasons"].startswith("nan_dropout:")
    assert df.loc[3, "reasons"].startswith("flatline:")
    assert df.loc[0, "reasons"] == ""
    assert {"nan_dropout", "flatline", "clipping"} <= set(df.columns)


def test_make_windows_drops_partial_tail():
    w = make_windows(1250 + 7, 125.0, 2.0)
    assert w.shape == (5, 2)
    assert w[0].tolist() == [0, 250] and w[-1].tolist() == [1000, 1250]
    assert make_windows(10, 125.0, 2.0).shape == (0, 2)


def test_make_windows_overlap():
    w = make_windows(1000, 100.0, 2.0, step_s=1.0)
    assert w[:3].tolist() == [[0, 200], [100, 300], [200, 400]]


def test_window_index_of():
    w = make_windows(1000, 100.0, 2.0)
    s = np.array([0, 199, 200, 999, 1000, 5000])
    assert window_index_of(s, w).tolist() == [0, 0, 1, 4, -1, -1]
    assert window_index_of(s, w[:0]).tolist() == [-1] * 6
