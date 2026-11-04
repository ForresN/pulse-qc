import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy import signal as ss

import pulse_qc
from pulse_qc import detect as detect_mod
from pulse_qc.qc import rules as rules_mod

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "synthetic.yaml"


@pytest.fixture(autouse=True)
def _clean_registry():
    before_d, before_r = dict(detect_mod.DETECTORS), dict(rules_mod.RULES)
    yield
    detect_mod.DETECTORS.clear()
    detect_mod.DETECTORS.update(before_d)
    rules_mod.RULES.clear()
    rules_mod.RULES.update(before_r)


def test_evaluate_returns_result_with_provenance(tmp_path):
    r = pulse_qc.evaluate(CONFIG, root=tmp_path, report=False)
    assert isinstance(r, pulse_qc.Result)
    assert r.detectors == ["elgendi", "prominence"]
    gt = r.gating_table()
    assert {"detector", "F1 (all)", "F1 (passed)"} <= set(gt.columns)
    assert r.scores("elgendi", gating="passed")["f1"] > 0.99
    prov = r.provenance
    assert prov["config_hash"] == r.config_hash and "packages" in prov and "git" in prov
    again = pulse_qc.Result.load(r.run_dir)
    assert again.config_hash == r.config_hash and len(again.summary) == len(r.summary)


def test_evaluate_overrides_and_custom_detector(tmp_path):
    @pulse_qc.register_detector("test_peaks")
    def test_peaks(y, fs, *, min_rr_s=0.3):
        peaks, _ = ss.find_peaks(y, distance=int(min_rr_s * fs), prominence=np.std(y))
        return peaks.astype(np.int64)

    base = pulse_qc.evaluate(CONFIG, root=tmp_path, report=False)
    r = pulse_qc.evaluate(
        CONFIG,
        root=tmp_path,
        detectors=["elgendi", {"name": "test_peaks", "min_rr_s": 0.4}],
        name="variant",
        report=False,
    )
    assert r.detectors == ["elgendi", "test_peaks"]
    assert r.run_dir != base.run_dir
    assert r.cache_summary["detect:elgendi"]["hits"] == 6  # shared cache
    assert r.cache_summary["detect:test_peaks"]["misses"] == 6
    assert r.scores("test_peaks")["f1"] > 0.9
    assert r.config["detectors"][1] == {"name": "test_peaks", "min_rr_s": 0.4}


def test_unknown_detector_is_a_clear_error(tmp_path):
    with pytest.raises(KeyError, match="register_detector"):
        pulse_qc.evaluate(CONFIG, root=tmp_path, detectors=["nope"])


def test_duplicate_registration_rejected():
    @pulse_qc.register_detector("dup")
    def a(y, fs):
        return np.empty(0, np.int64)

    with pytest.raises(ValueError):

        @pulse_qc.register_detector("dup")
        def b(y, fs):
            return np.empty(0, np.int64)


def test_custom_rule_with_threshold(tmp_path):
    from pulse_qc import holdout

    @pulse_qc.register_rule("p2p_floor", threshold=("min_p2p", "below"), candidates=[0.5, 1.0, 2.0])
    def p2p_floor(x, fs, *, min_p2p=0.0):
        f = x[np.isfinite(x)]
        p2p = float(f.max() - f.min()) if f.size else 0.0
        return pulse_qc.RuleResult(p2p >= min_p2p, p2p, "" if p2p >= min_p2p else "low")

    try:
        r = pulse_qc.evaluate(
            CONFIG,
            root=tmp_path,
            rules=[{"name": "nan_dropout"}, {"name": "p2p_floor", "min_p2p": 0.5}],
            name="ruled",
            report=False,
        )
        assert "p2p_floor" in r.window_qc.columns
        out = r.holdout(rules=["p2p_floor"])
        assert "p2p_floor" in out.read_text()
    finally:
        holdout.THRESHOLD_RULES.pop("p2p_floor", None)
        holdout.DEFAULT_CANDIDATES.pop("p2p_floor", None)


def test_plugins_are_imported_from_config(tmp_path, monkeypatch):
    mod = tmp_path / "my_plugin_mod.py"
    mod.write_text(
        textwrap.dedent(
            """
            import numpy as np
            import pulse_qc
            from scipy import signal as ss

            @pulse_qc.register_detector("plugin_peaks")
            def plugin_peaks(y, fs):
                p, _ = ss.find_peaks(y, distance=int(0.3 * fs), prominence=np.std(y))
                return p.astype(np.int64)
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["name"] = "plugged"
    cfg["plugins"] = ["my_plugin_mod"]
    cfg["detectors"] = [{"name": "plugin_peaks"}]
    r = pulse_qc.evaluate(cfg, root=tmp_path, report=False)
    assert r.scores("plugin_peaks")["f1"] > 0.9
    sys.modules.pop("my_plugin_mod", None)


def test_compare_api(tmp_path):
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())
    b = dict(cfg, name="b")
    b["windows"] = {"length_s": 5}
    md = pulse_qc.compare(cfg, b, root=tmp_path)
    assert "windows.length_s" in md


def test_example_script_runs(tmp_path, monkeypatch):
    import runpy

    monkeypatch.setattr(sys, "argv", ["custom_detector.py"])

    script = Path(__file__).resolve().parents[1] / "examples" / "custom_detector.py"
    monkey_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)
        (tmp_path / "configs").mkdir()
        (tmp_path / "configs" / "synthetic.yaml").write_text(CONFIG.read_text())
        runpy.run_path(str(script), run_name="__main__")
        assert any((tmp_path / "runs").glob("custom-detector-demo-*"))
    finally:
        os.chdir(monkey_cwd)
