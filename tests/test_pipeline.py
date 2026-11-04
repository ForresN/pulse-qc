"""End-to-end tests on synthetic fixtures: no network, no PhysioNet data."""

import copy
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from pulse_qc.cache import Cache
from pulse_qc.cli import app
from pulse_qc.pipeline import config_fingerprint, load_config, run
from pulse_qc.report import compare_runs, write_report

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "synthetic.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def first_run(cfg, tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    return root, run(cfg, root=root)


def test_run_produces_outputs_and_scores(first_run):
    root, r = first_run
    assert (r.run_dir / "provenance.json").exists()
    assert (r.run_dir / "summary.csv").exists()
    assert set(r.summary["gating"]) == {"all", "passed", "failed"}
    overall = r.summary[(r.summary["activity"] == "all") & (r.summary["gating"] == "passed")]
    assert (overall["f1"] > 0.99).all()
    # fixtures with injected faults fail QC; clean ones pass
    per_rec = r.window_qc.groupby("record")["passed"].mean()
    assert per_rec["synthetic/clean_a:signal"] == 1.0
    assert per_rec["synthetic/clipped_a:signal"] == 0.0
    assert per_rec["synthetic/dropout_a:signal"] < 1.0
    # every stage ran once per record on a cold cache
    assert all(v["misses"] == 5 and v["hits"] == 0 for v in r.cache_summary.values())


def test_second_run_is_all_cache_hits(first_run, cfg):
    root, _ = first_run
    r2 = run(cfg, root=root)
    assert all(v["hits"] == 5 and v["misses"] == 0 for v in r2.cache_summary.values())


def test_changed_qc_config_reruns_only_qc_and_evaluate(first_run, cfg):
    root, _ = first_run
    cfg2 = copy.deepcopy(cfg)
    cfg2["qc"]["rules"][0]["max_frac"] = 0.5  # loosen nan_dropout
    r2 = run(cfg2, root=root)
    s = r2.cache_summary
    assert s["preprocess"] == {"hits": 5, "misses": 0}
    assert s["detect:elgendi"] == {"hits": 5, "misses": 0}
    assert s["detect:prominence"] == {"hits": 5, "misses": 0}
    assert s["qc"] == {"hits": 0, "misses": 5}
    assert s["evaluate:elgendi"] == {"hits": 0, "misses": 5}
    assert config_fingerprint(cfg2) != config_fingerprint(cfg)
    assert r2.run_dir != run(cfg, root=root).run_dir


def test_new_detector_reruns_only_that_detector(first_run, cfg):
    root, _ = first_run
    cfg2 = copy.deepcopy(cfg)
    cfg2["detectors"].append({"name": "msptd"})
    r2 = run(cfg2, root=root)
    s = r2.cache_summary
    assert s["detect:msptd"] == {"hits": 0, "misses": 5}
    assert s["evaluate:msptd"] == {"hits": 0, "misses": 5}
    assert s["detect:elgendi"] == {"hits": 5, "misses": 0}
    assert s["qc"] == {"hits": 5, "misses": 0}


def test_changed_preprocess_reruns_detect_but_not_qc(first_run, cfg):
    root, _ = first_run
    cfg2 = copy.deepcopy(cfg)
    cfg2["preprocess"]["band_hz"] = [0.5, 6.0]
    s = run(cfg2, root=root).cache_summary
    assert s["preprocess"]["misses"] == 5
    assert s["detect:elgendi"]["misses"] == 5
    assert s["qc"] == {"hits": 5, "misses": 0}


def test_output_dir_does_not_affect_fingerprint(cfg):
    cfg2 = {**cfg, "output_dir": "elsewhere", "cache_dir": "x"}
    assert config_fingerprint(cfg2) == config_fingerprint(cfg)


def test_report_and_compare(first_run, cfg):
    root, r = first_run
    md = write_report(r.run_dir, figures=True).read_text()
    assert "## Beat detection vs annotations" in md
    assert "clipping" in md
    assert (r.run_dir / "fig_f1_by_activity.png").exists()
    assert (r.run_dir / "fig_example_qc.png").exists()
    cfg2 = copy.deepcopy(cfg)
    cfg2["name"] = "synthetic-loose"
    cfg2["qc"]["rules"] = [{"name": "nan_dropout", "max_frac": 0.5}]
    r2 = run(cfg2, root=root)
    cmp_md = compare_runs(r.run_dir, r2.run_dir)
    assert "qc.rules" in cmp_md
    assert "Windows passed" in cmp_md


def test_cli_run_and_report(tmp_path):
    runner = CliRunner()
    res = runner.invoke(app, ["run", str(CONFIG), "--root", str(tmp_path), "--no-figures"])
    assert res.exit_code == 0, res.output
    assert "misses=5" in res.output and "F1=" in res.output
    run_dir = next((tmp_path / "runs").iterdir())
    res = runner.invoke(app, ["report", str(run_dir), "--no-figures"])
    assert res.exit_code == 0, res.output
    assert (run_dir / "report.md").exists()


def test_cli_compare(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text())
    b = copy.deepcopy(cfg)
    b["name"] = "b"
    b["windows"]["length_s"] = 5
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(b))
    res = CliRunner().invoke(
        app,
        ["compare", str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml"), "--root", str(tmp_path)],
    )
    assert res.exit_code == 0, res.output
    assert "windows.length_s" in res.output


def test_cache_object_can_be_shared(cfg, tmp_path):
    cache = Cache(tmp_path / "c")
    run(cfg, cache=cache, root=tmp_path)
    assert cache.stage_summary()["qc"]["misses"] == 5
