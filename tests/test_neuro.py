from pathlib import Path

import numpy as np
import pytest

from pulse_qc.neuro import masked_means, moving_correlation, pulse_amplitude
from pulse_qc.pipeline import load_config, run
from pulse_qc.report import write_report
from pulse_qc.synth import synthetic_ppg
from pulse_qc.windows import make_windows

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "synthetic-neuro.yaml"


def test_pulse_amplitude_on_synthetic():
    s = synthetic_ppg(duration_s=20, fs=125, hr_bpm=60, noise_sd=0.0, seed=3)
    amp = pulse_amplitude(s.signal, s.fs, s.beat_samples)
    assert len(amp) == len(s.beat_samples) - 1
    # each pulse rises ~1 unit above the diastolic floor
    assert 0.8 < amp["amplitude"].median() < 1.2
    assert (amp["t_s"].diff().dropna() > 0.5).all()


def test_pulse_amplitude_nan_interval():
    x = np.ones(100)
    x[20:30] = np.nan
    amp = pulse_amplitude(x, 10.0, np.array([10, 40, 70]))
    assert np.isnan(amp["amplitude"].iloc[0]) and amp["amplitude"].iloc[1] == 0.0


def test_masked_means():
    x = np.arange(100, dtype=float)
    w = make_windows(100, 10.0, 2.0)  # 5 windows of 20
    m = masked_means(x, 10.0, w, np.array([True, False, True, True, True]))
    assert m[0] == 9.5 and np.isnan(m[1]) and m[2] == 49.5


def test_moving_correlation_sign_and_validity():
    t = np.arange(60)
    x = np.sin(t / 5)
    y = 2 * x + 0.01 * np.cos(t)
    c = moving_correlation(x, y, corr_n=10, step_n=5)
    assert (c["r"] > 0.95).all()
    c2 = moving_correlation(x, -y, corr_n=10, step_n=5)
    assert (c2["r"] < -0.95).all()
    x_nan = x.copy()
    x_nan[:5] = np.nan  # first block has 50 % valid -> NaN
    c3 = moving_correlation(x_nan, y, corr_n=10, step_n=5, min_valid_frac=0.8)
    assert np.isnan(c3["r"].iloc[0]) and c3["n_valid"].iloc[0] == 5
    assert c3["r"].iloc[1] > 0.95


@pytest.fixture(scope="module")
def neuro_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    return run(load_config(CONFIG), root=root)


def test_neuro_stage_pairs_records(neuro_run):
    r = neuro_run
    assert set(r.neuro["pairs"]) == {"synthetic/pair_a", "synthetic/pair_b"}
    s = {row["record"]: row for row in r.neuro["summary"]}
    for row in s.values():
        assert row["windows_both_passed"] == row["n_windows"]
        assert row["corr_blocks_valid"] == row["n_corr_blocks"] > 0
        assert row["corr_median"] > 0.9  # shared slow wave -> strongly positive
        assert 2.0 < row["icp_pulse_amp_median"] < 4.0
        assert 15 < row["abp_pulse_amp_median"] < 25
    assert r.cache_summary["neuro"] == {"hits": 0, "misses": 2}
    assert (r.run_dir / "neuro" / "summary.csv").exists()


def test_neuro_report_and_figure(neuro_run):
    r = neuro_run
    md = write_report(r.run_dir).read_text()
    assert "Neuro-ICU waveform features (non-clinical)" in md
    assert (r.run_dir / "fig_neuro_synthetic__pair_a.png").exists()


def test_neuro_stage_is_cached(neuro_run):
    r2 = run(load_config(CONFIG), root=neuro_run.run_dir.parents[1])
    assert r2.cache_summary["neuro"] == {"hits": 2, "misses": 0}
