import numpy as np
import pytest

from pulse_qc.qc.rules import (
    RULES,
    _longest_run,
    clipping,
    flatline,
    nan_dropout,
    plausible_range,
    sampling_gap,
)
from pulse_qc.synth import inject_clipping, inject_flatline, inject_nan

FS = 125.0


def test_longest_run():
    assert _longest_run(np.array([], dtype=bool)) == 0
    assert _longest_run(np.array([False, False])) == 0
    assert _longest_run(np.array([True, True, False, True])) == 2
    assert _longest_run(np.array([False, True, True, True])) == 3


def test_registry_lists_every_rule():
    assert set(RULES) == {"nan_dropout", "sampling_gap", "flatline", "clipping", "plausible_range"}


class TestNanDropout:
    def test_clean_passes(self, x):
        r = nan_dropout(x, FS, max_frac=0.01)
        assert r.passed and r.value == 0.0 and r.reason == ""

    def test_fails_above_threshold(self, x):
        y = inject_nan(x, 0, len(x) // 10)  # 10 % missing
        r = nan_dropout(y, FS, max_frac=0.05)
        assert not r.passed
        assert r.value == pytest.approx(0.1, abs=1e-3)
        assert "NaN" in r.reason

    def test_boundary_is_inclusive(self):
        y = np.ones(100)
        y[:5] = np.nan
        assert nan_dropout(y, FS, max_frac=0.05).passed
        assert not nan_dropout(y, FS, max_frac=0.049).passed

    def test_empty_window_fails(self):
        assert not nan_dropout(np.array([]), FS).passed


class TestSamplingGap:
    def test_scattered_nans_pass_but_fail_dropout(self, x):
        y = x.copy()
        y[::50] = np.nan  # 2 % scattered, each gap one sample
        assert sampling_gap(y, FS, max_gap_s=0.1).passed
        assert not nan_dropout(y, FS, max_frac=0.01).passed

    def test_one_long_hole_fails(self, x):
        y = inject_nan(x, 1000, 1000 + int(0.5 * FS))
        r = sampling_gap(y, FS, max_gap_s=0.2)
        assert not r.passed
        assert r.value == pytest.approx(0.5, abs=1 / FS)


class TestFlatline:
    def test_clean_passes(self, x):
        assert flatline(x, FS, max_flat_s=1.0).passed

    def test_flat_segment_fails(self, x):
        y = inject_flatline(x, 500, 500 + int(2 * FS))
        r = flatline(y, FS, max_flat_s=1.0)
        assert not r.passed
        assert r.value == pytest.approx(2.0, abs=2 / FS)

    def test_nan_run_is_not_flat(self, x):
        y = inject_nan(x, 500, 500 + int(2 * FS))
        assert flatline(y, FS, max_flat_s=1.0).passed

    def test_eps_tolerates_quantisation_noise(self):
        y = np.ones(500) + np.random.default_rng(0).integers(0, 2, 500) * 1e-6
        assert flatline(y, FS, max_flat_s=1.0, eps=0.0).passed  # toggling 1 LSB is "not flat"
        assert not flatline(y, FS, max_flat_s=1.0, eps=1e-5).passed


class TestClipping:
    def test_clean_passes(self, x):
        assert clipping(x, FS, max_frac=0.02).passed

    def test_ceiling_fails(self, x):
        y = inject_clipping(x, np.percentile(x, 80))
        r = clipping(y, FS, max_frac=0.02)
        assert not r.passed
        assert r.value > 0.1

    def test_single_sample_at_max_is_not_clipping(self):
        y = np.sin(np.linspace(0, 200, 1000))
        assert clipping(y, FS, max_frac=0.0, min_run=3).passed

    def test_explicit_rails(self):
        y = np.sin(np.linspace(0, 200, 1000))
        assert clipping(y, FS, max_frac=0.0, lo=-2, hi=2).passed
        yy = np.clip(y, -0.5, 0.5)
        assert not clipping(yy, FS, max_frac=0.0, lo=-0.5, hi=0.5).passed

    def test_constant_window_defers_to_flatline(self):
        assert clipping(np.ones(100), FS).passed


class TestPlausibleRange:
    def test_inside_passes(self, x):
        assert plausible_range(x, FS, lo=0, hi=10).passed

    def test_outside_fails(self, x):
        r = plausible_range(x, FS, lo=0, hi=1.5)
        assert not r.passed and r.value > 0

    def test_ignores_nan(self):
        y = np.array([1.0, np.nan, 2.0])
        assert plausible_range(y, FS, lo=0, hi=3).passed

    def test_tolerates_fraction(self):
        y = np.r_[np.ones(99), 100.0]
        assert not plausible_range(y, FS, lo=0, hi=10).passed
        assert plausible_range(y, FS, lo=0, hi=10, max_frac=0.02).passed
