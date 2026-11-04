import numpy as np
import pytest

from pulse_qc.detect import DETECTORS, preprocess, run_detector
from pulse_qc.evaluate import estimate_lag, match_beats, per_window_counts, scores, summarise
from pulse_qc.synth import inject_nan, synthetic_ppg
from pulse_qc.windows import make_windows


@pytest.mark.parametrize("name", sorted(DETECTORS))
def test_detectors_find_synthetic_beats(clean, name):
    y = preprocess(clean.signal, clean.fs)
    det = run_detector(name, y, clean.fs)
    m = match_beats(clean.beat_samples, det, clean.fs, tolerance_s=0.05, lag_s=0.0)
    s = scores(len(m.tp_ref), len(m.fn_ref), len(m.fp_det))
    assert s["f1"] > 0.95, (name, s)
    assert abs(m.errors_s).max() <= 0.05


def test_preprocess_handles_nan_and_inversion(clean):
    x = inject_nan(clean.signal, 100, 200)
    y = preprocess(x, clean.fs)
    assert np.isfinite(y).all()
    assert np.allclose(preprocess(x, clean.fs, invert=True), -y)
    assert np.all(preprocess(np.full(100, np.nan), 125.0) == 0)


def test_unknown_detector():
    with pytest.raises(KeyError):
        run_detector("nope", np.zeros(10), 10.0)


def test_lag_is_estimated_and_removed():
    fs = 100.0
    ref = np.arange(100, 5000, 80)  # 0.8 s beats
    det = ref + 25  # 250 ms pulse-arrival delay
    assert estimate_lag(ref, det, fs, (0.0, 0.6)) == pytest.approx(0.25)
    m = match_beats(ref, det, fs, tolerance_s=0.05)
    assert m.lag_s == pytest.approx(0.25)
    assert len(m.tp_ref) == len(ref) and len(m.fp_det) == 0 and len(m.fn_ref) == 0
    # without lag correction the same detections are all misses
    m0 = match_beats(ref, det, fs, tolerance_s=0.05, lag_s=0.0)
    assert len(m0.tp_ref) == 0


def test_matching_is_one_to_one_and_tolerance_bounded():
    fs = 100.0
    ref = np.array([100, 200, 300, 400])
    det = np.array([101, 103, 300, 450])  # duplicate near 100, miss 200, 450 is a FP (>tol)
    m = match_beats(ref, det, fs, tolerance_s=0.05, lag_s=0.0)
    assert m.tp_ref.tolist() == [0, 2]
    assert m.tp_det.tolist() == [0, 2]  # closest duplicate wins
    assert m.fn_ref.tolist() == [1, 3]
    assert m.fp_det.tolist() == [1, 3]


def test_per_window_counts_and_summary():
    fs = 100.0
    ref = np.array([50, 150, 250, 350, 1050])  # last beat is in the dropped tail
    det = np.array([50, 150, 260, 500])  # 260 misses at 5 samples tol? no: tol 0.1s=10 -> hit
    windows = make_windows(1000, fs, 5.0)  # two windows: [0,500), [500,1000)
    m = match_beats(ref, det, fs, tolerance_s=0.1, lag_s=0.0)
    c = per_window_counts(m, ref, det, fs, windows)
    assert c.set_index("window").loc[0].tolist() == [3, 1, 0]  # tp, fn, fp
    assert c.set_index("window").loc[1].tolist() == [0, 0, 1]
    assert c.set_index("window").loc[-1].tolist() == [0, 1, 0]
    c["passed"] = c["window"] == 0
    s = summarise(c[c["window"] >= 0], ["passed"]).set_index("passed")
    assert s.loc[True, "sensitivity"] == pytest.approx(0.75)
    assert s.loc[True, "ppv"] == pytest.approx(1.0)
    assert np.isnan(s.loc[False, "sensitivity"])
    assert s.loc[False, "ppv"] == 0.0


def test_scores_edge_cases():
    assert np.isnan(scores(0, 0, 0)["f1"])
    assert scores(10, 0, 0) == {"sensitivity": 1.0, "ppv": 1.0, "f1": 1.0}


def test_synthetic_has_expected_beat_count():
    s = synthetic_ppg(duration_s=30, fs=100, hr_bpm=60, hr_jitter=0.0)
    assert 28 <= len(s.beat_samples) <= 30
