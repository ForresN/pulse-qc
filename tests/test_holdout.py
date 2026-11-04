from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pulse_qc.cli import app
from pulse_qc.holdout import choose_threshold, fails, loso, rank_auc, window_table, write_holdout
from pulse_qc.pipeline import load_config, run

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "synthetic.yaml"


def test_rank_auc():
    assert rank_auc(np.array([1, 2, 3, 4]), np.array([False, False, True, True])) == 1.0
    assert rank_auc(np.array([4, 3, 2, 1]), np.array([False, False, True, True])) == 0.0
    assert rank_auc(np.array([1, 2, 3, 4]), np.array([True, False, True, False])) == 0.25
    assert np.isnan(rank_auc(np.array([1, 2]), np.array([True, True])))


def test_choose_threshold_direction_and_ties():
    v = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    bad = np.array([True, True, True, False, False, False])
    assert choose_threshold(v, bad, np.array([0.25, 0.5, 0.75]), "below") == 0.5
    # "above" fails high values: perfect separation at 0.5 too
    bad_hi = ~bad
    assert choose_threshold(v, bad_hi, np.array([0.25, 0.5, 0.75]), "above") == 0.5
    # a tie between 0.35 and 0.5 (both perfect) goes to whichever flags fewer: identical here,
    # so the first wins
    assert choose_threshold(v, bad, np.array([0.35, 0.5]), "below") == 0.35
    assert fails(v, 0.5, "below").tolist() == [True, True, True, False, False, False]


def test_window_table_marks_bad_windows():
    wq = pd.DataFrame(
        {
            "record": ["r"] * 2,
            "window": [0, 1],
            "periodicity": [0.8, 0.2],
            "subject": ["a", "a"],
            "activity": ["x", "x"],
        }
    )
    bc = pd.DataFrame(
        {
            "record": ["r"] * 4,
            "window": [0, 1, 0, 1],
            "detector": list("ppqq"),
            "tp": [10, 5, 10, 6],
            "fn": [0, 5, 0, 4],
            "fp": [0, 1, 0, 0],
        }
    )
    t = window_table(wq, bc)
    assert t["bad"].tolist() == [False, True]
    assert t["mean_f1"].iloc[0] == 1.0


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("root")
    cfg = load_config(CONFIG)
    return run(cfg, root=root)


def test_loso_on_synthetic(synthetic_run):
    r = synthetic_run
    assert set(r.window_qc["subject"]) == {"a", "b"}
    # the synthetic accelerometer shakes in the second half of the motion_b record only
    mb = r.window_qc[r.window_qc["record"] == "synthetic/motion_b:signal"].set_index("window")
    assert (mb.loc[:2, "motion"] < 0.05).all() and (mb.loc[3:, "motion"] > 0.1).all()
    others = r.window_qc[r.window_qc["record"] != "synthetic/motion_b:signal"]
    assert (others["motion"] < 0.05).all()
    res = loso(r.window_qc, r.beat_counts, "periodicity")
    assert sorted(res.folds["held_out_subject"]) == ["a", "b"]
    assert len(res.gated_windows) == len(r.window_qc)
    assert {"all", "passed"} <= set(res.summary["gating"])
    pooled = res.summary[(res.summary["activity"] == "all") & (res.summary["gating"] == "all")]
    base = r.summary[(r.summary["activity"] == "all") & (r.summary["gating"] == "all")]
    # ungated held-out scores are the run's own ungated scores: same beats, same detections
    assert np.allclose(
        pooled.sort_values("detector")["f1"].values, base.sort_values("detector")["f1"].values
    )


def test_loso_rejects_unknown_rule(synthetic_run):
    r = synthetic_run
    with pytest.raises(KeyError):
        loso(r.window_qc, r.beat_counts, "clipping")


def test_write_holdout_and_cli(synthetic_run):
    r = synthetic_run
    out = write_holdout(r.run_dir, ["periodicity", "motion"])
    md = out.read_text()
    assert "Per fold" in md and "held_out_subject" in md
    assert "Which rule explains the errors better?" in md
    assert (r.run_dir / "holdout_comparison.csv").exists()
    assert (r.run_dir / "fig_f1_heldout_motion.png").exists()
    assert (r.run_dir / "holdout_periodicity_folds.csv").exists()
    res = CliRunner().invoke(app, ["holdout", str(r.run_dir)])
    assert res.exit_code == 0, res.output
    assert "Pooled held-out scores" in res.output
