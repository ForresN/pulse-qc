"""Choose a rule threshold on held-out subjects and score gating honestly.

A threshold picked by looking at the evaluation data gives optimistic gated
scores. This module runs leave-one-subject-out (LOSO): for each subject, the
threshold is chosen on every *other* subject, then applied to the held-out
one. The held-out windows from all folds are pooled and scored exactly as the
main report scores them (all / passed / failed windows), so the "passed" column
is what gating of this kind achieves on subjects it has never seen.

The threshold objective is Youden's J for flagging "bad" windows, where a
window is bad when its mean beat-detection F1 across the run's detectors is
below ``bad_f1``. This needs only the per-window tables a run already saves:
the rule's measured value is stored per window regardless of the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Rules whose pass/fail is a single threshold on the stored value:
#: name -> (parameter name, direction in which the rule FAILS).
THRESHOLD_RULES: dict[str, tuple[str, str]] = {
    "periodicity": ("min_acf", "below"),
    "motion": ("max_std_g", "above"),
}

DEFAULT_CANDIDATES: dict[str, np.ndarray] = {
    "periodicity": np.round(np.arange(0.30, 0.86, 0.05), 2),
    "motion": np.round(np.geomspace(0.005, 3.0, 40), 4),
}


def window_table(
    window_qc: pd.DataFrame, beat_counts: pd.DataFrame, bad_f1: float = 0.9
) -> pd.DataFrame:
    """One row per (record, window) with the rule values, a ``bad`` label and the per-detector
    counts kept in long form alongside (``counts``)."""
    bc = beat_counts[beat_counts["window"] >= 0].copy()
    bc["f1w"] = 2 * bc["tp"] / (2 * bc["tp"] + bc["fn"] + bc["fp"]).replace(0, np.nan)
    per_window = bc.groupby(["record", "window"])["f1w"].mean().rename("mean_f1").reset_index()
    tbl = window_qc.merge(per_window, on=["record", "window"], how="inner")
    tbl["bad"] = tbl["mean_f1"] < bad_f1
    return tbl


def rank_auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUC for ``label`` (True = positive) given a score where higher means more positive."""
    from scipy.stats import rankdata

    label = np.asarray(label, dtype=bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fails(value: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    return value < threshold if direction == "below" else value > threshold


def choose_threshold(
    values: np.ndarray, bad: np.ndarray, candidates: np.ndarray, direction: str
) -> float:
    """Candidate that maximises Youden's J (sensitivity + specificity - 1) for flagging bad windows.
    Ties go to the candidate that flags fewer windows."""
    bad = np.asarray(bad, dtype=bool)
    best, best_j, best_flag = float(candidates[0]), -np.inf, np.inf
    for c in candidates:
        f = fails(values, c, direction)
        sens = f[bad].mean() if bad.any() else 0.0
        spec = (~f[~bad]).mean() if (~bad).any() else 0.0
        j = sens + spec - 1
        if j > best_j + 1e-12 or (abs(j - best_j) <= 1e-12 and f.sum() < best_flag):
            best, best_j, best_flag = float(c), j, f.sum()
    return best


@dataclass
class HoldoutResult:
    rule: str
    direction: str
    folds: pd.DataFrame  # one row per held-out subject
    auc: pd.DataFrame  # AUC of the rule value for flagging bad windows, overall and by activity
    summary: pd.DataFrame  # pooled held-out scores: detector x activity x gating
    gated_windows: pd.DataFrame  # (record, window, passed_heldout)


def loso(
    window_qc: pd.DataFrame,
    beat_counts: pd.DataFrame,
    rule: str,
    candidates: np.ndarray | None = None,
    bad_f1: float = 0.9,
) -> HoldoutResult:
    if rule not in THRESHOLD_RULES:
        raise KeyError(f"{rule!r} is not a single-threshold rule; known: {sorted(THRESHOLD_RULES)}")
    if rule not in window_qc.columns:
        raise KeyError(f"rule {rule!r} was not run: no column in window_qc")
    _, direction = THRESHOLD_RULES[rule]
    candidates = DEFAULT_CANDIDATES[rule] if candidates is None else np.asarray(candidates)
    tbl = window_table(window_qc, beat_counts, bad_f1)
    if "subject" not in tbl.columns or tbl["subject"].isna().all():
        raise ValueError("window_qc has no subject column; rerun with a subject_pattern")
    subjects = sorted(tbl["subject"].dropna().unique())
    if len(subjects) < 2:
        raise ValueError("leave-one-subject-out needs at least two subjects")

    sign = -1.0 if direction == "below" else 1.0
    auc_rows = [
        {
            "activity": "all",
            "auc": rank_auc(sign * tbl[rule].values, tbl["bad"].values),
            "n_windows": len(tbl),
            "n_bad": int(tbl["bad"].sum()),
        }
    ]
    for act, sub in tbl.groupby("activity"):
        auc_rows.append(
            {
                "activity": act,
                "auc": rank_auc(sign * sub[rule].values, sub["bad"].values),
                "n_windows": len(sub),
                "n_bad": int(sub["bad"].sum()),
            }
        )

    fold_rows, gated = [], []
    for held in subjects:
        train = tbl[tbl["subject"] != held]
        test = tbl[tbl["subject"] == held]
        thr = choose_threshold(train[rule].values, train["bad"].values, candidates, direction)
        f = fails(test[rule].values, thr, direction)
        fold_rows.append(
            {
                "held_out_subject": held,
                "threshold": thr,
                "n_windows": len(test),
                "flagged": int(f.sum()),
                "flagged_frac": float(f.mean()) if len(test) else np.nan,
                "bad_windows": int(test["bad"].sum()),
                "bad_caught": float(f[test["bad"].values].mean()) if test["bad"].any() else np.nan,
                "precision": float(test["bad"].values[f].mean()) if f.any() else np.nan,
            }
        )
        gated.append(
            pd.DataFrame(
                {
                    "record": test["record"].values,
                    "window": test["window"].values,
                    "passed_heldout": ~f,
                }
            )
        )
    gated_df = pd.concat(gated, ignore_index=True)

    bc = beat_counts[beat_counts["window"] >= 0].merge(
        gated_df, on=["record", "window"], how="inner"
    )
    bc = bc.drop(columns=["passed"]).rename(columns={"passed_heldout": "passed"})
    from pulse_qc.pipeline import summarise_counts

    summary = summarise_counts(bc)
    return HoldoutResult(
        rule, direction, pd.DataFrame(fold_rows), pd.DataFrame(auc_rows), summary, gated_df
    )


def write_holdout(run_dir: Path, rules: list[str], bad_f1: float = 0.9) -> Path:
    """Run LOSO for each rule on a saved run and write ``holdout.md`` plus CSVs."""
    from pulse_qc.report import _md_table, f1_bar_figure, gating_table, load_run

    r = load_run(run_dir)
    md = [f"# Held-out threshold selection: `{r['config']['name']}`", ""]
    md.append(
        f"Leave-one-subject-out over {r['window_qc']['subject'].nunique()} subjects. A window is "
        f"'bad' when its mean beat F1 across detectors is below {bad_f1}. The threshold for "
        "each fold maximises Youden's J on the other subjects; the pooled scores below are on "
        "held-out subjects only."
    )
    md.append("")
    results: dict[str, HoldoutResult] = {}
    for rule in rules:
        res = loso(r["window_qc"], r["beat_counts"], rule, bad_f1=bad_f1)
        results[rule] = res
        md.append(f"## `{rule}` (fails when value is {res.direction} the threshold)")
        md.append("")
        md.append("AUC of the raw rule value for flagging bad windows:")
        md.append("")
        md.append(
            _md_table(
                res.auc, ["activity", "auc", "n_windows", "n_bad"], {"n_windows": str, "n_bad": str}
            )
        )
        md.append("")
        md.append("Per fold:")
        md.append("")
        md.append(
            _md_table(
                res.folds,
                list(res.folds.columns),
                {
                    "n_windows": str,
                    "flagged": str,
                    "bad_windows": str,
                    "threshold": lambda v: f"{v:g}",
                },
            )
        )
        md.append("")
        gt = gating_table(res.summary)
        md.append("Pooled held-out scores (gating applied with each fold's own threshold):")
        md.append("")
        md.append(_md_table(gt, list(gt.columns), {"beats (all)": str, "beats (passed)": str}))
        md.append("")
        fig = Path(run_dir) / f"fig_f1_heldout_{rule}.png"
        f1_bar_figure(gt, fig, f"Held-out gating by `{rule}` (leave-one-subject-out thresholds)")
        md.append(f"![held-out F1 for {rule}]({fig.name})")
        md.append("")
        res.folds.to_csv(Path(run_dir) / f"holdout_{rule}_folds.csv", index=False)
        res.summary.to_csv(Path(run_dir) / f"holdout_{rule}_summary.csv", index=False)
        res.auc.to_csv(Path(run_dir) / f"holdout_{rule}_auc.csv", index=False)
    if len(results) > 1:
        md.append("## Which rule explains the errors better?")
        md.append("")
        md.append(
            "Same windows, same beats, same held-out protocol. AUC says how well the raw value "
            "ranks bad windows; the gain and the beats kept say what a held-out threshold buys."
        )
        md.append("")
        rows = []
        for rule, res in results.items():
            gt = gating_table(res.summary)
            for act in list(dict.fromkeys(gt["activity"])):
                sub = gt[gt["activity"] == act]
                a = res.auc.set_index("activity")["auc"].get(act, np.nan)
                rows.append(
                    {
                        "rule": rule,
                        "activity": act,
                        "AUC": a,
                        "F1 (all)": float(sub["F1 (all)"].mean()),
                        "F1 (passed, held-out)": float(sub["F1 (passed)"].mean()),
                        "beats kept": float(sub["beats (passed)"].sum() / sub["beats (all)"].sum())
                        if sub["beats (all)"].sum()
                        else np.nan,
                    }
                )
        cmp_df = pd.DataFrame(rows)
        order = {"all": 0, "sit": 1, "walk": 2, "run": 3}
        cmp_df["_o"] = cmp_df["activity"].map(lambda a: order.get(a, 9))
        cmp_df = cmp_df.sort_values(
            ["activity", "rule"], key=lambda c: c.map(order) if c.name == "activity" else c
        )
        cmp_df = cmp_df.drop(columns="_o")
        md.append(_md_table(cmp_df, list(cmp_df.columns), {"beats kept": lambda v: f"{v:.0%}"}))
        md.append("")
        cmp_df.to_csv(Path(run_dir) / "holdout_comparison.csv", index=False)
    out = Path(run_dir) / "holdout.md"
    out.write_text("\n".join(md))
    return out
