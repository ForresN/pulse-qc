# Adding a detector or a rule, and running a comparison

You do not need to edit anything under `src/` to try a new beat detector or a new
quality rule against the ones already here. Everything below works from a script, a
notebook, or a YAML config.

## 1. Run an evaluation from Python

```python
import pulse_qc

result = pulse_qc.evaluate("configs/ptt-ppg.yaml")      # cached: a second call is instant
print(result.gating_table())                            # F1 all / QC-passed / QC-failed, per detector and activity
result.scores("elgendi", gating="passed")               # {"sensitivity": ..., "ppv": ..., "f1": ..., ...}
result.provenance["git"], result.provenance["packages"] # what produced these numbers
result.holdout()                                        # leave-one-subject-out thresholds -> holdout.md
```

`evaluate` accepts a path or a dict, and two overrides so that a comparison is one call:

```python
result = pulse_qc.evaluate(
    "configs/ptt-ppg.yaml",
    detectors=["elgendi", "msptd", {"name": "prominence", "prominence_sd": 0.8}],
    rules=[{"name": "nan_dropout", "max_frac": 0.01}, {"name": "periodicity", "min_acf": 0.6}],
    name="my-variant",          # separate run directory, same cache
)
```

Every result is also written to `runs/<name>-<confighash>/` with `provenance.json`,
exactly as the CLI does. `pulse_qc.Result.load(run_dir)` reopens one.

## 2. Add a detector

A detector is a function `fn(y, fs, **params) -> np.ndarray` of peak sample indices.
`y` is the preprocessed signal: NaN-free, band-passed, peaks pointing up. Register it by
name and it is available to `evaluate`, to YAML configs and to the CLI:

```python
import numpy as np
import pulse_qc
from scipy import signal as ss

@pulse_qc.register_detector("my_peaks")
def my_peaks(y: np.ndarray, fs: float, *, min_rr_s: float = 0.3) -> np.ndarray:
    peaks, _ = ss.find_peaks(y, distance=int(min_rr_s * fs), height=np.percentile(y, 60))
    return peaks.astype(np.int64)

result = pulse_qc.evaluate("configs/ptt-ppg.yaml", detectors=["elgendi", "my_peaks"])
print(result.gating_table())
```

The detector's cache key includes its own source code, so editing the function reruns it
and nothing else. Keep it linear in the record length: records are 250k samples and more.

## 3. Add a rule

A rule is a function `fn(x, fs, *, **params) -> pulse_qc.RuleResult` of one raw window.
Return the value the decision was made on and a reason when it fails. Declare an `aux`
keyword to receive auxiliary channels (such as an accelerometer) loaded via
`aux_channels:` in the config.

```python
@pulse_qc.register_rule("amplitude_floor", threshold=("min_p2p", "below"))
def amplitude_floor(x, fs, *, min_p2p=0.0):
    finite = x[np.isfinite(x)]
    p2p = float(finite.max() - finite.min()) if finite.size else 0.0
    ok = p2p >= min_p2p
    return pulse_qc.RuleResult(ok, p2p, "" if ok else f"peak-to-peak {p2p:.3g} < {min_p2p:.3g}")
```

`threshold=(param, "below"|"above")` tells `pulse-qc holdout` that the rule is a single
threshold on its value and which direction fails, so it can choose that threshold on
held-out subjects like the built-in `periodicity` and `motion` rules.

## 4. Use a plugin from YAML or the CLI

Put the registrations in a module on the Python path and list it under `plugins:`; the
module is imported before the run, so its decorators execute:

```yaml
plugins: [my_detectors]
detectors:
  - {name: elgendi}
  - {name: my_peaks, min_rr_s: 0.35}
```

```bash
uv run pulse-qc run configs/my-variant.yaml
uv run pulse-qc compare configs/ptt-ppg.yaml configs/my-variant.yaml
```

`examples/custom_detector.py` is a complete, runnable version of all of the above on the
synthetic fixtures.

## What the harness does for you

- Downloads and checksum-verifies only the records named in the config; never commits data.
- Runs the quality rules on every window and attributes each annotated beat to its window,
  so your detector is scored over all windows, QC-passed windows and QC-failed windows.
- Removes one pulse-arrival lag per record before matching (the annotations are ECG
  R-peaks), and matches one-to-one within the configured tolerance.
- Caches every stage; adding your detector reruns only your detector and its scoring.
- Records git SHA, package versions, config hash, record IDs and file hashes with every
  result.
