# pulse-qc — notes for agents

Data-quality gating and beat-detection evaluation on open PhysioNet pulsatile waveforms.
Read `README.md` first; this file is only what is not obvious from it.

- **Env:** Python 3.12 via mise (`mise.toml`), deps via `uv sync --all-groups`. Never pip-install
  into the system. Run everything with `uv run ...`.
- **Tests:** `uv run pytest` uses synthetic fixtures only and must stay network-free; CI runs it.
  `uv run ruff check src tests && uv run ruff format --check src tests` before committing.
- **Data:** `data/`, `cache/`, `runs/` are git-ignored and must stay that way. PhysioNet data is
  never committed. Figures that the README shows are copied into `docs/` by hand from a run.
- **Cache keys** include the source text of each stage (`cache.code_hash`). Editing a stage's
  code invalidates its cache entries on purpose. Bump `RULES_VERSION` in `qc/rules.py` or
  `DETECT_VERSION` in `detect.py` when semantics change in a way the source hash would not catch.
- **NeuroKit2's `bishop` method is O(n²)** in memory and time; it was tried and dropped for
  the 254k-sample records. `elgendi` and `charlton` (MSPTD) are fine.
- **Raw MAX30101 PPG in PTT-PPG dips at systole**: `preprocess.invert: true` in that config.
  BIDMC `PLETH` is already upright.
- The `periodicity` rule's threshold (0.5) was chosen on the same six PTT-PPG subjects used
  for evaluation. Say so wherever its numbers are quoted.
