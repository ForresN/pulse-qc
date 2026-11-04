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
- Thresholds for `periodicity` and `motion` in `configs/ptt-ppg.yaml` are the leave-one-subject-out
  consensus from `pulse-qc holdout`; the README quotes the held-out numbers, `report.md` is
  in-sample. Rerun `holdout` after touching either rule. The motion candidate grid is log-spaced
  because walking and running differ by an order of magnitude in g.
- CHARIS is streamed as 2 h segments (`segment:` in the config); no whole-file checksum is
  possible for a segment, so provenance records the SHA-256 of the cached samples.
