# Repo Layout

Root keeps source code and shared entrypoints:
- `train.py`, `test.py`, `utils.py`
- `model/`, `dataset/`

Organized folders:
- `runs/baseline/`: baseline smoke/local baseline artifacts
- `runs/phase1/`: Phase1 numbered runs
- `runs/phase1_ablation/`: Phase1 ablations, e.g. `text_adapt_weight=0.10`
- `runs/phase1_dora_test/`: abandoned DoRA tests, kept as historical Phase1 tests
- `runs/smoke/`: smoke/debug artifacts
- `scripts/baseline/`: baseline train scripts
- `scripts/phase1/`: Phase1 train/test scripts
- `scripts/phase1_ablation/`: current ablation wrappers
- `scripts/common/`: shared test wrappers
- `scripts/diagnostics/`: diagnostic scripts
- `docs/`: notes/plans/debug docs

Compatibility symlinks are kept at root for the most-used scripts and current
best run, so old commands continue to work.
