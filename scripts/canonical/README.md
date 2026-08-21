# Canonical Phase2B + SABRA execution package

This package is a reproducible, stage-gated wrapper around the canonical
research entry points.  The scientific implementation identity is pinned to
`4aa9b465ddeb072e9218b74982306d6324c62375`; the workflow-package identity is
derived from the current repository `HEAD` at runtime.  The guard accepts
workflow-only commits layered above the scientific SHA and does not duplicate
model or metric implementation.

## Required environment

The scripts use these defaults, which can be overridden explicitly:

```bash
export PYTHON=/home/ai4/ENTER/envs/torchhuy/bin/python
export CLIP_ASSET=/home/ai4/.cache/clip/ViT-L-14-336px.pt
export VISA_ROOT=/home/ai4/caohuy/data/VisA_20220922
export RUN_ROOT=/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0
export MVTEC_ROOT=/path/to/mvtec_anomaly_detection
export MEDICAL_ROOT=/path/to/medical_root
```

`MVTEC_ROOT` is required for checkpoint selection and lambda selection.
`MEDICAL_ROOT` is required only for the final Medical stage.  The same values
can be supplied as `--mvtec-root`, `--medical-root`, and `--run-root` options.
The scripts require the scientific SHA to be an ancestor of `HEAD` and reject
any committed path changed since it unless it is under `scripts/canonical/**`
or `tests/test_canonical_exporter.py`.  Tracked working-tree changes are
rejected by default; the known `.codebase-memory/**` generated files remain
excluded.  `ALLOW_DIRTY_CODE=1` is an explicit emergency override, prints a
loud warning, and marks scientific identity verification false.

## Recommended first scientific execution

Set the development/final roots explicitly, then run and inspect each gate in
order.  Do not use `all` as the first real execution path.

```bash
export MVTEC_ROOT=/actual/path/to/MVTec
export MEDICAL_ROOT=/actual/path/to/medical
export RUN_ROOT=/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0

./scripts/canonical/run_pipeline.sh preflight
./scripts/canonical/run_pipeline.sh train
./scripts/canonical/run_pipeline.sh select
# STOP AND INSPECT: $RUN_ROOT/phase2b_selection/phase2b_selection.json
./scripts/canonical/run_pipeline.sh fit-sabra
./scripts/canonical/run_pipeline.sh lambda
# STOP AND INSPECT: $RUN_ROOT/sabra_lambda/SABRA_FREEZE.json
./scripts/canonical/run_pipeline.sh medical
./scripts/canonical/run_pipeline.sh export
```

## Preflight

```bash
./scripts/canonical/run_pipeline.sh preflight
```

Preflight checks the scientific SHA lineage, current workflow-package SHA,
Python/Torch/CUDA, GPU, hydrated CLIP asset, VisA, and the canonical config.
It records both provenance identities plus `scientific_code_verified=true`.
If `MVTEC_ROOT` is set it invokes the
existing read-only MVTec adapter preflight.  `MEDICAL_ROOT`, when set, is only
checked as a directory; no Medical samples are opened.  The real preflight
writes `manifests/preflight.json` without Medical statistics.

## Phase2B 20e training

```bash
./scripts/canonical/run_pipeline.sh train
```

The wrapper calls `train.py` for exactly 20 epochs with micro-batch 6,
accumulation 1, FP32, workers 4, prefetch 2, pinned memory, persistent
workers, and the canonical code SHA.  It requires `adapter_10.pth` through
`adapter_20.pth` (the six preregistered candidate epochs) after completion.

To resume explicitly from a compatible checkpoint:

```bash
RESUME_CHECKPOINT="$RUN_ROOT/phase2b/last.pth" \
  ./scripts/canonical/run_pipeline.sh train
```

The wrapper validates resume compatibility through `train.py` before passing
`--resume` once.  It never auto-selects a resume checkpoint.  A completed
`run_manifest.json` is not overwritten unless `FORCE_RERUN=1` is explicit.

## MVTec checkpoint selection

```bash
./scripts/canonical/run_pipeline.sh select
```

This calls `select_phase2b_checkpoint.py` on E10/E12/E14/E16/E18/E20 using
the exact evaluator and preregistered score
`.35*pAUROC+.35*pAP+.15*iAUROC+.15*iAP`.  The output is required to be
`phase2b_selection.json` with status `FROZEN`, an allowed epoch, and a
checkpoint SHA256 matching the selected file.

## SABRA source fitting

```bash
./scripts/canonical/run_pipeline.sh fit-sabra
```

This calls `calibrate_sabra.py fit-source` on VisA only, with the frozen E*,
backend `fast`, and Batch-6 data loading.  The stage verifies finalized
GT-free and GT-target manifests, exact Trust/Need feature orders, the
`P90(abs(native_margin))` scale, zero Medical reads, and a lambda-free source
calibration artifact.

## MVTec lambda selection

```bash
./scripts/canonical/run_pipeline.sh lambda
```

This calls `calibrate_sabra.py select-lambda` with backend `fast`, lambda
chunk size 8, and the canonical coarse/refined grids.  It requires a frozen
SABRA artifact whose checkpoint SHA equals E*, whose authority is `T*N`, whose
direction is positive abnormal-only, whose normal delta is zero, and whose
Medical flag is false.

## Medical final zero-shot

```bash
./scripts/canonical/run_pipeline.sh medical
```

The strongest guard runs first: selection and SABRA must be frozen, checkpoint
hashes must match, backend must be `fast`, and `medical_seen` must be false.
For every authoritative Medical dataset name from `dataset.info`, the wrapper
calls `test.py --method compare` once with Batch-6 inference, exact metrics,
pixel stride 1, and the same frozen E*/SABRA artifacts.  It never refits or
changes lambda between datasets.  Undefined Medical image metrics remain
null/N/A in the evaluator outputs.

## Result export

```bash
./scripts/canonical/run_pipeline.sh export
```

`60_export_results.py` reads only completed JSON/CSV artifacts and writes the
deterministic final summary, deltas, and provenance.  It never loads a model
or opens a dataset.  An optional publication-only reference can be supplied:

```bash
./scripts/canonical/run_pipeline.sh export \
  --acdclip-reference-json /path/to/acdclip_reference.json
```

Use `--force` only for an explicit deterministic replacement of an existing
final export.  Null metrics remain null; they are never converted to zero.

## Output tree

```text
$RUN_ROOT/
├── manifests/preflight.json
├── logs/
├── phase2b/
│   ├── checkpoints/adapter_{10,12,14,16,18,20}.pth
│   ├── last.pth
│   ├── config_resolved.json
│   └── run_manifest.json
├── phase2b_selection/
│   ├── phase2b_selection.json
│   └── phase2b_selection_metrics.csv
├── sabra_source/
│   ├── gt_free_cache/
│   ├── GT_FREE_MANIFEST.json
│   ├── GT_TARGET_MANIFEST.json
│   └── sabra_source_calibration.json
├── sabra_lambda/
│   ├── lambda_selection.csv
│   ├── lambda_runtime.json
│   └── SABRA_FREEZE.json
├── medical/<dataset>/
│   ├── metrics.json
│   └── per_class_metrics.csv
└── final/
    ├── summary.csv
    ├── summary.json
    ├── deltas.csv
    └── provenance.json
```

## Scientific data roles

* VisA is the Phase2B training and SABRA source-calibration dataset.
* MVTecAD is development data for checkpoint and global lambda selection.
* Medical is the final unseen zero-shot test only.

## Dry-run validation

Every shell stage supports `DRY_RUN=1` (or `--dry-run`).  It validates
prerequisites and prints the exact command without executing training,
evaluation, calibration, or lambda selection.  The `export` dry run validates
completed artifacts but writes no final files.  `all` preserves every stage
gate and stops at the first failure.
