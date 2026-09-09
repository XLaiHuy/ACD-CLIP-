# R3 Anchor + E1 Optimization Status

Status: `TTA_F_LOCKED_TARGET_REPLAY_PASS`

This branch is the reproducible R3 workspace for the source-only E1 anchor
optimization study. The dataset and runtime gates are recorded before any
new training or target-guided decision.

## Completed gates

- Dataset archives downloaded, ZIP-tested, staged, validated, promoted, and
  recorded in `/workspace/R3_DATA_PROVISIONING.md`.
- Repo data paths are symlinks to `/workspace/datasets`; dataset contents are
  ignored and are not intended for Git tracking.
- Canonical CLIP, shared E1, and A15 checkpoint SHA256 values verified.
- CUDA smoke passed on the installed Blackwell-compatible environment:
  `torch 2.7.1+cu128`, `torchvision 0.22.1+cu128`, CUDA `12.8`.
- A15 model forward smoke passed at `518x518` with finite segmentation and
  detection outputs.

## Completed

- Full six-dataset Medical migration parity replay with the frozen
  `benchmark_exact`, `pixel_stride=1`, `cls_only` image-score contract.
- Full MVTec migration parity replay with the established industrial image
  score is complete; metrics and provenance are in `results/r3_migration_parity/`.
- Source-only TTA gate complete; `TTA-F` is locked by source Pixel AP.
- Locked target TTA-F replay complete on Medical and all 15 MVTec categories;
  target results are recorded as a post-lock audit only.

## In progress

- Source-only alpha screen is the next gate. No target labels or target metrics
  may affect alpha/rho/lambda selection.

## Frozen anchors

- Base H2 commit: `88abf174dea1744b8977b934142a9809aff6e96f`
- Shared E1 SHA256:
  `7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35`
- A15 SHA256:
  `727dc4813db1ef0c5a6db3a4cf15e916ad1413dcf629913358419d2734edecfe`
- CLIP SHA256:
  `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`

No target-label selection, tuning, or new training has been performed on this
branch yet. The last valid checkpoint remains the canonical A15; the
authoritative shared E1 is unchanged. The locked target replay artifact is
`results/r3_anchor_e1_optimization/TTA_TARGET_EVAL.json` and the raw run was
`/tmp/r3_locked_tta_target_20260909_b16/metrics.json`.
