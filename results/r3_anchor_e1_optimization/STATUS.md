# R3 Anchor + E1 Optimization Status

Status: `PROVISIONED_MIGRATION_IN_PROGRESS`

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

## In progress

- Full six-dataset Medical migration parity replay with the frozen
  `benchmark_exact`, `pixel_stride=1`, `cls_only` image-score contract.
- Full MVTec migration parity replay is complete; final metrics and
  provenance will be committed under `results/r3_migration_parity/`.

## Frozen anchors

- Base H2 commit: `88abf174dea1744b8977b934142a9809aff6e96f`
- Shared E1 SHA256:
  `7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35`
- A15 SHA256:
  `727dc4813db1ef0c5a6db3a4cf15e916ad1413dcf629913358419d2734edecfe`
- CLIP SHA256:
  `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`

No target-label selection, tuning, or new training has been performed on
this branch yet.
