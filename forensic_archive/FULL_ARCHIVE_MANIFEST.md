# Full cross-machine archive manifest

- Archive status: `ARCHIVAL_ONLY_NO_SCIENTIFIC_CODE_CHANGE`
- Branch: `research/h2-clean-repro-anchor-cir-v1`
- Pre-archive HEAD: `930d79dfaa5d3875f15a4967f0f795f7488d0e6d`
- Post-archive HEAD: `<filled after archive commit>`
- Source timestamp: `2026-09-04T16:37:10.101113+07:00`

## Separation of scientific and forensic artifacts

- Eligible seed 0: discovery/factorial evidence; `FINAL=A` under the preregistered rule.
- Seed 1 and Seed 2: forensic-only invalid confirmatory trajectories; no target metrics.
- `ANCHOR_REPLICATION_SUPPORT=NOT_CONFIRMED`.

## Counts and parity

- Eligible seed-0 checkpoints: `9`; SHA verification: `PASS`.
- Seed 1: `34` files, `29` checkpoints, `7323790251` bytes.
- Seed 2: `34` files, `29` checkpoints, `7323788587` bytes.
- Total forensic checkpoint payload: `14647578838` bytes.
- Forensic checkpoint byte parity: `PASS`.

Full per-file inventories and every checkpoint SHA256 are in:

- [`seed1_files.csv`](manifests/seed1_files.csv)
- [`seed2_files.csv`](manifests/seed2_files.csv)
- [`checkpoint_sha256.csv`](manifests/checkpoint_sha256.csv)
- [`context_inventory.csv`](manifests/context_inventory.csv)

## Scientific identity

- CLIP SHA256: `3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02`
- VisA dataset manifest SHA256: `468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842`
- Scientific code fingerprint: `31167af5ee3dfff80b74af1e9ee0da4ecc475d2e`
- Replication implementation SHA: `67888aad3eba2e7d2eecf90bfb8ba132c2c137aa`
- Base H2 commit: `e03966997d4cecfd985943a4053a93e1e40197ec`
- CIR reference commit: `9cc0ad4cc6b34e34a8c15e74df881866516b3181`

## Environment and restore

Environment files are under [`machine_env/`](machine_env/). The exact Medical evaluator is restored using [`MEDICAL_EVALUATOR_HANDOFF.md`](MEDICAL_EVALUATOR_HANDOFF.md); raw datasets are excluded and documented in [`DATASET_RESTORE.md`](DATASET_RESTORE.md).

## Required next scientific action

`Numerical-stability forensics; no target evaluation and no redesign yet.`
