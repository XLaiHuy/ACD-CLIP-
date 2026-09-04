# FORENSIC ARCHIVE ONLY

This directory preserves the complete invalid H/A replication state for
cross-machine numerical-stability forensics. It is not a scientific result
selection and must not be used as target-evaluation evidence.

## Scientific label

Seed 1 and Seed 2 are **not**:

- target-evaluation eligible;
- confirmatory evidence;
- publication-selected checkpoints.

Known validity:

- Seed 1 H/A successful global steps: `5410 / 5411`.
- Seed 1 gradient skips: `3 / 2`.
- Seed 2 H/A successful global steps: `5410 / 5411`.
- Seed 2 gradient skips: `3 / 2`.
- Non-finite loss skips: `0` for both H and A in both seeds.
- Final model checkpoints were finite.
- The replication validity failed before target evaluation.

`ANCHOR_REPLICATION_SUPPORT=NOT_CONFIRMED`

Do not write “Anchor failed.” The correct interpretation is that the
replication trajectory was invalid before target evaluation, so robustness of
the positive seed-0 Anchor result remains unconfirmed.

## Contents

- `seed1/h2_ha_replication_e15_seed1/`: complete preserved Seed 1 tree.
- `seed2/h2_ha_replication_e15_seed2/`: complete preserved Seed 2 tree.
- `manifests/seed1_files.csv` and `seed2_files.csv`: all copied files and
  source/archive byte hashes.
- `manifests/checkpoint_sha256.csv`: every eligible and forensic checkpoint
  hash, size, and byte-parity result.
- `manifests/context_inventory.csv`: required research-context inventory.
- `machine_env/`: secret-free source environment snapshots.
- `MEDICAL_EVALUATOR_HANDOFF.md`: exact external evaluator restore path.
- `DATASET_RESTORE.md`: external dataset requirements; no raw data is stored.
- `FULL_ARCHIVE_MANIFEST.json` and `.md`: complete archive provenance.

The eligible seed-0 checkpoints remain only under
`runs/h2_clean_factorial_e20_20260902_ampfix/`; they are not duplicated here.
