#!/usr/bin/env bash
set -euo pipefail

# Mainbase reproduction check: same architecture and LR schedule, but no AMP.
# Use this only after confirming the current checkpoint is the likely issue.

SAVE_PATH="${SAVE_PATH:-test_train_main_base_noamp_n3}"
BATCH_SIZE="${BATCH_SIZE:-6}"
NUM_WORKERS="${NUM_WORKERS:-6}"

conda run --no-capture-output -n torchhuy python train.py \
  --dataset VisA \
  --n_groups 3 \
  --batch_size "${BATCH_SIZE}" \
  --epoch 20 \
  --grad_checkpointing \
  --num_workers "${NUM_WORKERS}" \
  --save_path "${SAVE_PATH}"
