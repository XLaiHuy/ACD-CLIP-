#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
conda run --no-capture-output -n torchhuy python phase2c_train.py \
  --condition P --save-path runs/phase2c_bf16/P_pcgrad_seed42 \
  --hybrid-alpha-max 0.20 --train-manifest splits/visa_train_seed42.csv \
  --val-manifest splits/visa_val_seed42.csv --split-metadata splits/visa_split_seed42_metadata.json \
  --batch-size 6 --num-workers 6 --diagnostic-batch-size 1 --bf16 "$@"
