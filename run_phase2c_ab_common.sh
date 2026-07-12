#!/usr/bin/env bash
set -euo pipefail

CONDITION="$1"
SAVE_PATH="$2"
HYBRID_ALPHA_MAX="$3"

CMD=(
  conda run --no-capture-output -n torchhuy python phase2c_train.py
  --condition "$CONDITION"
  --save-path "$SAVE_PATH"
  --hybrid-alpha-max "$HYBRID_ALPHA_MAX"
  --train-manifest splits/visa_train_seed42.csv
  --val-manifest splits/visa_val_seed42.csv
  --split-metadata splits/visa_split_seed42_metadata.json
  --batch-size "${BATCH_SIZE:-6}"
  --num-workers "${NUM_WORKERS:-6}"
  --diagnostic-batch-size "${DIAGNOSTIC_BATCH_SIZE:-1}"
)
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry-run)
fi
if [[ "${AMP:-1}" == "0" ]]; then
  CMD+=(--no-amp)
fi
if [[ "${BF16:-0}" == "1" ]]; then
  CMD+=(--bf16)
fi
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
