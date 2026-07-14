#!/usr/bin/env bash
# Phase2C condition P_LoRA_only — local launcher
#
# Assumes the conda/virtualenv environment is already activated and that
# python resolves to the correct interpreter.  Does not hardcode conda run.
#
# Usage:
#   bash run_phase2c_PL_pcgrad_lora_only_seed42.sh
#   bash run_phase2c_PL_pcgrad_lora_only_seed42.sh --dry-run
#   bash run_phase2c_PL_pcgrad_lora_only_seed42.sh --max-train-batches 5 --max-val-batches 2
#
# Environment overrides:
#   PYTHON_BIN      python interpreter (default: python)
#   NUM_WORKERS     DataLoader workers (default: 6)
#   SAVE_PATH       override output directory
#   DRY_RUN         set to 1 to run --dry-run
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SAVE_PATH="${SAVE_PATH:-runs/phase2c_bf16/PL_lora_only_seed42}"

CMD=(
  "${PYTHON_BIN}" phase2c_train.py
  --condition     P_LoRA_only
  --save-path     "${SAVE_PATH}"
  --hybrid-alpha-max 0.20
  --train-manifest   splits/visa_train_seed42.csv
  --val-manifest     splits/visa_val_seed42.csv
  --split-metadata   splits/visa_split_seed42_metadata.json
  --batch-size    6
  --num-workers   "${NUM_WORKERS}"
  --diagnostic-batch-size 1
  --bf16
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  CMD+=(--dry-run)
fi

# Pass through any extra arguments (e.g. --dry-run, --max-train-batches N)
CMD+=("$@")

printf 'Running: '
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
