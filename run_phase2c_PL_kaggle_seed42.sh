#!/usr/bin/env bash
# Phase2C P_LoRA_only — Kaggle launcher
#
# Designed for a Kaggle notebook with one T4 GPU.
# Does NOT use DDP or DataParallel; single GPU only.
#
# ── Environment variables ──────────────────────────────────────────────────
# REPO_ROOT        Path to the cloned repository (default: /kaggle/working/ACD-CLIP)
# TRAIN_MANIFEST   Train CSV (default: /kaggle/working/runtime_splits/visa_train_seed42_kaggle.csv)
# VAL_MANIFEST     Val CSV   (default: /kaggle/working/runtime_splits/visa_val_seed42_kaggle.csv)
# SPLIT_METADATA   Metadata JSON (default: REPO_ROOT/splits/visa_split_seed42_metadata.json)
# RUN_DIR          Output directory
# NUM_WORKERS      DataLoader workers (default: 2 — Kaggle T4 limit)
# PYTHON_BIN       Python interpreter (default: python)
# ──────────────────────────────────────────────────────────────────────────
#
# DO NOT RUN UNTIL MANUAL APPROVAL after dry-run and smoke-test review.
#
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/kaggle/working/ACD-CLIP}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUN_DIR="${RUN_DIR:-/kaggle/working/runs/phase2c_kaggle/PL_lora_only_seed42}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-/kaggle/working/runtime_splits/visa_train_seed42_kaggle.csv}"
VAL_MANIFEST="${VAL_MANIFEST:-/kaggle/working/runtime_splits/visa_val_seed42_kaggle.csv}"
SPLIT_METADATA="${SPLIT_METADATA:-${REPO_ROOT}/splits/visa_split_seed42_metadata.json}"

# Use a single GPU.  Only set CUDA_VISIBLE_DEVICES when the variable is not
# already defined to avoid overriding a higher-level setting.
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

cd "${REPO_ROOT}"

CMD=(
  "${PYTHON_BIN}" phase2c_train.py
  --condition     P_LoRA_only
  --save-path     "${RUN_DIR}"
  --hybrid-alpha-max 0.20
  --train-manifest   "${TRAIN_MANIFEST}"
  --val-manifest     "${VAL_MANIFEST}"
  --split-metadata   "${SPLIT_METADATA}"
  --cuda-device   0
  --batch-size    6
  --num-workers   "${NUM_WORKERS}"
  --diagnostic-batch-size 1
  --bf16
)

# Pass through any extra arguments (e.g. --dry-run, --max-train-batches N)
CMD+=("$@")

printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'Running: '
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
