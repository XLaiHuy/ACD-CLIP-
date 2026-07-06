#!/usr/bin/env bash
set -euo pipefail

# Phase 1A main run: replace original DFG MLP gate with conservative
# dual-softmax visual-text attention. Keep all other ACD-CLIP settings fixed.

SAVE_PATH="${SAVE_PATH:-phase1_v1_attn_tau4}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCH="${EPOCH:-20}"
NUM_WORKERS="${NUM_WORKERS:-6}"
AMP="${AMP:-1}"
DFG_ATTN_DIM="${DFG_ATTN_DIM:-256}"
DFG_ATTN_TAU="${DFG_ATTN_TAU:-4.0}"

CMD=(
  conda run --no-capture-output -n torchhuy python train.py
  --dataset VisA
  --n_groups 3
  --dfg_mode attn
  --dfg_attn_dim "${DFG_ATTN_DIM}"
  --dfg_attn_tau "${DFG_ATTN_TAU}"
  --batch_size "${BATCH_SIZE}"
  --epoch "${EPOCH}"
  --grad_checkpointing
  --num_workers "${NUM_WORKERS}"
  --save_path "${SAVE_PATH}"
)

if [ "${AMP}" != "0" ]; then
  CMD+=(--amp)
fi

echo "Running Phase 1A attention DFG training:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
