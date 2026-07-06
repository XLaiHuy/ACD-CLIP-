#!/usr/bin/env bash
set -euo pipefail

# Phase 1B / V3c-beta010:
# keep GAP attention as the main path and mix in SS2D attention weights with fixed beta=0.10.

SAVE_PATH="${SAVE_PATH:-phase1_v3c_attn_tau8_ss2d_weightres_beta010}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCH="${EPOCH:-20}"
NUM_WORKERS="${NUM_WORKERS:-6}"
AMP="${AMP:-1}"

CMD=(
  conda run --no-capture-output -n torchhuy python train.py
  --dataset VisA
  --n_groups 3
  --dfg_mode attn
  --dfg_attn_dim 256
  --dfg_attn_tau 8.0
  --use_ss2d_dfg
  --dfg_gamma_max 0.2
  --dfg_ss2d_fusion weight_residual
  --dfg_beta 0.10
  --batch_size "${BATCH_SIZE}"
  --epoch "${EPOCH}"
  --grad_checkpointing
  --num_workers "${NUM_WORKERS}"
  --save_path "${SAVE_PATH}"
)

if [ "${AMP}" != "0" ]; then
  CMD+=(--amp)
fi

echo "Running Phase 1B / V3c-beta010 attention-weight residual training:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
