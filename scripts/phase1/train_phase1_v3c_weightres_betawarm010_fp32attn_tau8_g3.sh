#!/usr/bin/env bash
set -euo pipefail

# Phase 1B / V3c fp32-attn stability fix:
# weight_residual + beta warmup010 + fp32 q/k attention scores/softmax.

SAVE_PATH="${SAVE_PATH:-phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCH="${EPOCH:-20}"
NUM_WORKERS="${NUM_WORKERS:-6}"
AMP="${AMP:-1}"
NON_FINITE_LOSS_ABORT_THRESHOLD="${NON_FINITE_LOSS_ABORT_THRESHOLD:-5}"

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
  --dfg_beta_schedule warmup010
  --dfg_beta_target 0.10
  --non_finite_loss_abort_threshold "${NON_FINITE_LOSS_ABORT_THRESHOLD}"
  --batch_size "${BATCH_SIZE}"
  --epoch "${EPOCH}"
  --grad_checkpointing
  --num_workers "${NUM_WORKERS}"
  --save_path "${SAVE_PATH}"
)

if [ "${AMP}" != "0" ]; then
  CMD+=(--amp)
fi

echo "Running Phase 1B / V3c weight_residual beta-warm010 fp32-attn tau8 g3 training:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
