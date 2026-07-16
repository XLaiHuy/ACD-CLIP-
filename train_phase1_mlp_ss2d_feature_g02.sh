#!/usr/bin/env bash
set -euo pipefail

# Clean factorial control: original GAP-to-MLP DFG with an SS2D feature residual.
# raw_gamma starts at 0, so the first step is exactly the original MLP baseline.

SAVE_PATH="${SAVE_PATH:-phase1_mlp_ss2d_feature_g02}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCH="${EPOCH:-20}"
NUM_WORKERS="${NUM_WORKERS:-6}"
AMP="${AMP:-1}"
NON_FINITE_LOSS_ABORT_THRESHOLD="${NON_FINITE_LOSS_ABORT_THRESHOLD:-5}"

CMD=(
  conda run --no-capture-output -n torchhuy python train.py
  --dataset VisA
  --n_groups 3
  --dfg_mode mlp
  --use_ss2d_dfg
  --dfg_gamma_max 0.2
  --dfg_ss2d_fusion feature_residual
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

echo "Running Phase 1 MLP + SS2D feature-residual ablation (gamma_max=0.2):"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
