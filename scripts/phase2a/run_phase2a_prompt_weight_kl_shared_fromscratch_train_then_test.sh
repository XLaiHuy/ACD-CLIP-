#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-runs/phase2a/phase2a_prompt_weight_kl_shared_fromscratch}"
PROMPT_WEIGHT_LR="${PROMPT_WEIGHT_LR:-5e-5}"
PROMPT_WEIGHT_LAMBDA_KL="${PROMPT_WEIGHT_LAMBDA_KL:-1e-4}"
PROMPT_WEIGHT_TEMPERATURE="${PROMPT_WEIGHT_TEMPERATURE:-2.0}"
PROMPT_WEIGHT_FREEZE_EPOCHS="${PROMPT_WEIGHT_FREEZE_EPOCHS:-3}"
NON_FINITE_LOSS_ABORT_THRESHOLD="${NON_FINITE_LOSS_ABORT_THRESHOLD:-5}"
TEST_EPOCHS=("$@")
if [ "${#TEST_EPOCHS[@]}" -eq 0 ]; then
  TEST_EPOCHS=(8 9 10 11 12 13 14 15 16 17 18 19 20)
fi

echo "==== Train Phase2A prompt_weight_kl_shared_fromscratch ===="
echo "==== SAVE_PATH=${SAVE_PATH} ===="

python train.py \
  --dataset VisA \
  --n_groups 3 \
  --dfg_mode attn \
  --dfg_attn_dim 256 \
  --dfg_attn_tau 8.0 \
  --use_ss2d_dfg \
  --dfg_gamma_max 0.2 \
  --dfg_ss2d_fusion weight_residual \
  --dfg_beta 0.10 \
  --dfg_beta_schedule warmup010 \
  --dfg_beta_target 0.10 \
  --non_finite_loss_abort_threshold "${NON_FINITE_LOSS_ABORT_THRESHOLD}" \
  --text_adapt_weight 0.2 \
  --use_prompt_weighting \
  --prompt_weight_temperature "${PROMPT_WEIGHT_TEMPERATURE}" \
  --prompt_weight_lr "${PROMPT_WEIGHT_LR}" \
  --prompt_weight_lambda_kl "${PROMPT_WEIGHT_LAMBDA_KL}" \
  --prompt_weight_freeze_epochs "${PROMPT_WEIGHT_FREEZE_EPOCHS}" \
  --batch_size 6 \
  --epoch 20 \
  --grad_checkpointing \
  --num_workers 6 \
  --amp \
  --save_path "${SAVE_PATH}"

echo "==== Test ${SAVE_PATH} | epochs: ${TEST_EPOCHS[*]} ===="

SAVE_PATH="${SAVE_PATH}" \
DFG_MODE=attn \
DFG_ATTN_DIM=256 \
DFG_ATTN_TAU=8.0 \
USE_SS2D_DFG=1 \
DFG_GAMMA_MAX=0.2 \
DFG_SS2D_FUSION=weight_residual \
DFG_BETA=0.10 \
DFG_BETA_SCHEDULE=warmup010 \
DFG_BETA_TARGET=0.10 \
USE_PROMPT_WEIGHTING=1 \
PROMPT_WEIGHT_TEMPERATURE="${PROMPT_WEIGHT_TEMPERATURE}" \
METRIC_THRESHOLDS=none \
PIXEL_STRIDE=4 \
BATCH_SIZE=8 \
NUM_WORKERS=6 \
bash test_6medical_selected_epochs.sh "${TEST_EPOCHS[@]}"

python parse_test_log.py --log "${SAVE_PATH}/test.log" --paper-summary

echo "==== Done: Phase2A prompt_weight_kl_shared_fromscratch ===="
