#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-runs/phase3b/phase3b_stagecons_alpha02_kreg2e3_lstage5e4_m002_train15}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCH="${EPOCH:-15}"
NUM_WORKERS="${NUM_WORKERS:-6}"
AMP="${AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-fp16}"
SOFT_PROMPT_LR="${SOFT_PROMPT_LR:-0.00005}"
LAMBDA_KG="${LAMBDA_KG:-0.01}"
LAMBDA_K="${LAMBDA_K:-0.002}"
LAMBDA_STAGE="${LAMBDA_STAGE:-0.0005}"
STAGE_CONSISTENCY_LOSS="${STAGE_CONSISTENCY_LOSS:-js_margin}"
STAGE_CONSISTENCY_MARGIN="${STAGE_CONSISTENCY_MARGIN:-0.02}"
STAGE_CONSISTENCY_WARMUP_EPOCHS="${STAGE_CONSISTENCY_WARMUP_EPOCHS:-0}"
HYBRID_ALPHA_MAX="${HYBRID_ALPHA_MAX:-0.2}"
SOFT_PROMPT_FREEZE_EPOCHS="${SOFT_PROMPT_FREEZE_EPOCHS:-3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
NON_FINITE_LOSS_ABORT_THRESHOLD="${NON_FINITE_LOSS_ABORT_THRESHOLD:-20}"
TEST_EPOCHS=(${TEST_EPOCHS:-7 8 9 10 11 12 13 14 15})

TRAIN_CMD=(
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
  --text_adapt_weight 0.2
  --use_hybrid_soft_prompt
  --hybrid_alpha_max "${HYBRID_ALPHA_MAX}"
  --soft_prompt_ctx_len 4
  --soft_prompt_lr "${SOFT_PROMPT_LR}"
  --soft_prompt_freeze_epochs "${SOFT_PROMPT_FREEZE_EPOCHS}"
  --lambda_kg "${LAMBDA_KG}"
  --lambda_k "${LAMBDA_K}"
  --lambda_stage "${LAMBDA_STAGE}"
  --stage_consistency_loss "${STAGE_CONSISTENCY_LOSS}"
  --stage_consistency_margin "${STAGE_CONSISTENCY_MARGIN}"
  --stage_consistency_warmup_epochs "${STAGE_CONSISTENCY_WARMUP_EPOCHS}"
  --stage_consistency_update_soft_only
  --stage_consistency_detach_visual
  --stage_consistency_detach_qk
  --grad_clip_norm "${GRAD_CLIP_NORM}"
  --non_finite_loss_abort_threshold "${NON_FINITE_LOSS_ABORT_THRESHOLD}"
  --batch_size "${BATCH_SIZE}"
  --epoch "${EPOCH}"
  --grad_checkpointing
  --num_workers "${NUM_WORKERS}"
  --save_path "${SAVE_PATH}"
)

if [ "${AMP}" != "0" ]; then
  TRAIN_CMD+=(--amp --amp_dtype "${AMP_DTYPE}")
fi

echo "==== Train Phase3B stage consistency ===="
echo "==== SAVE_PATH=${SAVE_PATH} ===="
echo "==== lambda_stage=${LAMBDA_STAGE} loss=${STAGE_CONSISTENCY_LOSS} margin=${STAGE_CONSISTENCY_MARGIN} warmup=${STAGE_CONSISTENCY_WARMUP_EPOCHS} amp_dtype=${AMP_DTYPE} ===="
printf ' %q' "${TRAIN_CMD[@]}"
echo
"${TRAIN_CMD[@]}"

echo "==== 6-medical test epochs ${TEST_EPOCHS[*]} ===="
SAVE_PATH="${SAVE_PATH}" \
BATCH_SIZE=8 \
NUM_WORKERS="${NUM_WORKERS}" \
N_GROUPS=3 \
DFG_MODE=attn \
DFG_ATTN_DIM=256 \
DFG_ATTN_TAU=8.0 \
USE_SS2D_DFG=1 \
DFG_GAMMA_MAX=0.2 \
DFG_SS2D_FUSION=weight_residual \
DFG_BETA=0.10 \
DFG_BETA_SCHEDULE=warmup010 \
DFG_BETA_TARGET=0.10 \
METRIC_THRESHOLDS=none \
PIXEL_STRIDE=4 \
bash test_6medical_selected_epochs.sh "${TEST_EPOCHS[@]}"

echo "==== Parsed full 6-medical results ===="
python parse_test_log.py --log "${SAVE_PATH}/test.log" | tee "${SAVE_PATH}/parsed_results.csv"

echo "==== Key AP by epoch: Brain / Retina / Colon variants ===="
awk -F, '
  NR == 1 || $1 == "Brain" || $1 == "Retina" || $1 == "Colon_clinicDB" || $1 == "Colon_colonDB" || $1 == "Colon_Kvasir" {
    print
  }
' "${SAVE_PATH}/parsed_results.csv" | tee "${SAVE_PATH}/key_ap_summary.csv"
