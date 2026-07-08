#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-runs/phase3a/phase3a_dyn_dwexpert_convlora_k35_e2_t30_h0125_alpha02_kreg2e3_train20_bf16}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCH="${EPOCH:-20}"
NUM_WORKERS="${NUM_WORKERS:-6}"
AMP="${AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
SOFT_PROMPT_LR="${SOFT_PROMPT_LR:-0.00005}"
LAMBDA_KG="${LAMBDA_KG:-0.01}"
LAMBDA_K="${LAMBDA_K:-0.002}"
HYBRID_ALPHA_MAX="${HYBRID_ALPHA_MAX:-0.2}"
SOFT_PROMPT_FREEZE_EPOCHS="${SOFT_PROMPT_FREEZE_EPOCHS:-3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
NON_FINITE_LOSS_ABORT_THRESHOLD="${NON_FINITE_LOSS_ABORT_THRESHOLD:-20}"
CONVLORA_VARIANT="${CONVLORA_VARIANT:-dynamic_depthwise_expert}"
DYNAMIC_DW_NUM_EXPERTS="${DYNAMIC_DW_NUM_EXPERTS:-2}"
DYNAMIC_DW_TEMPERATURE="${DYNAMIC_DW_TEMPERATURE:-30.0}"
DYNAMIC_DW_GATE_HIDDEN_RATIO="${DYNAMIC_DW_GATE_HIDDEN_RATIO:-0.125}"
DYNAMIC_DW_USE_BN="${DYNAMIC_DW_USE_BN:-1}"
DYNAMIC_DW_ACTIVATION="${DYNAMIC_DW_ACTIVATION:-silu}"
DYNAMIC_DW_ZERO_INIT="${DYNAMIC_DW_ZERO_INIT:-0}"
TEST_EPOCHS=(${TEST_EPOCHS:-7 8 9 10 11 12 13 14 15 16 17 18 19 20})

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
  --convlora_variant "${CONVLORA_VARIANT}"
  --dynamic_dw_num_experts "${DYNAMIC_DW_NUM_EXPERTS}"
  --dynamic_dw_temperature "${DYNAMIC_DW_TEMPERATURE}"
  --dynamic_dw_gate_hidden_ratio "${DYNAMIC_DW_GATE_HIDDEN_RATIO}"
  --dynamic_dw_activation "${DYNAMIC_DW_ACTIVATION}"
  --grad_clip_norm "${GRAD_CLIP_NORM}"
  --non_finite_loss_abort_threshold "${NON_FINITE_LOSS_ABORT_THRESHOLD}"
  --batch_size "${BATCH_SIZE}"
  --epoch "${EPOCH}"
  --grad_checkpointing
  --num_workers "${NUM_WORKERS}"
  --save_path "${SAVE_PATH}"
)

if [ "${DYNAMIC_DW_USE_BN}" != "0" ]; then
  TRAIN_CMD+=(--dynamic_dw_use_bn)
else
  TRAIN_CMD+=(--no-dynamic_dw_use_bn)
fi
if [ "${DYNAMIC_DW_ZERO_INIT}" != "0" ]; then
  TRAIN_CMD+=(--dynamic_dw_zero_init)
fi
if [ "${AMP}" != "0" ]; then
  TRAIN_CMD+=(--amp)
  TRAIN_CMD+=(--amp_dtype "${AMP_DTYPE}")
fi

echo "==== Train Phase3A Conv-LoRA variant=${CONVLORA_VARIANT} ===="
echo "==== SAVE_PATH=${SAVE_PATH} ===="
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
CONVLORA_VARIANT="${CONVLORA_VARIANT}" \
DYNAMIC_DW_NUM_EXPERTS="${DYNAMIC_DW_NUM_EXPERTS}" \
DYNAMIC_DW_TEMPERATURE="${DYNAMIC_DW_TEMPERATURE}" \
DYNAMIC_DW_GATE_HIDDEN_RATIO="${DYNAMIC_DW_GATE_HIDDEN_RATIO}" \
DYNAMIC_DW_USE_BN="${DYNAMIC_DW_USE_BN}" \
DYNAMIC_DW_ACTIVATION="${DYNAMIC_DW_ACTIVATION}" \
DYNAMIC_DW_ZERO_INIT="${DYNAMIC_DW_ZERO_INIT}" \
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
