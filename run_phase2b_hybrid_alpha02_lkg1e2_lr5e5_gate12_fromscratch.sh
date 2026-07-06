#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-runs/phase2b/phase2b_hybrid_alpha02_lkg1e2_lr5e5_train20_test7to20_fromscratch}"
BATCH_SIZE="${BATCH_SIZE:-6}"
EPOCH="${EPOCH:-20}"
NUM_WORKERS="${NUM_WORKERS:-6}"
AMP="${AMP:-1}"
SOFT_PROMPT_LR="${SOFT_PROMPT_LR:-0.00005}"
LAMBDA_KG="${LAMBDA_KG:-0.01}"
HYBRID_ALPHA_MAX="${HYBRID_ALPHA_MAX:-0.2}"
SOFT_PROMPT_FREEZE_EPOCHS="${SOFT_PROMPT_FREEZE_EPOCHS:-3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
NON_FINITE_LOSS_ABORT_THRESHOLD="${NON_FINITE_LOSS_ABORT_THRESHOLD:-5}"
BRAIN_GATE_AP="${BRAIN_GATE_AP:-44}"
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
  --grad_clip_norm "${GRAD_CLIP_NORM}"
  --non_finite_loss_abort_threshold "${NON_FINITE_LOSS_ABORT_THRESHOLD}"
  --batch_size "${BATCH_SIZE}"
  --epoch "${EPOCH}"
  --grad_checkpointing
  --num_workers "${NUM_WORKERS}"
  --save_path "${SAVE_PATH}"
)

if [ "${AMP}" != "0" ]; then
  TRAIN_CMD+=(--amp)
fi

echo "==== Train Phase2B hybrid hard-soft alpha=${HYBRID_ALPHA_MAX} from scratch ===="
echo "==== SAVE_PATH=${SAVE_PATH} ===="
printf ' %q' "${TRAIN_CMD[@]}"
echo
"${TRAIN_CMD[@]}"

echo "==== Brain gate test epochs ${TEST_EPOCHS[*]} ===="
DATASETS=Brain \
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

BEST_BRAIN_AP="$(
  python parse_test_log.py --log "${SAVE_PATH}/test.log" \
    | awk -F, 'NR > 1 && $1 == "Brain" && $8 + 0 > best { best = $8 + 0 } END { printf "%.2f", best }'
)"

echo "==== Brain gate best pixel AP: ${BEST_BRAIN_AP} (threshold ${BRAIN_GATE_AP}) ===="
python - "$BEST_BRAIN_AP" "$BRAIN_GATE_AP" <<'PY'
import sys
best = float(sys.argv[1])
threshold = float(sys.argv[2])
if best < threshold:
    raise SystemExit(f"Brain gate failed: best pixel AP {best:.2f} < {threshold:.2f}")
print(f"Brain gate passed: best pixel AP {best:.2f} >= {threshold:.2f}")
PY
