#!/usr/bin/env bash
set -euo pipefail

# Single-variable ablation from Phase1 best:
# keep V3c weight-residual/fp32-attn/tau8/g3 settings, change only text_adapt_weight.

TEXT_ADAPT_WEIGHT="${TEXT_ADAPT_WEIGHT:-0.1}"
TAG="${TEXT_ADAPT_WEIGHT/./}"
SAVE_PATH="${SAVE_PATH:-phase1_v3c_w${TAG}_nokeyanchor}"
TEST_EPOCHS=("$@")
if [ "${#TEST_EPOCHS[@]}" -eq 0 ]; then
  TEST_EPOCHS=(8 9 10 11 12)
fi

echo "==== Train Phase1 V3c text_adapt_weight=${TEXT_ADAPT_WEIGHT} ===="
echo "==== SAVE_PATH=${SAVE_PATH} ===="

set +e
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
  --text_adapt_weight "${TEXT_ADAPT_WEIGHT}" \
  --batch_size 6 \
  --epoch 20 \
  --grad_checkpointing \
  --num_workers 6 \
  --amp \
  --save_path "${SAVE_PATH}"
TRAIN_STATUS=$?
set -e

if [ "${TRAIN_STATUS}" -ne 0 ]; then
  echo "==== Train exited with status ${TRAIN_STATUS}; testing completed checkpoints only ===="
fi

AVAILABLE_TEST_EPOCHS=()
for EPOCH in "${TEST_EPOCHS[@]}"; do
  if [ -f "${SAVE_PATH}/adapter_${EPOCH}.pth" ]; then
    AVAILABLE_TEST_EPOCHS+=("${EPOCH}")
  else
    echo "==== Skip epoch ${EPOCH}: ${SAVE_PATH}/adapter_${EPOCH}.pth not found ===="
  fi
done

if [ "${#AVAILABLE_TEST_EPOCHS[@]}" -eq 0 ]; then
  echo "No requested checkpoints are available for testing."
  exit "${TRAIN_STATUS}"
fi

echo "==== Test ${SAVE_PATH} | epochs: ${AVAILABLE_TEST_EPOCHS[*]} ===="

SAVE_PATH="${SAVE_PATH}" \
bash test_phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3_selected_epochs.sh "${AVAILABLE_TEST_EPOCHS[@]}"

echo "==== Done: train + test completed for ${SAVE_PATH} ===="
exit "${TRAIN_STATUS}"
