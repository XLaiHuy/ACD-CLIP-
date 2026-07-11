#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch}"
OUTPUT_DIR="${OUTPUT_DIR:-${SAVE_PATH}/anchor_e10_diagnosis}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-6}"
PIXEL_STRIDE="${PIXEL_STRIDE:-4}"
METRIC_THRESHOLDS="${METRIC_THRESHOLDS:-none}"
MAX_SAMPLES="${MAX_SAMPLES:-none}"
MAX_SAMPLES_PER_LABEL="${MAX_SAMPLES_PER_LABEL:-none}"

CMD=(
  conda run --no-capture-output -n torchhuy python phase2b_anchor_diagnosis.py
  --mode anchor
  --save_path "${SAVE_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --epochs 10
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
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --pixel_stride "${PIXEL_STRIDE}"
)

if [ "${METRIC_THRESHOLDS}" != "none" ]; then
  CMD+=(--metric_thresholds "${METRIC_THRESHOLDS}")
fi
if [ "${MAX_SAMPLES}" != "none" ]; then
  CMD+=(--max_samples "${MAX_SAMPLES}")
fi
if [ "${MAX_SAMPLES_PER_LABEL}" != "none" ]; then
  CMD+=(--max_samples_per_label "${MAX_SAMPLES_PER_LABEL}")
fi

echo "==== Phase2B e10 anchor diagnosis ===="
echo "SAVE_PATH=${SAVE_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"

echo "==== Suggested exploratory config ===="
cat "${OUTPUT_DIR}/anchor_best_config.txt"
