#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-test_train_main_base}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-6}"
N_GROUPS="${N_GROUPS:-3}"
DFG_MODE="${DFG_MODE:-mlp}"
DFG_ATTN_DIM="${DFG_ATTN_DIM:-256}"
DFG_ATTN_TAU="${DFG_ATTN_TAU:-4.0}"
USE_SS2D_DFG="${USE_SS2D_DFG:-0}"
DFG_GAMMA_MAX="${DFG_GAMMA_MAX:-0.2}"
DFG_SS2D_FUSION="${DFG_SS2D_FUSION:-feature_residual}"
DFG_BETA="${DFG_BETA:-0.10}"
DFG_BETA_SCHEDULE="${DFG_BETA_SCHEDULE:-fixed}"
DFG_BETA_TARGET="${DFG_BETA_TARGET:-0.10}"
METRIC_THRESHOLDS="${METRIC_THRESHOLDS:-1000}"
PIXEL_STRIDE="${PIXEL_STRIDE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-none}"
MAX_SAMPLES_PER_LABEL="${MAX_SAMPLES_PER_LABEL:-none}"

if [ "$#" -gt 0 ]; then
  EPOCHS=("$@")
else
  EPOCHS=(10 15 20)
fi

if [ -n "${DATASETS:-}" ]; then
  read -r -a DATASET_LIST <<< "${DATASETS}"
else
  DATASET_LIST=(
    Brain
    Liver
    Retina
    Colon_clinicDB
    Colon_colonDB
    Colon_Kvasir
  )
fi

for DATASET in "${DATASET_LIST[@]}"; do
  echo "==== Testing ${DATASET} | epochs: ${EPOCHS[*]} ===="
  CMD=(
    conda run --no-capture-output -n torchhuy python test.py
    --dataset "${DATASET}" \
    --n_groups "${N_GROUPS}" \
    --dfg_mode "${DFG_MODE}" \
    --dfg_attn_dim "${DFG_ATTN_DIM}" \
    --dfg_attn_tau "${DFG_ATTN_TAU}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --save_path "${SAVE_PATH}" \
    --pixel_stride "${PIXEL_STRIDE}" \
    --epochs "${EPOCHS[@]}"
  )
  if [ "${USE_SS2D_DFG}" != "0" ]; then
    CMD+=(
      --use_ss2d_dfg
      --dfg_gamma_max "${DFG_GAMMA_MAX}"
      --dfg_ss2d_fusion "${DFG_SS2D_FUSION}"
      --dfg_beta "${DFG_BETA}"
      --dfg_beta_schedule "${DFG_BETA_SCHEDULE}"
      --dfg_beta_target "${DFG_BETA_TARGET}"
    )
  fi
  if [ "${MAX_SAMPLES}" != "none" ]; then
    CMD+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [ "${MAX_SAMPLES_PER_LABEL}" != "none" ]; then
    CMD+=(--max_samples_per_label "${MAX_SAMPLES_PER_LABEL}")
  fi
  if [ "${METRIC_THRESHOLDS}" != "none" ]; then
    CMD+=(--metric_thresholds "${METRIC_THRESHOLDS}")
  fi
  "${CMD[@]}"
done

echo "==== Done. Results are appended to ${SAVE_PATH}/test.log ===="
