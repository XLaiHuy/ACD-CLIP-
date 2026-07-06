#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-runs/phase2b/phase2b_kgsoftprompt_ctx4_fromscratch}"
NUM_WORKERS="${NUM_WORKERS:-6}"

DATASETS="${DATASETS:-Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir}" \
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
bash test_6medical_selected_epochs.sh 8 9 10 11 12 13 14 15 16 17 18 19 20

python parse_test_log.py --log "${SAVE_PATH}/test.log" --paper-summary
