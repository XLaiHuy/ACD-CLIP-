#!/usr/bin/env bash
set -euo pipefail

# Test Phase 1B / V3 with the fixed local protocol:
# no binned thresholds, pixel_stride=4, six medical datasets.

RUN_SAVE_PATH="${SAVE_PATH:-phase1_v3_attn_tau8_ss2d_g02}"

SAVE_PATH="${RUN_SAVE_PATH}" \
DFG_MODE=attn \
DFG_ATTN_DIM=256 \
DFG_ATTN_TAU=8.0 \
USE_SS2D_DFG=1 \
DFG_GAMMA_MAX=0.2 \
METRIC_THRESHOLDS=none \
PIXEL_STRIDE=4 \
bash test_6medical_selected_epochs.sh "$@"

python parse_test_log.py --log "${RUN_SAVE_PATH}/test.log" --paper-summary
