#!/usr/bin/env bash
set -euo pipefail

# Test Phase 1 V2 tau=8 with the fixed local protocol:
# no binned thresholds, pixel_stride=4, six medical datasets.

RUN_SAVE_PATH="${SAVE_PATH:-phase1_v2_attn_tau8}"

SAVE_PATH="${RUN_SAVE_PATH}" \
DFG_MODE=attn \
DFG_ATTN_DIM=256 \
DFG_ATTN_TAU=8.0 \
METRIC_THRESHOLDS=none \
PIXEL_STRIDE=4 \
bash test_6medical_selected_epochs.sh "$@"

python parse_test_log.py --log "${RUN_SAVE_PATH}/test.log"
