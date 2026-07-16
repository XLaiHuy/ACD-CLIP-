#!/usr/bin/env bash
set -euo pipefail

# Match the fixed six-medical-dataset Phase 1 evaluation protocol.
# Default gamma starts from zero during training and is learned per group.

RUN_SAVE_PATH="${SAVE_PATH:-phase1_mlp_ss2d_feature_g02}"

SAVE_PATH="${RUN_SAVE_PATH}" \
DFG_MODE=mlp \
USE_SS2D_DFG=1 \
DFG_GAMMA_MAX=0.2 \
DFG_SS2D_FUSION=feature_residual \
METRIC_THRESHOLDS=none \
PIXEL_STRIDE=4 \
bash test_6medical_selected_epochs.sh "$@"

python parse_test_log.py --log "${RUN_SAVE_PATH}/test.log" --paper-summary
