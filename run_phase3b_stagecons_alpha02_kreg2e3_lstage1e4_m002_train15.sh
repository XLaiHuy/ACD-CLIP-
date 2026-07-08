#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-runs/phase3b/phase3b_stagecons_alpha02_kreg2e3_lstage1e4_m002_train15}" \
LAMBDA_STAGE="${LAMBDA_STAGE:-0.0001}" \
STAGE_CONSISTENCY_MARGIN="${STAGE_CONSISTENCY_MARGIN:-0.02}" \
bash run_phase3b_stagecons_alpha02_kreg2e3_lstage5e4_m002_train15.sh
