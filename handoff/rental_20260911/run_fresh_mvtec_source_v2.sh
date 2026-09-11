#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_v2.sh"
DATASET=MVTec_all_supervised
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/fresh_mvtec_all_source_20260911}"
require_common_inputs "${DATASET}"
if [[ -e "${RUN_ROOT}" ]]; then echo "Refusing to reuse existing run root: ${RUN_ROOT}" >&2; exit 2; fi
mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/orchestration.log") 2>&1
record_run_environment "${RUN_ROOT}/environment.txt"
frozen_train_args
echo "RUN_M fresh MVTec-AD-all supervised source training"
echo "anchor_reference=exact fresh E1 checkpoint in this run"
echo "resume_state=full optimizer/scheduler/scaler/RNG/dataloader state from E1"
"${PYTHON}" "${ROOT}/train.py" "${TRAIN_ARGS[@]}" --epoch 1 --protocol_horizon 20 --save_path "${RUN_ROOT}"
test -s "${RUN_ROOT}/adapter_1.pth"
verify_full_checkpoint "${RUN_ROOT}/adapter_1.pth"
echo "fresh_e1_anchor_sha256=$(sha256sum "${RUN_ROOT}/adapter_1.pth" | awk '{print $1}')"
"${PYTHON}" "${ROOT}/train.py" "${TRAIN_ARGS[@]}" --epoch 20 --protocol_horizon 20 \
  --resume "${RUN_ROOT}/adapter_1.pth" --use_safe_anchor \
  --anchor_lambda 0.0021633926715180626 --anchor_reference_path "${RUN_ROOT}/adapter_1.pth" \
  --anchor_gradient_budget --anchor_family_budget 0.10 --anchor_family_audit \
  --save_path "${RUN_ROOT}"
for epoch in $(seq 1 20); do test -s "${RUN_ROOT}/adapter_${epoch}.pth"; done
echo "RUN_M checkpoints=adapter_1.pth..adapter_20.pth"
