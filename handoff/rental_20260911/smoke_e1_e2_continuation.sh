#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_v2.sh"
DATASET="${1:-VisA}"
require_common_inputs "${DATASET}"
TMP_ROOT="$(mktemp -d /tmp/acd-clip-e1-e2.XXXXXX)"
trap 'rm -rf "${TMP_ROOT}"' EXIT
frozen_train_args
"${PYTHON}" "${ROOT}/train.py" "${TRAIN_ARGS[@]}" --epoch 1 --protocol_horizon 20 --max_batches 1 \
  --trace_batch_identity --save_path "${TMP_ROOT}"
E1_SHA256="$(sha256sum "${TMP_ROOT}/adapter_1.pth" | awk '{print $1}')"
"${PYTHON}" "${ROOT}/train.py" "${TRAIN_ARGS[@]}" --epoch 2 --protocol_horizon 20 --max_batches 1 \
  --resume "${TMP_ROOT}/adapter_1.pth" --use_safe_anchor \
  --anchor_lambda 0.0021633926715180626 --anchor_reference_path "${TMP_ROOT}/adapter_1.pth" \
  --anchor_gradient_budget --anchor_family_budget 0.10 --anchor_family_audit --save_path "${TMP_ROOT}"
# train.py writes the peak allocator measurement to train.log; forward it
# before the temporary smoke directory is removed so preflight can report it.
if [[ -f "${TMP_ROOT}/train.log" ]]; then
  rg 'peak_cuda_memory_allocated_gb=' "${TMP_ROOT}/train.log" || true
fi
"${PYTHON}" - "${TMP_ROOT}/adapter_1.pth" "${TMP_ROOT}/adapter_2.pth" "${E1_SHA256}" <<'PY'
import sys
import torch
e1 = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
e2 = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
assert e1["epoch"] == 1 and e2["epoch"] == 2
assert e2["global_step"] > e1["global_step"]
assert e2["scheduler_state"]["last_epoch"] > e1["scheduler_state"]["last_epoch"]
assert e2["resolved_scientific_config"]["anchor_reference_sha256"] == sys.argv[3]
assert e2["resolved_scientific_config"]["use_safe_anchor"] is True
assert e2["parent_scientific_config"] == e1["parent_scientific_config"]
print("E1_TO_E2_CONTINUATION_TEST_PASS")
PY
