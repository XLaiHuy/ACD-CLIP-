#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# ACD-CLIP Phase 4 Progress 1 v8.2 — Resume Pipeline Script
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="configs/phase4/p1_v8_2_candidate1.json"
RUN_DIR="runs/phase4/p1_v8_2_full20_seed0"

echo "=== [P1-V8.2] Resume Pipeline Check ==="

if [ ! -d "${RUN_DIR}" ]; then
  echo "[ERROR] Run directory ${RUN_DIR} does not exist. Nothing to resume."
  exit 1
fi

LATEST_EPOCH=0
for e in $(seq 1 20); do
  if [ -f "${RUN_DIR}/adapter_${e}.pth" ]; then
    LATEST_EPOCH="${e}"
  fi
done

echo "Latest available adapter checkpoint: epoch ${LATEST_EPOCH}"

if [ "${LATEST_EPOCH}" -lt 20 ]; then
  echo "Training is incomplete. Resume required from epoch $((LATEST_EPOCH + 1))."
else
  echo "Training is complete (all 20 adapter checkpoints present)."
fi

echo "Checking test epoch completeness (epochs 10-20)..."
COMPLETED_TEST_EPOCHS=()
MISSING_TEST_EPOCHS=()
for e in $(seq 10 20); do
  if [ -f "${RUN_DIR}/test_epoch_${e}/metrics.json" ]; then
    COMPLETED_TEST_EPOCHS+=("${e}")
  else
    MISSING_TEST_EPOCHS+=("${e}")
  fi
done

echo "Completed test epochs: ${COMPLETED_TEST_EPOCHS[*]:-none}"
echo "Missing test epochs: ${MISSING_TEST_EPOCHS[*]:-none}"
echo "Resume inspection completed cleanly. Existing valid results will not be overwritten."
