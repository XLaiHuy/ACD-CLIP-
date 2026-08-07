#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# ACD-CLIP Phase 4 Progress 1 v8.2 — Master Train -> Test Pipeline
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="configs/phase4/p1_v8_2_candidate1.json"
RUN_DIR="runs/phase4/p1_v8_2_full20_seed0"
LOG_FILE="${RUN_DIR}/train.log"

echo "=== [P1-V8.2] Starting Master Preflight & Launch Checklist ==="

# 1. Preflight Check
PYTHONPATH=. python tools/preflight_p1_v8_2_full20.py --stage all --config "${CONFIG_PATH}"

mkdir -p "${RUN_DIR}"
rm -f "${RUN_DIR}/exit_code.txt" "${RUN_DIR}/failure_reason.txt"

# Failure classification function
classify_failure() {
  local exit_code="$1"
  local reason="UNKNOWN_FAILURE"
  if [ "${exit_code}" -eq 137 ]; then
    reason="HOST_OOM_OR_SIGKILL"
  elif [ "${exit_code}" -eq 143 ]; then
    reason="SIGTERM"
  elif grep -q "CUDA out of memory" "${LOG_FILE}" 2>/dev/null; then
    reason="CUDA_OOM"
  elif grep -q "DataLoader worker" "${LOG_FILE}" 2>/dev/null; then
    reason="DATALOADER_WORKER_FAILURE"
  elif grep -q "No space left on device" "${LOG_FILE}" 2>/dev/null; then
    reason="DISK_FULL"
  elif grep -q "non-finite H6 loss" "${LOG_FILE}" 2>/dev/null; then
    reason="NONFINITE_LOSS"
  fi
  echo "${exit_code}" > "${RUN_DIR}/exit_code.txt"
  echo "${reason}" > "${RUN_DIR}/failure_reason.txt"
  echo "[FAILURE DETECTED] exit_code=${exit_code}, reason=${reason}"
}

# Resource monitor function
MONITOR_PID=""
start_monitor() {
  (
    while true; do
      date -Is
      nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
      free -m | grep Mem | awk '{print "RAM_Used_MB: "$3" RAM_Free_MB: "$4}' || true
      sleep 30
    done
  ) > "${RUN_DIR}/resource_monitor.log" 2>&1 &
  MONITOR_PID=$!
}

stop_monitor() {
  if [ -n "${MONITOR_PID}" ] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}

trap 'stop_monitor' EXIT INT TERM

# Prepare launcher scripts
bash scripts/phase4/run_p1_v8_2_full20_train.sh
bash scripts/phase4/run_p1_v8_2_full_test_e10_e20.sh

echo "Master setup complete. To launch training & testing when ready:"
echo "  bash scripts/phase4/run_p1_v8_2_full20_train_then_test.sh --execute"

if [[ "${1:-}" == "--execute" ]]; then
  echo "Executing 20-epoch training..."
  start_monitor
  set +e
  bash "${RUN_DIR}/launch.command.txt" 2>&1 | tee "${LOG_FILE}"
  TRAIN_EXIT=${PIPESTATUS[0]}
  set -e
  stop_monitor

  if [ "${TRAIN_EXIT}" -ne 0 ]; then
    classify_failure "${TRAIN_EXIT}"
    exit "${TRAIN_EXIT}"
  fi

  echo "0" > "${RUN_DIR}/exit_code.txt"
  echo "SUCCESS" > "${RUN_DIR}/failure_reason.txt"
  date -Is > "${RUN_DIR}/finished_at.txt"

  echo "Evaluating epochs 10 through 20..."
  for EPOCH in $(seq 10 20); do
    EPOCH_DIR="${RUN_DIR}/test_epoch_${EPOCH}"
    CKPT="${RUN_DIR}/adapter_${EPOCH}.pth"
    if [ ! -f "${CKPT}" ]; then
      echo "[ERROR] Missing adapter checkpoint ${CKPT} for epoch ${EPOCH} evaluation!"
      exit 1
    fi
    bash "${EPOCH_DIR}/test.command.txt" 2>&1 | tee "${EPOCH_DIR}/test.log"
  done

  PYTHONPATH=. python tools/summarize_p1_v8_2_full_test_epochs.py --run-dir "${RUN_DIR}"
  echo "Master train-then-test pipeline completed successfully."
fi
