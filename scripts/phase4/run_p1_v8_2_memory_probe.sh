#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "=== [P1-V8.2] Controlled Train & Test Memory Probe ==="
echo "Repo Root: ${REPO_ROOT}"

OUT_DIR="runs/phase4/p1_v8_2_full20_prelaunch_audit"
mkdir -p "${OUT_DIR}"

PYTHONPATH=. python tools/probe_p1_v8_2_full20_memory.py 2>&1 | tee "${OUT_DIR}/probe_train.log"
PYTHONPATH=. python tools/probe_p1_v8_2_test_memory.py 2>&1 | tee "${OUT_DIR}/probe_test.log"

echo "[OK] Memory probes executed successfully."
