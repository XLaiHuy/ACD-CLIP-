#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_PATH="${SAVE_PATH:-${SCRIPT_DIR}/../../runs/phase4/k1_noop/short64_seed0}"
export SAVE_PATH
export H6_PROGRESS_VERSION="P4-CSF-K1-NOOP"
exec "${SCRIPT_DIR}/run_p4_k1_short64.sh"
