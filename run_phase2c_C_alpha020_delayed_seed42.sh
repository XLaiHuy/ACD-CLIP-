#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_ROOT="runs/phase2c_bf16"
C_DIR="$RUN_ROOT/C_alpha020_delayed_seed42"
MASTER_LOG="$RUN_ROOT/phase2c_c_terminal.log"

mkdir -p "$C_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[$(date -Is)] Phase2C C delayed-activation BF16 terminal run started"
for locked_dir in "$RUN_ROOT/A_alpha020_seed42" "$RUN_ROOT/B_alpha015_seed42"; do
  (cd "$locked_dir" && sha256sum -c ARTIFACTS_LOCKED.sha256 >/dev/null)
done
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

sha256sum \
  dataset/__init__.py \
  phase2c_split.py \
  phase2c_prepare_visa_split.py \
  phase2c_utils.py \
  phase2c_train.py \
  phase2c_analyze_ab.py \
  run_phase2c_ab_common.sh \
  run_phase2c_C_alpha020_delayed_seed42.sh \
  PHASE2C_C_PREREGISTRATION.md \
  splits/visa_train_seed42.csv \
  splits/visa_val_seed42.csv \
  splits/visa_split_seed42_metadata.json > "$C_DIR/code_fingerprint.sha256"
git rev-parse HEAD > "$C_DIR/git_head.txt"
git status --short > "$C_DIR/git_status_before.txt"

BF16=1 bash run_phase2c_ab_common.sh C "$C_DIR" 0.20
test -s "$C_DIR/selection.json"
date -Is > "$C_DIR/completed_at.txt"
echo "[$(date -Is)] Phase2C C delayed-activation BF16 completed successfully"
