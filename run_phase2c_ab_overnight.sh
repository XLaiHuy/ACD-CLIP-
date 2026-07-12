#!/usr/bin/env bash
# Run the deterministic Phase2C A-prime/B comparison sequentially from a real terminal.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_ROOT="runs/phase2c"
A_DIR="$RUN_ROOT/A_alpha020_seed42"
B_DIR="$RUN_ROOT/B_alpha015_seed42"
MASTER_LOG="$RUN_ROOT/phase2c_ab_terminal.log"

mkdir -p "$A_DIR" "$B_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[$(date -Is)] Phase2C A-prime/B terminal run started"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

# Keep an exact record of the code and deterministic split used by both conditions.
sha256sum \
  dataset/__init__.py \
  phase2c_split.py \
  phase2c_prepare_visa_split.py \
  phase2c_utils.py \
  phase2c_train.py \
  run_phase2c_ab_common.sh \
  run_phase2c_A_alpha020_seed42.sh \
  run_phase2c_B_alpha015_seed42.sh \
  splits/visa_train_seed42.csv \
  splits/visa_val_seed42.csv \
  splits/visa_split_seed42_metadata.json > "$A_DIR/code_fingerprint.sha256"
cp "$A_DIR/code_fingerprint.sha256" "$B_DIR/code_fingerprint.sha256"
git rev-parse HEAD > "$A_DIR/git_head.txt"
cp "$A_DIR/git_head.txt" "$B_DIR/git_head.txt"
git status --short > "$A_DIR/git_status_before.txt"
cp "$A_DIR/git_status_before.txt" "$B_DIR/git_status_before.txt"

echo "[$(date -Is)] Starting A-prime (hybrid alpha 0.20)"
bash run_phase2c_A_alpha020_seed42.sh
test -s "$A_DIR/selection.json"

echo "[$(date -Is)] A-prime finished; starting B (hybrid alpha 0.15)"
bash run_phase2c_B_alpha015_seed42.sh
test -s "$B_DIR/selection.json"

date -Is > "$RUN_ROOT/phase2c_ab_completed_at.txt"
echo "[$(date -Is)] Phase2C A-prime/B completed successfully"
