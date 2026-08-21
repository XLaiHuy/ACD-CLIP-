#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

init_common "$@"
((${#COMMON_ARGS[@]} == 0)) || die "unknown training arguments: ${COMMON_ARGS[*]}"
require_base_assets

banner "STAGE 1 — PHASE2B TRAINING (20 EPOCHS, BATCH 6)"

completed_manifest="$RUN_ROOT/phase2b/run_manifest.json"
if [[ -f "$completed_manifest" ]]; then
  status="$($PYTHON - "$completed_manifest" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("status", ""))
PY
)"
  if [[ "$status" == "COMPLETED" ]]; then
    if [[ "$FORCE_RERUN" == "1" ]]; then
      warn "FORCE_RERUN=1: completed Phase2B output will be rerun in place"
    else
      die "Phase2B run_manifest.json is already COMPLETED; set FORCE_RERUN=1 for an explicit rerun"
    fi
  fi
fi

if [[ ! -e "$completed_manifest" && -d "$RUN_ROOT/phase2b" && -z "$RESUME_CHECKPOINT" ]]; then
  first_entry="$(find "$RUN_ROOT/phase2b" -mindepth 1 -maxdepth 1 -print -quit)"
  if [[ -n "$first_entry" ]]; then
    die "Phase2B output directory contains incomplete files; provide RESUME_CHECKPOINT or handle it explicitly before rerun: $RUN_ROOT/phase2b"
  fi
fi

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  require_file "$RESUME_CHECKPOINT"
  printf '[canonical] validating resume checkpoint compatibility through train.py\n'
  printf 'COMMAND: %q - %q %q %q\n' "$PYTHON" "<resume-checkpoint>" "$RESUME_CHECKPOINT" "$CONFIG"
  validate_resume_checkpoint "$RESUME_CHECKPOINT"
fi

report_training_interruption() {
  local resume_checkpoint="$RUN_ROOT/phase2b/last.pth"
  printf 'TRAIN_INTERRUPTED=YES\n'
  if [[ -f "$resume_checkpoint" ]] && validate_resume_checkpoint "$resume_checkpoint" >/dev/null 2>&1; then
    printf 'RESUME_AVAILABLE=YES\n'
    printf 'RESUME_VALIDATION=PASS\n'
    printf 'RESUME_COMMAND=RESUME_CHECKPOINT=%q RUN_ROOT=%q %q train\n' "$resume_checkpoint" "$RUN_ROOT" "$SCRIPT_DIR/run_pipeline.sh"
  else
    printf 'RESUME_AVAILABLE=NO\n'
    if [[ -f "$resume_checkpoint" ]]; then
      printf 'RESUME_VALIDATION=FAIL\n'
    fi
  fi
}
train_cmd=(
  "$PYTHON" "$REPO_ROOT/train.py"
  --visa-root "$VISA_ROOT"
  --clip-asset "$CLIP_ASSET"
  --config "$CONFIG"
  --run-root "$RUN_ROOT"
  --device cuda
  --epochs 20
  --micro-batch-size 6
  --grad-accum-steps 1
  --num-workers 4
  --prefetch-factor 2
  --pin-memory
  --persistent-workers
  --git-sha "$SCIENTIFIC_CODE_SHA"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  train_cmd+=(--resume "$RESUME_CHECKPOINT")
fi

if run_logged "$RUN_ROOT/logs/phase2b_train.log" "${train_cmd[@]}"; then
  :
else
  train_status=$?
  report_training_interruption
  exit "$train_status"
fi
candidate_epochs=(10 12 14 16 18 20)
if [[ "$DRY_RUN" == "1" ]]; then
  printf '[canonical] DRY_RUN: would require checkpoints under %s/phase2b/checkpoints\n' "$RUN_ROOT"
else
  for epoch in "${candidate_epochs[@]}"; do
    require_file "$RUN_ROOT/phase2b/checkpoints/adapter_${epoch}.pth"
  done
  require_file "$RUN_ROOT/phase2b/last.pth"
  require_file "$RUN_ROOT/phase2b/config_resolved.json"
  require_file "$RUN_ROOT/phase2b/run_manifest.json"
fi

printf 'STAGE1_STATUS=PASS\n'
