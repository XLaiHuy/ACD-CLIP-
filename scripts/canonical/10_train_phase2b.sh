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
  "$PYTHON" - "$RESUME_CHECKPOINT" "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import torch

from train import _validate_resume

checkpoint_path = Path(sys.argv[1]).expanduser()
config_path = Path(sys.argv[2]).expanduser()
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = json.loads(config_path.read_text(encoding="utf-8"))
config.update({
    "micro_batch_size": 6,
    "batch_size": 6,
    "grad_accum_steps": 1,
    "effective_batch_size": 6,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2,
})
_validate_resume(checkpoint, config)
print("RESUME_COMPATIBILITY=PASS")
PY
fi

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
  --git-sha "$CANONICAL_SHA"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  train_cmd+=(--resume "$RESUME_CHECKPOINT")
fi

run_logged "$RUN_ROOT/logs/phase2b_train.log" "${train_cmd[@]}"

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
