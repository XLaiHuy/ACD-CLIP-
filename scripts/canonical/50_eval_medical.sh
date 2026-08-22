#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

init_common "$@"
((${#COMMON_ARGS[@]} == 0)) || die "unknown Medical-evaluation arguments: ${COMMON_ARGS[*]}"
require_base_assets
require_medical_root

banner "STAGE 5 — MEDICAL FINAL ZERO-SHOT (COMPARE ONLY)"
printf '[canonical] WARNING: FINAL unseen Medical evaluation; no training, fitting, tuning, or lambda changes are permitted.\n'

selection_json="$RUN_ROOT/phase2b_selection/phase2b_selection.json"
freeze_json="$RUN_ROOT/sabra_lambda/SABRA_FREEZE.json"
require_file "$selection_json"
require_file "$freeze_json"

"$PYTHON" - "$selection_json" "$freeze_json" "$SCIENTIFIC_CODE_SHA" <<'PY'
import hashlib
import sys
from pathlib import Path

from tools.sabra.artifacts import load_json, validate_sabra_freeze

selection = load_json(sys.argv[1])
freeze = load_json(sys.argv[2])
expected_code_sha = sys.argv[3]
if freeze.get("provenance", {}).get("git_sha") != expected_code_sha:
    raise SystemExit("SABRA freeze provenance SHA differs from scientific code SHA")
if selection.get("status") != "FROZEN":
    raise SystemExit("Phase2B selection must be FROZEN before Medical evaluation")
checkpoint = Path(str(selection.get("selected_checkpoint", ""))).expanduser()
expected = str(selection.get("selected_checkpoint_sha256", ""))
if not checkpoint.is_file() or not expected:
    raise SystemExit("Phase2B selection lacks a valid selected checkpoint")
digest = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit("Phase2B selected checkpoint SHA256 mismatch")
validate_sabra_freeze(freeze, checkpoint_sha256=actual)
if freeze.get("medical_seen") is not False:
    raise SystemExit("SABRA freeze is not Medical-clean")
if freeze.get("relational", {}).get("backend") != "fast":
    raise SystemExit("Medical evaluation requires frozen fast relational backend")
print(f"MEDICAL_GUARD=PASS checkpoint_sha256={actual} backend=fast medical_seen=false")
PY

medical_names="$($PYTHON - <<'PY'
from dataset.info import MEDICAL_EVAL_PATHS

for name in MEDICAL_EVAL_PATHS:
    print(name)
PY
)"
[[ -n "$medical_names" ]] || die "authoritative Medical dataset metadata returned no datasets"
mapfile -t medical_datasets <<< "$medical_names"

for dataset in "${medical_datasets[@]}"; do
  [[ -n "$dataset" ]] || continue
  output_dir="$RUN_ROOT/medical/$dataset"
  work_dir="$RUN_ROOT/medical_work/$dataset"
  medical_cmd=(
    "$PYTHON" "$SCRIPT_DIR/medical_compare_external.py"
    --dataset "$dataset"
    --data-root "$MEDICAL_ROOT"
    --phase2b-selection "$selection_json"
    --sabra-freeze "$freeze_json"
    --config "$CONFIG"
    --clip-asset "$CLIP_ASSET"
    --device cuda
    --batch-size 6
    --num-workers 4
    --prefetch-factor 2
    --pin-memory
    --metric-mode exact
    --pixel-stride 1
    --output-dir "$output_dir"
    --work-dir "$work_dir"
    --reuse-inference-cache
  )
  if [[ -f "$output_dir/metrics.json" && -f "$output_dir/per_class_metrics.csv" ]]; then
    # Validate and preserve a completed result without truncating its log.
    run_cmd "${medical_cmd[@]}"
    continue
  fi
  log_path="$RUN_ROOT/logs/medical_${dataset}.log"
  if [[ -e "$log_path" ]]; then
    resume_index=1
    while [[ -e "$RUN_ROOT/logs/medical_${dataset}.resume${resume_index}.log" ]]; do
      ((resume_index += 1))
    done
    log_path="$RUN_ROOT/logs/medical_${dataset}.resume${resume_index}.log"
  fi
  run_logged "$log_path" "${medical_cmd[@]}"
  if [[ "$DRY_RUN" != "1" ]]; then
    require_file "$output_dir/metrics.json"
    require_file "$output_dir/per_class_metrics.csv"
  fi
done

printf 'STAGE5_STATUS=PASS\n'
