#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

init_common "$@"
((${#COMMON_ARGS[@]} == 0)) || die "unknown SABRA source arguments: ${COMMON_ARGS[*]}"
require_base_assets

banner "STAGE 3 — SABRA SOURCE CALIBRATION (VisA ONLY)"

selection_json="$RUN_ROOT/phase2b_selection/phase2b_selection.json"
require_file "$selection_json"
source_dir="$RUN_ROOT/sabra_source"
source_json="$source_dir/sabra_source_calibration.json"
require_clean_stage_output "$source_json"

"$PYTHON" - "$selection_json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if selection.get("status") != "FROZEN":
    raise SystemExit("Phase2B selection must be FROZEN before SABRA source fitting")
checkpoint = Path(str(selection.get("selected_checkpoint", ""))).expanduser()
expected = str(selection.get("selected_checkpoint_sha256", ""))
if not checkpoint.is_file() or not expected:
    raise SystemExit("Phase2B selection lacks a usable selected checkpoint")
digest = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != expected:
    raise SystemExit("Phase2B selected checkpoint SHA256 mismatch")
PY

fit_cmd=(
  "$PYTHON" "$REPO_ROOT/calibrate_sabra.py" fit-source
  --phase2b-selection "$selection_json"
  --visa-root "$VISA_ROOT"
  --output-dir "$source_dir"
  --dataset VisA
  --config "$CONFIG"
  --clip-asset "$CLIP_ASSET"
  --device cuda
  --batch-size 6
  --num-workers 4
  --prefetch-factor 2
  --backend fast
  --git-sha "$SCIENTIFIC_CODE_SHA"
)
run_logged "$RUN_ROOT/logs/sabra_fit_source.log" "${fit_cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[canonical] DRY_RUN: would validate GT-free/GT-target manifests and lambda-free source calibration\n'
else
  require_file "$source_dir/GT_FREE_MANIFEST.json"
  require_file "$source_dir/GT_TARGET_MANIFEST.json"
  require_file "$source_json"
  require_dir "$source_dir/gt_free_cache"
  "$PYTHON" - "$selection_json" "$source_json" "$source_dir/GT_FREE_MANIFEST.json" "$source_dir/GT_TARGET_MANIFEST.json" "$SCIENTIFIC_CODE_SHA" <<'PY'
import sys
from pathlib import Path

from tools.sabra.artifacts import load_json, validate_source_calibration
from tools.sabra.relational import FEATURE_ORDER, NEED_ORDER

selection = load_json(sys.argv[1])
source = load_json(sys.argv[2])
gt_free = load_json(sys.argv[3])
gt_target = load_json(sys.argv[4])
expected_code_sha = sys.argv[5]
validate_source_calibration(source)
if source.get("provenance", {}).get("git_sha") != expected_code_sha:
    raise SystemExit("source calibration provenance SHA differs from scientific code SHA")
if source["phase2b"]["checkpoint_sha256"] != selection["selected_checkpoint_sha256"]:
    raise SystemExit("source calibration checkpoint SHA256 differs from Phase2B selection")
if source.get("relational", {}).get("backend") != "fast":
    raise SystemExit("source calibration backend is not fast")
if source.get("trust", {}).get("feature_order") != list(FEATURE_ORDER):
    raise SystemExit("Trust feature order mismatch")
if source.get("need", {}).get("feature_order") != list(NEED_ORDER):
    raise SystemExit("Need feature order mismatch")
if source.get("margin_scale", {}).get("definition") != "P90(abs(native_margin))":
    raise SystemExit("margin scale definition mismatch")
if gt_free.get("GT_FREE_CACHE_FINALIZED") is not True or gt_free.get("dataset") != "VisA":
    raise SystemExit("GT-free cache was not finalized for VisA")
if gt_free.get("medical_reads") != 0 or gt_target.get("medical_reads") != 0:
    raise SystemExit("Medical reads were recorded during source calibration")
if gt_target.get("GT_FREE_CACHE_FINALIZED") is not True:
    raise SystemExit("GT target manifest does not bind to finalized GT-free cache")


def contains_lambda(value):
    if isinstance(value, dict):
        return any(str(key).lower() == "lambda" or contains_lambda(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_lambda(item) for item in value)
    return False


if contains_lambda(source):
    raise SystemExit("source calibration unexpectedly contains lambda")
print("SABRA_SOURCE=PASS GT_FREE_CACHE_FINALIZED=true medical_reads=0")
PY
fi

printf 'STAGE3_STATUS=PASS\n'
