#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

init_common "$@"
((${#COMMON_ARGS[@]} == 0)) || die "unknown lambda-selection arguments: ${COMMON_ARGS[*]}"
require_base_assets
require_mvtec_root

banner "STAGE 4 — MVTecAD SABRA LAMBDA SELECTION"

selection_json="$RUN_ROOT/phase2b_selection/phase2b_selection.json"
source_json="$RUN_ROOT/sabra_source/sabra_source_calibration.json"
lambda_dir="$RUN_ROOT/sabra_lambda"
freeze_json="$lambda_dir/SABRA_FREEZE.json"
require_file "$selection_json"
require_file "$source_json"
require_clean_stage_output "$freeze_json"

"$PYTHON" - "$selection_json" "$source_json" "$CANONICAL_SHA" <<'PY'
import hashlib
import sys
from pathlib import Path

from tools.sabra.artifacts import load_json, validate_source_calibration

selection = load_json(sys.argv[1])
source = load_json(sys.argv[2])
expected_code_sha = sys.argv[3]
if source.get("provenance", {}).get("git_sha") != expected_code_sha:
    raise SystemExit("source calibration provenance SHA differs from canonical HEAD")
if selection.get("status") != "FROZEN":
    raise SystemExit("Phase2B selection must be FROZEN before lambda selection")
validate_source_calibration(source)
if source.get("relational", {}).get("backend") != "fast":
    raise SystemExit("lambda selection requires source backend=fast")
if source["phase2b"]["checkpoint_sha256"] != selection["selected_checkpoint_sha256"]:
    raise SystemExit("source calibration checkpoint SHA256 differs from Phase2B selection")
checkpoint = Path(str(selection["selected_checkpoint"])).expanduser()
digest = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
if digest.hexdigest() != selection["selected_checkpoint_sha256"]:
    raise SystemExit("selected checkpoint SHA256 mismatch")
PY

lambda_cmd=(
  "$PYTHON" "$REPO_ROOT/calibrate_sabra.py" select-lambda
  --source-calibration "$source_json"
  --mvtec-root "$MVTEC_ROOT"
  --output-dir "$lambda_dir"
  --config "$CONFIG"
  --clip-asset "$CLIP_ASSET"
  --device cuda
  --batch-size 6
  --num-workers 4
  --prefetch-factor 2
  --lambda-chunk-size 8
  --backend fast
  --git-sha "$CANONICAL_SHA"
)
run_logged "$RUN_ROOT/logs/sabra_lambda.log" "${lambda_cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[canonical] DRY_RUN: would validate SABRA_FREEZE.json and lambda runtime provenance\n'
else
  require_file "$lambda_dir/lambda_selection.csv"
  require_file "$lambda_dir/lambda_runtime.json"
  require_file "$freeze_json"
  "$PYTHON" - "$selection_json" "$source_json" "$freeze_json" "$CANONICAL_SHA" <<'PY'
import sys

from tools.sabra.artifacts import load_json, validate_sabra_freeze, validate_source_calibration

selection = load_json(sys.argv[1])
source = load_json(sys.argv[2])
freeze = load_json(sys.argv[3])
expected_code_sha = sys.argv[4]
if freeze.get("provenance", {}).get("git_sha") != expected_code_sha:
    raise SystemExit("SABRA freeze provenance SHA differs from canonical HEAD")
validate_source_calibration(source)
validate_sabra_freeze(freeze, checkpoint_sha256=selection["selected_checkpoint_sha256"])
if freeze["phase2b"]["selected_epoch"] != selection["selected_epoch"]:
    raise SystemExit("SABRA freeze selected epoch differs from Phase2B selection")
if freeze["relational"].get("backend") != "fast" or source["relational"].get("backend") != "fast":
    raise SystemExit("source, lambda, and final backend must all be fast")
correction = freeze["correction"]
if correction.get("authority") != "T*N":
    raise SystemExit("SABRA authority is not T*N")
if correction.get("direction") != "positive_abnormal_only":
    raise SystemExit("SABRA correction direction mismatch")
if correction.get("normal_delta") != 0 or correction.get("shared_across_stages") is not True:
    raise SystemExit("SABRA stage-sharing/normal-channel contract mismatch")
value = float(correction["lambda"])
if not 0.0 <= value <= 1.0:
    raise SystemExit("selected lambda is outside [0,1]")
if freeze.get("medical_seen") is not False:
    raise SystemExit("SABRA freeze records Medical data")
print(f"SABRA_FREEZE=FROZEN lambda={value:.6f} backend=fast")
PY
fi

printf 'STAGE4_STATUS=PASS\n'
