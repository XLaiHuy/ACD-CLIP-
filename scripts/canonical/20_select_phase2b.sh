#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

init_common "$@"
((${#COMMON_ARGS[@]} == 0)) || die "unknown checkpoint-selection arguments: ${COMMON_ARGS[*]}"
require_base_assets
require_mvtec_root

banner "STAGE 2 — MVTecAD PHASE2B CHECKPOINT SELECTION"

checkpoint_dir="$RUN_ROOT/phase2b/checkpoints"
selection_dir="$RUN_ROOT/phase2b_selection"
selection_json="$selection_dir/phase2b_selection.json"
for epoch in 10 12 14 16 18 20; do
  require_file "$checkpoint_dir/adapter_${epoch}.pth"
done
require_clean_stage_output "$selection_json"

select_cmd=(
  "$PYTHON" "$REPO_ROOT/select_phase2b_checkpoint.py"
  --checkpoint-dir "$checkpoint_dir"
  --candidate-epochs "10,12,14,16,18,20"
  --mvtec-root "$MVTEC_ROOT"
  --config "$CONFIG"
  --clip-asset "$CLIP_ASSET"
  --device cuda
  --batch-size 6
  --num-workers 4
  --output-dir "$selection_dir"
)
run_logged "$RUN_ROOT/logs/phase2b_select.log" "${select_cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  printf '[canonical] DRY_RUN: would validate %s as FROZEN and verify selected checkpoint SHA256\n' "$selection_json"
else
  require_file "$selection_json"
  require_file "$selection_dir/phase2b_selection_metrics.csv"
  "$PYTHON" - "$selection_json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

selection_path = Path(sys.argv[1])
payload = json.loads(selection_path.read_text(encoding="utf-8"))
if payload.get("status") != "FROZEN":
    raise SystemExit("Phase2B selection status is not FROZEN")
if int(payload.get("selected_epoch", -1)) not in {10, 12, 14, 16, 18, 20}:
    raise SystemExit("selected epoch is outside the canonical candidate set")
checkpoint = Path(str(payload.get("selected_checkpoint", ""))).expanduser()
expected = str(payload.get("selected_checkpoint_sha256", ""))
if not checkpoint.is_file() or not expected:
    raise SystemExit("selection lacks a valid selected checkpoint and SHA256")
digest = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(f"selected checkpoint SHA256 mismatch: {expected} != {actual}")
print(f"PHASE2B_SELECTION=FROZEN E*={int(payload['selected_epoch'])} SHA256={actual}")
PY
fi

printf 'STAGE2_STATUS=PASS\n'
