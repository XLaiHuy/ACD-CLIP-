#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
SOURCE="${CIR_SMOKE_SOURCE:-visa}"
if [[ "$SOURCE" == visa ]]; then SOURCE_ROOT="${VISA_ROOT:-${ACDCLIP_VISA_ROOT:-}}"; else SOURCE_ROOT="${MVTEC_ROOT:-${ACDCLIP_MVTEC_ROOT:-}}"; fi
CLIP_ASSET="${CIR_CLIP_ASSET:-}"
DEVICE="${CIR_DEVICE:-cpu}"
RUN_ROOT="${CIR_SMOKE_RUN_ROOT:-$ROOT/runs/cir_rmt/CIR_DFG_RMT_V1}"
SMOKE_STEPS="${CIR_SMOKE_STEPS:-50}"
GIT_SHA="$(git rev-parse HEAD)"
[[ -n "$CLIP_ASSET" && -f "$CLIP_ASSET" ]] || { echo "CIR_CLIP_ASSET is required" >&2; exit 2; }
[[ -n "$SOURCE_ROOT" && -d "$SOURCE_ROOT" ]] || { echo "source root is required for $SOURCE" >&2; exit 2; }
[[ "$DEVICE" == cuda:* || "$DEVICE" == cpu ]] || DEVICE="cuda:$DEVICE"
python -m scripts.cir_rmt.train_full --source "$SOURCE" --source-root "$SOURCE_ROOT" --clip-asset "$CLIP_ASSET" --config configs/cir_dfg_rmt_v1.json --run-root "$RUN_ROOT" --seed "${CIR_SEED:-0}" --epochs 2 --device "$DEVICE" --git-sha "$GIT_SHA" --smoke-steps "$SMOKE_STEPS"
LAST="$RUN_ROOT/$SOURCE/seed${CIR_SEED:-0}/last.pth"
SMOKE_JSON="$RUN_ROOT/$SOURCE/seed${CIR_SEED:-0}/G5_SMOKE.json"
test -s "$LAST"
test -s "$SMOKE_JSON"
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "PASS" and int(p["steps_completed"]) >= int(sys.argv[2]) and p["checkpoint_resume_contract"]' "$SMOKE_JSON" "$SMOKE_STEPS"
# A one-step resumed invocation exercises identity validation, state restore,
# and a real post-resume optimizer step without becoming a training run.
python -m scripts.cir_rmt.train_full --source "$SOURCE" --source-root "$SOURCE_ROOT" --clip-asset "$CLIP_ASSET" --config configs/cir_dfg_rmt_v1.json --run-root "$RUN_ROOT" --seed "${CIR_SEED:-0}" --epochs 3 --resume "$LAST" --device "$DEVICE" --git-sha "$GIT_SHA" --smoke-steps 1
echo "CIR/G5-SMOKE PASS: $SMOKE_STEPS optimizer-step path plus one resumed step completed; checkpoint is resumable at $LAST"
