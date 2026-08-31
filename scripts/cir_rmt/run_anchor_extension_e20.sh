#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_ROOT="${CIR_EXTENSION_RUN_ROOT:-$ROOT/runs/cir_rmt/CIR_DFG_RMT_V2/matched_horizon_anchor_e14_20260831}"
ARCHIVE_ROOT="${CIR_EXTENSION_ARCHIVE_ROOT:-$ROOT/research_artifacts/cir_rmt_v2/final_extension_anchor_e20_20260831}"
CONFIG="${CIR_CONFIG:-$ROOT/configs/cir_dfg_rmt_v2.json}"
CLIP_ASSET="${CIR_CLIP_ASSET:-$ROOT/model/ViT-L-14-336px.pt}"
VISA_ROOT="${VISA_ROOT:-${ACDCLIP_VISA_ROOT:-/home/ai4/caohuy/data/VisA_20220922}}"
DEVICE="${CIR_DEVICE:-cuda:0}"
SEED="${CIR_SEED:-0}"
ANCHOR_CHECKPOINT="${CIR_ANCHOR_CHECKPOINT:-$ROOT/runs/cir_rmt/CIR_DFG_RMT_V2/corrective_matched_retrain_20260830/parent/phase2b/checkpoints/adapter_14.pth}"
EXPECTED_ANCHOR_SHA="3eb6e2fe12f96b84745baf0f8a013f88c7f3a739283493a2ba5e31a35ad2f6c2"
EXPECTED_CLIP_SHA="3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
EXPECTED_E14_SHA="9b8fc5e7760037e772c9bd63d98ce56fcbbaa04f021258ca0d23aa8f2bf5ab81"

if [[ "$DEVICE" =~ ^[0-9]+$ ]]; then
  DEVICE="cuda:$DEVICE"
fi

[[ -d "$VISA_ROOT" ]] || { echo "missing VisA root: $VISA_ROOT" >&2; exit 2; }
[[ -s "$CLIP_ASSET" ]] || { echo "missing CLIP asset: $CLIP_ASSET" >&2; exit 2; }
[[ -s "$ANCHOR_CHECKPOINT" ]] || { echo "missing E14 Phase2B anchor: $ANCHOR_CHECKPOINT" >&2; exit 2; }
[[ "$(sha256sum "$CLIP_ASSET" | awk '{print $1}')" == "$EXPECTED_CLIP_SHA" ]] || { echo "CLIP asset SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "$ANCHOR_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_ANCHOR_SHA" ]] || { echo "anchor SHA mismatch" >&2; exit 3; }

BASE="$RUN_ROOT/visa/seed$SEED"
CHECKPOINT_ROOT="$BASE/checkpoints"
LAST="$BASE/last.pth"
MANIFEST="$BASE/run_manifest.json"
mkdir -p "$ARCHIVE_ROOT"
[[ -s "$LAST" && -s "$MANIFEST" ]] || { echo "completed E14 anchor cursor/manifest is required" >&2; exit 4; }

exec 9>"$RUN_ROOT/.process.lock"
flock -n 9 || { echo "anchor extension lock is held: $RUN_ROOT/.process.lock" >&2; exit 11; }

TRAINING_GIT_SHA="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_sha"])' "$MANIFEST")"
RESUME_EPOCH="$(python -c 'import sys,torch; print(int(torch.load(sys.argv[1],map_location="cpu",weights_only=False).get("epoch",-1)))' "$LAST")"
[[ "$RESUME_EPOCH" -ge 14 ]] || { echo "resume cursor is before E14: $RESUME_EPOCH" >&2; exit 5; }

if [[ ! -s "$ARCHIVE_ROOT/E14_PRE_EXTENSION_RUN_MANIFEST.json" ]]; then
  cp "$MANIFEST" "$ARCHIVE_ROOT/E14_PRE_EXTENSION_RUN_MANIFEST.json"
fi

if [[ "$RESUME_EPOCH" -lt 20 ]]; then
  echo "ANCHOR_EXTENSION_LAUNCH resume_epoch=$RESUME_EPOCH next_epoch=$((RESUME_EPOCH + 1)) target=20 training_git=$TRAINING_GIT_SHA"
  python -m scripts.cir_rmt.train_full \
    --source visa \
    --source-root "$VISA_ROOT" \
    --clip-asset "$CLIP_ASSET" \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --seed "$SEED" \
    --epochs 20 \
    --resume "$LAST" \
    --device "$DEVICE" \
    --git-sha "$TRAINING_GIT_SHA" \
    --matched-extension-e20 \
    --quiet-progress \
    --image-anchor-checkpoint "$ANCHOR_CHECKPOINT" \
    --image-anchor-lambda 0.001 \
    --micro-batch-size 6 \
    --grad-accum-steps 1 \
    --num-workers 0 \
    --no-persistent-workers \
    2>&1 | tee -a "$BASE/train_extension.log"
else
  echo "ANCHOR_EXTENSION_RESUME_COMPLETE existing_epoch=$RESUME_EPOCH"
fi

python -m tools.cir_rmt.verify_anchor_extension \
  --run-root "$RUN_ROOT" \
  --config "$CONFIG" \
  --anchor-checkpoint "$ANCHOR_CHECKPOINT" \
  --expected-anchor-sha256 "$EXPECTED_ANCHOR_SHA" \
  --training-git-sha "$TRAINING_GIT_SHA" \
  --output "$ARCHIVE_ROOT/EXTENSION_TRAINING_AUDIT.json"

echo "ANCHOR_EXTENSION_STATUS PASS: E14 cursor extended to E20 and audited"
