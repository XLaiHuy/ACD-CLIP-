#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

RUN_ROOT="${CIR_MATCHED_RUN_ROOT:-$ROOT/runs/cir_rmt/CIR_DFG_RMT_V2/matched_horizon_anchor_e14_20260831}"
ARCHIVE_ROOT="${CIR_MATCHED_ARCHIVE_ROOT:-$ROOT/research_artifacts/cir_rmt_v2/matched_horizon_anchor_e14_20260831}"
CONFIG="${CIR_CONFIG:-$ROOT/configs/cir_dfg_rmt_v2.json}"
CLIP_ASSET="${CIR_CLIP_ASSET:-$ROOT/model/ViT-L-14-336px.pt}"
VISA_ROOT="${VISA_ROOT:-${ACDCLIP_VISA_ROOT:-/home/ai4/caohuy/data/VisA_20220922}}"
DEVICE="${CIR_DEVICE:-cuda:0}"
SEED="${CIR_SEED:-0}"
ANCHOR_CHECKPOINT="${CIR_ANCHOR_CHECKPOINT:-$ROOT/runs/cir_rmt/CIR_DFG_RMT_V2/corrective_matched_retrain_20260830/parent/phase2b/checkpoints/adapter_14.pth}"
PARENT_ARCHIVE="${CIR_PARENT_ARCHIVE:-$ROOT/research_artifacts/cir_rmt_v2/pre_full_run_root_cause_lock_20260831}"

[[ "$DEVICE" == cuda:* || "$DEVICE" == cpu ]] || DEVICE="cuda:$DEVICE"
[[ -d "$VISA_ROOT" ]] || { echo "VISA_ROOT is required for the matched source run" >&2; exit 2; }
[[ -f "$CLIP_ASSET" ]] || { echo "missing CLIP asset: $CLIP_ASSET" >&2; exit 2; }
[[ -f "$ANCHOR_CHECKPOINT" ]] || { echo "missing Phase2B E14 anchor: $ANCHOR_CHECKPOINT" >&2; exit 2; }
[[ -s "$PARENT_ARCHIVE/SOURCE_BOUNDED_METRICS.csv" ]] || { echo "frozen P/C0 source artifact is required" >&2; exit 2; }
[[ -s "$PARENT_ARCHIVE/SOURCE_SAMPLE_IDENTITY.json" ]] || { echo "frozen source sample identity is required" >&2; exit 2; }

EXPECTED_ANCHOR_SHA="3eb6e2fe12f96b84745baf0f8a013f88c7f3a739283493a2ba5e31a35ad2f6c2"
ACTUAL_ANCHOR_SHA="$(sha256sum "$ANCHOR_CHECKPOINT" | awk '{print $1}')"
[[ "$ACTUAL_ANCHOR_SHA" == "$EXPECTED_ANCHOR_SHA" ]] || { echo "anchor checkpoint hash mismatch: $ACTUAL_ANCHOR_SHA" >&2; exit 3; }

BASE="$RUN_ROOT/visa/seed$SEED"
CHECKPOINT_ROOT="$BASE/checkpoints"
LAST="$BASE/last.pth"
TRAIN_LOG="$BASE/train.log"
mkdir -p "$BASE" "$ARCHIVE_ROOT"
[[ -s "$ARCHIVE_ROOT/ANCHOR_OVERHEAD_PROFILE.json" ]] || { echo "one-time anchor overhead profile is required before training" >&2; exit 16; }
exec 9>"$RUN_ROOT/.process.lock"
flock -n 9 || { echo "matched-horizon run lock is held: $RUN_ROOT/.process.lock" >&2; exit 11; }

GIT_SHA="$(git rev-parse HEAD)"
TRAIN_ARGS=(
  --source visa
  --source-root "$VISA_ROOT"
  --clip-asset "$CLIP_ASSET"
  --config "$CONFIG"
  --run-root "$RUN_ROOT"
  --seed "$SEED"
  --epochs 14
  --device "$DEVICE"
  --git-sha "$GIT_SHA"
  --matched-horizon-e14
  --quiet-progress
  --image-anchor-checkpoint "$ANCHOR_CHECKPOINT"
  --image-anchor-lambda 0.001
  --micro-batch-size 6
  --grad-accum-steps 1
  --num-workers 0
  --no-persistent-workers
)

RESUME_EPOCH=-1
if [[ -s "$LAST" ]]; then
  RESUME_EPOCH="$(python -c 'import sys,torch; print(int(torch.load(sys.argv[1],map_location="cpu",weights_only=False).get("epoch",-1)))' "$LAST")"
fi

NEED_TRAIN=1
if [[ "$RESUME_EPOCH" -ge 14 && -s "$CHECKPOINT_ROOT/epoch_10.pth" && -s "$CHECKPOINT_ROOT/epoch_12.pth" && -s "$CHECKPOINT_ROOT/epoch_14.pth" ]]; then
  NEED_TRAIN=0
elif [[ "$RESUME_EPOCH" -ge 14 ]]; then
  echo "E14 resume cursor exists but a required candidate checkpoint is missing" >&2
  exit 12
elif [[ "$RESUME_EPOCH" -ge 1 ]]; then
  TRAIN_ARGS+=(--resume "$LAST")
fi

if [[ "$NEED_TRAIN" -eq 1 ]]; then
  echo "MATCHED_HORIZON_LAUNCH source=visa max_epoch=14 candidate_epochs=10,12,14 seed=$SEED device=$DEVICE git=$GIT_SHA"
  python -m scripts.cir_rmt.train_full "${TRAIN_ARGS[@]}" 2>&1 | tee "$TRAIN_LOG"
else
  echo "MATCHED_HORIZON_RESUME_COMPLETE existing_epoch=$RESUME_EPOCH"
fi

for epoch in 10 12 14; do
  checkpoint="$CHECKPOINT_ROOT/epoch_$(printf '%02d' "$epoch").pth"
  [[ -s "$checkpoint" ]] || { echo "missing matched-horizon checkpoint E$epoch: $checkpoint" >&2; exit 13; }
done
[[ ! -e "$CHECKPOINT_ROOT/epoch_15.pth" && ! -e "$CHECKPOINT_ROOT/epoch_16.pth" ]] || { echo "matched-horizon runner produced a checkpoint after E14" >&2; exit 14; }
[[ -s "$BASE/E10_CATASTROPHIC_FAILURE_GATE.json" ]] || { echo "E10 catastrophic-failure gate artifact missing" >&2; exit 15; }
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "PASS", p' "$BASE/E10_CATASTROPHIC_FAILURE_GATE.json"

echo "MATCHED_HORIZON_POSTHOC_SOURCE_EVAL epochs=10,12,14 reuse_existing_p_c0=true"
python -m tools.cir_rmt.matched_horizon_source_eval \
  --archive-root "$ARCHIVE_ROOT" \
  --baseline-archive-root "$PARENT_ARCHIVE" \
  --run-root "$RUN_ROOT" \
  --source-root "$VISA_ROOT" \
  --clip-asset "$CLIP_ASSET" \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --batch-size 6 \
  --num-workers 0

python -m tools.cir_rmt.finalize_matched_horizon \
  --run-root "$RUN_ROOT" \
  --archive-root "$ARCHIVE_ROOT" \
  --config "$CONFIG" \
  --clip-asset "$CLIP_ASSET" \
  --anchor-checkpoint "$ANCHOR_CHECKPOINT" \
  --source-root "$VISA_ROOT" \
  --git-sha "$GIT_SHA"

echo "MATCHED_HORIZON_STATUS PASS: E10/E12/E14 trained, saved, gated, and source-evaluated; no P/C0 recomputation or target evaluation"
