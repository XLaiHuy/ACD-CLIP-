#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

RUN_ROOT="${PA_RUN_ROOT:-$ROOT/runs/cir_rmt/CIR_DFG_RMT_V2/pa_control_20260831}"
ARCHIVE_ROOT="${PA_ARCHIVE_ROOT:-$ROOT/research_artifacts/cir_rmt_v2/pa_control_20260831}"
CONFIG="${PA_CONFIG:-$ROOT/configs/phase2b_canonical_v1.json}"
CLIP_ASSET="${PA_CLIP_ASSET:-$ROOT/model/ViT-L-14-336px.pt}"
VISA_ROOT="${VISA_ROOT:-${ACDCLIP_VISA_ROOT:-/home/ai4/caohuy/data/VisA_20220922}}"
DEVICE="${PA_DEVICE:-cuda:0}"
SEED="${PA_SEED:-0}"
ANCHOR_CHECKPOINT="${PA_ANCHOR_CHECKPOINT:-$ROOT/runs/cir_rmt/CIR_DFG_RMT_V2/corrective_matched_retrain_20260830/parent/phase2b/checkpoints/adapter_14.pth}"
PARENT_RUN_ROOT="${PA_PARENT_RUN_ROOT:-$ROOT/runs/cir_rmt/CIR_DFG_RMT_V2/corrective_matched_retrain_20260830/parent}"

EXPECTED_CLIP_SHA="3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
EXPECTED_ANCHOR_SHA="3eb6e2fe12f96b84745baf0f8a013f88c7f3a739283493a2ba5e31a35ad2f6c2"
EXPECTED_PARENT_CONFIG_SHA="d24cf942684b0be3c12838699ec6fe452697bd7f0a58eabbf316fb79b1b18cdb"

if [[ "$DEVICE" =~ ^[0-9]+$ ]]; then
  DEVICE="cuda:$DEVICE"
fi

[[ -d "$VISA_ROOT" ]] || { echo "missing VisA root: $VISA_ROOT" >&2; exit 2; }
[[ -s "$CLIP_ASSET" ]] || { echo "missing CLIP asset: $CLIP_ASSET" >&2; exit 2; }
[[ -s "$ANCHOR_CHECKPOINT" ]] || { echo "missing P_E14 anchor: $ANCHOR_CHECKPOINT" >&2; exit 2; }
[[ "$(sha256sum "$CLIP_ASSET" | awk '{print $1}')" == "$EXPECTED_CLIP_SHA" ]] || { echo "CLIP SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "$ANCHOR_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_ANCHOR_SHA" ]] || { echo "P_E14 anchor SHA mismatch" >&2; exit 3; }

python - "$CONFIG" "$EXPECTED_PARENT_CONFIG_SHA" <<'PY'
import hashlib, json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if actual != sys.argv[2]:
    raise SystemExit(f"parent config compact SHA mismatch: {actual}")
PY

BASE="$RUN_ROOT/visa/seed$SEED"
CHECKPOINT_ROOT="$BASE/checkpoints"
LAST="$BASE/last.pth"
MANIFEST="$BASE/run_manifest.json"
TRAIN_LOG="$BASE/train.log"
RECOVERY_LOG="$BASE/RECOVERY_LOG.md"
mkdir -p "$BASE" "$ARCHIVE_ROOT"
exec 9>"$RUN_ROOT/.process.lock"
flock -n 9 || { echo "PA control lock is held: $RUN_ROOT/.process.lock" >&2; exit 11; }

if [[ -s "$LAST" ]]; then
  RESUME_EPOCH="$(python -c 'import sys,torch; print(int(torch.load(sys.argv[1],map_location="cpu",weights_only=False).get("epoch",-1)))' "$LAST")"
else
  RESUME_EPOCH=-1
fi

if [[ "$RESUME_EPOCH" -ge 20 ]]; then
  echo "PA_CONTROL_RESUME_COMPLETE existing_epoch=$RESUME_EPOCH"
else
  GIT_SHA="$(git rev-parse HEAD)"
  TRAIN_ARGS=(
    --visa-root "$VISA_ROOT"
    --clip-asset "$CLIP_ASSET"
    --config "$CONFIG"
    --run-root "$RUN_ROOT"
    --image-anchor-checkpoint "$ANCHOR_CHECKPOINT"
    --image-anchor-lambda 0.001
    --seed "$SEED"
    --epochs 20
    --device "$DEVICE"
    --git-sha "$GIT_SHA"
    --micro-batch-size 6
    --grad-accum-steps 1
    --num-workers 4
    --prefetch-factor 2
    --pin-memory
    --persistent-workers
    --quiet-progress
  )
  if [[ "$RESUME_EPOCH" -ge 1 ]]; then
    TRAIN_ARGS+=(--resume "$LAST")
  fi

  attempt=0
  while true; do
    attempt=$((attempt + 1))
    echo "PA_CONTROL_LAUNCH attempt=$attempt resume_epoch=$RESUME_EPOCH next_epoch=$((RESUME_EPOCH + 1)) target=20 git=$GIT_SHA"
    set +e
    python -m scripts.cir_rmt.train_pa "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$TRAIN_LOG"
    python_status=${PIPESTATUS[0]}
    set -e
    if [[ "$python_status" -eq 0 ]]; then
      break
    fi
    if [[ "$attempt" -ge 2 ]]; then
      {
        echo "# PA control recovery log"
        echo
        echo "- status: STOPPED_AFTER_RECOVERY_LIMIT"
        echo "- attempts: $attempt"
        echo "- last_exit_status: $python_status"
        echo "- scientific_changes: none"
        echo "- action: human review required"
      } > "$RECOVERY_LOG"
      exit "$python_status"
    fi
    if [[ ! -s "$LAST" ]]; then
      echo "PA_CONTROL_RETRY no valid last.pth; retrying identical E1 protocol" | tee -a "$RECOVERY_LOG"
    else
      RESUME_EPOCH="$(python -c 'import sys,torch; print(int(torch.load(sys.argv[1],map_location="cpu",weights_only=False).get("epoch",-1)))' "$LAST")"
      TRAIN_ARGS+=(--resume "$LAST")
      echo "PA_CONTROL_AUTO_RESUME failure_attempt=$attempt resume_epoch=$RESUME_EPOCH" | tee -a "$RECOVERY_LOG"
    fi
  done
fi

for epoch in 10 12 14 16 18 20; do
  checkpoint="$CHECKPOINT_ROOT/adapter_${epoch}.pth"
  [[ -s "$checkpoint" ]] || { echo "missing PA checkpoint E$epoch: $checkpoint" >&2; exit 12; }
done
[[ -s "$MANIFEST" ]] || { echo "missing PA run manifest" >&2; exit 13; }
python - "$MANIFEST" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
if p.get("status") != "COMPLETED":
    raise SystemExit(f"PA manifest status is not COMPLETED: {p.get('status')!r}")
if p.get("control_id") != "PA_PHASE2B_IMAGE_ANCHOR_V1":
    raise SystemExit("PA control identity mismatch")
PY

python -m tools.cir_rmt.verify_pa_training \
  --run-root "$RUN_ROOT" \
  --parent-run-root "$PARENT_RUN_ROOT" \
  --config "$CONFIG" \
  --anchor-checkpoint "$ANCHOR_CHECKPOINT" \
  --clip-asset "$CLIP_ASSET" \
  --source-root "$VISA_ROOT" \
  --output "$ARCHIVE_ROOT/PA_TRAINING_AUDIT.json"

echo "PA_CONTROL_STATUS PASS: native Phase2B + image anchor trained E1-E20 and passed identity/scheduler/resume audit"
