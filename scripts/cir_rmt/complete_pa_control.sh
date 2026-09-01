#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

RUN_ROOT="$ROOT/runs/cir_rmt/CIR_DFG_RMT_V2/pa_control_20260831"
PA_RUN_ROOT="$RUN_ROOT"
ARCHIVE_ROOT="$ROOT/research_artifacts/cir_rmt_v2/pa_control_20260831"
PARENT_RUN_ROOT="$ROOT/runs/cir_rmt/CIR_DFG_RMT_V2/corrective_matched_retrain_20260830/parent"
OLD_CIR_RUN_ROOT="$ROOT/runs/cir_rmt/CIR_DFG_RMT_V2/corrective_matched_retrain_20260830/cir"
ANCHOR_RUN_ROOT="$ROOT/runs/cir_rmt/CIR_DFG_RMT_V2/matched_horizon_anchor_e14_20260831"
FROZEN_SOURCE_ARCHIVE="$ROOT/research_artifacts/cir_rmt_v2/final_extension_anchor_e20_20260831"
FROZEN_MEDICAL="$FROZEN_SOURCE_ARCHIVE/FINAL_MEDICAL_MATRIX.csv"
CONFIG="$ROOT/configs/phase2b_canonical_v1.json"
CIR_CONFIG="$ROOT/configs/cir_dfg_rmt_v2.json"
CLIP_ASSET="$ROOT/model/ViT-L-14-336px.pt"
SOURCE_ROOT="${VISA_ROOT:-/home/ai4/caohuy/data/VisA_20220922}"
MEDICAL_ROOT="${MEDICAL_ROOT:-/home/ai4/caohuy/data}"
MEDICAL_RUN_ROOT="$RUN_ROOT/medical_eval"
RESOURCE_JSON="$ARCHIVE_ROOT/PA_MEDICAL_RESOURCE_ADMISSION.json"
RESOURCE_SPOOL="$RUN_ROOT/medical_preflight_spool"

python -m tools.cir_rmt.verify_pa_training \
  --run-root "$RUN_ROOT" \
  --parent-run-root "$PARENT_RUN_ROOT" \
  --config "$CONFIG" \
  --anchor-checkpoint "$PARENT_RUN_ROOT/phase2b/checkpoints/adapter_14.pth" \
  --clip-asset "$CLIP_ASSET" \
  --source-root "$SOURCE_ROOT" \
  --output "$ARCHIVE_ROOT/PA_TRAINING_AUDIT.json"
python -m tools.cir_rmt.finalize_pa_training --run-root "$RUN_ROOT" --output-root "$ARCHIVE_ROOT"

PA_CONFIG_SHA="$(python -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["config_sha256"])' "$RUN_ROOT/visa/seed0/run_manifest.json")"
python -m tools.cir_rmt.pa_source_eval \
  --output-root "$ARCHIVE_ROOT" \
  --pa-run-root "$PA_RUN_ROOT" \
  --parent-run-root "$PARENT_RUN_ROOT" \
  --parent-config "$CONFIG" \
  --pa-config-sha256 "$PA_CONFIG_SHA" \
  --source-root "$SOURCE_ROOT" \
  --clip-asset "$CLIP_ASSET" \
  --device cuda:0 \
  --batch-size 6 \
  --num-workers 0

python -m tools.cir_rmt.finalize_pa_source \
  --output-root "$ARCHIVE_ROOT" \
  --frozen-source-archive "$FROZEN_SOURCE_ARCHIVE" \
  --parent-run-root "$PARENT_RUN_ROOT" \
  --pa-run-root "$PA_RUN_ROOT" \
  --old-cir-run-root "$OLD_CIR_RUN_ROOT" \
  --anchor-run-root "$ANCHOR_RUN_ROOT"

python tools/cir_rmt/resource_admission_preflight.py \
  --source-root "$SOURCE_ROOT" \
  --medical-root "$MEDICAL_ROOT" \
  --cir-checkpoint "$ANCHOR_RUN_ROOT/visa/seed0/checkpoints/epoch_20.pth" \
  --clip-asset "$CLIP_ASSET" \
  --config "$CIR_CONFIG" \
  --output "$RESOURCE_JSON" \
  --spool-root "$RESOURCE_SPOOL" \
  --batch-size 6 \
  --num-workers 4 \
  --prefetch-factor 2

python -m tools.cir_rmt.pa_medical_eval \
  --freeze "$ARCHIVE_ROOT/PRE_PA_MEDICAL_FREEZE.json" \
  --resource-admission "$RESOURCE_JSON" \
  --output-root "$MEDICAL_RUN_ROOT" \
  --pa-run-root "$PA_RUN_ROOT" \
  --parent-config "$CONFIG" \
  --source-root "$SOURCE_ROOT" \
  --medical-root "$MEDICAL_ROOT" \
  --clip-asset "$CLIP_ASSET" \
  --device cuda:0 \
  --batch-size 6 \
  --num-workers 4 \
  --prefetch-factor 2 \
  --resume

python -m tools.cir_rmt.finalize_pa_medical \
  --output-root "$ARCHIVE_ROOT" \
  --medical-run-root "$MEDICAL_RUN_ROOT" \
  --pa-run-root "$PA_RUN_ROOT" \
  --frozen-medical "$FROZEN_MEDICAL"

python -m tools.cir_rmt.hash_pa_artifacts \
  --archive "$ARCHIVE_ROOT" \
  --repo-root "$ROOT" \
  --script scripts/cir_rmt/train_pa.py \
  --script scripts/cir_rmt/run_pa_control.sh \
  --script scripts/cir_rmt/complete_pa_control.sh \
  --script tools/cir_rmt/parameter_anchor.py \
  --script tools/cir_rmt/verify_pa_training.py \
  --script tools/cir_rmt/pa_source_eval.py \
  --script tools/cir_rmt/finalize_pa_source.py \
  --script tools/cir_rmt/pa_medical_eval.py \
  --script tools/cir_rmt/finalize_pa_medical.py \
  --script tools/cir_rmt/finalize_pa_training.py \
  --script tools/cir_rmt/hash_pa_artifacts.py

echo "PA_CONTROL_POSTPROCESS_STATUS PASS"
