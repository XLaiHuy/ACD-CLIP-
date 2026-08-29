#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

SOURCE_MODE="both"
DEVICE="${CIR_DEVICE:-0}"
SEED="${CIR_SEED:-0}"
RUN_ROOT="${CIR_RUN_ROOT:-$ROOT/runs/cir_rmt/CIR_DFG_RMT_V1}"
CONFIG="${CIR_CONFIG:-$ROOT/configs/cir_dfg_rmt_v1.json}"
CLIP_ASSET="${CIR_CLIP_ASSET:-}"
VISA_ROOT="${VISA_ROOT:-${ACDCLIP_VISA_ROOT:-}}"
MVTEC_ROOT="${MVTEC_ROOT:-${ACDCLIP_MVTEC_ROOT:-}}"
MEDICAL_ROOT="${MEDICAL_ROOT:-${ACDCLIP_DATA_ROOT:-}}"
RESUME="${CIR_RESUME:-1}"
DRY_RUN=0

usage() { echo "Usage: $0 [--source visa|mvtec|both] [--device 0|cuda:0] [--seed N] [--clip-asset PATH] [--dry-run]"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE_MODE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --clip-asset) CLIP_ASSET="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$SOURCE_MODE" in visa|mvtec) SOURCES=("$SOURCE_MODE") ;; both) SOURCES=(visa mvtec) ;; *) echo "--source must be visa, mvtec, or both" >&2; exit 2 ;; esac
[[ -n "$CLIP_ASSET" ]] || { echo "CIR_CLIP_ASSET or --clip-asset is required" >&2; exit 2; }
[[ -f "$CLIP_ASSET" ]] || { echo "missing CLIP asset: $CLIP_ASSET" >&2; exit 2; }
[[ "$DEVICE" == cuda:* || "$DEVICE" == cpu ]] || DEVICE="cuda:$DEVICE"
if [[ "$DRY_RUN" -eq 0 && "${RELEASE_LOCK:-FALSE}" != TRUE ]]; then echo "Refusing full CIR run: set RELEASE_LOCK=TRUE after G0-G5 pass." >&2; exit 3; fi

run_source() {
  local source="$1" source_root target target_root epoch base
  base="$RUN_ROOT/$source/seed$SEED"
  if [[ "$source" == visa ]]; then source_root="$VISA_ROOT"; else source_root="$MVTEC_ROOT"; fi
  [[ -n "$source_root" ]] || { echo "missing source root for $source" >&2; return 2; }
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "CIR/TRAIN-${source^^}: $source_root -> $base"
    echo "CIR/EVAL targets: $([[ "$source" == visa ]] && echo MVTec || echo VisA) plus Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir"
    return 0
  fi
  local train_args=(--source "$source" --source-root "$source_root" --clip-asset "$CLIP_ASSET" --config "$CONFIG" --run-root "$RUN_ROOT" --seed "$SEED" --epochs 20 --device "$DEVICE")
  if [[ "$RESUME" == 1 && -s "$base/last.pth" ]]; then train_args+=(--resume "$base/last.pth"); fi
  python -m scripts.cir_rmt.train_full "${train_args[@]}"
  for epoch in 12 14 16 18 20; do
    [[ -s "$base/checkpoints/epoch_$(printf '%02d' "$epoch").pth" ]] || { echo "missing checkpoint epoch $epoch for $source" >&2; return 4; }
  done
  if [[ "$source" == visa ]]; then target="MVTec"; target_root="$MVTEC_ROOT"; else target="VisA"; target_root="$VISA_ROOT"; fi
  [[ -n "$target_root" ]] || { echo "missing target root for $target" >&2; return 2; }
  for epoch in 12 14 16 18 20; do
    python -m scripts.cir_rmt.eval_full --source "$source" --target "$target" --target-root "$target_root" --checkpoint "$base/checkpoints/epoch_$(printf '%02d' "$epoch").pth" --clip-asset "$CLIP_ASSET" --config "$CONFIG" --output-dir "$base/eval/$target/epoch_$(printf '%02d' "$epoch")" --device "$DEVICE"
  done
  for target in Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir; do
    [[ -n "$MEDICAL_ROOT" ]] || { echo "missing medical root" >&2; return 2; }
    for epoch in 12 14 16 18 20; do
      python -m scripts.cir_rmt.eval_full --source "$source" --target "$target" --target-root "$MEDICAL_ROOT" --checkpoint "$base/checkpoints/epoch_$(printf '%02d' "$epoch").pth" --clip-asset "$CLIP_ASSET" --config "$CONFIG" --output-dir "$base/eval/$target/epoch_$(printf '%02d' "$epoch")" --device "$DEVICE"
    done
  done
  python -m tools.cir_rmt.summarize_results --run-root "$base"
}
for source in "${SOURCES[@]}"; do run_source "$source"; done
