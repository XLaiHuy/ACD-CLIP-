#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$BASH_SOURCE")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
ALLOW_DIRTY=0
DEVICE="${CIR_DEVICE:-cpu}"
OUT="${CIR_AUDIT_OUT:-/tmp/cir_rmt_audit}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --output) OUT="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--allow-dirty] [--device cpu|cuda:0] [--output DIR]"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT"
identity_args=(--config configs/cir_dfg_rmt_v1.json --output "$OUT/g0_identity.json")
if [[ "$ALLOW_DIRTY" -eq 1 ]]; then identity_args+=(--allow-dirty); fi
python -m tools.cir_rmt.audit_identity "${identity_args[@]}"
python -m pytest tests/cir_rmt -q
python -m tools.cir_rmt.audit_numerics --output "$OUT/g1_math.json"
python -m tools.cir_rmt.audit_parity --config configs/cir_dfg_rmt_v1.json --output "$OUT/g2_parity.json"
python -m tools.cir_rmt.audit_preflight --output "$OUT/g3_preflight.json"
python -m tools.cir_rmt.profile_runtime --config configs/cir_dfg_rmt_v1.json --device "$DEVICE" --output "$OUT/g4_profile.json"
echo "CIR/G0-IDENTITY PASS"
echo "CIR/G1-MATH PASS"
echo "CIR/G2-PARITY BLOCKED/NOT_RUN (real CLIP/input/checkpoint unavailable)"
echo "CIR/G3-PREFLIGHT PARTIAL (synthetic-only PASS; real source preflight NOT_RUN)"
echo "CIR/G4-PROFILE PARTIAL (CPU micro-profile only; GPU latency/VRAM NOT_RUN)"
echo "CIR/G5-SMOKE BLOCKED/NOT_RUN (real source dataset/CLIP asset unavailable)"
