#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/workspace/venv-acdclip/bin/python}"
SMOKE_ROOT="${SMOKE_ROOT:-/workspace/h2_mixed_fp16_fp32_v1_smoke}"
SMOKE_BATCHES="${SMOKE_BATCHES:-5}"
RUN_SMOKE="${RUN_SMOKE:-NO}"
TRACE="${ROOT}/audit/H2_MIXED_FP16_FP32_V1_DTYPE_TRACE.json"
ANCHOR_LAMBDA="0.0021633926715180626"

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"${PYTHON}" - "${ROOT}/audit/H2_MIXED_FP16_FP32_V1_PARITY.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
bad = [row for row in rows if row["result"] not in {"PASS", "NO"}]
required = {
    "ARCHITECTURE_PARITY", "ANCHOR_PARITY", "DFG_PARITY", "SS2D_PARITY",
    "PROMPT_PARITY", "OPTIMIZER_PARITY", "LR_PARITY", "SCHEDULER_PARITY",
    "LOSS_PARITY", "DATA_PARITY", "AUGMENTATION_PARITY", "BATCH_PARITY",
    "HORIZON_PARITY", "FP16_AUTOCAST", "GRADSCALER_ON", "BF16_DISABLED",
    "DFG_FP32_RESIDUAL", "HISTORICAL_TRANSFORMER_PATH",
    "LATER_FP32_TRANSFORMER_ISLANDS_ACTIVE",
}
names = {row["gate"] for row in rows}
assert not bad and required <= names, (bad, sorted(required - names))
print("STATIC_PARITY_GATE=PASS")
PY

if [[ "${RUN_SMOKE}" != "YES" ]]; then
  echo "Prepared only. Set RUN_SMOKE=YES to run the bounded 30-batch source smoke."
  exit 0
fi
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'
[[ ! -e "${SMOKE_ROOT}" ]] || { echo "refusing existing ${SMOKE_ROOT}" >&2; exit 2; }
mkdir -p "${SMOKE_ROOT}"

base=(
  "${ROOT}/train.py" --protocol_horizon 15 --dataset VisA
  --model_name ViT-L-14-336 --img_size 518 --n_groups 3
  --image_adapt_weight .2 --text_adapt_weight .2
  --conv_lora_rank 8 --conv_lora_alpha 2 --conv_kernel_size_list 3 5
  --lora_rank 16 --lora_alpha 2 --image_lr .001 --text_lr .0005
  --use_hybrid_soft_prompt --hybrid_alpha_max .2 --soft_prompt_freeze_epochs 3
  --soft_prompt_ctx_len 4 --soft_prompt_lr .00005 --soft_prompt_init phrase
  --soft_prompt_init_phrase "a photo of a" --lambda_kg .01 --lambda_k .002
  --lr_gamma .9 --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8
  --use_ss2d_dfg --dfg_gamma_max .2 --dfg_ss2d_fusion weight_residual
  --dfg_beta .10 --dfg_beta_schedule warmup010 --dfg_beta_target .10
  --dfg_weight_residual_fp32 --grad_clip_norm 1 --batch_size 6
  --num_workers 6 --pin_memory --no-persistent_workers --prefetch_factor 2
  --no-non_blocking_copy --grad_checkpointing
  --precision_protocol HISTORICAL_MIXED_FP16_FP32_V1
  --seed 0 --deterministic_algorithms --non_finite_loss_abort_threshold 20
  --trace_batch_identity --telemetry_interval 1 --family_telemetry_interval 1
)

"${PYTHON}" "${base[@]}" --epoch 1 --max_batches "${SMOKE_BATCHES}" \
  --save_path "${SMOKE_ROOT}/shared_e1"
shared="${SMOKE_ROOT}/shared_e1/adapter_1.pth"

"${PYTHON}" "${base[@]}" --epoch 6 --resume "${shared}" \
  --max_batches "${SMOKE_BATCHES}" --save_path "${SMOKE_ROOT}/A" \
  --use_safe_anchor --anchor_lambda "${ANCHOR_LAMBDA}" \
  --anchor_reference_path "${shared}" --anchor_gradient_budget \
  --anchor_family_budget .10 --anchor_family_audit --dtype_trace_path "${TRACE}"

"${PYTHON}" "${ROOT}/scripts/validate_h2_historical_mixed_fp16_fp32_v1_smoke.py" \
  --root "${SMOKE_ROOT}" --trace "${TRACE}" --output "${ROOT}/audit/H2_MIXED_FP16_FP32_V1_SMOKE.md"
