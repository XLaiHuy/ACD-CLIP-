#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/tmp/r3_full_e15_confirmation_20260909}"
E1="${ROOT}/runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth"
FORKER="${ROOT}/scripts/r3_fork_shared_e1.py"
PY="${PYTHON:-/workspace/.venv-acd-r3/bin/python}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

test -s "${E1}"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse existing E15 confirmation root: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

fork="${RUN_ROOT}/fork_e15/adapter_1.pth"
run="${RUN_ROOT}/final"
"${PY}" "${FORKER}" --source "${E1}" --output "${fork}" \
  --hybrid-alpha-max 0.20 \
  --anchor-lambda 0.0021633926715180626 --anchor-family-budget 0.10 --enable-anchor

"${PY}" "${ROOT}/train.py" \
  --protocol_horizon 20 --dataset VisA --model_name ViT-L-14-336 --img_size 518 \
  --n_groups 3 --image_adapt_weight 0.2 --text_adapt_weight 0.2 \
  --conv_lora_rank 8 --conv_lora_alpha 2.0 --conv_kernel_size_list 3 5 \
  --lora_rank 16 --lora_alpha 2.0 --image_lr 0.001 --text_lr 0.0005 \
  --use_hybrid_soft_prompt --soft_prompt_freeze_epochs 3 --soft_prompt_ctx_len 4 \
  --soft_prompt_lr 0.00005 --soft_prompt_init phrase --soft_prompt_init_phrase "a photo of a" \
  --lambda_kg 0.01 --lambda_k 0.002 --lr_gamma 0.9 \
  --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0 --use_ss2d_dfg \
  --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual --dfg_beta 0.10 \
  --dfg_beta_schedule warmup010 --dfg_beta_target 0.10 --grad_clip_norm 1.0 \
  --non_finite_loss_abort_threshold 20 --batch_size 6 --num_workers 2 \
  --grad_checkpointing --amp --seed 0 --deterministic_algorithms \
  --epoch 15 --hybrid_alpha_max 0.20 --resume "${fork}" --save_path "${run}" \
  --use_safe_anchor --anchor_lambda 0.0021633926715180626 \
  --anchor_reference_path "${E1}" --anchor_gradient_budget \
  --anchor_family_budget 0.10 --anchor_family_audit

checkpoint="${run}/adapter_15.pth"
test -s "${checkpoint}"
sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
"${PY}" "${ROOT}/scripts/r3_source_tta.py" \
  --checkpoint "${checkpoint}" --expected-sha256 "${sha}" \
  --output "${RUN_ROOT}/source_tta.json" --batch-size 8 --num-workers 2

echo "R3 full E15 source confirmation complete: ${RUN_ROOT}"
echo "CHECKPOINT=${checkpoint}"
echo "CHECKPOINT_SHA256=${sha}"
