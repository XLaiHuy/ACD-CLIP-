#!/usr/bin/env bash
set -euo pipefail

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)"
PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/fresh_visa_source_20260910}"
ANCHOR_REFERENCE="${ROOT}/runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse existing run root: ${RUN_ROOT}" >&2
  exit 2
fi
test -s "${ROOT}/model/ViT-L-14-336px.pt"
test -s "${ROOT}/dataset/hub/VisA.jsonl"
test -s "${ANCHOR_REFERENCE}"
mkdir -p "${RUN_ROOT}"

echo "RUN_V fresh VisA source training"
echo "epochs=20 save_every_epoch=true cir=false hard_background=false"
echo "target_selection=Medical macro Pixel AP, then Pixel AUROC, then earlier epoch"

exec "${PYTHON}" "${ROOT}/train.py" \
  --protocol_horizon 20 \
  --dataset VisA \
  --model_name ViT-L-14-336 \
  --img_size 518 \
  --n_groups 3 \
  --image_adapt_weight 0.2 \
  --text_adapt_weight 0.2 \
  --conv_lora_rank 8 \
  --conv_lora_alpha 2.0 \
  --conv_kernel_size_list 3 5 \
  --lora_rank 16 \
  --lora_alpha 2.0 \
  --image_lr 0.001 \
  --text_lr 0.0005 \
  --use_hybrid_soft_prompt \
  --hybrid_alpha_max 0.2 \
  --soft_prompt_freeze_epochs 3 \
  --soft_prompt_ctx_len 4 \
  --soft_prompt_lr 0.00005 \
  --soft_prompt_init phrase \
  --soft_prompt_init_phrase "a photo of a" \
  --lambda_kg 0.01 \
  --lambda_k 0.002 \
  --lr_gamma 0.9 \
  --dfg_mode attn \
  --dfg_attn_dim 256 \
  --dfg_attn_tau 8.0 \
  --use_ss2d_dfg \
  --dfg_gamma_max 0.2 \
  --dfg_ss2d_fusion weight_residual \
  --dfg_beta 0.10 \
  --dfg_beta_schedule warmup010 \
  --dfg_beta_target 0.10 \
  --grad_clip_norm 1.0 \
  --non_finite_loss_abort_threshold 20 \
  --batch_size 6 \
  --num_workers 6 \
  --grad_checkpointing \
  --amp \
  --seed 0 \
  --deterministic_algorithms \
  --anchor_grad_audit_interval 0 \
  --use_safe_anchor \
  --anchor_lambda 0.0021633926715180626 \
  --anchor_reference_path "${ANCHOR_REFERENCE}" \
  --anchor_gradient_budget \
  --anchor_family_budget 0.10 \
  --anchor_family_audit \
  --save_path "${RUN_ROOT}" \
  --epoch 20
