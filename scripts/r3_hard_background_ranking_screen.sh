#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/tmp/r3_hard_background_ranking_screen_20260910}"
E1="${ROOT}/runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth"
FORKER="${ROOT}/scripts/r3_fork_shared_e1.py"
PY="${PYTHON:-/workspace/.venv-acd-r3/bin/python}"
RANKING_LAMBDAS="${RANKING_LAMBDAS:-002 005}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

test -s "${E1}"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse existing hard-background ranking screen root: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

base_args=(
  "${ROOT}/train.py"
  --protocol_horizon 20 --dataset VisA --model_name ViT-L-14-336 --img_size 518
  --n_groups 3 --image_adapt_weight 0.2 --text_adapt_weight 0.2
  --conv_lora_rank 8 --conv_lora_alpha 2.0 --conv_kernel_size_list 3 5
  --lora_rank 16 --lora_alpha 2.0 --image_lr 0.001 --text_lr 0.0005
  --use_hybrid_soft_prompt --soft_prompt_freeze_epochs 3 --soft_prompt_ctx_len 4
  --soft_prompt_lr 0.00005 --soft_prompt_init phrase --soft_prompt_init_phrase "a photo of a"
  --lambda_kg 0.01 --lambda_k 0.002 --lr_gamma 0.9
  --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0 --use_ss2d_dfg
  --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual --dfg_beta 0.10
  --dfg_beta_schedule warmup010 --dfg_beta_target 0.10 --grad_clip_norm 1.0
  --non_finite_loss_abort_threshold 20 --batch_size 6 --num_workers 2
  --grad_checkpointing --amp --seed 0 --deterministic_algorithms
  --epoch 5 --max_batches 50
)

for lambda_tag in ${RANKING_LAMBDAS}; do
  case "${lambda_tag}" in
    002) lambda_value="0.02" ;;
    005) lambda_value="0.05" ;;
    *) echo "unknown ranking lambda tag: ${lambda_tag}" >&2; exit 2 ;;
  esac
  fork="${RUN_ROOT}/fork_${lambda_tag}/adapter_1.pth"
  run="${RUN_ROOT}/lambda_${lambda_tag}"
  "${PY}" "${FORKER}" --source "${E1}" --output "${fork}" \
    --hybrid-alpha-max 0.20 \
    --anchor-lambda 0.0021633926715180626 --anchor-family-budget 0.10 --enable-anchor \
    --use-hard-background-patch-ranking \
    --hard-background-ranking-lambda "${lambda_value}" \
    --hard-background-topk-fraction 0.05 --hard-background-margin 0.05
  "${PY}" "${base_args[@]}" --hybrid_alpha_max 0.20 \
    --resume "${fork}" --save_path "${run}" \
    --use_safe_anchor --anchor_lambda 0.0021633926715180626 \
    --anchor_reference_path "${E1}" --anchor_gradient_budget \
    --anchor_family_budget 0.10 --anchor_family_audit \
    --use_hard_background_patch_ranking \
    --hard_background_ranking_lambda "${lambda_value}" \
    --hard_background_topk_fraction 0.05 --hard_background_margin 0.05
  checkpoint="${run}/adapter_5.pth"
  test -s "${checkpoint}"
  sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
  "${PY}" "${ROOT}/scripts/r3_source_tta.py" \
    --checkpoint "${checkpoint}" --expected-sha256 "${sha}" \
    --output "${RUN_ROOT}/lambda_${lambda_tag}_source.json" \
    --batch-size 8 --num-workers 2
done

echo "R3 hard-background ranking S1 source screen complete: ${RUN_ROOT}"
