#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/phase4/p1_v83_full20_seed0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
export ACDCLIP_DATA_ROOT="${ACDCLIP_DATA_ROOT:-${REPO_ROOT}/data}"

if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to reuse existing RUN_DIR; choose a new path to preserve prior artifacts: ${RUN_DIR}" >&2
  exit 2
fi

CMD=(
  python train.py
  --save_path "${RUN_DIR}"
  --dataset VisA --img_size 518 --epoch 20
  --batch_size 1 --grad_accum_steps 6 --precision fp32 --grad_checkpointing
  --cuda_device "${CUDA_DEVICE}" --num_workers "${NUM_WORKERS}" --seed 0
  --n_groups 3
  --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0
  --use_ss2d_dfg --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual
  --dfg_beta 0.10 --dfg_beta_schedule warmup010 --dfg_beta_target 0.10
  --h6_progress 1 --h6_progress_version P1-v8.3
  --h6_num_factors 4 --h6_top_k 2 --h6_prediction_routing dense
  --h6_global_text_mode phase2b_hybrid --use_hybrid_soft_prompt
  --h6_local_factor_mode center_spread --h6_local_center_mix 0.05 --h6_local_factor_spread 0.10
  --no-h6_expert_enabled --no-h6_load_bias_enabled --no-h6_cluster_responsibility
  --lambda_h6_balance 0 --lambda_h6_center 0 --lambda_h6_orth 0
  --lambda_h6_route 0 --lambda_h6_factor_role 0 --lambda_h6_actual_local 0
  --lambda_h6_func_div 0 --lambda_h6_router_teacher 0 --h6_lambda_cluster_resp 0
  --lambda_h6_factor 0.03 --lambda_h6_router 0.10
  --h6_utility_factor_effective_beta 0.999 --h6_router_support_normalized
  --h6_primary_anchored_factor_surgery
  --h6_utility_denominator_floor 0.10 --h6_tau_utility 0.05
  --h6_utility_gain_threshold 0.02 --h6_utility_entropy_threshold 0.98
  --h6_exploration_start 0.15 --h6_exploration_end 0.05
  --h6_drift_diagnostics --h6_factor_grad_diagnostics --pin_memory
)

for token in "${CMD[@]}"; do
  case "${token}" in
    --phase2b_checkpoint|--resume|--load_adapter|--init_from_progress)
      echo "Forbidden non-OpenAI initialization argument: ${token}" >&2
      exit 2
      ;;
  esac
done

printf 'Launching canonical P1-v8.3 final-20 command:\n'
printf '%q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
