#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_BATCHES="${MAX_BATCHES:-8}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/runs/p1_v83_smoke_${MAX_BATCHES}}"
export ACDCLIP_DATA_ROOT="${ACDCLIP_DATA_ROOT:-${REPO_ROOT}/data}"
: "${ACDCLIP_CLIP_VITL14_336:=${REPO_ROOT}/model/ViT-L-14-336px.pt}"
export ACDCLIP_CLIP_VITL14_336
python "${REPO_ROOT}/train.py" \
  --dataset VisA --img_size 518 --epoch 1 --save_path "${SAVE_PATH}" \
  --batch_size 1 --grad_accum_steps 6 --precision fp32 --grad_checkpointing \
  --n_groups 3 --h6_progress 1 --h6_progress_version P1-v8.3 \
  --h6_num_factors 4 --h6_top_k 2 --h6_prediction_routing dense \
  --h6_global_text_mode hard_anchor \
  --h6_local_factor_mode center_spread --h6_local_center_mix 0.05 --h6_local_factor_spread 0.10 \
  --no-h6_expert_enabled --no-h6_load_bias_enabled --no-h6_cluster_responsibility \
  --lambda_h6_balance 0 --lambda_h6_center 0 --lambda_h6_orth 0 \
  --lambda_h6_route 0 --lambda_h6_factor_role 0 --lambda_h6_actual_local 0 \
  --lambda_h6_func_div 0 --lambda_h6_router_teacher 0 \
  --lambda_h6_factor 0.10 --lambda_h6_router 0.10 \
  --h6_utility_denominator_floor 0.10 --h6_tau_utility 0.05 \
  --h6_utility_gain_threshold 0.02 --h6_utility_entropy_threshold 0.98 \
  --h6_exploration_start 0.15 --h6_exploration_end 0.05 \
  --h6_smoke_max_batches "${MAX_BATCHES}" --pin_memory
