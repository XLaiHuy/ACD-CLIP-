#!/usr/bin/env bash
# PREPARED ONLY. This file was not executed by the post-300 lambda audit.
set -euo pipefail

REPO_ROOT=/home/ai4/caohuy/ACD-CLIP-phase4
SAVE_PATH="${REPO_ROOT}/runs/p1_v83_dev/300batch_corrected_lambda003_001"
test ! -e "${SAVE_PATH}"

cd "${REPO_ROOT}"
export ACDCLIP_DATA_ROOT="${REPO_ROOT}/data"
export ACDCLIP_CLIP_VITL14_336="${REPO_ROOT}/model/ViT-L-14-336px.pt"

conda run --no-capture-output -n torchhuy python "${REPO_ROOT}/train.py" \
  --dataset VisA --img_size 518 --epoch 1 --seed 0 --save_path "${SAVE_PATH}" \
  --batch_size 1 --grad_accum_steps 6 --precision fp32 --grad_checkpointing \
  --n_groups 3 --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0 \
  --use_ss2d_dfg --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual \
  --dfg_beta 0.10 --dfg_beta_schedule warmup010 --dfg_beta_target 0.10 \
  --h6_progress 1 --h6_progress_version P1-v8.3 \
  --h6_num_factors 4 --h6_top_k 2 --h6_prediction_routing dense \
  --h6_global_text_mode phase2b_hybrid --use_hybrid_soft_prompt \
  --h6_local_factor_mode center_spread --h6_local_center_mix 0.05 --h6_local_factor_spread 0.10 \
  --no-h6_expert_enabled --no-h6_load_bias_enabled --no-h6_cluster_responsibility \
  --lambda_h6_balance 0 --lambda_h6_center 0 --lambda_h6_orth 0 \
  --lambda_h6_route 0 --lambda_h6_factor_role 0 --lambda_h6_actual_local 0 \
  --lambda_h6_func_div 0 --lambda_h6_router_teacher 0 \
  --lambda_h6_factor 0.03 --lambda_h6_router 0.01 \
  --h6_utility_denominator_floor 0.10 --h6_tau_utility 0.05 \
  --h6_utility_gain_threshold 0.02 --h6_utility_entropy_threshold 0.98 \
  --h6_exploration_start 0.15 --h6_exploration_end 0.05 \
  --h6_exploration_total_epochs 20 \
  --h6_drift_diagnostics --h6_factor_grad_diagnostics \
  --h6_smoke_max_batches 300 --h6_wiring_probe_batches 1 300 \
  --h6_trajectory_milestones 32 64 128 192 256 300 \
  --num_workers 4 --pin_memory
