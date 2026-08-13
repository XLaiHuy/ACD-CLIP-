#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAX_BATCHES="${MAX_BATCHES:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_PATH="${SAVE_PATH:-${REPO_ROOT}/runs/phase4/k1/short64_seed0}"
if [[ -z "${ACDCLIP_DATA_ROOT:-}" ]]; then
  if [[ -d "${REPO_ROOT}/data" ]]; then
    export ACDCLIP_DATA_ROOT="${REPO_ROOT}/data"
  else
    export ACDCLIP_DATA_ROOT="$(cd "${REPO_ROOT}/.." && pwd)/data"
  fi
fi
: "${ACDCLIP_CLIP_VITL14_336:=${REPO_ROOT}/model/ViT-L-14-336px.pt}"
export ACDCLIP_CLIP_VITL14_336

if (( MAX_BATCHES < 1 || MAX_BATCHES > 64 )); then
  echo "MAX_BATCHES must be in [1,64]" >&2
  exit 2
fi

exec conda run --no-capture-output -n torchhuy python "${REPO_ROOT}/train.py" \
  --dataset VisA --img_size 518 --epoch 1 --seed 0 --save_path "${SAVE_PATH}" \
  --batch_size 1 --grad_accum_steps 6 --precision fp32 --grad_checkpointing \
  --n_groups 3 --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0 \
  --use_ss2d_dfg --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual \
  --dfg_beta 0.10 --dfg_beta_schedule warmup010 --dfg_beta_target 0.10 \
  --h6_progress 1 --h6_progress_version P4-CSF-K1 \
  --h6_num_factors 1 --h6_top_k 1 --h6_prediction_routing dense \
  --h6_global_text_mode phase2b_hybrid --use_hybrid_soft_prompt \
  --h6_local_factor_mode legacy_mix \
  --no-h6_expert_enabled --no-h6_load_bias_enabled --no-h6_cluster_responsibility \
  --lambda_h6_balance 0 --lambda_h6_center 0 --lambda_h6_orth 0 \
  --lambda_h6_route 0 --lambda_h6_factor_role 0 --lambda_h6_actual_local 0 \
  --lambda_h6_func_div 0 --lambda_h6_delta_div 0 --lambda_h6_router_teacher 0 \
  --lambda_h6_factor 0 --lambda_h6_router 0 --lambda_h6_act 0 \
  --lambda_h6_concept_key_diversity 0 --lambda_h6_dynamic_mean_anchor 0 \
  --lambda_h6_expert 0 --lambda_h6_advantage 0 --lambda_h6_etf 0 \
  --lambda_h6_vae_rec 0.05 --beta_h6_vae_kl 0.0001 \
  --h6_wiring_probe_batches 1 32 64 --h6_smoke_max_batches "${MAX_BATCHES}" --num_workers "${NUM_WORKERS}" --pin_memory
