#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-runs/phase4/progress1_v6_structural_diagnostic12_seed0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-6}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-12}"

H6_ROUTER_QUERY_MODE="${H6_ROUTER_QUERY_MODE:-local_global_bypass}"
H6_ROUTER_QUERY_GLOBAL_WEIGHT="${H6_ROUTER_QUERY_GLOBAL_WEIGHT:-0.10}"
H6_ROUTER_LOCAL_BYPASS_SCALE="${H6_ROUTER_LOCAL_BYPASS_SCALE:-0.10}"
H6_ROUTER_LOCAL_BYPASS_MAX_RATIO="${H6_ROUTER_LOCAL_BYPASS_MAX_RATIO:-0.20}"

H6_ROUTER_KEY_ANCHOR_ENABLED="${H6_ROUTER_KEY_ANCHOR_ENABLED:-1}"
H6_ROUTER_KEY_ADAPTATION_INITIAL_RATIO="${H6_ROUTER_KEY_ADAPTATION_INITIAL_RATIO:-0.10}"
H6_ROUTER_KEY_ADAPTATION_MAX_RATIO="${H6_ROUTER_KEY_ADAPTATION_MAX_RATIO:-0.25}"

H6_FACTOR_CONTEXT_ANCHOR_ENABLED="${H6_FACTOR_CONTEXT_ANCHOR_ENABLED:-1}"
H6_FACTOR_CONTEXT_ADAPTATION_INITIAL_RATIO="${H6_FACTOR_CONTEXT_ADAPTATION_INITIAL_RATIO:-0.10}"
H6_FACTOR_CONTEXT_ADAPTATION_MAX_RATIO="${H6_FACTOR_CONTEXT_ADAPTATION_MAX_RATIO:-0.25}"
H6_FACTOR_IDENTITY_TANGENT_PROJECTION_ENABLED="${H6_FACTOR_IDENTITY_TANGENT_PROJECTION_ENABLED:-1}"

H6_FACTOR_ID_SCALE="${H6_FACTOR_ID_SCALE:-0.02}"
H6_FACTOR_ID_MAX_RATIO="${H6_FACTOR_ID_MAX_RATIO:-0.05}"

LAMBDA_H6_DYNAMIC_MEAN_ANCHOR="${LAMBDA_H6_DYNAMIC_MEAN_ANCHOR:-0.001}"
H6_DYNAMIC_MEAN_ANCHOR_MIN_COSINE="${H6_DYNAMIC_MEAN_ANCHOR_MIN_COSINE:-0.70}"
H6_DYNAMIC_MEAN_ANCHOR_START_EPOCH="${H6_DYNAMIC_MEAN_ANCHOR_START_EPOCH:-4}"
H6_DYNAMIC_MEAN_ANCHOR_WARMUP_EPOCHS="${H6_DYNAMIC_MEAN_ANCHOR_WARMUP_EPOCHS:-3}"

ROUTER_KEY_ANCHOR_FLAG="--no-h6_router_key_anchor_enabled"
if [ "${H6_ROUTER_KEY_ANCHOR_ENABLED}" = "1" ]; then
  ROUTER_KEY_ANCHOR_FLAG="--h6_router_key_anchor_enabled"
fi
FACTOR_CONTEXT_ANCHOR_FLAG="--no-h6_factor_context_anchor_enabled"
if [ "${H6_FACTOR_CONTEXT_ANCHOR_ENABLED}" = "1" ]; then
  FACTOR_CONTEXT_ANCHOR_FLAG="--h6_factor_context_anchor_enabled"
fi
TANGENT_FLAG="--no-h6_factor_identity_tangent_projection_enabled"
if [ "${H6_FACTOR_IDENTITY_TANGENT_PROJECTION_ENABLED}" = "1" ]; then
  TANGENT_FLAG="--h6_factor_identity_tangent_projection_enabled"
fi

echo "[PHASE4-P1-V6-STRUCTURAL-DIAG] train_only=true validation=false medical=false test=false save_path=${SAVE_PATH}"
echo "[PHASE4-P1-V6-STRUCTURAL-DIAG] query=${H6_ROUTER_QUERY_MODE} global_weight=${H6_ROUTER_QUERY_GLOBAL_WEIGHT} bypass=${H6_ROUTER_LOCAL_BYPASS_SCALE}/${H6_ROUTER_LOCAL_BYPASS_MAX_RATIO}"
echo "[PHASE4-P1-V6-STRUCTURAL-DIAG] router_key_anchor=${H6_ROUTER_KEY_ANCHOR_ENABLED} factor_context_anchor=${H6_FACTOR_CONTEXT_ANCHOR_ENABLED} tangent=${H6_FACTOR_IDENTITY_TANGENT_PROJECTION_ENABLED}"

conda run --no-capture-output -n torchhuy python train.py \
  --dataset VisA \
  --img_size 518 \
  --epoch "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --cuda_device "${CUDA_DEVICE}" \
  --grad_accum_steps "${GRAD_ACCUM}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --precision "${PRECISION}" \
  --n_groups 3 \
  --image_adapt_weight 0.2 \
  --text_adapt_weight 0.2 \
  --lora_rank 16 \
  --lora_alpha 2.0 \
  --conv_lora_rank 8 \
  --conv_lora_alpha 2.0 \
  --conv_kernel_size_list 3 5 \
  --dfg_mode attn \
  --dfg_attn_dim 256 \
  --dfg_attn_tau 8.0 \
  --use_ss2d_dfg \
  --dfg_gamma_max 0.2 \
  --dfg_ss2d_fusion weight_residual \
  --dfg_beta 0.10 \
  --dfg_beta_schedule warmup010 \
  --dfg_beta_target 0.10 \
  --image_lr 0.001 \
  --text_lr 0.0005 \
  --soft_prompt_ctx_len 4 \
  --soft_prompt_lr 0.00005 \
  --hybrid_alpha_max 0.20 \
  --soft_prompt_freeze_epochs 3 \
  --lambda_kg 0.001 \
  --lambda_k 0.0 \
  --grad_clip_norm 1.0 \
  --grad_checkpointing \
  --h6_progress 1 \
  --h6_num_factors 4 \
  --h6_top_k 2 \
  --h6_bank_dim 256 \
  --h6_router_dim 128 \
  --h6_router_temperature 1.0 \
  --h6_dense_routing_epochs 8 \
  --h6_sparse_transition_epochs 4 \
  --lambda_h6_router_teacher 0.01 \
  --h6_router_teacher_temperature 0.15 \
  --h6_router_teacher_start_epoch 3 \
  --h6_router_teacher_warmup_epochs 3 \
  --h6_router_teacher_mode state_centered_cosine \
  --h6_teacher_confidence_gate \
  --h6_teacher_entropy_threshold 0.98 \
  --h6_teacher_prob_std_threshold 0.001 \
  --h6_router_query_mode "${H6_ROUTER_QUERY_MODE}" \
  --h6_router_query_global_weight "${H6_ROUTER_QUERY_GLOBAL_WEIGHT}" \
  --h6_router_local_bypass_scale "${H6_ROUTER_LOCAL_BYPASS_SCALE}" \
  --h6_router_local_bypass_max_ratio "${H6_ROUTER_LOCAL_BYPASS_MAX_RATIO}" \
  --h6_router_local_projection_seed_offset 7200 \
  "${ROUTER_KEY_ANCHOR_FLAG}" \
  --h6_router_key_anchor_seed_offset 7300 \
  --h6_router_key_adaptation_initial_ratio "${H6_ROUTER_KEY_ADAPTATION_INITIAL_RATIO}" \
  --h6_router_key_adaptation_max_ratio "${H6_ROUTER_KEY_ADAPTATION_MAX_RATIO}" \
  "${FACTOR_CONTEXT_ANCHOR_FLAG}" \
  --h6_factor_context_anchor_seed_offset 7400 \
  --h6_factor_context_adaptation_initial_ratio "${H6_FACTOR_CONTEXT_ADAPTATION_INITIAL_RATIO}" \
  --h6_factor_context_adaptation_max_ratio "${H6_FACTOR_CONTEXT_ADAPTATION_MAX_RATIO}" \
  "${TANGENT_FLAG}" \
  --lambda_h6_dynamic_mean_anchor "${LAMBDA_H6_DYNAMIC_MEAN_ANCHOR}" \
  --h6_dynamic_mean_anchor_min_cosine "${H6_DYNAMIC_MEAN_ANCHOR_MIN_COSINE}" \
  --h6_dynamic_mean_anchor_start_epoch "${H6_DYNAMIC_MEAN_ANCHOR_START_EPOCH}" \
  --h6_dynamic_mean_anchor_warmup_epochs "${H6_DYNAMIC_MEAN_ANCHOR_WARMUP_EPOCHS}" \
  --lambda_h6_center 0.10 \
  --h6_center_factor_aware \
  --h6_center_detach_assignment \
  --h6_center_margin 0.0 \
  --lambda_h6_vae_rec 0.05 \
  --beta_h6_vae_kl 0.00001 \
  --h6_kl_zero_epochs 8 \
  --h6_kl_warmup_epochs 4 \
  --h6_kl_free_bits 0.02 \
  --h6_vae_class_ratio 0.25 \
  --h6_slot_init_enabled \
  --h6_slot_init_scale 0.02 \
  --h6_slot_init_seed_offset 6100 \
  --h6_factor_grad_diagnostics \
  --h6_late_factor_identity_enabled \
  --h6_factor_id_scale "${H6_FACTOR_ID_SCALE}" \
  --h6_factor_id_max_ratio "${H6_FACTOR_ID_MAX_RATIO}" \
  --lambda_h6_orth 0.001 \
  --lambda_h6_balance 0.001 \
  --lambda_h6_concept_key_diversity 0.0001 \
  --h6_concept_key_cosine_margin 0.5 \
  --h6_concept_key_diversity_start_epoch 1 \
  --h6_concept_key_diversity_warmup_epochs 3 \
  --h6_load_bias_enabled \
  --h6_load_bias_momentum 0.9 \
  --h6_load_bias_step 0.001 \
  --h6_load_bias_max 0.03 \
  --h6_router_failure_patience 2 \
  --h6_router_max_sparse_dead_factors 1 \
  --h6_router_min_unique_topk_pairs 2 \
  --h6_vae_hidden_dim 512 \
  --h6_vae_latent_dim 256 \
  --save_path "${SAVE_PATH}"
