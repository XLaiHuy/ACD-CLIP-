#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-runs/phase4/progress1_v4_minimal_symmetry_break_seed0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-6}"
NUM_WORKERS="${NUM_WORKERS:-6}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-20}"

H6_DENSE_ROUTING_EPOCHS="${H6_DENSE_ROUTING_EPOCHS:-8}"
H6_SPARSE_TRANSITION_EPOCHS="${H6_SPARSE_TRANSITION_EPOCHS:-4}"
LAMBDA_H6_ROUTER_TEACHER="${LAMBDA_H6_ROUTER_TEACHER:-0.01}"
H6_ROUTER_TEACHER_TEMPERATURE="${H6_ROUTER_TEACHER_TEMPERATURE:-0.15}"
H6_ROUTER_TEACHER_START_EPOCH="${H6_ROUTER_TEACHER_START_EPOCH:-3}"
H6_ROUTER_TEACHER_WARMUP_EPOCHS="${H6_ROUTER_TEACHER_WARMUP_EPOCHS:-3}"
LAMBDA_H6_BALANCE="${LAMBDA_H6_BALANCE:-0.001}"
H6_LOAD_BIAS_ENABLED="${H6_LOAD_BIAS_ENABLED:-1}"
H6_LOAD_BIAS_MOMENTUM="${H6_LOAD_BIAS_MOMENTUM:-0.9}"
H6_LOAD_BIAS_STEP="${H6_LOAD_BIAS_STEP:-0.001}"
H6_LOAD_BIAS_MAX="${H6_LOAD_BIAS_MAX:-0.03}"
H6_ROUTER_FAILURE_PATIENCE="${H6_ROUTER_FAILURE_PATIENCE:-2}"
H6_ROUTER_MAX_SPARSE_DEAD_FACTORS="${H6_ROUTER_MAX_SPARSE_DEAD_FACTORS:-1}"
H6_ROUTER_MIN_UNIQUE_TOPK_PAIRS="${H6_ROUTER_MIN_UNIQUE_TOPK_PAIRS:-2}"
H6_KL_ZERO_EPOCHS="${H6_KL_ZERO_EPOCHS:-8}"
H6_KL_WARMUP_EPOCHS="${H6_KL_WARMUP_EPOCHS:-4}"
BETA_H6_VAE_KL="${BETA_H6_VAE_KL:-0.00001}"
H6_KL_FREE_BITS="${H6_KL_FREE_BITS:-0.02}"
H6_VAE_CLASS_RATIO="${H6_VAE_CLASS_RATIO:-0.25}"
H6_SLOT_INIT_ENABLED="${H6_SLOT_INIT_ENABLED:-1}"
H6_SLOT_INIT_SCALE="${H6_SLOT_INIT_SCALE:-0.02}"
H6_SLOT_INIT_SEED_OFFSET="${H6_SLOT_INIT_SEED_OFFSET:-6100}"
LAMBDA_H6_CONCEPT_KEY_DIVERSITY="${LAMBDA_H6_CONCEPT_KEY_DIVERSITY:-0.0001}"
H6_CONCEPT_KEY_COSINE_MARGIN="${H6_CONCEPT_KEY_COSINE_MARGIN:-0.5}"
H6_CONCEPT_KEY_DIVERSITY_START_EPOCH="${H6_CONCEPT_KEY_DIVERSITY_START_EPOCH:-1}"
H6_CONCEPT_KEY_DIVERSITY_WARMUP_EPOCHS="${H6_CONCEPT_KEY_DIVERSITY_WARMUP_EPOCHS:-3}"
H6_FACTOR_GRAD_DIAGNOSTICS="${H6_FACTOR_GRAD_DIAGNOSTICS:-1}"

BRANCH="$(git branch --show-current)"
COMMIT_SHA="$(git rev-parse HEAD)"
LOAD_BIAS_FLAG="--no-h6_load_bias_enabled"
if [ "${H6_LOAD_BIAS_ENABLED}" = "1" ]; then
  LOAD_BIAS_FLAG="--h6_load_bias_enabled"
fi
SLOT_INIT_FLAG="--no-h6_slot_init_enabled"
if [ "${H6_SLOT_INIT_ENABLED}" = "1" ]; then
  SLOT_INIT_FLAG="--h6_slot_init_enabled"
fi
FACTOR_GRAD_FLAG="--no-h6_factor_grad_diagnostics"
if [ "${H6_FACTOR_GRAD_DIAGNOSTICS}" = "1" ]; then
  FACTOR_GRAD_FLAG="--h6_factor_grad_diagnostics"
fi

echo "[PHASE4-P1-V4] branch=${BRANCH} commit=${COMMIT_SHA}"
echo "[PHASE4-P1-V4] save_path=${SAVE_PATH} cuda=${CUDA_DEVICE} seed=${SEED} precision=${PRECISION}"
echo "[PHASE4-P1-V4] batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} effective_batch=$((BATCH_SIZE * GRAD_ACCUM)) workers=${NUM_WORKERS} pin_memory=false"
echo "[PHASE4-P1-V4] routing=dense_${H6_DENSE_ROUTING_EPOCHS}_transition_${H6_SPARSE_TRANSITION_EPOCHS}_st_topk2"
echo "[PHASE4-P1-V4] slot_init=${H6_SLOT_INIT_ENABLED} scale=${H6_SLOT_INIT_SCALE} seed_offset=${H6_SLOT_INIT_SEED_OFFSET}"
echo "[PHASE4-P1-V4] key_diversity=${LAMBDA_H6_CONCEPT_KEY_DIVERSITY} margin=${H6_CONCEPT_KEY_COSINE_MARGIN}"
echo "[PHASE4-P1-V4] train_from=OpenAI_CLIP_only no_phase2b_checkpoint=true"

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
  --h6_dense_routing_epochs "${H6_DENSE_ROUTING_EPOCHS}" \
  --h6_sparse_transition_epochs "${H6_SPARSE_TRANSITION_EPOCHS}" \
  --lambda_h6_router_teacher "${LAMBDA_H6_ROUTER_TEACHER}" \
  --h6_router_teacher_temperature "${H6_ROUTER_TEACHER_TEMPERATURE}" \
  --h6_router_teacher_start_epoch "${H6_ROUTER_TEACHER_START_EPOCH}" \
  --h6_router_teacher_warmup_epochs "${H6_ROUTER_TEACHER_WARMUP_EPOCHS}" \
  --lambda_h6_center 0.10 \
  --h6_center_factor_aware \
  --h6_center_detach_assignment \
  --h6_center_margin 0.0 \
  --lambda_h6_vae_rec 0.05 \
  --beta_h6_vae_kl "${BETA_H6_VAE_KL}" \
  --h6_kl_zero_epochs "${H6_KL_ZERO_EPOCHS}" \
  --h6_kl_warmup_epochs "${H6_KL_WARMUP_EPOCHS}" \
  --h6_kl_free_bits "${H6_KL_FREE_BITS}" \
  --h6_vae_class_ratio "${H6_VAE_CLASS_RATIO}" \
  "${SLOT_INIT_FLAG}" \
  --h6_slot_init_scale "${H6_SLOT_INIT_SCALE}" \
  --h6_slot_init_seed_offset "${H6_SLOT_INIT_SEED_OFFSET}" \
  "${FACTOR_GRAD_FLAG}" \
  --lambda_h6_orth 0.001 \
  --lambda_h6_balance "${LAMBDA_H6_BALANCE}" \
  --lambda_h6_concept_key_diversity "${LAMBDA_H6_CONCEPT_KEY_DIVERSITY}" \
  --h6_concept_key_cosine_margin "${H6_CONCEPT_KEY_COSINE_MARGIN}" \
  --h6_concept_key_diversity_start_epoch "${H6_CONCEPT_KEY_DIVERSITY_START_EPOCH}" \
  --h6_concept_key_diversity_warmup_epochs "${H6_CONCEPT_KEY_DIVERSITY_WARMUP_EPOCHS}" \
  "${LOAD_BIAS_FLAG}" \
  --h6_load_bias_momentum "${H6_LOAD_BIAS_MOMENTUM}" \
  --h6_load_bias_step "${H6_LOAD_BIAS_STEP}" \
  --h6_load_bias_max "${H6_LOAD_BIAS_MAX}" \
  --h6_router_failure_patience "${H6_ROUTER_FAILURE_PATIENCE}" \
  --h6_router_max_sparse_dead_factors "${H6_ROUTER_MAX_SPARSE_DEAD_FACTORS}" \
  --h6_router_min_unique_topk_pairs "${H6_ROUTER_MIN_UNIQUE_TOPK_PAIRS}" \
  --h6_vae_hidden_dim 512 \
  --h6_vae_latent_dim 256 \
  --save_path "${SAVE_PATH}"
