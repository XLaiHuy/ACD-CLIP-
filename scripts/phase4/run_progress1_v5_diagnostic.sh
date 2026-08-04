#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-runs/phase4/progress1_v5_diagnostic_seed0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-6}"
NUM_WORKERS="${NUM_WORKERS:-6}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-3}"

echo "[PHASE4-P1-V5-DIAG] train_only=true medical_val=false medical_test=false save_path=${SAVE_PATH}"
echo "[PHASE4-P1-V5-DIAG] teacher_confidence_gate=true entropy_threshold=0.98 prob_std_threshold=0.001"

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
  --h6_teacher_confidence_gate \
  --h6_teacher_entropy_threshold 0.98 \
  --h6_teacher_prob_std_threshold 0.001 \
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
