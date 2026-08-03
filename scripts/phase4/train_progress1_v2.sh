#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-runs/phase4/progress1_v2_specialization_seed0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-6}"
NUM_WORKERS="${NUM_WORKERS:-6}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-20}"
H6_DENSE_ROUTING_EPOCHS="${H6_DENSE_ROUTING_EPOCHS:-6}"

BRANCH="$(git branch --show-current)"
COMMIT_SHA="$(git rev-parse HEAD)"

echo "[PHASE4-P1-V2] branch=${BRANCH}"
echo "[PHASE4-P1-V2] commit=${COMMIT_SHA}"
echo "[PHASE4-P1-V2] progress=P1-v2 save_path=${SAVE_PATH} cuda_device=${CUDA_DEVICE} seed=${SEED}"
echo "[PHASE4-P1-V2] precision=${PRECISION} epochs=${EPOCHS} batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} effective_batch=$((BATCH_SIZE * GRAD_ACCUM))"
echo "[PHASE4-P1-V2] num_workers=${NUM_WORKERS} pin_memory=false"
echo "[PHASE4-P1-V2] routing=dense_epochs_${H6_DENSE_ROUTING_EPOCHS}_then_topk2 M=4 K=2"
echo "[PHASE4-P1-V2] text_norm=pre_fusion_l2 frozen_anchor=functional_layer_norm_no_adapter diversity=dynamic_residual"
echo "[PHASE4-P1-V2] vae_prompt=decoder_mu sampled_z=reconstruction_only eta_class=none"
echo "[PHASE4-P1-V2] train_from=OpenAI_CLIP_only no_phase2b_checkpoint=true"

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
  --h6_router_soft_epochs "${H6_DENSE_ROUTING_EPOCHS}" \
  --h6_vae_hidden_dim 512 \
  --h6_vae_latent_dim 256 \
  --lambda_h6_center 0.10 \
  --lambda_h6_vae_rec 0.05 \
  --beta_h6_vae_kl 0.0001 \
  --lambda_h6_orth 0.001 \
  --lambda_h6_balance 0.01 \
  --save_path "${SAVE_PATH}"
