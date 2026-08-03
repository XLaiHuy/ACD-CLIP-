#!/usr/bin/env bash
set -euo pipefail

: "${SAVE_PATH:?Set SAVE_PATH to the completed Phase4 training directory}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-6}"
if [ "$#" -gt 0 ]; then
  EPOCHS=("$@")
else
  EPOCHS=(8 9 10 11 12 13 14 15)
fi
DATASETS=(Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir)

for index in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$index]}"
  echo "[PHASE4-P1][TEST][$((index + 1))/6] dataset=${DATASET} epochs=${EPOCHS[*]} exact_mode=true"
  conda run --no-capture-output -n torchhuy python test.py \
    --dataset "${DATASET}" \
    --img_size 518 \
    --cuda_device "${CUDA_DEVICE}" \
    --save_path "${SAVE_PATH}" \
    --batch_size "${TEST_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --pixel_stride 1 \
    --epochs "${EPOCHS[@]}" \
    --n_groups 3 \
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
    --h6_progress 1 \
    --h6_num_factors 4 \
    --h6_top_k 2 \
    --h6_bank_dim 256 \
    --h6_router_dim 128 \
    --h6_router_temperature 1.0 \
    --h6_router_soft_epochs 2 \
    --h6_vae_hidden_dim 512 \
    --h6_vae_latent_dim 256
done
