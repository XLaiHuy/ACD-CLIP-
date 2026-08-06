#!/usr/bin/env bash
set -euo pipefail

export SAVE_PATH="runs/phase4/p1_v8_minimal_wiring_smoke"

echo "Running wiring smoke (50 batches)"
conda run --no-capture-output -n torchhuy python train.py \
    --dataset Brain --img_size 518 --cuda_device 0 \
    --save_path "${SAVE_PATH}/train" --batch_size 1 --num_workers 4 \
    --n_groups 3 --lora_rank 16 --lora_alpha 2.0 \
    --conv_lora_rank 8 --conv_lora_alpha 2.0 --conv_kernel_size_list 3 5 \
    --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0 \
    --use_ss2d_dfg --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual \
    --dfg_beta 0.10 --dfg_beta_schedule warmup010 --dfg_beta_target 0.10 \
    --h6_progress 1 --h6_progress_version "P1-v8-minimal" \
    --h6_global_text_mode phase2b --h6_prediction_routing dense \
    --no-h6_expert_enabled \
    --h6_diagnostics_mode light --h6_num_factors 4 --h6_top_k 2 \
    --h6_bank_dim 256 --h6_router_dim 128 --h6_router_temperature 1.0 \
    --h6_dense_routing_epochs 6 --h6_sparse_start_epoch 7 \
    --h6_vae_hidden_dim 512 --h6_vae_latent_dim 256 \
    --epoch 1 --h6_smoke_max_batches 50

echo "Checking if checkpoint was saved"
if [ ! -f "${SAVE_PATH}/train/adapter_1.pth" ]; then
    echo "Checkpoint not saved!"
    exit 1
fi

echo "Testing checkpoint load"
conda run --no-capture-output -n torchhuy python test.py \
    --dataset Brain --img_size 518 --cuda_device 0 \
    --save_path "${SAVE_PATH}/train" --batch_size 1 --num_workers 4 \
    --medical_split val --medical_manifest_root "${SAVE_PATH}/train/protocol/medical_manifests" \
    --external_exact_pixel_metrics --external_metric_chunk_pixels 5000000 \
    --pixel_stride 1 --epochs 1 \
    --n_groups 3 --lora_rank 16 --lora_alpha 2.0 \
    --conv_lora_rank 8 --conv_lora_alpha 2.0 --conv_kernel_size_list 3 5 \
    --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0 \
    --use_ss2d_dfg --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual \
    --dfg_beta 0.10 --dfg_beta_schedule warmup010 --dfg_beta_target 0.10 \
    --h6_progress 1 --h6_progress_version "P1-v8-minimal" \
    --h6_global_text_mode phase2b --h6_prediction_routing dense \
    --no-h6_expert_enabled \
    --h6_diagnostics_mode light --h6_num_factors 4 --h6_top_k 2 \
    --h6_bank_dim 256 --h6_router_dim 128 --h6_router_temperature 1.0 \
    --h6_dense_routing_epochs 6 --h6_sparse_start_epoch 7 \
    --h6_vae_hidden_dim 512 --h6_vae_latent_dim 256

echo "Wiring smoke successful."
