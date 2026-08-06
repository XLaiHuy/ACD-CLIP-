#!/usr/bin/env bash
set -euo pipefail

mkdir -p runs/phase4/p1_fast_audit
python tools/instrument_train.py

export SAVE_PATH="runs/phase4/p1_fast_audit"

run_profile() {
    local name="$1"
    local json_out="$2"
    shift 2
    echo "=========================================="
    echo "Profiling ${name}"
    conda run --no-capture-output -n torchhuy python train_profile.py \
        --dataset Brain --img_size 518 --cuda_device 0 \
        --save_path "${SAVE_PATH}" --batch_size 1 --num_workers 4 \
        --n_groups 3 --lora_rank 16 --lora_alpha 2.0 \
        --conv_lora_rank 8 --conv_lora_alpha 2.0 --conv_kernel_size_list 3 5 \
        --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0 \
        --use_ss2d_dfg --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual \
        --dfg_beta 0.10 --dfg_beta_schedule warmup010 --dfg_beta_target 0.10 \
        --h6_progress 1 --h6_progress_version "P1-v7-full" \
        --h6_num_factors 4 --h6_top_k 2 \
        --h6_bank_dim 256 --h6_router_dim 128 --h6_router_temperature 1.0 \
        --h6_dense_routing_epochs 6 --h6_sparse_start_epoch 7 \
        --h6_vae_hidden_dim 512 --h6_vae_latent_dim 256 \
        --h6_global_text_mode dynamic --h6_prediction_routing scheduled_topk \
        --epoch 1 \
        "$@" || true
    
    if [ -f "${SAVE_PATH}/runtime_profile_before.json" ]; then
        mv "${SAVE_PATH}/runtime_profile_before.json" "${SAVE_PATH}/${json_out}"
    fi
}

run_profile "R0 full P1-v7" "R0.json" \
    --h6_expert_enabled \
    --h6_diagnostics_mode full

run_profile "R1 experts off" "R1.json" \
    --no-h6_expert_enabled \
    --h6_diagnostics_mode full

run_profile "R2 diagnostics light" "R2.json" \
    --h6_expert_enabled \
    --h6_diagnostics_mode light

run_profile "R3 experts off + diagnostics light" "R3.json" \
    --no-h6_expert_enabled \
    --h6_diagnostics_mode light
