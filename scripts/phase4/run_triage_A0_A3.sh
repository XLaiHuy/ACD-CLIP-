#!/usr/bin/env bash
set -euo pipefail

export SAVE_PATH="runs/phase4/progress1_v7_full_seed0_ready3/train"
EPOCH="12"
MEDICAL_MANIFEST_ROOT="${SAVE_PATH}/protocol/medical_manifests"
DATASETS=(Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir)

run_ablation() {
    local name="$1"
    shift
    echo "=========================================="
    echo "Running ${name}"
    for DATASET in "${DATASETS[@]}"; do
        echo "Dataset: ${DATASET}"
        conda run --no-capture-output -n torchhuy python test.py \
            --dataset "${DATASET}" --img_size 518 --cuda_device 0 \
            --save_path "${SAVE_PATH}" --batch_size 1 --num_workers 4 \
            --medical_split val --medical_manifest_root "${MEDICAL_MANIFEST_ROOT}" \
            --pixel_stride 1 --epochs "${EPOCH}" \
            --n_groups 3 --lora_rank 16 --lora_alpha 2.0 \
            --conv_lora_rank 8 --conv_lora_alpha 2.0 --conv_kernel_size_list 3 5 \
            --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0 \
            --use_ss2d_dfg --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual \
            --dfg_beta 0.10 --dfg_beta_schedule warmup010 --dfg_beta_target 0.10 \
            --h6_progress 1 --h6_progress_version "P1-v7-full" \
            --h6_diagnostics_mode light --h6_num_factors 4 --h6_top_k 2 \
            --h6_bank_dim 256 --h6_router_dim 128 --h6_router_temperature 1.0 \
            --h6_dense_routing_epochs 6 --h6_sparse_start_epoch 7 \
            --h6_vae_hidden_dim 512 --h6_vae_latent_dim 256 \
            "$@"
    done
    # Extract save_path if it was overridden, otherwise use default
    local actual_save_path="${SAVE_PATH}"
    for i in "$@"; do
        if [[ "$i" == "--save_path" ]]; then
            :
        fi
    done
    local next_is_save_path=0
    for arg in "$@"; do
        if [[ "$next_is_save_path" == "1" ]]; then
            actual_save_path="$arg"
            break
        fi
        if [[ "$arg" == "--save_path" ]]; then
            next_is_save_path=1
        fi
    done

    # Run support-aware aggregation for the triage epoch
    local safe_name=$(echo "${name}" | tr ' ' '_')
    local out_dir="${SAVE_PATH}/${safe_name}"
    mkdir -p "${out_dir}"
    
    python tools/reaggregate_support_aware.py \
        --save_path "${actual_save_path}" \
        --split val \
        --epochs "${EPOCH}" \
        --output_dir "${out_dir}" \
        --manifest_root "${MEDICAL_MANIFEST_ROOT}" || true

    # Backup exact results to prevent overwrite
    mv "${actual_save_path}"/exact_results_*_val_epoch_"${EPOCH}".csv "${out_dir}/" || true
    
    # Clean up any test results if they were generated
    rm -f "${actual_save_path}"/exact_results_*_test_epoch_*.csv || true
}

run_ablation "A0 - legacy P1-v7" \
    --h6_global_text_mode dynamic_legacy \
    --h6_prediction_routing scheduled_topk \
    --h6_expert_enabled

run_ablation "A1 - hard-anchor baseline inside P1 checkpoint" \
    --h6_global_text_mode hard_anchor \
    --h6_test_rho_override 0.0 \
    --no-h6_expert_enabled

run_ablation "A2 - target P1-v8" \
    --h6_global_text_mode hard_anchor \
    --h6_prediction_routing dense \
    --no-h6_expert_enabled

run_ablation "A3 - sparse comparison" \
    --h6_global_text_mode hard_anchor \
    --h6_prediction_routing scheduled_topk \
    --no-h6_expert_enabled
