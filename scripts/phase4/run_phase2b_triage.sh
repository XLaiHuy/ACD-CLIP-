#!/usr/bin/env bash
set -euo pipefail

export SAVE_PATH="runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch"
EPOCH="10"
MEDICAL_MANIFEST_ROOT="${SAVE_PATH}/protocol/medical_manifests"
DATASETS=(Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir)

python tools/prepare_phase4_medical_splits.py \
  --output-root "${MEDICAL_MANIFEST_ROOT}" --val-ratio 0.30 --seed 0

echo "Running Phase2B baseline"
for DATASET in "${DATASETS[@]}"; do
    echo "Dataset: ${DATASET}"
    conda run --no-capture-output -n torchhuy python test.py \
        --dataset "${DATASET}" --img_size 518 --cuda_device 0 \
        --save_path "${SAVE_PATH}" --batch_size 1 --num_workers 4 \
        --medical_split val --medical_manifest_root "${MEDICAL_MANIFEST_ROOT}" \
        --external_exact_pixel_metrics --external_metric_chunk_pixels 5000000 \
        --pixel_stride 1 --epochs "${EPOCH}" \
        --n_groups 3 --lora_rank 16 --lora_alpha 2.0 \
        --conv_lora_rank 8 --conv_lora_alpha 2.0 --conv_kernel_size_list 3 5 \
        --use_hybrid_soft_prompt \
        --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0 \
        --use_ss2d_dfg --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual \
        --dfg_beta 0.10 --dfg_beta_schedule warmup010 --dfg_beta_target 0.10 \
        --h6_progress 0
done
mkdir -p "${SAVE_PATH}/phase2b_triage"
cp "${SAVE_PATH}/key_ap_summary.csv" "${SAVE_PATH}/phase2b_triage/" || true
cp "${SAVE_PATH}/parsed_results.csv" "${SAVE_PATH}/phase2b_triage/" || true
