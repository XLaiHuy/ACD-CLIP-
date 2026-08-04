#!/usr/bin/env bash
set -euo pipefail

: "${SAVE_PATH:?Set SAVE_PATH to the completed Phase4 P1-v3 training directory}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-6}"
MEDICAL_SPLIT="test"
MEDICAL_MANIFEST_ROOT="${MEDICAL_MANIFEST_ROOT:-${SAVE_PATH}/protocol/medical_manifests}"
EXTERNAL_EXACT_PIXEL_METRICS="${EXTERNAL_EXACT_PIXEL_METRICS:-1}"
EXTERNAL_METRIC_CHUNK_PIXELS="${EXTERNAL_METRIC_CHUNK_PIXELS:-5000000}"
H6_DENSE_ROUTING_EPOCHS="${H6_DENSE_ROUTING_EPOCHS:-8}"
H6_SPARSE_TRANSITION_EPOCHS="${H6_SPARSE_TRANSITION_EPOCHS:-4}"
H6_LOAD_BIAS_ENABLED="${H6_LOAD_BIAS_ENABLED:-1}"
H6_LOAD_BIAS_MOMENTUM="${H6_LOAD_BIAS_MOMENTUM:-0.9}"
H6_LOAD_BIAS_STEP="${H6_LOAD_BIAS_STEP:-0.001}"
H6_LOAD_BIAS_MAX="${H6_LOAD_BIAS_MAX:-0.03}"
H6_VAE_CLASS_RATIO="${H6_VAE_CLASS_RATIO:-0.25}"
EPOCHS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --split)
      MEDICAL_SPLIT="$2"
      shift 2
      ;;
    --manifest-root)
      MEDICAL_MANIFEST_ROOT="$2"
      shift 2
      ;;
    *)
      EPOCHS+=("$1")
      shift
      ;;
  esac
done

if [ "${MEDICAL_SPLIT}" != "val" ] && [ "${MEDICAL_SPLIT}" != "test" ]; then
  echo "--split must be val or test, got ${MEDICAL_SPLIT}" >&2
  exit 2
fi
if [ "${#EPOCHS[@]}" -eq 0 ]; then
  EPOCHS=(8 9 10 11 12 13 14 15)
fi

EXTRA_TEST_ARGS=()
if [ "${EXTERNAL_EXACT_PIXEL_METRICS}" = "1" ]; then
  EXTRA_TEST_ARGS+=(--external_exact_pixel_metrics --external_metric_chunk_pixels "${EXTERNAL_METRIC_CHUNK_PIXELS}")
fi
LOAD_BIAS_FLAG="--no-h6_load_bias_enabled"
if [ "${H6_LOAD_BIAS_ENABLED}" = "1" ]; then
  LOAD_BIAS_FLAG="--h6_load_bias_enabled"
fi

python tools/prepare_phase4_medical_splits.py \
  --output-root "${MEDICAL_MANIFEST_ROOT}" --val-ratio 0.30 --seed 0

DATASETS=(Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir)
for index in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$index]}"
  echo "[PHASE4-P1-V3][${MEDICAL_SPLIT^^}][$((index + 1))/6] dataset=${DATASET} epochs=${EPOCHS[*]} exact_mode=true external_exact=${EXTERNAL_EXACT_PIXEL_METRICS} chunk_pixels=${EXTERNAL_METRIC_CHUNK_PIXELS}"
  conda run --no-capture-output -n torchhuy python test.py \
    --dataset "${DATASET}" \
    --img_size 518 \
    --cuda_device "${CUDA_DEVICE}" \
    --save_path "${SAVE_PATH}" \
    --batch_size "${TEST_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --medical_split "${MEDICAL_SPLIT}" \
    --medical_manifest_root "${MEDICAL_MANIFEST_ROOT}" \
    "${EXTRA_TEST_ARGS[@]}" \
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
    --h6_dense_routing_epochs "${H6_DENSE_ROUTING_EPOCHS}" \
    --h6_sparse_transition_epochs "${H6_SPARSE_TRANSITION_EPOCHS}" \
    "${LOAD_BIAS_FLAG}" \
    --h6_load_bias_momentum "${H6_LOAD_BIAS_MOMENTUM}" \
    --h6_load_bias_step "${H6_LOAD_BIAS_STEP}" \
    --h6_load_bias_max "${H6_LOAD_BIAS_MAX}" \
    --h6_vae_class_ratio "${H6_VAE_CLASS_RATIO}" \
    --h6_vae_hidden_dim 512 \
    --h6_vae_latent_dim 256
done
