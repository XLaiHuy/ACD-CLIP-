#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${SAVE_PATH:?Set SAVE_PATH to the completed canonical P1-v8.3 training directory}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EXTERNAL_METRIC_CHUNK_PIXELS="${EXTERNAL_METRIC_CHUNK_PIXELS:-5000000}"
export ACDCLIP_DATA_ROOT="${ACDCLIP_DATA_ROOT:-${REPO_ROOT}/data}"

CHECKPOINT="${SAVE_PATH}/adapter_20.pth"
DATASETS=(Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir)
EXPECTED_DATASETS="Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir"
if [[ "${#DATASETS[@]}" -ne 6 || "${DATASETS[*]}" != "${EXPECTED_DATASETS}" ]]; then
  echo "Medical protocol must contain exactly the frozen six datasets" >&2
  exit 2
fi

PYTHONPATH=. python tools/preflight_p1_v83_final_checkpoint.py "${CHECKPOINT}"

for index in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$index]}"
  echo "[P1-v8.3 medical][$((index + 1))/6] dataset=${dataset} split=test epoch=20 stride=1 exact=true"
  python test.py \
    --save_path "${SAVE_PATH}" --epochs 20 \
    --dataset "${dataset}" --medical_split test \
    --img_size 518 --batch_size 1 --cuda_device "${CUDA_DEVICE}" --num_workers "${NUM_WORKERS}" \
    --external_exact_pixel_metrics --external_metric_chunk_pixels "${EXTERNAL_METRIC_CHUNK_PIXELS}" \
    --pixel_stride 1 --n_groups 3 \
    --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0 \
    --use_ss2d_dfg --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual \
    --dfg_beta 0.10 --dfg_beta_schedule warmup010 --dfg_beta_target 0.10 \
    --h6_progress 1 --h6_progress_version P1-v8.3 \
    --h6_num_factors 4 --h6_top_k 2 --h6_prediction_routing dense \
    --h6_global_text_mode phase2b_hybrid --use_hybrid_soft_prompt \
    --h6_local_factor_mode center_spread --h6_local_center_mix 0.05 --h6_local_factor_spread 0.10 \
    --no-h6_expert_enabled --no-h6_load_bias_enabled --no-h6_cluster_responsibility
done
