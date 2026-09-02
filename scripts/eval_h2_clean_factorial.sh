#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/h2_clean_factorial_v1}"
ARM="${ARM:-H}"
CONDA_ENV="${CONDA_ENV:-torchhuy}"
RUN_EVAL="${RUN_EVAL:-NO}"
FINAL_FROZEN="${FINAL_FROZEN:-NO}"
EVAL_EPOCH="${EVAL_EPOCH:-15}"
EVALUATOR_MODE="${EVALUATOR_MODE:-benchmark_exact}"
NUM_WORKERS="${NUM_WORKERS:-6}"

if [[ "${EVAL_EPOCH}" != "15" && "${EVAL_EPOCH}" != "20" ]]; then
  echo "EVAL_EPOCH must be 15 or 20" >&2
  exit 2
fi
PY=(conda run --no-capture-output -n "${CONDA_ENV}" python)

if [[ "${RUN_EVAL}" != "YES" || "${FINAL_FROZEN}" != "YES" ]]; then
  echo "Prepared only. Set RUN_EVAL=YES FINAL_FROZEN=YES after the fixed E15/E20 checkpoint is frozen."
  echo "Set EVAL_EPOCH=15 for primary or EVAL_EPOCH=20 for secondary; this command never selects a checkpoint."
  exit 0
fi

SAVE_PATH="${RUN_ROOT}/${ARM}"
test -s "${SAVE_PATH}/adapter_${EVAL_EPOCH}.pth"

for dataset in Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir; do
  "${PY[@]}" "${ROOT}/test.py" \
    --dataset "${dataset}" \
    --model_name ViT-L-14-336 \
    --img_size 518 \
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
    --batch_size 8 \
    --num_workers "${NUM_WORKERS}" \
    --save_path "${SAVE_PATH}" \
    --epochs "${EVAL_EPOCH}" \
    --evaluator_mode "${EVALUATOR_MODE}" \
    --pixel_stride 1
done
