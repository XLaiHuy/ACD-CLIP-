#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)"
PYTHON="${PYTHON:-python}"
RUN_ROOT=""
OUTPUT_ROOT=""
NUM_WORKERS="${NUM_WORKERS:-6}"

usage() {
  echo "Usage: $0 --run-root RUN_ROOT --output-root OUTPUT_ROOT"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${RUN_ROOT}" || -z "${OUTPUT_ROOT}" ]]; then
  usage >&2
  exit 2
fi

for epoch in $(seq 1 20); do
  test -s "${RUN_ROOT}/adapter_${epoch}.pth"
done

mkdir -p "${OUTPUT_ROOT}"
DATASETS=(Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir)
EPOCHS=( $(seq 1 20) )

for dataset in "${DATASETS[@]}"; do
  eval_root="${OUTPUT_ROOT}/${dataset}"
  mkdir -p "${eval_root}"
  for checkpoint in "${RUN_ROOT}"/adapter_*.pth; do
    link="${eval_root}/$(basename "${checkpoint}")"
    if [[ -e "${link}" && ! -L "${link}" ]]; then
      echo "Refusing to overwrite non-symlink: ${link}" >&2
      exit 2
    fi
    ln -sfn "${checkpoint}" "${link}"
  done
  echo "Evaluating ${dataset} for epochs 1 through 20"
  "${PYTHON}" "${ROOT}/test.py" \
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
    --save_path "${eval_root}" \
    --epochs "${EPOCHS[@]}" \
    --evaluator_mode benchmark_exact \
    --pixel_stride 1
done

echo "Medical epoch evaluation complete"
