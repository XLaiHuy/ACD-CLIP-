#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_v2.sh"
SOURCE=""
CHECKPOINT=""
OUTPUT_ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --help|-h) echo "Usage: $0 --source V|M --checkpoint CHECKPOINT --output-root OUTPUT_ROOT"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "${SOURCE}" != "V" && "${SOURCE}" != "M" ]]; then echo "--source must be V or M" >&2; exit 2; fi
test -s "${CHECKPOINT}"
if [[ -z "${OUTPUT_ROOT}" ]]; then echo "--output-root is required" >&2; exit 2; fi
if [[ -e "${OUTPUT_ROOT}" ]]; then echo "Refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 2; fi
mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${OUTPUT_ROOT}/orchestration.log") 2>&1
CHECKPOINT_NAME="$(basename "${CHECKPOINT}")"
if [[ "${CHECKPOINT_NAME}" =~ ^adapter_([0-9]+)\.pth$ ]]; then EPOCH="${BASH_REMATCH[1]}"; else echo "checkpoint must be named adapter_N.pth" >&2; exit 2; fi
CHECKPOINT_SHA256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
MEDICAL=(Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir)
if [[ "${SOURCE}" = "V" ]]; then TARGET="MVTec"; else TARGET="VisA"; fi
require_common_inputs "$([[ "${TARGET}" = "MVTec" ]] && echo VisA || echo MVTec_all_supervised)"
echo "source=${SOURCE} selected_epoch=${EPOCH} checkpoint_sha256=${CHECKPOINT_SHA256}"
echo "single_view=identity"
echo "four_view=identity,hflip,vflip,hvflip implementation=sequential_reference"
record_run_environment "${OUTPUT_ROOT}/environment.txt"

for dataset in "${MEDICAL[@]}" "${TARGET}"; do
  eval_root="${OUTPUT_ROOT}/single_view/${dataset}"
  mkdir -p "${eval_root}"
  ln -s "$(readlink -f "${CHECKPOINT}")" "${eval_root}/adapter_${EPOCH}.pth"
  "${PYTHON}" "${ROOT}/test.py" \
    --dataset "${dataset}" --model_name ViT-L-14-336 --img_size 518 --n_groups 3 \
    --lora_rank 16 --lora_alpha 2.0 --conv_lora_rank 8 --conv_lora_alpha 2.0 \
    --conv_kernel_size_list 3 5 --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0 \
    --use_ss2d_dfg --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual --dfg_beta 0.10 \
    --dfg_beta_schedule warmup010 --dfg_beta_target 0.10 --use_hybrid_soft_prompt \
    --soft_prompt_ctx_len 4 --soft_prompt_init phrase --soft_prompt_init_phrase "a photo of a" \
    --batch_size 6 --num_workers "${NUM_WORKERS}" --evaluator_mode benchmark_exact \
    --pixel_stride 1 --save_path "${eval_root}"
done

"${PYTHON}" "${ROOT}/scripts/rental_locked_tta.py" \
  --checkpoint "${CHECKPOINT}" --expected-sha256 "${CHECKPOINT_SHA256}" \
  --output "${OUTPUT_ROOT}/four_view_tta.json" --datasets "${MEDICAL[@]}" "${TARGET}" \
  --batch-size 6 --num-workers "${NUM_WORKERS}"
"${PYTHON}" "${SCRIPT_DIR}/summarize_delta_tta.py" \
  --single-root "${OUTPUT_ROOT}/single_view" \
  --tta-json "${OUTPUT_ROOT}/four_view_tta.json" \
  --output "${OUTPUT_ROOT}/DELTA_TTA.json"
echo "SELECTED_CHECKPOINT_EVALUATION_COMPLETE"
