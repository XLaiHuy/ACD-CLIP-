#!/usr/bin/env bash
# Native RTX 4090 launcher for Phase2C P_LoRA_only.
#
# This intentionally does not use Kaggle paths or runtime manifest rewrites.
# It keeps the reviewed Phase2C protocol (seed 42, 15 epochs, batch 6) while
# using native BF16 Tensor Cores on Ada GPUs.
#
# Usage:
#   bash run_phase2c_PL_rtx4090_seed42.sh --dry-run
#   SAVE_PATH=runs/phase2c_4090/PL_lora_only_SMOKE \
#     bash run_phase2c_PL_rtx4090_seed42.sh --max-train-batches 3 --max-val-batches 2
#   bash run_phase2c_PL_rtx4090_seed42.sh
#
# Environment overrides:
#   PYTHON_BIN, CUDA_DEVICE, NUM_WORKERS, BATCH_SIZE, SAVE_PATH,
#   TRAIN_MANIFEST, VAL_MANIFEST, SPLIT_METADATA
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Keep batch 6 for protocol compatibility. Do not raise this for a final run
# unless all compared conditions use the same batch size.
BATCH_SIZE="${BATCH_SIZE:-6}"
SAVE_PATH="${SAVE_PATH:-runs/phase2c_4090/PL_lora_only_seed42}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-splits/visa_train_seed42.csv}"
VAL_MANIFEST="${VAL_MANIFEST:-splits/visa_val_seed42.csv}"
SPLIT_METADATA="${SPLIT_METADATA:-splits/visa_split_seed42_metadata.json}"
MODEL_WEIGHTS="model/ViT-L-14-336px.pt"
VISA_ROOT="data/VisA_20220922"

if [[ ! -f "${MODEL_WEIGHTS}" ]]; then
  echo "Missing ${MODEL_WEIGHTS}. Download the OpenAI ViT-L-14-336px weights first." >&2
  exit 1
fi
if [[ ! -d "${VISA_ROOT}" ]]; then
  echo "Missing ${VISA_ROOT}. Create a symlink to the VisA_20220922 dataset first." >&2
  exit 1
fi

# Helpful for variable-size allocations; harmless on the RTX 4090.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PHASE2C_CUDA_DEVICE="${CUDA_DEVICE}"

"${PYTHON_BIN}" -c 'import os, torch; d = int(os.environ["PHASE2C_CUDA_DEVICE"]); assert torch.cuda.is_available(), "CUDA is unavailable"; assert d < torch.cuda.device_count(), f"CUDA device {d} is unavailable"; assert torch.cuda.is_bf16_supported(), "This launcher requires native CUDA BF16 support"; print("GPU:", torch.cuda.get_device_name(d), "Torch:", torch.__version__, "CUDA:", torch.version.cuda)'

CMD=(
  "${PYTHON_BIN}" phase2c_train.py
  --condition P_LoRA_only
  --save-path "${SAVE_PATH}"
  --hybrid-alpha-max 0.20
  --train-manifest "${TRAIN_MANIFEST}"
  --val-manifest "${VAL_MANIFEST}"
  --split-metadata "${SPLIT_METADATA}"
  --cuda-device "${CUDA_DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --diagnostic-batch-size 1
  --bf16
)
CMD+=("$@")

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
