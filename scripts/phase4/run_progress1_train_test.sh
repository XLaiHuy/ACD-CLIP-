#!/usr/bin/env bash
set -euo pipefail

: "${SAVE_PATH:?Set SAVE_PATH, e.g. runs/phase4/progress1_cops_dynamic_prompt_seed0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-6}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-6}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-20}"
VALIDATION_EPOCHS_TEXT="${VALIDATION_EPOCHS:-8 9 10 11 12 13 14 15}"
read -r -a VALIDATION_EPOCHS <<< "${VALIDATION_EPOCHS_TEXT}"
MEDICAL_MANIFEST_ROOT="${MEDICAL_MANIFEST_ROOT:-${SAVE_PATH}/protocol/medical_manifests}"

if [[ "${SAVE_PATH}" == *"--resume"* || "${SAVE_PATH}" == *"--phase2b_checkpoint"* ]]; then
  echo "SAVE_PATH must be a directory, not an accidental resume/checkpoint flag." >&2
  exit 2
fi
if grep -Eq -- '--(phase2b_checkpoint|load_adapter|resume|init_from_progress|freeze_phase2b)' scripts/phase4/train_progress1.sh; then
  echo "The final Phase4 train script must not initialize from an existing checkpoint." >&2
  exit 2
fi
mkdir -p "${SAVE_PATH}"

echo "[PHASE4-P1] save_path=${SAVE_PATH} cuda=${CUDA_DEVICE} batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} test_batch=${TEST_BATCH_SIZE} workers=${NUM_WORKERS} precision=${PRECISION} seed=${SEED} epochs=${EPOCHS}"
echo "[PHASE4-P1] protocol=VisA-train -> medical-val sweep -> one medical-test epoch; val_epochs=${VALIDATION_EPOCHS[*]}"

python tools/prepare_phase4_medical_splits.py \
  --output-root "${MEDICAL_MANIFEST_ROOT}" --val-ratio 0.30 --seed 0

SAVE_PATH="${SAVE_PATH}" CUDA_DEVICE="${CUDA_DEVICE}" BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACCUM="${GRAD_ACCUM}" NUM_WORKERS="${NUM_WORKERS}" PRECISION="${PRECISION}" \
SEED="${SEED}" EPOCHS="${EPOCHS}" bash scripts/phase4/train_progress1.sh

SAVE_PATH="${SAVE_PATH}" CUDA_DEVICE="${CUDA_DEVICE}" TEST_BATCH_SIZE="${TEST_BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" MEDICAL_MANIFEST_ROOT="${MEDICAL_MANIFEST_ROOT}" \
bash scripts/phase4/test_6medical_exact.sh --split val "${VALIDATION_EPOCHS[@]}"

conda run --no-capture-output -n torchhuy python tools/summarize_phase4_results.py \
  --save_path "${SAVE_PATH}" --split val --epochs "${VALIDATION_EPOCHS[@]}"

BEST_EPOCH="$(conda run --no-capture-output -n torchhuy python -c \"import json; print(int(json.load(open('${SAVE_PATH}/medical_validation_selection.json'))['best_epoch']['epoch']))\")"
echo "[PHASE4-P1] selected_epoch=${BEST_EPOCH} from medical validation only"

SAVE_PATH="${SAVE_PATH}" CUDA_DEVICE="${CUDA_DEVICE}" TEST_BATCH_SIZE="${TEST_BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" MEDICAL_MANIFEST_ROOT="${MEDICAL_MANIFEST_ROOT}" \
bash scripts/phase4/test_6medical_exact.sh --split test "${BEST_EPOCH}"

conda run --no-capture-output -n torchhuy python tools/summarize_phase4_results.py \
  --save_path "${SAVE_PATH}" --split test --epochs "${BEST_EPOCH}"
