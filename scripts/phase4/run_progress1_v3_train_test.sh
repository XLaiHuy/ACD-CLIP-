#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="${SAVE_PATH:-runs/phase4/progress1_v3_safe_specialization_seed0}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-6}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-6}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-20}"
H6_DENSE_ROUTING_EPOCHS="${H6_DENSE_ROUTING_EPOCHS:-8}"
H6_SPARSE_TRANSITION_EPOCHS="${H6_SPARSE_TRANSITION_EPOCHS:-4}"
EXTERNAL_EXACT_PIXEL_METRICS="${EXTERNAL_EXACT_PIXEL_METRICS:-1}"
EXTERNAL_METRIC_CHUNK_PIXELS="${EXTERNAL_METRIC_CHUNK_PIXELS:-5000000}"
VALIDATION_EPOCHS_TEXT="${VALIDATION_EPOCHS:-8 9 10 11 12 13 14 15}"
read -r -a VALIDATION_EPOCHS <<< "${VALIDATION_EPOCHS_TEXT}"
MEDICAL_MANIFEST_ROOT="${MEDICAL_MANIFEST_ROOT:-${SAVE_PATH}/protocol/medical_manifests}"

if [[ "${SAVE_PATH}" == *"--resume"* || "${SAVE_PATH}" == *"--phase2b_checkpoint"* ]]; then
  echo "SAVE_PATH must be a directory, not an accidental resume/checkpoint flag." >&2
  exit 2
fi
if grep -Eq -- '--(phase2b_checkpoint|load_adapter|resume|init_from_progress|freeze_phase2b)' scripts/phase4/run_progress1_v3_train.sh; then
  echo "The Phase4 P1-v3 train script must not initialize from an existing checkpoint." >&2
  exit 2
fi
mkdir -p "${SAVE_PATH}"

echo "[PHASE4-P1-V3] save_path=${SAVE_PATH} cuda=${CUDA_DEVICE} batch=${BATCH_SIZE} grad_accum=${GRAD_ACCUM} test_batch=${TEST_BATCH_SIZE} workers=${NUM_WORKERS} precision=${PRECISION} seed=${SEED} epochs=${EPOCHS}"
echo "[PHASE4-P1-V3] selection_split=val selection_rule=six_dataset_macro_combined_score validation_epochs=${VALIDATION_EPOCHS[*]}"
echo "[PHASE4-P1-V3] test_split=test test_epochs_count=1 exact_external=${EXTERNAL_EXACT_PIXEL_METRICS} chunk_pixels=${EXTERNAL_METRIC_CHUNK_PIXELS}"

python tools/prepare_phase4_medical_splits.py \
  --output-root "${MEDICAL_MANIFEST_ROOT}" --val-ratio 0.30 --seed 0

SAVE_PATH="${SAVE_PATH}" CUDA_DEVICE="${CUDA_DEVICE}" BATCH_SIZE="${BATCH_SIZE}" \
GRAD_ACCUM="${GRAD_ACCUM}" NUM_WORKERS="${NUM_WORKERS}" PRECISION="${PRECISION}" \
SEED="${SEED}" EPOCHS="${EPOCHS}" H6_DENSE_ROUTING_EPOCHS="${H6_DENSE_ROUTING_EPOCHS}" \
H6_SPARSE_TRANSITION_EPOCHS="${H6_SPARSE_TRANSITION_EPOCHS}" \
bash scripts/phase4/run_progress1_v3_train.sh

SAVE_PATH="${SAVE_PATH}" CUDA_DEVICE="${CUDA_DEVICE}" TEST_BATCH_SIZE="${TEST_BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" MEDICAL_MANIFEST_ROOT="${MEDICAL_MANIFEST_ROOT}" \
H6_DENSE_ROUTING_EPOCHS="${H6_DENSE_ROUTING_EPOCHS}" H6_SPARSE_TRANSITION_EPOCHS="${H6_SPARSE_TRANSITION_EPOCHS}" \
EXTERNAL_EXACT_PIXEL_METRICS="${EXTERNAL_EXACT_PIXEL_METRICS}" EXTERNAL_METRIC_CHUNK_PIXELS="${EXTERNAL_METRIC_CHUNK_PIXELS}" \
bash scripts/phase4/run_progress1_v3_test.sh --split val "${VALIDATION_EPOCHS[@]}"

conda run --no-capture-output -n torchhuy python tools/summarize_phase4_results.py \
  --save_path "${SAVE_PATH}" --split val --epochs "${VALIDATION_EPOCHS[@]}"

BEST_EPOCH="$(SAVE_PATH="${SAVE_PATH}" conda run --no-capture-output -n torchhuy python -c 'import json, os; path = os.path.join(os.environ["SAVE_PATH"], "medical_validation_selection.json"); print(int(json.load(open(path))["best_epoch"]["epoch"]))')"
echo "[PHASE4-P1-V3] selected_epoch=${BEST_EPOCH} selection_split=val selection_rule=six_dataset_macro_combined_score"
echo "[PHASE4-P1-V3] test_split=test test_epochs_count=1"

SAVE_PATH="${SAVE_PATH}" CUDA_DEVICE="${CUDA_DEVICE}" TEST_BATCH_SIZE="${TEST_BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" MEDICAL_MANIFEST_ROOT="${MEDICAL_MANIFEST_ROOT}" \
H6_DENSE_ROUTING_EPOCHS="${H6_DENSE_ROUTING_EPOCHS}" H6_SPARSE_TRANSITION_EPOCHS="${H6_SPARSE_TRANSITION_EPOCHS}" \
EXTERNAL_EXACT_PIXEL_METRICS="${EXTERNAL_EXACT_PIXEL_METRICS}" EXTERNAL_METRIC_CHUNK_PIXELS="${EXTERNAL_METRIC_CHUNK_PIXELS}" \
bash scripts/phase4/run_progress1_v3_test.sh --split test "${BEST_EPOCH}"

conda run --no-capture-output -n torchhuy python tools/summarize_phase4_results.py \
  --save_path "${SAVE_PATH}" --split test --epochs "${BEST_EPOCH}"
