#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEDICAL_ROOT="${MEDICAL_ROOT:-/home/ai4/caohuy/ACD-CLIP-medical-test}"
RUN_ROOT="${RUN_ROOT:-/tmp/h2_clean_factorial_e20_20260902}"
RESULTS="${RESULTS:-${MEDICAL_ROOT}/medical_phase2cd_results/h2_clean_factorial_e20_20260902}"
CONDA_ENV="${CONDA_ENV:-torchhuy}"
RUN_EVAL="${RUN_EVAL:-NO}"
FINAL_FROZEN="${FINAL_FROZEN:-NO}"
EVAL_EPOCH="${EVAL_EPOCH:-15}"
NUM_WORKERS="${NUM_WORKERS:-6}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PY=(conda run --no-capture-output -n "${CONDA_ENV}" python)

if [[ "${EVAL_EPOCH}" != "15" && "${EVAL_EPOCH}" != "20" ]]; then
  echo "EVAL_EPOCH must be 15 or 20" >&2
  exit 2
fi

if [[ "${RUN_EVAL}" != "YES" || "${FINAL_FROZEN}" != "YES" ]]; then
  echo "Prepared only. Set RUN_EVAL=YES FINAL_FROZEN=YES after the E15/E20 freeze audit is committed."
  echo "This evaluates all H/A/C/AC arms at fixed E${EVAL_EPOCH}; it performs no checkpoint selection or tuning."
  echo "RUN_ROOT=${RUN_ROOT} RESULTS=${RESULTS}"
  exit 0
fi

test -d "${MEDICAL_ROOT}"
test -s "${MEDICAL_ROOT}/phase2cd_medical_eval.py"
mkdir -p "${RESULTS}"

for arm in H A C AC; do
  checkpoint="${RUN_ROOT}/${arm}/adapter_${EVAL_EPOCH}.pth"
  test -s "${checkpoint}"
  state="${arm}_E${EVAL_EPOCH}"
  output="${RESULTS}/${state}"
  if [[ -e "${output}/complete" ]]; then
    echo "Already complete: ${state}"
    continue
  fi
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite incomplete output: ${output}" >&2
    exit 1
  fi
  sha256="$(sha256sum "${checkpoint}" | awk '{print $1}')"
  echo "Evaluating ${state} checkpoint=${checkpoint} sha256=${sha256}"
  (
    cd "${MEDICAL_ROOT}"
    "${PY[@]}" phase2cd_medical_eval.py \
      --state "${state}" \
      --checkpoint "${checkpoint}" \
      --expected-sha256 "${sha256}" \
      --output-dir "${output}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --cuda-device "${CUDA_DEVICE}" \
      --pixel-stride 1
  ) 2>&1 | tee "${RESULTS}/${state}.evaluation.log"
done
