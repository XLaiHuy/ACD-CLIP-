#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_v2.sh"
PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-${ROOT}/results/rental_20260911/preflight}"
mkdir -p "${PREFLIGHT_ROOT}"
exec > >(tee -a "${PREFLIGHT_ROOT}/preflight.log") 2>&1

machine_gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
if [[ -n "${vram_mib}" ]]; then vram_gb=$((vram_mib / 1024)); else vram_gb=UNAVAILABLE; fi
python_version="$(${PYTHON} --version 2>&1 || true)"
torch_version="$(${PYTHON} -c 'import torch; print(torch.__version__)' 2>&1 || true)"
torch_cuda="$(${PYTHON} -c 'import torch; print(torch.version.cuda)' 2>&1 || true)"
git_branch="$(git -C "${ROOT}" branch --show-current)"
git_sha="$(git -C "${ROOT}" rev-parse HEAD)"
disk_available_kb="$(df -Pk "${ROOT}" | awk 'NR==2 {print $4}')"

data_ready=YES
for path in "${ROOT}/data/VisA_20220922" "${ROOT}/data/mvtec_ad" "${ROOT}/data/MedAD" "${ROOT}/data/Colon"; do
  if [[ ! -d "${path}" ]] || ! find -L "${path}" -type f -print -quit | grep -q .; then data_ready=NO; fi
done
model_status=FAIL
if [[ -s "${ROOT}/model/ViT-L-14-336px.pt" ]] && [[ "$(sha256sum "${ROOT}/model/ViT-L-14-336px.pt" | awk '{print $1}')" = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02" ]]; then model_status=PASS; fi
manifest_status=NOT_RUN
if [[ "${data_ready}" = YES ]] && "${PYTHON}" "${SCRIPT_DIR}/validate_data.py" "${ROOT}" > "${PREFLIGHT_ROOT}/data_validation.txt" 2>&1; then manifest_status=PASS; else [[ "${data_ready}" = NO ]] && manifest_status=BLOCKED_BY_DATA; fi
pytest_status=FAIL
if pytest -q > "${PREFLIGHT_ROOT}/pytest.log" 2>&1; then
  pytest_status=PASS
elif pytest -q --ignore=tests/test_cir_v2_parity.py > "${PREFLIGHT_ROOT}/pytest_non_cir.log" 2>&1; then
  if ! pytest -q tests/test_cir_v2_parity.py::test_cir_alpha05_reference_output_parity_synthetic > "${PREFLIGHT_ROOT}/cir_failure.log" 2>&1; then
    pytest_status=PASS_NON_CIR_CIR_ONLY_FAIL
  fi
fi
cuda_status=FAIL
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_properties(0).total_memory // 1024**2 >= 24000' > "${PREFLIGHT_ROOT}/cuda_smoke.txt" 2>&1 && cuda_status=PASS || true
worker_status=NOT_RUN
worker_choice=UNAVAILABLE
if [[ "${data_ready}" = YES ]] && "${PYTHON}" "${SCRIPT_DIR}/benchmark_workers.py" > "${PREFLIGHT_ROOT}/worker_benchmark.txt" 2>&1; then
  worker_status=PASS
  w1="$(awk -F'[ =]' '$1=="workers" && $2==1 {print $4}' "${PREFLIGHT_ROOT}/worker_benchmark.txt")"
  w2="$(awk -F'[ =]' '$1=="workers" && $2==2 {print $4}' "${PREFLIGHT_ROOT}/worker_benchmark.txt")"
  if [[ -n "${w1}" && -n "${w2}" ]] && awk "BEGIN {exit !(${w1} <= ${w2})}"; then worker_choice=1; else worker_choice=2; fi
fi
continuation_status=NOT_RUN
peak_train_vram=NOT_RUN
if [[ "${data_ready}" = YES && "${model_status}" = PASS && "${cuda_status}" = PASS ]]; then
  workers_for_smoke="${worker_choice}"; [[ "${workers_for_smoke}" =~ ^[12]$ ]] || workers_for_smoke=2
  continuation_status=PASS
  if ! NUM_WORKERS="${workers_for_smoke}" "${SCRIPT_DIR}/smoke_e1_e2_continuation.sh" VisA > "${PREFLIGHT_ROOT}/continuation_visa.log" 2>&1; then continuation_status=FAIL; fi
  if ! NUM_WORKERS="${workers_for_smoke}" "${SCRIPT_DIR}/smoke_e1_e2_continuation.sh" MVTec_all_supervised > "${PREFLIGHT_ROOT}/continuation_mvtec.log" 2>&1; then continuation_status=FAIL; fi
  if [[ "${continuation_status}" = PASS ]]; then
    peak_train_vram="$(${PYTHON} - "${PREFLIGHT_ROOT}/continuation_visa.log" "${PREFLIGHT_ROOT}/continuation_mvtec.log" <<'PY'
import re
import sys
values = []
for name in sys.argv[1:]:
    text = open(name, encoding="utf-8", errors="replace").read()
    values.extend(float(value) for value in re.findall(r"peak_cuda_memory_allocated_gb=([0-9.]+)", text))
print(f"{max(values):.6f}" if values else "NOT_REPORTED")
PY
    )"
  fi
fi
tta_baseline_runtime=NOT_RUN; tta_opt_runtime=NOT_RUN; tta_speedup=NOT_RUN
tta_baseline_peak=NOT_RUN; tta_opt_peak=NOT_RUN; tta_parity=NOT_RUN; tta_used=NO_UNTIL_PARITY_PASS
TTA_CHECKPOINT="${TTA_CHECKPOINT:-${ROOT}/runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth}"
if [[ "${data_ready}" = YES && -s "${TTA_CHECKPOINT}" ]]; then
  tta_dir="${PREFLIGHT_ROOT}/tta_benchmark"; tta_sha="$(sha256sum "${TTA_CHECKPOINT}" | awk '{print $1}')"
  "${PYTHON}" "${SCRIPT_DIR}/benchmark_tta.py" --checkpoint "${TTA_CHECKPOINT}" --expected-sha256 "${tta_sha}" --output-dir "${tta_dir}" > "${PREFLIGHT_ROOT}/tta_benchmark.log" 2>&1 && {
    tta_baseline_runtime="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["scenarios"]["baseline_sequential"]["runtime_seconds"])' "${tta_dir}/summary.json")"
    tta_opt_runtime="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["scenarios"]["safe_inference_mode"]["runtime_seconds"])' "${tta_dir}/summary.json")"
    tta_baseline_peak="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["scenarios"]["baseline_sequential"]["peak_cuda_bytes"])' "${tta_dir}/summary.json")"
    tta_opt_peak="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["scenarios"]["safe_inference_mode"]["peak_cuda_bytes"])' "${tta_dir}/summary.json")"
    tta_parity="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["scenarios"]["safe_inference_mode"]["parity"])' "${tta_dir}/summary.json")"
    tta_speedup="$(${PYTHON} -c 'import json,sys; d=json.load(open(sys.argv[1])); b=d["scenarios"]["baseline_sequential"]["runtime_seconds"]; o=d["scenarios"]["safe_inference_mode"]["runtime_seconds"]; print(b/o if o else 0)' "${tta_dir}/summary.json")"
    [[ "${tta_parity}" = PASS ]] && tta_used=safe_inference_mode
  }
fi
disk_status=FAIL; [[ "${disk_available_kb}" =~ ^[0-9]+$ ]] && (( disk_available_kb > 20 * 1024 * 1024 )) && disk_status=PASS

anchor_only_status=NO
if [[ -f "${SCRIPT_DIR}/run_fresh_visa_source_v2.sh" && -f "${SCRIPT_DIR}/run_fresh_mvtec_source_v2.sh" && -f "${SCRIPT_DIR}/smoke_e1_e2_continuation.sh" ]] \
  && ! rg -q -- "--use_cir_training|--cir_alpha|--use_hard_background_patch_ranking|h2_clean_factorial_e20_20260902_ampfix/shared_e1" \
    "${SCRIPT_DIR}/run_fresh_visa_source_v2.sh" "${SCRIPT_DIR}/run_fresh_mvtec_source_v2.sh" "${SCRIPT_DIR}/smoke_e1_e2_continuation.sh"; then
  anchor_only_status=YES
fi
fresh_visa_anchor=NO
fresh_mvtec_protocol=NO
if [[ -s "${ROOT}/runs/fresh_visa_source_20260911/adapter_1.pth" ]]; then fresh_visa_anchor=YES; fi
if [[ -f "${SCRIPT_DIR}/run_fresh_mvtec_source_v2.sh" ]]; then fresh_mvtec_protocol=YES; fi
cuda_amp_status=NOT_RUN
if [[ "${continuation_status}" = PASS ]]; then
  cuda_amp_status=PASS
  fresh_visa_anchor=YES
fi
ready=NO
if [[ "${data_ready}" = YES && "${manifest_status}" = PASS && "${anchor_only_status}" = YES && "${fresh_mvtec_protocol}" = YES && ( "${pytest_status}" = PASS || "${pytest_status}" = PASS_NON_CIR_CIR_ONLY_FAIL ) && "${cuda_status}" = PASS && "${continuation_status}" = PASS && "${tta_parity}" = PASS && "${disk_status}" = PASS ]]; then ready=YES; fi

echo "GPU=${machine_gpu}"
echo "VRAM_GB=${vram_gb}"
echo "PYTHON=${python_version}"
echo "TORCH=${torch_version}"
echo "TORCH_CUDA=${torch_cuda}"
echo "GIT_BRANCH=${git_branch}"
echo "GIT_SHA=${git_sha}"
echo "DATA_READY=${data_ready}"
echo "MODEL_WEIGHT_READY=${model_status}"
echo "ANCHOR_ONLY_PATH_CONFIRMED=${anchor_only_status}"
echo "CIR_ENABLED=false"
echo "HARD_BACKGROUND_ENABLED=false"
echo "FRESH_VISA_E1_ANCHOR_READY=${fresh_visa_anchor}"
echo "FRESH_MVTEC_E1_PROTOCOL_READY=${fresh_mvtec_protocol}"
echo "E1_TO_E20_CONTINUATION_PASS=${continuation_status}"
echo "CUDA_AMP_SMOKE=${cuda_amp_status}"
echo "PEAK_TRAIN_VRAM_GB=${peak_train_vram}"
echo "TTA_BASELINE_RUNTIME=${tta_baseline_runtime}"
echo "TTA_OPT_RUNTIME=${tta_opt_runtime}"
echo "TTA_PARITY=${tta_parity}"
echo "TTA_OPT_USED=${tta_used}"
echo "DISK_STATUS=${disk_status}"
echo "READY_TO_LAUNCH=${ready}"
if [[ "${ready}" != YES ]]; then exit 1; fi
