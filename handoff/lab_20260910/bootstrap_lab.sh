#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)"
PYTHON="${PYTHON:-python}"
EXPECTED_BRANCH="handoff/lab-transfer-20260910"
FROZEN_COMMIT="aa77c76baa49cd2fb621fde7a11a8850313782de"
CLIP_SHA256="3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
MVTEC_ALL_SHA256="1b850c1b52de3a08db0ad4e14adb51933f716bb69e66776a9ed5cde71f6d7129"

failures=0
pass_check() { echo "PASS: $*"; }
fail_check() { echo "FAIL: $*" >&2; failures=$((failures + 1)); }

echo "LAB BOOTSTRAP START"
echo "repo_root=${ROOT}"

branch="$(git -C "${ROOT}" branch --show-current)"
if [[ "${branch}" == "${EXPECTED_BRANCH}" ]]; then pass_check "branch ${branch}"; else fail_check "branch is ${branch}, expected ${EXPECTED_BRANCH}"; fi
if git -C "${ROOT}" merge-base --is-ancestor "${FROZEN_COMMIT}" HEAD; then
  pass_check "frozen commit ${FROZEN_COMMIT} is an ancestor"
else
  fail_check "frozen commit ${FROZEN_COMMIT} is not reachable"
fi

if git -C "${ROOT}" lfs version >/dev/null 2>&1; then pass_check "Git LFS available"; else fail_check "Git LFS unavailable"; fi
if git -C "${ROOT}" lfs ls-files >/dev/null 2>&1; then pass_check "Git LFS index readable"; else fail_check "Git LFS index unreadable"; fi

if "${PYTHON}" --version >/dev/null 2>&1; then pass_check "Python available"; else fail_check "Python unavailable"; fi
torch_report="$(${PYTHON} -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())' 2>&1)" || true
if [[ "${torch_report}" == *$'\nTrue' ]]; then
  pass_check "PyTorch CUDA available"
else
  torch_report_flat="$(printf '%s' "${torch_report}" | tr '\n' ';')"
  fail_check "PyTorch CUDA check failed: ${torch_report_flat}"
fi
if command -v nvidia-smi >/dev/null 2>&1; then pass_check "nvidia-smi available"; else fail_check "nvidia-smi unavailable"; fi

clip_path="${ROOT}/model/ViT-L-14-336px.pt"
if [[ -s "${clip_path}" ]] && [[ "$(sha256sum "${clip_path}" | awk '{print $1}')" == "${CLIP_SHA256}" ]]; then
  pass_check "CLIP base weight and SHA256"
else
  fail_check "CLIP base weight missing or SHA256 mismatch"
fi

for data_path in data/VisA_20220922 data/mvtec_ad data/MedAD data/Colon; do
  if [[ -d "${ROOT}/${data_path}" ]]; then pass_check "dataset path ${data_path}"; else fail_check "dataset path missing ${data_path}"; fi
done

for manifest in VisA MVTec Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir; do
  if [[ -s "${ROOT}/dataset/hub/${manifest}.jsonl" ]]; then pass_check "metadata ${manifest}.jsonl"; else fail_check "metadata missing ${manifest}.jsonl"; fi
done
all_manifest="${ROOT}/dataset/hub/MVTec_all_supervised.jsonl"
if [[ -s "${all_manifest}" ]] && [[ "$(sha256sum "${all_manifest}" | awk '{print $1}')" == "${MVTEC_ALL_SHA256}" ]]; then
  pass_check "MVTec all manifest SHA256"
else
  fail_check "MVTec all manifest missing or SHA256 mismatch"
fi

set +e
"${PYTHON}" - "${ROOT}" "${MVTEC_ALL_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_sha = sys.argv[2]
manifest = root / "dataset/hub/MVTec_all_supervised.jsonl"
raw_root = root / "data/mvtec_ad"
rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == expected_sha
assert len(rows) == 5354
assert sum(int(row["label"]) == 0 for row in rows) == 4096
assert sum(int(row["label"]) == 1 for row in rows) == 1258
assert len({row["image_path"] for row in rows}) == len(rows)
for row in rows:
    assert (raw_root / row["image_path"]).is_file(), row["image_path"]
    if row["label"]:
        assert (raw_root / row["mask_path"]).is_file(), row["mask_path"]
print("manifest_records=5354 normal=4096 anomaly=1258 missing=0 duplicates=0")
PY
manifest_status=$?
set -e
if [[ ${manifest_status} -eq 0 ]]; then pass_check "MVTec all manifest records and masks"; else fail_check "MVTec all manifest validation"; fi

set +e
"${PYTHON}" - "${ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checks = {
    "VisA": (root / "data/VisA_20220922", root / "dataset/hub/VisA.jsonl"),
    "MVTec": (root / "data/mvtec_ad", root / "dataset/hub/MVTec.jsonl"),
    "Brain": (root / "data/MedAD/Brain_AD/test", root / "dataset/hub/Brain.jsonl"),
    "Liver": (root / "data/MedAD/Liver_AD/test", root / "dataset/hub/Liver.jsonl"),
    "Retina": (root / "data/MedAD/Retina_RESC_AD/test", root / "dataset/hub/Retina.jsonl"),
    "Colon_clinicDB": (root / "data/Colon/CVC-ClinicDB", root / "dataset/hub/Colon_clinicDB.jsonl"),
    "Colon_colonDB": (root / "data/Colon/CVC-ColonDB", root / "dataset/hub/Colon_colonDB.jsonl"),
    "Colon_Kvasir": (root / "data/Colon/Kvasir", root / "dataset/hub/Colon_Kvasir.jsonl"),
}
expected_hashes = {
    "VisA": "468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842",
    "MVTec": "3a5e304ea16bba82e6e525d188698e91ca92b718696f8c257ed435d235b4cc2c",
    "Brain": "89092dd5f3e36d2e611b115b2a97e4e9ee83af183ebec298abac983a7a323e4e",
    "Liver": "1483b5a43f011a3ef02211d5fa81c5b09031423bd0ca5c0ef6cbf0375fee4fc8",
    "Retina": "d0de975045262b321851ac3770eb7b5e68d4d7fb3bdba833b1cbbbe32f212e24",
    "Colon_clinicDB": "1f057657a64221672a5123c3e87b926d226b9eb6a3276768385ca3a7554cdb5c",
    "Colon_colonDB": "e3be9a5e158bef9a2c7f481827339798152c78e92590df9631c4281a6b6c31c3",
    "Colon_Kvasir": "ac948309511f02e8ec66b9c3b5dbc4a4be5e85d22dbd17742c74212ceee2ee94",
}
for name, (data_root, manifest) in checks.items():
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert rows, name
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == expected_hashes[name], name
    for row in rows:
        assert (data_root / row["image_path"]).is_file(), (name, row["image_path"])
        if int(row["label"]):
            assert (data_root / row["mask_path"]).is_file(), (name, row["mask_path"])
print("dataset_manifests=8 image_paths_and_masks=valid hashes=valid")
PY
medical_status=$?
set -e
if [[ ${medical_status} -eq 0 ]]; then pass_check "VisA and Medical dataset manifests"; else fail_check "VisA or Medical dataset manifest validation"; fi

echo "Running unit tests"
set +e
test_output="$(${PYTHON} -m pytest -q "${ROOT}/tests" 2>&1)"
test_status=$?
set -e
if [[ ${test_status} -eq 0 ]]; then
  pass_check "unit tests"
else
  fail_check "unit tests"
  echo "${test_output}" | tail -80
fi

echo "Running model and dataloader smoke test"
set +e
smoke_output="$(${PYTHON} - "${ROOT}" <<'PY'
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from dataset import get_text_and_image_dataset
from model.adapter import ACDCLIP
from model.clip import create_model

for name in ("VisA", "MVTec_all_supervised"):
    dataset = get_text_and_image_dataset(name, 518, "train")
    assert len(dataset) > 0, name
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 518, 518), name
    assert tuple(sample["mask"].shape) == (1, 518, 518), name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = create_model(
    model_name="ViT-L-14-336",
    img_size=518,
    device=device,
    pretrained="openai",
    require_pretrained=True,
)
model = ACDCLIP(
    clip_model=clip_model,
    n_groups=3,
    image_adapt_weight=0.2,
    conv_lora_rank=8,
    conv_lora_alpha=2.0,
    conv_kernel_size_list=[3, 5],
    text_adapt_weight=0.2,
    lora_rank=16,
    lora_alpha=2.0,
    dfg_mode="attn",
    dfg_attn_dim=256,
    dfg_attn_tau=8.0,
    use_ss2d_dfg=True,
    dfg_gamma_max=0.2,
    dfg_ss2d_fusion="weight_residual",
    dfg_beta=0.10,
    dfg_beta_schedule="warmup010",
    dfg_beta_target=0.10,
    dfg_beta_current=0.10,
).to(device)
model.eval()
print("model_type=ACDCLIP device=%s dataloader_samples=ok" % device)
PY
)"
smoke_status=$?
set -e
if [[ ${smoke_status} -eq 0 ]]; then
  pass_check "model and dataloader smoke test"
  echo "${smoke_output}"
else
  fail_check "model and dataloader smoke test"
  echo "${smoke_output}" | tail -80
fi

if [[ ${failures} -eq 0 ]]; then
  echo "READY_FOR_FRESH_TRAIN"
  exit 0
fi
echo "BOOTSTRAP_BLOCKED failures=${failures}" >&2
exit 1
