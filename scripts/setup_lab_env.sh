#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_PATH="${VENV_PATH:-${REPO_ROOT}/.venv}"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
PYPI_INDEX="https://pypi.org/simple"

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip

# Install the frozen CUDA wheels from the official cu121 index explicitly.
python -m pip install \
  --index-url "${TORCH_INDEX}" \
  --extra-index-url "${PYPI_INDEX}" \
  "torch==2.5.1+cu121" "torchvision==0.20.1+cu121"

# Install the remaining repository pins without asking the default index to
# resolve the local-version CUDA requirements a second time.
REQ_TMP="$(mktemp)"
trap 'rm -f "${REQ_TMP}"' EXIT
sed -e '/^torch==/d' -e '/^torchvision==/d' "${REPO_ROOT}/requirements.txt" > "${REQ_TMP}"
python -m pip install --index-url "${PYPI_INDEX}" -r "${REQ_TMP}"
python -m pip install --index-url "${PYPI_INDEX}" "scikit-learn==1.7.2"

python - <<'PY'
import torch
import torchvision
import sklearn
assert torch.__version__ == "2.5.1+cu121", torch.__version__
assert torchvision.__version__ == "0.20.1+cu121", torchvision.__version__
assert sklearn.__version__ == "1.7.2", sklearn.__version__
print({
    "python": __import__("sys").version,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
PY

sha256sum -c "${REPO_ROOT}/runs/phase5/sabra/LAB_20E_READY_V2/ASSET_HASHES.sha256"
printf '%s\n' "LAB_ENVIRONMENT_BOOTSTRAP_PASS"
