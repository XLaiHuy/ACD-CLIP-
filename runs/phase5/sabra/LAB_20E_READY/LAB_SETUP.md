Lab setup

Run on the lab GPU machine from a clean clone. Do not run full training on the
rented machine.

    git clone <repository-url> ACD-CLIP
    cd ACD-CLIP
    git checkout <LAB_20E_READY_SHA>
    git lfs install
    git lfs pull
    python3.12 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    python -m pip install scikit-learn==1.7.2
    cp runs/phase5/sabra/LAB_20E_READY/.env.example .env
    set -a; source .env; set +a
    python -c 'import torch, torchvision, sklearn; print(torch.__version__, torchvision.__version__, sklearn.__version__); print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda")'
    sha256sum -c runs/phase5/sabra/LAB_20E_READY/ASSET_HASHES.sha256
    python tools/sabra/lab_preflight.py --dataset visa --root "$VISA_ROOT"

MVTec is post-training-only. Medical is sealed and must not be inspected during
setup. Do not install a new NVIDIA driver.
