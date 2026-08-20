# LAB_20E_READY_V2 setup

Run this only on the lab GPU. The rented-machine package boundary is source-only: no 20e training, MVTec rerun, or Medical read occurred here.

```bash
git clone <repository-url> ACD-CLIP
cd ACD-CLIP
git fetch origin research/p5-sabra-g
# Set LAB_20E_READY_SHA_V2 to the commit printed by the final handoff.
git checkout "$LAB_20E_READY_SHA_V2"
git lfs install
git lfs pull
bash scripts/setup_lab_env.sh
cp runs/phase5/sabra/LAB_20E_READY_V2/.env.example .env
set -a; source .env; set +a
sha256sum -c runs/phase5/sabra/LAB_20E_READY_V2/ASSET_HASHES.sha256
python -c 'import torch, torchvision, sklearn; print(torch.__version__, torchvision.__version__, sklearn.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The setup script installs the frozen `torch==2.5.1+cu121` and
`torchvision==0.20.1+cu121` from the official PyTorch cu121 index before the remaining pins. It does not install a driver. Do not run Medical preflight during setup.
