# RTX 4090 setup: Phase2C P_LoRA_only

Use branch `phase2c-pl-kaggle`, not `phase2c`: the latter does not define the
`P_LoRA_only` condition.

## Required assets

The repository already contains the VisA metadata and seed-42 split files.
Before training, provide only:

1. OpenAI `ViT-L-14-336px.pt` at `model/ViT-L-14-336px.pt`.
2. The VisA image tree, exposed as `data/VisA_20220922` (a symbolic link is
   fine).

No Phase-1 adapter checkpoint is loaded by `phase2c_train.py`.

```bash
mkdir -p model data
curl -L --fail --retry 3 \
  -o model/ViT-L-14-336px.pt \
  https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt
ln -s /absolute/path/to/VisA_20220922 data/VisA_20220922
```

## Verify and run

Do not reinstall PyTorch if the rental image already exposes CUDA and the RTX
4090. Install only missing Python packages, run the test suite, then smoke
test. RTX 4090 supports BF16 natively, so the launcher uses BF16; its
numerically sensitive text, fusion, logits, and losses remain FP32.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
python -m pip install -U numpy pandas regex ftfy==6.3.1 kornia==0.8.1 Pillow==11.3.0 torchmetrics tqdm
python -m py_compile phase2c_train.py phase2c_pcgrad.py phase2c_utils.py
python -m unittest discover -s tests -p 'test_phase2c_*.py' -v

SAVE_PATH=runs/phase2c_4090/PL_lora_only_seed42_SMOKE \
  bash run_phase2c_PL_rtx4090_seed42.sh --max-train-batches 3 --max-val-batches 2
```

After the smoke test is clean, run the protocol-compatible full experiment:

```bash
SAVE_PATH=runs/phase2c_4090/PL_lora_only_seed42 \
  bash run_phase2c_PL_rtx4090_seed42.sh
```

The default batch size is 6. Keep it unchanged for a comparable final result.
