# Kaggle Setup Guide — Phase2C P_LoRA_only

This document describes the exact procedure for running the P_LoRA_only
ablation on Kaggle without violating the Phase2C scientific protocol.

## Hardware note

> [!WARNING]
> Kaggle T4 x2 does **not** automatically combine VRAM.
> Each T4 has approximately 15 GB of addressable VRAM.
> This ablation uses one GPU only.
> Do not add DDP or DataParallel for this experiment.

## Comparison validity note

A Kaggle PL run is exploratory relative to the local A-prime result because
different hardware, precision pipelines, and memory bandwidths may produce
numerically different results.

A fair comparison between A-prime and P_LoRA_only on Kaggle requires
rerunning **both** conditions on the same:

- commit
- GPU (single T4)
- precision (BF16)
- batch size (6)
- manifests (same runtime rewrites of the same source split)
- seed (42)
- protocol (15 epochs, same selection rule)

Alternatively, rerun P_LoRA_only on the local/lab RTX GPU when it becomes
available and compare directly against the local A-prime result.

---

## Workflow

### Step 1 — Enable Kaggle GPU

In the Kaggle notebook settings, select **Accelerator: GPU T4 x2**.

> [!NOTE]
> Even though two T4 GPUs are available, only GPU 0 is used.
> `CUDA_VISIBLE_DEVICES=0` is set automatically by the launcher.

### Step 2 — Attach the VisA dataset

In Kaggle > Input datasets, attach your uploaded VisA dataset.

Note the mount point (typically `/kaggle/input/<dataset-name>/`).

### Step 3 — Clone or upload the repository at the reviewed commit

```bash
# In a notebook cell
!git clone https://github.com/XLaiHuy/ACD-CLIP-.git /kaggle/working/ACD-CLIP
%cd /kaggle/working/ACD-CLIP
!git checkout <reviewed-commit-sha>
```

> [!IMPORTANT]
> Always work from the exact reviewed and tested commit, not HEAD.

### Step 4 — Confirm active branch/commit

```bash
!git branch --show-current
!git log -1 --oneline
```

Expected output:

```
phase2c-pl-kaggle
<sha>  <commit message>
```

### Step 5 — Inspect /kaggle/input

```bash
!ls /kaggle/input/
!ls /kaggle/input/<dataset-name>/   # replace with actual dataset name
```

Identify the root of the VisA image tree.

### Step 6 — Install only missing dependencies

```bash
!pip install -q torchmetrics tqdm
```

Do not reinstall PyTorch; Kaggle provides a compatible version.

### Step 7 — Generate runtime manifests

```bash
!python scripts/prepare_kaggle_manifests.py \
    --train-manifest splits/visa_train_seed42.csv \
    --val-manifest   splits/visa_val_seed42.csv \
    --old-root       /home/ai4/data/VisA \
    --new-root       /kaggle/input/<dataset-name>/VisA \
    --output-dir     /kaggle/working/runtime_splits
```

Confirm the script exits with code 0 and reports 0 missing paths.

### Step 8 — Run preflight check

```bash
!python scripts/kaggle_preflight.py \
    --train-manifest /kaggle/working/runtime_splits/visa_train_seed42_kaggle.csv \
    --val-manifest   /kaggle/working/runtime_splits/visa_val_seed42_kaggle.csv \
    --split-metadata splits/visa_split_seed42_metadata.json \
    --output-dir     /kaggle/working/runs/phase2c_kaggle/PL_lora_only_seed42
```

Expected: `Preflight PASSED.`

### Step 9 — Source compilation checks

```bash
!python -m py_compile \
    phase2c_pcgrad.py \
    phase2c_pcgrad_diagnostics.py \
    phase2c_utils.py \
    phase2c_train.py
```

### Step 10 — Run Phase2C unit and regression tests

```bash
!python -m unittest discover -s tests -p "test_phase2c_*.py" -v 2>&1 | head -100
```

All tests must pass.

### Step 11 — Run PL dry-run

```bash
!TRAIN_MANIFEST=/kaggle/working/runtime_splits/visa_train_seed42_kaggle.csv \
 VAL_MANIFEST=/kaggle/working/runtime_splits/visa_val_seed42_kaggle.csv \
 RUN_DIR=/kaggle/working/runs/phase2c_kaggle/PL_lora_only_seed42 \
 bash run_phase2c_PL_kaggle_seed42.sh --dry-run
```

Inspect the printed JSON config.  Confirm:

- `condition`: `P_LoRA_only`
- `pcgrad_groups`: `["shared_image_lora"]`
- `pcgrad_enabled`: `true`
- `hybrid_alpha_max`: `0.20`
- `seed`: `42`
- `epochs`: `15`

### Step 12 — Bounded smoke test (1 epoch, 3 batches)

```bash
!TRAIN_MANIFEST=/kaggle/working/runtime_splits/visa_train_seed42_kaggle.csv \
 VAL_MANIFEST=/kaggle/working/runtime_splits/visa_val_seed42_kaggle.csv \
 RUN_DIR=/kaggle/working/runs/phase2c_kaggle/PL_lora_only_SMOKE \
 bash run_phase2c_PL_kaggle_seed42.sh --max-train-batches 3 --max-val-batches 2
```

> [!NOTE]
> The smoke run uses a distinct output directory (`..._SMOKE`).
> Smoke output is NOT a scientific result.

### Step 13 — Inspect smoke output

Check:

- VRAM usage during the run (should stay below ~14 GB)
- Training throughput (items/second printed by tqdm)
- No NaN or Inf in loss or gradients
- `runs/phase2c_kaggle/PL_lora_only_SMOKE/` contains:
  - `config.json` with `"smoke_test": true`
  - `train.log`
  - `gradient_diagnostics.csv` (all four groups logged)
  - `pcgrad_diagnostics.csv` (only `shared_image_lora` logged)

### Step 14 — Stop for manual review

> [!CAUTION]
> Do not start the 15-epoch run without reviewing the smoke output and
> obtaining explicit approval.

### Step 15 — Full 15-epoch run (only after approval)

```bash
# DO NOT RUN UNTIL MANUAL APPROVAL
!TRAIN_MANIFEST=/kaggle/working/runtime_splits/visa_train_seed42_kaggle.csv \
 VAL_MANIFEST=/kaggle/working/runtime_splits/visa_val_seed42_kaggle.csv \
 RUN_DIR=/kaggle/working/runs/phase2c_kaggle/PL_lora_only_seed42 \
 bash run_phase2c_PL_kaggle_seed42.sh
```

### Step 16 — Export artifacts before the session ends

Kaggle notebooks lose all data when the session ends.  Before stopping:

```python
# In a notebook cell
import shutil
shutil.make_archive(
    "/kaggle/working/PL_lora_only_seed42_artifacts",
    "zip",
    "/kaggle/working/runs/phase2c_kaggle/PL_lora_only_seed42",
)
```

Then download the ZIP from the Kaggle output panel.

Key files to preserve:

- `config.json`
- `train.log`
- `visa_val_metrics.csv`
- `selection.json`
- `gradient_diagnostics.csv`
- `pcgrad_diagnostics.csv`
- `diagnostic_batches.json`
- `split_metadata.json`
- `checkpoints/adapter_<selected_epoch>.pth`

---

## Exact commands summary

| Step | Command |
|------|---------|
| Source checks | `python -m py_compile phase2c_pcgrad.py phase2c_pcgrad_diagnostics.py phase2c_utils.py phase2c_train.py` |
| Unit tests | `python -m unittest discover -s tests -p "test_phase2c_*.py" -v` |
| Manifest generation | `python scripts/prepare_kaggle_manifests.py --train-manifest ... --val-manifest ... --old-root ... --new-root ... --output-dir ...` |
| Preflight | `python scripts/kaggle_preflight.py --train-manifest ... --val-manifest ... --split-metadata ... --output-dir ...` |
| Dry-run | `bash run_phase2c_PL_kaggle_seed42.sh --dry-run` |
| Smoke test | `bash run_phase2c_PL_kaggle_seed42.sh --max-train-batches 3 --max-val-batches 2` |
| **Full run (approval required)** | `bash run_phase2c_PL_kaggle_seed42.sh` |
