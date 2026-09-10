# ACD-CLIP++ Laboratory Machine Handoff

This handoff preserves the frozen research result and prepares exactly two
future fresh source-training runs. Do not start either run until the bootstrap
reports `READY_FOR_FRESH_TRAIN`.

## Checkout and Git LFS

```bash
git clone https://github.com/XLaiHuy/ACD-CLIP-.git
cd ACD-CLIP-
git checkout handoff/lab-transfer-20260910
git lfs pull
```

Verify that `aa77c76` is an ancestor of the checked-out handoff commit. Never
rewrite or force-push the frozen commit.

## Environment

Create and activate a laboratory environment with a CUDA-compatible PyTorch
build. The source machine used Python 3.11.16, PyTorch 2.7.1+cu128,
torchvision 0.22.1+cu128, and CUDA runtime 12.8. The exact source freeze is in
`handoff/lab_20260910/environment/requirements_exact.txt` and the runtime
details are in the other environment files.

```bash
python3 -m venv .venv-acd-clip
source .venv-acd-clip/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r handoff/lab_20260910/environment/requirements_exact.txt
```

The historical `requirements.txt` retains upstream torch pins for provenance.
Do not use it to downgrade a working laboratory CUDA/PyTorch installation.
If the laboratory GPU requires another compatible PyTorch wheel, install that
wheel first and then install the remaining packages from the exact freeze.
Record any deliberate hardware-specific difference before training.

## Dataset paths

Transfer raw datasets separately. Do not commit raw VisA, MVTec-AD, MedAD, or
Colon data to GitHub. See `DATASET_TRANSFER_MANIFEST.md` for safe rsync/scp
examples. The repository loader expects these paths:

```bash
mkdir -p data
ln -sfn /lab/data/acd-clip/VisA_20220922 data/VisA_20220922
ln -sfn /lab/data/acd-clip/mvtec-ad data/mvtec_ad
ln -sfn /lab/data/acd-clip/MedAD data/MedAD
ln -sfn /lab/data/acd-clip/Colon data/Colon
```

Review every destination path before creating the links. The metadata JSONL
files are tracked in `dataset/hub/`; the all-supervised MVTec manifest is
`dataset/hub/MVTec_all_supervised.jsonl` and its SHA256 is recorded in the
handoff files.

## Bootstrap checks

```bash
PYTHON="$PWD/.venv-acd-clip/bin/python" \
  bash handoff/lab_20260910/bootstrap_lab.sh
```

Bootstrap verifies branch ancestry, Git LFS, Python/CUDA, CLIP SHA256, all raw
dataset paths, metadata and manifest hashes, unit tests, and a small model and
dataloader smoke test. It does not train and does not evaluate a target
dataset.

## Future Run V: fresh VisA source

```bash
RUN_ROOT="$PWD/runs/fresh_visa_source_20260910" \
  PYTHON="$PWD/.venv-acd-clip/bin/python" \
  bash handoff/lab_20260910/run_fresh_visa_source.sh
```

This saves E1 through E20. Do not add `--resume` or change the source and
target-selection rule. The run is fresh training on VisA, then target-selected
Medical transfer.

## Future Run M: fresh MVTec-AD-all source

```bash
RUN_ROOT="$PWD/runs/fresh_mvtec_all_source_20260910" \
  PYTHON="$PWD/.venv-acd-clip/bin/python" \
  bash handoff/lab_20260910/run_fresh_mvtec_all_source.sh
```

This saves E1 through E20. It is supervised MVTec-AD-all source training; the
source includes normal and labelled anomalous images with official masks. Do
not describe MVTec as an untouched target for this run.

## Medical E1-E20 evaluation

Run this separately after the selected fresh training run completes:

```bash
RUN_ROOT="$PWD/runs/fresh_visa_source_20260910"
EVAL_ROOT="$PWD/results/fresh_visa_source_20260910/medical_epochs"
bash handoff/lab_20260910/run_medical_epoch_evaluation.sh \
  --run-root "$RUN_ROOT" --output-root "$EVAL_ROOT"
```

Repeat with the Run M paths. This evaluates Brain, Liver, Retina,
Colon_clinicDB, Colon_colonDB, and Colon_Kvasir at every epoch using
identity/no TTA, `benchmark_exact`, and pixel stride 1.

## Epoch-ranking summary

```bash
python handoff/lab_20260910/rank_medical_epochs.py \
  --results-root "$EVAL_ROOT" \
  --output-csv "${EVAL_ROOT%/medical_epochs}/MEDICAL_EPOCH_RANKING.csv" \
  --output-json "${EVAL_ROOT%/medical_epochs}/MEDICAL_EPOCH_RANKING.json"
```

The winner is the highest Medical macro Pixel AP, then higher Medical macro
Pixel AUROC, then earlier epoch. Keep all 20 rows. These are target-selected
exploratory transfer results, not untouched target confirmation.

## Frozen result reference

The current clean frozen result is still commit `aa77c76`, canonical H2 A15,
with locked TTA-F. Its reported Medical Pixel AP is 40.2459 percent and its
MVTec Pixel AP is 47.2975 percent. Do not replace these values with results
from Run V or Run M.

## Checkpoint and data handoff records

- `CHECKPOINT_MANIFEST.csv` records required checkpoint paths, sizes, hashes,
  epochs, roles, source datasets, interventions, and full-state contents.
- `DATASET_TRANSFER_MANIFEST.csv` records raw data paths, counts, sizes,
  class names, metadata hashes, mask/label status, and transfer requirements.
- `MVTEC_ALL_MANIFEST_AUDIT.md` records the new all-supervised MVTec manifest
  verification.
- `UPSTREAM_ACDCLIP_REFERENCE.md` records upstream commit
  `2685a6633d5466ff0255733d799ab9ce7b65a59a` and the upstream/project
  distinction.
