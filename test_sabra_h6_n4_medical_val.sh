#!/usr/bin/env bash
set -euo pipefail

REPO=/home/ai4/caohuy/ACD-CLIP-lab20e-live
PYTHON=/home/ai4/ENTER/bin/python
RUN_ROOT=/home/ai4/caohuy/acdclip_runs/lab20e_b1_accum6_seed0

cd "$REPO"

# Fail early if the PyTorch-2.6 checkpoint compatibility fix is missing.
# Use POSIX grep so this launcher does not require the optional ripgrep binary.
grep -Fq 'torch.load(file, map_location=device, weights_only=False)' test.py

exec "$PYTHON" tools/sabra/run_medical_val_sweep.py \
  --run-root "$RUN_ROOT" \
  --checkpoint-epochs 10 12 14 16 18 20 \
  --medical-root /home/ai4/caohuy/data \
  --medical-manifest-root /home/ai4/caohuy/acdclip_runs/protocol/medical_seed0_val030 \
  --output-root "$RUN_ROOT/validation/h6_n4_exploratory_medical_val_even_epochs" \
  --python "$PYTHON" \
  --device 0
