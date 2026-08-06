#!/usr/bin/env bash
set -euo pipefail

SAVE_PATH="runs/phase4/progress1_v8_structural_smoke_seed0"
rm -rf "${SAVE_PATH}"
mkdir -p "${SAVE_PATH}"

# Structural smoke test: 3 epochs, max 300 batches per epoch.
# Uses OpenAI pretrained CLIP initialization and newly initialized Phase2B/H6 trainable modules.
# We do NOT pass --checkpoint_path so it starts from scratch.
# We do NOT pass --medical_manifest_root (no exact medical test).
conda run --no-capture-output -n torchhuy python train.py \
  --save_path "${SAVE_PATH}" \
  --dataset "Brain" \
  --img_size 518 \
  --cuda_device 0 \
  --epoch 3 \
  --batch_size 2 \
  --h6_smoke_max_batches 300 \
  --h6_progress 1 \
  --h6_progress_version "P1-v8-minimal" \
  --h6_global_text_mode hard_anchor \
  --h6_prediction_routing dense \
  --no-h6_expert_enabled \
  --h6_diagnostics_mode light \
  --h6_diagnostics_interval 1
