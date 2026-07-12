#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SAVE_PATH="${SAVE_PATH:-runs/phase1/05_phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3_best}"
OUTPUT_DIR="${OUTPUT_DIR:-${SAVE_PATH}/cls_only_rescore_e9}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-6}"
PIXEL_STRIDE="${PIXEL_STRIDE:-4}"

python phase2b_anchor_diagnosis.py \
  --mode sweep \
  --save_path "${SAVE_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs 9 \
  --fixed_prompt_config phase1_hard \
  --fixed_score_rule cls_only \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --pixel_stride "${PIXEL_STRIDE}" \
  --dfg_mode attn \
  --dfg_attn_dim 256 \
  --dfg_attn_tau 8.0 \
  --use_ss2d_dfg \
  --dfg_gamma_max 0.2 \
  --dfg_ss2d_fusion weight_residual \
  --dfg_beta 0.10 \
  --dfg_beta_schedule warmup010 \
  --dfg_beta_target 0.10
