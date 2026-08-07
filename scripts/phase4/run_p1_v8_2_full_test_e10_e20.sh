#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# ACD-CLIP Phase 4 Progress 1 v8.2 — Full-Test Evaluation (Epochs 10–20)
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="configs/phase4/p1_v8_2_candidate1.json"
RUN_DIR="runs/phase4/p1_v8_2_full20_seed0"

echo "=== [P1-V8.2] Full-Test Evaluation Setup (Epochs 10–20) ==="
echo "Repo Root: ${REPO_ROOT}"
echo "Run Directory: ${RUN_DIR}"

# 1. Preflight check
PYTHONPATH=. python tools/preflight_p1_v8_2_full20.py --stage test --config "${CONFIG_PATH}"

# 2. Setup test evaluation script generation


echo "Checking status of checkpoints adapter_10.pth through adapter_20.pth..."
MISSING_CKPTS=0
for EPOCH in $(seq 10 20); do
  CKPT="${RUN_DIR}/adapter_${EPOCH}.pth"
  if [ ! -f "${CKPT}" ]; then
    MISSING_CKPTS=$((MISSING_CKPTS + 1))
  fi
done

if [ "${MISSING_CKPTS}" -gt 0 ]; then
  echo "[INFO] ${MISSING_CKPTS}/11 checkpoints not yet present. Test evaluation scripts will be generated and will execute after training completes."
else
  echo "[OK] All required evaluation checkpoints exist."
fi

# 3. Test loop over epochs 10-20 using exact test.py CLI argument parser
for EPOCH in $(seq 10 20); do
  EPOCH_DIR="${RUN_DIR}/test_epoch_${EPOCH}"
  mkdir -p "${EPOCH_DIR}"

  echo "--- Evaluator prepared for Epoch ${EPOCH} ---"
  date -Is > "${EPOCH_DIR}/started_at.txt"

  CMD=(
    python test.py
    --save_path "${RUN_DIR}"
    --epochs "${EPOCH}"
    --dataset VisA
    --img_size 518
    --batch_size 1
    --cuda_device 0
    --num_workers 2
    --n_groups 3
    --dfg_mode attn
    --dfg_attn_dim 256
    --dfg_attn_tau 8.0
    --use_ss2d_dfg
    --dfg_gamma_max 0.2
    --dfg_ss2d_fusion weight_residual
    --dfg_beta 0.10
    --dfg_beta_schedule fixed
    --h6_progress 1
    --h6_progress_version P1-v8-minimal
    --h6_global_text_mode hard_anchor
    --h6_local_factor_mode center_spread
    --h6_local_center_mix 0.05
    --h6_local_factor_spread 0.10
    --h6_prediction_routing dense
    --external_exact_pixel_metrics
    --h6_num_factors 4
    --h6_top_k 2
    --h6_bank_dim 256
    --h6_router_dim 128
    --no-h6_expert_enabled
    --no-h6_load_bias_enabled
    --no-h6_cluster_responsibility
    --lambda_h6_dynamic_mean_anchor 0.0
  )

  printf '%q ' "${CMD[@]}" > "${EPOCH_DIR}/test.command.txt"
  printf '\n' >> "${EPOCH_DIR}/test.command.txt"
done

echo "[OK] Evaluation scripts & commands generated for epochs 10 through 20."
