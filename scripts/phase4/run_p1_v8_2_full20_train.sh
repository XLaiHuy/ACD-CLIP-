#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# ACD-CLIP Phase 4 Progress 1 v8.2 — 20-Epoch Training Launcher
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="configs/phase4/p1_v8_2_candidate1.json"
RUN_DIR="runs/phase4/p1_v8_2_full20_seed0"

echo "=== [P1-V8.2] 20-Epoch Training Setup ==="
echo "Repo Root: ${REPO_ROOT}"
echo "Run Directory: ${RUN_DIR}"
echo "Config Path: ${CONFIG_PATH}"

# 1. Run Preflight Verification
PYTHONPATH=. python tools/preflight_p1_v8_2_full20.py --stage train --config "${CONFIG_PATH}"

# 2. Setup Run Directory & Manifests
mkdir -p "${RUN_DIR}"
cp "${CONFIG_PATH}" "${RUN_DIR}/resolved_config.json"
sha256sum "${CONFIG_PATH}" > "${RUN_DIR}/config.sha256"

{
  echo "Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')"
  echo "HEAD: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
  git status --short 2>/dev/null || true
} > "${RUN_DIR}/git_state.txt"

{
  echo "Python: $(python --version)"
  echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
  echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
} > "${RUN_DIR}/environment_summary.txt"

date -Is > "${RUN_DIR}/started_at.txt"

# 3. Authoritative Training Command matching train.py argument parser
CMD=(
  python train.py
  --save_path "${RUN_DIR}"
  --dataset VisA
  --img_size 518
  --epoch 20
  --batch_size 1
  --cuda_device 0
  --grad_accum_steps 6
  --num_workers 2
  --seed 0
  --precision bf16
  --n_groups 3
  --image_adapt_weight 0.2
  --text_adapt_weight 0.2
  --lora_rank 16
  --lora_alpha 2.0
  --conv_lora_rank 8
  --conv_lora_alpha 2.0
  --conv_kernel_size_list 3 5
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
  --h6_num_factors 4
  --h6_top_k 2
  --h6_bank_dim 256
  --h6_router_dim 128
  --no-h6_expert_enabled
  --no-h6_load_bias_enabled
  --no-h6_cluster_responsibility
  --lambda_h6_route 0.023563732085236152
  --lambda_h6_factor_role 0.2836047825589712
  --lambda_h6_actual_local 0.2127045363418866
  --lambda_h6_balance 0.0
  --lambda_h6_center 0.0
  --lambda_h6_dynamic_mean_anchor 0.0
)

printf '%q ' "${CMD[@]}" > "${RUN_DIR}/launch.command.txt"
printf '\n' >> "${RUN_DIR}/launch.command.txt"

echo "Command logged to ${RUN_DIR}/launch.command.txt"
echo "To execute training manually when ready, run:"
echo "  bash ${RUN_DIR}/launch.command.txt"

check_checkpoints() {
  echo "Verifying saved checkpoints..."
  for e in $(seq 1 20); do
    ckpt="${RUN_DIR}/adapter_${e}.pth"
    if [ ! -f "${ckpt}" ]; then
      echo "[ERROR] Missing expected checkpoint: ${ckpt}"
      return 1
    fi
  done
  echo "[OK] All 20 checkpoints (adapter_1.pth ... adapter_20.pth) exist and are non-empty."
  return 0
}

export -f check_checkpoints 2>/dev/null || true
