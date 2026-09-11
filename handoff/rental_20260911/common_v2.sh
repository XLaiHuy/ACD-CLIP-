#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)"
PYTHON="${PYTHON:-python}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

require_common_inputs() {
  local dataset="$1"
  local manifest="${ROOT}/dataset/hub/${dataset}.jsonl"
  test "$(git -C "${ROOT}" branch --show-current)" = "experiment/fresh-source-rental-20260911"
  test -s "${ROOT}/model/ViT-L-14-336px.pt"
  test -s "${manifest}"
  test -d "${ROOT}/data/VisA_20220922"
  test -d "${ROOT}/data/mvtec_ad"
  test -d "${ROOT}/data/MedAD"
  test -d "${ROOT}/data/Colon"
  if ! find -L "${ROOT}/data/VisA_20220922" -type f -print -quit | grep -q .; then
    echo "VisA data path is empty" >&2
    return 2
  fi
  if ! find -L "${ROOT}/data/mvtec_ad" -type f -print -quit | grep -q .; then
    echo "MVTec data path is empty" >&2
    return 2
  fi
  if ! find -L "${ROOT}/data/MedAD" -type f -print -quit | grep -q .; then
    echo "MedAD data path is empty" >&2
    return 2
  fi
  if ! find -L "${ROOT}/data/Colon" -type f -print -quit | grep -q .; then
    echo "Colon data path is empty" >&2
    return 2
  fi
  if [[ "${dataset}" = "MVTec_all_supervised" ]]; then
    test "$(wc -l < "${manifest}")" -eq 5354
  fi
}

frozen_train_args() {
  TRAIN_ARGS=(
    --dataset "${DATASET}"
    --model_name ViT-L-14-336
    --img_size 518
    --n_groups 3
    --image_adapt_weight 0.2
    --text_adapt_weight 0.2
    --conv_lora_rank 8
    --conv_lora_alpha 2.0
    --conv_kernel_size_list 3 5
    --lora_rank 16
    --lora_alpha 2.0
    --image_lr 0.001
    --text_lr 0.0005
    --use_hybrid_soft_prompt
    --hybrid_alpha_max 0.2
    --soft_prompt_freeze_epochs 3
    --soft_prompt_ctx_len 4
    --soft_prompt_lr 0.00005
    --soft_prompt_init phrase
    --soft_prompt_init_phrase "a photo of a"
    --lambda_kg 0.01
    --lambda_k 0.002
    --lr_gamma 0.9
    --dfg_mode attn
    --dfg_attn_dim 256
    --dfg_attn_tau 8.0
    --use_ss2d_dfg
    --dfg_gamma_max 0.2
    --dfg_ss2d_fusion weight_residual
    --dfg_beta 0.10
    --dfg_beta_schedule warmup010
    --dfg_beta_target 0.10
    --grad_clip_norm 1.0
    --non_finite_loss_abort_threshold 20
    --batch_size 6
    --num_workers "${NUM_WORKERS}"
    --cuda_device "${CUDA_DEVICE}"
    --grad_checkpointing
    --amp
    --seed 0
    --deterministic_algorithms
  )
}

verify_full_checkpoint() {
  "${PYTHON}" - "$1" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
required = {
    "checkpoint_version", "protocol_version", "model_state", "image_adapter",
    "text_adapter", "optimizer_state", "scheduler_state", "scaler_state",
    "resolved_scientific_config", "parent_scientific_config",
    "resolved_operational_config", "config_sha256", "git_sha", "clip_sha256",
    "dataset_manifest_sha256", "epoch", "global_step", "dataloader_generator_state",
}
missing = sorted(required.difference(payload))
if missing:
    raise SystemExit(f"missing full-state checkpoint keys: {missing}")
print(
    "checkpoint_epoch=%d global_step=%d optimizer_groups=%d scheduler_last_epoch=%s" % (
        payload["epoch"], payload["global_step"],
        len(payload["optimizer_state"]["param_groups"]),
        payload["scheduler_state"].get("last_epoch"),
    )
)
PY
}

record_run_environment() {
  local output="$1"
  mkdir -p "$(dirname "${output}")"
  {
    echo "recorded_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "repo_root=${ROOT}"
    echo "git_branch=$(git -C "${ROOT}" branch --show-current)"
    echo "git_sha=$(git -C "${ROOT}" rev-parse HEAD)"
    echo "dataset=${DATASET:-unknown}"
    echo "python=$(${PYTHON} --version 2>&1)"
    echo "torch=$(${PYTHON} -c 'import torch; print(torch.__version__)' 2>&1)"
    echo "torch_cuda=$(${PYTHON} -c 'import torch; print(torch.version.cuda)' 2>&1)"
    echo "clip_sha256=$(sha256sum "${ROOT}/model/ViT-L-14-336px.pt" | awk '{print $1}')"
    echo "dataset_manifest_sha256=$(sha256sum "${ROOT}/dataset/hub/${DATASET:-unknown}.jsonl" 2>/dev/null | awk '{print $1}' || true)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
  } > "${output}"
}
