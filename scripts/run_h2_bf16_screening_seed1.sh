#!/usr/bin/env bash
set -euo pipefail

# Fresh Seed-1 BF16 H/A source training only.  This script deliberately does
# not invoke any target evaluator.  The caller must use a clean, committed
# checkout because the resulting checkpoints are a new scientific protocol.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/workspace/venv-acdclip/bin/python}"
SEED="${SEED:-1}"
RUN_ROOT="${RUN_ROOT:-/workspace/h2_bf16_seed1_v1}"
ANCHOR_LAMBDA="0.0021633926715180626"

if [[ "${SEED}" != "1" ]]; then
  echo "This frozen screening script is Seed-1 only." >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse an existing run root: ${RUN_ROOT}" >&2
  exit 2
fi
if ! git -C "${ROOT}" diff --quiet || ! git -C "${ROOT}" diff --cached --quiet; then
  echo "Refusing to train from a dirty checkout." >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable is unavailable: ${PYTHON}" >&2
  exit 2
fi

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

base=(
  "${ROOT}/train.py"
  --protocol_horizon 15 --dataset VisA --model_name ViT-L-14-336 --img_size 518
  --n_groups 3 --image_adapt_weight 0.2 --text_adapt_weight 0.2
  --conv_lora_rank 8 --conv_lora_alpha 2.0 --conv_kernel_size_list 3 5
  --lora_rank 16 --lora_alpha 2.0 --image_lr 0.001 --text_lr 0.0005
  --use_hybrid_soft_prompt --hybrid_alpha_max 0.2 --soft_prompt_freeze_epochs 3
  --soft_prompt_ctx_len 4 --soft_prompt_lr 0.00005 --soft_prompt_init phrase
  --soft_prompt_init_phrase "a photo of a" --lambda_kg 0.01 --lambda_k 0.002
  --lr_gamma 0.9 --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0
  --use_ss2d_dfg --dfg_gamma_max 0.2 --dfg_ss2d_fusion weight_residual
  --dfg_beta 0.10 --dfg_beta_schedule warmup010 --dfg_beta_target 0.10
  --grad_clip_norm 1.0 --batch_size 6 --num_workers 6 --pin_memory
  --no-persistent_workers --prefetch_factor 2 --no-non_blocking_copy
  --grad_checkpointing --precision bf16 --no-bf16_local_fp32_islands
  --seed "${SEED}" --deterministic_algorithms --non_finite_loss_abort_threshold 0
  --abort_on_nonfinite --telemetry_interval 25 --family_telemetry_interval 25
)

mkdir -p "${RUN_ROOT}"
{
  echo "git_sha=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "python=$(${PYTHON} -c 'import sys; print(sys.executable)')"
  echo "precision=bf16"
  echo "bf16_local_fp32_islands=false"
  echo "gradscaler_enabled=false"
  echo "dataset_source=$(readlink -f /workspace/data/VisA_20220922)"
} > "${RUN_ROOT}/runtime_environment.txt"

verify_checkpoint() {
  "${PYTHON}" - "$1" "$2" <<'PY'
import sys
import torch

path, expected_epoch = sys.argv[1], int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=False)
bad = []
for key, expected in {
    "epoch": expected_epoch,
    "precision": "bf16",
    "amp_enabled": True,
    "gradscaler_enabled": False,
    "tf32_enabled": False,
}.items():
    if payload.get(key) != expected:
        bad.append((key, payload.get(key), expected))
if payload.get("scaler_state") != {}:
    bad.append(("scaler_state", "nonempty", "{}"))
config = payload.get("resolved_scientific_config", {})
if config.get("precision") != "bf16" or config.get("bf16_local_fp32_islands") is not False:
    bad.append(("precision_config", config.get("precision"), config.get("bf16_local_fp32_islands")))
for section in ("model_state",):
    for group in payload.get(section, {}).values():
        for value in group.values():
            if torch.is_tensor(value) and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
                bad.append((section, str(value.dtype), "finite float32"))
for state in payload.get("optimizer_state", {}).get("state", {}).values():
    for value in state.values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            bad.append(("optimizer_state", "nonfinite", "finite"))
if bad:
    raise SystemExit(f"checkpoint invariant failure: {bad[:5]}")
print(f"BF16_CHECKPOINT=PASS epoch={expected_epoch} global_step={payload['global_step']}")
PY
}

shared="${RUN_ROOT}/shared_e1/adapter_1.pth"
"${PYTHON}" "${base[@]}" --epoch 1 --save_path "${RUN_ROOT}/shared_e1"
verify_checkpoint "${shared}" 1

"${PYTHON}" "${base[@]}" --epoch 15 --resume "${shared}" --save_path "${RUN_ROOT}/H"
verify_checkpoint "${RUN_ROOT}/H/adapter_15.pth" 15

"${PYTHON}" "${base[@]}" --epoch 15 --resume "${shared}" --save_path "${RUN_ROOT}/A" \
  --use_safe_anchor --anchor_lambda "${ANCHOR_LAMBDA}" --anchor_reference_path "${shared}" \
  --anchor_gradient_budget --anchor_family_budget 0.10 --anchor_family_audit
verify_checkpoint "${RUN_ROOT}/A/adapter_15.pth" 15

echo "BF16_SEED1_SOURCE_HA_E15_COMPLETE=${RUN_ROOT}"
