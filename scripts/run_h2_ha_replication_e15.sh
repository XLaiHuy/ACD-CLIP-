#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED="${SEED:-}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/h2_ha_replication_e15_seed${SEED}}"
CONDA_ENV="${CONDA_ENV:-torchhuy}"
DRY_RUN="${DRY_RUN:-NO}"
ANCHOR_LAMBDA_ACTIVE="0.0021633926715180626"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF

if [[ "${SEED}" != "1" && "${SEED}" != "2" ]]; then
  echo "SEED must be exactly 1 or 2" >&2
  exit 2
fi
if [[ "${RUN_ROOT}" == /tmp || "${RUN_ROOT}" == /tmp/* ]]; then
  echo "Refusing non-persistent RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse existing replication root: ${RUN_ROOT}" >&2
  exit 2
fi

SHARED_E1="${RUN_ROOT}/shared_e1/adapter_1.pth"
H15="${RUN_ROOT}/H/adapter_15.pth"
A15="${RUN_ROOT}/A/adapter_15.pth"
PY=(conda run --no-capture-output -n "${CONDA_ENV}" python)

base_args=(
  "${ROOT}/train.py"
  --protocol_horizon 15
  --dataset VisA
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
  --num_workers 6
  --grad_checkpointing
  --amp
  --seed "${SEED}"
  --anchor_grad_audit_interval 0
  --deterministic_algorithms
)

if [[ "${DRY_RUN}" == "YES" ]]; then
  echo "REPLICATION_DRY_RUN=PASS seed=${SEED} horizon=15 arms=H,A root=${RUN_ROOT} anchor_lambda=${ANCHOR_LAMBDA_ACTIVE} rho=0.10"
  exit 0
fi

mkdir -p "${RUN_ROOT}"
{
  echo "seed=${SEED}"
  echo "run_root=${RUN_ROOT}"
  echo "protocol_horizon=15"
  echo "arms=H,A"
  echo "PYTHONHASHSEED=${PYTHONHASHSEED}"
  echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}"
  echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  echo "conda_env=${CONDA_ENV}"
  echo "git_sha=$(git -C "${ROOT}" rev-parse HEAD)"
  echo "git_status=$(git -C "${ROOT}" status --short)"
  echo "started_at=$(date -Is)"
} > "${RUN_ROOT}/replication_runtime_environment.txt"

"${PY[@]}" "${base_args[@]}" \
  --epoch 1 \
  --save_path "${RUN_ROOT}/shared_e1"
test -s "${SHARED_E1}"

"${PY[@]}" -c 'import sys, torch
p=sys.argv[1]
x=torch.load(p, map_location="cpu", weights_only=False)
required=("checkpoint_version","protocol_version","model_state","image_parameter_reference","optimizer_state","scheduler_state","scaler_state","python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","dataloader_generator_state","epoch","global_step","parent_scientific_config","resolved_operational_config","resolved_scientific_config","config_sha256","base_h2_commit","implementation_git_sha","working_tree_diff_sha256","git_sha","clip_sha256","dataset_manifest_sha256","seed","precision","amp_enabled","tf32_enabled")
missing=[k for k in required if k not in x]
cfg=x.get("resolved_scientific_config", {})
bad=[]
if int(x.get("checkpoint_version", 0)) != 3: bad.append(("checkpoint_version", x.get("checkpoint_version")))
if x.get("protocol_version") != "H2_CLEAN_REPRO_ANCHOR_CIR_V2_REDTEAM": bad.append(("protocol_version", x.get("protocol_version")))
if x.get("epoch") != 1: bad.append(("epoch", x.get("epoch")))
if x.get("seed") != int(sys.argv[2]): bad.append(("seed", x.get("seed")))
for k,v in (("training_horizon",15),("primary_horizon",15),("secondary_horizon",20)):
    if cfg.get(k) != v: bad.append((k, cfg.get(k)))
if missing or bad: raise SystemExit(f"invalid shared E1: missing={missing} bad={bad}")
for name,state in (("model_state",x["model_state"]),("image_parameter_reference",x["image_parameter_reference"])):
    for key,value in state.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all(): raise SystemExit(f"nonfinite {name}:{key}")
' "${SHARED_E1}" "${SEED}"

for arm in H A; do
  args=("${base_args[@]}" --epoch 15 --resume "${SHARED_E1}" --save_path "${RUN_ROOT}/${arm}")
  if [[ "${arm}" == A ]]; then
    args+=(--use_safe_anchor --anchor_lambda "${ANCHOR_LAMBDA_ACTIVE}" --anchor_reference_path "${SHARED_E1}" --anchor_gradient_budget --anchor_family_budget 0.10 --anchor_family_audit)
  fi
  "${PY[@]}" "${args[@]}"
done

for checkpoint in "${H15}" "${A15}"; do
  test -s "${checkpoint}"
  "${PY[@]}" -c 'import sys, torch
p=sys.argv[1]
x=torch.load(p, map_location="cpu", weights_only=False)
required=("checkpoint_version","protocol_version","model_state","image_parameter_reference","optimizer_state","scheduler_state","scaler_state","python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","dataloader_generator_state","epoch","global_step","parent_scientific_config","resolved_operational_config","resolved_scientific_config","config_sha256","base_h2_commit","implementation_git_sha","working_tree_diff_sha256","git_sha","clip_sha256","dataset_manifest_sha256","seed","precision","amp_enabled","tf32_enabled")
missing=[k for k in required if k not in x]
cfg=x.get("resolved_scientific_config", {})
bad=[]
if int(x.get("checkpoint_version", 0)) != 3: bad.append(("checkpoint_version", x.get("checkpoint_version")))
if x.get("protocol_version") != "H2_CLEAN_REPRO_ANCHOR_CIR_V2_REDTEAM": bad.append(("protocol_version", x.get("protocol_version")))
if x.get("epoch") != 15: bad.append(("epoch", x.get("epoch")))
if x.get("seed") != int(sys.argv[2]): bad.append(("seed", x.get("seed")))
for k,v in (("training_horizon",15),("primary_horizon",15),("secondary_horizon",20)):
    if cfg.get(k) != v: bad.append((k, cfg.get(k)))
if x.get("parent_scientific_config") is None: bad.append(("parent_scientific_config", None))
if missing or bad: raise SystemExit(f"invalid E15 checkpoint: missing={missing} bad={bad}")
for name,state in (("model_state",x["model_state"]),("image_parameter_reference",x["image_parameter_reference"])):
    for key,value in state.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all(): raise SystemExit(f"nonfinite {name}:{key}")
' "${checkpoint}" "${SEED}"
done
date -Is > "${RUN_ROOT}/training_complete_at.txt"
echo "H/A E15 training completed: ${RUN_ROOT}"
