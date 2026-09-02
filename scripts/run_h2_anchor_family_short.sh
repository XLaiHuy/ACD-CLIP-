#!/usr/bin/env bash
set -euo pipefail

export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHORT_ROOT="${SHORT_ROOT:-/tmp/h2_anchor_family_short_e20_v1}"
CONDA_ENV="${CONDA_ENV:-torchhuy}"
RUN_SHORT="${RUN_SHORT:-NO}"
SHORT_BATCHES="${SHORT_BATCHES:-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRAINING_HORIZON=20
PRIMARY_HORIZON=15
SECONDARY_HORIZON=20
SHORT_END_EPOCH=5
ANCHOR_LAMBDA_ACTIVE="0.0021633926715180626"
PY=(conda run --no-capture-output -n "${CONDA_ENV}" python)
SHARED_E1="${SHORT_ROOT}/shared_e1/adapter_1.pth"

base_args=(
  "${ROOT}/train.py"
  --protocol_horizon "${TRAINING_HORIZON}"
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
  --num_workers "${NUM_WORKERS}"
  --deterministic_algorithms
  --grad_checkpointing
  --amp
  --seed 0
  --anchor_grad_audit_interval 0
  --anchor_family_audit
  --trace_batch_identity
)

if [[ "${RUN_SHORT}" != "YES" ]]; then
  echo "Prepared only. Set RUN_SHORT=YES to run native H_short and calibrated A_active_short E2-E5."
  echo "Scientific horizon: E20; target-valid checkpoints: E15 primary, E20 secondary."
  echo "Short root: ${SHORT_ROOT}"
  exit 0
fi

if [[ -e "${SHORT_ROOT}" ]]; then
  echo "Refusing to reuse existing short-run root: ${SHORT_ROOT}" >&2
  exit 2
fi
mkdir -p "${SHORT_ROOT}"

"${PY[@]}" "${base_args[@]}" \
  --epoch 1 \
  --max_batches "${SHORT_BATCHES}" \
  --save_path "${SHORT_ROOT}/shared_e1"

"${PY[@]}" -c 'import sys, torch; p=sys.argv[1]; x=torch.load(p,map_location="cpu",weights_only=False); required=("checkpoint_version","protocol_version","model_state","image_parameter_reference","optimizer_state","scheduler_state","scaler_state","python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","dataloader_generator_state","epoch","global_step","parent_scientific_config","resolved_operational_config","resolved_scientific_config","config_sha256","base_h2_commit","implementation_git_sha","working_tree_diff_sha256","git_sha","clip_sha256","dataset_manifest_sha256","seed","precision","amp_enabled","tf32_enabled"); missing=[k for k in required if k not in x]; bad=int(x.get("checkpoint_version",0)) != 3 or x.get("protocol_version") != "H2_CLEAN_REPRO_ANCHOR_CIR_V2_REDTEAM"; assert not (missing or bad), "incomplete/wrong-protocol E1: missing=%r version=%r protocol=%r" % (missing,x.get("checkpoint_version"),x.get("protocol_version"))' "${SHARED_E1}"

"${PY[@]}" "${base_args[@]}" \
  --epoch "${SHORT_END_EPOCH}" \
  --resume "${SHARED_E1}" \
  --max_batches "${SHORT_BATCHES}" \
  --save_path "${SHORT_ROOT}/H_short"

"${PY[@]}" "${base_args[@]}" \
  --epoch "${SHORT_END_EPOCH}" \
  --resume "${SHARED_E1}" \
  --max_batches "${SHORT_BATCHES}" \
  --save_path "${SHORT_ROOT}/A_active_short" \
  --use_safe_anchor \
  --anchor_lambda "${ANCHOR_LAMBDA_ACTIVE}" \
  --anchor_reference_path "${SHARED_E1}" \
  --anchor_gradient_budget \
  --anchor_family_budget 0.10

"${PY[@]}" "${ROOT}/scripts/validate_h2_anchor_family_short.py" \
  --root "${SHORT_ROOT}" \
  --expected-batches "${SHORT_BATCHES}" \
  --output "${ROOT}/audit/h2_anchor_family_short.json"

echo "H_short/A_active_short E2-E5 activation-gate validation completed: ${SHORT_ROOT}"
