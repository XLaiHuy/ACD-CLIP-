#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/h2_clean_factorial_v1}"
CONDA_ENV="${CONDA_ENV:-torchhuy}"
RUN_FULL_TRAIN="${RUN_FULL_TRAIN:-NO}"
TRAINING_HORIZON=20
PRIMARY_HORIZON=15
SECONDARY_HORIZON=20
ANCHOR_LAMBDA_ACTIVE="0.0021633926715180626"
ACTIVATION_AUDIT="${ACTIVATION_AUDIT:-${ROOT}/audit/h2_anchor_family_short.json}"
PY=(conda run --no-capture-output -n "${CONDA_ENV}" python)
SHARED_E1="${RUN_ROOT}/shared_e1/adapter_1.pth"

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
  --num_workers 6
  --grad_checkpointing
  --amp
  --seed 0
  --anchor_grad_audit_interval 0
  --deterministic_algorithms
)

if [[ "${RUN_FULL_TRAIN}" != "YES" ]]; then
  echo "Prepared only. Set RUN_FULL_TRAIN=YES after every readiness gate passes."
  echo "Shared trajectory: native E1 -> full-state checkpoint -> H/A/C/AC from E2-E20."
  echo "Target-valid checkpoints: E15 primary, E20 secondary."
  echo "Run root: ${RUN_ROOT}"
  exit 0
fi

test -s "${ACTIVATION_AUDIT}"
"${PY[@]}" -c 'import json, sys; p=sys.argv[1]; x=json.load(open(p)); assert x.get("ACTIVATION_GATE") == "PASS", "activation gate did not pass: %r" % x.get("ACTIVATION_GATE"); assert x.get("ANCHOR_STATUS") == "FAMILY_SAFE_ACTIVE", "full training requires FAMILY_SAFE_ACTIVE, got %r" % x.get("ANCHOR_STATUS"); assert x.get("TRAINING_HORIZON") == 20, "activation training horizon mismatch: %r" % x.get("TRAINING_HORIZON"); assert x.get("PRIMARY_HORIZON") == 15, "activation primary horizon mismatch: %r" % x.get("PRIMARY_HORIZON"); assert x.get("SECONDARY_HORIZON") == 20, "activation secondary horizon mismatch: %r" % x.get("SECONDARY_HORIZON"); assert x.get("ANCHOR_R_MED") == 23.11184680352771, "activation R_MED mismatch: %r" % x.get("ANCHOR_R_MED"); assert x.get("ANCHOR_TARGET_EFFECTIVE_RATIO") == 0.05, "activation target ratio mismatch: %r" % x.get("ANCHOR_TARGET_EFFECTIVE_RATIO"); assert x.get("MEANINGFUL_TASK_ACTIVE_RATIO_MAX", 0.0) >= 0.02 - 1.0e-6, "Anchor signal is negligible: %r" % x.get("MEANINGFUL_TASK_ACTIVE_RATIO_MAX"); assert all(x.get("HARD_CHECKS", {}).get(k, False) for k in ("cap", "near_zero", "finite", "H_native", "A_only", "meaningful_active", "anchor_after_drift", "no_40000x_pathology")), "activation hard checks did not all pass"; assert x.get("ANCHOR_LAMBDA") == 0.0021633926715180626, "activation lambda mismatch: %r" % x.get("ANCHOR_LAMBDA"); assert x.get("ANCHOR_FAMILY_BUDGET_RHO") == 0.1, "activation rho mismatch: %r" % x.get("ANCHOR_FAMILY_BUDGET_RHO")' "${ACTIVATION_AUDIT}"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "Refusing to reuse existing factorial root: ${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${RUN_ROOT}"

"${PY[@]}" "${base_args[@]}" \
  --epoch 1 \
  --save_path "${RUN_ROOT}/shared_e1"
test -s "${SHARED_E1}"

"${PY[@]}" -c 'import sys, torch; p=sys.argv[1]; x=torch.load(p,map_location="cpu",weights_only=False); required=("checkpoint_version","protocol_version","model_state","image_parameter_reference","optimizer_state","scheduler_state","scaler_state","python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","dataloader_generator_state","epoch","global_step","parent_scientific_config","resolved_operational_config","resolved_scientific_config","config_sha256","base_h2_commit","implementation_git_sha","working_tree_diff_sha256","git_sha","clip_sha256","dataset_manifest_sha256","seed","precision","amp_enabled","tf32_enabled"); missing=[k for k in required if k not in x]; bad_version=int(x.get("checkpoint_version",0)) != 3 or x.get("protocol_version") != "H2_CLEAN_REPRO_ANCHOR_CIR_V2_REDTEAM"; sys.exit("incomplete or wrong-protocol shared E1 checkpoint: missing=%r version=%r protocol=%r" % (missing,x.get("checkpoint_version"),x.get("protocol_version"))) if missing or bad_version else None' "${SHARED_E1}"
"${PY[@]}" -c 'import sys, torch; p=sys.argv[1]; x=torch.load(p,map_location="cpu",weights_only=False); cfg=x["resolved_scientific_config"]; expected={"epoch":20,"training_horizon":20,"primary_horizon":15,"secondary_horizon":20}; bad={k:(cfg.get(k),v) for k,v in expected.items() if cfg.get(k) != v}; sys.exit("shared E1 protocol horizon mismatch: %r" % bad) if x.get("epoch") != 1 or bad else None' "${SHARED_E1}"

for arm in H A C AC; do
  args=(
    "${base_args[@]}"
    --epoch "${TRAINING_HORIZON}"
    --resume "${SHARED_E1}"
    --save_path "${RUN_ROOT}/${arm}"
  )
  case "${arm}" in
    A)
      args+=(--use_safe_anchor --anchor_lambda "${ANCHOR_LAMBDA_ACTIVE}" --anchor_reference_path "${SHARED_E1}" --anchor_gradient_budget --anchor_family_budget 0.10 --anchor_family_audit)
      ;;
    C)
      args+=(--use_cir_training --cir_alpha 0.5 --cir_peer_count 8 --cir_spatial_radius 3)
      ;;
    AC)
      args+=(
        --use_safe_anchor --anchor_lambda "${ANCHOR_LAMBDA_ACTIVE}" --anchor_reference_path "${SHARED_E1}"
        --anchor_gradient_budget --anchor_family_budget 0.10 --anchor_family_audit
        --use_cir_training --cir_alpha 0.5 --cir_peer_count 8 --cir_spatial_radius 3
      )
      ;;
  esac
  "${PY[@]}" "${args[@]}"
done
