#!/usr/bin/env bash
set -euo pipefail
# The one authorized full precision/seed disambiguation trajectory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/workspace/venv-acdclip/bin/python}"
RUN_ROOT="${RUN_ROOT:-/workspace/h2_bf16_screening/fresh_seed0_a_e15}"
[[ ! -e "$RUN_ROOT" ]] || { echo "refusing existing $RUN_ROOT" >&2; exit 2; }
git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
base=("$ROOT/train.py" --protocol_horizon 15 --dataset VisA --model_name ViT-L-14-336 --img_size 518 --n_groups 3 --image_adapt_weight .2 --text_adapt_weight .2 --conv_lora_rank 8 --conv_lora_alpha 2 --conv_kernel_size_list 3 5 --lora_rank 16 --lora_alpha 2 --image_lr .001 --text_lr .0005 --use_hybrid_soft_prompt --hybrid_alpha_max .2 --soft_prompt_freeze_epochs 3 --soft_prompt_ctx_len 4 --soft_prompt_lr .00005 --soft_prompt_init phrase --soft_prompt_init_phrase 'a photo of a' --lambda_kg .01 --lambda_k .002 --lr_gamma .9 --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8 --use_ss2d_dfg --dfg_gamma_max .2 --dfg_ss2d_fusion weight_residual --dfg_beta .10 --dfg_beta_schedule warmup010 --dfg_beta_target .10 --grad_clip_norm 1 --batch_size 6 --num_workers 6 --pin_memory --no-persistent_workers --prefetch_factor 2 --no-non_blocking_copy --grad_checkpointing --precision bf16 --no-bf16_local_fp32_islands --seed 0 --deterministic_algorithms --non_finite_loss_abort_threshold 0 --abort_on_nonfinite --telemetry_interval 25 --family_telemetry_interval 25)
mkdir -p "$RUN_ROOT"
"$PYTHON" "${base[@]}" --epoch 1 --save_path "$RUN_ROOT/shared_e1"
shared="$RUN_ROOT/shared_e1/adapter_1.pth"
"$PYTHON" "${base[@]}" --epoch 15 --resume "$shared" --save_path "$RUN_ROOT/A" --use_safe_anchor --anchor_lambda .0021633926715180626 --anchor_reference_path "$shared" --anchor_gradient_budget --anchor_family_budget .10 --anchor_family_audit
"$PYTHON" - "$RUN_ROOT/A/adapter_15.pth" <<'PY'
import sys,torch
x=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
assert x['epoch']==15 and x['global_step']==5415 and x['precision']=='bf16'
assert x['gradscaler_enabled'] is False and x['scaler_state']=={}
assert x['resolved_scientific_config']['seed']==0
for tree in (x['model_state'],x['optimizer_state']):
 def walk(v):
  if torch.is_tensor(v): assert torch.isfinite(v).all()
  elif isinstance(v,dict):
   for q in v.values(): walk(q)
  elif isinstance(v,(tuple,list)):
   for q in v: walk(q)
 walk(tree)
print('A_BF16_SEED0_E15=PASS')
PY
