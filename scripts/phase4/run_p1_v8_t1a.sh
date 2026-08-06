#!/usr/bin/env bash
set -euo pipefail

ROOT="runs/phase4/p1_v8_specialization_overnight/${P1_V8_CANDIDATE_NAME:-T1-A_delta001}"
DELTA_DIV_WEIGHT="${P1_V8_DELTA_DIV_WEIGHT:-0.001}"
FUNC_DIV_WEIGHT="${P1_V8_FUNC_DIV_WEIGHT:-0.0}"
PHASE="${1:-wiring}"
RUN_TAG="${2:-}"
mkdir -p "${ROOT}"

if [[ "${PHASE}" == "wiring" ]]; then
  SAVE_PATH="${ROOT}/${RUN_TAG:-wiring_50}"
  EPOCHS=1
  MAX_BATCHES=50
elif [[ "${PHASE}" == "smoke" ]]; then
  SAVE_PATH="${ROOT}/${RUN_TAG:-smoke_3x300}"
  EPOCHS=3
  MAX_BATCHES=300
else
  echo "usage: $0 [wiring|smoke] [unique-run-tag]" >&2
  exit 2
fi

CMD=(
  conda run --no-capture-output -n torchhuy python train.py
  --save_path "${SAVE_PATH}"
  --dataset Brain --img_size 518 --cuda_device 0
  --epoch "${EPOCHS}" --batch_size 2 --num_workers 4 --precision bf16
  --h6_smoke_max_batches "${MAX_BATCHES}"
  --h6_wiring_probe_batches 1 10 25 50
  --h6_progress 1 --h6_progress_version P1-v8-minimal
  --h6_global_text_mode hard_anchor --h6_prediction_routing dense
  --h6_router_soft_epochs 1000 --no-h6_expert_enabled
  --h6_diagnostics_mode light --h6_diagnostics_interval 1
  --h6_num_factors 4 --h6_top_k 2 --h6_bank_dim 256 --h6_router_dim 128
  --h6_factor_generator_specialization_enabled
  --h6_factor_head_init_scale 0.001
  --h6_factor_local_dynamic_mix 0.05
  --h6_factor_id_scale 0.02 --h6_factor_id_max_ratio 0.05
  --h6_factor_grad_diagnostics
  --lambda_h6_orth 0.0 --lambda_h6_delta_div "${DELTA_DIV_WEIGHT}" --lambda_h6_func_div "${FUNC_DIV_WEIGHT}"
)

printf '%q ' "${CMD[@]}" > "${SAVE_PATH}.command.txt"
printf '\n' >> "${SAVE_PATH}.command.txt"
"${CMD[@]}" 2>&1 | tee "${SAVE_PATH}.run.log"
