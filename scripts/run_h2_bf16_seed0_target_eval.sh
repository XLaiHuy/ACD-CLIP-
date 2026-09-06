#!/usr/bin/env bash
set -euo pipefail
# Fixed post-freeze target evaluation for the single BF16 Seed0 A E15 checkpoint.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/workspace/venv-acdclip/bin/python}"
RUN_ROOT="${RUN_ROOT:-/workspace/h2_bf16_screening/fresh_seed0_a_e15}"
MEDICAL_ROOT="${MEDICAL_ROOT:-/workspace/ACD-CLIP-medical-test}"
EVAL_ROOT="${EVAL_ROOT:-/workspace/h2_bf16_screening/target_eval/seed0_bf16_a_e15}"
CHECKPOINT="$RUN_ROOT/A/adapter_15.pth"
MEDICAL_OUT="$EVAL_ROOT/medical/A_E15_BF16_Seed0"
MVTEC_OUT="$EVAL_ROOT/mvtec/A_E15_BF16_Seed0"

"$PYTHON" "$ROOT/scripts/freeze_h2_bf16_seed0_a_e15.py" --run-root "$RUN_ROOT"
test -s "$CHECKPOINT"
test -d "$MEDICAL_ROOT"
mkdir -p "$EVAL_ROOT"
SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [[ ! -e "$MEDICAL_OUT/complete" ]]; then
  [[ ! -e "$MEDICAL_OUT" ]] || { echo "Refusing incomplete Medical output: $MEDICAL_OUT" >&2; exit 2; }
  (
    cd "$MEDICAL_ROOT"
    "$PYTHON" phase2cd_medical_eval.py --state A_E15_BF16_Seed0 --checkpoint "$CHECKPOINT" \
      --expected-sha256 "$SHA256" --output-dir "$MEDICAL_OUT" --batch-size 8 --num-workers 6 \
      --cuda-device 0 --pixel-stride 1
  ) 2>&1 | tee "$EVAL_ROOT/medical.evaluation.log"
fi
if [[ ! -e "$MVTEC_OUT/test.log" ]]; then
  [[ ! -e "$MVTEC_OUT" ]] || { echo "Refusing incomplete MVTec output: $MVTEC_OUT" >&2; exit 2; }
  mkdir -p "$MVTEC_OUT"
  ln -s "$CHECKPOINT" "$MVTEC_OUT/adapter_15.pth"
  "$PYTHON" "$ROOT/test.py" --dataset MVTec --model_name ViT-L-14-336 --img_size 518 --n_groups 3 \
    --lora_rank 16 --lora_alpha 2.0 --conv_lora_rank 8 --conv_lora_alpha 2.0 --conv_kernel_size_list 3 5 \
    --dfg_mode attn --dfg_attn_dim 256 --dfg_attn_tau 8.0 --use_ss2d_dfg --dfg_gamma_max 0.2 \
    --dfg_ss2d_fusion weight_residual --dfg_beta 0.10 --dfg_beta_schedule warmup010 --dfg_beta_target 0.10 \
    --batch_size 8 --num_workers 6 --save_path "$MVTEC_OUT" --epochs 15 --evaluator_mode benchmark_exact --pixel_stride 1
fi
