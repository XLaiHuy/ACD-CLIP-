#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ai4/caohuy/ACD-CLIP-medical-test"
HISTORICAL="/home/ai4/caohuy/ACD-CLIP-base-new-phase1"
RESULTS="${RESULTS:-$ROOT/medical_phase2cd_results}"
mkdir -p "$RESULTS"

run_one() {
  local state="$1" checkpoint="$2" sha256="$3"
  local output="$RESULTS/$state"
  if [[ -e "$output/complete" ]]; then
    printf 'Already complete: %s\n' "$state"
    return
  fi
  if [[ -e "$output" ]]; then
    printf 'Refusing to overwrite incomplete output: %s\n' "$output" >&2
    exit 1
  fi
  conda run --no-capture-output -n torchhuy python phase2cd_medical_eval.py \
    --state "$state" --checkpoint "$checkpoint" --expected-sha256 "$sha256" \
    --output-dir "$output" --batch-size 8 --num-workers 6 --pixel-stride 4 \
    2>&1 | tee "$RESULTS/${state}.evaluation.log"
}

cd "$ROOT"
run_one A_prime "$ROOT/checkpoints/phase2c/A_prime_seed42/A_prime_seed42_e13_pixelAUC94.8038_pixelAP55.5341_imageAP98.4225.pth" 036143f9ff940716684174e569ca07a8a060a9b81de94c14e8ba49d748783752
run_one B "$ROOT/checkpoints/phase2c/B_seed42/B_seed42_e13_pixelAUC96.2236_pixelAP55.1342_imageAP98.4287.pth" b556a2083555b1b9a2d29050b515808d191f224832613a203a90b74f5847cc2d
run_one C "$HISTORICAL/runs/phase2c_bf16/C_alpha020_delayed_seed42/checkpoints/adapter_14.pth" 8c9a463acdce6e617ddf2078cc7401c7e226dbddeb12415e14136364f749e69a
run_one P "$HISTORICAL/runs/phase2c_bf16/P_pcgrad_seed42/checkpoints/adapter_13.pth" 6e0b5d3ba987cfbe5b89ee51578a61e0ab3d5c737863cc54edef69de58c7e234
run_one PL "$ROOT/checkpoints/phase2c/PL_lora_only_seed42/PL_seed42_bs8_e15_pixelAUC96.6840_pixelAP52.7478_imageAP97.9956.pth" 1da52b88cd8009ad377e6c82377c7957cc1ac6df687596a800299bf54eab04f4
run_one LB_0p1 "$ROOT/checkpoints/phase2d/LB_0p1_seed42/LB_0p1_seed42_e15_pixelAUC97.4206_pixelAP53.4980_imageAP98.2721.pth" cf59bcbed5e00d60bfcbc9955ffb16928ccd887d97b03d53774eed237bce922d
run_one AB25 "$ROOT/checkpoints/phase2d/AB_interpolation_seed42/AB25_lambdaB0p25.pth" e9f6d3339b8d6766b2227ed718fa05c8a0effcc407343dffbe8a5557ea7cbff7
run_one AB50 "$ROOT/checkpoints/phase2d/AB_interpolation_seed42/AB50_lambdaB0p50.pth" 610b8e9a89d339dfc7893a006cf9a848d99e218a4abfef3da8b6576a2442d824
run_one AB75 "$ROOT/checkpoints/phase2d/AB_interpolation_seed42/AB75_lambdaB0p75.pth" 606f312654c70317db10bb2d5d137643d53ffa409cbed065e0eb87b7427d37ce
