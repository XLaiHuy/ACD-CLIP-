#!/usr/bin/env bash
set -euo pipefail

# Force running from script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Configuration
CONDA_ENV_NAME="torchhuy"
export PATH="/home/ai4/miniconda3/bin:$PATH"

GIT_SHA="$(git rev-parse HEAD)"
echo "==== Workflow Started ===="
echo "Evaluating Git Commit: ${GIT_SHA}"
echo "Current Directory: $(pwd)"

# Directories
SAVE_PATH_P1="runs/phase1/05_phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3_best"
OUTPUT_DIR_P1="${SAVE_PATH_P1}/cls_only_rescore_e9"

SAVE_PATH_P2B="runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch"
OUTPUT_DIR_P2B="${SAVE_PATH_P2B}/cls_only_rescore_e10"

ENSEMBLE_OUT="${SAVE_PATH_P2B}/ensemble_e9_e10"

# Clean old outputs safely to prevent mixing up data
echo "==== Cleaning old output directories ===="
rm -rf "${OUTPUT_DIR_P1}"
rm -rf "${OUTPUT_DIR_P2B}"
rm -rf "${ENSEMBLE_OUT}"

mkdir -p "${OUTPUT_DIR_P1}" "${OUTPUT_DIR_P2B}" "${ENSEMBLE_OUT}"

# Record Git Metadata
GIT_STATUS="$(git status --porcelain)"
{
  echo "git_sha=${GIT_SHA}"
  if [ -n "${GIT_STATUS}" ]; then
    echo "working_tree=dirty"
    echo "${GIT_STATUS}"
  else
    echo "working_tree=clean"
  fi
} > "${ENSEMBLE_OUT}/evaluation_metadata.txt"

echo "==== Step 1: Rescore Phase1 e9 with cls_only ===="
conda run --no-capture-output -n ${CONDA_ENV_NAME} python phase2b_anchor_diagnosis.py \
  --mode sweep \
  --save_path "${SAVE_PATH_P1}" \
  --output_dir "${OUTPUT_DIR_P1}" \
  --epochs 9 \
  --fixed_prompt_config phase1_hard \
  --fixed_score_rule cls_only \
  --batch_size 8 \
  --num_workers 6 \
  --pixel_stride 4 \
  --dfg_mode attn \
  --dfg_attn_dim 256 \
  --dfg_attn_tau 8.0 \
  --use_ss2d_dfg \
  --dfg_gamma_max 0.2 \
  --dfg_ss2d_fusion weight_residual \
  --dfg_beta 0.10 \
  --dfg_beta_schedule warmup010 \
  --dfg_beta_target 0.10

# Verify file exists and is non-empty
if [ ! -s "${OUTPUT_DIR_P1}/image_score_raw_predictions.csv" ]; then
  echo "Error: Phase 1 rescore did not output predictions or the output file is empty."
  exit 1
fi
if [ ! -s "${OUTPUT_DIR_P1}/fixed_config_epoch_sweep.csv" ]; then
  echo "Error: Phase 1 rescore summary file is missing or empty."
  exit 1
fi
echo "Phase 1 rescore outputs verified."


echo "==== Step 2: Rescore Phase2B e10 with cls_only ===="
conda run --no-capture-output -n ${CONDA_ENV_NAME} python phase2b_anchor_diagnosis.py \
  --mode sweep \
  --save_path "${SAVE_PATH_P2B}" \
  --output_dir "${OUTPUT_DIR_P2B}" \
  --epochs 10 \
  --fixed_prompt_config current_shared \
  --fixed_score_rule cls_only \
  --batch_size 8 \
  --num_workers 6 \
  --pixel_stride 4 \
  --dfg_mode attn \
  --dfg_attn_dim 256 \
  --dfg_attn_tau 8.0 \
  --use_ss2d_dfg \
  --dfg_gamma_max 0.2 \
  --dfg_ss2d_fusion weight_residual \
  --dfg_beta 0.10 \
  --dfg_beta_schedule warmup010 \
  --dfg_beta_target 0.10

# Verify file exists and is non-empty
if [ ! -s "${OUTPUT_DIR_P2B}/image_score_raw_predictions.csv" ]; then
  echo "Error: Phase 2B rescore did not output predictions or the output file is empty."
  exit 1
fi
if [ ! -s "${OUTPUT_DIR_P2B}/fixed_config_epoch_sweep.csv" ]; then
  echo "Error: Phase 2B rescore summary file is missing or empty."
  exit 1
fi
echo "Phase 2B rescore outputs verified."


echo "==== Step 3: Run Probability Ensemble with Sanity checks ===="
conda run --no-capture-output -n ${CONDA_ENV_NAME} python phase1_phase2b_probability_ensemble.py \
  --phase1_raw "${OUTPUT_DIR_P1}/image_score_raw_predictions.csv" \
  --phase2b_raw "${OUTPUT_DIR_P2B}/image_score_raw_predictions.csv" \
  --phase1_summary "${OUTPUT_DIR_P1}/fixed_config_epoch_sweep.csv" \
  --phase2b_summary "${OUTPUT_DIR_P2B}/fixed_config_epoch_sweep.csv" \
  --output_dir "${ENSEMBLE_OUT}" \
  --betas 0.0 0.25 0.5 0.75 1.0

# Verify ensemble output
if [ ! -s "${ENSEMBLE_OUT}/probability_ensemble_summary.csv" ]; then
  echo "Error: Ensemble did not produce summary or the output file is empty."
  exit 1
fi

echo "==== Step 4: Final Summary ===="
echo ""
echo "--- Phase 1 e9 cls_only results ---"
cat "${OUTPUT_DIR_P1}/fixed_config_epoch_sweep.csv"
echo ""
echo "--- Phase 2B e10 cls_only results ---"
cat "${OUTPUT_DIR_P2B}/fixed_config_epoch_sweep.csv"
echo ""
echo "--- Ensemble metrics ---"
cat "${ENSEMBLE_OUT}/probability_ensemble_summary.csv"
echo ""
echo "==== Workflow Completed successfully! ===="
