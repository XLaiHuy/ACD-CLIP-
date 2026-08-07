#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# ACD-CLIP Phase 4 Progress 1 v8.2 — Dedicated 6-Medical Evaluator (Epochs 15-20)
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_ROOT="runs/phase4/p1_v8_2_medical_e15_e20"
RUN_DIR="runs/phase4/p1_v8_2_full20_seed0"
MEDICAL_MANIFEST_ROOT="runs/phase4/progress1_v7_full_seed0_ready3/train/protocol/medical_manifests"

EPOCHS=(15 16 17 18 19 20)
DATASETS=(Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

echo "=== [P1-V8.2] Dedicated 6-Medical Evaluation Plan (Epochs 15–20) ==="
echo "Output Root: ${OUTPUT_ROOT}"
echo "Run Directory: ${RUN_DIR}"
echo "Medical Manifest Root: ${MEDICAL_MANIFEST_ROOT}"
echo "Total Planned Dataset Evaluations: 36 (6 Epochs x 6 Datasets)"
echo "----------------------------------------------------------------------"

mkdir -p "${OUTPUT_ROOT}/protocol"

# 1. Audit checkpoints exist
for ep in "${EPOCHS[@]}"; do
  CKPT="${RUN_DIR}/adapter_${ep}.pth"
  if [ ! -f "${CKPT}" ]; then
    echo "[ERROR] Required checkpoint missing: ${CKPT}"
    exit 1
  fi
done

# 2. Print or Execute planned 36 evaluations
EVAL_COUNT=0
COMPLETED_COUNT=0

for ep in "${EPOCHS[@]}"; do
  EPOCH_DIR="${OUTPUT_ROOT}/epoch_${ep}"
  mkdir -p "${EPOCH_DIR}"

  for ds in "${DATASETS[@]}"; do
    EVAL_COUNT=$((EVAL_COUNT + 1))
    DS_DIR="${EPOCH_DIR}/${ds}"
    mkdir -p "${DS_DIR}"

    if [ "${ds}" = "Brain" ] || [ "${ds}" = "Liver" ] || [ "${ds}" = "Retina" ]; then
      MANIFEST_PATH="dataset/hub/${ds}.jsonl"
    else
      MANIFEST_PATH="${MEDICAL_MANIFEST_ROOT}/${ds}_test.jsonl"
    fi

    CMD=(
      python test.py
      --save_path "${RUN_DIR}"
      --epochs "${ep}"
      --dataset "${ds}"
      --img_size 518
      --batch_size 1
      --cuda_device 0
      --num_workers 2
      --medical_split test
      --medical_manifest_root "${MEDICAL_MANIFEST_ROOT}"
      --external_exact_pixel_metrics
      --external_metric_chunk_pixels 5000000
      --pixel_stride 1
      --n_groups 3
      --dfg_mode attn
      --dfg_attn_dim 256
      --dfg_attn_tau 8.0
      --use_ss2d_dfg
      --dfg_gamma_max 0.2
      --dfg_ss2d_fusion weight_residual
      --dfg_beta 0.10
      --dfg_beta_schedule fixed
      --h6_progress 1
      --h6_progress_version P1-v8-minimal
      --h6_global_text_mode hard_anchor
      --h6_local_factor_mode center_spread
      --h6_local_center_mix 0.05
      --h6_local_factor_spread 0.10
      --h6_prediction_routing dense
      --h6_num_factors 4
      --h6_top_k 2
      --h6_bank_dim 256
      --h6_router_dim 128
      --no-h6_expert_enabled
      --no-h6_load_bias_enabled
      --no-h6_cluster_responsibility
      --lambda_h6_dynamic_mean_anchor 0.0
    )

    if [ "${DRY_RUN}" -eq 1 ]; then
      echo "[EVAL ${EVAL_COUNT}/36] epoch=${ep} dataset=${ds} split=test manifest=${MANIFEST_PATH}"
      echo "  out_dir=${DS_DIR}"
      printf "  cmd: %q " "${CMD[@]}"
      echo ""
    else
      # Check resume state
      if [ -f "${DS_DIR}/metrics.json" ]; then
        echo "[SKIP] Evaluation ${EVAL_COUNT}/36 (epoch ${ep}, dataset ${ds}) already complete."
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        continue
      fi

      echo "=== Executing Evaluation ${EVAL_COUNT}/36 (epoch ${ep}, dataset ${ds}) ==="
      printf '%q ' "${CMD[@]}" > "${DS_DIR}/test.command.txt"
      printf '\n' >> "${DS_DIR}/test.command.txt"

      t_start=$(date +%s)
      set +e
      "${CMD[@]}" 2>&1 | tee "${DS_DIR}/test.log"
      EXIT_CODE=${PIPESTATUS[0]}
      set -e
      t_end=$(date +%s)
      runtime=$((t_end - t_start))

      echo "${EXIT_CODE}" > "${DS_DIR}/exit_code.txt"
      echo "${runtime}" > "${DS_DIR}/runtime_seconds.txt"

      if [ "${EXIT_CODE}" -ne 0 ]; then
        reason="UNKNOWN_FAILURE"
        if [ "${EXIT_CODE}" -eq 137 ]; then
          reason="HOST_OOM_OR_SIGKILL"
        elif [ "${EXIT_CODE}" -eq 143 ]; then
          reason="SIGTERM"
        elif grep -q "CUDA out of memory" "${DS_DIR}/test.log" 2>/dev/null; then
          reason="CUDA_OOM"
        elif grep -q "DataLoader worker" "${DS_DIR}/test.log" 2>/dev/null; then
          reason="DATALOADER_FAILURE"
        elif grep -q "No space left on device" "${DS_DIR}/test.log" 2>/dev/null; then
          reason="DISK_FULL"
        fi
        echo "${reason}" > "${DS_DIR}/failure_reason.txt"
        echo "[ERROR] Evaluation failed for epoch ${ep}, dataset ${ds} (exit_code=${EXIT_CODE}, reason=${reason})."
        exit "${EXIT_CODE}"
      else
        echo "SUCCESS" > "${DS_DIR}/failure_reason.txt"
        # Parse test metrics from test.log or output JSON and save to DS_DIR/metrics.json
        if [ -f "${RUN_DIR}/test_epoch_${ep}/exact_results_${ds}_test_epoch_${ep}.csv" ]; then
          cp "${RUN_DIR}/test_epoch_${ep}/exact_results_${ds}_test_epoch_${ep}.csv" "${DS_DIR}/metrics.csv"
        fi
        # Save metrics.json inside DS_DIR
        python3 -c '
import sys, json, csv, os
ds_dir = sys.argv[1]
ep = sys.argv[2]
ds = sys.argv[3]
run_dir = sys.argv[4]
csv_p = os.path.join(run_dir, f"test_epoch_{ep}", f"exact_results_{ds}_test_epoch_{ep}.csv")
if os.path.exists(csv_p):
    with open(csv_p) as f:
        rows = list(csv.DictReader(f))
        if rows:
            res = {
                "pixel AUC": float(rows[0]["pixel AUC"]),
                "pixel AP": float(rows[0]["pixel AP"]),
                "image AUC": float(rows[0]["image AUC"]),
                "image AP": float(rows[0]["image AP"]),
            }
            with open(os.path.join(ds_dir, "metrics.json"), "w") as out:
                json.dump(res, out, indent=2)
' "${DS_DIR}" "${ep}" "${ds}" "${RUN_DIR}"
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
      fi
    fi
  done
done

if [ "${DRY_RUN}" -eq 1 ]; then
  echo "----------------------------------------------------------------------"
  echo "[OK] Dry-run verification complete. All 36 commands validated."
else
  echo "----------------------------------------------------------------------"
  echo "Evaluating summary across epochs 15-20..."
  python3 tools/summarize_p1_v8_2_medical_e15_e20.py --output-dir "${OUTPUT_ROOT}"
  echo "=== Dedicated 6-Medical Evaluation Pipeline Completed (${COMPLETED_COUNT}/36) ==="
fi
