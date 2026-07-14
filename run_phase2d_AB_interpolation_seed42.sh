#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="runs/phase2d_ab_interpolation_seed42"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
A_PATH="checkpoints/phase2c/A_prime_seed42/A_prime_seed42_e13_pixelAUC94.8038_pixelAP55.5341_imageAP98.4225.pth"
B_PATH="checkpoints/phase2c/B_seed42/B_seed42_e13_pixelAUC96.2236_pixelAP55.1342_imageAP98.4287.pth"
A_SHA="036143f9ff940716684174e569ca07a8a060a9b81de94c14e8ba49d748783752"
B_SHA="b556a2083555b1b9a2d29050b515808d191f224832613a203a90b74f5847cc2d"
BATCH_SIZE=6
NUM_WORKERS=6
CUDA_DEVICE="${CUDA_DEVICE:-0}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  "$PYTHON_BIN" phase2d_interpolation.py --a-checkpoint "$A_PATH" --b-checkpoint "$B_PATH" --a-sha256 "$A_SHA" --b-sha256 "$B_SHA" --output-dir "$CHECKPOINT_DIR" --dry-run
  "$PYTHON_BIN" phase2d_evaluate.py --checkpoint "A_prime=$A_PATH" --checkpoint "B=$B_PATH" --output-csv "$RUN_DIR/dry_run.csv" --output-json "$RUN_DIR/dry_run.json" --dry-run
  exit 0
fi

mkdir -p "$RUN_DIR"
"$PYTHON_BIN" phase2d_results.py --run-dir "$RUN_DIR" --write-config --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --a-path "$A_PATH" --a-sha "$A_SHA" --b-path "$B_PATH" --b-sha "$B_SHA"
printf "Phase2D AB interpolation: zero-training run started\n" > "$RUN_DIR/run.log"
"$PYTHON_BIN" phase2d_interpolation.py --a-checkpoint "$A_PATH" --b-checkpoint "$B_PATH" --a-sha256 "$A_SHA" --b-sha256 "$B_SHA" --output-dir "$CHECKPOINT_DIR"
"$PYTHON_BIN" phase2d_evaluate.py --checkpoint "A_prime=$A_PATH" --checkpoint "B=$B_PATH" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --cuda-device "$CUDA_DEVICE" --output-csv "$RUN_DIR/parent_reproduction.csv" --output-json "$RUN_DIR/parent_reproduction.json"
"$PYTHON_BIN" - "$RUN_DIR/parent_reproduction.csv" <<'PY'
import csv
import sys

expected = {
    "A_prime": {"pixel_auc": 94.8038, "pixel_ap": 55.5341, "image_auc": 97.9028, "image_ap": 98.4225},
    "B": {"pixel_auc": 96.2236, "pixel_ap": 55.1342, "image_auc": 97.8750, "image_ap": 98.4287},
}
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = [row for row in csv.DictReader(handle) if row["scope"] == "macro"]
if {row["checkpoint_name"] for row in rows} != set(expected):
    raise SystemExit("parent reproduction macro rows are incomplete")
for row in rows:
    for metric, target in expected[row["checkpoint_name"]].items():
        if abs(float(row[metric]) - target) > 0.05:
            raise SystemExit(f"parent reproduction failed: {row['checkpoint_name']} {metric}={row[metric]} target={target}")
PY
"$PYTHON_BIN" phase2d_results.py --run-dir "$RUN_DIR" --parent-gate
printf "Parent reproduction gate passed\n" >> "$RUN_DIR/run.log"
"$PYTHON_BIN" phase2d_evaluate.py --checkpoint "AB25=$CHECKPOINT_DIR/AB25_lambdaB0p25.pth" --checkpoint "AB50=$CHECKPOINT_DIR/AB50_lambdaB0p50.pth" --checkpoint "AB75=$CHECKPOINT_DIR/AB75_lambdaB0p75.pth" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --cuda-device "$CUDA_DEVICE" --output-csv "$RUN_DIR/candidate_metrics.csv" --output-json "$RUN_DIR/candidate_metrics.json"
"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import csv
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
rows = []
for name in ("parent_reproduction.csv", "candidate_metrics.csv"):
    with (run_dir / name).open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))
with (run_dir / "visa_val_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
PY
"$PYTHON_BIN" phase2d_results.py --run-dir "$RUN_DIR" --select --results-markdown PHASE2D_AB_RESULTS.md
printf "Candidate evaluation and preregistered selection completed\n" >> "$RUN_DIR/run.log"
