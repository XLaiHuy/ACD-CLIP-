#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="runs/phase2d_lb_0p1_seed42"
VISA_ROOT="data/VisA_20220922"
TRAIN_MANIFEST="splits/visa_train_seed42.csv"
VAL_MANIFEST="splits/visa_val_seed42.csv"
SPLIT_METADATA="splits/visa_split_seed42_metadata.json"
PRETRAINED="model/ViT-L-14-336px.pt"

check_assets() {
  test -d "$VISA_ROOT" || { echo "Missing VisA root: $VISA_ROOT" >&2; exit 1; }
  test -f "$TRAIN_MANIFEST" || { echo "Missing train manifest: $TRAIN_MANIFEST" >&2; exit 1; }
  test -f "$VAL_MANIFEST" || { echo "Missing val manifest: $VAL_MANIFEST" >&2; exit 1; }
  test -f "$SPLIT_METADATA" || { echo "Missing split metadata: $SPLIT_METADATA" >&2; exit 1; }
  test -f "$PRETRAINED" || { echo "Missing pretrained model: $PRETRAINED" >&2; exit 1; }
}

mkdir -p "$RUN_DIR"
check_assets

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  "$PYTHON_BIN" phase2c_train.py \
    --condition A_prime \
    --save-path "$RUN_DIR" \
    --hybrid-alpha-max 0.20 \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --split-metadata "$SPLIT_METADATA" \
    --batch-size 6 \
    --num-workers 6 \
    --diagnostic-batch-size 1 \
    --cls-loss-weight 0.1 \
    --seg-loss-weight 1.0 \
    --bf16 \
    --dry-run > "$RUN_DIR/config.json"
  cp "$SPLIT_METADATA" "$RUN_DIR/split_metadata.json"
  "$PYTHON_BIN" - "$RUN_DIR/config.json" "$RUN_DIR/protocol_diff.json" <<'PY'
import json
import sys
from pathlib import Path

current_path, diff_path = map(Path, sys.argv[1:])
current = json.loads(current_path.read_text())
historical = json.loads(Path("runs/phase2c_bf16/A_alpha020_seed42/config.json").read_text())
historical["cls_loss_weight"] = 1.0
historical["seg_loss_weight"] = 1.0
historical["activation_delay_epochs"] = 0
historical["pcgrad_enabled"] = False
historical["pcgrad_groups"] = []
historical["pcgrad_scale_factor"] = 1.0
ignored = {"save_path"}
diff = {"intended_change": "cls_loss_weight", "historical": 1.0, "proposed": 0.1, "fields": {}, "medical_evaluation": False}
for key in sorted(set(current) | set(historical)):
    if key in ignored:
        continue
    left = historical.get(key)
    right = current.get(key)
    identical = left == right
    intended = key == "cls_loss_weight" and left == 1.0 and right == 0.1
    diff["fields"][key] = {"historical": left, "proposed": right, "identical": identical, "intended_difference": intended}
    if not identical and not intended:
        raise SystemExit(f"unintended protocol difference: {key}: {left!r} -> {right!r}")
diff_path.write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n")
PY
  echo "LB_0p1 preflight passed"
  exit 0
fi

exec "$PYTHON_BIN" phase2c_train.py \
  --condition A_prime \
  --save-path "$RUN_DIR" \
  --hybrid-alpha-max 0.20 \
  --train-manifest "$TRAIN_MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --split-metadata "$SPLIT_METADATA" \
  --batch-size 6 \
  --num-workers 6 \
  --diagnostic-batch-size 1 \
  --cls-loss-weight 0.1 \
  --seg-loss-weight 1.0 \
  --bf16
