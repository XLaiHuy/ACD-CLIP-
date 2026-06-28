#!/usr/bin/env bash
set -euo pipefail

# Quick diagnostic only. Do not use subset results for reporting.
# This compares current binned metrics vs phase2-style no-threshold stride-4
# metrics on the same label-balanced Retina subset.

EPOCH="${1:-19}"
SAMPLES_PER_LABEL="${SAMPLES_PER_LABEL:-128}"

echo "==== Retina diagnostic: binned thresholds=1000, stride=1 ===="
MAX_SAMPLES_PER_LABEL="${SAMPLES_PER_LABEL}" \
METRIC_THRESHOLDS=1000 \
PIXEL_STRIDE=1 \
DATASETS="Retina" \
bash test_6medical_selected_epochs.sh "${EPOCH}"

echo "==== Retina diagnostic: no threshold, stride=4 ===="
MAX_SAMPLES_PER_LABEL="${SAMPLES_PER_LABEL}" \
METRIC_THRESHOLDS=none \
PIXEL_STRIDE=4 \
DATASETS="Retina" \
bash test_6medical_selected_epochs.sh "${EPOCH}"

echo "==== Parsed log ===="
python parse_test_log.py --log test_train_main_base/test.log
