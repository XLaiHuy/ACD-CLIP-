#!/usr/bin/env bash
set -euo pipefail

# Train the clean MLP + SS2D feature-residual ablation, then evaluate the
# likely-best checkpoints on all six medical datasets at image and pixel level.

SAVE_PATH="${SAVE_PATH:-phase1_mlp_ss2d_feature_g02}"
if [ "$#" -gt 0 ]; then
  EPOCHS_TO_TEST=("$@")
else
  EPOCHS_TO_TEST=(8 9 10 11 12 13 14)
fi

export SAVE_PATH
bash train_phase1_mlp_ss2d_feature_g02.sh

SAVE_PATH="${SAVE_PATH}" \
bash test_phase1_mlp_ss2d_feature_g02_selected_epochs.sh "${EPOCHS_TO_TEST[@]}"

echo "Completed. Review ${SAVE_PATH}/test.log for per-dataset image/pixel metrics."
echo "The paper-summary aggregate is printed above."
