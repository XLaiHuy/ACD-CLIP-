#!/usr/bin/env bash
set -euo pipefail
exec bash run_phase2c_ab_common.sh \
  B runs/phase2c/B_alpha015_seed42 0.15
