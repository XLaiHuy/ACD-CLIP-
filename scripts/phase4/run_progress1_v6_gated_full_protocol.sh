#!/usr/bin/env bash
set -euo pipefail

PROTOCOL_ROOT="${PROTOCOL_ROOT:-runs/phase4/progress1_v6_gated_full_seed0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-6}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PRECISION="${PRECISION:-bf16}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXACT_TEST_RERUN="${ALLOW_EXACT_TEST_RERUN:-0}"

export PROTOCOL_ROOT EPOCHS BATCH_SIZE GRAD_ACCUM NUM_WORKERS PRECISION SEED DRY_RUN ALLOW_EXACT_TEST_RERUN

conda run --no-capture-output -n torchhuy python tools/phase4_gated_protocol.py
