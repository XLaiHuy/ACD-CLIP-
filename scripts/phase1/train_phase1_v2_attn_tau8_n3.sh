#!/usr/bin/env bash
set -euo pipefail

# Phase 1 V2: pure dual-softmax attention DFG with tau=8.
# Only tau changes from V1. This tests whether V1 attention was too sharp.

DFG_ATTN_DIM="${DFG_ATTN_DIM:-256}" \
DFG_ATTN_TAU=8.0 \
SAVE_PATH="${SAVE_PATH:-phase1_v2_attn_tau8}" \
bash train_phase1a_attn_n3.sh
