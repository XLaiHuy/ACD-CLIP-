#!/usr/bin/env bash
set -euo pipefail

# Phase 1 V1: pure dual-softmax attention DFG with tau=4.

DFG_ATTN_DIM="${DFG_ATTN_DIM:-256}" \
DFG_ATTN_TAU=4.0 \
SAVE_PATH="${SAVE_PATH:-phase1_v1_attn_tau4}" \
bash train_phase1a_attn_n3.sh
