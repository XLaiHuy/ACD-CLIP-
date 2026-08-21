"""Compatibility import surface for the canonical Phase2B runtime.

The implementation lives in ``model.phase2b_runtime`` so checkpoint
selection, SABRA calibration, and final testing cannot silently diverge.
"""
from __future__ import annotations

from model.phase2b_runtime import (
    IMAGE_SIZE,
    PATCH_COUNT,
    PATCH_GRID,
    PROJECTED_PATCH_DIM,
    STAGES,
    build_phase2b_frozen,
    deploy_native_logits,
    deploy_with_delta,
    forward_phase2b,
    load_phase2b_checkpoint,
)

__all__ = [
    "IMAGE_SIZE",
    "PATCH_COUNT",
    "PATCH_GRID",
    "PROJECTED_PATCH_DIM",
    "STAGES",
    "build_phase2b_frozen",
    "deploy_native_logits",
    "deploy_with_delta",
    "forward_phase2b",
    "load_phase2b_checkpoint",
]
