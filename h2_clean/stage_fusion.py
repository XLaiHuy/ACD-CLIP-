"""Frozen H2 stage-fusion weights for the bounded R2 screen.

The fusion operation is deliberately parameter-free.  Equal weights use the
historical ``torch.mean`` path exactly; the candidate applies its fixed convex
weights to pre-softmax stage logits.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch


H2_EQUAL_STAGE_FUSION_WEIGHTS = (
    1.0 / 3.0,
    1.0 / 3.0,
    1.0 / 3.0,
)
H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS = (
    0.3736138197153701,
    0.3270300383596602,
    0.2993561419249697,
)


def validate_stage_fusion_weights(weights: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in weights)
    if len(values) != 3:
        raise ValueError(f"H2 stage fusion requires exactly 3 weights, got {len(values)}")
    if not all(torch.isfinite(torch.tensor(value)) for value in values):
        raise ValueError("H2 stage fusion weights must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("H2 stage fusion weights must be non-negative")
    if abs(sum(values) - 1.0) > 1e-12:
        raise ValueError(f"H2 stage fusion weights must sum to 1, got {sum(values)}")
    return values


def fuse_stage_logits(
        stage_logits: torch.Tensor,
        weights: Sequence[float] = H2_EQUAL_STAGE_FUSION_WEIGHTS,
) -> torch.Tensor:
    """Fuse ``[stage, batch, class, height, width]`` logits before softmax."""
    if stage_logits.ndim != 5 or stage_logits.shape[0] != 3:
        raise ValueError("stage_logits must have shape [3, batch, class, height, width]")
    checked = validate_stage_fusion_weights(weights)
    # Preserve the historical equal-fusion operation bit-for-bit in the
    # control path; only the candidate takes the weighted sum branch.
    if checked == H2_EQUAL_STAGE_FUSION_WEIGHTS:
        return torch.mean(stage_logits, dim=0)
    weight_tensor = stage_logits.new_tensor(checked).view(3, 1, 1, 1, 1)
    return torch.sum(stage_logits * weight_tensor, dim=0)
