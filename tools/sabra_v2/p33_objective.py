"""Frozen P33 continuous actionability-weighted functional objective.

P33 keeps the P32 deployed functional-effect target but uses its detached
absolute magnitude as a bounded, training-only actionability weight.  The
weight changes where the target is learned; it does not shrink the signed
target itself and it is never needed at inference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tools.sabra_v2.p29_contract import CORRECTION_SCALE
from tools.sabra_v2.p32_objective import deployed_margin_effect
from tools.sabra_v2.region_pool import REGION_GRID, STAGES


IMAGE_SIZE = 518
SMOOTH_L1_BETA = 1.0
P33_OBJECTIVE_NAME = "P33_CONTINUOUS_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER_V1"
P33_PREREGISTRATION_SHA256 = "d2460555be14af7d23316e43ad16c8585faeecbedf1698ee71f29dce765aed6c"


def _validate_shapes(student_region: torch.Tensor, teacher_region: torch.Tensor) -> None:
    if (
        student_region.ndim != 4
        or student_region.shape[0] != STAGES
        or tuple(student_region.shape[-2:]) != REGION_GRID
    ):
        raise ValueError("student_region must be [3,B,9,9]")
    if teacher_region.ndim != 3 or tuple(teacher_region.shape[-2:]) != REGION_GRID:
        raise ValueError("teacher_region must be [B,9,9]")
    if teacher_region.shape[0] != student_region.shape[1]:
        raise ValueError("teacher batch must match student batch")


def p33_actionability_components(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return loss, student effect, teacher effect, and detached actionability."""
    _validate_shapes(student_region, teacher_region)
    student_effect = deployed_margin_effect(student_region.mean(dim=0))
    teacher_effect = deployed_margin_effect(teacher_region.detach())
    weight = (teacher_effect.abs() / CORRECTION_SCALE).clamp(0.0, 1.0).detach()
    pointwise_error = F.smooth_l1_loss(
        student_effect,
        teacher_effect,
        beta=SMOOTH_L1_BETA,
        reduction="none",
    )
    loss = (weight * pointwise_error).mean()
    return loss, student_effect, teacher_effect, weight


def p33_actionability_loss(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> torch.Tensor:
    """Compute the one and only P33 scientific objective."""
    return p33_actionability_components(student_region, teacher_region)[0]


def p33_objective_contract() -> dict[str, object]:
    """Expose immutable constants for provenance and engineering audits."""
    return {
        "name": P33_OBJECTIVE_NAME,
        "preregistration_sha256": P33_PREREGISTRATION_SHA256,
        "objective_count": 1,
        "student_shape": "[3,B,9,9]",
        "teacher_shape": "[B,9,9]",
        "effect_shape": "[B,518,518]",
        "weight_shape": "[B,518,518]",
        "inherited_correction_scale_C": CORRECTION_SCALE,
        "weight_formula": "clamp(abs(detached_teacher_effect)/C,0,1)",
        "weight_detached": True,
        "smooth_l1_beta": SMOOTH_L1_BETA,
        "gaussian_kernel": [7, 7],
        "gaussian_sigma": [1.0, 1.0],
        "align_corners": True,
        "teacher_detached": True,
        "student_self_normalized": False,
        "target_shrinkage": False,
        "hard_threshold": False,
        "auxiliary_terms": [],
        "teacher_at_inference": False,
        "incremental_inference_overhead_percent": 0,
    }
