"""Frozen P34 explicit actionability-target functional objective.

P34 changes one semantic operation from P33: the bounded, source-only
actionability signal shapes the detached target instead of weighting the loss.
Consequently a zero-actionability location has an explicit zero target and a
nonzero student effect receives a restoring gradient toward zero.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tools.sabra_v2.p29_contract import CORRECTION_SCALE
from tools.sabra_v2.p32_objective import deployed_margin_effect
from tools.sabra_v2.region_pool import REGION_GRID, STAGES


IMAGE_SIZE = 518
SMOOTH_L1_BETA = 1.0
P34_OBJECTIVE_NAME = "P34_EXPLICIT_ACTIONABILITY_TARGET_FUNCTIONAL_TRANSFER_V1"
P34_PREREGISTRATION_SHA256 = "b78f69487e665b62d9c81b58da45f8f0afe5d047e91996f18569c6d38f99abdb"


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


def p34_actionability_components(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return loss, effects, actionability, and the explicit shaped target."""
    _validate_shapes(student_region, teacher_region)
    student_effect = deployed_margin_effect(student_region.mean(dim=0))
    teacher_effect = deployed_margin_effect(teacher_region.detach())
    weight = (teacher_effect.abs() / CORRECTION_SCALE).clamp(0.0, 1.0).detach()
    target_effect = (weight * teacher_effect).detach()
    pointwise_error = F.smooth_l1_loss(
        student_effect,
        target_effect,
        beta=SMOOTH_L1_BETA,
        reduction="none",
    )
    loss = pointwise_error.mean()
    return loss, student_effect, teacher_effect, weight, target_effect


def p34_actionability_loss(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> torch.Tensor:
    """Compute the one and only P34 scientific objective."""
    return p34_actionability_components(student_region, teacher_region)[0]


def p34_objective_contract() -> dict[str, object]:
    """Expose immutable constants for provenance and engineering audits."""
    return {
        "name": P34_OBJECTIVE_NAME,
        "preregistration_sha256": P34_PREREGISTRATION_SHA256,
        "objective_count": 1,
        "student_shape": "[3,B,9,9]",
        "teacher_shape": "[B,9,9]",
        "effect_shape": "[B,518,518]",
        "weight_shape": "[B,518,518]",
        "target_shape": "[B,518,518]",
        "inherited_correction_scale_C": CORRECTION_SCALE,
        "weight_formula": "clamp(abs(detached_teacher_effect)/C,0,1)",
        "weight_detached": True,
        "target_formula": "detached(weight * teacher_effect)",
        "target_detached": True,
        "smooth_l1_beta": SMOOTH_L1_BETA,
        "gaussian_kernel": [7, 7],
        "gaussian_sigma": [1.0, 1.0],
        "align_corners": True,
        "teacher_detached": True,
        "student_self_normalized": False,
        "target_shrinkage": True,
        "explicit_zero_target": True,
        "hard_threshold": False,
        "sparsity_regularizer": False,
        "auxiliary_terms": [],
        "teacher_at_inference": False,
        "incremental_inference_overhead_percent": 0,
    }
