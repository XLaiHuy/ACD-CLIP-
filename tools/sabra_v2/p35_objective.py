"""Frozen P35 soft actionability-weighted functional objective.

P35 preserves the complete signed deployed teacher effect as the target.  It
changes only the detached source-example importance map from P33's hard clamp
to a parameter-free monotonic tanh map.  The weight is training-only; it is
not part of inference and no target shaping is performed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tools.sabra_v2.p29_contract import CORRECTION_SCALE
from tools.sabra_v2.p32_objective import deployed_margin_effect
from tools.sabra_v2.region_pool import REGION_GRID, STAGES


IMAGE_SIZE = 518
SMOOTH_L1_BETA = 1.0
P35_OBJECTIVE_NAME = "P35_SOFT_TANH_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER_V1"
P35_PREREGISTRATION_SHA256 = "d92a8144e071412608292b4c48f5fe69381f82c3b205f6990266f2383336e3d8"


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


def p35_actionability_components(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return loss, effects, detached importance, and the full target."""
    _validate_shapes(student_region, teacher_region)
    student_effect = deployed_margin_effect(student_region.mean(dim=0))
    teacher_effect = deployed_margin_effect(teacher_region.detach())
    normalized_effect = teacher_effect.abs() / CORRECTION_SCALE
    weight = torch.tanh(normalized_effect).detach()
    target_effect = teacher_effect.detach()
    pointwise_error = F.smooth_l1_loss(
        student_effect,
        target_effect,
        beta=SMOOTH_L1_BETA,
        reduction="none",
    )
    loss = (weight * pointwise_error).mean()
    return loss, student_effect, teacher_effect, weight, target_effect


def p35_actionability_loss(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> torch.Tensor:
    """Compute P35's one and only objective."""
    return p35_actionability_components(student_region, teacher_region)[0]


def p35_objective_contract() -> dict[str, object]:
    """Expose the frozen mathematical and inference contract."""
    return {
        "name": P35_OBJECTIVE_NAME,
        "preregistration_sha256": P35_PREREGISTRATION_SHA256,
        "objective_count": 1,
        "student_shape": "[3,B,9,9]",
        "teacher_shape": "[B,9,9]",
        "effect_shape": "[B,518,518]",
        "weight_shape": "[B,518,518]",
        "target_shape": "[B,518,518]",
        "inherited_correction_scale_C": CORRECTION_SCALE,
        "normalized_effect_formula": "abs(detached_teacher_effect)/C",
        "weight_formula": "tanh(abs(detached_teacher_effect)/C)",
        "weight_detached": True,
        "target_formula": "detached_teacher_effect (full E_t; never multiplied by weight)",
        "target_detached": True,
        "target_shrinkage": False,
        "smooth_l1_beta": SMOOTH_L1_BETA,
        "gaussian_kernel": [7, 7],
        "gaussian_sigma": [1.0, 1.0],
        "align_corners": True,
        "teacher_detached": True,
        "student_self_normalized": False,
        "radial_identifiable": True,
        "hard_threshold": False,
        "sparsity_regularizer": False,
        "learned_gate": False,
        "auxiliary_terms": [],
        "new_tuned_hyperparameters": 0,
        "new_learnable_parameters": 0,
        "teacher_at_inference": False,
        "incremental_inference_overhead_percent": 0,
    }
