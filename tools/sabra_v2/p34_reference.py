"""Readable P34 reference algebra for deterministic parity tests."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tools.sabra_v2.p29_contract import CORRECTION_SCALE
from tools.sabra_v2.p32_reference import _effect_from_staged_region
from tools.sabra_v2.region_pool import REGION_GRID, STAGES


SMOOTH_L1_BETA = 1.0


def _validate_shapes(student_region: torch.Tensor, teacher_region: torch.Tensor) -> None:
    if student_region.ndim != 4 or student_region.shape[0] != STAGES or tuple(student_region.shape[-2:]) != REGION_GRID:
        raise ValueError("student_region must be [3,B,9,9]")
    if teacher_region.ndim != 3 or tuple(teacher_region.shape[-2:]) != REGION_GRID:
        raise ValueError("teacher_region must be [B,9,9]")
    if teacher_region.shape[0] != student_region.shape[1]:
        raise ValueError("teacher batch must match student batch")


def p34_actionability_components(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference loss through the full symmetric deployment algebra."""
    _validate_shapes(student_region, teacher_region)
    student_effect = _effect_from_staged_region(student_region)
    teacher_staged = teacher_region.detach().unsqueeze(0).expand_as(student_region)
    teacher_effect = _effect_from_staged_region(teacher_staged)
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
    return p34_actionability_components(student_region, teacher_region)[0]
