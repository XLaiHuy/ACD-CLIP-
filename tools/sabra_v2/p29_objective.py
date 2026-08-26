"""Frozen P29 sign-guarded normalized source-only distillation objective."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


CORRECTION_SCALE = 4.960109710693359
PATCH_GRID = (37, 37)
REGION_GRID = (9, 9)


@dataclass(frozen=True)
class P29Loss:
    total: torch.Tensor
    value: torch.Tensor
    sign: torch.Tensor
    normal: torch.Tensor
    normalized_student: torch.Tensor
    normalized_teacher: torch.Tensor
    pure_normal: torch.Tensor


def source_pure_normal_regions(source_mask: torch.Tensor) -> torch.Tensor:
    """Return source-only pure-normal 9x9 regions using frozen max pooling."""
    if source_mask.ndim != 4 or tuple(source_mask.shape[1:]) != (1, 518, 518):
        raise ValueError(f"source mask must be [B,1,518,518], got {tuple(source_mask.shape)}")
    patch_occupancy = F.adaptive_max_pool2d(source_mask, PATCH_GRID)
    region_occupancy = F.adaptive_max_pool2d(patch_occupancy, REGION_GRID)
    return region_occupancy.eq(0)


def p29_sign_guarded_loss(
    student_region: torch.Tensor, teacher_region: torch.Tensor, source_mask: torch.Tensor
) -> P29Loss:
    """Compute the preregistered P29 objective without calibration or thresholds."""
    if student_region.shape != teacher_region.shape:
        raise ValueError("student and teacher region residuals must have identical shapes")
    if student_region.ndim != 4 or student_region.shape[0] != 3 or tuple(student_region.shape[-2:]) != REGION_GRID:
        raise ValueError("region residuals must be [3,B,9,9]")
    pure_normal = source_pure_normal_regions(source_mask)
    if pure_normal.shape[0] != student_region.shape[1]:
        raise ValueError("source mask batch size must match region residual batch size")
    pure_normal_staged = pure_normal.squeeze(1).unsqueeze(0).expand(student_region.shape[0], -1, -1, -1)
    normalized_student = student_region / CORRECTION_SCALE
    normalized_teacher = teacher_region / CORRECTION_SCALE
    value = F.smooth_l1_loss(normalized_student, normalized_teacher)
    sign = (normalized_teacher.abs() * F.relu(-normalized_teacher.sign() * normalized_student)).mean()
    if bool(pure_normal_staged.any()):
        normal = F.relu(normalized_student[pure_normal_staged]).square().mean()
    else:
        normal = normalized_student.sum() * 0.0
    return P29Loss(
        total=value + sign + normal,
        value=value,
        sign=sign,
        normal=normal,
        normalized_student=normalized_student,
        normalized_teacher=normalized_teacher,
        pure_normal=pure_normal,
    )
