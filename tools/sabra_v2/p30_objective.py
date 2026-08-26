"""The single preregistered P30 directional distillation objective."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tools.sabra_v2.p29_objective import CORRECTION_SCALE


STAGES = 3
REGION_GRID = (9, 9)
P30_NORMALIZATION_EPSILON = 0.01


@dataclass(frozen=True)
class P30Loss:
    total: torch.Tensor
    directional: torch.Tensor
    valid_count: int
    normalized_student: torch.Tensor
    normalized_teacher: torch.Tensor


def _validate_student(student_region: torch.Tensor) -> None:
    if student_region.ndim != 4 or student_region.shape[0] != STAGES or tuple(student_region.shape[-2:]) != REGION_GRID:
        raise ValueError("student_region must be [3,B,9,9]")


def _stage_teacher(teacher_region: torch.Tensor, student_region: torch.Tensor) -> torch.Tensor:
    if teacher_region.ndim != 3 or tuple(teacher_region.shape[-2:]) != REGION_GRID:
        raise ValueError("teacher_region must be [B,9,9]")
    if teacher_region.shape[0] != student_region.shape[1]:
        raise ValueError("teacher batch must match student batch")
    return teacher_region.detach().unsqueeze(0).expand_as(student_region)


def p30_directional_loss(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
    *,
    epsilon: float = P30_NORMALIZATION_EPSILON,
) -> P30Loss:
    """Compare only per-sample direction over staged 9x9 signed corrections."""
    _validate_student(student_region)
    if epsilon <= 0.0 or not torch.isfinite(torch.tensor(epsilon)):
        raise ValueError("epsilon must be positive and finite")
    staged_teacher = _stage_teacher(teacher_region, student_region)
    normalized_student = student_region / CORRECTION_SCALE
    normalized_teacher = staged_teacher / CORRECTION_SCALE
    student_vectors = normalized_student.permute(1, 0, 2, 3).reshape(student_region.shape[1], -1)
    teacher_vectors = normalized_teacher.permute(1, 0, 2, 3).reshape(student_region.shape[1], -1)
    student_rms = torch.sqrt(student_vectors.square().mean(dim=1, keepdim=True) + float(epsilon) ** 2)
    teacher_rms = torch.sqrt(teacher_vectors.square().mean(dim=1, keepdim=True) + float(epsilon) ** 2)
    student_hat = student_vectors / student_rms
    teacher_hat = teacher_vectors / teacher_rms
    valid = teacher_vectors.abs().sum(dim=1).gt(0)
    valid_count = int(valid.sum().detach().cpu())
    if valid_count == 0:
        directional = student_region.sum() * 0.0
    else:
        cosine = (student_hat * teacher_hat).mean(dim=1)
        directional = (1.0 - cosine[valid]).mean()
    return P30Loss(
        total=directional,
        directional=directional,
        valid_count=valid_count,
        normalized_student=normalized_student,
        normalized_teacher=normalized_teacher,
    )
