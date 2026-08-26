"""The exact frozen P30R1 teacher-relative single residual objective."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tools.sabra_v2.p29_objective import CORRECTION_SCALE


STAGES = 3
REGION_GRID = (9, 9)
COORDINATE_COUNT = STAGES * REGION_GRID[0] * REGION_GRID[1]
P30R1_NORMALIZATION_EPSILON = 0.01
P30R1_SMOOTH_L1_BETA = 1.0
P30R1_OBJECTIVE_NAME = "P30R1_TEACHER_RELATIVE_SMOOTHL1_V1"
P30R1_FORMULATION_HASH = "290aae42e04d9faae5a10b929eb58aa0da066b5dbd248b3fee40f20e9094781c"


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


def p30r1_teacher_relative_components(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return loss and normalized vectors for the frozen P30R1 formulation."""
    _validate_shapes(student_region, teacher_region)
    teacher_staged = teacher_region.detach().unsqueeze(0).expand_as(student_region)
    student_bar = student_region / CORRECTION_SCALE
    teacher_bar = teacher_staged / CORRECTION_SCALE
    student_vectors = student_bar.permute(1, 0, 2, 3).reshape(student_region.shape[1], COORDINATE_COUNT)
    teacher_vectors = teacher_bar.permute(1, 0, 2, 3).reshape(teacher_region.shape[0], COORDINATE_COUNT)
    teacher_scale = torch.sqrt(
        teacher_vectors.square().mean(dim=1, keepdim=True)
        + P30R1_NORMALIZATION_EPSILON**2
    ).detach()
    normalized_student = student_vectors / teacher_scale
    normalized_teacher = teacher_vectors / teacher_scale
    loss = F.smooth_l1_loss(
        normalized_student,
        normalized_teacher,
        beta=P30R1_SMOOTH_L1_BETA,
        reduction="mean",
    )
    return loss, normalized_student, normalized_teacher, teacher_scale


def p30r1_teacher_relative_loss(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> torch.Tensor:
    """Compute the one and only P30R1 scientific objective."""
    return p30r1_teacher_relative_components(student_region, teacher_region)[0]


def p30r1_objective_contract() -> dict[str, object]:
    """Expose immutable constants for provenance/audit code."""
    return {
        "name": P30R1_OBJECTIVE_NAME,
        "formulation_hash": P30R1_FORMULATION_HASH,
        "objective_count": 1,
        "correction_scale_C": CORRECTION_SCALE,
        "normalization_epsilon": P30R1_NORMALIZATION_EPSILON,
        "smooth_l1_beta": P30R1_SMOOTH_L1_BETA,
        "coordinate_count": COORDINATE_COUNT,
        "teacher_detached": True,
        "same_teacher_denominator": True,
        "student_self_normalized": False,
        "exact_zero_teacher_active": True,
    }


def p30r1_gradient_bounds(batch_size: int = 1) -> dict[str, float]:
    """Return the analytic per-coordinate and per-sample L2 bounds."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    coordinate = 1.0 / (batch_size * COORDINATE_COUNT * CORRECTION_SCALE * P30R1_NORMALIZATION_EPSILON)
    sample_l2 = 1.0 / (batch_size * CORRECTION_SCALE * P30R1_NORMALIZATION_EPSILON * math.sqrt(COORDINATE_COUNT))
    return {"gradient_max_abs": coordinate, "per_sample_gradient_l2": sample_l2}
