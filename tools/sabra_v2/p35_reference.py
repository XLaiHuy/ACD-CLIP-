"""Readable P35 reference algebra for deterministic parity tests."""
from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d

from tools.sabra_v2.p29_contract import CORRECTION_SCALE
from tools.sabra_v2.region_pool import REGION_GRID, STAGES


IMAGE_SIZE = 518
GAUSSIAN_KERNEL = (7, 7)
GAUSSIAN_SIGMA = (1.0, 1.0)
SMOOTH_L1_BETA = 1.0


def _build_effect_matrix_cpu() -> torch.Tensor:
    """Build the separable linear form of the frozen deployment transform."""
    basis = torch.eye(9, dtype=torch.float32).reshape(9, 1, 1, 9).expand(-1, 1, 9, -1)
    patch = F.interpolate(basis, size=(37, 37), mode="bilinear", align_corners=True)
    patch = gaussian_blur2d(patch, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    deployed = F.interpolate(patch, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True)
    return deployed[:, 0, 0, :].transpose(0, 1).contiguous()


_EFFECT_MATRIX_CPU = _build_effect_matrix_cpu()


@lru_cache(maxsize=8)
def _effect_matrix(device_name: str, dtype: torch.dtype) -> torch.Tensor:
    return _EFFECT_MATRIX_CPU.to(device=torch.device(device_name), dtype=dtype)


def _validate_shapes(student_region: torch.Tensor, teacher_region: torch.Tensor) -> None:
    if student_region.ndim != 4 or student_region.shape[0] != STAGES or tuple(student_region.shape[-2:]) != REGION_GRID:
        raise ValueError("student_region must be [3,B,9,9]")
    if teacher_region.ndim != 3 or tuple(teacher_region.shape[-2:]) != REGION_GRID:
        raise ValueError("teacher_region must be [B,9,9]")
    if teacher_region.shape[0] != student_region.shape[1]:
        raise ValueError("teacher batch must match student batch")


def _effect_from_region(region: torch.Tensor) -> torch.Tensor:
    matrix = _effect_matrix(str(region.device), region.dtype)
    return torch.matmul(torch.matmul(matrix, region), matrix.transpose(0, 1))


def p35_actionability_components(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute P35 through the full symmetric deployment algebra."""
    _validate_shapes(student_region, teacher_region)
    student_effect = _effect_from_region(student_region.mean(dim=0))
    teacher_effect = _effect_from_region(teacher_region.detach())
    weight = torch.tanh((teacher_effect.abs() / CORRECTION_SCALE)).detach()
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
    return p35_actionability_components(student_region, teacher_region)[0]
