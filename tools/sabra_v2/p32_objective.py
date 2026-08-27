"""Frozen P32 functional native-relative margin-effect objective.

The objective matches the scalar effect that the existing Industrial
deployment path applies to the abnormal-minus-normal margin.  It deliberately
does not require native logits: the symmetric two-class correction and the
linear blur/resize/stage-average deployment map make the native term cancel.
"""
from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d

from tools.sabra_v2.region_pool import REGION_GRID, STAGES


IMAGE_SIZE = 518
GAUSSIAN_KERNEL = (7, 7)
GAUSSIAN_SIGMA = (1.0, 1.0)
SMOOTH_L1_BETA = 1.0
P32_OBJECTIVE_NAME = "P32_FUNCTIONAL_MARGIN_EFFECT_SMOOTHL1_V1"
P32_PREREGISTRATION_SHA256 = "5141722b2c3e3d3aac721390a8943d54356dd17bdfdad8aaa6bd7302766a5cc2"


def _build_effect_matrix_cpu() -> torch.Tensor:
    """Build the exact separable deployment map once, in canonical FP32.

    The fixed deployment transform is separable: bilinear resize, Gaussian
    blur, and bilinear resize are each separable.  Applying the transform to
    width-only basis maps recovers its one-dimensional 518x9 matrix.  The
    production path then evaluates ``A @ x @ A.T``; this is algebraically the
    same transform as the repository deployment path, with much less redundant
    work than materializing a three-stage 518x518 map.
    """
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


def deployed_margin_effect(region: torch.Tensor) -> torch.Tensor:
    """Apply the frozen 9x9 -> 518x518 scalar deployment transform."""
    if region.ndim != 3 or tuple(region.shape[-2:]) != REGION_GRID:
        raise ValueError("region must be [B,9,9]")
    matrix = _effect_matrix(str(region.device), region.dtype)
    return torch.matmul(torch.matmul(matrix, region), matrix.transpose(0, 1))


def p32_functional_margin_components(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``loss, student_effect, detached_teacher_effect``."""
    _validate_shapes(student_region, teacher_region)
    student_effect = deployed_margin_effect(student_region.mean(dim=0))
    teacher_effect = deployed_margin_effect(teacher_region.detach())
    loss = F.smooth_l1_loss(
        student_effect,
        teacher_effect,
        beta=SMOOTH_L1_BETA,
        reduction="mean",
    )
    return loss, student_effect, teacher_effect


def p32_functional_margin_loss(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> torch.Tensor:
    """Compute the one and only P32 scientific objective."""
    return p32_functional_margin_components(student_region, teacher_region)[0]


def p32_objective_contract() -> dict[str, object]:
    """Expose immutable constants for provenance and engineering audits."""
    return {
        "name": P32_OBJECTIVE_NAME,
        "preregistration_sha256": P32_PREREGISTRATION_SHA256,
        "objective_count": 1,
        "student_shape": "[3,B,9,9]",
        "teacher_shape": "[B,9,9]",
        "effect_shape": "[B,518,518]",
        "smooth_l1_beta": SMOOTH_L1_BETA,
        "gaussian_kernel": list(GAUSSIAN_KERNEL),
        "gaussian_sigma": list(GAUSSIAN_SIGMA),
        "align_corners": True,
        "teacher_detached": True,
        "student_self_normalized": False,
        "auxiliary_terms": [],
        "teacher_at_inference": False,
        "incremental_inference_overhead_percent": 0,
    }
