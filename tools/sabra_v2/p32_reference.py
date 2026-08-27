"""Readable P32 reference algebra for deterministic parity tests.

This intentionally goes through the existing symmetric two-logit deployment
path.  P32 production code collapses the same linear operations to the
stage-mean scalar path in :mod:`p32_objective`; the two implementations must
agree within the recorded FP32 tolerance.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from model.phase2b_runtime import deploy_native_logits
from tools.sabra_v2.region_pool import symmetric_margin_delta, upsample_region_map


STAGES = 3
REGION_GRID = (9, 9)
PATCH_GRID = (37, 37)
IMAGE_SIZE = 518
SMOOTH_L1_BETA = 1.0


def _validate_shapes(student_region: torch.Tensor, teacher_region: torch.Tensor) -> None:
    if student_region.ndim != 4 or student_region.shape[0] != STAGES or tuple(student_region.shape[-2:]) != REGION_GRID:
        raise ValueError("student_region must be [3,B,9,9]")
    if teacher_region.ndim != 3 or tuple(teacher_region.shape[-2:]) != REGION_GRID:
        raise ValueError("teacher_region must be [B,9,9]")
    if teacher_region.shape[0] != student_region.shape[1]:
        raise ValueError("teacher batch must match student batch")


def _effect_from_staged_region(staged_region: torch.Tensor) -> torch.Tensor:
    """Compute a deployed margin effect through the two-logit path."""
    if staged_region.ndim != 4 or staged_region.shape[0] != STAGES or tuple(staged_region.shape[-2:]) != REGION_GRID:
        raise ValueError("staged_region must be [3,B,9,9]")
    batch = staged_region.shape[1]
    patch = upsample_region_map(staged_region)
    patch_delta = patch.reshape(STAGES, batch, -1)
    zero_native = torch.zeros(
        (STAGES, batch, PATCH_GRID[0] * PATCH_GRID[1], 2),
        dtype=staged_region.dtype,
        device=staged_region.device,
    )
    delta = symmetric_margin_delta(zero_native, patch)
    _unused_probability, deployed_logits = deploy_native_logits(
        delta,
        patch_grid=PATCH_GRID,
        image_size=IMAGE_SIZE,
        domain="Industrial",
    )
    zero_margin = zero_native[..., 1] - zero_native[..., 0]
    if not torch.equal(zero_margin, torch.zeros_like(zero_margin)):
        raise AssertionError("reference zero native margin is not zero")
    effect = deployed_logits[:, 1] - deployed_logits[:, 0]
    if patch_delta.shape != (STAGES, batch, PATCH_GRID[0] * PATCH_GRID[1]):
        raise AssertionError("reference patch shape changed")
    return effect


def p32_functional_margin_components(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference loss and effects using the full deployment algebra."""
    _validate_shapes(student_region, teacher_region)
    student_effect = _effect_from_staged_region(student_region)
    teacher_staged = teacher_region.detach().unsqueeze(0).expand_as(student_region)
    teacher_effect = _effect_from_staged_region(teacher_staged)
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
    """Compute the reference P32 loss."""
    return p32_functional_margin_components(student_region, teacher_region)[0]
