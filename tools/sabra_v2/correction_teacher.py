"""Source-only P27 teacher targets using the frozen historical R0 semantics.

This module is deliberately separate from the student forward path: its only
entrypoint requiring masks is for source-training supervision.
"""
from __future__ import annotations

import torch

from tools.sabra_v2.region_pool import PATCH_GRID, STAGES, pool_patch_map


R0_TEACHER_ALPHA = 0.25
R0_MARGIN_SCALE = 19.840438842773438


def _validate_actions(actions: torch.Tensor) -> None:
    if actions.ndim != 2 or actions.shape[1] != PATCH_GRID[0] * PATCH_GRID[1]:
        raise ValueError("actions must be [B,1369]")
    if not torch.all((actions == -1) | (actions == 0) | (actions == 1)):
        raise ValueError("actions must contain only -1, 0, or +1")


def teacher_patch_delta_from_actions(actions: torch.Tensor) -> torch.Tensor:
    """Convert historical R0 signed actions into a source-only patch correction."""
    _validate_actions(actions)
    return (actions.to(dtype=torch.float32) * (R0_TEACHER_ALPHA * R0_MARGIN_SCALE)).reshape(-1, *PATCH_GRID)


def build_source_teacher_region_target(native_logits: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
    """Create a 9x9 teacher target from source logits and source GT only.

    The R0 utility creates an abnormal-channel-only intervention strictly to
    derive historical signed utility.  That intervention is never deployed by
    P27; student inference instead uses the separately approved symmetric
    two-logit integration in ``region_pool.symmetric_margin_delta``.
    """
    if native_logits.ndim != 4 or native_logits.shape[0] != STAGES or native_logits.shape[-2:] != (1369, 2):
        raise ValueError("native_logits must be [3,B,1369,2]")
    if source_mask.ndim != 4 or source_mask.shape[:2] != (native_logits.shape[1], 1):
        raise ValueError("source_mask must be [B,1,H,W] aligned with native_logits")
    # Lazy import keeps student-only inference independent from the legacy R0
    # utility runtime while retaining its exact signed-utility implementation.
    from tools.sabra_car import r0_direction

    if r0_direction.MARGIN_SCALE != R0_MARGIN_SCALE:
        raise RuntimeError("historical R0 margin scale provenance mismatch")
    with torch.enable_grad():
        utility, _ = r0_direction.utility_for_batch(native_logits.detach(), source_mask.float())
    actions = r0_direction.classify_actions(utility)
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions, device=native_logits.device)
    return pool_patch_map(teacher_patch_delta_from_actions(actions))
