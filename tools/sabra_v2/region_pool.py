"""Deterministic P27 region geometry and symmetric margin integration."""
from __future__ import annotations

import torch
import torch.nn.functional as F


PATCH_GRID = (37, 37)
REGION_GRID = (9, 9)
STAGES = 3


def _validate_spatial_map(value: torch.Tensor, name: str, expected: tuple[int, int]) -> None:
    if value.ndim not in (3, 4) or tuple(value.shape[-2:]) != expected:
        raise ValueError(f"{name} must be [B,{expected[0]},{expected[1]}] or [S,B,{expected[0]},{expected[1]}]")


def _adapt_spatial(value: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
    leading = value.shape[:-2]
    spatial = value.reshape(-1, 1, *value.shape[-2:])
    if mode == "area":
        result = F.adaptive_avg_pool2d(spatial, size)
    else:
        result = F.interpolate(spatial, size=size, mode=mode, align_corners=True)
    return result.reshape(*leading, *size)


def pool_patch_map(patch_map: torch.Tensor) -> torch.Tensor:
    """Adaptively average a 37x37 patch map to the frozen 9x9 region grid."""
    _validate_spatial_map(patch_map, "patch_map", PATCH_GRID)
    return _adapt_spatial(patch_map, REGION_GRID, "area")


def upsample_region_map(region_map: torch.Tensor) -> torch.Tensor:
    """Bilinearly upsample a 9x9 region map to the canonical 37x37 grid."""
    _validate_spatial_map(region_map, "region_map", REGION_GRID)
    return _adapt_spatial(region_map, PATCH_GRID, "bilinear")


def symmetric_margin_delta(native_logits: torch.Tensor, patch_delta: torch.Tensor) -> torch.Tensor:
    """Apply a scalar margin residual with no common two-class logit offset."""
    if native_logits.ndim != 4 or native_logits.shape[0] != STAGES or native_logits.shape[-2:] != (37 * 37, 2):
        raise ValueError("native_logits must be [3,B,1369,2]")
    _validate_spatial_map(patch_delta, "patch_delta", PATCH_GRID)
    if patch_delta.ndim != 4 or tuple(patch_delta.shape[:2]) != tuple(native_logits.shape[:2]):
        raise ValueError("patch_delta must be [3,B,37,37] aligned with native_logits")
    flattened = patch_delta.reshape(STAGES, native_logits.shape[1], 37 * 37)
    return native_logits + torch.stack((-flattened * 0.5, flattened * 0.5), dim=-1)
