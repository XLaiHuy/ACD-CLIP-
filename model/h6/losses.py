"""Numerically explicit auxiliary losses for H6 Progress 1."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def _patch_labels(mask: torch.Tensor, patch_count: int) -> torch.Tensor:
    grid = int(math.isqrt(int(patch_count)))
    if grid * grid != patch_count:
        raise ValueError(f"H6 center loss requires a square patch grid, got P={patch_count}")
    reduced = F.adaptive_max_pool2d(mask.float(), output_size=(grid, grid))
    return reduced.flatten(start_dim=1) > 0


def _nearest_prototype_distance(tokens: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    # tokens [B,P,D], prototypes [B,M,D] -> nearest squared L2 [B,P]
    distance = (tokens.float().unsqueeze(2) - prototypes.float().unsqueeze(1)).pow(2).mean(dim=-1)
    return distance.min(dim=2).values


def center_loss(
    projected_levels: torch.Tensor,
    prototype_normal: torch.Tensor,
    prototype_abnormal: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Anomaly-preserving CoPS center loss with robust empty-set handling."""
    if projected_levels.ndim != 4:
        raise ValueError("projected_levels must be [G,B,P,D]")
    groups, batch, patches, _ = projected_levels.shape
    if masks.shape[0] != batch or labels.shape[0] != batch:
        raise ValueError("mask/label batch does not match projected visual features")
    anomaly = _patch_labels(masks, patches)
    labels = labels.reshape(batch).bool()
    terms = []
    for group in range(groups):
        distances_normal = _nearest_prototype_distance(projected_levels[group], prototype_normal)
        distances_abnormal = _nearest_prototype_distance(projected_levels[group], prototype_abnormal)
        normal_image = ~labels
        if normal_image.any():
            terms.append(distances_normal[normal_image].mean())
        anomalous_image = labels
        if anomalous_image.any():
            normal_patch = anomalous_image[:, None] & ~anomaly
            abnormal_patch = anomalous_image[:, None] & anomaly
            if normal_patch.any():
                terms.append(distances_normal[normal_patch].mean())
            if abnormal_patch.any():
                terms.append(distances_abnormal[abnormal_patch].mean())
    if not terms:
        return projected_levels.float().sum() * 0.0
    return torch.stack(terms).mean()


def factor_orthogonal_loss(factor_bank: torch.Tensor) -> torch.Tensor:
    """Weakly diversify normal-to-abnormal directions in FP32."""
    if factor_bank.ndim != 5:
        raise ValueError("factor_bank must be [G,B,M,768,2]")
    directions = factor_bank[..., 1] - factor_bank[..., 0]
    directions = F.normalize(directions.float(), dim=-1)
    gram = torch.einsum("gbmd,gbnd->gbmn", directions, directions)
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    return (gram - identity).pow(2).mean()


def dynamic_residual_diversity_loss(dynamic_text: torch.Tensor, hard_frozen: torch.Tensor) -> torch.Tensor:
    """Diversify dynamic text residual directions around the frozen hard anchor."""
    if dynamic_text.ndim != 5:
        raise ValueError("dynamic_text must be [G,B,M,768,2]")
    if hard_frozen.ndim == 4:
        hard_frozen = hard_frozen.unsqueeze(2)
    if hard_frozen.ndim != 5:
        raise ValueError("hard_frozen must be [G,B,768,2] or [G,B,M,768,2]")
    dynamic_text = F.normalize(dynamic_text.float(), dim=3)
    hard_frozen = F.normalize(hard_frozen.float(), dim=3).expand_as(dynamic_text)
    residual = dynamic_text - hard_frozen
    directions = residual[..., 1] - residual[..., 0]
    directions = F.normalize(directions.float(), dim=-1)
    gram = torch.einsum("gbmd,gbnd->gbmn", directions, directions)
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    return (gram - identity).pow(2).mean()


def routing_balance_loss(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 4:
        raise ValueError("routing probabilities must be [G,B,P,M]")
    usage = probabilities.float().mean(dim=(0, 1, 2))
    target = torch.full_like(usage, 1.0 / usage.numel())
    return (usage - target).pow(2).sum()


def h6_loss_diagnostics(
    factor_bank: torch.Tensor,
    probabilities: torch.Tensor,
    prototype_normal: torch.Tensor,
    prototype_abnormal: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    directions = F.normalize((factor_bank[..., 1] - factor_bank[..., 0]).float(), dim=-1)
    direction_cosine = torch.einsum("gbmd,gbnd->gbmn", directions, directions)
    return {
        "direction_offdiag_absmax": (direction_cosine - torch.eye(direction_cosine.shape[-1], device=direction_cosine.device)).abs().amax().detach(),
        "prototype_normal_norm": prototype_normal.float().norm(dim=-1).mean().detach(),
        "prototype_abnormal_norm": prototype_abnormal.float().norm(dim=-1).mean().detach(),
        "center_distance": (prototype_normal.float() - prototype_abnormal.float()).norm(dim=-1).mean().detach(),
        "factor_bank_finite": torch.isfinite(factor_bank).all().detach(),
        "routing_finite": torch.isfinite(probabilities).all().detach(),
    }
