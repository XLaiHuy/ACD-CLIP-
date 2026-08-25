"""Frozen-shape P27 shared region residual adapter."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from tools.sabra_v2.region_pool import PATCH_GRID, REGION_GRID, STAGES


class RegionResidualAdapter(nn.Module):
    """Project three frozen P26 stages and emit one 9x9 residual map per stage."""

    def __init__(self, visual_dim: int = 768, projection_dim: int = 64) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.projection_dim = projection_dim
        self.stage_projection = nn.Linear(visual_dim, projection_dim)
        self.context_1x1 = nn.Conv2d(STAGES * projection_dim, projection_dim, kernel_size=1)
        self.context_depthwise = nn.Conv2d(
            projection_dim, projection_dim, kernel_size=3, padding=1, groups=projection_dim
        )
        self.output = nn.Conv2d(projection_dim, STAGES, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, seg_features: torch.Tensor) -> torch.Tensor:
        """Return [3,B,9,9] residual maps from [3,B,1369,768] frozen features."""
        if seg_features.ndim != 4 or seg_features.shape[0] != STAGES:
            raise ValueError("seg_features must be [3,B,1369,768]")
        if seg_features.shape[2] != PATCH_GRID[0] * PATCH_GRID[1] or seg_features.shape[3] != self.visual_dim:
            raise ValueError(f"seg_features must be [3,B,1369,{self.visual_dim}]")
        stages, batch, _, _ = seg_features.shape
        projected = self.stage_projection(seg_features)
        maps = projected.reshape(stages, batch, *PATCH_GRID, self.projection_dim).permute(0, 1, 4, 2, 3)
        pooled = F.adaptive_avg_pool2d(maps.reshape(stages * batch, self.projection_dim, *PATCH_GRID), REGION_GRID)
        fused = pooled.reshape(stages, batch, self.projection_dim, *REGION_GRID).permute(1, 0, 2, 3, 4)
        fused = fused.reshape(batch, STAGES * self.projection_dim, *REGION_GRID)
        context = F.gelu(self.context_1x1(fused))
        context = F.gelu(self.context_depthwise(context))
        return self.output(context).permute(1, 0, 2, 3)
