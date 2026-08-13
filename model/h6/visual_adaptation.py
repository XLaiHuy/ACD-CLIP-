"""Minimal K=1 semantic-conditioned visual residual for Phase4-V."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


class ConditionalVisualAdapter(nn.Module):
    """One shared low-rank visual transform, optionally FiLM-conditioned by CoPS.

    ``gate`` is detached inside this module by contract. Exact no-op cases use
    the original input tensor rather than a redundant normalization, preserving
    the Phase2B feature bit-for-bit when correction is exactly zero.
    """

    def __init__(self, visual_dim: int = 768, state_dim: int = 256, bottleneck: int = 64, lambda_value: float = 0.05) -> None:
        super().__init__()
        if not 0.0 < lambda_value <= 0.05:
            raise ValueError("Phase4-V lambda must be in (0,.05]")
        self.visual_dim = int(visual_dim)
        self.state_dim = int(state_dim)
        self.bottleneck = int(bottleneck)
        self.down = nn.Linear(visual_dim, bottleneck)
        self.controller = nn.Linear(state_dim, 2 * bottleneck)
        self.up = nn.Linear(bottleneck, visual_dim)
        self.register_buffer("lambda_value", torch.tensor(float(lambda_value), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.controller.weight, std=1e-3)
        nn.init.zeros_(self.controller.bias)
        nn.init.normal_(self.up.weight, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        patches: torch.Tensor,
        semantic_code: torch.Tensor,
        gate: torch.Tensor,
        *,
        enabled: bool,
        semantic_conditioning: bool,
        spatial_gating: bool,
        lambda_override: float | None = None,
        force_delta_zero: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if patches.ndim != 3 or patches.shape[-1] != self.visual_dim:
            raise ValueError(f"patches must be [B,P,{self.visual_dim}]")
        batch, patch_count, _ = patches.shape
        if semantic_code.shape != (batch, self.state_dim):
            raise ValueError(f"semantic_code must be [B,{self.state_dim}]")
        if gate.shape != (batch, patch_count):
            raise ValueError("gate must be [B,P]")
        if not enabled:
            zeros = torch.zeros_like(patches)
            return {"adapted": patches, "delta_v": zeros, "correction": zeros, "gate": gate.detach()}
        lambda_value = float(self.lambda_value if lambda_override is None else lambda_override)
        if not 0.0 <= lambda_value <= float(self.lambda_value):
            raise ValueError("lambda override must stay within the fixed Phase4-V bound")
        h = self.down(patches.float())
        if semantic_conditioning:
            gamma, beta = self.controller(semantic_code.float()).chunk(2, dim=-1)
            gamma, beta = 0.10 * torch.tanh(gamma), 0.10 * torch.tanh(beta)
        else:
            gamma, beta = torch.zeros_like(h[:, 0]), torch.zeros_like(h[:, 0])
        h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)
        delta_v = self.up(F.gelu(h))
        if force_delta_zero:
            delta_v = delta_v * 0.0
        effective_gate = gate.detach() if spatial_gating else torch.ones_like(gate).detach()
        correction = lambda_value * effective_gate.unsqueeze(-1) * delta_v
        exact_zero = correction.abs().sum(dim=-1, keepdim=True) == 0
        normalized = F.normalize(patches.float() + correction.float(), dim=-1).to(dtype=patches.dtype)
        adapted = torch.where(exact_zero, patches, normalized)
        return {"adapted": adapted, "delta_v": delta_v, "correction": correction, "gate": effective_gate}
