"""Frozen-orthogonal functional semantic experts for P1-v7.

The frozen A matrices give each factor a disjoint input subspace while the
zero-initialised B matrices make this an exact P1-v6 no-op at construction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _fofs_basis(num_factors: int, bottleneck: int, text_dim: int, seed: int) -> torch.Tensor:
    if num_factors * bottleneck > text_dim:
        raise ValueError(f"FOFS needs num_factors*bottleneck <= text_dim; got {num_factors}*{bottleneck}>{text_dim}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    # QR columns are orthonormal; transposition yields orthonormal rows.
    matrix = torch.randn(text_dim, num_factors * bottleneck, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.T.reshape(num_factors, bottleneck, text_dim).contiguous()


class FOFSPairedSemanticExperts(nn.Module):
    def __init__(
        self, num_factors: int, text_dim: int, bank_dim: int, bottleneck: int = 64,
        seed_offset: int = 7500, state_condition_scale: float = 0.25,
        max_relative_ratio: float = 0.10,
    ) -> None:
        super().__init__()
        self.num_factors, self.text_dim, self.bank_dim = int(num_factors), int(text_dim), int(bank_dim)
        self.bottleneck = int(bottleneck)
        self.seed_offset = int(seed_offset)
        self.state_condition_scale = float(state_condition_scale)
        self.max_relative_ratio = float(max_relative_ratio)
        self.register_buffer("fofs_A", _fofs_basis(self.num_factors, self.bottleneck, self.text_dim, self.seed_offset))
        self.expert_B = nn.Parameter(torch.zeros(self.num_factors, self.text_dim, self.bottleneck))
        self.state_projection = nn.Linear(self.bank_dim, self.text_dim, bias=False)
        nn.init.zeros_(self.state_projection.weight)

    def forward(self, base_factor_bank: torch.Tensor, prototype_normal: torch.Tensor,
                prototype_abnormal: torch.Tensor, scale: float) -> dict[str, torch.Tensor]:
        if base_factor_bank.ndim != 5:
            raise ValueError("base_factor_bank must be [G,B,M,D,2]")
        groups, batch, factors, dim, states = base_factor_bank.shape
        if (factors, dim, states) != (self.num_factors, self.text_dim, 2):
            raise ValueError("base_factor_bank does not match expert configuration")
        if prototype_normal.shape != (batch, factors, self.bank_dim) or prototype_abnormal.shape != prototype_normal.shape:
            raise ValueError("prototypes must be [B,M,bank_dim]")
        normal, abnormal = base_factor_bank[..., 0].float(), base_factor_bank[..., 1].float()
        base_direction = abnormal - normal
        base_hat = F.normalize(base_direction, dim=-1, eps=1e-6)
        state = prototype_abnormal.float() - prototype_normal.float()
        state = self.state_projection(state).unsqueeze(0).expand(groups, -1, -1, -1)
        expert_input = F.normalize(base_direction + self.state_condition_scale * state, dim=-1, eps=1e-6)
        hidden = F.gelu(torch.einsum("mrd,gbmd->gbmr", self.fofs_A.to(expert_input), expert_input))
        delta_raw = torch.einsum("mdr,gbmr->gbmd", self.expert_B.float(), hidden)
        delta_tangent = delta_raw - (delta_raw * base_hat).sum(dim=-1, keepdim=True) * base_hat
        desired = delta_tangent * float(scale)
        direction_norm = base_direction.norm(dim=-1, keepdim=True)
        max_norm = direction_norm * self.max_relative_ratio
        desired_norm = desired.norm(dim=-1, keepdim=True)
        clamp_scale = torch.minimum(torch.ones_like(desired_norm), max_norm / desired_norm.clamp_min(1e-8))
        applied = desired * clamp_scale
        factor_bank = torch.stack((F.normalize(normal - applied, dim=-1), F.normalize(abnormal + applied, dim=-1)), dim=-1)
        return {
            "expert_factor_bank": factor_bank,
            "expert_delta_raw": delta_raw,
            "expert_delta_tangent": delta_tangent,
            "expert_applied_delta": applied,
            "expert_relative_ratio": applied.norm(dim=-1) / direction_norm.squeeze(-1).clamp_min(1e-8),
            "expert_clamp_fraction": (clamp_scale < 0.999999).float().mean(),
            "expert_delta_norm": applied.norm(dim=-1),
        }
