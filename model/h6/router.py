"""Patch-wise sparse text routing for the H6 dynamic factor bank."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class PatchRouter(nn.Module):
    def __init__(
        self,
        n_groups: int,
        num_factors: int = 4,
        text_dim: int = 768,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        soft_routing_epochs: int = 2,
        top_k: int = 2,
    ):
        super().__init__()
        if not 0 < top_k <= num_factors:
            raise ValueError("top_k must be in [1, num_factors]")
        if temperature <= 0:
            raise ValueError("router temperature must be positive")
        self.n_groups = int(n_groups)
        self.num_factors = int(num_factors)
        self.text_dim = int(text_dim)
        self.temperature = float(temperature)
        self.soft_routing_epochs = int(soft_routing_epochs)
        self.top_k = int(top_k)
        self.level_embedding = nn.Parameter(torch.empty(n_groups, text_dim))
        self.trunk = nn.Sequential(
            nn.Linear(3 * text_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, num_factors)
        )
        nn.init.normal_(self.level_embedding, std=0.02)

    @staticmethod
    def _stack(level_tokens: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        return level_tokens if torch.is_tensor(level_tokens) else torch.stack(list(level_tokens), dim=0)

    def forward(self, level_tokens: Sequence[torch.Tensor] | torch.Tensor, epoch_one_based: int) -> Dict[str, torch.Tensor]:
        stacked = self._stack(level_tokens)
        with torch.autocast(device_type=stacked.device.type, enabled=False):
            return self._forward_fp32(stacked, epoch_one_based)

    def _forward_fp32(self, tokens: torch.Tensor, epoch_one_based: int) -> Dict[str, torch.Tensor]:
        if tokens.ndim != 4 or tokens.shape[0] != self.n_groups or tokens.shape[-1] != self.text_dim:
            raise ValueError("router input must be [n_groups, B, P, 768]")
        tokens = F.normalize(tokens.float(), dim=-1)
        groups, batch, patches, _ = tokens.shape
        context = tokens.mean(dim=2, keepdim=True).expand(-1, -1, patches, -1)
        level = self.level_embedding[:, None, None, :].expand(groups, batch, patches, -1)
        logits = self.trunk(torch.cat([tokens, context, level], dim=-1)).float() / self.temperature
        if epoch_one_based <= self.soft_routing_epochs:
            probabilities = F.softmax(logits, dim=-1)
            indices = torch.arange(self.num_factors, device=logits.device).view(1, 1, 1, -1)
            indices = indices.expand(groups, batch, patches, -1)
            sparse = False
        else:
            _, indices = torch.topk(logits, k=self.top_k, dim=-1)
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits.scatter_(-1, indices, logits.gather(-1, indices))
            probabilities = F.softmax(masked_logits, dim=-1)
            sparse = True
        return {
            "logits": logits,
            "probabilities": probabilities,
            "topk_indices": indices,
            "sparse": torch.tensor(sparse, device=logits.device),
        }

    @staticmethod
    def aggregate_global(probabilities: torch.Tensor, factor_bank: torch.Tensor) -> torch.Tensor:
        """Return image-wise text [G,B,768,2] from the one shared factor bank."""
        routing_global = probabilities.float().mean(dim=2)
        text_global = torch.einsum("gbm,gbmds->gbds", routing_global, factor_bank.float())
        return F.normalize(text_global, dim=2)

    @staticmethod
    def local_text(probabilities: torch.Tensor, factor_bank: torch.Tensor) -> torch.Tensor:
        """Return patch-local normal/abnormal text [G,B,P,768,2]."""
        text = torch.einsum("gbpm,gbmds->gbpds", probabilities.float(), factor_bank.float())
        return F.normalize(text, dim=3)

    @staticmethod
    def diagnostics(probabilities: torch.Tensor) -> Dict[str, torch.Tensor]:
        probabilities = probabilities.float()
        factors = probabilities.shape[-1]
        usage = probabilities.mean(dim=(1, 2))
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy.mean(dim=(1, 2)) / torch.log(torch.tensor(float(factors), device=probabilities.device))
        return {
            "factor_usage": usage.detach(),
            "normalized_entropy": entropy.detach(),
            "top1_share": usage.max(dim=-1).values.detach(),
            "dead_factors": (usage < 0.01).sum(dim=-1).detach(),
            "max_factor_usage": usage.max(dim=-1).values.detach(),
        }
