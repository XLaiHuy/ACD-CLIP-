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
        bank_dim: int = 256,
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
        self.bank_dim = int(bank_dim)
        self.temperature = float(temperature)
        self.soft_routing_epochs = int(soft_routing_epochs)
        self.top_k = int(top_k)
        self.level_embedding = nn.Parameter(torch.empty(n_groups, text_dim))
        self.query_projector = nn.Sequential(
            nn.Linear(3 * text_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, bank_dim)
        )
        self.fallback_concept_keys = nn.Parameter(torch.empty(num_factors, bank_dim))
        nn.init.normal_(self.level_embedding, std=0.02)
        nn.init.normal_(self.fallback_concept_keys, std=0.02)

    @staticmethod
    def _stack(level_tokens: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        return level_tokens if torch.is_tensor(level_tokens) else torch.stack(list(level_tokens), dim=0)

    def forward(
        self,
        level_tokens: Sequence[torch.Tensor] | torch.Tensor,
        epoch_one_based: int,
        concept_keys: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        stacked = self._stack(level_tokens)
        with torch.autocast(device_type=stacked.device.type, enabled=False):
            return self._forward_fp32(stacked, epoch_one_based, concept_keys)

    def _forward_fp32(
        self,
        tokens: torch.Tensor,
        epoch_one_based: int,
        concept_keys: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if tokens.ndim != 4 or tokens.shape[0] != self.n_groups or tokens.shape[-1] != self.text_dim:
            raise ValueError("router input must be [n_groups, B, P, 768]")
        if concept_keys is None:
            concept_keys = self.fallback_concept_keys
        if concept_keys.ndim != 2 or concept_keys.shape != (self.num_factors, self.bank_dim):
            raise ValueError(f"concept_keys must be [{self.num_factors}, {self.bank_dim}]")
        tokens = F.normalize(tokens.float(), dim=-1)
        groups, batch, patches, _ = tokens.shape
        context = tokens.mean(dim=2, keepdim=True).expand(-1, -1, patches, -1)
        level = self.level_embedding[:, None, None, :].expand(groups, batch, patches, -1)
        query = self.query_projector(torch.cat([tokens, context, level], dim=-1)).float()
        query = F.normalize(query, dim=-1)
        keys = F.normalize(concept_keys.float(), dim=-1)
        logits = torch.einsum("gbpd,md->gbpm", query, keys) / self.temperature
        dense_probabilities = F.softmax(logits, dim=-1)
        _, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(-1, topk_indices, logits.gather(-1, topk_indices))
        sparse_probabilities = F.softmax(masked_logits, dim=-1)
        sparse_active = bool(epoch_one_based > self.soft_routing_epochs)
        prediction_probabilities = sparse_probabilities if sparse_active else dense_probabilities
        return {
            "logits": logits,
            "dense_probabilities": dense_probabilities,
            "sparse_probabilities": sparse_probabilities,
            "prediction_probabilities": prediction_probabilities,
            "probabilities": prediction_probabilities,
            "topk_indices": topk_indices,
            "sparse_active": torch.tensor(sparse_active, device=logits.device),
            "sparse": torch.tensor(sparse_active, device=logits.device),
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
    def _probability_diagnostics(probabilities: torch.Tensor) -> Dict[str, torch.Tensor]:
        probabilities = probabilities.float()
        factors = probabilities.shape[-1]
        usage = probabilities.mean(dim=(1, 2))
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy.mean(dim=(1, 2)) / torch.log(torch.tensor(float(factors), device=probabilities.device))
        return {
            "usage": usage.detach(),
            "entropy": entropy.detach(),
            "top1_share": usage.max(dim=-1).values.detach(),
            "dead_factors": (usage < 0.01).sum(dim=-1).detach(),
            "max_factor_usage": usage.max(dim=-1).values.detach(),
        }

    @staticmethod
    def diagnostics(
        prediction_probabilities: torch.Tensor,
        dense_probabilities: torch.Tensor | None = None,
        sparse_probabilities: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        prediction = PatchRouter._probability_diagnostics(prediction_probabilities)
        output = {
            "factor_usage": prediction["usage"],
            "normalized_entropy": prediction["entropy"],
            "top1_share": prediction["top1_share"],
            "dead_factors": prediction["dead_factors"],
            "max_factor_usage": prediction["max_factor_usage"],
            "prediction_factor_usage": prediction["usage"],
            "prediction_normalized_entropy": prediction["entropy"],
        }
        if dense_probabilities is not None:
            dense = PatchRouter._probability_diagnostics(dense_probabilities)
            output["dense_factor_usage"] = dense["usage"]
            output["dense_normalized_entropy"] = dense["entropy"]
        if sparse_probabilities is not None:
            sparse = PatchRouter._probability_diagnostics(sparse_probabilities)
            output["sparse_factor_usage"] = sparse["usage"]
            output["sparse_normalized_entropy"] = sparse["entropy"]
        if topk_indices is not None:
            selected = F.one_hot(topk_indices.long(), num_classes=prediction_probabilities.shape[-1]).float()
            output["selected_topk_frequency"] = selected.mean(dim=(1, 2, 3)).detach()
        return output
