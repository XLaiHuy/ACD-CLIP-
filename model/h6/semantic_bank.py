"""CoPS-style semantic state and class updates for H6 Progress 1."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


class BoundedPositiveGate(nn.Module):
    """A strictly positive scalar with a known, inspectable upper bound."""

    def __init__(self, initial: float, maximum: float, count: int = 1):
        super().__init__()
        if not 0.0 < initial < maximum:
            raise ValueError("gate initial value must be strictly inside (0, maximum)")
        self.maximum = float(maximum)
        initial_probability = float(initial) / self.maximum
        self.raw = nn.Parameter(torch.full((count,), _logit(initial_probability)))

    def forward(self, cap: float | None = None) -> torch.Tensor:
        value = self.maximum * torch.sigmoid(self.raw)
        if cap is not None:
            cap = min(float(cap), self.maximum)
            value = torch.minimum(value, torch.full_like(value, cap))
        return value

    def extra_repr(self) -> str:
        return f"maximum={self.maximum}"


class ClassVAE(nn.Module):
    """The detached global CLS auxiliary VAE specified for H6 Progress 1."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, cls24: torch.Tensor) -> Dict[str, torch.Tensor]:
        target = cls24.detach().float()
        hidden = self.encoder(target)
        mu = self.mu(hidden).float()
        logvar = self.logvar(hidden).float().clamp(-10.0, 10.0)
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        class_semantic = self.decoder(z).float()
        reconstruction = F.mse_loss(class_semantic, target, reduction="mean")
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return {
            "class_semantic": class_semantic,
            "reconstruction": reconstruction,
            "kl": kl,
            "mu": mu,
            "logvar": logvar,
        }


class CoPSSemanticCore(nn.Module):
    """Extract paired state prototypes and image-conditioned soft contexts.

    This module deliberately operates in FP32 for the attention-derived
    semantic state, VAE statistics, and context normalisation.  It receives
    Phase2B projected visual tokens; it never owns or reruns the visual tower.
    """

    def __init__(
        self,
        n_groups: int,
        num_factors: int = 4,
        bank_dim: int = 256,
        text_dim: int = 768,
        ctx_len: int = 4,
        vae_hidden_dim: int = 512,
        vae_latent_dim: int = 256,
    ):
        super().__init__()
        if bank_dim % 4 != 0:
            raise ValueError("bank_dim must be divisible by four attention heads")
        self.n_groups = int(n_groups)
        self.num_factors = int(num_factors)
        self.bank_dim = int(bank_dim)
        self.text_dim = int(text_dim)
        self.ctx_len = int(ctx_len)

        self.level_projectors = nn.ModuleList(
            [nn.Sequential(nn.Linear(text_dim, bank_dim), nn.LayerNorm(bank_dim)) for _ in range(n_groups)]
        )
        self.level_embedding = nn.Parameter(torch.empty(n_groups, bank_dim))
        self.concept_slots = nn.Parameter(torch.empty(num_factors, bank_dim))
        self.normal_query = nn.Linear(bank_dim, bank_dim, bias=False)
        self.abnormal_query = nn.Linear(bank_dim, bank_dim, bias=False)
        self.router_key = nn.Linear(bank_dim, bank_dim, bias=False)
        self.prototype_attention = nn.MultiheadAttention(
            bank_dim, num_heads=4, dropout=0.0, batch_first=True
        )
        self.normal_state_update = nn.Sequential(
            nn.LayerNorm(bank_dim), nn.Linear(bank_dim, bank_dim), nn.GELU(), nn.Linear(bank_dim, bank_dim)
        )
        self.abnormal_state_update = nn.Sequential(
            nn.LayerNorm(bank_dim), nn.Linear(bank_dim, bank_dim), nn.GELU(), nn.Linear(bank_dim, bank_dim)
        )
        self.state_to_context_normal = nn.Linear(bank_dim, ctx_len * text_dim)
        self.state_to_context_abnormal = nn.Linear(bank_dim, ctx_len * text_dim)
        self.class_vae = ClassVAE(text_dim, vae_hidden_dim, vae_latent_dim)
        self.class_to_context = nn.Linear(text_dim, ctx_len * text_dim)
        self.gamma_state = BoundedPositiveGate(initial=0.05, maximum=0.20)
        self.gamma_class = BoundedPositiveGate(initial=0.02, maximum=0.10)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.level_embedding, std=0.02)
        nn.init.normal_(self.concept_slots, std=0.02)
        for module in (self.normal_query, self.abnormal_query, self.router_key):
            nn.init.xavier_uniform_(module.weight)

    def concept_keys(self) -> torch.Tensor:
        """Shared factor keys reserved for a later joint visual/text router."""
        return self.router_key(self.concept_slots)

    def _project_levels(self, seg_tokens_pre_l2: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(seg_tokens_pre_l2) != self.n_groups:
            raise ValueError(f"expected {self.n_groups} visual levels, got {len(seg_tokens_pre_l2)}")
        projected = []
        for level, tokens in enumerate(seg_tokens_pre_l2):
            if tokens.ndim != 3 or tokens.shape[-1] != self.text_dim:
                raise ValueError("each visual level must have shape [B, P, 768]")
            projected.append(self.level_projectors[level](tokens.float()) + self.level_embedding[level])
        return torch.stack(projected, dim=0)  # [G, B, P, bank_dim]

    def forward(
        self,
        seg_tokens_pre_l2: Sequence[torch.Tensor],
        cls24: torch.Tensor,
        ctx_normal: torch.Tensor,
        ctx_abnormal: torch.Tensor,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # The semantic state, VAE, and context normalisation intentionally stay
        # FP32 even while the surrounding Phase2B visual forward uses BF16.
        with torch.autocast(device_type=cls24.device.type, enabled=False):
            return self._forward_fp32(seg_tokens_pre_l2, cls24, ctx_normal, ctx_abnormal, debug)

    def _forward_fp32(
        self,
        seg_tokens_pre_l2: Sequence[torch.Tensor],
        cls24: torch.Tensor,
        ctx_normal: torch.Tensor,
        ctx_abnormal: torch.Tensor,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        projected_levels = self._project_levels(seg_tokens_pre_l2)
        groups, batch, patches, _ = projected_levels.shape
        multi_features = projected_levels.permute(1, 0, 2, 3).reshape(batch, groups * patches, self.bank_dim)

        normal_queries = self.normal_query(self.concept_slots).unsqueeze(0).expand(batch, -1, -1)
        abnormal_queries = self.abnormal_query(self.concept_slots).unsqueeze(0).expand(batch, -1, -1)
        normal_attended, normal_maps = self.prototype_attention(
            normal_queries, multi_features, multi_features, need_weights=debug, average_attn_weights=False
        )
        abnormal_attended, abnormal_maps = self.prototype_attention(
            abnormal_queries, multi_features, multi_features, need_weights=debug, average_attn_weights=False
        )
        prototype_normal = normal_queries + self.normal_state_update(normal_attended)
        prototype_abnormal = abnormal_queries + self.abnormal_state_update(abnormal_attended)

        vae = self.class_vae(cls24)
        state_normal = self.state_to_context_normal(prototype_normal).view(
            batch, self.num_factors, self.ctx_len, self.text_dim
        )
        state_abnormal = self.state_to_context_abnormal(prototype_abnormal).view(
            batch, self.num_factors, self.ctx_len, self.text_dim
        )
        state_delta = torch.stack([state_normal, state_abnormal], dim=2)
        state_delta = F.normalize(state_delta.float(), dim=-1)
        class_delta = self.class_to_context(vae["class_semantic"]).view(batch, self.ctx_len, self.text_dim)
        class_delta = F.normalize(class_delta.float(), dim=-1).unsqueeze(1).unsqueeze(1)

        base_context = torch.stack([ctx_normal.float(), ctx_abnormal.float()], dim=0)
        if tuple(base_context.shape) != (2, self.ctx_len, self.text_dim):
            raise ValueError(
                f"base contexts must be [2, {self.ctx_len}, {self.text_dim}], got {tuple(base_context.shape)}"
            )
        base_context = base_context.unsqueeze(0).unsqueeze(0)
        dynamic_contexts = (
            base_context
            + self.gamma_state().view(1, 1, 1, 1, 1) * state_delta
            + self.gamma_class().view(1, 1, 1, 1, 1) * class_delta
        )
        output: Dict[str, torch.Tensor] = {
            "projected_levels": projected_levels,
            "multi_features": multi_features,
            "prototype_normal": prototype_normal,
            "prototype_abnormal": prototype_abnormal,
            "dynamic_contexts": dynamic_contexts,
            "concept_keys": self.concept_keys(),
            "gamma_state": self.gamma_state(),
            "gamma_class": self.gamma_class(),
            **vae,
        }
        if debug:
            output["normal_attention"] = normal_maps
            output["abnormal_attention"] = abnormal_maps
        return output
