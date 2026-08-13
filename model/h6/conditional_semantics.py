"""Conditional semantic synthesis for the Phase 4 K1 scientific path.

This module is intentionally independent of the legacy factor bank and patch
Router. It produces one image-conditioned Normal/Abnormal context pair and
uses exactly one conditioning formulation: context deltas added to the current
Phase2B soft context.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .semantic_bank import BoundedPositiveGate, ClassVAE


class ConditionalSemanticCore(nn.Module):
    """Build one CoPS N/A state pair plus an optional VAE class context."""

    def __init__(
        self,
        n_groups: int,
        bank_dim: int = 256,
        text_dim: int = 768,
        ctx_len: int = 4,
        vae_hidden_dim: int = 512,
        vae_latent_dim: int = 256,
        vae_class_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        if bank_dim % 4 != 0:
            raise ValueError("bank_dim must be divisible by four attention heads")
        self.n_groups = int(n_groups)
        self.bank_dim = int(bank_dim)
        self.text_dim = int(text_dim)
        self.ctx_len = int(ctx_len)

        self.level_projectors = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(text_dim, bank_dim), nn.LayerNorm(bank_dim))
                for _ in range(n_groups)
            ]
        )
        self.level_embedding = nn.Parameter(torch.empty(n_groups, bank_dim))
        self.normal_query = nn.Parameter(torch.empty(1, bank_dim))
        self.abnormal_query = nn.Parameter(torch.empty(1, bank_dim))
        self.prototype_attention = nn.MultiheadAttention(
            bank_dim, num_heads=4, dropout=0.0, batch_first=True
        )
        self.normal_state_update = nn.Sequential(
            nn.LayerNorm(bank_dim),
            nn.Linear(bank_dim, bank_dim),
            nn.GELU(),
            nn.Linear(bank_dim, bank_dim),
        )
        self.abnormal_state_update = nn.Sequential(
            nn.LayerNorm(bank_dim),
            nn.Linear(bank_dim, bank_dim),
            nn.GELU(),
            nn.Linear(bank_dim, bank_dim),
        )
        self.state_to_context_normal = nn.Linear(bank_dim, ctx_len * text_dim)
        self.state_to_context_abnormal = nn.Linear(bank_dim, ctx_len * text_dim)

        # CLS24 is a visual representation. The VAE may model it, but its
        # result reaches CLIP text space only through this learned bridge; it
        # is never inserted as a standalone raw token.
        self.class_vae = ClassVAE(
            text_dim, vae_hidden_dim, vae_latent_dim, class_ratio=vae_class_ratio
        )
        self.class_to_context = nn.Linear(text_dim, ctx_len * text_dim)
        self.gamma_state = BoundedPositiveGate(initial=0.05, maximum=0.20)
        self.gamma_class = BoundedPositiveGate(initial=0.02, maximum=0.10)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.level_embedding, std=0.02)
        nn.init.normal_(self.normal_query, std=0.02)
        nn.init.normal_(self.abnormal_query, std=0.02)

    def _project_levels(self, seg_tokens_pre_l2: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(seg_tokens_pre_l2) != self.n_groups:
            raise ValueError(
                f"expected {self.n_groups} visual levels, got {len(seg_tokens_pre_l2)}"
            )
        projected = []
        for level, tokens in enumerate(seg_tokens_pre_l2):
            if tokens.ndim != 3 or tokens.shape[-1] != self.text_dim:
                raise ValueError(
                    f"level {level} must be [B,P,{self.text_dim}], got {tuple(tokens.shape)}"
                )
            projected.append(
                self.level_projectors[level](tokens.float()) + self.level_embedding[level]
            )
        return torch.stack(projected, dim=0)

    def forward(
        self,
        seg_tokens_pre_l2: Sequence[torch.Tensor],
        cls24: torch.Tensor,
        ctx_normal: torch.Tensor,
        ctx_abnormal: torch.Tensor,
        *,
        state_scale: float = 1.0,
        class_scale: float = 1.0,
        debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        with torch.autocast(device_type=cls24.device.type, enabled=False):
            projected = self._project_levels(seg_tokens_pre_l2)
            groups, batch, patches, _ = projected.shape
            multi_features = projected.permute(1, 0, 2, 3).reshape(
                batch, groups * patches, self.bank_dim
            )
            normal_query = self.normal_query.unsqueeze(0).expand(batch, -1, -1)
            abnormal_query = self.abnormal_query.unsqueeze(0).expand(batch, -1, -1)
            normal_attended, normal_attention = self.prototype_attention(
                normal_query,
                multi_features,
                multi_features,
                need_weights=debug,
                average_attn_weights=False,
            )
            abnormal_attended, abnormal_attention = self.prototype_attention(
                abnormal_query,
                multi_features,
                multi_features,
                need_weights=debug,
                average_attn_weights=False,
            )
            prototype_normal = normal_query + self.normal_state_update(normal_attended)
            prototype_abnormal = abnormal_query + self.abnormal_state_update(abnormal_attended)

            state_normal_raw = self.state_to_context_normal(prototype_normal).view(
                batch, self.ctx_len, self.text_dim
            )
            state_abnormal_raw = self.state_to_context_abnormal(prototype_abnormal).view(
                batch, self.ctx_len, self.text_dim
            )
            state_delta_raw = torch.stack([state_normal_raw, state_abnormal_raw], dim=1)
            state_delta = F.normalize(state_delta_raw.float(), dim=-1)

            vae = self.class_vae(cls24.detach())
            class_delta_raw = self.class_to_context(vae["class_semantic"]).view(
                batch, self.ctx_len, self.text_dim
            )
            class_delta = F.normalize(class_delta_raw.float(), dim=-1)

            base_context = torch.stack(
                [ctx_normal.float(), ctx_abnormal.float()], dim=0
            )
            expected_shape = (2, self.ctx_len, self.text_dim)
            if tuple(base_context.shape) != expected_shape:
                raise ValueError(
                    f"base contexts must be {expected_shape}, got {tuple(base_context.shape)}"
                )
            base_context = base_context.unsqueeze(0).expand(batch, -1, -1, -1)
            dynamic_contexts = (
                base_context
                + float(state_scale) * self.gamma_state().view(1, 1, 1, 1) * state_delta
                + float(class_scale)
                * self.gamma_class().view(1, 1, 1, 1)
                * class_delta.unsqueeze(1)
            )

            output: Dict[str, torch.Tensor] = {
                "projected_levels": projected,
                "multi_features": multi_features,
                "prototype_normal": prototype_normal,
                "prototype_abnormal": prototype_abnormal,
                # Singleton semantic-bank dimension expected by the shared
                # dynamic text encoder: [B,1,2,C,D].
                "dynamic_contexts": dynamic_contexts.unsqueeze(1),
                "base_contexts": base_context.unsqueeze(1),
                "state_delta_raw": state_delta_raw,
                "state_delta": state_delta,
                "class_delta_raw": class_delta_raw,
                "class_delta": class_delta,
                "gamma_state": self.gamma_state(),
                "gamma_class": self.gamma_class(),
                "state_scale": torch.tensor(float(state_scale), device=cls24.device),
                "class_scale": torch.tensor(float(class_scale), device=cls24.device),
                **vae,
            }
            if debug:
                output["normal_attention"] = normal_attention
                output["abnormal_attention"] = abnormal_attention
            return output


def predictor_aligned_abnormal_residual(
    base_group_logits: torch.Tensor,
    dynamic_abnormal_logits: torch.Tensor,
    rho: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Apply the exact bounded abnormal-only residual in the base DFG frame."""
    if base_group_logits.ndim != 4 or base_group_logits.shape[-1] != 2:
        raise ValueError("base_group_logits must be [G,B,P,2]")
    if dynamic_abnormal_logits.shape != base_group_logits.shape[:-1]:
        raise ValueError("dynamic abnormal logits must be [G,B,P]")
    if rho.shape != (base_group_logits.shape[0],):
        raise ValueError("rho must contain one scalar per visual group")
    residual = dynamic_abnormal_logits - base_group_logits[..., 1].detach()
    final = torch.stack(
        [
            base_group_logits[..., 0],
            base_group_logits[..., 1]
            + rho.view(-1, 1, 1).to(base_group_logits.dtype) * residual,
        ],
        dim=-1,
    )
    return {
        "predictor_residual_logits": residual,
        "final_group_logits": final,
        "normal_invariant_error": (final[..., 0] - base_group_logits[..., 0]).abs().max(),
    }
