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


def deterministic_slot_directions(
    num_slots: int,
    dim: int,
    seed: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate fixed slot directions without consuming global RNG state."""
    if num_slots < 1 or dim < 1:
        raise ValueError("num_slots and dim must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(dim, num_slots, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    directions = q.T.contiguous()
    return F.normalize(directions.to(device=device, dtype=dtype), dim=-1)


def apply_relative_slot_offsets_(
    parameter: torch.Tensor,
    *,
    scale: float,
    seed: int,
    fallback_std: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """Make slot rows slightly distinct while preserving a shared base.

    ``scale`` is interpreted as perturbation_norm / base_norm.  If the shared
    base is near zero, use a documented fallback norm equivalent to a normal
    vector with ``fallback_std`` per dimension.
    """
    if parameter.ndim != 2:
        raise ValueError("slot parameter must be [num_slots, dim]")
    with torch.no_grad():
        base = parameter.detach().mean(dim=0, keepdim=True)
        base_norm = base.float().norm().clamp_min(0.0)
        if float(base_norm.item()) < 1e-8:
            base_norm = torch.tensor(
                math.sqrt(parameter.shape[1]) * float(fallback_std),
                device=parameter.device,
                dtype=torch.float32,
            )
        directions = deterministic_slot_directions(
            parameter.shape[0],
            parameter.shape[1],
            seed,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        offsets = directions * (base_norm.to(parameter.dtype) * float(scale))
        parameter.copy_(base.to(parameter.dtype).expand_as(parameter) + offsets)
        values = parameter.detach().float()
        cosine = F.normalize(values, dim=-1) @ F.normalize(values, dim=-1).T
        offdiag_mask = ~torch.eye(values.shape[0], device=values.device, dtype=torch.bool)
        l2 = torch.cdist(values, values)
        offdiag_l2 = l2[offdiag_mask]
        return {
            "slot_initial_cos_mean": cosine[offdiag_mask].mean().detach(),
            "slot_initial_cos_max": cosine[offdiag_mask].abs().max().detach(),
            "slot_initial_l2_min": offdiag_l2.min().detach(),
            "slot_initial_relative_scale": (offsets.float().norm(dim=-1).mean() / base_norm).detach(),
        }


def deterministic_xavier_uniform_(parameter: torch.Tensor, seed: int) -> None:
    """Xavier-uniform init using a local CPU generator, not global RNG."""
    if parameter.ndim < 2:
        raise ValueError("xavier init expects at least 2-D tensor")
    fan_in = parameter.shape[1]
    fan_out = parameter.shape[0]
    bound = math.sqrt(6.0 / float(fan_in + fan_out))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = torch.rand(parameter.shape, generator=generator, dtype=torch.float32)
    values = values.mul(2.0).sub(1.0).mul(bound)
    with torch.no_grad():
        parameter.copy_(values.to(device=parameter.device, dtype=parameter.dtype))


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

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, latent_dim: int = 256, class_ratio: float = 0.25):
        super().__init__()
        if not 0.0 <= float(class_ratio) <= 1.0:
            raise ValueError("class_ratio must be in [0, 1]")
        self.class_ratio = float(class_ratio)
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
        reconstruction_sample = self.decoder(z).float()
        decoded_mu = self.decoder(mu).float()
        decoded_semantic = F.normalize(decoded_mu, dim=-1)
        cls_semantic = F.normalize(target, dim=-1)
        class_semantic = F.normalize(
            (1.0 - self.class_ratio) * cls_semantic + self.class_ratio * decoded_semantic,
            dim=-1,
        )
        reconstruction = F.mse_loss(reconstruction_sample, target, reduction="mean")
        # Scalar KL units: sum over latent dimensions, then mean over batch.
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        return {
            "class_semantic": class_semantic,
            "decoded_mu": decoded_mu,
            "decoded_semantic": decoded_semantic,
            "cls_semantic": cls_semantic.detach(),
            "reconstruction_sample": reconstruction_sample,
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
        vae_class_ratio: float = 0.25,
        slot_init_enabled: bool = False,
        slot_init_scale: float = 0.02,
        slot_init_seed_offset: int = 6100,
        slot_init_method: str = "qr_relative_offset",
        late_factor_identity_enabled: bool = False,
        factor_id_scale: float = 0.02,
        factor_id_max_ratio: float = 0.05,
    ):
        super().__init__()
        if bank_dim % 4 != 0:
            raise ValueError("bank_dim must be divisible by four attention heads")
        self.n_groups = int(n_groups)
        self.num_factors = int(num_factors)
        self.bank_dim = int(bank_dim)
        self.text_dim = int(text_dim)
        self.ctx_len = int(ctx_len)
        self.slot_init_enabled = bool(slot_init_enabled)
        self.slot_init_scale = float(slot_init_scale)
        self.slot_init_seed_offset = int(slot_init_seed_offset)
        self.slot_init_method = str(slot_init_method)
        self.late_factor_identity_enabled = bool(late_factor_identity_enabled)
        self.factor_id_scale = float(factor_id_scale)
        self.factor_id_max_ratio = float(factor_id_max_ratio)
        self.slot_init_applied_components: list[str] = []
        self._slot_initial_diagnostics: Dict[str, torch.Tensor] = {}
        self.register_buffer(
            "factor_id_directions",
            deterministic_slot_directions(num_factors, bank_dim, self.slot_init_seed_offset),
        )

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
        self.factor_id_projection = nn.Linear(bank_dim, text_dim, bias=False)
        self.class_vae = ClassVAE(text_dim, vae_hidden_dim, vae_latent_dim, class_ratio=vae_class_ratio)
        self.class_to_context = nn.Linear(text_dim, ctx_len * text_dim)
        self.gamma_state = BoundedPositiveGate(initial=0.05, maximum=0.20)
        self.gamma_class = BoundedPositiveGate(initial=0.02, maximum=0.10)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.level_embedding, std=0.02)
        nn.init.normal_(self.concept_slots, std=0.02)
        for module in (self.normal_query, self.abnormal_query, self.router_key):
            nn.init.xavier_uniform_(module.weight)
        deterministic_xavier_uniform_(self.factor_id_projection.weight, self.slot_init_seed_offset + 31)
        self.slot_init_applied_components = []
        self._slot_initial_diagnostics = {}
        if self.slot_init_enabled:
            diagnostics = apply_relative_slot_offsets_(
                self.concept_slots,
                scale=self.slot_init_scale,
                seed=self.slot_init_seed_offset,
            )
            self.slot_init_applied_components.append("semantic_core.concept_slots")
            self._slot_initial_diagnostics = diagnostics

    def apply_late_factor_identity(self, state_delta_raw: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Add small factor-ID residual after state_to_context and before text encoding.

        state_delta_raw is [B,M,2,ctx_len,D].  The same factor identity m is
        broadcast across normal/abnormal state and context-token positions.
        Residual scale is relative to a detached per-token base norm.
        """
        zero = state_delta_raw.float().sum() * 0.0
        if not self.late_factor_identity_enabled:
            return state_delta_raw, {
                "factor_id_residual_norm_mean": zero.detach(),
                "factor_id_residual_norm_max": zero.detach(),
                "factor_id_residual_to_context_ratio_mean": zero.detach(),
                "factor_id_residual_to_context_ratio_max": zero.detach(),
            }
        directions = self.factor_id_directions.to(device=state_delta_raw.device, dtype=state_delta_raw.dtype)
        factor_identity = F.normalize(self.factor_id_projection(directions).float(), dim=-1)
        factor_identity = factor_identity.view(1, self.num_factors, 1, 1, self.text_dim)
        base_norm = state_delta_raw.detach().float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        residual = float(self.factor_id_scale) * base_norm * factor_identity
        max_norm = float(self.factor_id_max_ratio) * base_norm
        residual_norm = residual.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        residual = residual * torch.clamp(max_norm / residual_norm, max=1.0)
        ratio = residual.float().norm(dim=-1, keepdim=True) / base_norm
        return state_delta_raw + residual.to(state_delta_raw.dtype), {
            "factor_id_residual_norm_mean": residual.float().norm(dim=-1).mean().detach(),
            "factor_id_residual_norm_max": residual.float().norm(dim=-1).max().detach(),
            "factor_id_residual_to_context_ratio_mean": ratio.mean().detach(),
            "factor_id_residual_to_context_ratio_max": ratio.max().detach(),
        }

    def concept_keys(self) -> torch.Tensor:
        """Shared factor keys reserved for a later joint visual/text router."""
        return self.router_key(self.concept_slots)

    def initialization_diagnostics(self) -> Dict[str, torch.Tensor]:
        values = self.concept_slots.detach().float()
        normalized = F.normalize(values, dim=-1)
        cosine = normalized @ normalized.T
        offdiag_mask = ~torch.eye(values.shape[0], device=values.device, dtype=torch.bool)
        l2 = torch.cdist(values, values)[offdiag_mask]
        storage_offsets = torch.tensor(
            [values[i].storage_offset() for i in range(values.shape[0])],
            device=values.device,
            dtype=torch.long,
        )
        out = {
            "slot_initial_cos_mean": cosine[offdiag_mask].mean().detach(),
            "slot_initial_cos_max": cosine[offdiag_mask].abs().max().detach(),
            "slot_initial_l2_min": l2.min().detach(),
            "slot_residual_norm": values.norm(dim=-1).detach(),
            "slot_storage_offsets": storage_offsets.detach(),
        }
        out.update({k: v.detach() for k, v in self._slot_initial_diagnostics.items()})
        return out

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
        state_delta_raw = torch.stack([state_normal, state_abnormal], dim=2)
        state_delta_with_identity, factor_id_diagnostics = self.apply_late_factor_identity(state_delta_raw)
        state_delta = F.normalize(state_delta_with_identity.float(), dim=-1)
        class_delta_raw = self.class_to_context(vae["class_semantic"]).view(batch, self.ctx_len, self.text_dim)
        class_delta = F.normalize(class_delta_raw.float(), dim=-1).unsqueeze(1).unsqueeze(1)

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
            "concept_slots": self.concept_slots,
            "normal_queries": normal_queries,
            "abnormal_queries": abnormal_queries,
            "state_delta_raw": state_delta_raw,
            "state_delta_with_identity": state_delta_with_identity,
            "state_delta": state_delta,
            "late_factor_identity_enabled": torch.tensor(
                self.late_factor_identity_enabled, device=state_delta.device
            ),
            "factor_id_scale": torch.tensor(self.factor_id_scale, device=state_delta.device),
            "factor_id_max_ratio": torch.tensor(self.factor_id_max_ratio, device=state_delta.device),
            **factor_id_diagnostics,
            "class_delta_raw": class_delta_raw,
            "class_delta": class_delta,
            "gamma_state": self.gamma_state(),
            "gamma_class": self.gamma_class(),
            **vae,
        }
        if debug:
            output["normal_attention"] = normal_maps
            output["abnormal_attention"] = abnormal_maps
        return output
