"""Progress 1 integration: one dynamic semantic bank shared by both paths."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .losses import (
    compute_final_expert_hard_cosine,
    dynamic_residual_diagnostics,
    dynamic_residual_diversity_loss,
    factor_bank_against_reference_diagnostics,
    factor_bank_comparison_diagnostics,
    factor_stage_diagnostics,
)
from .router import PatchRouter
from .semantic_bank import BoundedPositiveGate, CoPSSemanticCore, deterministic_slot_directions
from .paired_experts import FOFSPairedSemanticExperts


class H6Progress1(nn.Module):
    """The complete Progress 1 semantic core, without visual experts.

    ``build_batch`` deliberately accepts the owning ACDCLIP model.  H6 owns the
    image-conditioned state and router, while ACDCLIP remains the sole owner of
    the Phase2B text encoder and hard-prompt helpers.  That keeps a single text
    bank and avoids a second visual forward for dynamic prompts.
    """

    progress = 1

    def __init__(
        self,
        n_groups: int,
        num_factors: int = 4,
        top_k: int = 2,
        prediction_routing: str = "dense",
        bank_dim: int = 256,
        router_dim: int = 128,
        router_temperature: float = 1.0,
        router_soft_epochs: int = 2,
        sparse_transition_epochs: int = 1,
        load_bias_enabled: bool = False,
        load_bias_momentum: float = 0.9,
        load_bias_step: float = 0.001,
        load_bias_max: float = 0.03,
        vae_hidden_dim: int = 512,
        vae_latent_dim: int = 256,
        vae_class_ratio: float = 0.25,
        slot_init_enabled: bool = False,
        slot_init_scale: float = 0.02,
        slot_init_seed_offset: int = 6100,
        slot_init_method: str = "qr_relative_offset",
        factor_grad_diagnostics: bool = False,
        diagnostics_mode: str = "light",
        diagnostics_interval: int = 1,
        test_rho_override: float | None = None,
        late_factor_identity_enabled: bool = False,
        factor_id_scale: float = 0.02,
        factor_id_max_ratio: float = 0.05,
        router_query_mode: str = "local_global_bypass",
        router_query_global_weight: float = 0.10,
        router_local_bypass_scale: float = 0.10,
        router_local_bypass_max_ratio: float = 0.20,
        router_local_projection_seed_offset: int = 7200,
        router_key_anchor_enabled: bool = True,
        router_key_anchor_seed_offset: int = 7300,
        router_key_adaptation_initial_ratio: float = 0.10,
        router_key_adaptation_max_ratio: float = 0.25,
        factor_context_anchor_enabled: bool = True,
        factor_context_anchor_seed_offset: int = 7400,
        factor_context_adaptation_initial_ratio: float = 0.10,
        factor_context_adaptation_max_ratio: float = 0.25,
        factor_identity_tangent_projection_enabled: bool = True,
        factor_generator_specialization_enabled: bool = False,
        factor_head_init_scale: float = 1e-3,
        factor_local_dynamic_mix: float = 0.0,
        lambda_dynamic_mean_anchor: float = 0.001,
        dynamic_mean_anchor_min_cosine: float = 0.70,
        dynamic_mean_anchor_start_epoch: int = 4,
        dynamic_mean_anchor_warmup_epochs: int = 3,
        router_teacher_mode: str = "raw_cosine",
        progress_version: str = "P1-v6",
        local_factor_mode: str = "center_spread",
        local_center_mix: float = 0.05,
        local_factor_spread: float = 0.10,
        expert_enabled: bool = False,
        expert_bottleneck: int = 64,
        expert_fofs_seed_offset: int = 7500,
        expert_state_condition_scale: float = 0.25,
        expert_scale_target: float = 0.10,
        expert_scale_start_epoch: int = 1,
        expert_scale_warmup_epochs: int = 6,
        expert_max_relative_ratio: float = 0.10,
        text_dim: int = 768,
        ctx_len: int = 4,
        h6_logit_temperature: float = 10.0,
        cluster_responsibility_enabled: bool = False,
        cluster_temperature: float = 0.10,
        **kwargs,
    ):
        super().__init__()
        self.n_groups = int(n_groups)
        self.num_factors = int(num_factors)
        self.top_k = int(top_k)
        self.bank_dim = int(bank_dim)
        self.router_dim = int(router_dim)
        self.router_temperature = float(router_temperature)
        self.router_soft_epochs = int(router_soft_epochs)
        self.sparse_transition_epochs = max(1, int(sparse_transition_epochs))
        self.load_bias_enabled = bool(load_bias_enabled)
        self.load_bias_momentum = float(load_bias_momentum)
        self.load_bias_step = float(load_bias_step)
        self.load_bias_max = float(load_bias_max)
        self.vae_hidden_dim = int(vae_hidden_dim)
        self.vae_latent_dim = int(vae_latent_dim)
        self.vae_class_ratio = float(vae_class_ratio)
        self.slot_init_enabled = bool(slot_init_enabled)
        self.slot_init_scale = float(slot_init_scale)
        self.slot_init_seed_offset = int(slot_init_seed_offset)
        self.slot_init_method = str(slot_init_method)
        self.factor_grad_diagnostics = bool(factor_grad_diagnostics)
        self.diagnostics_mode = str(diagnostics_mode)
        self.diagnostics_interval = int(diagnostics_interval)
        self.late_factor_identity_enabled = bool(late_factor_identity_enabled)
        self.factor_id_scale = float(factor_id_scale)
        self.factor_id_max_ratio = float(factor_id_max_ratio)
        self.router_query_mode = str(router_query_mode)
        self.router_query_global_weight = float(router_query_global_weight)
        self.router_local_bypass_scale = float(router_local_bypass_scale)
        self.router_local_bypass_max_ratio = float(router_local_bypass_max_ratio)
        self.router_local_projection_seed_offset = int(router_local_projection_seed_offset)
        self.router_key_anchor_enabled = bool(router_key_anchor_enabled)
        self.router_key_anchor_seed_offset = int(router_key_anchor_seed_offset)
        self.router_key_adaptation_initial_ratio = float(router_key_adaptation_initial_ratio)
        self.router_key_adaptation_max_ratio = float(router_key_adaptation_max_ratio)
        self.factor_context_anchor_enabled = bool(factor_context_anchor_enabled)
        self.factor_context_anchor_seed_offset = int(factor_context_anchor_seed_offset)
        self.factor_context_adaptation_initial_ratio = float(factor_context_adaptation_initial_ratio)
        self.factor_context_adaptation_max_ratio = float(factor_context_adaptation_max_ratio)
        self.factor_identity_tangent_projection_enabled = bool(factor_identity_tangent_projection_enabled)
        self.factor_generator_specialization_enabled = bool(factor_generator_specialization_enabled)
        self.factor_head_init_scale = float(factor_head_init_scale)
        self.factor_local_dynamic_mix = float(factor_local_dynamic_mix)
        if not 0.0 <= self.factor_local_dynamic_mix <= 1.0:
            raise ValueError("factor_local_dynamic_mix must be in [0, 1]")
            
        # P1-v8.2 Geometry: explicit runtime contract fields.
        self.local_factor_mode = str(local_factor_mode)
        if self.local_factor_mode not in ("legacy_mix", "center_spread"):
            raise ValueError("local_factor_mode must be legacy_mix or center_spread")
        self.local_center_mix = float(local_center_mix)
        self.local_factor_spread = float(local_factor_spread)
        
        self.lambda_dynamic_mean_anchor = float(lambda_dynamic_mean_anchor)
        self.dynamic_mean_anchor_min_cosine = float(dynamic_mean_anchor_min_cosine)
        self.dynamic_mean_anchor_start_epoch = int(dynamic_mean_anchor_start_epoch)
        self.dynamic_mean_anchor_warmup_epochs = int(dynamic_mean_anchor_warmup_epochs)
        self.router_teacher_mode = str(router_teacher_mode)
        self.progress_version = str(progress_version)
        self.expert_enabled = bool(expert_enabled)
        self.expert_bottleneck = int(expert_bottleneck)
        self.expert_fofs_seed_offset = int(expert_fofs_seed_offset)
        self.expert_state_condition_scale = float(expert_state_condition_scale)
        self.expert_scale_target = float(expert_scale_target)
        self.expert_scale_start_epoch = int(expert_scale_start_epoch)
        self.expert_scale_warmup_epochs = int(expert_scale_warmup_epochs)
        self.expert_max_relative_ratio = float(expert_max_relative_ratio)
        self.text_dim = int(text_dim)
        self.ctx_len = int(ctx_len)
        self.h6_logit_temperature = float(h6_logit_temperature)
        self.cluster_responsibility_enabled = bool(cluster_responsibility_enabled)
        self.cluster_temperature = float(cluster_temperature)
        if self.cluster_temperature <= 0:
            raise ValueError("cluster_temperature must be positive")
        self.semantic_core = CoPSSemanticCore(
            n_groups=n_groups,
            num_factors=num_factors,
            bank_dim=bank_dim,
            text_dim=text_dim,
            ctx_len=ctx_len,
            vae_hidden_dim=vae_hidden_dim,
            vae_latent_dim=vae_latent_dim,
            vae_class_ratio=vae_class_ratio,
            slot_init_enabled=slot_init_enabled,
            slot_init_scale=slot_init_scale,
            slot_init_seed_offset=slot_init_seed_offset,
            slot_init_method=slot_init_method,
            late_factor_identity_enabled=late_factor_identity_enabled,
            factor_id_scale=factor_id_scale,
            factor_id_max_ratio=factor_id_max_ratio,
            factor_context_anchor_enabled=factor_context_anchor_enabled,
            factor_context_anchor_seed_offset=factor_context_anchor_seed_offset,
            factor_context_adaptation_initial_ratio=factor_context_adaptation_initial_ratio,
            factor_context_adaptation_max_ratio=factor_context_adaptation_max_ratio,
            factor_identity_tangent_projection_enabled=factor_identity_tangent_projection_enabled,
            factor_generator_specialization_enabled=factor_generator_specialization_enabled,
            factor_head_init_scale=factor_head_init_scale,
        )
        self.router = PatchRouter(
            n_groups=n_groups,
            num_factors=num_factors,
            text_dim=text_dim,
            bank_dim=bank_dim,
            hidden_dim=router_dim,
            prediction_routing=prediction_routing,
            temperature=router_temperature,
            soft_routing_epochs=router_soft_epochs,
            sparse_transition_epochs=sparse_transition_epochs,
            top_k=top_k,
            load_bias_enabled=load_bias_enabled,
            load_bias_momentum=load_bias_momentum,
            load_bias_step=load_bias_step,
            load_bias_max=load_bias_max,
            slot_init_enabled=slot_init_enabled,
            slot_init_scale=slot_init_scale,
            slot_init_seed_offset=slot_init_seed_offset,
            router_query_mode=router_query_mode,
            router_query_global_weight=router_query_global_weight,
            router_local_bypass_scale=router_local_bypass_scale,
            router_local_bypass_max_ratio=router_local_bypass_max_ratio,
            router_local_projection_seed_offset=router_local_projection_seed_offset,
            router_key_anchor_enabled=router_key_anchor_enabled,
            router_key_anchor_seed_offset=router_key_anchor_seed_offset,
            router_key_adaptation_initial_ratio=router_key_adaptation_initial_ratio,
            router_key_adaptation_max_ratio=router_key_adaptation_max_ratio,
        )
        self.paired_experts = FOFSPairedSemanticExperts(
            num_factors=num_factors, text_dim=text_dim, bank_dim=bank_dim,
            bottleneck=expert_bottleneck, seed_offset=expert_fofs_seed_offset,
            state_condition_scale=expert_state_condition_scale,
            max_relative_ratio=expert_max_relative_ratio,
        ) if self.expert_enabled else None
        self.rho = BoundedPositiveGate(initial=0.05, maximum=0.50, count=n_groups)
        # Keep the legacy raw state key for checkpoint compatibility, but make
        # the canonical correction a fixed .05 residual from the first step.
        self.rho.raw.requires_grad_(False)
        with torch.no_grad():
            self.rho.raw.fill_(float(torch.logit(torch.tensor(0.05 / 0.50))))
        # Persistent Tier-3 provenance.  Empty buffers preserve the Tier-2
        # path and are populated only by bind_cluster_centroids().
        self.register_buffer("cluster_centroids", torch.empty(0, text_dim))
        self.register_buffer("cluster_identity", torch.empty(0, bank_dim))
        self.register_buffer(
            "cluster_identity_projection",
            deterministic_slot_directions(bank_dim, text_dim, slot_init_seed_offset + 8100).T.contiguous(),
        )
        self.test_rho_override = test_rho_override
        self.epoch_one_based = 1

    def config_dict(self) -> Dict[str, int | float | str]:
        return {
            "variant": "p1_v7_full_fofs_paired_semantic_moe" if self.expert_enabled else "p1_v6_structural_specialization",
            "progress_version": self.progress_version,
            "progress": self.progress,
            "n_groups": self.n_groups,
            "num_factors": self.num_factors,
            "top_k": self.top_k,
            "bank_dim": self.bank_dim,
            "router_dim": self.router_dim,
            "router_temperature": self.router_temperature,
            "router_soft_epochs": self.router_soft_epochs,
            "dense_routing_epochs": self.router_soft_epochs,
            "sparse_start_epoch": self.router_soft_epochs + 1,
            "sparse_transition_epochs": self.sparse_transition_epochs,
            "sparse_full_epoch": self.router_soft_epochs + self.sparse_transition_epochs,
            "sparse_mode": "straight_through_topk",
            "prediction_interpolation_enabled": True,
            "router_mode": "concept_key_dot",
            "router_scoring": "concept_key_dot",
            "load_bias_enabled": self.load_bias_enabled,
            "load_bias_momentum": self.load_bias_momentum,
            "load_bias_step": self.load_bias_step,
            "load_bias_max": self.load_bias_max,
            "load_bias_selection_only": True,
            "load_bias_within_topk_weights_use_semantic_logits": True,
            "vae_hidden_dim": self.vae_hidden_dim,
            "vae_latent_dim": self.vae_latent_dim,
            "vae_class_ratio": self.vae_class_ratio,
            "text_dim": self.text_dim,
            "ctx_len": self.ctx_len,
            "h6_logit_temperature": self.h6_logit_temperature,
            "rho_init": 0.05,
            "rho_max": 0.05,
            "rho_fixed": True,
            "rho_trainable": False,
            "local_factor_mode": self.local_factor_mode,
            "local_center_mix": self.local_center_mix,
            "local_factor_spread": self.local_factor_spread,
            "dynamic_text_normalized": True,
            "text_fusion_norm": "pre_fusion_l2",
            "anchor_encoder_mode": "frozen",
            "frozen_anchor_mode": "functional_layer_norm_no_adapter",
            "diversity_target": "dynamic_residual",
            "center_factor_aware": True,
            "center_assignment_detached": True,
            "center_loss": "factor_aware_dense_detached",
            "kl_schedule": "zero_then_linear",
            "vae_prompt_use_mu": True,
            "vae_sample_used_for_reconstruction_only": True,
            "vae_class_skip_enabled": True,
            "vae_prompt_path": "decoder_mu",
            "structured_text_layout": "[C1][C2][C3][C4][STATE_m][CLASS][literal_state][REAL_NAME]",
            "structured_text_enabled": self.progress_version == "P1-v8.3",
            "dynamic_text_adapt_text": self.progress_version == "P1-v8.3",
            "state_token_factor_specific": True,
            "utility_denominator_floor": 0.10,
            "tau_utility": 0.05,
            "utility_gain_threshold": 0.02,
            "utility_entropy_threshold": 0.98,
            "exploration_schedule": [0.15, 0.05],
            "utility_teacher_detached": True,
            "dense_router_only": self.progress_version == "P1-v8.3",
            "class_token_deterministic_decoder_mu": True,
            "slot_init_enabled": self.slot_init_enabled,
            "slot_init_scale": self.slot_init_scale,
            "slot_init_seed_offset": self.slot_init_seed_offset,
            "slot_init_method": self.slot_init_method,
            "slot_init_applied_components": list(self.semantic_core.slot_init_applied_components)
            + list(getattr(self.router, "slot_init_applied_components", [])),
            "factor_grad_diagnostics_enabled": self.factor_grad_diagnostics,
            "three_level_router_mode": "shared_router_level_specific_inputs",
            "level_specific_input_verified": True,
            "teacher_diagnostics_version": 1,
            "teacher_confidence_gate": True,
            "teacher_candidate_diagnostics": True,
            "factor_identity_stage_tracing": True,
            "router_granularity_diagnostics": True,
            "query_patchwise_diagnostics": True,
            "late_factor_identity_enabled": self.late_factor_identity_enabled,
            "factor_id_scale": self.factor_id_scale,
            "factor_id_max_ratio": self.factor_id_max_ratio,
            "router_query_mode": self.router_query_mode,
            "router_query_global_weight": self.router_query_global_weight,
            "router_local_bypass_scale": self.router_local_bypass_scale,
            "router_local_bypass_max_ratio": self.router_local_bypass_max_ratio,
            "router_local_projection_method": "qr_semi_orthogonal_buffer",
            "router_local_projection_seed_offset": self.router_local_projection_seed_offset,
            "router_key_anchor_enabled": self.router_key_anchor_enabled,
            "router_key_anchor_method": "qr_orthonormal_rows_buffer",
            "router_key_anchor_seed_offset": self.router_key_anchor_seed_offset,
            "router_key_adaptation_initial_ratio": self.router_key_adaptation_initial_ratio,
            "router_key_adaptation_max_ratio": self.router_key_adaptation_max_ratio,
            "factor_context_anchor_enabled": self.factor_context_anchor_enabled,
            "factor_context_anchor_method": "qr_orthonormal_rows_buffer",
            "factor_context_anchor_seed_offset": self.factor_context_anchor_seed_offset,
            "factor_context_adaptation_initial_ratio": self.factor_context_adaptation_initial_ratio,
            "factor_context_adaptation_max_ratio": self.factor_context_adaptation_max_ratio,
            "factor_identity_tangent_projection_enabled": self.factor_identity_tangent_projection_enabled,
            "factor_generator_specialization_enabled": self.factor_generator_specialization_enabled,
            "factor_head_init_scale": self.factor_head_init_scale,
            "factor_local_dynamic_mix": self.factor_local_dynamic_mix,
            "local_factor_mode": self.local_factor_mode,
            "local_center_mix": self.local_center_mix,
            "local_factor_spread": self.local_factor_spread,
            "lambda_dynamic_mean_anchor": self.lambda_dynamic_mean_anchor,
            "dynamic_mean_anchor_min_cosine": self.dynamic_mean_anchor_min_cosine,
            "dynamic_mean_anchor_start_epoch": self.dynamic_mean_anchor_start_epoch,
            "dynamic_mean_anchor_warmup_epochs": self.dynamic_mean_anchor_warmup_epochs,
            "factor_id_direction_method": "tangent_context_anchor_shared_buffer",
            "factor_id_projection_mode": "shared_linear_bankdim_to_textdim",
            "factor_id_shared_across_states": True,
            "router_teacher_mode": self.router_teacher_mode,
            "router_teacher_center_detached": True,
            "router_teacher_probability_detached": True,
            "teacher_gate_scope": "patch",
            "expert_enabled": self.expert_enabled,
            "expert_bottleneck": self.expert_bottleneck,
            "expert_fofs_seed_offset": self.expert_fofs_seed_offset,
            "expert_state_condition_scale": self.expert_state_condition_scale,
            "expert_scale_target": self.expert_scale_target,
            "expert_scale_start_epoch": self.expert_scale_start_epoch,
            "expert_scale_warmup_epochs": self.expert_scale_warmup_epochs,
            "expert_max_relative_ratio": self.expert_max_relative_ratio,
            "cluster_responsibility_enabled": self.cluster_responsibility_enabled,
            "cluster_temperature": self.cluster_temperature,
            "cluster_identity_tied": bool(self.cluster_centroids.numel()),
            "cluster_centroid_count": int(self.cluster_centroids.shape[0]),
        }

    @property
    def cluster_ready(self) -> bool:
        return bool(self.cluster_centroids.shape == (self.num_factors, self.text_dim))

    @torch.no_grad()
    def bind_cluster_centroids(self, centroids: torch.Tensor) -> None:
        """Bind centroid m to the same m across all Tier-3 identity paths."""
        if centroids.ndim != 2 or tuple(centroids.shape) != (self.num_factors, self.text_dim):
            raise ValueError(
                f"centroids must be [{self.num_factors}, {self.text_dim}], got {tuple(centroids.shape)}"
            )
        if not self.semantic_core.factor_generator_specialization_enabled:
            raise ValueError("Tier-3 requires --h6_factor_generator_specialization_enabled")
        normalized_centroids = F.normalize(
            centroids.detach().to(device=self.cluster_identity_projection.device, dtype=torch.float32), dim=-1
        )
        identity = F.normalize(normalized_centroids @ self.cluster_identity_projection.float(), dim=-1)
        self.cluster_centroids = normalized_centroids.to(
            device=self.cluster_identity_projection.device, dtype=self.cluster_identity_projection.dtype
        )
        self.cluster_identity = identity.to(
            device=self.cluster_identity_projection.device, dtype=self.cluster_identity_projection.dtype
        )
        core = self.semantic_core
        slot_scale = core.concept_slots.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
        factor_scale = core.factor_id_embedding.detach().float().norm(dim=-1).mean().clamp_min(1e-6)
        # The m-th values share one projected centroid identity.  concept_slots
        # feed semantic prototypes and core.router_key(concept_slots), while
        # factor_id_embedding[m] feeds the factor generator before text encode.
        core.concept_slots.copy_(identity.to(core.concept_slots) * slot_scale)
        core.factor_id_embedding.copy_(identity.to(core.factor_id_embedding) * factor_scale)
        core.tier3_cluster_identity_enabled = True

    def load_cluster_centroids(self, path: str) -> dict:
        payload = torch.load(path, map_location="cpu")
        centroids = payload.get("centroids") if isinstance(payload, dict) else payload
        if not torch.is_tensor(centroids):
            raise ValueError("cluster centroid file must contain a tensor or {'centroids': tensor}")
        self.bind_cluster_centroids(centroids)
        return payload.get("metadata", {}) if isinstance(payload, dict) else {}

    def set_epoch(self, epoch_one_based: int) -> None:
        self.epoch_one_based = int(epoch_one_based)

    def rho_cap(self) -> float:
        return 0.05

    def rho_values(self) -> torch.Tensor:
        if not self.training and self.test_rho_override is not None:
            return torch.full_like(self.rho.raw, float(self.test_rho_override))
        return torch.full_like(self.rho.raw, 0.05)

    def expert_scale(self) -> float:
        if not self.expert_enabled or self.epoch_one_based < self.expert_scale_start_epoch:
            return 0.0
        step = self.epoch_one_based - self.expert_scale_start_epoch + 1
        return min(self.expert_scale_target, self.expert_scale_target * step / max(1, self.expert_scale_warmup_epochs))

    def forward_core(self, visual_output: Dict[str, torch.Tensor], ctx_normal: torch.Tensor, ctx_abnormal: torch.Tensor, debug: bool = False) -> Dict[str, torch.Tensor]:
        return self.semantic_core(
            visual_output["seg_tokens_pre_l2"],
            visual_output["cls24"],
            ctx_normal,
            ctx_abnormal,
            debug=debug,
        )

    @staticmethod
    def _batch_hard_embeddings(base_model, dataset_name: str, class_names: Sequence[str], device: torch.device):
        # Local import avoids an adapter -> h6 -> utils import cycle during module import.
        from utils import get_hard_anchor_single_class_text_embedding, get_hard_phase1_single_class_text_embedding

        adapted_cache = {}
        frozen_cache = {}
        adapted = []
        frozen = []
        for class_name in class_names:
            if class_name not in adapted_cache:
                adapted_cache[class_name] = get_hard_phase1_single_class_text_embedding(
                    base_model, dataset_name, class_name, device, adapt_text=True
                )
                frozen_cache[class_name] = get_hard_anchor_single_class_text_embedding(
                    base_model, dataset_name, class_name, device
                )
            adapted.append(adapted_cache[class_name])
            frozen.append(frozen_cache[class_name])
        hard_adapted = torch.stack(adapted, dim=1).float()  # [G,B,D,2]
        hard_frozen = torch.stack(frozen, dim=1).detach().float()  # [G,B,D,2]
        return hard_adapted, hard_frozen

    def _encode_dynamic_bank(
        self,
        base_model,
        dataset_name: str,
        class_names: Sequence[str],
        dynamic_contexts: torch.Tensor,
        state_tokens: torch.Tensor | None = None,
        class_token: torch.Tensor | None = None,
        structured: bool = False,
        return_raw: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from utils import get_real_name, get_soft_prompt_sentence, get_structured_prompt_sentence
        from model.tokenizer import tokenize

        batch, factors, states, ctx_len, text_dim = dynamic_contexts.shape
        if factors != self.num_factors or states != 2 or ctx_len != self.ctx_len or text_dim != self.text_dim:
            raise ValueError("dynamic contexts do not match the H6 configuration")
        sentences = []
        for class_name in class_names:
            real_name = get_real_name(dataset_name, class_name)
            for _ in range(self.num_factors):
                for state in range(2):
                    sentence_builder = get_structured_prompt_sentence if structured else get_soft_prompt_sentence
                    sentences.append(sentence_builder(real_name, state, self.ctx_len))
        token_ids = tokenize(sentences).to(dynamic_contexts.device)
        contexts = dynamic_contexts.reshape(batch * factors * states, ctx_len, text_dim)
        if structured:
            if state_tokens is None or class_token is None:
                raise ValueError("structured prompts require STATE and CLASS tensors")
            if state_tokens.shape != (batch, factors, states, text_dim):
                raise ValueError("state_tokens must be [B,M,2,D]")
            if class_token.shape != (batch, text_dim):
                raise ValueError("class_token must be [B,D]")
            flat_state = state_tokens.reshape(batch * factors * states, text_dim)
            flat_class = class_token[:, None, None, :].expand(
                batch, factors, states, text_dim
            ).reshape(batch * factors * states, text_dim)
            text_levels = base_model.encode_dynamic_prompt_text(
                token_ids, contexts, flat_state, flat_class, adapt_text=True
            )
        else:
            text_levels = base_model.encode_dynamic_prompt_text(token_ids, contexts, adapt_text=False)
        dynamic = torch.stack(text_levels, dim=0).view(
            self.n_groups, batch, factors, states, self.text_dim
        )
        dynamic_raw = dynamic.permute(0, 1, 2, 4, 3).contiguous().float()
        dynamic = F.normalize(dynamic_raw, dim=3)
        if return_raw:
            return dynamic, dynamic_raw
        return dynamic

    @staticmethod
    def _fuse_factor_bank_legacy(hard_adapted: torch.Tensor, dynamic_text: torch.Tensor, hybrid_alpha: float) -> torch.Tensor:
        if hard_adapted.ndim != 4:
            raise ValueError("hard_adapted must be [G,B,768,2]")
        if dynamic_text.ndim != 5:
            raise ValueError("dynamic_text must be [G,B,M,768,2]")
        hard_adapted = F.normalize(hard_adapted.float(), dim=2)
        dynamic_text = F.normalize(dynamic_text.float(), dim=3)
        mixed = (1.0 - float(hybrid_alpha)) * hard_adapted.unsqueeze(2) + float(hybrid_alpha) * dynamic_text
        return F.normalize(mixed, dim=3)

    @staticmethod
    def _fuse_factor_bank_center_spread(hard_adapted: torch.Tensor, dynamic_text: torch.Tensor, center_mix: float, factor_spread: float) -> torch.Tensor:
        if hard_adapted.ndim != 4:
            raise ValueError("hard_adapted must be [G,B,768,2]")
        if dynamic_text.ndim != 5:
            raise ValueError("dynamic_text must be [G,B,M,768,2]")
        
        # 1. raw factor mean (semantic center)
        raw_factor_mean = dynamic_text.mean(dim=2, keepdim=True)
        # 2. normalize semantic center and local center
        semantic_center_norm = F.normalize(raw_factor_mean.float(), dim=3)
        hard_adapted_norm = F.normalize(hard_adapted.float(), dim=2).unsqueeze(2)
        local_center = (1.0 - float(center_mix)) * hard_adapted_norm + float(center_mix) * semantic_center_norm
        local_center_norm = F.normalize(local_center, dim=3)
        
        # 3. subtract raw factor mean
        residual = dynamic_text.float() - raw_factor_mean.float()
        
        # 4. project residual onto tangent space of normalized local center
        # Tangent projection: R - (R dot C) * C
        dot_product = (residual * local_center_norm).sum(dim=3, keepdim=True)
        tangent_residual = residual - dot_product * local_center_norm
        
        # 5. normalize tangent residual with zero-norm protection
        tangent_norm = tangent_residual.norm(dim=3, keepdim=True).clamp_min(1e-8)
        normalized_tangent = tangent_residual / tangent_norm
        
        # 6. apply factor spread and normalize final factor
        final_factor = local_center_norm + float(factor_spread) * normalized_tangent
        return F.normalize(final_factor, dim=3)

    def dynamic_mean_anchor_loss(
        self,
        dynamic_text: torch.Tensor,
        hard_frozen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dynamic_text.ndim != 5:
            raise ValueError("dynamic_text must be [G,B,M,D,2]")
        mean_dynamic = F.normalize(dynamic_text.float().mean(dim=2), dim=2)
        hard_anchor = hard_frozen.detach().float()
        dynamic_mean_hard_cos = F.cosine_similarity(mean_dynamic, hard_anchor, dim=2)
        loss = F.relu(float(self.dynamic_mean_anchor_min_cosine) - dynamic_mean_hard_cos).pow(2).mean()
        return loss, dynamic_mean_hard_cos.detach()

    def build_batch(
        self,
        base_model,
        dataset_name: str,
        class_names: Sequence[str],
        visual_output: Dict[str, torch.Tensor],
        hybrid_alpha: float,
        debug: bool = False,
        update_load_bias: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Create the unique dynamic factor bank and route every patch through it."""
        if len(class_names) != visual_output["cls24"].shape[0]:
            raise ValueError("class_names must have one entry per image")
        core = self.forward_core(
            visual_output,
            base_model.soft_prompt.ctx_normal,
            base_model.soft_prompt.ctx_abnormal,
            debug=debug,
        )
        structured_text = self.progress_version == "P1-v8.3"
        prompt_contexts = core["structured_contexts"] if structured_text else core["dynamic_contexts"]
        dynamic, dynamic_raw = self._encode_dynamic_bank(
            base_model, dataset_name, class_names, prompt_contexts,
            state_tokens=core["state_tokens"] if structured_text else None,
            class_token=core["class_token"] if structured_text else None,
            structured=structured_text,
            return_raw=True,
        )
        hard_adapted, hard_frozen = self._batch_hard_embeddings(
            base_model, dataset_name, class_names, visual_output["cls24"].device
        )
        hard_adapted = F.normalize(hard_adapted.float(), dim=2)
        hard_frozen = F.normalize(hard_frozen.float(), dim=2)
        dynamic = F.normalize(dynamic.float(), dim=3)
        dynamic_mean_anchor_loss_raw, dynamic_mean_hard_cos = self.dynamic_mean_anchor_loss(dynamic, hard_frozen)
        # Global hard-anchor text remains independent of the factor-local
        # residual blend.  The latter is opt-in and only makes generated
        # factor differences reachable by the local CoPS pathway while the
        # shared global text stays at its hard-anchor blend.
        global_factor_bank = self._fuse_factor_bank_legacy(hard_adapted, dynamic, hybrid_alpha)
        
        if self.local_factor_mode == "center_spread":
            factor_bank = self._fuse_factor_bank_center_spread(hard_adapted, dynamic, self.local_center_mix, self.local_factor_spread)
        else:
            local_factor_mix = max(float(hybrid_alpha), self.factor_local_dynamic_mix)
            factor_bank = self._fuse_factor_bank_legacy(hard_adapted, dynamic, local_factor_mix)
            
        # This is the exact fusion-path no-op, not an assumed raw CLIP bank.
        expected_noop_pre_expert_bank = self._fuse_factor_bank_legacy(hard_adapted, dynamic, hybrid_alpha=0.0)
        anchor = hard_frozen.unsqueeze(2).expand_as(dynamic)
        kg_loss = (1.0 - F.cosine_similarity(dynamic.float(), anchor, dim=3)).mean()
        residual_diversity = dynamic_residual_diversity_loss(dynamic, hard_frozen)
        routing = self.router(
            visual_output["seg_tokens"],
            epoch_one_based=self.epoch_one_based,
            concept_keys=core["concept_keys"],
            update_load_bias=self.training if update_load_bias is None else bool(update_load_bias),
        )
        prediction_probabilities = routing["prediction_probabilities"]
        raw_semantic_keys = core["concept_keys"]
        final_router_keys = routing["final_router_keys"]
        expert_payload = {}
        active_factor_bank = factor_bank
        if self.paired_experts is not None:
            expert_payload = self.paired_experts(
                factor_bank, core["prototype_normal"], core["prototype_abnormal"], self.expert_scale()
            )
            active_factor_bank = expert_payload["expert_factor_bank"]
        local_text = self.router.local_text(prediction_probabilities, active_factor_bank)
        patches = torch.stack(visual_output["seg_tokens"], dim=0).float()
        patches = F.normalize(patches, dim=-1)
        h6_logits = self.h6_logit(patches, local_text)
        
        factor_patch_logits = self.h6_logit(
            patches.unsqueeze(3), active_factor_bank.unsqueeze(2)
        )
        expert_patch_logits = factor_patch_logits if self.paired_experts is not None else None
        expert_diagnostics = {}
        if self.paired_experts is not None:
            delta_raw = expert_payload["expert_delta_raw"].float()
            delta_tangent = expert_payload["expert_delta_tangent"].float()
            delta_applied = expert_payload["expert_applied_delta"].float()
            tangent_norm = delta_tangent.norm(dim=-1)
            valid = tangent_norm > 1e-8
            direction = delta_tangent / tangent_norm.unsqueeze(-1).clamp_min(1e-8)
            gram = torch.einsum("gbmd,gbnd->gbmn", direction, direction)
            offdiag = ~torch.eye(self.num_factors, dtype=torch.bool, device=gram.device)
            pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
            valid_offdiag = pair_valid[..., offdiag]
            pair_values = gram[..., offdiag]
            safe_pair_mean = pair_values[valid_offdiag].mean() if valid_offdiag.any() else gram.sum() * 0.0
            safe_pair_max = pair_values[valid_offdiag].abs().max() if valid_offdiag.any() else gram.sum() * 0.0
            final_direction = active_factor_bank[..., 1].float() - active_factor_bank[..., 0].float()
            final_norm = final_direction.norm(dim=-1)
            final_unit = final_direction / final_norm.unsqueeze(-1).clamp_min(1e-8)
            final_gram = torch.einsum("gbmd,gbnd->gbmn", final_unit, final_unit)
            final_pairs = final_gram[..., offdiag]
            # Shared with anchor loss and the structural trust gate diagnostics.
            pre_state_cos = compute_final_expert_hard_cosine(factor_bank, hard_frozen)
            state_cos = compute_final_expert_hard_cosine(active_factor_bank, hard_frozen)  # [G,B,2]
            final_pre_state_cos = compute_final_expert_hard_cosine(
                active_factor_bank, factor_bank.float().mean(dim=2)
            )
            bank_comparisons = {
                "hard_frozen_vs_pre_expert": factor_bank_against_reference_diagnostics(factor_bank, hard_frozen),
                "hard_adapted_vs_pre_expert": factor_bank_against_reference_diagnostics(factor_bank, hard_adapted),
                "pre_expert_vs_expected_noop": factor_bank_comparison_diagnostics(
                    factor_bank, expected_noop_pre_expert_bank
                ),
                "final_expert_vs_pre_expert": factor_bank_comparison_diagnostics(active_factor_bank, factor_bank),
            }
            expert_diagnostics = {
                "expert_scale_current": torch.tensor(self.expert_scale(), device=patches.device),
                "expert_scale_target": torch.tensor(self.expert_scale_target, device=patches.device),
                "expert_delta_raw_norm_mean": delta_raw.norm(dim=-1).mean().detach(),
                "expert_delta_raw_norm_max": delta_raw.norm(dim=-1).max().detach(),
                "expert_delta_tangent_norm_mean": tangent_norm.mean().detach(),
                "expert_delta_tangent_norm_min": tangent_norm.min().detach(),
                "expert_delta_valid_fraction": valid.float().mean().detach(),
                "expert_delta_tangent_cos_mean": safe_pair_mean.detach(),
                "expert_delta_tangent_cos_max": safe_pair_max.detach(),
                "expert_delta_tangent_l2_min": torch.pdist(delta_tangent.reshape(-1, self.text_dim)).min().detach() if delta_tangent.numel() > self.text_dim else tangent_norm.sum().detach() * 0.0,
                "expert_residual_relative_ratio_mean": expert_payload["expert_relative_ratio"].mean().detach(),
                "expert_residual_relative_ratio_max": expert_payload["expert_relative_ratio"].max().detach(),
                "expert_residual_clamp_fraction": expert_payload["expert_clamp_fraction"].detach(),
                "expert_delta_norm_mean": delta_applied.norm(dim=-1).mean().detach(),
                "expert_direction_cos_mean": safe_pair_mean.detach(),
                "expert_direction_cos_max": safe_pair_max.detach(),
                "final_expert_direction_cos_mean": final_pairs.mean().detach(),
                "final_expert_direction_cos_max": final_pairs.abs().max().detach(),
                "final_expert_direction_l2_min": torch.pdist(final_direction.reshape(-1, self.text_dim)).min().detach(),
                "final_expert_direction_norm_mean": final_norm.mean().detach(),
                "final_expert_orth_raw": (final_pairs + 1.0 / float(self.num_factors - 1)).pow(2).mean().detach(),
                "expert_patch_logit_std_across_experts": expert_patch_logits.float().std(dim=-1, unbiased=False).mean().detach(),
                "expert_patch_logit_std_across_patches": expert_patch_logits.float().std(dim=2, unbiased=False).mean().detach(),
                "expert_patch_logit_pairwise_difference_mean": torch.pdist(expert_patch_logits.float().movedim(-1, 0).reshape(self.num_factors, -1)).mean().detach(),
                "expert_mean_hard_cos": state_cos.detach(),
                "pre_expert_mean_hard_cos_per_level_normal": pre_state_cos[..., 0].mean(dim=1).detach(),
                "pre_expert_mean_hard_cos_per_level_abnormal": pre_state_cos[..., 1].mean(dim=1).detach(),
                "pre_expert_mean_hard_cos_mean": pre_state_cos.mean().detach(),
                "pre_expert_mean_hard_cos_min": pre_state_cos.min().detach(),
                "pre_expert_mean_hard_cos_p05": torch.quantile(pre_state_cos.reshape(-1), 0.05).detach(),
                "final_expert_mean_hard_cos_per_level_normal": state_cos[..., 0].mean(dim=1).detach(),
                "final_expert_mean_hard_cos_per_level_abnormal": state_cos[..., 1].mean(dim=1).detach(),
                "final_expert_mean_hard_cos_mean": state_cos.mean().detach(),
                "final_expert_mean_hard_cos_min": state_cos.min().detach(),
                "final_expert_mean_hard_cos_p05": torch.quantile(state_cos.reshape(-1), 0.05).detach(),
                "final_expert_mean_pre_expert_cos_mean": final_pre_state_cos.mean().detach(),
                "final_expert_mean_pre_expert_cos_min": final_pre_state_cos.min().detach(),
                **{
                    f"{comparison_name}_{metric_name}": metric_value
                    for comparison_name, comparison in bank_comparisons.items()
                    for metric_name, metric_value in comparison.items()
                },
            }
        rho = self.rho_values()
        G, B, P, M = factor_patch_logits.shape
        rho_scaled_factor_correction = factor_patch_logits * rho.view(G, 1, 1, 1).to(factor_patch_logits.dtype)
        rho_scaled_actual_correction = h6_logits * rho.view(G, 1, 1).to(h6_logits.dtype)
        
        return {
            **core,
            **routing,
            "hard_adapted": hard_adapted,
            "hard_frozen": hard_frozen,
            "dynamic_text": dynamic,
            "dynamic_text_raw": dynamic_raw,
            "factor_bank": factor_bank,
            "global_factor_bank": global_factor_bank,
            "expected_noop_pre_expert_bank": expected_noop_pre_expert_bank,
            "active_factor_bank": active_factor_bank,
            "raw_semantic_keys": raw_semantic_keys,
            "final_router_keys": final_router_keys,
            "router_patch_features": routing["router_input_features"],
            "kg_loss": kg_loss,
            "residual_diversity": residual_diversity,
            "dynamic_mean_anchor_loss_raw": dynamic_mean_anchor_loss_raw,
            "dynamic_mean_hard_cos": dynamic_mean_hard_cos,
            "text_global": self.router.aggregate_global(prediction_probabilities, global_factor_bank),
            "topk_indices": routing.get("topk_indices", None),
            "prediction_logits": routing.get("logits", None),
            "prediction_probabilities": routing["prediction_probabilities"],
            "h6_logits": h6_logits,
            "rho": self.rho_values(),
            "expert_scale": torch.tensor(self.expert_scale(), device=patches.device),
            "factor_patch_logits": factor_patch_logits,
            "expert_patch_logits": expert_patch_logits,
            "local_text": local_text,
            **expert_payload,
            "actual_local_text": local_text,
            "rho_scaled_factor_correction": rho_scaled_factor_correction,
            "rho_scaled_actual_correction": rho_scaled_actual_correction,
            "router_diagnostics": {
                **self.router.diagnostics(
                prediction_probabilities,
                dense_probabilities=routing["dense_probabilities"],
                sparse_probabilities=routing["sparse_probabilities"],
                topk_indices=routing["topk_indices"],
                ),
                **self.router.concept_key_diagnostics(final_router_keys),
                "raw_concept_key_cos_mean": routing["raw_concept_key_cos_mean"].detach(),
                "raw_concept_key_cos_max": routing["raw_concept_key_cos_max"].detach(),
                "raw_concept_key_l2_min": routing["raw_concept_key_l2_min"].detach(),
                "final_router_key_cos_mean": routing["final_router_key_cos_mean"].detach(),
                "final_router_key_cos_max": routing["final_router_key_cos_max"].detach(),
                "final_router_key_l2_min": routing["final_router_key_l2_min"].detach(),
                "router_key_adaptation_ratio_mean": routing["router_key_adaptation_ratio_mean"].detach(),
                "router_key_adaptation_ratio_max": routing["router_key_adaptation_ratio_max"].detach(),
                "router_logit_std": routing["logits"].float().std(dim=(1, 2, 3), unbiased=False).detach(),
                "router_prob_std": routing["dense_probabilities"].float().std(dim=(1, 2, 3), unbiased=False).detach(),
                "query_variance": routing["queries"].float().var(dim=(1, 2, 3), unbiased=False).detach(),
                "query_norm": routing["queries"].float().norm(dim=-1).mean(dim=(1, 2)).detach(),
                "sparse_ratio": routing["sparse_ratio"].detach(),
                "load_bias": routing["load_bias"].detach(),
                "ema_topk_usage": routing["ema_topk_usage"].detach(),
                "level_input_alias": routing["level_input_alias"].detach(),
                "level_input_difference": routing["level_input_difference"].detach(),
                "level_query_difference": routing["level_query_difference"].detach(),
                "level_logit_difference": routing["level_logit_difference"].detach(),
                "router_patch_count": routing["router_patch_count"].detach(),
                "router_softmax_dim": routing["router_softmax_dim"].detach(),
                "router_topk_dim": routing["router_topk_dim"].detach(),
                "query_pairwise_cos_mean_across_patches": routing[
                    "query_pairwise_cos_mean_across_patches"
                ].detach(),
                "query_pairwise_cos_max_across_patches": routing[
                    "query_pairwise_cos_max_across_patches"
                ].detach(),
                "query_variance_across_patches": routing["query_variance_across_patches"].detach(),
                "query_effective_rank": routing["query_effective_rank"].detach(),
                "query_singular_value_ratio": routing["query_singular_value_ratio"].detach(),
                "per_factor_logit_std_across_patches": routing[
                    "per_factor_logit_std_across_patches"
                ].detach(),
                "raw_query_pairwise_cos_mean": routing["raw_query_pairwise_cos_mean"].detach(),
                "local_query_pairwise_cos_mean": routing["local_query_pairwise_cos_mean"].detach(),
                "final_query_pairwise_cos_mean": routing["final_query_pairwise_cos_mean"].detach(),
                "raw_query_variance_across_patches": routing["raw_query_variance_across_patches"].detach(),
                "local_query_variance_across_patches": routing["local_query_variance_across_patches"].detach(),
                "final_query_variance_across_patches": routing["final_query_variance_across_patches"].detach(),
                "final_query_effective_rank": routing["final_query_effective_rank"].detach(),
                "final_query_top1_energy_ratio": routing["final_query_top1_energy_ratio"].detach(),
                "local_bypass_norm_mean": routing["local_bypass_norm_mean"].detach(),
                "local_bypass_to_learned_ratio_mean": routing["local_bypass_to_learned_ratio_mean"].detach(),
                "local_bypass_to_learned_ratio_max": routing["local_bypass_to_learned_ratio_max"].detach(),
                **self.semantic_core.initialization_diagnostics(),
                **factor_stage_diagnostics(core["concept_slots"], "stage_concept_slots", factor_dim=0),
                **factor_stage_diagnostics(core["concept_keys"], "stage_concept_keys", factor_dim=0),
                **factor_stage_diagnostics(core["normal_queries"], "stage_normal_queries", factor_dim=1),
                **factor_stage_diagnostics(core["abnormal_queries"], "stage_abnormal_queries", factor_dim=1),
                **factor_stage_diagnostics(core["prototype_normal"], "stage_prototype_normal", factor_dim=1),
                **factor_stage_diagnostics(core["prototype_abnormal"], "stage_prototype_abnormal", factor_dim=1),
                **factor_stage_diagnostics(core["state_tokens"], "structured_STATE", factor_dim=1),
                **factor_stage_diagnostics(core["state_delta_raw"], "stage_state_to_context_raw", factor_dim=1),
                **factor_stage_diagnostics(core["state_delta_with_identity"], "stage_state_to_context_with_identity", factor_dim=1),
                **factor_stage_diagnostics(core["state_delta"], "stage_state_to_context_norm", factor_dim=1),
                **factor_stage_diagnostics(core["dynamic_contexts"], "stage_context_before_encoder", factor_dim=1),
                **factor_stage_diagnostics(dynamic_raw, "stage_dynamic_text_raw", factor_dim=2),
                **factor_stage_diagnostics(dynamic, "stage_dynamic_text_norm", factor_dim=2),
                **dynamic_residual_diagnostics(dynamic, hard_frozen),
                "late_factor_identity_enabled": core["late_factor_identity_enabled"].detach(),
                "factor_generator_specialization_enabled": core[
                    "factor_generator_specialization_enabled"
                ].detach(),
                "factor_generator_id_norm_mean": core["factor_generator_id_norm_mean"].detach(),
                "factor_generator_head_delta_norm_mean": core[
                    "factor_generator_head_delta_norm_mean"
                ].detach(),
                "factor_local_dynamic_mix": torch.tensor(
                    self.factor_local_dynamic_mix, device=patches.device
                ),
                "factor_id_scale": core["factor_id_scale"].detach(),
                "factor_id_max_ratio": core["factor_id_max_ratio"].detach(),
                "factor_id_residual_norm_mean": core["factor_id_residual_norm_mean"].detach(),
                "factor_id_residual_norm_max": core["factor_id_residual_norm_max"].detach(),
                "factor_id_residual_to_context_ratio_mean": core["factor_id_residual_to_context_ratio_mean"].detach(),
                "factor_id_residual_to_context_ratio_max": core["factor_id_residual_to_context_ratio_max"].detach(),
                "factor_context_anchor_cos_mean": core["factor_context_anchor_cos_mean"].detach(),
                "factor_context_anchor_cos_max": core["factor_context_anchor_cos_max"].detach(),
                "factor_identity_tangent_base_abs_cos_mean": core[
                    "factor_identity_tangent_base_abs_cos_mean"
                ].detach(),
                "factor_identity_tangent_base_abs_cos_max": core[
                    "factor_identity_tangent_base_abs_cos_max"
                ].detach(),
                "factor_identity_tangent_pair_cos_mean": core["factor_identity_tangent_pair_cos_mean"].detach(),
                "factor_identity_tangent_pair_cos_max": core["factor_identity_tangent_pair_cos_max"].detach(),
                "factor_identity_tangent_l2_min": core["factor_identity_tangent_l2_min"].detach(),
                "context_angle_change_degrees_mean": core["context_angle_change_degrees_mean"].detach(),
                "context_angle_change_degrees_max": core["context_angle_change_degrees_max"].detach(),
                "dynamic_mean_hard_cos": dynamic_mean_hard_cos.detach(),
                "dynamic_mean_anchor_loss_raw": dynamic_mean_anchor_loss_raw.detach(),
                **expert_diagnostics,
            },
        }

    def h6_logit(self, normalized_patches: torch.Tensor, local_text: torch.Tensor) -> torch.Tensor:
        normal = local_text[..., 0]
        abnormal = local_text[..., 1]
        normal_similarity = (normalized_patches.float() * normal.float()).sum(dim=-1)
        abnormal_similarity = (normalized_patches.float() * abnormal.float()).sum(dim=-1)
        return self.h6_logit_temperature * (abnormal_similarity - normal_similarity)

    def parameter_partitions(self) -> Dict[str, Iterable[nn.Parameter]]:
        """Named optimizer partitions matching the Progress 1 protocol."""
        partitions = {
            "h6_projectors": self.semantic_core.level_projectors.parameters(),
            "h6_concepts": list(self.semantic_core.normal_query.parameters())
            + list(self.semantic_core.abnormal_query.parameters())
            + list(self.semantic_core.router_key.parameters())
            + [self.semantic_core.concept_slots, self.semantic_core.level_embedding],
            "h6_prototype": list(self.semantic_core.prototype_attention.parameters())
            + list(self.semantic_core.normal_state_update.parameters())
            + list(self.semantic_core.abnormal_state_update.parameters()),
            "h6_vae": self.semantic_core.class_vae.parameters(),
            "h6_router": self.router.parameters(),
            "h6_dynamic_prompt": list(self.semantic_core.state_to_context_normal.parameters())
            + list(self.semantic_core.state_to_context_abnormal.parameters())
            + list(self.semantic_core.class_to_context.parameters()),
            "h6_gates": list(self.semantic_core.gamma_state.parameters())
            + list(self.semantic_core.gamma_class.parameters()),
            "h6_late_factor_identity": self.semantic_core.factor_id_projection.parameters(),
            "h6_factor_generator": (
                [self.semantic_core.factor_id_embedding]
                + list(self.semantic_core.factor_id_to_context.parameters())
                + list(self.semantic_core.factor_output_heads.parameters())
                if self.factor_generator_specialization_enabled
                else []
            ),
        }
        if self.paired_experts is not None:
            partitions["h6_paired_experts"] = self.paired_experts.parameters()
        return partitions
