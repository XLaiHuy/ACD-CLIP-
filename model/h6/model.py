"""Progress 1 integration: one dynamic semantic bank shared by both paths."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .conditional_semantics import (
    ConditionalSemanticCore,
    predictor_aligned_abnormal_residual,
)
from .losses import (
    compute_final_expert_hard_cosine,
    dynamic_residual_diagnostics,
    dynamic_residual_diversity_loss,
    factor_bank_against_reference_diagnostics,
    factor_bank_comparison_diagnostics,
    factor_stage_diagnostics,
)
from .router import PatchRouter
from .visual_adaptation import ConditionalVisualAdapter
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
        role_topology: str = "flat",
        role_teacher_scale: float | None = None,
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
        router_boundary_mode: str = "none",
        router_boundary_trust_scale: float | None = None,
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
        phase4v_bottleneck: int = 64,
        phase4v_lambda: float = 0.05,
        intrinsic_factor_responsibility: bool = False,
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
        self.role_topology = str(role_topology)
        if self.role_topology not in {"flat", "r2_normal_anomaly"}:
            raise ValueError("role_topology must be 'flat' or 'r2_normal_anomaly'")
        if self.role_topology == "r2_normal_anomaly" and self.num_factors != 2:
            raise ValueError("r2_normal_anomaly requires num_factors=2")
        self.role_teacher_scale = (
            None if role_teacher_scale is None else float(role_teacher_scale)
        )
        if self.role_topology == "r2_normal_anomaly" and (
            self.role_teacher_scale is None or self.role_teacher_scale <= 0.0
        ):
            raise ValueError("r2_normal_anomaly requires a positive role_teacher_scale")
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
        self.router_boundary_mode = str(router_boundary_mode)
        self.router_boundary_trust_scale = None if router_boundary_trust_scale is None else float(router_boundary_trust_scale)
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
        self.intrinsic_factor_responsibility = bool(intrinsic_factor_responsibility)
        if self.intrinsic_factor_responsibility and self.num_factors != 2:
            raise ValueError("intrinsic factor responsibility currently requires R2")
        self.progress_version = str(progress_version)
        self.phase4v_enabled = self.progress_version == "P4V-K1"
        self.semantic_factorization_enabled = self.progress_version in {
            "P4-CSF-K1", "P4-CSF-K1-NOOP", "P4V-K1"
        }
        self.k1_noop_selectivity_enabled = self.progress_version == "P4-CSF-K1-NOOP"
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
        if self.semantic_factorization_enabled:
            forbidden = {
                "num_factors": self.num_factors != 1,
                "top_k": self.top_k != 1,
                "role_topology": self.role_topology != "flat",
                "expert_enabled": self.expert_enabled,
                "intrinsic_factor_responsibility": self.intrinsic_factor_responsibility,
                "cluster_responsibility_enabled": self.cluster_responsibility_enabled,
                "late_factor_identity_enabled": self.late_factor_identity_enabled,
                "factor_generator_specialization_enabled": self.factor_generator_specialization_enabled,
                "router_boundary_mode": self.router_boundary_mode != "none",
                "local_factor_mode": self.local_factor_mode != "legacy_mix",
            }
            active = [name for name, enabled in forbidden.items() if enabled]
            if active:
                raise ValueError(
                    "P4-CSF-K1 forbids legacy factor/router/expert paths: "
                    + ", ".join(active)
                )
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
            boundary_mode=router_boundary_mode,
            boundary_trust_scale=router_boundary_trust_scale,
        )
        self.conditional_semantic_core = (
            ConditionalSemanticCore(
                n_groups=n_groups,
                bank_dim=bank_dim,
                text_dim=text_dim,
                ctx_len=ctx_len,
                vae_hidden_dim=vae_hidden_dim,
                vae_latent_dim=vae_latent_dim,
                vae_class_ratio=vae_class_ratio,
            )
            if self.semantic_factorization_enabled
            else None
        )
        self.visual_adapter = (
            ConditionalVisualAdapter(text_dim, bank_dim, phase4v_bottleneck, phase4v_lambda)
            if self.phase4v_enabled
            else None
        )
        if self.semantic_factorization_enabled:
            # Retain legacy modules only so old state-dict layouts remain
            # loadable. They are neither trainable nor executed in P4-CSF-K1.
            self.semantic_core.requires_grad_(False)
            self.router.requires_grad_(False)
        if self.intrinsic_factor_responsibility:
            # Retained solely for legacy checkpoint load compatibility; it is
            # neither executed nor optimized by the intrinsic production path.
            self.router.requires_grad_(False)
        self.residual_act_enabled = self.progress_version == "P1-v8.4-A"
        self.act_head = None
        if self.residual_act_enabled:
            # Minimum-capacity ACT gate over the exact already-computed router
            # patch features. Zero initialization starts from continuous a=.5
            # without adding another encoder or a deep gating network.
            self.act_head = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, 1))
            nn.init.zeros_(self.act_head[-1].weight)
            nn.init.zeros_(self.act_head[-1].bias)
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
        is_v83 = self.progress_version == "P1-v8.3"
        is_v84a = self.progress_version == "P1-v8.4-A"
        is_p4_csf = self.semantic_factorization_enabled
        is_structured_utility = is_v83 or is_v84a
        return {
            "variant": (
                "p4_conditional_semantic_k1" if is_p4_csf
                else "p1_v8_4_a_true_residual_act" if is_v84a
                else
                "p1_v8_3_structured_utility_routing" if is_v83
                else "p1_v7_full_fofs_paired_semantic_moe" if self.expert_enabled
                else "p1_v6_structural_specialization"
            ),
            "progress_version": self.progress_version,
            "semantic_factorization_enabled": is_p4_csf,
            "k1_noop_selectivity_enabled": self.k1_noop_selectivity_enabled,
            "conditioning_path": "context_only" if is_p4_csf else "legacy",
            "semantic_bank_count": 1 if is_p4_csf else self.num_factors,
            "legacy_router_active": False if is_p4_csf else not self.intrinsic_factor_responsibility,
            "legacy_factor_roles_active": False if is_p4_csf else self.role_topology != "flat",
            "predictor_residual": (
                "dynamic_abnormal_minus_stopgrad_base_abnormal" if is_p4_csf else None
            ),
            "progress": self.progress,
            "n_groups": self.n_groups,
            "num_factors": self.num_factors,
            "top_k": self.top_k,
            "role_topology": self.role_topology,
            "role_teacher_scale": self.role_teacher_scale,
            "intrinsic_factor_responsibility": self.intrinsic_factor_responsibility,
            "bank_dim": self.bank_dim,
            "router_dim": self.router_dim,
            "router_temperature": self.router_temperature,
            "router_soft_epochs": self.router_soft_epochs,
            "dense_routing_epochs": self.router_soft_epochs,
            "sparse_start_epoch": self.router_soft_epochs + 1,
            "sparse_transition_epochs": self.sparse_transition_epochs,
            "sparse_full_epoch": self.router_soft_epochs + self.sparse_transition_epochs,
            "sparse_mode": "diagnostic_only" if is_structured_utility else "straight_through_topk",
            "prediction_interpolation_enabled": not is_structured_utility,
            "prediction_routing": self.router.prediction_routing,
            "routing": "dense" if is_structured_utility else self.router.prediction_routing,
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
            "structured_text_layout": (
                "[R_NORMAL][R_ANOMALY][STATE_r][CLASS][literal_state][REAL_NAME]"
                if self.role_topology == "r2_normal_anomaly"
                else (
                    "[C1][C2][C3][C4][STATE_m][CLASS][literal_state][REAL_NAME]"
                    if self.num_factors == 4
                    else f"[C1..C{self.num_factors}][STATE_m][CLASS][literal_state][REAL_NAME]"
                )
            ),
            "structured_text_enabled": is_structured_utility,
            "dynamic_text_adapt_text": is_structured_utility,
            "state_token_factor_specific": True,
            "utility_denominator_floor": 0.10,
            "tau_utility": 0.05,
            "utility_gain_threshold": 0.02,
            "utility_entropy_threshold": 0.98,
            "exploration_schedule": [0.15, 0.05],
            "utility_teacher_detached": True,
            "dense_router_only": is_structured_utility,
            "local_correction_semantics": (
                "act_times_routed_true_residual" if is_v84a else "routed_absolute_factor_margin"
            ),
            "noop_reference": (
                "expected_noop_pre_expert_bank" if is_v84a else None
            ),
            "act_enabled": is_v84a,
            "act_model": "layernorm_linear" if is_v84a else None,
            "act_probability_mode": "continuous_sigmoid" if is_v84a else None,
            "act_initial_probability": 0.5 if is_v84a else None,
            "act_parameter_count": (
                sum(parameter.numel() for parameter in self.act_head.parameters())
                if self.act_head is not None else 0
            ),
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
            "router_boundary_mode": self.router_boundary_mode,
            "router_boundary_trust_scale": self.router_boundary_trust_scale,
            "router_boundary_residual": self.router.boundary_residual is not None,
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


    def phase4v_state_code(self, base_model, visual_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return controller-only CoPS contrast; never encode dynamic text."""
        if not self.phase4v_enabled or self.conditional_semantic_core is None:
            raise RuntimeError("Phase4-V visual controller is unavailable")
        core = self.conditional_semantic_core(
            visual_output["seg_tokens_pre_l2"],
            visual_output["cls24"],
            base_model.soft_prompt.ctx_normal,
            base_model.soft_prompt.ctx_abnormal,
        )
        return {
            "semantic_code": (core["prototype_abnormal"] - core["prototype_normal"]).squeeze(1),
            "core": core,
        }

    def phase4v_adapt(self, patches: torch.Tensor, semantic_code: torch.Tensor, gate: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        if not self.phase4v_enabled or self.visual_adapter is None:
            raise RuntimeError("Phase4-V visual adapter is unavailable")
        return self.visual_adapter(patches, semantic_code, gate, **kwargs)

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


    @staticmethod
    def _responsibility_factor_view_center_spread(
        hard_adapted: torch.Tensor,
        dynamic_text: torch.Tensor,
        center_mix: float,
        factor_spread: float,
    ) -> torch.Tensor:
        """Exact center/spread values with responsibility gradients only through factor deviations.

        This is deliberately not a second factor bank. Its forward values are
        identical to ``_fuse_factor_bank_center_spread``; the stop-gradients
        prevent the auxiliary role CE from moving shared hard/center semantics.
        """
        if hard_adapted.ndim != 4 or dynamic_text.ndim != 5:
            raise ValueError("responsibility view expects hard [G,B,D,2] and dynamic [G,B,M,D,2]")
        raw_mean = dynamic_text.float().mean(dim=2, keepdim=True).detach()
        deviations = dynamic_text.float() - dynamic_text.float().mean(dim=2, keepdim=True)
        dynamic_view = dynamic_text.detach().float() + deviations - deviations.detach()
        semantic_center = F.normalize(raw_mean, dim=3).detach()
        hard_center = F.normalize(hard_adapted.float(), dim=2).unsqueeze(2).detach()
        center = F.normalize(
            (1.0 - float(center_mix)) * hard_center + float(center_mix) * semantic_center,
            dim=3,
        ).detach()
        residual = dynamic_view - raw_mean
        tangent = residual - (residual * center).sum(dim=3, keepdim=True) * center
        tangent = tangent / tangent.norm(dim=3, keepdim=True).clamp_min(1e-8)
        return F.normalize(center + float(factor_spread) * tangent, dim=3)

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
        base_text_features: torch.Tensor | None = None,
        state_scale: float = 1.0,
        class_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Create the unique dynamic factor bank and route every patch through it."""
        if len(class_names) != visual_output["cls24"].shape[0]:
            raise ValueError("class_names must have one entry per image")
        if self.phase4v_enabled:
            raise RuntimeError("P4V-K1 forbids dynamic-text batch construction; use phase4v_state_code only")
        if self.semantic_factorization_enabled:
            return self._build_conditional_semantic_batch(
                base_model=base_model,
                dataset_name=dataset_name,
                class_names=class_names,
                visual_output=visual_output,
                base_text_features=base_text_features,
                state_scale=state_scale,
                class_scale=class_scale,
                debug=debug,
            )
        core = self.forward_core(
            visual_output,
            base_model.soft_prompt.ctx_normal,
            base_model.soft_prompt.ctx_abnormal,
            debug=debug,
        )
        structured_text = self.progress_version in {"P1-v8.3", "P1-v8.4-A"}
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
        routing = None if self.intrinsic_factor_responsibility else self.router(
            visual_output["seg_tokens"],
            epoch_one_based=self.epoch_one_based,
            concept_keys=core["concept_keys"],
            update_load_bias=self.training if update_load_bias is None else bool(update_load_bias),
            frozen_reference_features=visual_output.get("frozen_router_reference"),
        )

        if self.intrinsic_factor_responsibility:
            if self.paired_experts is not None:
                raise RuntimeError("intrinsic factor responsibility requires the frozen non-expert Factor Bank")
            patches_for_responsibility = F.normalize(
                torch.stack(visual_output["seg_tokens"], dim=0).float(), dim=-1
            )
            responsibility_bank = (
                self._responsibility_factor_view_center_spread(
                    hard_adapted, dynamic, self.local_center_mix, self.local_factor_spread
                )
                if self.local_factor_mode == "center_spread"
                else factor_bank
            )
            intrinsic_state_similarity = self.h6_logit_temperature * torch.einsum(
                "gbpd,gbmds->gbpms",
                patches_for_responsibility.detach(), responsibility_bank,
            )
            intrinsic_responsibility_logits = torch.logsumexp(
                intrinsic_state_similarity, dim=-1
            )
            intrinsic_responsibility_probabilities = F.softmax(
                intrinsic_responsibility_logits, dim=-1
            )
            # The task correction receives the detached responsibility decision;
            # its selected residual values remain fully differentiable.
            prediction_probabilities = intrinsic_responsibility_probabilities.detach()
            topk_indices = prediction_probabilities.topk(self.top_k, dim=-1).indices
            routing = {
                "router_input_features": patches_for_responsibility,
                "qk_logits": intrinsic_responsibility_logits,
                "logits": intrinsic_responsibility_logits,
                "selection_logits": intrinsic_responsibility_logits,
                "prediction_probabilities": prediction_probabilities,
                "dense_probabilities": prediction_probabilities,
                "sparse_probabilities": prediction_probabilities,
                "topk_indices": topk_indices,
                "final_router_keys": core["concept_keys"].detach(),
                "intrinsic_responsibility_logits": intrinsic_responsibility_logits,
                "intrinsic_responsibility_probabilities": intrinsic_responsibility_probabilities,
                "intrinsic_state_similarity": intrinsic_state_similarity,
                "responsibility_factor_bank": responsibility_bank,
            }
            diag_zero = intrinsic_responsibility_logits.new_zeros(
                intrinsic_responsibility_logits.shape[0]
            )
            diag_usage = prediction_probabilities.mean(dim=(1, 2)).detach()
            routing.update({
                "raw_concept_key_cos_mean": diag_zero,
                "raw_concept_key_cos_max": diag_zero,
                "raw_concept_key_l2_min": diag_zero,
                "final_router_key_cos_mean": diag_zero,
                "final_router_key_cos_max": diag_zero,
                "final_router_key_l2_min": diag_zero,
                "router_key_adaptation_ratio_mean": diag_zero,
                "router_key_adaptation_ratio_max": diag_zero,
                "queries": patches_for_responsibility.detach(),
                "sparse_ratio": intrinsic_responsibility_logits.sum() * 0.0,
                "load_bias": torch.zeros_like(diag_usage),
                "ema_topk_usage": diag_usage,
                "level_input_alias": diag_zero,
                "level_input_difference": diag_zero,
                "level_query_difference": diag_zero,
                "level_logit_difference": diag_zero,
                "router_patch_count": torch.full_like(diag_zero, intrinsic_responsibility_logits.shape[2]),
                "router_softmax_dim": torch.full_like(diag_zero, -1),
                "router_topk_dim": torch.full_like(diag_zero, -1),
                "query_pairwise_cos_mean_across_patches": diag_zero,
                "query_pairwise_cos_max_across_patches": diag_zero,
                "query_variance_across_patches": diag_zero,
                "query_effective_rank": diag_zero,
                "query_singular_value_ratio": diag_zero,
                "per_factor_logit_std_across_patches": intrinsic_responsibility_logits.float().std(dim=2, unbiased=False).mean(dim=1).detach(),
                "raw_query_pairwise_cos_mean": diag_zero,
                "local_query_pairwise_cos_mean": diag_zero,
                "final_query_pairwise_cos_mean": diag_zero,
                "raw_query_variance_across_patches": diag_zero,
                "local_query_variance_across_patches": diag_zero,
                "final_query_variance_across_patches": diag_zero,
                "final_query_effective_rank": diag_zero,
                "final_query_top1_energy_ratio": diag_zero,
                "local_bypass_norm_mean": diag_zero,
                "local_bypass_to_learned_ratio_mean": diag_zero,
                "local_bypass_to_learned_ratio_max": diag_zero,
            })
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
        local_text = PatchRouter.local_text(prediction_probabilities, active_factor_bank)
        patches = torch.stack(visual_output["seg_tokens"], dim=0).float()
        patches = F.normalize(patches, dim=-1)
        absolute_h6_logits = self.h6_logit(patches, local_text)
        
        factor_patch_logits = self.h6_logit(
            patches.unsqueeze(3), active_factor_bank.unsqueeze(2)
        )
        noop_reference_patch_logit = self.h6_logit(
            patches, expected_noop_pre_expert_bank[:, :, 0].unsqueeze(2)
        )
        factor_residual_logits = (
            factor_patch_logits - noop_reference_patch_logit.unsqueeze(-1)
        )
        qk_probabilities = F.softmax(routing["qk_logits"], dim=-1)
        if self.residual_act_enabled:
            from .utility_routing import routed_residual_correction

            act_logits = self.act_head(routing["router_input_features"]).squeeze(-1)
            act_probability = torch.sigmoid(act_logits)
            h6_logits = routed_residual_correction(
                act_probability, prediction_probabilities, factor_residual_logits
            )
            # Instrumentation-only legacy branch on the identical tensors.  It
            # exposes the exact Q/K-only downstream correction for zero-init
            # boundary parity without a second model or altered production path.
            qk_routed_residual = (qk_probabilities * factor_residual_logits).sum(dim=-1)
            qk_h6_logits = act_probability * qk_routed_residual
        else:
            act_logits = None
            act_probability = None
            h6_logits = absolute_h6_logits
            qk_routed_residual = None
            qk_h6_logits = None
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
        active_factor_evidence = (
            factor_residual_logits if self.residual_act_enabled else factor_patch_logits
        )
        rho_scaled_factor_correction = active_factor_evidence * rho.view(G, 1, 1, 1).to(factor_patch_logits.dtype)
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
            "text_global": PatchRouter.aggregate_global(prediction_probabilities, global_factor_bank),
            "topk_indices": routing.get("topk_indices", None),
            "prediction_logits": routing.get("logits", None),
            "prediction_probabilities": routing["prediction_probabilities"],
            "qk_probabilities": qk_probabilities,
            "qk_routed_residual": qk_routed_residual,
            "qk_h6_logits": qk_h6_logits,
            "h6_logits": h6_logits,
            "rho": self.rho_values(),
            "expert_scale": torch.tensor(self.expert_scale(), device=patches.device),
            "factor_patch_logits": factor_patch_logits,
            "factor_absolute_logits": factor_patch_logits,
            "noop_reference_logit": noop_reference_patch_logit,
            "factor_residual_logits": factor_residual_logits,
            "act_logits": act_logits,
            "act_probability": act_probability,
            "residual_act_enabled": torch.tensor(self.residual_act_enabled, device=patches.device),
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

    def _build_conditional_semantic_batch(
        self,
        *,
        base_model,
        dataset_name: str,
        class_names: Sequence[str],
        visual_output: Dict[str, torch.Tensor],
        base_text_features: torch.Tensor | None,
        state_scale: float,
        class_scale: float,
        debug: bool,
    ) -> Dict[str, torch.Tensor]:
        """Build the Stage-0 K1 context path in the frozen base DFG frame."""
        if self.conditional_semantic_core is None:
            raise RuntimeError("conditional semantic core is unavailable")
        if self.num_factors != 1 or self.residual_act_enabled:
            raise RuntimeError("P4-CSF-K1 requires K=1 and ACT disabled")
        if self.paired_experts is not None or self.intrinsic_factor_responsibility:
            raise RuntimeError("P4-CSF-K1 cannot execute legacy experts or responsibility")
        if any(parameter.requires_grad for parameter in self.router.parameters()):
            raise RuntimeError("legacy Router must remain frozen in P4-CSF-K1")
        if any(parameter.requires_grad for parameter in self.semantic_core.parameters()):
            raise RuntimeError("legacy semantic factor core must remain frozen in P4-CSF-K1")

        if base_text_features is None:
            from utils import get_phase2b_global_text_features

            base_text_features = get_phase2b_global_text_features(
                base_model,
                dataset_name,
                class_names,
                visual_output["cls24"].device,
                use_hybrid_soft_prompt=getattr(base_model, "use_hybrid_soft_prompt", False),
                use_soft_prompt=getattr(base_model, "use_soft_prompt", False),
            )
        expected_base_shape = (
            self.n_groups,
            len(class_names),
            self.text_dim,
            2,
        )
        if tuple(base_text_features.shape) != expected_base_shape:
            raise ValueError(
                f"base_text_features must be {expected_base_shape}, "
                f"got {tuple(base_text_features.shape)}"
            )

        core = self.conditional_semantic_core(
            visual_output["seg_tokens_pre_l2"],
            visual_output["cls24"],
            base_model.soft_prompt.ctx_normal,
            base_model.soft_prompt.ctx_abnormal,
            state_scale=state_scale,
            class_scale=class_scale,
            debug=debug,
        )
        dynamic_text, dynamic_text_raw = self._encode_dynamic_bank(
            base_model,
            dataset_name,
            class_names,
            core["dynamic_contexts"],
            structured=False,
            return_raw=True,
        )
        dynamic_text = dynamic_text.squeeze(2)
        dynamic_text_raw = dynamic_text_raw.squeeze(2)
        base_cross_level = base_text_features.permute(1, 0, 2, 3).float()
        dynamic_cross_level = dynamic_text.permute(1, 0, 2, 3).float()

        base_logits = []
        dynamic_abnormal_logits = []
        noop_alpha = []
        noop_scores = []
        noop_base_abnormal_semantic = []
        noop_dynamic_abnormal_semantic = []
        noop_original_k1_dynamic_abnormal_logits = []
        normal_weights = []
        abnormal_weights = []
        for group_index, group_patches in enumerate(visual_output["seg_tokens"]):
            weights = base_model.compute_dfg_weights(
                group_patches, base_cross_level, group_index
            )
            weight_normal = weights["normal"].detach()
            weight_abnormal = weights["abnormal"].detach()
            normal_weights.append(weight_normal)
            abnormal_weights.append(weight_abnormal)
            base_dfg_text = base_model.apply_dfg_weights(
                base_cross_level, weight_normal, weight_abnormal
            )
            dynamic_dfg_text = base_model.apply_dfg_weights(
                dynamic_cross_level, weight_normal, weight_abnormal
            )
            scaled_patches = 10.0 * group_patches.float()
            base_logits.append(torch.matmul(scaled_patches, base_dfg_text.float()))
            base_abnormal_semantic = base_dfg_text[..., 1].float()
            dynamic_abnormal_semantic = dynamic_dfg_text[..., 1].float()
            if self.k1_noop_selectivity_enabled:
                noop_original_k1_dynamic_abnormal_logits.append(
                    torch.einsum("bpd,bd->bp", scaled_patches, dynamic_abnormal_semantic)
                )
                scores = torch.stack(
                    [
                        torch.einsum("bpd,bd->bp", group_patches.float(), base_abnormal_semantic),
                        torch.einsum("bpd,bd->bp", group_patches.float(), dynamic_abnormal_semantic),
                    ],
                    dim=-1,
                )
                alpha = F.softmax(scores, dim=-1)
                patch_semantic = F.normalize(
                    alpha[..., 0:1] * base_abnormal_semantic.unsqueeze(1)
                    + alpha[..., 1:2] * dynamic_abnormal_semantic.unsqueeze(1),
                    dim=-1,
                )
                dynamic_abnormal_logits.append(
                    torch.einsum("bpd,bpd->bp", scaled_patches, patch_semantic)
                )
                noop_alpha.append(alpha)
                noop_scores.append(scores)
                noop_base_abnormal_semantic.append(base_abnormal_semantic)
                noop_dynamic_abnormal_semantic.append(dynamic_abnormal_semantic)
            else:
                dynamic_abnormal_logits.append(
                    torch.einsum("bpd,bd->bp", scaled_patches, dynamic_abnormal_semantic)
                )

        base_group_logits = torch.stack(base_logits, dim=0)
        dynamic_abnormal = torch.stack(dynamic_abnormal_logits, dim=0)
        predictor = predictor_aligned_abnormal_residual(
            base_group_logits,
            dynamic_abnormal,
            self.rho_values(),
        )
        residual = predictor["predictor_residual_logits"]
        return {
            **core,
            **predictor,
            "base_text_features": base_text_features,
            "dynamic_text": dynamic_text,
            "dynamic_text_raw": dynamic_text_raw,
            "base_group_logits": base_group_logits,
            "dynamic_abnormal_logits": dynamic_abnormal,
            "base_dfg_weights_normal": torch.stack(normal_weights, dim=0),
            "base_dfg_weights_abnormal": torch.stack(abnormal_weights, dim=0),
            "h6_logits": residual,
            "factor_patch_logits": dynamic_abnormal.unsqueeze(-1),
            "rho": self.rho_values(),
            "residual_act_enabled": torch.tensor(False, device=residual.device),
            "legacy_router_executed": torch.tensor(False, device=residual.device),
            "legacy_factor_core_executed": torch.tensor(False, device=residual.device),
            "conditioning_path_count": torch.tensor(1, device=residual.device),
            "dense_probabilities": torch.ones_like(dynamic_abnormal.unsqueeze(-1)),
            "prediction_probabilities": torch.ones_like(dynamic_abnormal.unsqueeze(-1)),
            "probabilities": torch.ones_like(dynamic_abnormal.unsqueeze(-1)),
            "topk_indices": torch.zeros_like(dynamic_abnormal.unsqueeze(-1), dtype=torch.long),
            "hard_frozen": base_text_features,
            "factor_bank": dynamic_text.unsqueeze(2),
            "text_global": base_text_features,
            "residual_diversity": residual.sum() * 0.0,
            "raw_semantic_keys": core["prototype_abnormal"].reshape(-1, self.bank_dim)[:1],
            "dynamic_mean_anchor_loss_raw": residual.sum() * 0.0,
            "kg_loss": residual.sum() * 0.0,
            "sparse_ratio": residual.sum() * 0.0,
            "router_diagnostics": {
                "dense_factor_usage": torch.ones(self.n_groups, 1, device=residual.device),
                "sparse_factor_usage": torch.ones(self.n_groups, 1, device=residual.device),
                "unique_topk_pairs": torch.ones(self.n_groups, device=residual.device),
            },
            "noop_selectivity_enabled": torch.tensor(
                self.k1_noop_selectivity_enabled, device=residual.device
            ),
            "noop_alpha": torch.stack(noop_alpha, dim=0) if noop_alpha else None,
            "noop_scores": torch.stack(noop_scores, dim=0) if noop_scores else None,
            "noop_base_abnormal_semantic": (
                torch.stack(noop_base_abnormal_semantic, dim=0)
                if noop_base_abnormal_semantic else None
            ),
            "noop_dynamic_abnormal_semantic": (
                torch.stack(noop_dynamic_abnormal_semantic, dim=0)
                if noop_dynamic_abnormal_semantic else None
            ),
            "noop_original_k1_dynamic_abnormal_logits": (
                torch.stack(noop_original_k1_dynamic_abnormal_logits, dim=0)
                if noop_original_k1_dynamic_abnormal_logits else None
            ),
        }

    def h6_logit(self, normalized_patches: torch.Tensor, local_text: torch.Tensor) -> torch.Tensor:
        normal = local_text[..., 0]
        abnormal = local_text[..., 1]
        normal_similarity = (normalized_patches.float() * normal.float()).sum(dim=-1)
        abnormal_similarity = (normalized_patches.float() * abnormal.float()).sum(dim=-1)
        return self.h6_logit_temperature * (abnormal_similarity - normal_similarity)

    def parameter_partitions(self) -> Dict[str, Iterable[nn.Parameter]]:
        """Named optimizer partitions matching the Progress 1 protocol."""
        if self.semantic_factorization_enabled:
            core = self.conditional_semantic_core
            if core is None:
                raise RuntimeError("conditional semantic core is unavailable")
            return {
                "h6_projectors": core.level_projectors.parameters(),
                "h6_concepts": [core.level_embedding, core.normal_query, core.abnormal_query],
                "h6_prototype": list(core.prototype_attention.parameters())
                + list(core.normal_state_update.parameters())
                + list(core.abnormal_state_update.parameters()),
                "h6_vae": core.class_vae.parameters(),
                "h6_dynamic_prompt": list(core.state_to_context_normal.parameters())
                + list(core.state_to_context_abnormal.parameters())
                + list(core.class_to_context.parameters()),
                "h6_gates": list(core.gamma_state.parameters())
                + list(core.gamma_class.parameters()),
                "h6_router": [],
                "h6_late_factor_identity": [],
                "h6_factor_generator": [],
            }
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
            "h6_router": [] if self.intrinsic_factor_responsibility else self.router.parameters(),
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
        if self.act_head is not None:
            partitions["h6_act"] = self.act_head.parameters()
        if self.paired_experts is not None:
            partitions["h6_paired_experts"] = self.paired_experts.parameters()
        return partitions
