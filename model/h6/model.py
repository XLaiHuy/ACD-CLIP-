"""Progress 1 integration: one dynamic semantic bank shared by both paths."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .losses import dynamic_residual_diagnostics, dynamic_residual_diversity_loss, factor_stage_diagnostics
from .router import PatchRouter
from .semantic_bank import BoundedPositiveGate, CoPSSemanticCore


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
        late_factor_identity_enabled: bool = False,
        factor_id_scale: float = 0.02,
        factor_id_max_ratio: float = 0.05,
        router_teacher_mode: str = "raw_cosine",
        text_dim: int = 768,
        ctx_len: int = 4,
        h6_logit_temperature: float = 10.0,
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
        self.late_factor_identity_enabled = bool(late_factor_identity_enabled)
        self.factor_id_scale = float(factor_id_scale)
        self.factor_id_max_ratio = float(factor_id_max_ratio)
        self.router_teacher_mode = str(router_teacher_mode)
        self.text_dim = int(text_dim)
        self.ctx_len = int(ctx_len)
        self.h6_logit_temperature = float(h6_logit_temperature)
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
        )
        self.router = PatchRouter(
            n_groups=n_groups,
            num_factors=num_factors,
            text_dim=text_dim,
            bank_dim=bank_dim,
            hidden_dim=router_dim,
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
        )
        self.rho = BoundedPositiveGate(initial=0.05, maximum=0.50, count=n_groups)
        self.epoch_one_based = 1

    def config_dict(self) -> Dict[str, int | float | str]:
        return {
            "variant": "p1_v5_late_identity_centered_teacher",
            "progress_version": "P1-v5-fix",
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
            "rho_max": 0.50,
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
            "factor_id_direction_method": "qr_relative_offset_shared_buffer",
            "factor_id_projection_mode": "shared_linear_bankdim_to_textdim",
            "factor_id_shared_across_states": True,
            "router_teacher_mode": self.router_teacher_mode,
            "router_teacher_center_detached": True,
            "router_teacher_probability_detached": True,
            "teacher_gate_scope": "patch",
        }

    def set_epoch(self, epoch_one_based: int) -> None:
        self.epoch_one_based = int(epoch_one_based)

    def rho_cap(self) -> float:
        return min(0.50, 0.10 * max(1, self.epoch_one_based))

    def rho_values(self) -> torch.Tensor:
        return self.rho(cap=self.rho_cap())

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
        return_raw: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from utils import get_real_name, get_soft_prompt_sentence
        from model.tokenizer import tokenize

        batch, factors, states, ctx_len, text_dim = dynamic_contexts.shape
        if factors != self.num_factors or states != 2 or ctx_len != self.ctx_len or text_dim != self.text_dim:
            raise ValueError("dynamic contexts do not match the H6 configuration")
        sentences = []
        for class_name in class_names:
            real_name = get_real_name(dataset_name, class_name)
            for _ in range(self.num_factors):
                for state in range(2):
                    sentences.append(get_soft_prompt_sentence(real_name, state, self.ctx_len))
        token_ids = tokenize(sentences).to(dynamic_contexts.device)
        contexts = dynamic_contexts.reshape(batch * factors * states, ctx_len, text_dim)
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
    def _fuse_factor_bank(hard_adapted: torch.Tensor, dynamic_text: torch.Tensor, hybrid_alpha: float) -> torch.Tensor:
        if hard_adapted.ndim != 4:
            raise ValueError("hard_adapted must be [G,B,768,2]")
        if dynamic_text.ndim != 5:
            raise ValueError("dynamic_text must be [G,B,M,768,2]")
        hard_adapted = F.normalize(hard_adapted.float(), dim=2)
        dynamic_text = F.normalize(dynamic_text.float(), dim=3)
        mixed = (1.0 - float(hybrid_alpha)) * hard_adapted.unsqueeze(2) + float(hybrid_alpha) * dynamic_text
        return F.normalize(mixed, dim=3)

    def build_batch(
        self,
        base_model,
        dataset_name: str,
        class_names: Sequence[str],
        visual_output: Dict[str, torch.Tensor],
        hybrid_alpha: float,
        debug: bool = False,
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
        dynamic, dynamic_raw = self._encode_dynamic_bank(
            base_model, dataset_name, class_names, core["dynamic_contexts"], return_raw=True
        )
        hard_adapted, hard_frozen = self._batch_hard_embeddings(
            base_model, dataset_name, class_names, visual_output["cls24"].device
        )
        hard_adapted = F.normalize(hard_adapted.float(), dim=2)
        hard_frozen = F.normalize(hard_frozen.float(), dim=2)
        dynamic = F.normalize(dynamic.float(), dim=3)
        factor_bank = self._fuse_factor_bank(hard_adapted, dynamic, hybrid_alpha)
        anchor = hard_frozen.unsqueeze(2).expand_as(dynamic)
        kg_loss = (1.0 - F.cosine_similarity(dynamic.float(), anchor, dim=3)).mean()
        residual_diversity = dynamic_residual_diversity_loss(dynamic, hard_frozen)
        routing = self.router(
            visual_output["seg_tokens"],
            epoch_one_based=self.epoch_one_based,
            concept_keys=core["concept_keys"],
            update_load_bias=self.training,
        )
        prediction_probabilities = routing["prediction_probabilities"]
        local_text = self.router.local_text(prediction_probabilities, factor_bank)
        patches = torch.stack(visual_output["seg_tokens"], dim=0).float()
        patches = F.normalize(patches, dim=-1)
        h6_logits = self.h6_logit(patches, local_text)
        return {
            **core,
            **routing,
            "hard_adapted": hard_adapted,
            "hard_frozen": hard_frozen,
            "dynamic_text": dynamic,
            "dynamic_text_raw": dynamic_raw,
            "factor_bank": factor_bank,
            "kg_loss": kg_loss,
            "residual_diversity": residual_diversity,
            "text_global": self.router.aggregate_global(prediction_probabilities, factor_bank),
            "local_text": local_text,
            "h6_logits": h6_logits,
            "rho": self.rho_values(),
            "router_diagnostics": {
                **self.router.diagnostics(
                prediction_probabilities,
                dense_probabilities=routing["dense_probabilities"],
                sparse_probabilities=routing["sparse_probabilities"],
                topk_indices=routing["topk_indices"],
                ),
                **self.router.concept_key_diagnostics(core["concept_keys"]),
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
                **self.semantic_core.initialization_diagnostics(),
                **factor_stage_diagnostics(core["concept_slots"], "stage_concept_slots", factor_dim=0),
                **factor_stage_diagnostics(core["concept_keys"], "stage_concept_keys", factor_dim=0),
                **factor_stage_diagnostics(core["normal_queries"], "stage_normal_queries", factor_dim=1),
                **factor_stage_diagnostics(core["abnormal_queries"], "stage_abnormal_queries", factor_dim=1),
                **factor_stage_diagnostics(core["prototype_normal"], "stage_prototype_normal", factor_dim=1),
                **factor_stage_diagnostics(core["prototype_abnormal"], "stage_prototype_abnormal", factor_dim=1),
                **factor_stage_diagnostics(core["state_delta_raw"], "stage_state_to_context_raw", factor_dim=1),
                **factor_stage_diagnostics(core["state_delta_with_identity"], "stage_state_to_context_with_identity", factor_dim=1),
                **factor_stage_diagnostics(core["state_delta"], "stage_state_to_context_norm", factor_dim=1),
                **factor_stage_diagnostics(core["dynamic_contexts"], "stage_context_before_encoder", factor_dim=1),
                **factor_stage_diagnostics(dynamic_raw, "stage_dynamic_text_raw", factor_dim=2),
                **factor_stage_diagnostics(dynamic, "stage_dynamic_text_norm", factor_dim=2),
                **dynamic_residual_diagnostics(dynamic, hard_frozen),
                "late_factor_identity_enabled": core["late_factor_identity_enabled"].detach(),
                "factor_id_scale": core["factor_id_scale"].detach(),
                "factor_id_max_ratio": core["factor_id_max_ratio"].detach(),
                "factor_id_residual_norm_mean": core["factor_id_residual_norm_mean"].detach(),
                "factor_id_residual_norm_max": core["factor_id_residual_norm_max"].detach(),
                "factor_id_residual_to_context_ratio_mean": core["factor_id_residual_to_context_ratio_mean"].detach(),
                "factor_id_residual_to_context_ratio_max": core["factor_id_residual_to_context_ratio_max"].detach(),
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
        return {
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
            + list(self.semantic_core.gamma_class.parameters())
            + list(self.rho.parameters()),
            "h6_late_factor_identity": self.semantic_core.factor_id_projection.parameters(),
        }
