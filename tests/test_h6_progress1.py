import tempfile

import torch
from torch import nn

from model.checkpoint_utils import build_phase4_checkpoint, load_adapter_checkpoint
from model.h6.losses import center_loss, factor_orthogonal_loss, routing_balance_loss
from model.h6.model import H6Progress1


class _TinyPhase4Model(nn.Module):
    def __init__(
        self,
        h6_router_soft_epochs=2,
        h6_sparse_transition_epochs=1,
        h6_late_factor_identity_enabled=False,
        h6_factor_id_scale=0.02,
        h6_factor_id_max_ratio=0.05,
        h6_router_teacher_mode="raw_cosine",
    ):
        super().__init__()
        self.image_adapter = nn.Linear(3, 3)
        self.text_adapter = nn.Linear(3, 3)
        self.soft_prompt = nn.Module()
        self.soft_prompt.ctx = nn.Parameter(torch.randn(4, 768))
        self.h6_enabled = True
        self.h6 = H6Progress1(
            n_groups=3,
            router_soft_epochs=h6_router_soft_epochs,
            sparse_transition_epochs=h6_sparse_transition_epochs,
            load_bias_enabled=True,
            vae_class_ratio=0.25,
            late_factor_identity_enabled=h6_late_factor_identity_enabled,
            factor_id_scale=h6_factor_id_scale,
            factor_id_max_ratio=h6_factor_id_max_ratio,
            router_teacher_mode=h6_router_teacher_mode,
        )
        self.n_groups = 3
        self.dfg_mode = "attn"
        self.dfg_attn_dim = 256
        self.dfg_attn_tau = 8.0
        self.use_ss2d_dfg = True
        self.dfg_gamma_max = 0.2
        self.dfg_ss2d_fusion = "weight_residual"
        self.dfg_beta = 0.1
        self.dfg_beta_schedule = "warmup010"
        self.dfg_beta_target = 0.1
        self.dfg_weight_residual_fp32 = True
        self.soft_prompt_ctx_len = 4
        self.soft_prompt_init = "phrase"
        self.soft_prompt_init_phrase = "a photo of a"
        self.hybrid_alpha_current = 0.05
        self.hybrid_alpha_max = 0.2
        self.soft_prompt_freeze_epochs = 3


def test_progress1_synthetic_backward_and_isolation():
    torch.manual_seed(1)
    h6 = H6Progress1(n_groups=3)
    legacy_adapter = nn.Linear(768, 768)
    frozen_clip_parameter = nn.Parameter(torch.ones(1), requires_grad=False)
    raw = torch.randn(2, 16, 768)
    levels = [legacy_adapter(raw) for _ in range(3)]
    visual = {
        "seg_tokens_pre_l2": levels,
        "seg_tokens": [torch.nn.functional.normalize(level, dim=-1) for level in levels],
        "cls24": torch.randn(2, 768),
    }
    core = h6.forward_core(visual, torch.randn(4, 768), torch.randn(4, 768))
    routing = h6.router(visual["seg_tokens"], epoch_one_based=3)
    factor_bank = torch.nn.functional.normalize(torch.randn(3, 2, 4, 768, 2, requires_grad=True), dim=3)
    local_text = h6.router.local_text(routing["probabilities"], factor_bank)
    h6_logits = h6.h6_logit(torch.stack(visual["seg_tokens"]), local_text)
    loss = (
        h6_logits.square().mean()
        + center_loss(core["projected_levels"], core["prototype_normal"], core["prototype_abnormal"], torch.zeros(2, 1, 32, 32), torch.tensor([0, 1]))
        + factor_orthogonal_loss(factor_bank)
        + routing_balance_loss(routing["probabilities"])
        + core["reconstruction"]
        + core["kl"] * 1e-4
        + frozen_clip_parameter.sum() * 0.0
    )
    loss.backward()
    assert legacy_adapter.weight.grad is not None
    assert h6.semantic_core.level_projectors[0][0].weight.grad is not None
    if h6.router.router_query_mode == "raw":
        assert h6.router.query_projector[0].weight.grad is not None
    else:
        assert h6.router.local_query_projector[0].weight.grad is not None
    assert frozen_clip_parameter.grad is None
    assert not hasattr(h6, "visual_experts")
    assert not hasattr(h6, "consistency")
    assert torch.all(h6.rho_values() > 0)
    assert torch.all(h6.rho_values() <= 0.50)


def test_synthetic_old_and_phase4_checkpoint_compatibility():
    source = _TinyPhase4Model(
        h6_router_soft_epochs=8,
        h6_sparse_transition_epochs=4,
        h6_late_factor_identity_enabled=True,
        h6_factor_id_scale=0.02,
        h6_factor_id_max_ratio=0.05,
        h6_router_teacher_mode="state_centered_cosine",
    )
    payload = build_phase4_checkpoint(
        source,
        epoch=3,
        seed=0,
        precision="fp32",
        phase2b_config={
            "n_groups": 3,
            "h6_dense_routing_epochs": 8,
            "h6_sparse_start_epoch": 9,
            "h6_sparse_transition_epochs": 4,
            "h6_router_failure_patience": 2,
            "h6_router_max_sparse_dead_factors": 1,
            "h6_router_min_unique_topk_pairs": 2,
            "h6_center_factor_aware": True,
            "h6_center_detach_assignment": True,
            "h6_kl_zero_epochs": 8,
            "h6_kl_warmup_epochs": 4,
            "h6_kl_free_bits": 0.02,
            "h6_vae_class_ratio": 0.25,
            "beta_h6_vae_kl": 1e-5,
            "lambda_h6_concept_key_diversity": 0.0,
            "h6_late_factor_identity_enabled": True,
            "h6_factor_id_scale": 0.02,
            "h6_factor_id_max_ratio": 0.05,
            "h6_router_teacher_mode": "state_centered_cosine",
        },
        loss_weights={
            "center": 0.1,
            "center_factor_aware": True,
            "center_detach_assignment": True,
            "router_teacher": 0.01,
            "router_teacher_temperature": 0.15,
            "router_teacher_start_epoch": 3,
            "router_teacher_warmup_epochs": 3,
            "router_teacher_mode": "state_centered_cosine",
            "teacher_confidence_gate": True,
            "teacher_entropy_threshold": 0.98,
            "teacher_prob_std_threshold": 0.001,
            "balance": 0.001,
            "vae_kl_zero_epochs": 8,
        },
    )
    restored = _TinyPhase4Model(
        h6_router_soft_epochs=8,
        h6_sparse_transition_epochs=4,
        h6_late_factor_identity_enabled=True,
        h6_factor_id_scale=0.02,
        h6_factor_id_max_ratio=0.05,
        h6_router_teacher_mode="state_centered_cosine",
    )
    assert load_adapter_checkpoint(restored, payload) is True
    assert restored.h6.epoch_one_based == 3
    for key, value in source.h6.state_dict().items():
        assert torch.allclose(restored.h6.state_dict()[key], value)
    assert restored.h6.config_dict()["vae_prompt_path"] == "decoder_mu"
    assert restored.h6.config_dict()["frozen_anchor_mode"] == "functional_layer_norm_no_adapter"
    old_payload = {
        "image_adapter": source.image_adapter.state_dict(),
        "text_adapter": source.text_adapter.state_dict(),
    }
    assert load_adapter_checkpoint(restored, old_payload) is False
    with tempfile.NamedTemporaryFile(suffix=".pth") as handle:
        torch.save(payload, handle.name)
        loaded = torch.load(handle.name, map_location="cpu")
    assert loaded["checkpoint_version"] == 6
    assert loaded["phase4_progress"] == 1
    assert loaded["h6_enabled"] is True
    assert loaded["h6_config"]["variant"] == "p1_v6_structural_specialization"
    assert loaded["h6_config"]["progress_version"] == "P1-v6"
    assert loaded["h6_config"]["dense_routing_epochs"] == 8
    assert loaded["h6_config"]["sparse_start_epoch"] == 9
    assert loaded["h6_config"]["sparse_full_epoch"] == 12
    assert loaded["h6_config"]["sparse_mode"] == "straight_through_topk"
    assert loaded["h6_config"]["router_mode"] == "concept_key_dot"
    assert loaded["h6_config"]["router_teacher_enabled"] is True
    assert loaded["h6_config"]["router_teacher_detached"] is True
    assert loaded["h6_config"]["load_bias_enabled"] is True
    assert loaded["h6_config"]["load_bias_selection_only"] is True
    assert loaded["h6_config"]["dynamic_text_normalized"] is True
    assert loaded["h6_config"]["anchor_encoder_mode"] == "frozen"
    assert loaded["h6_config"]["diversity_target"] == "dynamic_residual"
    assert loaded["h6_config"]["router_scoring"] == "concept_key_dot"
    assert loaded["h6_config"]["center_factor_aware"] is True
    assert loaded["h6_config"]["center_assignment_detached"] is True
    assert loaded["h6_config"]["center_loss"] == "factor_aware_dense_detached"
    assert loaded["h6_config"]["kl_schedule"] == "zero_then_linear"
    assert loaded["h6_config"]["vae_prompt_use_mu"] is True
    assert loaded["h6_config"]["vae_class_skip_enabled"] is True
    assert loaded["h6_config"]["kl_free_bits"] == 0.02
    assert loaded["h6_config"]["three_level_router_mode"] == "shared_router_level_specific_inputs"
    assert loaded["h6_config"]["teacher_diagnostics_version"] == 1
    assert loaded["h6_config"]["late_factor_identity_enabled"] is True
    assert loaded["h6_config"]["factor_id_scale"] == 0.02
    assert loaded["h6_config"]["factor_id_max_ratio"] == 0.05
    assert loaded["h6_config"]["factor_id_direction_method"] == "tangent_context_anchor_shared_buffer"
    assert loaded["h6_config"]["router_query_mode"] == "local_global_bypass"
    assert loaded["h6_config"]["router_key_anchor_enabled"] is True
    assert loaded["h6_config"]["factor_context_anchor_enabled"] is True
    assert loaded["h6_config"]["factor_identity_tangent_projection_enabled"] is True
    assert loaded["h6_config"]["factor_id_projection_mode"] == "shared_linear_bankdim_to_textdim"
    assert loaded["h6_config"]["factor_id_shared_across_states"] is True
    assert loaded["h6_config"]["router_teacher_mode"] == "state_centered_cosine"
    assert loaded["h6_config"]["router_teacher_center_detached"] is True
    assert loaded["h6_config"]["router_teacher_probability_detached"] is True
    assert loaded["h6_config"]["teacher_confidence_gate_enabled"] is True
    assert loaded["h6_config"]["teacher_entropy_threshold"] == 0.98
    assert loaded["h6_config"]["teacher_probability_std_threshold"] == 0.001
    assert loaded["h6_config"]["teacher_gate_scope"] == "patch"
    assert loaded["loss_weights"]["center_factor_aware"] is True
    assert loaded["loss_weights"]["center_detach_assignment"] is True
