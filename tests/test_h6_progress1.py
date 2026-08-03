import tempfile

import torch
from torch import nn

from model.checkpoint_utils import build_phase4_checkpoint, load_adapter_checkpoint
from model.h6.losses import center_loss, factor_orthogonal_loss, routing_balance_loss
from model.h6.model import H6Progress1


class _TinyPhase4Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_adapter = nn.Linear(3, 3)
        self.text_adapter = nn.Linear(3, 3)
        self.soft_prompt = nn.Module()
        self.soft_prompt.ctx = nn.Parameter(torch.randn(4, 768))
        self.h6_enabled = True
        self.h6 = H6Progress1(n_groups=3)
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
    assert h6.router.query_projector[0].weight.grad is not None
    assert frozen_clip_parameter.grad is None
    assert not hasattr(h6, "visual_experts")
    assert not hasattr(h6, "consistency")
    assert torch.all(h6.rho_values() > 0)
    assert torch.all(h6.rho_values() <= 0.50)


def test_synthetic_old_and_phase4_checkpoint_compatibility():
    source = _TinyPhase4Model()
    payload = build_phase4_checkpoint(
        source,
        epoch=3,
        seed=0,
        precision="fp32",
        phase2b_config={"n_groups": 3},
        loss_weights={"center": 0.1},
    )
    restored = _TinyPhase4Model()
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
    assert loaded["checkpoint_version"] == 2
    assert loaded["phase4_progress"] == 1
    assert loaded["h6_enabled"] is True
    assert loaded["h6_config"]["variant"] == "p1_v2_specialization_fix"
    assert loaded["h6_config"]["diversity_target"] == "dynamic_residual"
    assert loaded["h6_config"]["router_scoring"] == "concept_key_dot"
    assert loaded["h6_config"]["center_loss"] == "factor_aware_dense_detached"
    assert loaded["h6_config"]["kl_schedule"] == "zero_then_linear"
