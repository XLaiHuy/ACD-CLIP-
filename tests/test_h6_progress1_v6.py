import torch
import torch.nn.functional as F

from model.checkpoint_utils import validate_h6_configuration
from model.h6.model import H6Progress1
from model.h6.router import PatchRouter
from model.h6.semantic_bank import CoPSSemanticCore, deterministic_slot_directions


def _router(**kwargs):
    return PatchRouter(
        n_groups=1,
        num_factors=4,
        text_dim=8,
        bank_dim=4,
        hidden_dim=6,
        soft_routing_epochs=8,
        sparse_transition_epochs=4,
        **kwargs,
    )


def test_local_residual_common_shift_and_masked_center_detached():
    router = _router(router_query_mode="local_residual")
    tokens = F.normalize(torch.randn(1, 2, 4, 8), dim=-1)
    common = torch.randn(1, 2, 1, 8)
    local_a, _, _ = router._local_query_inputs(tokens)
    local_b, _, _ = router._local_query_inputs(tokens + common)
    assert torch.allclose(local_a, local_b, atol=1e-6)

    masked = tokens.clone()
    masked[:, :, 3] = 100.0
    valid = torch.tensor([[[1, 1, 1, 0], [1, 1, 1, 0]]], dtype=torch.bool)
    local_masked, _, _ = router._local_query_inputs(masked, valid)
    expected_center = masked[:, :, :3].mean(dim=2, keepdim=True).detach()
    assert torch.allclose(local_masked, masked - expected_center, atol=1e-6)
    assert expected_center.requires_grad is False


def test_raw_query_mode_preserves_legacy_projection_path_and_shapes():
    torch.manual_seed(1)
    router = _router(router_query_mode="raw")
    tokens = F.normalize(torch.randn(1, 2, 5, 8), dim=-1)
    keys = torch.randn(4, 4)
    out = router(tokens, epoch_one_based=9, concept_keys=keys)
    context = tokens.mean(dim=2, keepdim=True).expand(-1, -1, tokens.shape[2], -1)
    level = router.level_embedding[:, None, None, :].expand_as(tokens)
    expected = F.normalize(router.query_projector(torch.cat([tokens, context, level], dim=-1)).float(), dim=-1)
    assert torch.allclose(out["queries"], expected, atol=1e-6)
    assert out["queries"].shape == (1, 2, 5, 4)
    assert out["logits"].shape == (1, 2, 5, 4)
    assert out["router_softmax_dim"].item() == 3
    assert out["router_topk_dim"].item() == 3


def test_local_global_bypass_preserves_differences_gradients_and_rng_state():
    state_before = torch.random.get_rng_state()
    _ = deterministic_slot_directions(4, 8, 7200)
    _ = deterministic_slot_directions(4, 4, 7300)
    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after)

    router_a = _router(router_query_mode="local_global_bypass")
    router_b = _router(router_query_mode="local_global_bypass")
    assert torch.allclose(router_a.frozen_local_projection, router_b.frozen_local_projection)
    gram = router_a.frozen_local_projection.T @ router_a.frozen_local_projection
    assert torch.allclose(gram, torch.eye(4), atol=1e-5)
    assert router_a.frozen_local_projection.requires_grad is False

    tokens = F.normalize(torch.randn(1, 1, 4, 8), dim=-1).requires_grad_(True)
    out = router_a(tokens, epoch_one_based=9, concept_keys=torch.randn(4, 4))
    assert out["queries"][:, :, 0].sub(out["queries"][:, :, 1]).norm().item() > 1e-5
    assert out["local_bypass_to_learned_ratio_max"].item() <= 0.20001
    out["queries"].sum().backward()
    grads = [p.grad for p in router_a.local_query_projector.parameters()]
    assert any(g is not None and g.float().norm().item() > 0 for g in grads)
    assert getattr(router_a.frozen_local_projection, "grad", None) is None


def test_zero_local_residual_is_finite():
    router = _router(router_query_mode="local_global_bypass")
    tokens = torch.ones(1, 1, 4, 8)
    out = router(tokens, epoch_one_based=9, concept_keys=torch.randn(4, 4))
    assert torch.isfinite(out["queries"]).all()
    assert out["local_bypass_norm_mean"].item() == 0.0


def test_router_key_anchors_separate_identical_raw_keys_and_keep_gradients():
    router = _router(router_key_anchor_enabled=True)
    raw = torch.ones(4, 4, requires_grad=True)
    final, diag = router.final_router_keys(raw)
    assert torch.allclose(final.norm(dim=-1), torch.ones(4), atol=1e-6)
    assert diag["final_router_key_cos_max"].item() < 0.30
    assert diag["router_key_adaptation_ratio_max"].item() <= 0.25001
    final.sum().backward()
    assert raw.grad is not None
    assert raw.grad.float().norm().item() > 0
    assert router.router_key_anchors.requires_grad is False


def test_tangent_factor_identity_is_orthogonal_capped_and_directional():
    torch.manual_seed(2)
    core = CoPSSemanticCore(
        n_groups=1,
        num_factors=4,
        bank_dim=8,
        text_dim=8,
        ctx_len=2,
        late_factor_identity_enabled=True,
        factor_id_scale=0.02,
        factor_id_max_ratio=0.05,
    )
    state_delta = torch.randn(2, 4, 2, 2, 8, requires_grad=True)
    base = torch.randn(1, 1, 2, 2, 8)
    updated, diag = core.apply_late_factor_identity(state_delta, base_context=base)
    residual = updated - state_delta
    ratio = residual.float().norm(dim=-1) / base.detach().float().norm(dim=-1).clamp_min(1e-6)
    assert updated.shape == state_delta.shape
    assert diag["factor_identity_tangent_base_abs_cos_max"].item() < 1e-5
    assert diag["factor_identity_tangent_l2_min"].item() > 1e-4
    assert ratio.max().item() <= 0.05001
    assert diag["context_angle_change_degrees_mean"].item() > 0
    updated.sum().backward()
    assert core.factor_id_projection.weight.grad is not None
    assert core.factor_id_projection.weight.grad.float().norm().item() > 0
    assert core.factor_context_anchors.requires_grad is False
    assert torch.allclose(core.factor_context_anchors[0], core.factor_context_anchors[0])


def test_dynamic_mean_anchor_loss_detaches_hard_anchor_and_uses_factor_mean():
    h6 = H6Progress1(n_groups=1, num_factors=4, bank_dim=8, router_dim=6, text_dim=8, ctx_len=2)
    hard = F.normalize(torch.randn(1, 2, 8, 2), dim=2).requires_grad_(True)
    aligned = hard.unsqueeze(2).expand(1, 2, 4, 8, 2).clone().requires_grad_(True)
    loss, cos = h6.dynamic_mean_anchor_loss(aligned, hard)
    assert loss.item() == 0.0
    assert cos.shape == (1, 2, 2)

    bad = (-hard.detach()).unsqueeze(2).expand(1, 2, 4, 8, 2).clone().requires_grad_(True)
    bad_loss, _ = h6.dynamic_mean_anchor_loss(bad, hard)
    bad_loss.backward()
    assert bad_loss.item() > 0
    assert bad.grad is not None
    assert bad.grad.float().norm().item() > 0
    assert hard.grad is None


def test_p1_v6_checkpoint_rejects_p1_v5_metadata():
    class _Model:
        h6_enabled = True

        def __init__(self):
            self.h6 = H6Progress1(n_groups=1, num_factors=4, bank_dim=8, router_dim=6, text_dim=8, ctx_len=2)

    checkpoint = {
        "checkpoint_version": 5,
        "h6_enabled": True,
        "phase4_progress": 1,
        "h6_config": {"progress_version": "P1-v5-fix"},
    }
    try:
        validate_h6_configuration(_Model(), checkpoint)
    except ValueError as exc:
        assert "checkpoint_version=6" in str(exc)
    else:
        raise AssertionError("P1-v6 accepted a P1-v5 checkpoint")
