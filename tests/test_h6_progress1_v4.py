import torch
import torch.nn.functional as F

from model.h6.losses import (
    concept_key_diversity_loss,
    dynamic_residual_diversity_loss,
    factor_aware_center_loss,
    router_teacher_loss,
)
from model.h6.model import H6Progress1
from model.h6.router import PatchRouter
from model.h6.semantic_bank import deterministic_slot_directions
from train import factor_gradient_diagnostics, linear_ramp_weight


def test_slot_init_is_reproducible_distinct_and_rng_safe():
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    directions_a = deterministic_slot_directions(4, 16, 6100)
    after = torch.random.get_rng_state()
    directions_b = deterministic_slot_directions(4, 16, 6100)
    assert torch.equal(before, after)
    assert torch.allclose(directions_a, directions_b)
    assert torch.allclose(directions_a @ directions_a.T, torch.eye(4), atol=1e-5)

    torch.manual_seed(7)
    h6_a = H6Progress1(n_groups=3, slot_init_enabled=True, slot_init_scale=0.02, slot_init_seed_offset=6100)
    torch.manual_seed(7)
    h6_b = H6Progress1(n_groups=3, slot_init_enabled=True, slot_init_scale=0.02, slot_init_seed_offset=6100)
    assert torch.allclose(h6_a.semantic_core.concept_slots, h6_b.semantic_core.concept_slots)
    assert h6_a.config_dict()["slot_init_applied_components"]
    rows = h6_a.semantic_core.concept_slots.detach()
    assert not torch.allclose(rows[0], rows[1])
    assert len({rows[i].storage_offset() for i in range(rows.shape[0])}) == rows.shape[0]
    diag = h6_a.semantic_core.initialization_diagnostics()
    assert torch.isfinite(diag["slot_initial_l2_min"])
    assert diag["slot_initial_l2_min"] > 0


def test_three_level_router_uses_distinct_inputs_and_diagnostics():
    router = PatchRouter(n_groups=3, num_factors=4, text_dim=8, bank_dim=4, hidden_dim=12, top_k=2)
    levels = [
        F.normalize(torch.ones(2, 5, 8), dim=-1),
        F.normalize(torch.randn(2, 5, 8) + 2.0, dim=-1),
        F.normalize(torch.randn(2, 5, 8) - 2.0, dim=-1),
    ]
    keys = F.normalize(torch.randn(4, 4), dim=-1)
    out = router(levels, epoch_one_based=12, concept_keys=keys)
    assert out["queries"].shape[:3] == (3, 2, 5)
    assert out["level_input_alias"].item() is False
    assert out["level_input_difference"] > 0
    assert out["level_query_difference"] > 0
    assert out["level_logit_difference"] > 0
    assert not torch.allclose(out["queries"][0], out["queries"][1])


def test_concept_key_diversity_margin_loss_and_gradients():
    identical = torch.ones(4, 8, requires_grad=True)
    loss = concept_key_diversity_loss(identical, margin=0.5)
    assert loss > 0
    loss.backward()
    assert identical.grad is not None
    assert torch.isfinite(identical.grad).all()

    separated = torch.eye(4, 8, requires_grad=True)
    zeroish = concept_key_diversity_loss(separated, margin=0.5)
    assert zeroish.item() == 0.0
    zeroish.backward()
    assert torch.isfinite(separated.grad).all()

    assert linear_ramp_weight(1, 1, 3, 0.0001) > 0
    assert linear_ramp_weight(3, 1, 3, 0.0001) == 0.0001


def test_asymmetric_factor_gradients_are_detected_and_symmetric_control_allowed():
    projected = torch.randn(1, 1, 4, 3)
    normal = F.normalize(torch.randn(1, 4, 3), dim=-1).requires_grad_()
    abnormal = F.normalize(torch.randn(1, 4, 3), dim=-1).requires_grad_()
    dense = F.softmax(torch.tensor([[[[4.0, 1.0, 0.0, -1.0],
                                      [0.0, 4.0, 1.0, -1.0],
                                      [-1.0, 0.0, 4.0, 1.0],
                                      [1.0, -1.0, 0.0, 4.0]]]]), dim=-1)
    masks = torch.zeros(1, 1, 2, 2)
    labels = torch.tensor([0])
    loss = factor_aware_center_loss(projected, normal, abnormal, dense, masks, labels, detach_assignment=True)
    loss.backward()
    diag = factor_gradient_diagnostics(normal.grad)
    assert torch.isfinite(diag["factor_grad_norms"]).all()
    assert diag["factor_grad_l2_min"] > 0

    symmetric = torch.ones(4, 3)
    control = factor_gradient_diagnostics(symmetric)
    assert control["factor_grad_l2_min"].item() == 0.0


def test_teacher_and_dynamic_residual_audit_behavior():
    projected = torch.randn(1, 1, 4, 3)
    proto = F.normalize(torch.randn(1, 4, 3), dim=-1)
    dense = torch.full((1, 1, 4, 4), 0.25)
    masks = torch.zeros(1, 1, 2, 2)
    labels = torch.tensor([0])
    loss, diag = router_teacher_loss(projected, proto, proto, dense, masks, labels, temperature=0.15)
    assert torch.isfinite(loss)
    assert diag["teacher_entropy"].shape == (1,)
    assert diag["teacher_usage"].shape == (1, 4)
    assert diag["teacher_unique_topk_pairs"].shape == (1,)
    assert torch.isfinite(diag["teacher_router_kl"]).all()

    hard = torch.zeros(1, 1, 768, 2)
    dynamic = torch.zeros(1, 1, 4, 768, 2)
    dynamic[..., 0, 1] = 1.0
    dynamic = dynamic.requires_grad_()
    orth = dynamic_residual_diversity_loss(dynamic, hard)
    orth.backward()
    assert abs(orth.item() - 0.75) < 1e-6
    assert dynamic.grad is not None
    assert torch.isfinite(dynamic.grad).all()
