from __future__ import annotations

import pytest
import torch

from model.h6.losses import build_semantic_roles
from model.h6.model import H6Progress1
from model.h6.utility_routing import (
    r2_region_normalized_utility_router_loss,
    r2_responsibility_balanced_utility_router_loss,
    utility_teacher,
)


SCALE = 0.0005203147302381694


def test_r2_teacher_preserves_gap_magnitude_and_shapes():
    base = torch.zeros(1, 1, 2)
    # The same winner (role 0) with two different utility-gap magnitudes.
    evidence = torch.tensor([[[[0.10, 0.00], [0.20, 0.00]]]])
    y = torch.ones(1, 2)
    valid = torch.ones(1, 2, dtype=torch.bool)
    payload = utility_teacher(
        base, evidence, y, valid, rho=0.05,
        role_topology="r2_normal_anomaly", role_teacher_scale=SCALE,
    )
    assert payload["q_factor_utility"].shape == (1, 1, 2, 2)
    assert torch.allclose(payload["q_factor_utility"].sum(-1), torch.ones(1, 1, 2))
    p = payload["role_probability"][0, 0]
    assert p[0] > 0.5 and p[1] > p[0]
    assert torch.allclose(payload["q_router_utility"], payload["q_factor_utility"])
    assert payload["q_utility"].requires_grad is False
    assert torch.isfinite(payload["role_entropy"]).all()


def test_r2_teacher_rejects_degenerate_configuration():
    base = torch.zeros(1, 1, 1)
    evidence = torch.zeros(1, 1, 1, 2)
    y = torch.zeros(1, 1)
    valid = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="positive global role_teacher_scale"):
        utility_teacher(
            base, evidence, y, valid, role_topology="r2_normal_anomaly",
        )


def test_r2_semantic_roles_are_patch_level_normal_background_vs_anomaly():
    masks = torch.zeros(2, 8, 8)
    masks[1, 0:2, 0:2] = 1.0
    labels = torch.tensor([0, 1])
    valid = torch.ones_like(masks)
    q_role, hard_role, coverage, local_valid, local_image = build_semantic_roles(
        masks, labels, patch_count=4, local_mask_valid=valid,
        num_roles=2, role_topology="r2_normal_anomaly", boundary_threshold=0.01,
    )
    assert q_role.shape == (2, 4, 2)
    assert torch.all(hard_role[0] == 0)
    assert torch.any(hard_role[1] == 1)
    assert torch.any(hard_role[1] == 0)  # anomaly-image background remains normal role
    assert torch.allclose(q_role.sum(-1), torch.ones(2, 4))
    assert torch.all(local_valid)
    assert torch.equal(local_image, torch.tensor([False, True]))


def test_r2_model_config_and_shapes_are_explicit():
    model = H6Progress1(
        n_groups=1, num_factors=2, top_k=2, bank_dim=8, router_dim=4,
        vae_hidden_dim=8, vae_latent_dim=4, text_dim=8, ctx_len=2,
        progress_version="P1-v8.4-A", role_topology="r2_normal_anomaly",
        role_teacher_scale=SCALE, prediction_routing="dense",
    )
    config = model.config_dict()
    assert config["num_factors"] == 2
    assert config["role_topology"] == "r2_normal_anomaly"
    assert config["role_teacher_scale"] == pytest.approx(SCALE)
    assert config["structured_text_layout"].startswith("[R_NORMAL][R_ANOMALY]")
    assert model.rho_values().shape == (1,)


def test_r2_router_responsibility_balanced_loss_is_finite_and_role_normalized():
    q = torch.tensor([[[[0.35, 0.65], [0.35, 0.65]]]])
    dense_logits = torch.tensor([[[[0.2, -0.2], [0.1, -0.1]]]], requires_grad=True)
    dense = torch.softmax(dense_logits, dim=-1)
    payload = {
        "role_topology": "r2_normal_anomaly",
        "q_utility": q,
        "informative": torch.ones(1, 1, 2, dtype=torch.bool),
    }
    role_weights = torch.tensor((1.0 / (2 * 0.35), 1.0 / (2 * 0.65)))
    role_mass = (q * role_weights).sum(dim=(0, 1, 2))
    assert torch.allclose(role_mass / role_mass.sum(), torch.tensor([0.5, 0.5]))
    loss = r2_responsibility_balanced_utility_router_loss(
        dense, payload, role_weights=tuple(float(value) for value in role_weights),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert dense_logits.grad is not None
    assert torch.isfinite(dense_logits.grad).all()



def _region_payload(q: torch.Tensor, informative: torch.Tensor) -> dict[str, torch.Tensor]:
    gap = torch.tensor([[[0.0002, -0.0010, 0.0004]]], dtype=q.dtype)
    return {
        "role_topology": "r2_normal_anomaly",
        "q_utility": q,
        "informative": informative,
        "valid": torch.ones_like(informative),
        "role_gap": gap,
        "role_scale": torch.full_like(gap, SCALE),
    }


def test_r2_region_normalized_loss_balances_present_regions_and_keeps_role_gradient():
    q = torch.tensor([[[[0.99, 0.01], [0.02, 0.98], [0.40, 0.60]]]])
    logits = torch.tensor([[[[0.2, -0.2], [0.1, -0.1], [-0.3, 0.3]]]], requires_grad=True)
    dense = torch.softmax(logits, dim=-1)
    targets = torch.tensor([[0.0, 1.0, 0.0]])
    loss, components = r2_region_normalized_utility_router_loss(
        dense, _region_payload(q, torch.ones(1, 1, 3, dtype=torch.bool)), targets,
        return_components=True,
    )
    assert torch.isfinite(loss)
    assert components["normal_support_count"].item() == 2
    assert components["anomaly_support_count"].item() == 1
    assert components["normal_contribution"].item() == pytest.approx(
        0.5 * components["normal_loss"].item()
    )
    assert components["anomaly_contribution"].item() == pytest.approx(
        0.5 * components["anomaly_loss"].item()
    )
    assert 1.0 <= components["utility_weight_min"].item() <= components["utility_weight_max"].item() <= 2.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    # The anomaly teacher strongly favors role 1, so descent increases logit 1 relative to role 0.
    assert logits.grad[0, 0, 1, 1] < logits.grad[0, 0, 1, 0]


@pytest.mark.parametrize("target", [0.0, 1.0])
def test_r2_region_normalized_loss_handles_one_present_region(target: float):
    q = torch.tensor([[[[0.95, 0.05] if target == 0.0 else [0.05, 0.95]]]])
    logits = torch.zeros(1, 1, 1, 2, requires_grad=True)
    loss, components = r2_region_normalized_utility_router_loss(
        torch.softmax(logits, dim=-1),
        _region_payload(q, torch.ones(1, 1, 1, dtype=torch.bool)),
        torch.tensor([[target]]),
        return_components=True,
    )
    assert torch.isfinite(loss)
    assert bool(components["normal_present"]) is (target == 0.0)
    assert bool(components["anomaly_present"]) is (target == 1.0)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
