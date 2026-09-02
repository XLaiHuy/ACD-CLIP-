import json
import math
from pathlib import Path

import torch
import pytest
from torch import nn

from h2_clean.contract import (
    ANCHOR_FAMILY_BUDGET_DEFAULT,
    ANCHOR_LAMBDA_ACTIVE,
    ANCHOR_LAMBDA_OLD,
    ANCHOR_R_MED,
    ANCHOR_TARGET_EFFECTIVE_RATIO,
    PRIMARY_HORIZON,
    SafeImageAdapterAnchor,
    CIR_LOGIT_SHIFT_EXPERIMENTAL,
    SECONDARY_HORIZON,
    TRAINING_HORIZON,
    scientific_config_from_mapping,
)
from h2_clean.cir_v2 import select_gt_free_peers


class AdapterHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.large = nn.Parameter(torch.ones(4))
        self.zero_reference = nn.Parameter(torch.zeros(4))


def test_safe_anchor_is_global_and_immutable():
    module = AdapterHolder()
    anchor = SafeImageAdapterAnchor.from_module(module)
    assert torch.equal(anchor.reference["large"], torch.ones(4))
    assert torch.equal(anchor.reference["zero_reference"], torch.zeros(4))
    assert anchor.loss(module).item() == 0.0

    with torch.no_grad():
        module.large.add_(0.1)
        module.zero_reference.fill_(0.2)
    loss = anchor.loss(module)
    expected = (4 * 0.1**2 + 4 * 0.2**2) / (4.0 + anchor.eps)
    assert math.isclose(loss.item(), expected, rel_tol=1e-5, abs_tol=1e-8)
    loss.backward()
    assert torch.isfinite(module.large.grad).all()
    assert torch.isfinite(module.zero_reference.grad).all()
    assert torch.equal(anchor.reference["large"], torch.ones(4))
    assert torch.equal(anchor.reference["zero_reference"], torch.zeros(4))


def test_safe_anchor_zero_reference_cannot_dominate():
    module = AdapterHolder()
    anchor = SafeImageAdapterAnchor.from_module(module)
    with torch.no_grad():
        module.zero_reference.fill_(1.0)
    loss = anchor.loss(module)
    assert math.isclose(loss.item(), 1.0, rel_tol=1e-5)
    assert loss.item() < 10.0


def test_experimental_logit_shift_is_disabled():
    native = torch.zeros(1, 1, 2, 4, requires_grad=True)
    features = torch.ones(1, 1, 4, 3)
    with pytest.raises(RuntimeError, match="disabled experimental logit shift"):
        CIR_LOGIT_SHIFT_EXPERIMENTAL(native, features, 0.5, peer_count=3, spatial_radius=3)


def test_cir_peer_selector_has_finite_gt_free_indices():
    torch.manual_seed(19)
    features = torch.randn(2, 3, 16, 4, requires_grad=True)
    margins = torch.randn(2, 3, 16, requires_grad=True)
    info = select_gt_free_peers(features, margins, peer_count=2, spatial_radius=0)
    assert info["peer_indices"].dtype == torch.long
    assert info["peer_indices"].shape == (3, 16, 2)
    assert torch.isfinite(info["candidate_count"].float()).all()
    assert info["valid"].all()
    assert not info["peer_indices"].requires_grad
    assert not info["valid"].requires_grad


def test_e20_protocol_identity_and_single_anchor_calibration():
    repo = Path(__file__).resolve().parents[1]
    config = json.loads((repo / "configs" / "h2_clean_factorial_v1.json").read_text())
    assert config["horizon"] == TRAINING_HORIZON == 20
    assert config["training_horizon"] == TRAINING_HORIZON
    assert config["primary_horizon"] == PRIMARY_HORIZON == 15
    assert config["secondary_horizon"] == SECONDARY_HORIZON == 20
    assert config["target_valid_epochs"] == [PRIMARY_HORIZON, SECONDARY_HORIZON]
    calibration = config["anchor_calibration"]
    assert calibration["valid_rows"] == [
        "historical_E5/vision_text_k",
        "historical_E10/vision_text_k",
        "historical_E15/vision_text_k",
    ]
    assert calibration["lambda_sweep"] == "NO"
    assert calibration["r_med"] == ANCHOR_R_MED
    assert calibration["target_effective_ratio"] == ANCHOR_TARGET_EFFECTIVE_RATIO
    assert calibration["lambda_old"] == ANCHOR_LAMBDA_OLD
    assert calibration["lambda_active"] == ANCHOR_LAMBDA_ACTIVE
    identity = scientific_config_from_mapping(
        {
            "protocol_horizon": TRAINING_HORIZON,
            "anchor_lambda": ANCHOR_LAMBDA_ACTIVE,
            "anchor_gradient_budget": True,
            "anchor_family_budget": ANCHOR_FAMILY_BUDGET_DEFAULT,
        },
        clip_sha256=None,
        dataset_manifest_sha256=None,
        implementation_git_sha=None,
    )
    assert identity["epoch"] == TRAINING_HORIZON
    assert identity["training_horizon"] == TRAINING_HORIZON
    assert identity["primary_horizon"] == PRIMARY_HORIZON
    assert identity["secondary_horizon"] == SECONDARY_HORIZON
    assert identity["anchor_lambda"] == ANCHOR_LAMBDA_ACTIVE
    assert identity["anchor_gradient_budget"] is True
    assert identity["anchor_family_budget"] == ANCHOR_FAMILY_BUDGET_DEFAULT
