import math

import torch
import pytest
from torch import nn

from h2_clean.contract import (
    SafeImageAdapterAnchor,
    CIR_LOGIT_SHIFT_EXPERIMENTAL,
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
