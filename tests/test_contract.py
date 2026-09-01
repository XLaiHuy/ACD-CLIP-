import math

import torch
from torch import nn

from h2_clean.contract import (
    SafeImageAdapterAnchor,
    cir_adjust_native_logits,
    select_gt_free_peers,
)


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


def test_cir_alpha_zero_output_and_gradient_identity():
    torch.manual_seed(7)
    native = torch.randn(2, 2, 2, 16, requires_grad=True)
    features = torch.randn(2, 2, 16, 5, requires_grad=True)
    adjusted, stats = cir_adjust_native_logits(native, features, 0.0, peer_count=2)
    assert adjusted is native
    assert stats == {"enabled": False, "alpha": 0.0}
    adjusted.sum().backward()
    assert torch.equal(native.grad, torch.ones_like(native))
    assert features.grad is None


def test_cir_nonzero_is_detached_finite_and_changes_logits():
    torch.manual_seed(11)
    native = torch.randn(2, 2, 2, 16, requires_grad=True)
    features = torch.randn(2, 2, 16, 5, requires_grad=True)
    adjusted, stats = cir_adjust_native_logits(
        native,
        features,
        0.2,
        peer_count=2,
        spatial_radius=0,
    )
    assert stats["enabled"] is True
    assert stats["valid"].all()
    assert torch.isfinite(stats["delta"]).all()
    assert not stats["delta"].requires_grad
    assert not stats["peer_indices"].requires_grad
    assert torch.isfinite(adjusted).all()
    assert not torch.equal(adjusted, native)
    adjusted.sum().backward()
    assert torch.isfinite(native.grad).all()
    assert features.grad is None


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


def test_cir_invalid_sparse_peer_queries_have_zero_shift():
    native = torch.zeros(1, 1, 2, 4, requires_grad=True)
    features = torch.ones(1, 1, 4, 3)
    adjusted, stats = cir_adjust_native_logits(
        native,
        features,
        0.5,
        peer_count=3,
        spatial_radius=3,
    )
    assert not stats["valid"].any()
    assert torch.equal(stats["delta"], torch.zeros_like(stats["delta"]))
    assert torch.equal(adjusted, native)
