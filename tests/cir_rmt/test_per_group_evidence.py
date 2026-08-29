import pytest
import torch

from tools.cir_rmt.core import cir_logits_from_native_weights, peer_delta_from_native_margins, robust_peer_delta
from tools.cir_rmt.runtime import _per_group_margins


def test_per_group_margin_axis_isolated_before_transport():
    torch.manual_seed(101)
    stages, batch, patches, groups, dim = 3, 2, 16, 3, 9
    features = torch.randn(stages, batch, patches, dim)
    text = torch.randn(batch, groups, dim, 2)
    baseline = _per_group_margins(features, text)
    changed_text = text.clone()
    changed_text[:, 1] = torch.roll(changed_text[:, 1], shifts=1, dims=1)
    changed = _per_group_margins(features, changed_text)
    difference = (changed - baseline).abs()
    assert difference[..., 0].max().item() <= 1e-6
    assert difference[..., 2].max().item() <= 1e-6
    assert difference[..., 1].max().item() > 1e-5


def test_peer_delta_has_stage_group_geometry_and_group_isolation():
    torch.manual_seed(102)
    stages, batch, patches, groups, dim = 3, 2, 49, 3, 8
    features = torch.nn.functional.normalize(torch.randn(stages, batch, patches, dim), dim=-1)
    margins = torch.randn(stages, batch, patches, groups)
    delta, stats = peer_delta_from_native_margins(features, margins)
    assert delta.shape == (stages, batch, patches, groups)
    assert stats["peer_margins"].shape == (stages, batch, patches, 8, groups)
    changed = margins.clone()
    changed[0, 0, 0, 1] += 3.0
    baseline_fixed = robust_peer_delta(margins, stats["peer_margins"], peer_dim=-2)
    changed_fixed = robust_peer_delta(changed, stats["peer_margins"], peer_dim=-2)
    difference = (changed_fixed - baseline_fixed).abs()
    assert difference[..., 0].max().item() <= 1e-6
    assert difference[..., 2].max().item() <= 1e-6
    assert difference[0, 0, 0, 1].item() > 1e-5


def test_visual_stage_and_text_group_axes_are_not_interchangeable():
    torch.manual_seed(103)
    stages, batch, patches, groups, dim = 3, 1, 16, 3, 7
    image = torch.nn.functional.normalize(torch.randn(stages, batch, patches, dim), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(batch, groups, dim, 2), dim=-2)
    native = torch.rand(stages, batch, groups, 2) + 0.2
    native = native / native.sum(dim=-2, keepdim=True)
    delta = torch.randn(stages, batch, patches, groups).tanh()
    with pytest.raises(ValueError, match=r"legacy \[B,P\] or contract \[S,B,P,G\]"):
        cir_logits_from_native_weights(image, text, native, delta.mean(dim=-1), 0.5)
    scores, _ = cir_logits_from_native_weights(image, text, native, delta, 0.5, score_mode="reference")
    swapped = delta.permute(3, 1, 2, 0).contiguous()
    swapped_scores, _ = cir_logits_from_native_weights(image, text, native, swapped, 0.5, score_mode="reference")
    assert scores.shape == swapped_scores.shape == (stages, batch, patches, 2)
    assert (scores - swapped_scores).abs().max().item() > 1e-6


def test_per_group_peer_permutation_invariance():
    torch.manual_seed(104)
    observed = torch.randn(3, 2, 16, 3)
    peers = torch.randn(3, 2, 16, 8, 3)
    permutation = torch.tensor([7, 1, 5, 0, 3, 6, 2, 4])
    assert torch.allclose(
        robust_peer_delta(observed, peers, peer_dim=-2),
        robust_peer_delta(observed, peers[:, :, :, permutation, :], peer_dim=-2),
    )
