import pytest
import torch

from tools.cir_rmt.core import midpoint_median, robust_peer_delta


def test_even_midpoint_median_k8():
    values = torch.tensor([8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0])
    assert midpoint_median(values).item() == pytest.approx(4.5)


def test_mad_and_delta_are_finite_and_bounded():
    peers = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    observed = torch.tensor([9.0])
    delta, stats = robust_peer_delta(observed, peers, return_stats=True)
    assert stats["center"].item() == pytest.approx(3.5)
    assert stats["mad"].item() == pytest.approx(2.0)
    assert -1.0 < delta.item() < 1.0
    assert torch.isfinite(delta).all()


def test_peer_permutation_does_not_change_robust_evidence():
    observed = torch.tensor([0.15, -0.8])
    peers = torch.tensor([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [-1.0, -0.5, -0.2, 0.0, 0.1, 0.2, 0.5, 1.0]])
    permuted = peers[:, torch.tensor([7, 1, 5, 0, 3, 6, 2, 4])]
    assert torch.allclose(robust_peer_delta(observed, peers), robust_peer_delta(observed, permuted))


def test_tiny_mad_uses_eps_without_nan():
    peers = torch.full((2, 8), 0.25)
    observed = torch.tensor([0.25, 100.0])
    delta = robust_peer_delta(observed, peers)
    assert torch.isfinite(delta).all()
    assert torch.all(delta.abs() <= 1.0)
