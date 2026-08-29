import torch

from tools.cir_rmt.core import transport_pair, transport_weights


def test_transport_normalizes_and_moves_in_expected_direction():
    native = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    delta = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32)
    normal, abnormal = transport_pair(native, native, delta, alpha=1.0)
    assert torch.allclose(normal.sum(-1), torch.ones(1))
    assert torch.allclose(abnormal.sum(-1), torch.ones(1))
    assert abnormal[0, 2] > native[0, 2]
    assert normal[0, 2] < native[0, 2]


def test_alpha_zero_is_exact_and_relative_weights_are_preserved():
    native = torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float32)
    delta = torch.tensor([[-2.0, -2.0, -2.0]], dtype=torch.float32)
    out = transport_weights(native, delta, alpha=0.0)
    assert torch.equal(out, native)
    shifted = transport_weights(native, delta, alpha=0.4)
    ratio_before = native[0, 2] / native[0, 1]
    ratio_after = shifted[0, 2] / shifted[0, 1]
    assert torch.allclose(ratio_before, ratio_after)


def test_delta_is_stop_gradient_but_native_weights_remain_differentiable():
    native = torch.tensor([[0.2, 0.3, 0.5]], requires_grad=True)
    delta = torch.tensor([[-0.5, 0.0, 0.5]], requires_grad=True)
    transport_weights(native, delta, alpha=0.7).sum().backward()
    assert native.grad is not None
    assert delta.grad is None
