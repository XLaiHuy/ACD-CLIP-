import torch

from model.h6.conditional_semantics import predictor_aligned_abnormal_residual


def test_predictor_residual_is_abnormal_only_and_base_reference_is_detached():
    base = torch.randn(3, 2, 5, 2, requires_grad=True)
    dynamic_abnormal = torch.randn(3, 2, 5, requires_grad=True)
    rho = torch.full((3,), 0.05)

    result = predictor_aligned_abnormal_residual(base, dynamic_abnormal, rho)
    final = result["final_group_logits"]

    assert torch.equal(final[..., 0], base[..., 0])
    assert result["normal_invariant_error"].item() == 0.0
    expected = dynamic_abnormal - base[..., 1].detach()
    assert torch.equal(result["predictor_residual_logits"], expected)

    result["predictor_residual_logits"].sum().backward()
    assert base.grad is None
    assert dynamic_abnormal.grad is not None
    assert torch.equal(dynamic_abnormal.grad, torch.ones_like(dynamic_abnormal))
