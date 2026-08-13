import torch

from model.h6.conditional_semantics import ConditionalSemanticCore


def _core():
    torch.manual_seed(7)
    return ConditionalSemanticCore(
        n_groups=3,
        bank_dim=32,
        text_dim=16,
        ctx_len=4,
        vae_hidden_dim=24,
        vae_latent_dim=8,
    )


def test_k1_has_one_context_path_and_exact_zero_scale_parity():
    core = _core()
    visual = [torch.randn(2, 5, 16) for _ in range(3)]
    cls24 = torch.randn(2, 16)
    ctx_normal = torch.randn(4, 16)
    ctx_abnormal = torch.randn(4, 16)

    result = core(
        visual,
        cls24,
        ctx_normal,
        ctx_abnormal,
        state_scale=0.0,
        class_scale=0.0,
    )

    assert result["dynamic_contexts"].shape == (2, 1, 2, 4, 16)
    assert torch.equal(result["dynamic_contexts"], result["base_contexts"])
    assert torch.isfinite(result["dynamic_contexts"]).all()


def test_state_and_class_context_paths_receive_gradient():
    core = _core()
    result = core(
        [torch.randn(2, 5, 16) for _ in range(3)],
        torch.randn(2, 16),
        torch.randn(4, 16),
        torch.randn(4, 16),
    )
    result["dynamic_contexts"].square().mean().backward()

    assert core.state_to_context_normal.weight.grad.abs().sum() > 0
    assert core.state_to_context_abnormal.weight.grad.abs().sum() > 0
    assert core.class_to_context.weight.grad.abs().sum() > 0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in core.class_vae.parameters()
    )
