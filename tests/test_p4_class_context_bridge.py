import torch

from model.h6.conditional_semantics import ConditionalSemanticCore


def test_cls24_is_detached_and_vae_output_crosses_explicit_context_bridge():
    torch.manual_seed(11)
    core = ConditionalSemanticCore(
        n_groups=2,
        bank_dim=16,
        text_dim=8,
        ctx_len=2,
        vae_hidden_dim=12,
        vae_latent_dim=4,
    )
    cls24 = torch.randn(1, 8, requires_grad=True)
    result = core(
        [torch.randn(1, 3, 8) for _ in range(2)],
        cls24,
        torch.randn(2, 8),
        torch.randn(2, 8),
    )
    result["class_delta_raw"].sum().backward()

    assert cls24.grad is None
    assert core.class_to_context.weight.grad.abs().sum() > 0
    assert result["class_delta_raw"].shape == (1, 2, 8)
