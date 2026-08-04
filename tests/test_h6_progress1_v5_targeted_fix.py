import torch

from model.h6.losses import router_teacher_loss
from model.h6.semantic_bank import CoPSSemanticCore


def test_late_factor_identity_adds_capped_residual_and_updates_projection():
    torch.manual_seed(0)
    core = CoPSSemanticCore(
        n_groups=1,
        num_factors=4,
        bank_dim=8,
        text_dim=8,
        ctx_len=2,
        late_factor_identity_enabled=True,
        factor_id_scale=0.02,
        factor_id_max_ratio=0.05,
    )
    state_delta_raw = torch.randn(2, 4, 2, 2, 8, requires_grad=True)
    updated, diagnostics = core.apply_late_factor_identity(state_delta_raw)
    residual = updated - state_delta_raw
    ratio = residual.float().norm(dim=-1) / state_delta_raw.detach().float().norm(dim=-1).clamp_min(1e-6)

    assert updated.shape == state_delta_raw.shape
    assert residual.float().norm().item() > 0
    assert ratio.max().item() <= 0.05001
    assert diagnostics["factor_id_residual_to_context_ratio_max"].item() <= 0.05001

    updated.sum().backward()
    assert core.factor_id_projection.weight.grad is not None
    assert core.factor_id_projection.weight.grad.float().norm().item() > 0


def test_late_factor_identity_disabled_is_exact_passthrough():
    core = CoPSSemanticCore(
        n_groups=1,
        num_factors=4,
        bank_dim=8,
        text_dim=8,
        ctx_len=2,
        late_factor_identity_enabled=False,
    )
    state_delta_raw = torch.randn(2, 4, 2, 2, 8)
    updated, diagnostics = core.apply_late_factor_identity(state_delta_raw)
    assert torch.equal(updated, state_delta_raw)
    assert diagnostics["factor_id_residual_norm_mean"].item() == 0.0


def test_state_centered_router_teacher_updates_router_only():
    torch.manual_seed(1)
    projected = torch.randn(1, 2, 4, 8)
    prototype_normal = torch.randn(2, 4, 8, requires_grad=True)
    prototype_abnormal = torch.randn(2, 4, 8, requires_grad=True)
    dense_probabilities = torch.full((1, 2, 4, 4), 0.25, requires_grad=True)
    masks = torch.zeros(2, 1, 2, 2)
    masks[1, 0, 0, 0] = 1.0
    labels = torch.tensor([0, 1])

    loss, diagnostics = router_teacher_loss(
        projected,
        prototype_normal,
        prototype_abnormal,
        dense_probabilities,
        masks,
        labels,
        temperature=0.15,
        mode="state_centered_cosine",
        confidence_gate_enabled=True,
        entropy_threshold=1.0,
        probability_std_threshold=0.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert dense_probabilities.grad is not None
    assert dense_probabilities.grad.float().norm().item() > 0
    assert prototype_normal.grad is None
    assert prototype_abnormal.grad is None
    assert diagnostics["teacher_informative_patch_count"].sum().item() > 0
    assert diagnostics["teacher_active_levels"].all().item() is True


def test_router_teacher_patch_gate_zeroes_uninformative_teacher():
    projected = torch.zeros(1, 1, 4, 8)
    prototype_normal = torch.zeros(1, 4, 8)
    prototype_abnormal = torch.zeros(1, 4, 8)
    dense_probabilities = torch.full((1, 1, 4, 4), 0.25, requires_grad=True)
    masks = torch.zeros(1, 1, 2, 2)
    labels = torch.tensor([0])

    loss, diagnostics = router_teacher_loss(
        projected,
        prototype_normal,
        prototype_abnormal,
        dense_probabilities,
        masks,
        labels,
        mode="state_centered_cosine",
        confidence_gate_enabled=True,
        entropy_threshold=0.98,
        probability_std_threshold=0.001,
    )

    assert loss.item() == 0.0
    assert diagnostics["teacher_informative_patch_count"].sum().item() == 0
    assert diagnostics["teacher_valid_patch_count"].sum().item() == 4
    assert diagnostics["teacher_active_levels"].any().item() is False
    assert diagnostics["teacher_gate_reason"].item() == 1
