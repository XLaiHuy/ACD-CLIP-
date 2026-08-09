import torch

from model.h6.utility_routing import (
    effective_number_utility_factor_loss,
    support_normalized_utility_router_loss,
)
from train import (
    apply_primary_anchored_factor_correction,
    pcgrad_project_two_task,
    primary_anchored_factor_surgery,
)


def _payload(losses, valid, informative=None):
    losses = torch.tensor(losses, dtype=torch.float32, requires_grad=True)
    valid = torch.tensor(valid, dtype=torch.bool)
    groups, batch, patches, factors = losses.shape
    responsibility = torch.full_like(losses, 1.0 / factors)
    q = torch.full_like(losses, 1.0 / factors)
    return losses, {
        "loss_per_factor": losses,
        "responsibility": responsibility,
        "q_utility": q,
        "valid": valid,
        "informative": valid.clone() if informative is None else torch.tensor(informative, dtype=torch.bool),
    }


def test_effective_number_single_region_reduces_to_valid_mean():
    losses, payload = _payload(
        [[[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]]],
        [[[True, True, False]]],
    )
    target = torch.zeros(1, 3)
    result = effective_number_utility_factor_loss(payload, target, beta=0.99)
    assert result.item() == 2.0
    result.backward()
    assert losses.grad is not None and torch.isfinite(losses.grad).all()


def test_effective_number_is_patch_weighted_not_region_mean_weighted():
    _, payload = _payload(
        [[[[1.0, 1.0], [1.0, 1.0], [9.0, 9.0]]]],
        [[[True, True, True]]],
    )
    target = torch.tensor([[0.0, 0.0, 1.0]])
    beta = 0.99
    result = effective_number_utility_factor_loss(payload, target, beta=beta)
    effective_normal = (1.0 - beta**2) / (1.0 - beta)
    expected = ((2.0 / effective_normal) + 9.0) / ((2.0 / effective_normal) + 1.0)
    assert result.item() == torch.tensor(expected).item()


def test_support_aware_router_uses_all_valid_denominator():
    probabilities = torch.tensor([[[[0.8, 0.2], [0.6, 0.4], [0.5, 0.5], [0.5, 0.5]]]], requires_grad=True)
    teacher = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]])
    payload = {
        "q_utility": teacher,
        "valid": torch.ones(1, 1, 4, dtype=torch.bool),
        "informative": torch.tensor([[[True, False, False, False]]]),
    }
    result = support_normalized_utility_router_loss(probabilities, payload)
    assert torch.allclose(result, -torch.log(torch.tensor(0.8)) / 4.0)
    result.backward()
    assert probabilities.grad is not None


def test_support_aware_router_zero_informative_has_zero_gradient():
    probabilities = torch.full((1, 1, 3, 2), 0.5, requires_grad=True)
    payload = {
        "q_utility": torch.full_like(probabilities, 0.5),
        "valid": torch.ones(1, 1, 3, dtype=torch.bool),
        "informative": torch.zeros(1, 1, 3, dtype=torch.bool),
    }
    result = support_normalized_utility_router_loss(probabilities, payload)
    result.backward()
    assert result.item() == 0.0
    assert torch.equal(probabilities.grad, torch.zeros_like(probabilities))


def test_support_aware_router_scales_with_informative_fraction():
    probabilities = torch.tensor([[[[0.8, 0.2], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]])
    teacher = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]])
    common = {
        "q_utility": teacher,
        "informative": torch.tensor([[[True, False, False, False]]]),
    }
    loss_two = support_normalized_utility_router_loss(
        probabilities, {**common, "valid": torch.tensor([[[True, True, False, False]]])}
    )
    loss_four = support_normalized_utility_router_loss(
        probabilities, {**common, "valid": torch.ones(1, 1, 4, dtype=torch.bool)}
    )
    assert torch.allclose(loss_four, loss_two / 2.0)


def test_pcgrad_aligned_gradients_are_unchanged():
    main = [torch.tensor([1.0, 0.0])]
    factor = [torch.tensor([2.0, 1.0])]
    projected_main, projected_factor, report = pcgrad_project_two_task(main, factor)
    assert not report["conflict"]
    assert torch.equal(projected_main[0], main[0])
    assert torch.equal(projected_factor[0], factor[0])


def test_pcgrad_conflicting_gradients_are_projected_without_nan():
    main = [torch.tensor([1.0, 0.0]), torch.tensor([0.0])]
    factor = [torch.tensor([-1.0, 1.0]), torch.tensor([0.0])]
    projected_main, projected_factor, report = pcgrad_project_two_task(main, factor)
    assert report["conflict"]
    assert torch.dot(projected_main[0], factor[0]).abs().item() < 1e-7
    assert torch.dot(projected_factor[0], main[0]).abs().item() < 1e-7
    assert all(torch.isfinite(value).all() for value in projected_main + projected_factor)


def test_primary_anchored_aligned_gradients_are_unchanged():
    main = [torch.tensor([1.0, 0.0])]
    factor = [torch.tensor([2.0, 1.0])]
    safe_main, safe_factor, report = primary_anchored_factor_surgery(main, factor)
    assert not report["conflict"]
    assert torch.equal(safe_main[0], main[0])
    assert torch.equal(safe_factor[0], factor[0])
    assert report["main_gradient_exact_change_norm"] == 0.0


def test_primary_anchored_conflict_preserves_main_and_projects_only_factor():
    main = [torch.tensor([1.0, 0.0])]
    factor = [torch.tensor([-1.0, 1.0])]
    safe_main, safe_factor, report = primary_anchored_factor_surgery(main, factor)
    assert report["conflict"]
    assert torch.equal(safe_main[0], main[0])
    assert torch.dot(main[0], safe_factor[0]).abs().item() < 1e-7
    assert torch.equal(safe_factor[0], torch.tensor([0.0, 1.0]))
    assert report["main_gradient_exact_change_norm"] == 0.0
    assert report["removed_factor_component_norm"] == 1.0


def test_primary_anchored_zero_norms_are_finite_and_unchanged():
    for main, factor in (
        ([torch.zeros(2)], [torch.tensor([-1.0, 1.0])]),
        ([torch.tensor([1.0, 0.0])], [torch.zeros(2)]),
    ):
        safe_main, safe_factor, report = primary_anchored_factor_surgery(main, factor)
        assert not report["conflict"]
        assert torch.equal(safe_main[0], main[0])
        assert torch.equal(safe_factor[0], factor[0])
        assert torch.isfinite(safe_main[0]).all()
        assert torch.isfinite(safe_factor[0]).all()


def test_primary_anchored_uses_one_vector_group_dot_across_tensors():
    main = [torch.tensor([1.0]), torch.tensor([1.0])]
    factor = [torch.tensor([-2.0]), torch.tensor([1.0])]
    _, safe_factor, report = primary_anchored_factor_surgery(main, factor)
    assert report["conflict"]
    assert torch.allclose(safe_factor[0], torch.tensor([-1.5]))
    assert torch.allclose(safe_factor[1], torch.tensor([1.5]))
    assert abs(sum(torch.dot(m, f) for m, f in zip(main, safe_factor)).item()) < 1e-7


def test_primary_anchored_accum6_matches_window_level_formula():
    main_microbatches = [torch.tensor([float(i), 1.0]) for i in range(1, 7)]
    factor_microbatches = [torch.tensor([-2.0 * i, 0.5]) for i in range(1, 7)]
    accumulated_main = [sum(main_microbatches) / 6.0]
    accumulated_factor = [sum(factor_microbatches) / 6.0]
    _, expected_factor, _ = primary_anchored_factor_surgery(
        accumulated_main, accumulated_factor
    )
    window_main = [torch.zeros(2)]
    window_factor = [torch.zeros(2)]
    for main, factor in zip(main_microbatches, factor_microbatches):
        window_main[0].add_(main / 6.0)
        window_factor[0].add_(factor / 6.0)
    safe_main, actual_factor, _ = primary_anchored_factor_surgery(
        window_main, window_factor
    )
    assert torch.allclose(safe_main[0], accumulated_main[0])
    assert torch.allclose(actual_factor[0], expected_factor[0])


def test_primary_anchored_correction_leaves_nonshared_and_router_gradients_unchanged():
    shared = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    nonshared = torch.nn.Parameter(torch.tensor([0.0]))
    router = torch.nn.Parameter(torch.tensor([0.0]))
    shared.grad = torch.tensor([3.0, 4.0])
    nonshared.grad = torch.tensor([5.0])
    router.grad = torch.tensor([6.0])
    nonshared_before = nonshared.grad.clone()
    router_before = router.grad.clone()
    raw_factor = [torch.tensor([-1.0, 1.0])]
    safe_factor = [torch.tensor([0.0, 1.0])]
    correction_norm = apply_primary_anchored_factor_correction(
        [shared], raw_factor, safe_factor
    )
    assert torch.equal(shared.grad, torch.tensor([4.0, 4.0]))
    assert torch.equal(nonshared.grad, nonshared_before)
    assert torch.equal(router.grad, router_before)
    assert correction_norm == 1.0
