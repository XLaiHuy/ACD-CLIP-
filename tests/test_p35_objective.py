from __future__ import annotations

import inspect

import pytest
import torch

from tools.sabra_v2 import p35_objective, p35_reference


def _tensors(seed: int = 35002, batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn((3, batch, 9, 9), generator=generator, dtype=torch.float32)
    teacher = torch.randn((batch, 9, 9), generator=generator, dtype=torch.float32)
    return student, teacher


def test_p35_contract_is_full_target_single_objective_and_no_inference_cost() -> None:
    contract = p35_objective.p35_objective_contract()
    assert contract["objective_count"] == 1
    assert contract["inherited_correction_scale_C"] == 4.960109710693359
    assert contract["weight_formula"] == "tanh(abs(detached_teacher_effect)/C)"
    assert contract["target_formula"] == "detached_teacher_effect (full E_t; never multiplied by weight)"
    assert contract["target_shrinkage"] is False
    assert contract["weight_detached"] is True
    assert contract["target_detached"] is True
    assert contract["teacher_detached"] is True
    assert contract["student_self_normalized"] is False
    assert contract["new_tuned_hyperparameters"] == 0
    assert contract["new_learnable_parameters"] == 0
    assert contract["auxiliary_terms"] == []
    assert contract["teacher_at_inference"] is False
    assert contract["incremental_inference_overhead_percent"] == 0


def test_p35_preserves_full_teacher_target_and_does_not_reuse_p34_target_shaping() -> None:
    student, teacher = _tensors(batch=1)
    student = student.requires_grad_(True)
    loss, student_effect, teacher_effect, weight, target = p35_objective.p35_actionability_components(student, teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student_effect).all()
    assert torch.isfinite(teacher_effect).all()
    assert torch.isfinite(weight).all()
    assert torch.isfinite(target).all()
    assert target.requires_grad is False
    assert torch.equal(target, teacher_effect)
    active = (weight > 1e-4) & (weight < 0.99) & (teacher_effect.abs() > 1e-4)
    assert torch.any(active)
    assert not torch.allclose(target[active], (weight * teacher_effect)[active])
    assert torch.isfinite(student.grad).all()


def test_p35_zero_actionability_is_zero_importance_not_p34_zero_target_restoration() -> None:
    student = torch.full((3, 1, 9, 9), 0.5, dtype=torch.float32, requires_grad=True)
    teacher = torch.zeros((1, 9, 9), dtype=torch.float32)
    loss, student_effect, teacher_effect, weight, target = p35_objective.p35_actionability_components(student, teacher)
    loss.backward()
    assert torch.count_nonzero(weight) == 0
    assert torch.count_nonzero(target) == 0
    assert torch.count_nonzero(student_effect) > 0
    assert torch.equal(target, teacher_effect)
    assert loss.item() == 0.0
    assert torch.count_nonzero(student.grad) == 0


@pytest.mark.parametrize("scale", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_p35_scale_cases_are_finite_and_weight_bounded(scale: float) -> None:
    student, teacher = _tensors(batch=1)
    student = (student * scale).requires_grad_(True)
    loss, _student_effect, _teacher_effect, weight, target = p35_objective.p35_actionability_components(student, teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
    assert torch.isfinite(target).all()
    assert bool((weight >= 0.0).all())
    assert bool((weight <= 1.0).all())


def test_p35_reference_and_production_match_outputs_targets_weights_and_gradients() -> None:
    cases = (
        _tensors(),
        (torch.zeros((3, 2, 9, 9), dtype=torch.float32), torch.zeros((2, 9, 9), dtype=torch.float32)),
        (_tensors(35003)[0] * 100.0, _tensors(35003)[1]),
        (_tensors(35004)[0], _tensors(35004)[1] * 0.01),
    )
    for student, teacher in cases:
        production_student = student.detach().clone().requires_grad_(True)
        reference_student = student.detach().clone().requires_grad_(True)
        production = p35_objective.p35_actionability_components(production_student, teacher)
        reference = p35_reference.p35_actionability_components(reference_student, teacher)
        production_gradient = torch.autograd.grad(production[0], production_student)[0]
        reference_gradient = torch.autograd.grad(reference[0], reference_student)[0]
        for actual, expected in zip(production, reference):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)
        torch.testing.assert_close(production_gradient, reference_gradient, rtol=1e-5, atol=1e-6)


def test_p35_source_has_no_p34_target_shaping_or_auxiliary_objective() -> None:
    source = inspect.getsource(p35_objective.p35_actionability_components)
    assert "tanh" in source
    assert "target_effect = teacher_effect.detach()" in source
    assert "weight * teacher_effect" not in source
    assert "cosine" not in source.lower()
    assert "ranking" not in source.lower()
    assert "sparsity" not in source.lower()
    assert "p34" not in source.lower()
