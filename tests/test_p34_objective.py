from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from tools.sabra_v2 import p32_objective, p34_objective, p34_reference, p33_objective


def _tensors(seed: int = 34002, batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn((3, batch, 9, 9), generator=generator, dtype=torch.float32)
    teacher = torch.randn((batch, 9, 9), generator=generator, dtype=torch.float32)
    return student, teacher


def test_p34_contract_is_one_objective_and_explicit_target_only() -> None:
    contract = p34_objective.p34_objective_contract()
    assert contract["objective_count"] == 1
    assert contract["inherited_correction_scale_C"] == 4.960109710693359
    assert contract["weight_formula"] == "clamp(abs(detached_teacher_effect)/C,0,1)"
    assert contract["target_formula"] == "detached(weight * teacher_effect)"
    assert contract["weight_detached"] is True
    assert contract["target_detached"] is True
    assert contract["teacher_detached"] is True
    assert contract["student_self_normalized"] is False
    assert contract["hard_threshold"] is False
    assert contract["sparsity_regularizer"] is False
    assert contract["auxiliary_terms"] == []
    assert contract["teacher_at_inference"] is False
    assert contract["incremental_inference_overhead_percent"] == 0


def test_zero_weight_regression_proves_p33_weighting_does_not_restore_zero() -> None:
    # Effect-space algebra isolates the causal difference, including the
    # decoupled w=0, teacher!=0 case that the actual teacher-derived rule does
    # not normally produce.
    student_effect = torch.ones((1, 4), dtype=torch.float32)
    teacher_effect = torch.full((1, 4), 2.0, dtype=torch.float32)
    zero_weight = torch.zeros_like(student_effect)

    p33_student = student_effect.clone().requires_grad_(True)
    p33_loss = (zero_weight * F.smooth_l1_loss(p33_student, teacher_effect, beta=1.0, reduction="none")).mean()
    p33_loss.backward()

    p34_student = student_effect.clone().requires_grad_(True)
    p34_target = (zero_weight * teacher_effect).detach()
    p34_loss = F.smooth_l1_loss(p34_student, p34_target, beta=1.0, reduction="none").mean()
    p34_loss.backward()

    assert torch.count_nonzero(p33_student.grad) == 0
    assert torch.count_nonzero(p34_student.grad) > 0
    assert torch.all(p34_student.grad > 0)
    assert torch.dot(p34_student.grad.reshape(-1), p34_student.detach().reshape(-1)) > 0


def test_actual_zero_teacher_effect_has_explicit_zero_target_and_restoring_gradient() -> None:
    student = torch.full((3, 1, 9, 9), 0.5, dtype=torch.float32, requires_grad=True)
    teacher = torch.zeros((1, 9, 9), dtype=torch.float32, requires_grad=True)
    loss, student_effect, teacher_effect, weight, target = p34_objective.p34_actionability_components(student, teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
    assert torch.count_nonzero(weight) == 0
    assert torch.count_nonzero(target) == 0
    assert teacher_effect.requires_grad is False
    assert target.requires_grad is False
    assert torch.linalg.vector_norm(student.grad) > 0
    assert torch.dot(student.grad.reshape(-1), student.detach().reshape(-1)) > 0


def test_zero_student_is_stable_zero_optimum_for_zero_teacher() -> None:
    student = torch.zeros((3, 1, 9, 9), dtype=torch.float32, requires_grad=True)
    teacher = torch.zeros((1, 9, 9), dtype=torch.float32)
    loss = p34_objective.p34_actionability_loss(student, teacher)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(student.grad) == 0


def test_w1_reduces_to_p32_functional_behavior() -> None:
    # A teacher with a large enough effect is used to make the actionability
    # weight exactly one at every effect pixel in this deterministic test.
    student = torch.zeros((3, 1, 9, 9), dtype=torch.float32)
    teacher = torch.full((1, 9, 9), 100.0, dtype=torch.float32)
    p34 = p34_objective.p34_actionability_components(student.requires_grad_(True), teacher)
    p32 = p32_objective.p32_functional_margin_components(student.detach(), teacher)
    torch.testing.assert_close(p34[0], p32[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(p34[1], p32[1], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(p34[2], p32[2], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(p34[3], torch.ones_like(p34[3]), rtol=0.0, atol=0.0)
    torch.testing.assert_close(p34[4], p34[2], rtol=0.0, atol=0.0)


def test_intermediate_target_is_exactly_attenuated_and_continuous() -> None:
    student_effect = torch.zeros((1, 4), dtype=torch.float32)
    teacher_effect = torch.full((1, 4), 2.0, dtype=torch.float32)
    weight = torch.full_like(student_effect, 0.5)
    target = (weight * teacher_effect).detach()
    assert torch.equal(target, torch.full_like(target, 1.0))
    loss = F.smooth_l1_loss(student_effect, target, beta=1.0, reduction="mean")
    assert loss.item() == 0.5


@pytest.mark.parametrize("scale", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_scale_cases_are_finite_and_weight_is_monotone_bounded(scale: float) -> None:
    _student, teacher = _tensors(batch=1)
    student = torch.zeros((3, 1, 9, 9), dtype=torch.float32, requires_grad=True)
    scaled_teacher = teacher * scale
    loss, _student_effect, _teacher_effect, weight, target = p34_objective.p34_actionability_components(student, scaled_teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
    assert torch.isfinite(target).all()
    assert torch.all(weight >= 0.0)
    assert torch.all(weight <= 1.0)


@pytest.mark.parametrize("seed", [34010, 34011, 34012])
def test_reference_and_production_match_outputs_targets_and_student_gradients(seed: int) -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    student, teacher = _tensors(seed=seed)
    for device in devices:
        production_student = student.to(device).detach().clone().requires_grad_(True)
        reference_student = student.to(device).detach().clone().requires_grad_(True)
        teacher_device = teacher.to(device)
        production = p34_objective.p34_actionability_components(production_student, teacher_device)
        reference = p34_reference.p34_actionability_components(reference_student, teacher_device)
        for observed, expected in zip(production, reference):
            torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)
        production_gradient = torch.autograd.grad(production[0], production_student)[0]
        reference_gradient = torch.autograd.grad(reference[0], reference_student)[0]
        torch.testing.assert_close(production_gradient, reference_gradient, rtol=1e-5, atol=1e-6)


def test_heavy_tail_mixed_batch_is_finite_and_radially_identifiable() -> None:
    generator = torch.Generator().manual_seed(34020)
    student = torch.randn((3, 4, 9, 9), generator=generator, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn((4, 9, 9), generator=generator, dtype=torch.float32)
    teacher[0] *= 100.0
    loss, student_effect, _teacher_effect, weight, target = p34_objective.p34_actionability_components(student, teacher)
    gradient = torch.autograd.grad(loss, student, retain_graph=True)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(target).all()
    assert torch.all(weight >= 0.0)
    assert torch.all(weight <= 1.0)
    beta = torch.tensor(0.5, dtype=torch.float32)
    low = p34_objective.p34_actionability_loss(student * beta, teacher)
    high = p34_objective.p34_actionability_loss(student * (beta + 0.5), teacher)
    assert not torch.equal(low, high)
    assert student_effect.shape == (4, 518, 518)


def test_production_contains_no_extra_loss_or_inference_module() -> None:
    source = inspect.getsource(p34_objective)
    assert source.count("smooth_l1_loss") == 1
    assert "cosine" not in source.lower()
    assert "ranking" not in source.lower()
    assert "sparsity" in source.lower()  # contract explicitly records it as forbidden
    assert "native_logits" not in source
    assert p34_objective.p34_objective_contract()["auxiliary_terms"] == []
