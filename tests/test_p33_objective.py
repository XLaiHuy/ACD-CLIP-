from __future__ import annotations

import inspect

import pytest
import torch

from tools.sabra_v2 import p33_objective, p33_reference


def _tensors(seed: int = 33002, batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn((3, batch, 9, 9), generator=generator, dtype=torch.float32)
    teacher = torch.randn((batch, 9, 9), generator=generator, dtype=torch.float32)
    return student, teacher


def test_p33_contract_is_one_objective_and_has_no_new_tuned_scalar() -> None:
    contract = p33_objective.p33_objective_contract()
    assert contract["objective_count"] == 1
    assert contract["inherited_correction_scale_C"] == 4.960109710693359
    assert contract["weight_formula"] == "clamp(abs(detached_teacher_effect)/C,0,1)"
    assert contract["weight_detached"] is True
    assert contract["teacher_detached"] is True
    assert contract["student_self_normalized"] is False
    assert contract["target_shrinkage"] is False
    assert contract["hard_threshold"] is False
    assert contract["auxiliary_terms"] == []
    assert contract["teacher_at_inference"] is False
    assert contract["incremental_inference_overhead_percent"] == 0


def test_exact_zero_teacher_effect_is_abstention_with_zero_loss_and_gradient() -> None:
    student = torch.full((3, 1, 9, 9), 0.5, dtype=torch.float32, requires_grad=True)
    teacher = torch.zeros((1, 9, 9), dtype=torch.float32, requires_grad=True)
    loss, _student_effect, teacher_effect, weight = p33_objective.p33_actionability_components(student, teacher)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.count_nonzero(weight) == 0
    assert torch.count_nonzero(student.grad) == 0
    assert teacher.grad is None
    assert teacher_effect.requires_grad is False


def test_nonzero_teacher_effect_retains_signed_target_and_bounded_weight() -> None:
    student = torch.zeros((3, 1, 9, 9), dtype=torch.float32, requires_grad=True)
    teacher = torch.full((1, 9, 9), -0.75, dtype=torch.float32)
    loss, student_effect, teacher_effect, weight = p33_objective.p33_actionability_components(student, teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
    assert torch.all(weight >= 0.0)
    assert torch.all(weight <= 1.0)
    assert torch.any(teacher_effect < 0.0)
    assert torch.linalg.vector_norm(student.grad) > 0.0


@pytest.mark.parametrize("scale", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_scale_cases_are_finite_and_weight_is_monotone_bounded(scale: float) -> None:
    _student, teacher = _tensors(batch=1)
    student = torch.zeros((3, 1, 9, 9), dtype=torch.float32, requires_grad=True)
    scaled_teacher = teacher * scale
    loss, _student_effect, _teacher_effect, weight = p33_objective.p33_actionability_components(student, scaled_teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
    assert torch.all(weight >= 0.0)
    assert torch.all(weight <= 1.0)
    if scale != 1.0:
        assert float(loss) >= 0.0


def test_reference_and_production_match_outputs_weights_and_student_gradients() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    cases = (
        _tensors(),
        (torch.zeros((3, 2, 9, 9), dtype=torch.float32), torch.zeros((2, 9, 9), dtype=torch.float32)),
        (_tensors(33003)[0] * 100.0, _tensors(33003)[1]),
        (_tensors(33004)[0], _tensors(33004)[1] * 0.01),
    )
    for device in devices:
        for student, teacher in cases:
            student = student.to(device)
            teacher = teacher.to(device)
            production_student = student.detach().clone().requires_grad_(True)
            reference_student = student.detach().clone().requires_grad_(True)
            production = p33_objective.p33_actionability_components(production_student, teacher)
            reference = p33_reference.p33_actionability_components(reference_student, teacher)
            production_gradient = torch.autograd.grad(production[0], production_student)[0]
            reference_gradient = torch.autograd.grad(reference[0], reference_student)[0]
            torch.testing.assert_close(production[0], reference[0], rtol=1e-5, atol=1e-4)
            torch.testing.assert_close(production[1], reference[1], rtol=1e-5, atol=1e-4)
            torch.testing.assert_close(production[2], reference[2], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(production[3], reference[3], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(production_gradient, reference_gradient, rtol=1e-5, atol=1e-6)


def test_production_has_no_forbidden_auxiliary_or_inference_path() -> None:
    source = inspect.getsource(p33_objective)
    assert "smooth_l1_loss" in source
    assert "cosine" not in source.lower()
    assert "ranking" not in source.lower()
    assert "deploy_native_logits" not in source
    assert "native_logits" not in source
    assert "teacher_at_inference" in source
