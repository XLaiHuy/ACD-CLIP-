from __future__ import annotations

import inspect

import pytest
import torch

from tools.sabra_v2 import p32_objective, p32_reference


def _tensors(seed: int = 32002, batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn((3, batch, 9, 9), generator=generator, dtype=torch.float32)
    teacher = torch.randn((batch, 9, 9), generator=generator, dtype=torch.float32)
    return student, teacher


def test_objective_contract_has_one_term_and_frozen_effect_constants() -> None:
    contract = p32_objective.p32_objective_contract()
    assert contract["objective_count"] == 1
    assert contract["smooth_l1_beta"] == 1.0
    assert contract["teacher_detached"] is True
    assert contract["student_self_normalized"] is False
    assert contract["auxiliary_terms"] == []
    assert contract["teacher_at_inference"] is False
    assert contract["incremental_inference_overhead_percent"] == 0


def test_objective_shapes_and_exact_null_behavior() -> None:
    student = torch.zeros((3, 2, 9, 9), dtype=torch.float32, requires_grad=True)
    teacher = torch.zeros((2, 9, 9), dtype=torch.float32, requires_grad=True)
    loss, student_effect, teacher_effect = p32_objective.p32_functional_margin_components(student, teacher)
    loss.backward()
    assert student_effect.shape == (2, 518, 518)
    assert teacher_effect.shape == (2, 518, 518)
    assert loss.item() == 0.0
    assert torch.count_nonzero(student.grad) == 0
    assert teacher.grad is None


def test_zero_teacher_nonzero_student_has_a_finite_restoring_gradient() -> None:
    student = torch.full((3, 1, 9, 9), 0.5, dtype=torch.float32, requires_grad=True)
    teacher = torch.zeros((1, 9, 9), dtype=torch.float32)
    loss = p32_objective.p32_functional_margin_loss(student, teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
    assert torch.linalg.vector_norm(student.grad) > 0.0
    assert float((student.grad * student).sum()) > 0.0


@pytest.mark.parametrize("scale", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_scale_cases_are_finite_and_have_expected_radial_order(scale: float) -> None:
    _student, teacher = _tensors(batch=1)
    target = teacher.unsqueeze(0).expand(3, -1, -1, -1)
    student = (target * scale).detach().requires_grad_(True)
    loss = p32_objective.p32_functional_margin_loss(student, teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
    if scale != 1.0:
        assert float(loss) > 0.0
    if scale != 1.0:
        # Convexity gives a positive alignment with the displacement from the
        # target; subtracting the gradient therefore moves toward the target.
        assert float((student.grad * (student - target)).sum()) > 0.0


def test_reference_and_production_match_outputs_and_student_gradients() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        for student, teacher in (
            _tensors(),
            (torch.zeros((3, 2, 9, 9), dtype=torch.float32), torch.zeros((2, 9, 9), dtype=torch.float32)),
            (_tensors(32003)[0] * 100.0, _tensors(32003)[1]),
        ):
            student = student.to(device)
            teacher = teacher.to(device)
            production_student = student.detach().clone().requires_grad_(True)
            reference_student = student.detach().clone().requires_grad_(True)
            production = p32_objective.p32_functional_margin_components(production_student, teacher)
            reference = p32_reference.p32_functional_margin_components(reference_student, teacher)
            production_gradient = torch.autograd.grad(production[0], production_student)[0]
            reference_gradient = torch.autograd.grad(reference[0], reference_student)[0]
            torch.testing.assert_close(production[0], reference[0], rtol=1e-5, atol=1e-4)
            torch.testing.assert_close(production[1], reference[1], rtol=1e-5, atol=1e-4)
            torch.testing.assert_close(production[2], reference[2], rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(production_gradient, reference_gradient, rtol=1e-5, atol=1e-6)


def test_production_has_no_forbidden_auxiliary_objective_or_native_forward() -> None:
    source = inspect.getsource(p32_objective)
    assert "smooth_l1_loss" in source
    assert "cosine" not in source.lower()
    assert "ranking" not in source.lower()
    assert "deploy_native_logits" not in source
    assert "native_logits" not in source
