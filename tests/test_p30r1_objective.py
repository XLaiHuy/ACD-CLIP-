from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from tools.sabra_v2 import p30r1_preflight as reference
from tools.sabra_v2.p30r1_objective import (
    COORDINATE_COUNT,
    P30R1_FORMULATION_HASH,
    P30R1_NORMALIZATION_EPSILON,
    P30R1_OBJECTIVE_NAME,
    p30r1_gradient_bounds,
    p30r1_teacher_relative_components,
    p30r1_teacher_relative_loss,
)
from tools.sabra_v2.run_p30r1_engineering import production_reference_parity


def _teacher(batch: int = 1) -> torch.Tensor:
    values = torch.linspace(-2.0, 2.0, 81, dtype=torch.float32).reshape(1, 9, 9)
    values[0, 4, 4] = 0.25
    return values.expand(batch, -1, -1).clone()


def _student(teacher: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return teacher.unsqueeze(0).expand(3, -1, -1, -1).clone() * scale


def test_exact_match_is_zero_and_has_shared_frozen_denominator() -> None:
    teacher = _teacher()
    student = _student(teacher)
    loss, normalized_student, normalized_teacher, teacher_scale = p30r1_teacher_relative_components(student, teacher)
    assert loss.item() == 0.0
    assert torch.equal(normalized_student, normalized_teacher)
    assert teacher_scale.shape == (1, 1)
    assert teacher_scale.requires_grad is False


def test_scale_is_identifiable_and_gross_errors_are_not_equivalent() -> None:
    teacher = _teacher()
    losses = {
        scale: float(p30r1_teacher_relative_loss(_student(teacher, scale), teacher))
        for scale in (0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0)
    }
    assert losses[1.0] == 0.0
    assert losses[1.0] < min(loss for scale, loss in losses.items() if scale != 1.0)
    assert losses[10.0] > 0.1
    assert losses[100.0] > losses[10.0]


def test_zero_teacher_is_active_and_restores_student_toward_zero() -> None:
    teacher = torch.zeros((1, 9, 9), dtype=torch.float32, requires_grad=True)
    student = (0.5 * _student(_teacher())).requires_grad_(True)
    loss = p30r1_teacher_relative_loss(student, teacher)
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert torch.linalg.vector_norm(student.grad) > 1e-8
    assert torch.sum(student.grad * student.detach()) > 0.0
    assert teacher.grad is None


def test_near_zero_teacher_is_finite_and_bounded() -> None:
    student = 0.5 * _student(_teacher())
    for scale in (1e-8, 1e-6, 1e-4):
        teacher = (_teacher() * scale).requires_grad_(True)
        value = student.detach().clone().requires_grad_(True)
        loss = p30r1_teacher_relative_loss(value, teacher)
        loss.backward()
        assert torch.isfinite(loss)
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert value.grad.abs().max() < 100.0
        assert torch.linalg.vector_norm(value.grad) < 1000.0
        assert teacher.grad is None


def test_production_reference_parity_covers_required_cases() -> None:
    result = production_reference_parity()
    assert result["status"] == "PASS"
    assert len(result["cases"]) == 5
    assert result["teacher_gradients_not_backpropagated"] is True
    assert result["max_abs_errors"]["loss"] <= result["tolerance"]["atol"]
    assert result["max_abs_errors"]["student_gradient"] <= result["tolerance"]["atol"]


def test_shape_and_reduction_match_frozen_contract() -> None:
    generator = torch.Generator().manual_seed(13)
    teacher = torch.randn((2, 9, 9), generator=generator)
    student = torch.randn((3, 2, 9, 9), generator=generator)
    loss, normalized_student, normalized_teacher, scale = p30r1_teacher_relative_components(student, teacher)
    expected = F.smooth_l1_loss(normalized_student, normalized_teacher, beta=1.0, reduction="mean")
    assert normalized_student.shape == (2, COORDINATE_COUNT)
    assert normalized_teacher.shape == (2, COORDINATE_COUNT)
    assert scale.shape == (2, 1)
    assert torch.equal(loss, expected)


def test_production_constants_and_bounds_are_frozen() -> None:
    assert P30R1_OBJECTIVE_NAME == "P30R1_TEACHER_RELATIVE_SMOOTHL1_V1"
    assert P30R1_FORMULATION_HASH == "290aae42e04d9faae5a10b929eb58aa0da066b5dbd248b3fee40f20e9094781c"
    assert P30R1_NORMALIZATION_EPSILON == 0.01
    bounds = p30r1_gradient_bounds()
    assert bounds["gradient_max_abs"] < 0.083
    assert bounds["per_sample_gradient_l2"] < 1.294


def test_p30r1_production_objective_has_no_forbidden_terms() -> None:
    source = inspect.getsource(p30r1_teacher_relative_components)
    assert "smooth_l1_loss" in source
    assert "cosine" not in source.lower()
    assert "sign" not in source.lower()
    assert "normal-region" not in source.lower()
    assert "source_mask" not in source
    assert "student_scale" not in source


def test_invalid_shape_is_fail_closed() -> None:
    teacher = _teacher()
    with pytest.raises(ValueError):
        p30r1_teacher_relative_loss(torch.zeros((1, 1, 9, 9)), teacher)
    with pytest.raises(ValueError):
        p30r1_teacher_relative_loss(_student(teacher), teacher[:, :8, :])
