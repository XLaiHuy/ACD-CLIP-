from __future__ import annotations

import inspect

import pytest
import torch

from tools.sabra_v2.p30_objective import (
    P30_NORMALIZATION_EPSILON,
    p30_directional_loss,
)


def _teacher() -> torch.Tensor:
    values = torch.linspace(-2.0, 2.0, 81, dtype=torch.float32)
    values[40] = 0.25
    return values.reshape(1, 9, 9)


def _student(teacher: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return teacher.unsqueeze(0).expand(3, -1, -1, -1).clone() * scale


def test_identical_direction_has_low_loss() -> None:
    teacher = _teacher()
    result = p30_directional_loss(_student(teacher), teacher)
    assert result.valid_count == 1
    assert result.directional.item() < 0.01


def test_opposite_direction_has_high_loss() -> None:
    teacher = _teacher()
    result = p30_directional_loss(-_student(teacher), teacher)
    assert result.directional.item() > 1.9


def test_same_direction_is_invariant_to_preregistered_magnitude_scales() -> None:
    teacher = _teacher()
    losses = [p30_directional_loss(_student(teacher, scale), teacher).directional.item() for scale in (0.1, 1.0, 10.0, 100.0)]
    assert max(losses) < 0.1
    assert max(losses) - min(losses) < 0.1


def test_zero_and_near_zero_targets_are_finite_and_zero_target_is_ignored() -> None:
    student = torch.randn((3, 2, 9, 9), dtype=torch.float32, requires_grad=True)
    teacher = torch.zeros((2, 9, 9), dtype=torch.float32)
    teacher[1, 0, 0] = 1e-8
    result = p30_directional_loss(student, teacher)
    assert result.valid_count == 1
    assert torch.isfinite(result.directional)
    zero_only = p30_directional_loss(student[:, :1], teacher[:1])
    assert zero_only.valid_count == 0
    assert zero_only.directional.item() == 0.0
    zero_only.directional.backward()
    assert student.grad is not None
    assert torch.equal(student.grad[:, :1], torch.zeros_like(student.grad[:, :1]))


def test_partial_sign_mismatch_is_monotonic() -> None:
    teacher = torch.ones((1, 9, 9), dtype=torch.float32)
    losses = []
    for mismatches in (0, 1, 10, 40, 81):
        student = torch.ones((3, 1, 9, 9), dtype=torch.float32)
        student.reshape(-1)[:mismatches] *= -1.0
        losses.append(p30_directional_loss(student, teacher).directional.item())
    assert losses == sorted(losses)
    assert losses[0] < losses[-1]


def test_known_spatial_ordering_is_preserved_by_directional_cosine() -> None:
    teacher = torch.arange(1, 82, dtype=torch.float32).reshape(1, 9, 9)
    matching = p30_directional_loss(_student(teacher), teacher).directional
    reversed_student = torch.flip(_student(teacher), dims=(-1, -2))
    reversed_loss = p30_directional_loss(reversed_student, teacher).directional
    assert matching.item() < 1e-5
    assert reversed_loss.item() > matching.item()


def test_zero_initialized_student_receives_finite_nonzero_gradient() -> None:
    teacher = _teacher()
    student = torch.zeros((3, 1, 9, 9), dtype=torch.float32, requires_grad=True)
    loss = p30_directional_loss(student, teacher).directional
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert torch.linalg.vector_norm(student.grad).item() > 1e-8
    assert float(student.grad.abs().max()) < 100.0


def test_objective_is_single_directional_term_without_p29_auxiliary_losses() -> None:
    source = inspect.getsource(p30_directional_loss)
    assert "smooth_l1" not in source.lower()
    assert "p29_sign_guarded_loss" not in source
    assert "pure_normal" not in source
    assert "source_mask" not in source
    assert P30_NORMALIZATION_EPSILON == 0.01


def test_invalid_shapes_and_epsilon_fail_closed() -> None:
    teacher = _teacher()
    with pytest.raises(ValueError):
        p30_directional_loss(torch.zeros((1, 1, 9, 9)), teacher)
    with pytest.raises(ValueError):
        p30_directional_loss(_student(teacher), teacher[:, :8, :])
    with pytest.raises(ValueError):
        p30_directional_loss(_student(teacher), teacher, epsilon=0.0)
