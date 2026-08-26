from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F

from tools.sabra_v2 import p29_objective

from tools.sabra_v2.p29_objective import (
    CORRECTION_SCALE,
    p29_sign_guarded_loss,
    source_pure_normal_regions,
)


def test_normalization_uses_only_the_frozen_global_correction_scale() -> None:
    teacher = torch.full((3, 1, 9, 9), CORRECTION_SCALE, dtype=torch.float32)
    student = torch.zeros_like(teacher, requires_grad=True)
    source_mask = torch.zeros((1, 1, 518, 518), dtype=torch.float32)

    loss = p29_sign_guarded_loss(student, teacher, source_mask)

    assert torch.equal(loss.normalized_teacher, torch.ones_like(teacher))
    assert loss.value.item() == 0.5


def test_sign_term_penalizes_only_opposite_signed_student_residuals() -> None:
    teacher = torch.full((3, 1, 9, 9), CORRECTION_SCALE, dtype=torch.float32)
    student = torch.full_like(teacher, -CORRECTION_SCALE, requires_grad=True)
    source_mask = torch.ones((1, 1, 518, 518), dtype=torch.float32)

    loss = p29_sign_guarded_loss(student, teacher, source_mask)

    assert loss.sign.item() == 1.0
    assert loss.normal.item() == 0.0


def test_pure_normal_regions_use_only_adaptive_max_pooling_from_source_mask() -> None:
    source_mask = torch.zeros((1, 1, 518, 518), dtype=torch.float32)
    source_mask[:, :, :57, :57] = 1.0

    pure_normal = source_pure_normal_regions(source_mask)

    assert pure_normal.shape == (1, 1, 9, 9)
    assert pure_normal.dtype == torch.bool
    assert pure_normal[0, 0, 0, 0].item() is False
    assert pure_normal[0, 0, -1, -1].item() is True


def test_normal_term_is_exact_zero_when_no_pure_normal_region_exists() -> None:
    teacher = torch.zeros((3, 1, 9, 9), dtype=torch.float32)
    student = torch.ones_like(teacher, requires_grad=True)
    source_mask = torch.ones((1, 1, 518, 518), dtype=torch.float32)

    loss = p29_sign_guarded_loss(student, teacher, source_mask)

    assert loss.normal.item() == 0.0
    assert loss.total.item() == loss.value.item() + loss.sign.item()


def _region(value: float) -> torch.Tensor:
    return torch.full((3, 1, 9, 9), value, dtype=torch.float32)


def test_correction_scale_is_the_exact_preregistered_float() -> None:
    assert CORRECTION_SCALE == 4.960109710693359


def test_objective_contains_no_per_fold_or_per_class_statistics() -> None:
    source = inspect.getsource(p29_objective)
    for forbidden in ("quantile", "median", "std(", "per_class", "per_fold", "clamp"):
        assert forbidden not in source


def test_value_term_matches_the_reference_smooth_l1_fixture() -> None:
    teacher = _region(CORRECTION_SCALE)
    student = _region(0.0).requires_grad_()
    loss = p29_sign_guarded_loss(student, teacher, torch.ones((1, 1, 518, 518)))
    assert torch.equal(loss.value, F.smooth_l1_loss(student / CORRECTION_SCALE, teacher / CORRECTION_SCALE))


def test_negative_teacher_positive_student_has_positive_sign_penalty() -> None:
    loss = p29_sign_guarded_loss(_region(CORRECTION_SCALE).requires_grad_(), _region(-CORRECTION_SCALE), torch.ones((1, 1, 518, 518)))
    assert loss.sign.item() == 1.0


def test_matching_and_zero_teacher_sign_cases_have_zero_penalty() -> None:
    matching = p29_sign_guarded_loss(_region(CORRECTION_SCALE).requires_grad_(), _region(CORRECTION_SCALE), torch.ones((1, 1, 518, 518)))
    zero = p29_sign_guarded_loss(_region(-CORRECTION_SCALE).requires_grad_(), _region(0.0), torch.ones((1, 1, 518, 518)))
    assert matching.sign.item() == 0.0
    assert zero.sign.item() == 0.0


def test_sign_penalty_uses_exact_absolute_teacher_weighting() -> None:
    teacher = _region(CORRECTION_SCALE * 0.25)
    student = _region(-CORRECTION_SCALE).requires_grad_()
    loss = p29_sign_guarded_loss(student, teacher, torch.ones((1, 1, 518, 518)))
    assert loss.sign.item() == 0.25


def test_pure_normal_term_only_penalizes_positive_residuals_in_pure_normal_regions() -> None:
    mask = torch.zeros((1, 1, 518, 518))
    positive = p29_sign_guarded_loss(_region(CORRECTION_SCALE).requires_grad_(), _region(0.0), mask)
    negative = p29_sign_guarded_loss(_region(-CORRECTION_SCALE).requires_grad_(), _region(0.0), mask)
    assert positive.normal.item() == 1.0
    assert negative.normal.item() == 0.0


def test_anomalous_or_mixed_regions_do_not_contribute_to_normal_penalty() -> None:
    mask = torch.zeros((1, 1, 518, 518))
    mask[:, :, :58, :58] = 1.0
    student = _region(0.0).requires_grad_()
    student.data[:, :, 0, 0] = CORRECTION_SCALE
    loss = p29_sign_guarded_loss(student, _region(0.0), mask)
    assert loss.normal.item() == 0.0


def test_total_is_exact_sum_of_three_preregistered_terms() -> None:
    loss = p29_sign_guarded_loss(_region(-CORRECTION_SCALE).requires_grad_(), _region(CORRECTION_SCALE), torch.zeros((1, 1, 518, 518)))
    assert torch.equal(loss.total, loss.value + loss.sign + loss.normal)
