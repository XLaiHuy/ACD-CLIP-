import pytest
import torch

from h2_clean.functional_anchor import (
    FUNCTIONAL_ANCHOR_STAGES,
    bounded_config_mismatches,
    freeze_e1_teacher,
    functional_feature_anchor_loss,
    lambda_from_gradient_norms,
    require_source_only,
)


def _stages(requires_grad=False):
    return [torch.randn(2, 5, 7, requires_grad=requires_grad) for _ in range(3)]


def test_e1_teacher_is_frozen_and_receives_no_gradient():
    teacher = freeze_e1_teacher(torch.nn.Linear(3, 3))
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    student = _stages(requires_grad=True)
    with torch.no_grad():
        reference = _stages()
    loss, _ = functional_feature_anchor_loss(student, reference)
    loss.backward()
    assert student[0].grad is None
    assert student[1].grad is not None and student[2].grad is not None


def test_only_stages_two_and_three_are_included_and_finite():
    student = _stages(requires_grad=True)
    reference = [value.detach().clone() for value in student]
    reference[0].add_(1000.0)
    loss, metrics = functional_feature_anchor_loss(student, reference)
    assert torch.isfinite(loss)
    assert set(metrics) == {"functional_anchor_stage2", "functional_anchor_stage3", "functional_anchor_total"}
    assert FUNCTIONAL_ANCHOR_STAGES == (1, 2)
    with pytest.raises(ValueError, match="exactly native stages 2 and 3"):
        functional_feature_anchor_loss(student, reference, stages=(0, 1))


def test_token_shape_alignment_is_required():
    student, reference = _stages(), _stages()
    reference[2] = torch.randn(2, 4, 7)
    with pytest.raises(ValueError, match="shape mismatch"):
        functional_feature_anchor_loss(student, reference)


def test_lambda_calibration_is_deterministic():
    first = lambda_from_gradient_norms([2.0, 4.0, 8.0], [1.0, 2.0, 4.0])
    second = lambda_from_gradient_norms([2.0, 4.0, 8.0], [1.0, 2.0, 4.0])
    assert first == second == (0.2, 0.5)


def test_only_feature_anchor_config_can_differ_and_target_paths_rejected():
    control = {"precision": "fp16", "functional_anchor_lambda": 0.0, "use_functional_feature_anchor": False}
    candidate = {"precision": "fp16", "functional_anchor_lambda": 0.2, "use_functional_feature_anchor": True}
    assert bounded_config_mismatches(control, candidate) == []
    assert bounded_config_mismatches(control, {**candidate, "precision": "bf16"}) == ["precision"]
    require_source_only("VisA")
    with pytest.raises(ValueError, match="source-only"):
        require_source_only("MVTec")
