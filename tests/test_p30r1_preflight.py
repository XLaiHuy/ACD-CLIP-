from __future__ import annotations

import inspect

import torch

from tools.sabra_v2 import p30r1_preflight as preflight


def test_teacher_relative_objective_has_one_frozen_teacher_denominator() -> None:
    direction = preflight._unit_direction()
    teacher = (preflight.CORRECTION_SCALE * direction[0]).requires_grad_(True)
    student = direction[0:1].expand(preflight.STAGES, -1, -1, -1) * preflight.CORRECTION_SCALE
    loss, z_s, z_t, a_t = preflight.teacher_relative_components(student, teacher)
    assert loss.item() == 0.0
    assert torch.allclose(z_s, z_t)
    assert a_t.requires_grad is False
    assert teacher.grad is None
    assert "smooth_l1_loss" in inspect.getsource(preflight.teacher_relative_components)


def test_scale_identifiability_and_zero_teacher_restoration_pass() -> None:
    result = preflight.synthetic_falsification()
    assert result["status"] == "PASS", result
    assert result["checks"]["scale_1x_unique_clear_minimum"]
    assert result["checks"]["zero_teacher_restoring_force"]
    assert result["checks"]["heavy_tail_detected"]


def test_preflight_does_not_define_a_scientific_trainer_or_marker() -> None:
    source = inspect.getsource(preflight)
    assert "class .*Trainer" not in source
    assert "optimizer.step" not in source
    assert "P30_EXECUTION_MARKER" not in source
    assert "P30R1_STAGE2" not in source
