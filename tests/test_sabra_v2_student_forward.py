from __future__ import annotations

import pytest
import torch
from torch import nn

from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.student_forward import assert_frozen_phase2b, forward_region_student


def test_zero_initialized_region_student_is_native_p26_deployment_parity() -> None:
    """P27 must begin as an exact P26 deployment identity, within FP tolerance."""
    torch.manual_seed(7)
    adapter = RegionResidualAdapter()
    features = torch.randn((3, 1, 1369, 768))
    native = torch.randn((3, 1, 1369, 2))

    output = forward_region_student(adapter, features, native)

    assert output.region_residual.shape == (3, 1, 9, 9)
    assert torch.allclose(output.corrected_logits, native, atol=1e-6, rtol=1e-6)
    assert torch.allclose(output.deployed_probability, output.native_probability, atol=1e-6, rtol=1e-6)


def test_loco_inventory_excludes_held_class_from_every_fit_record() -> None:
    """Held-class GT must be structurally unreachable from teacher/fitting input."""
    rows = [
        {"class_name": "candle", "image_path": "source/candle.png"},
        {"class_name": "capsules", "image_path": "source/capsules.png"},
        {"class_name": "candle", "image_path": "source/candle-2.png"},
    ]

    inventory = loco_inventory(rows, held_class="candle")

    assert [row["class_name"] for row in inventory.fit_rows] == ["capsules"]
    assert [row["class_name"] for row in inventory.held_rows] == ["candle", "candle"]
    assert all(row["class_name"] != inventory.held_class for row in inventory.fit_rows)


def test_only_adapter_is_trainable_and_receives_finite_backward_gradient() -> None:
    """Frozen Phase2B parameters cannot become optimizer-owned by P27."""
    phase2b = nn.Sequential(nn.Linear(4, 4), nn.GELU())
    adapter = RegionResidualAdapter()
    for parameter in phase2b.parameters():
        parameter.requires_grad_(False)

    assert_frozen_phase2b(phase2b, adapter)
    output = forward_region_student(adapter, torch.randn((3, 1, 1369, 768)), torch.randn((3, 1, 1369, 2)))
    output.corrected_logits.square().mean().backward()

    assert all(parameter.grad is None for parameter in phase2b.parameters())
    assert adapter.output.weight.grad is not None
    assert torch.isfinite(adapter.output.weight.grad).all()


def test_frozen_parameter_audit_rejects_trainable_phase2b() -> None:
    phase2b = nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="frozen"):
        assert_frozen_phase2b(phase2b, RegionResidualAdapter())
