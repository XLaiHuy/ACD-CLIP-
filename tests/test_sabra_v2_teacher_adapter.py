from __future__ import annotations

import io

import torch

from tools.sabra_v2.correction_teacher import (
    R0_TEACHER_ALPHA,
    build_source_teacher_region_target,
    teacher_patch_delta_from_actions,
)
from tools.sabra_v2.region_adapter import RegionResidualAdapter


def test_teacher_patch_delta_preserves_historical_r0_signed_alpha_semantics() -> None:
    """Teacher targets retain R0 signs and alpha=0.25 as provenance only."""
    actions = torch.zeros((1, 1369), dtype=torch.int64)
    actions[0, 0] = 1
    actions[0, 1] = -1

    patch_delta = teacher_patch_delta_from_actions(actions)

    expected = 19.840438842773438 * 0.25
    assert R0_TEACHER_ALPHA == 0.25
    assert patch_delta.shape == (1, 37, 37)
    assert torch.allclose(patch_delta[0, 0, :2], torch.tensor([expected, -expected]))
    assert torch.count_nonzero(patch_delta[0, 0, 2:]) == 0


def test_region_adapter_has_frozen_shape_contract_and_zero_residual_start() -> None:
    """Zero initialization preserves P26 logits until P27 learns a correction."""
    adapter = RegionResidualAdapter()
    seg_features = torch.randn((3, 2, 1369, 768), dtype=torch.float32)

    residual = adapter(seg_features)

    assert residual.shape == (3, 2, 9, 9)
    assert torch.equal(residual, torch.zeros_like(residual))


def test_region_adapter_serialization_preserves_residuals() -> None:
    """The training handoff checkpoint must faithfully reload the adapter."""
    original = RegionResidualAdapter()
    with torch.no_grad():
        original.output.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
    features = torch.randn((3, 1, 1369, 768), dtype=torch.float32)
    expected = original(features)
    payload = io.BytesIO()
    torch.save(original.state_dict(), payload)

    restored = RegionResidualAdapter()
    restored.load_state_dict(torch.load(io.BytesIO(payload.getvalue()), weights_only=True))

    assert torch.allclose(restored(features), expected, atol=1e-6, rtol=1e-6)


def test_source_teacher_requires_source_mask_and_emits_only_region_targets() -> None:
    """GT is consumed to form a source target, never by the student adapter."""
    native = torch.zeros((3, 1, 1369, 2), dtype=torch.float32)
    source_mask = torch.zeros((1, 1, 518, 518), dtype=torch.float32)

    target = build_source_teacher_region_target(native, source_mask)

    assert target.shape == (1, 9, 9)
    assert not target.requires_grad
