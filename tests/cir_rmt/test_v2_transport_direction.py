import pytest
import torch

from tools.cir_rmt.core import (
    V1_TRANSPORT_DIRECTION,
    V2_TRANSPORT_DIRECTION,
    cir_logits_from_native_weights,
    score_optimized,
    score_reference,
    transport_pair,
    transport_weights,
)
from tools.cir_rmt.identity import (
    checkpoint_metadata,
    config_sha256,
    load_cir_config,
    release_identity_fields,
    validate_checkpoint_identity,
)


def _fixture():
    torch.manual_seed(44)
    stages, batch, patches, groups, dim = 3, 2, 9, 3, 7
    image = torch.nn.functional.normalize(torch.randn(stages, batch, patches, dim), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(stages, batch, groups, dim, 2), dim=-2)
    native = torch.rand(stages, batch, groups, 2) + 0.2
    native = native / native.sum(dim=-2, keepdim=True)
    delta = torch.randn(stages, batch, patches, groups).tanh()
    return image, text, native, delta


def test_v2_direction_is_explicit_and_alpha_zero_is_exact():
    _, _, native, delta = _fixture()
    native_patch = native.unsqueeze(2).expand(-1, -1, delta.shape[2], -1, -1)
    normal, abnormal = transport_pair(
        native_patch[..., 0], native_patch[..., 1], delta, 0.7,
        transport_direction=V2_TRANSPORT_DIRECTION,
    )
    expected_normal = transport_weights(native_patch[..., 0], delta, 0.7)
    expected_abnormal = transport_weights(native_patch[..., 1], -delta, 0.7)
    assert torch.allclose(normal, expected_normal)
    assert torch.allclose(abnormal, expected_abnormal)
    zero_normal, zero_abnormal = transport_pair(
        native_patch[..., 0], native_patch[..., 1], delta, 0.0,
        transport_direction=V2_TRANSPORT_DIRECTION,
    )
    assert torch.equal(zero_normal, native_patch[..., 0])
    assert torch.equal(zero_abnormal, native_patch[..., 1])
    assert torch.allclose(normal.sum(dim=-1), torch.ones_like(normal[..., 0]))
    assert torch.allclose(abnormal.sum(dim=-1), torch.ones_like(abnormal[..., 0]))


def test_v2_reversing_delta_reverses_transport_effect_and_delta_is_detached():
    _, _, native, delta = _fixture()
    native_patch = native.unsqueeze(2).expand(-1, -1, delta.shape[2], -1, -1)
    plus_normal, plus_abnormal = transport_pair(native_patch[..., 0], native_patch[..., 1], delta, 0.5, V2_TRANSPORT_DIRECTION)
    minus_normal, minus_abnormal = transport_pair(native_patch[..., 0], native_patch[..., 1], -delta, 0.5, V2_TRANSPORT_DIRECTION)
    assert torch.allclose(plus_normal, transport_weights(native_patch[..., 0], delta, 0.5))
    assert torch.allclose(minus_normal, transport_weights(native_patch[..., 0], -delta, 0.5))
    assert torch.max((plus_normal - minus_normal).abs()) > 1e-6
    assert torch.max((plus_abnormal - minus_abnormal).abs()) > 1e-6
    native_patch = native_patch.clone().requires_grad_(True)
    evidence = delta.clone().requires_grad_(True)
    transport_pair(native_patch[..., 0], native_patch[..., 1], evidence, 0.5, V2_TRANSPORT_DIRECTION)[0].sum().backward()
    assert evidence.grad is None
    assert native_patch.grad is not None


def test_v2_reference_optimized_parity_and_v1_identity_is_rejected():
    image, text, native, delta = _fixture()
    v2, v2_native = cir_logits_from_native_weights(image, text, native, delta, 0.25, transport_direction=V2_TRANSPORT_DIRECTION, score_mode="optimized")
    v2_ref, _ = cir_logits_from_native_weights(image, text, native, delta, 0.25, transport_direction=V2_TRANSPORT_DIRECTION, score_mode="reference")
    assert torch.max((v2 - v2_ref).abs()).item() <= 1e-5
    assert torch.isfinite(v2).all()
    assert torch.isfinite(v2_native).all()

    v1_config = load_cir_config("configs/cir_dfg_rmt_v1.json")
    v2_config = load_cir_config("configs/cir_dfg_rmt_v2.json")
    assert release_identity_fields(v1_config)["arch_id"] != release_identity_fields(v2_config)["arch_id"]
    assert config_sha256(v1_config) != config_sha256(v2_config)
    metadata = checkpoint_metadata(v1_config, source_dataset="VisA", epoch=1, git_sha="v1-sha")
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_checkpoint_identity(metadata, v2_config, source_dataset="VisA", expected_git_sha="v1-sha", expected_epoch=1)
    assert metadata["rmt_transport_direction"] == V1_TRANSPORT_DIRECTION
