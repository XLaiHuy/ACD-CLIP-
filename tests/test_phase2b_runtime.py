from __future__ import annotations

import torch

from model import phase2b_legacy_bridge as bridge
from model import phase2b_runtime as runtime
from .canonical_fixtures import TinyAdapter, TinyClip, tiny_config


def _patched(monkeypatch):
    monkeypatch.setattr(bridge, "_adapter_and_clip", lambda: (TinyAdapter, lambda *args, **kwargs: TinyClip()))
    monkeypatch.setattr(runtime, "_text_features", lambda model, dataset_name, class_names, device, config: torch.ones((3, len(class_names), 768, 2), device=device))


def test_phase2b_shape_and_runtime_audit(monkeypatch, tmp_path):
    _patched(monkeypatch)
    asset = tmp_path / "clip.pt"
    asset.write_bytes(b"fixture")
    model = runtime.build_phase2b_trainable(tiny_config(), asset, torch.device("cpu"))
    audit = runtime.runtime_audit(model)
    assert audit["legacy_module_active"] is False
    assert audit["legacy_trainable_parameter_count"] == 0
    result = runtime.forward_phase2b(model, torch.zeros((1, 3, 518, 518)), ["candle"], torch.device("cpu"), tiny_config())
    assert tuple(result.native_logits.shape) == (3, 1, 1369, 2)
    assert tuple(result.native_margin.shape) == (3, 1, 1369)
    assert tuple(result.native_segmentation_probability.shape) == (1, 518, 518)


def test_native_deployment_zero_delta_parity():
    native = torch.randn(3, 1, 1369, 2)
    native_probability, native_logits = runtime.deploy_native_logits(native)
    zero_probability, zero_logits = runtime.deploy_with_delta(native, torch.zeros_like(native))
    assert torch.equal(native_logits, zero_logits)
    assert torch.equal(native_probability, zero_probability)
