from __future__ import annotations

import torch

from model import phase2b_legacy_bridge as bridge
from model import phase2b_runtime as runtime
from .canonical_fixtures import TinyAdapter, TinyClip, tiny_config


def test_training_side_native_and_frozen_deployment_parity():
    native = torch.randn(3, 1, 1369, 2)
    native_probability, native_deployed_logits = runtime.deploy_native_logits(native)
    zero_probability, zero_deployed_logits = runtime.deploy_with_delta(native, torch.zeros_like(native))
    assert torch.allclose(native_probability, zero_probability, atol=0.0, rtol=0.0)
    assert torch.allclose(native_deployed_logits, zero_deployed_logits, atol=0.0, rtol=0.0)


def test_eval_forward_save_load_matches_all_shared_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "_adapter_and_clip", lambda: (TinyAdapter, lambda *args, **kwargs: TinyClip()))
    monkeypatch.setattr(runtime, "_text_features", lambda model, dataset_name, class_names, device, config: torch.ones((3, len(class_names), 768, 2), device=device))
    asset = tmp_path / "clip.pt"
    asset.write_bytes(b"fixture")
    config = tiny_config()
    trainable = runtime.build_phase2b_trainable(config, asset, torch.device("cpu"))
    trainable.eval()
    payload = {"image_adapter": trainable.image_adapter.state_dict(), "text_adapter": trainable.text_adapter.state_dict(), "soft_prompt": trainable.soft_prompt.state_dict(), "dfg_beta": 0.1, "use_hybrid_soft_prompt": True, "use_soft_prompt": False}
    checkpoint = tmp_path / "adapter_10.pth"
    torch.save(payload, checkpoint)
    frozen = runtime.build_phase2b_frozen(config, checkpoint, asset, torch.device("cpu"))
    image = torch.zeros((1, 3, 518, 518))
    left = runtime.forward_phase2b(trainable, image, ["candle"], torch.device("cpu"), config)
    right = runtime.forward_phase2b(frozen, image, ["candle"], torch.device("cpu"), config)
    for name in ("seg_features", "det_features", "text_features", "native_logits", "native_margin", "native_segmentation_probability", "deployed_logits", "classification_probability"):
        assert torch.equal(getattr(left, name), getattr(right, name)), name
