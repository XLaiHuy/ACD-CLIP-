from __future__ import annotations

import torch

from model import phase2b_legacy_bridge as bridge
from model import phase2b_runtime as runtime
from .canonical_fixtures import TinyAdapter, TinyClip, tiny_config


def test_checkpoint_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "_adapter_and_clip", lambda: (TinyAdapter, lambda *args, **kwargs: TinyClip()))
    monkeypatch.setattr(runtime, "_text_features", lambda model, dataset_name, class_names, device, config: torch.ones((3, len(class_names), 768, 2), device=device))
    asset = tmp_path / "clip.pt"
    asset.write_bytes(b"fixture")
    config = tiny_config()
    trainable = runtime.build_phase2b_trainable(config, asset, torch.device("cpu"))
    payload = {"image_adapter": trainable.image_adapter.state_dict(), "text_adapter": trainable.text_adapter.state_dict(), "soft_prompt": trainable.soft_prompt.state_dict(), "dfg_beta": 0.1, "use_hybrid_soft_prompt": True, "use_soft_prompt": False}
    checkpoint = tmp_path / "adapter_10.pth"
    torch.save(payload, checkpoint)
    frozen = runtime.build_phase2b_frozen(config, checkpoint, asset, torch.device("cpu"))
    image = torch.zeros((1, 3, 518, 518))
    left = runtime.forward_phase2b(trainable, image, ["candle"], torch.device("cpu"), config)
    right = runtime.forward_phase2b(frozen, image, ["candle"], torch.device("cpu"), config)
    for name in ("native_logits", "native_margin", "native_segmentation_probability", "classification_probability"):
        assert torch.equal(getattr(left, name), getattr(right, name)), name
