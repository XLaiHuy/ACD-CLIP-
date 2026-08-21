from __future__ import annotations

import torch

from tools.sabra import pipeline


def test_compare_composes_one_existing_forward(monkeypatch):
    calls = []

    def fake_corrected(forward, freeze, domain="Industrial"):
        calls.append(forward)
        return {"corrected_probability": torch.zeros((1, 2, 4, 4)), "corrected_logits": torch.zeros((1, 2, 4, 4)), "delta": torch.zeros((3, 1, 4, 2)), "trust": 0.0, "need": 0.0, "authority": 0.0, "evidence": 0.0}

    monkeypatch.setattr(pipeline, "corrected_from_forward", fake_corrected)
    forward = type("Forward", (), {"native_segmentation_probability": torch.ones((1, 4, 4)), "classification_probability": torch.ones((1,))})()
    pipeline.compare_forward(forward, {}, domain="Industrial")
    assert len(calls) == 1
