from __future__ import annotations

import torch

from evaluation.evaluator import image_score


def test_domain_image_score_contract():
    cls = torch.tensor([0.8])
    pixel = torch.tensor([0.2])
    assert torch.allclose(image_score(cls, pixel, "Industrial"), torch.tensor([0.74]), atol=1e-6)
    assert torch.allclose(image_score(cls, pixel, "Medical"), torch.tensor([0.5]), atol=1e-6)
