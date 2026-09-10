import importlib

import pytest
import torch

importlib.import_module("test")
from utils import hard_background_patch_ranking_loss


def test_hard_background_ranking_penalizes_background_above_foreground():
    preds = torch.zeros(1, 2, 4, 4, requires_grad=True)
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, :2, :2] = 1.0
    preds.data[:, 1, :2, :2] = 0.20
    preds.data[:, 1, 2:, 2:] = 0.90

    loss = hard_background_patch_ranking_loss(preds, mask, topk_fraction=0.25, margin=0.05)
    assert loss.item() == pytest.approx(0.75, abs=1e-6)
    loss.backward()
    assert torch.isfinite(preds.grad).all()
    assert preds.grad[:, 1, 2:, 2:].abs().sum().item() > 0.0


def test_normal_image_suppresses_hard_background_scores():
    preds = torch.zeros(1, 2, 4, 4)
    preds[:, 1, 0, 0] = 0.8
    preds[:, 1, 1, 1] = 0.4
    mask = torch.zeros(1, 1, 4, 4)
    loss = hard_background_patch_ranking_loss(preds, mask, topk_fraction=0.01, margin=0.05)
    assert loss.item() == pytest.approx(0.8, abs=1e-6)


def test_hard_background_ranking_validates_inputs():
    preds = torch.zeros(1, 2, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    with pytest.raises(ValueError):
        hard_background_patch_ranking_loss(preds, mask, topk_fraction=0.0)
    with pytest.raises(ValueError):
        hard_background_patch_ranking_loss(preds, mask, margin=-0.1)
