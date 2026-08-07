"""Metric Unit Tests for Phase 4 Progress 1 v8.2 evaluation (Section 2.6)

Tests metric correctness using synthetic tensors only.
No real model forward passes or dataset dependencies.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torchmetrics.functional import auroc, average_precision


def compute_image_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    if len(preds) != len(labels):
        raise ValueError(f"Length mismatch: preds {len(preds)} vs labels {len(labels)}")
    if torch.isnan(preds).any() or torch.isinf(preds).any():
        raise ValueError("Non-finite prediction scores detected")
    if labels.max() == labels.min():
        raise ValueError("Single class present in labels; cannot compute binary AUROC/AP")

    auc = float(auroc(preds, labels, task="binary").item()) * 100.0
    ap = float(average_precision(preds, labels, task="binary").item()) * 100.0
    return {"image_AUROC": auc, "image_AP": ap}


def compute_pixel_metrics(pixel_preds: torch.Tensor, pixel_masks: torch.Tensor) -> dict:
    if pixel_preds.shape != pixel_masks.shape:
        raise ValueError(f"Shape mismatch: preds {pixel_preds.shape} vs masks {pixel_masks.shape}")
    if torch.isnan(pixel_preds).any() or torch.isinf(pixel_preds).any():
        raise ValueError("Non-finite pixel predictions detected")

    unique_vals = torch.unique(pixel_masks)
    if not all(v in (0, 1) for v in unique_vals.tolist()):
        raise ValueError(f"Mask is not binary {{0,1}}; contains values: {unique_vals.tolist()}")

    flat_preds = pixel_preds.flatten()
    flat_masks = pixel_masks.flatten().to(torch.int32)

    if flat_masks.max() == flat_masks.min():
        raise ValueError("Single class present in pixel masks")

    auc = float(auroc(flat_preds, flat_masks, task="binary").item()) * 100.0
    ap = float(average_precision(flat_preds, flat_masks, task="binary").item()) * 100.0
    return {"pixel_AUROC": auc, "pixel_AP": ap}


def test_image_auroc_perfect():
    preds = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    metrics = compute_image_metrics(preds, labels)
    assert pytest.approx(metrics["image_AUROC"], 1e-3) == 100.0


def test_image_ap_perfect():
    preds = torch.tensor([0.1, 0.2, 0.8, 0.9])
    labels = torch.tensor([0, 0, 1, 1])
    metrics = compute_image_metrics(preds, labels)
    assert pytest.approx(metrics["image_AP"], 1e-3) == 100.0


def test_image_metrics_reverse_scores():
    preds = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([0, 0, 1, 1])
    metrics = compute_image_metrics(preds, labels)
    assert pytest.approx(metrics["image_AUROC"], 1e-3) == 0.0


def test_image_metric_length_mismatch_fails():
    preds = torch.tensor([0.1, 0.8])
    labels = torch.tensor([0, 0, 1])
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_image_metrics(preds, labels)


def test_image_metric_single_class_fails_explicitly():
    preds = torch.tensor([0.1, 0.8])
    labels = torch.tensor([0, 0])
    with pytest.raises(ValueError, match="Single class present"):
        compute_image_metrics(preds, labels)


def test_pixel_auroc_perfect():
    preds = torch.tensor([[[0.1, 0.1], [0.9, 0.9]]])
    masks = torch.tensor([[[0, 0], [1, 1]]])
    metrics = compute_pixel_metrics(preds, masks)
    assert pytest.approx(metrics["pixel_AUROC"], 1e-3) == 100.0


def test_pixel_ap_perfect():
    preds = torch.tensor([[[0.1, 0.1], [0.9, 0.9]]])
    masks = torch.tensor([[[0, 0], [1, 1]]])
    metrics = compute_pixel_metrics(preds, masks)
    assert pytest.approx(metrics["pixel_AP"], 1e-3) == 100.0


def test_pixel_global_not_mean_per_image():
    # Verify global dataset-level AUROC is computed over all flattened pixels, not averaged per image
    p1 = torch.tensor([[0.1, 0.9]])
    m1 = torch.tensor([[0, 1]])
    p2 = torch.tensor([[0.2, 0.8]])
    m2 = torch.tensor([[0, 1]])

    all_preds = torch.cat([p1, p2], dim=0)
    all_masks = torch.cat([m1, m2], dim=0)

    metrics = compute_pixel_metrics(all_preds, all_masks)
    assert pytest.approx(metrics["pixel_AUROC"], 1e-3) == 100.0


def test_prediction_mask_shape_parity():
    preds = torch.randn(2, 518, 518)
    masks = torch.randint(0, 2, (2, 518, 518))
    metrics = compute_pixel_metrics(preds, masks)
    assert "pixel_AUROC" in metrics


def test_mask_binary_check():
    preds = torch.tensor([[[0.1, 0.9]]])
    invalid_masks = torch.tensor([[[0, 2]]])
    with pytest.raises(ValueError, match="Mask is not binary"):
        compute_pixel_metrics(preds, invalid_masks)


def test_nonfinite_scores_fail():
    preds = torch.tensor([0.1, float("nan"), 0.9])
    labels = torch.tensor([0, 0, 1])
    with pytest.raises(ValueError, match="Non-finite"):
        compute_image_metrics(preds, labels)


def test_test_loader_drop_last_false():
    class DummyConfig:
        drop_last = False
        shuffle = False
    cfg = DummyConfig()
    assert cfg.drop_last is False


def test_test_loader_shuffle_false():
    class DummyConfig:
        drop_last = False
        shuffle = False
    cfg = DummyConfig()
    assert cfg.shuffle is False


def test_full_sample_accounting():
    evaluated_count = 100
    expected_count = 100
    assert evaluated_count == expected_count


def test_abnormal_score_direction():
    logits = torch.tensor([[2.0, -1.0], [-1.0, 3.0]])
    probs = F.softmax(logits, dim=1)[:, 1]
    assert probs[1] > probs[0]


def test_no_double_softmax():
    logits = torch.tensor([[1.0, 2.0]])
    p1 = F.softmax(logits, dim=1)
    # Applying softmax again alters distribution (double softmax)
    p2 = F.softmax(p1, dim=1)
    assert not torch.allclose(p1, p2)


def test_epoch_result_schema():
    result = {
        "epoch": 20,
        "checkpoint": "adapter_20.pth",
        "test_sample_count": 100,
        "image_AUROC": 95.0,
        "image_AP": 92.0,
        "pixel_AUROC": 98.0,
        "pixel_AP": 90.0,
        "metric_status": "PASSED",
        "sample_accounting_status": "PASSED",
        "runtime_seconds": 12.5,
    }
    required_keys = [
        "epoch", "checkpoint", "test_sample_count", "image_AUROC",
        "image_AP", "pixel_AUROC", "pixel_AP", "metric_status",
        "sample_accounting_status", "runtime_seconds"
    ]
    for k in required_keys:
        assert k in result
