import torch
from torchmetrics.functional import auroc, average_precision

from h2_clean.exact_metrics import ExactBinaryAccumulator


def test_disk_exact_metrics_match_torchmetrics(tmp_path):
    torch.manual_seed(31)
    scores = torch.randn(257).float()
    scores[::17] = 0.0
    labels = torch.randint(0, 2, (257,), dtype=torch.int32)
    accumulator = ExactBinaryAccumulator(tmp_path / "pixel")
    for start in range(0, scores.numel(), 13):
        accumulator.update(scores[start:start + 13], labels[start:start + 13])

    actual_auc, actual_ap = accumulator.compute()
    expected_auc = float(auroc(scores, labels, task="binary").item())
    expected_ap = float(average_precision(scores, labels, task="binary").item())
    # TorchMetrics returns a float32 scalar; the disk implementation keeps
    # rank arithmetic in float64, so compare at the source scalar precision
    # while requiring exact score ordering and tie handling.
    assert abs(actual_auc - expected_auc) < 1e-7
    assert abs(actual_ap - expected_ap) < 1e-7
    accumulator.cleanup()
    assert not (tmp_path / "pixel").exists()


def test_disk_exact_metrics_rejects_single_class(tmp_path):
    accumulator = ExactBinaryAccumulator(tmp_path / "pixel")
    accumulator.update(torch.tensor([0.1, 0.2]), torch.zeros(2, dtype=torch.int32))
    try:
        accumulator.compute()
    except ValueError as exc:
        assert "both positive and negative" in str(exc)
    else:
        raise AssertionError("single-class exact metrics must be rejected")
    accumulator.cleanup()
