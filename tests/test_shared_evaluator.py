from __future__ import annotations

import numpy as np

from evaluation.evaluator import evaluate_records


def test_compare_uses_shared_metric_contract():
    records = [{
        "class_name": "a",
        "pixel_labels": np.asarray([0, 1, 0, 1]),
        "image_labels": np.asarray([0, 1]),
        "phase2b": {"pixel_scores": np.asarray([0.1, 0.9, 0.2, 0.8]), "image_scores": np.asarray([0.2, 0.8])},
        "sabra": {"pixel_scores": np.asarray([0.2, 0.8, 0.3, 0.7]), "image_scores": np.asarray([0.3, 0.7])},
    }]
    result = evaluate_records(records, method="compare")
    assert set(result["delta"]) == {"pixel_auroc", "pixel_ap", "image_auroc", "image_ap"}
