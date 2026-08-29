from __future__ import annotations

import json
import resource
import tracemalloc
from pathlib import Path

import numpy as np

from evaluation.evaluator import evaluate_records
from evaluation.metrics import (
    _arrays,
    _descending_groups,
    binary_average_precision,
    binary_auroc,
    macro_metrics,
)


def _legacy_descending_groups(scores: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    groups: list[tuple[int, int]] = []
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        groups.append((start, end))
        start = end
    return order, groups


def _legacy_binary_average_precision(scores, labels):
    arrays = _arrays(scores, labels)
    assert arrays is not None
    scores_array, labels_array = arrays
    order, groups = _legacy_descending_groups(scores_array)
    sorted_labels = labels_array[order]
    positives = float(labels_array.sum())
    true_positive = 0.0
    false_positive = 0.0
    previous_recall = 0.0
    result = 0.0
    for start, end in groups:
        group_labels = sorted_labels[start:end]
        true_positive += float(group_labels.sum())
        false_positive += float(group_labels.size - group_labels.sum())
        recall = true_positive / positives
        precision = true_positive / max(true_positive + false_positive, 1.0)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return float(result)


def _legacy_binary_auroc(scores, labels):
    arrays = _arrays(scores, labels)
    assert arrays is not None
    scores_array, labels_array = arrays
    order = np.argsort(scores_array, kind="mergesort")
    sorted_scores = scores_array[order]
    sorted_labels = labels_array[order]
    ranks = np.empty(sorted_scores.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    positives = float(labels_array.sum())
    negatives = float(labels_array.size - labels_array.sum())
    positive_rank_sum = ranks[sorted_labels == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives))


def _assert_same(actual: float, expected: float) -> None:
    assert actual == expected or abs(actual - expected) <= 1e-12


def _parity_cases() -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(20260829)
    random_scores = rng.normal(size=97)
    random_labels = rng.integers(0, 2, size=97, dtype=np.int8)
    random_labels[0], random_labels[1] = 0, 1

    heavy_tie_scores = np.asarray([0.2, 0.2, 0.9, 0.9, 0.9, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5])
    heavy_tie_labels = np.asarray([0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1], dtype=np.int8)

    unique_scores = np.linspace(-2.0, 2.0, 101, dtype=np.float64)
    unique_labels = np.zeros(101, dtype=np.int8)
    unique_labels[[2, 17, 44, 73, 100]] = 1

    model_like_scores = rng.random(257).astype(np.float32)
    model_like_labels = np.zeros(257, dtype=np.int8)
    model_like_labels[[4, 29, 121, 256]] = 1

    imbalanced_scores = rng.random(1003).astype(np.float32)
    imbalanced_labels = np.zeros(1003, dtype=np.int8)
    imbalanced_labels[::251] = 1
    return [
        (random_scores, random_labels),
        (heavy_tie_scores, heavy_tie_labels),
        (unique_scores, unique_labels),
        (model_like_scores, model_like_labels),
        (imbalanced_scores, imbalanced_labels),
    ]


def test_binary_pixel_and_image_metrics_match_legacy_across_score_cases():
    for scores, labels in _parity_cases():
        _assert_same(binary_auroc(scores, labels), _legacy_binary_auroc(scores, labels))
        _assert_same(binary_average_precision(scores, labels), _legacy_binary_average_precision(scores, labels))


def test_multiple_class_evaluator_and_macro_metrics_match_legacy():
    records = []
    expected_per_class = {}
    for index, (pixel_scores, pixel_labels) in enumerate(_parity_cases()[:3]):
        image_scores = np.asarray([0.05 + 0.1 * index, 0.95 - 0.1 * index, 0.25, 0.75], dtype=np.float32)
        image_labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
        class_name = f"class_{index}"
        records.append({
            "class_name": class_name,
            "pixel_scores": pixel_scores,
            "pixel_labels": pixel_labels,
            "image_scores": image_scores,
            "image_labels": image_labels,
        })
        expected_per_class[class_name] = {
            "pixel_auroc": _legacy_binary_auroc(pixel_scores, pixel_labels),
            "pixel_ap": _legacy_binary_average_precision(pixel_scores, pixel_labels),
            "image_auroc": _legacy_binary_auroc(image_scores, image_labels),
            "image_ap": _legacy_binary_average_precision(image_scores, image_labels),
        }

    observed = evaluate_records(records, method="phase2b")
    expected_macro = macro_metrics(expected_per_class)
    assert set(observed["per_class"]) == set(expected_per_class)
    for class_name in expected_per_class:
        for metric_name, expected in expected_per_class[class_name].items():
            _assert_same(observed["per_class"][class_name][metric_name], expected)
    for metric_name, expected in expected_macro.items():
        _assert_same(observed["macro"][metric_name], expected)


def test_descending_groups_are_streamed_not_materialized():
    scores = np.asarray([0.7, 0.1, 0.7, 0.3], dtype=np.float64)
    order, groups = _descending_groups(scores)
    assert not isinstance(groups, list)
    assert list(groups) == [(0, 2), (2, 3), (3, 4)]
    assert np.array_equal(scores[order], np.asarray([0.7, 0.7, 0.3, 0.1]))


def test_completed_mvtec_metrics_artifacts_are_self_consistent_if_present():
    root = Path("runs/cir_rmt/CIR_DFG_RMT_V2/visa/seed0/eval/MVTec")
    if not root.is_dir():
        return
    for epoch in (12, 14, 16, 18, 20):
        payload = json.loads((root / f"epoch_{epoch}" / "metrics.json").read_text(encoding="utf-8"))
        assert payload["status"] == "PASS"
        assert payload["evaluator_protocol"] == "CIR_FINAL_EXACT_V1"
        recomputed = macro_metrics(payload["per_class"])
        for name, expected in payload["macro"].items():
            _assert_same(recomputed[name], expected)


def test_ap_streaming_stress_millions_of_mostly_unique_float32_scores():
    count = 2_000_000
    rng = np.random.default_rng(20260829)
    scores = rng.random(count).astype(np.float32)
    labels = np.zeros(count, dtype=np.int8)
    labels[::997] = 1
    unique_count = np.unique(scores).size
    assert unique_count >= int(count * 0.95)

    tracemalloc.start()
    order, groups = _descending_groups(scores.astype(np.float64))
    group_count = sum(1 for _ in groups)
    _, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del order
    assert group_count == unique_count
    assert peak_python_bytes < 128 * 1024 * 1024

    before_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    value = binary_average_precision(scores, labels)
    after_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert np.isfinite(value)
    print(
        f"streaming AP stress: n={count} unique={unique_count} "
        f"peak_python_mib={peak_python_bytes / 2**20:.1f} "
        f"rss_delta_mib={(after_rss_kib - before_rss_kib) / 1024:.1f}"
    )
