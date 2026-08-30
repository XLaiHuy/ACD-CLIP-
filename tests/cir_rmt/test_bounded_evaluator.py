from __future__ import annotations

from pathlib import Path
import resource

import numpy as np
import pytest

import evaluation.metrics as metrics_module
from evaluation.evaluator import evaluate_records, evaluate_spool
from evaluation.metrics import (
    _arrays,
    binary_metrics,
    class_metrics,
    macro_metrics,
)
from evaluation.spool import EvaluationSpool


def _legacy_arrays(scores, labels):
    scores_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels_array = np.asarray(labels, dtype=np.int8).reshape(-1)
    if scores_array.shape != labels_array.shape or scores_array.size == 0:
        raise ValueError("invalid legacy arrays")
    positives = int(labels_array.sum())
    if positives == 0 or positives == labels_array.size:
        raise ValueError("undefined legacy metric")
    return scores_array, labels_array


def _legacy_metrics(scores, labels):
    scores_array, labels_array = _legacy_arrays(scores, labels)
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
    auroc = float(
        (ranks[sorted_labels == 1].sum() - positives * (positives + 1.0) / 2.0)
        / (positives * negatives)
    )

    order = np.argsort(-scores_array, kind="mergesort")
    sorted_scores = scores_array[order]
    sorted_labels = labels_array[order]
    true_positive = 0.0
    false_positive = 0.0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_labels = sorted_labels[start:end]
        true_positive += float(group_labels.sum())
        false_positive += float(group_labels.size - group_labels.sum())
        recall = true_positive / positives
        precision = true_positive / max(true_positive + false_positive, 1.0)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return auroc, float(average_precision)


def _assert_close(actual, expected):
    assert actual == expected or abs(actual - expected) <= 1e-12


def _cases():
    rng = np.random.default_rng(20260830)
    random64 = rng.normal(size=257)
    random64[:2] = [-1.0, 1.0]
    unique32 = rng.random(4099).astype(np.float32)
    labels_unique32 = np.zeros(unique32.size, dtype=np.int8)
    labels_unique32[[0, 17, 1000, -1]] = 1
    ties = rng.integers(0, 7, size=4097).astype(np.float32)
    labels_ties = rng.integers(0, 2, size=ties.size, dtype=np.int8)
    labels_ties[:2] = [0, 1]
    zeros_ones = np.zeros(103, dtype=np.float32)
    zeros_ones[::4] = 1.0
    labels_zeros_ones = np.zeros(103, dtype=np.int8)
    labels_zeros_ones[[1, 8, 31, 77]] = 1
    sparse = rng.random(1003).astype(np.float32)
    labels_sparse = np.zeros(1003, dtype=np.int8)
    labels_sparse[[0, 251, 502, 753, 1002]] = 1
    dense = rng.random(1003).astype(np.float32)
    labels_dense = np.ones(1003, dtype=np.int8)
    labels_dense[[0, 251, 502, 753, 1002]] = 0
    return [
        (random64, rng.integers(0, 2, size=random64.size, dtype=np.int8)),
        (unique32, labels_unique32),
        (ties, labels_ties),
        (zeros_ones, labels_zeros_ones),
        (sparse, labels_sparse),
        (dense, labels_dense),
    ]


def test_shared_metrics_match_legacy_across_exact_score_cases():
    for scores, labels in _cases():
        expected = _legacy_metrics(scores, labels)
        observed = binary_metrics(scores, labels, chunk_size=113)
        _assert_close(observed[0], expected[0])
        _assert_close(observed[1], expected[1])


def test_one_element_undefined_contract_is_preserved():
    with pytest.raises(ValueError, match="both positive and negative"):
        binary_metrics(np.asarray([0.5], dtype=np.float32), np.asarray([1], dtype=np.uint8))
    assert binary_metrics(
        np.asarray([0.5], dtype=np.float32),
        np.asarray([1], dtype=np.uint8),
        allow_undefined=True,
    ) == (None, None)


def test_ties_crossing_internal_chunks_match_legacy():
    scores = np.asarray(
        [0.25, 0.75, 0.5, 0.5, 0.5, 0.25, 0.75, 0.5, 0.25, 0.75, 0.5, 0.25, 0.75],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0], dtype=np.uint8)
    expected = _legacy_metrics(scores, labels)
    observed = binary_metrics(scores, labels, chunk_size=2)
    _assert_close(observed[0], expected[0])
    _assert_close(observed[1], expected[1])


def test_packed_spool_canonicalizes_signed_zero_ties(tmp_path):
    scores = np.asarray([-2.0, -0.0, 0.0, 1.5, -0.0, 1.5, -1.0], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    records = [
        {
            "class_name": "edge",
            "pixel_scores": scores,
            "pixel_labels": labels,
            "image_scores": [0.1],
            "image_labels": [0],
        },
        {
            "class_name": "edge",
            "pixel_scores": scores[::-1],
            "pixel_labels": labels[::-1],
            "image_scores": [0.9],
            "image_labels": [1],
        },
    ]
    expected = evaluate_records(records)
    spool = EvaluationSpool.create(tmp_path / "signed_zero")
    for record in records:
        spool.append(
            record["class_name"],
            record["pixel_scores"],
            record["pixel_labels"],
            record["image_scores"][0],
            record["image_labels"][0],
        )
    observed = evaluate_spool(spool)
    for name, value in expected["per_class"]["edge"].items():
        _assert_close(observed["per_class"]["edge"][name], value)
    spool.cleanup()


def test_million_pixel_exact_parity_with_legacy_reference():
    count = 1_000_003
    rng = np.random.default_rng(20260830)
    scores = (rng.integers(0, 257, size=count, dtype=np.int16) / 257.0).astype(np.float32)
    labels = rng.integers(0, 2, size=count, dtype=np.uint8)
    labels[0] = 0
    labels[1] = 1
    expected = _legacy_metrics(scores, labels)
    observed = binary_metrics(scores, labels, chunk_size=65_537)
    _assert_close(observed[0], expected[0])
    _assert_close(observed[1], expected[1])


def test_class_metrics_builds_one_order_per_score_array(monkeypatch):
    calls = []
    original = metrics_module.np.argsort

    def spy(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(metrics_module.np, "argsort", spy)
    result = class_metrics(
        np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
        np.asarray([0, 1, 0, 1], dtype=np.uint8),
        np.asarray([0.2, 0.8], dtype=np.float32),
        np.asarray([0, 1], dtype=np.uint8),
    )
    assert set(calls) == {(4,), (2,)}
    assert len(calls) == 2
    assert all(np.isfinite(value) for value in result.values())


def test_float32_scores_remain_unwidened_for_metric_storage():
    scores = np.linspace(0.0, 1.0, 17, dtype=np.float32)
    labels = np.asarray([0, 1] * 8 + [0], dtype=np.uint8)
    arrays = _arrays(scores, labels)
    assert arrays is not None
    observed_scores, observed_labels = arrays
    assert observed_scores.dtype == np.float32
    assert np.shares_memory(observed_scores, scores)
    assert observed_labels.dtype == np.uint8


def test_spool_evaluator_matches_record_evaluator_and_cleans(tmp_path):
    rng = np.random.default_rng(11)
    records = []
    seen_by_class = {}
    spool = EvaluationSpool.create(tmp_path / "eval")
    for class_name in ("class_b", "class_a", "class_b", "class_a"):
        scores = rng.random(37).astype(np.float32)
        labels = np.zeros(37, dtype=np.uint8)
        labels[[0, 19]] = 1
        image_score = float(rng.random())
        image_label = seen_by_class.get(class_name, 0)
        seen_by_class[class_name] = image_label + 1
        records.append({
            "class_name": class_name,
            "pixel_scores": scores,
            "pixel_labels": labels,
            "image_scores": [image_score],
            "image_labels": [image_label],
        })
        spool.append(class_name, scores, labels, image_score, image_label)

    expected = evaluate_records(records)
    observed = evaluate_spool(spool)
    for class_name in expected["per_class"]:
        for name, value in expected["per_class"][class_name].items():
            _assert_close(observed["per_class"][class_name][name], value)
    for name, value in expected["macro"].items():
        _assert_close(observed["macro"][name], value)
    for entry in spool.classes():
        with entry.open_arrays() as (scores, labels):
            assert scores.dtype == np.float32
            assert labels.dtype == np.uint8
            assert scores.size == entry.pixel_count
    spool.cleanup()
    assert not (tmp_path / "eval" / ".cir_eval_spool").exists()


def test_stale_spool_is_replaced_on_reuse(tmp_path):
    first = EvaluationSpool.create(tmp_path / "eval")
    first.append("class", np.asarray([0.1, 0.9], dtype=np.float32), np.asarray([0, 1], dtype=np.uint8), 0.5, 1)
    second = EvaluationSpool.create(tmp_path / "eval")
    assert second.classes() == ()
    second.cleanup()


def test_complete_spool_metric_stress_reports_rss(tmp_path):
    count = 2_000_003
    rng = np.random.default_rng(20260830)
    scores = rng.random(count).astype(np.float32)
    labels = np.zeros(count, dtype=np.uint8)
    labels[::997] = 1
    spool = EvaluationSpool.create(tmp_path / "stress")
    spool.append("synthetic", scores, labels, 0.5, 1)
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = evaluate_spool(spool, allow_undefined_image_metrics=True)
    final_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert np.isfinite(result["per_class"]["synthetic"]["pixel_auroc"])
    assert np.isfinite(result["per_class"]["synthetic"]["pixel_ap"])
    print(
        "complete spool stress: "
        f"n={count} baseline_maxrss_mib={baseline_rss / 1024:.1f} "
        f"final_maxrss_mib={final_rss / 1024:.1f}"
    )
    spool.cleanup()


def test_production_evaluator_is_explicitly_bounded():
    source = Path("scripts/cir_rmt/eval_full.py").read_text(encoding="utf-8")
    assert "records = []" not in source
    assert "EvaluationSpool.create" in source
    assert "evaluate_spool" in source
    assert "_shutdown_loader" in source
    assert "torch.cuda.empty_cache()" in source
