"""Full-precision binary metrics used by every selector and final report."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _arrays(scores: Iterable[float], labels: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    scores_array = np.asarray(list(scores) if not isinstance(scores, np.ndarray) else scores, dtype=np.float64).reshape(-1)
    labels_array = np.asarray(list(labels) if not isinstance(labels, np.ndarray) else labels, dtype=np.int8).reshape(-1)
    if scores_array.shape != labels_array.shape or scores_array.size == 0:
        raise ValueError("scores and labels must be non-empty arrays of equal shape")
    if not np.isfinite(scores_array).all():
        raise ValueError("metric scores must be finite")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("binary labels must be 0/1")
    if int(labels_array.sum()) == 0 or int(labels_array.sum()) == labels_array.size:
        raise ValueError("binary metric requires both positive and negative labels")
    return scores_array, labels_array


def _descending_groups(scores: np.ndarray, labels: np.ndarray) -> list[tuple[int, int, float]]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    groups: list[tuple[int, int, float]] = []
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        groups.append((start, end, float(sorted_scores[start])))
        start = end
    return groups


def binary_auroc(scores: Iterable[float], labels: Iterable[int]) -> float:
    """Exact tie-aware AUROC in raw [0, 1] units."""
    scores_array, labels_array = _arrays(scores, labels)
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
    positive_rank_sum = ranks[sorted_labels == 1].sum()
    positives = float(labels_array.sum())
    negatives = float(labels_array.size - labels_array.sum())
    return float((positive_rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives))


def binary_average_precision(scores: Iterable[float], labels: Iterable[int]) -> float:
    """Exact threshold-grouped average precision in raw [0, 1] units."""
    scores_array, labels_array = _arrays(scores, labels)
    positives = float(labels_array.sum())
    true_positive = 0.0
    false_positive = 0.0
    previous_recall = 0.0
    result = 0.0
    for start, end, _ in _descending_groups(scores_array, labels_array):
        group_labels = labels_array[np.argsort(-scores_array, kind="mergesort")[start:end]]
        true_positive += float(group_labels.sum())
        false_positive += float(group_labels.size - group_labels.sum())
        recall = true_positive / positives
        precision = true_positive / max(true_positive + false_positive, 1.0)
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return float(result)


def class_metrics(
    pixel_scores: Iterable[float],
    pixel_labels: Iterable[int],
    image_scores: Iterable[float],
    image_labels: Iterable[int],
) -> dict[str, float]:
    return {
        "pixel_auroc": binary_auroc(pixel_scores, pixel_labels),
        "pixel_ap": binary_average_precision(pixel_scores, pixel_labels),
        "image_auroc": binary_auroc(image_scores, image_labels),
        "image_ap": binary_average_precision(image_scores, image_labels),
    }


def macro_metrics(per_class: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_class:
        raise ValueError("cannot compute a macro metric over no classes")
    names = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")
    missing = {name: [key for key, value in per_class.items() if name not in value] for name in names}
    missing = {name: values for name, values in missing.items() if values}
    if missing:
        raise ValueError(f"every class must provide every metric: {missing}")
    return {name: float(np.mean([values[name] for values in per_class.values()], dtype=np.float64)) for name in names}


def selection_score(metrics: dict[str, float]) -> float:
    weights = {"pixel_auroc": 0.35, "pixel_ap": 0.35, "image_auroc": 0.15, "image_ap": 0.15}
    return float(sum(weights[name] * float(metrics[name]) for name in weights))
