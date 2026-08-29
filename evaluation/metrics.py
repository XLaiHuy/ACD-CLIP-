"""Full-precision, tie-aware binary metrics shared by every evaluator."""
from __future__ import annotations

from typing import Iterable, Iterator, Mapping

import numpy as np


def _arrays(scores: Iterable[float], labels: Iterable[int], *, allow_undefined: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
    scores_array = np.asarray(list(scores) if not isinstance(scores, np.ndarray) else scores, dtype=np.float64).reshape(-1)
    labels_array = np.asarray(list(labels) if not isinstance(labels, np.ndarray) else labels, dtype=np.int8).reshape(-1)
    if scores_array.shape != labels_array.shape or scores_array.size == 0:
        raise ValueError("scores and labels must be non-empty arrays of equal shape")
    if not np.isfinite(scores_array).all():
        raise ValueError("metric scores must be finite")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("binary labels must be 0/1")
    positives = int(labels_array.sum())
    if positives == 0 or positives == labels_array.size:
        if allow_undefined:
            return None
        raise ValueError("binary metric requires both positive and negative labels")
    return scores_array, labels_array


def _descending_groups(scores: np.ndarray) -> tuple[np.ndarray, Iterator[tuple[int, int]]]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]

    def groups() -> Iterator[tuple[int, int]]:
        start = 0
        while start < sorted_scores.size:
            end = start + 1
            while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
                end += 1
            yield start, end
            start = end

    return order, groups()


def binary_auroc(scores: Iterable[float], labels: Iterable[int], *, allow_undefined: bool = False) -> float | None:
    """Exact tie-aware AUROC in raw [0,1] units."""
    arrays = _arrays(scores, labels, allow_undefined=allow_undefined)
    if arrays is None:
        return None
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


def binary_average_precision(scores: Iterable[float], labels: Iterable[int], *, allow_undefined: bool = False) -> float | None:
    """Exact threshold-grouped AP with ties grouped before recall increments."""
    arrays = _arrays(scores, labels, allow_undefined=allow_undefined)
    if arrays is None:
        return None
    scores_array, labels_array = arrays
    order, groups = _descending_groups(scores_array)
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


def class_metrics(
    pixel_scores: Iterable[float],
    pixel_labels: Iterable[int],
    image_scores: Iterable[float],
    image_labels: Iterable[int],
    *,
    allow_undefined_image: bool = False,
) -> dict[str, float | None]:
    return {
        "pixel_auroc": binary_auroc(pixel_scores, pixel_labels),
        "pixel_ap": binary_average_precision(pixel_scores, pixel_labels),
        "image_auroc": binary_auroc(image_scores, image_labels, allow_undefined=allow_undefined_image),
        "image_ap": binary_average_precision(image_scores, image_labels, allow_undefined=allow_undefined_image),
    }


def macro_metrics(per_class: Mapping[str, Mapping[str, float | None]]) -> dict[str, float | None]:
    if not per_class:
        raise ValueError("cannot compute a macro metric over no classes")
    names = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")
    output: dict[str, float | None] = {}
    for name in names:
        values = [float(row[name]) for row in per_class.values() if row.get(name) is not None]
        if not values:
            output[name] = None
        else:
            output[name] = float(np.mean(values, dtype=np.float64))
    return output


def selection_score(metrics: Mapping[str, float | None]) -> float:
    weights = {"pixel_auroc": 0.35, "pixel_ap": 0.35, "image_auroc": 0.15, "image_ap": 0.15}
    missing = [name for name in weights if metrics.get(name) is None]
    if missing:
        raise ValueError(f"selection score requires all four metrics: {missing}")
    return float(sum(weights[name] * float(metrics[name]) for name in weights))
