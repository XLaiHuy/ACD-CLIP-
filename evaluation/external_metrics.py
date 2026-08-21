"""Bounded exact metric accumulators.

The accumulator keeps score/label chunks in host memory and performs one
stable threshold sort per metric at finalize time.  It is exact (including
ties) and avoids per-threshold or per-tie repeated full-array sorting.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .metrics import binary_average_precision, binary_auroc


class ExactBinaryAccumulator:
    def __init__(self) -> None:
        self._scores: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []

    def add(self, scores: Iterable[float], labels: Iterable[int]) -> None:
        score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
        label_array = np.asarray(labels, dtype=np.int8).reshape(-1)
        if score_array.shape != label_array.shape:
            raise ValueError("score/label chunk shape mismatch")
        self._scores.append(score_array.copy())
        self._labels.append(label_array.copy())

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._scores:
            raise ValueError("empty metric accumulator")
        return np.concatenate(self._scores), np.concatenate(self._labels)

    def finalize(self) -> dict[str, float | None]:
        scores, labels = self.arrays()
        return {"auroc": binary_auroc(scores, labels), "ap": binary_average_precision(scores, labels)}
