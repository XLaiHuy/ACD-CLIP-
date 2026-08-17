"""Pure CSRC cross-stage rank-consistency transform."""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from .common import aggregate_components, average_tie_percentile, validate_common, PACKED_GRAM_COUNT


def _is_constant(values: np.ndarray) -> bool:
    values = np.asarray(values)
    return bool(values.size == 0 or np.all(values == values[0]))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if _is_constant(x) and _is_constant(y):
        return 1.0
    if _is_constant(x) or _is_constant(y):
        return 0.0
    rx = average_tie_percentile(x); ry = average_tie_percentile(y)
    sx = float(rx.std()); sy = float(ry.std())
    if sx <= np.finfo(np.float64).eps or sy <= np.finfo(np.float64).eps:
        return 0.0
    order = np.lexsort((ry, rx))
    rx, ry = rx[order], ry[order]
    centered_x, centered_y = rx - rx.mean(), ry - ry.mean()
    numerator = float(np.sum(centered_x * centered_y, dtype=np.float64) / rx.size)
    return float(numerator / (sx * sy))


def _kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if _is_constant(x) and _is_constant(y):
        return 1.0
    if _is_constant(x) or _is_constant(y):
        return 0.0
    concordant = discordant = tie_x_only = tie_y_only = tied_both = 0
    for i, j in combinations(range(x.size), 2):
        dx = x[i] - x[j]; dy = y[i] - y[j]
        if dx == 0 and dy == 0:
            tied_both += 1
        elif dx == 0:
            tie_x_only += 1
        elif dy == 0:
            tie_y_only += 1
        elif dx * dy > 0:
            concordant += 1
        else:
            discordant += 1
    del tied_both
    denom = math.sqrt((concordant + discordant + tie_x_only) * (concordant + discordant + tie_y_only))
    return 0.0 if denom <= np.finfo(np.float64).eps else float((concordant - discordant) / denom)


def transform(c: np.ndarray, G: np.ndarray, valid_reference: np.ndarray, config: dict) -> dict[str, np.ndarray | dict]:
    del G
    validate_common(c, np.zeros((3, 1369, PACKED_GRAM_COUNT), dtype=np.float32), valid_reference)
    distances = 1.0 - np.asarray(c, dtype=np.float64)
    if config["pair_scope"] == "all_three":
        pairs = [(0, 1), (0, 2), (1, 2)]
    elif config["pair_scope"] == "adjacent":
        pairs = [(0, 1), (1, 2)]
    else:
        raise ValueError("unknown CSRC pair scope")
    components = np.zeros((len(pairs), 1369), dtype=np.float64)
    for component, (left, right) in enumerate(pairs):
        for patch in range(1369):
            assoc = _spearman(distances[left, patch], distances[right, patch]) if config["association"] == "spearman_average_tie" else _kendall_tau_b(distances[left, patch], distances[right, patch])
            components[component, patch] = (1.0 - assoc) / 2.0
    out = aggregate_components(components, valid_reference, config["pair_aggregation"])
    out["diagnostics"] = {"family": "CSRC", "config_id": config["config_id"], "pair_count": len(pairs), "valid_coverage": float(np.mean(valid_reference))}
    return out
