"""Pure PGM peer-geometry whitened coordinate transform."""
from __future__ import annotations

import numpy as np

from .common import aggregate_components, centered_eigensystem, decode_gram, validate_common


def transform(c: np.ndarray, G: np.ndarray, valid_reference: np.ndarray, config: dict) -> dict[str, np.ndarray | dict]:
    validate_common(c, G, valid_reference)
    c = np.asarray(c, dtype=np.float64)
    grams = decode_gram(G)
    stages = np.zeros((3, 1369), dtype=np.float64)
    for stage in range(3):
        for patch in range(1369):
            tol, eigenvalues, eigenvectors, _, rank, b = centered_eigensystem(grams[stage, patch], c[stage, patch], "machine_rank")
            whitened = np.asarray([(7.0 * (b @ eigenvectors[:, j]) ** 2) / (eigenvalues[j] ** 2) for j in range(rank) if eigenvalues[j] > tol], dtype=np.float64)
            if whitened.size:
                stages[stage, patch] = float(whitened.sum() if config["whitened_aggregation"] == "sum_whitened" else whitened.max())
    out = aggregate_components(stages, valid_reference, config["stage_aggregation"])
    out["diagnostics"] = {"family": "PGM", "config_id": config["config_id"], "rank_cap": 7, "valid_coverage": float(np.mean(valid_reference))}
    return out
