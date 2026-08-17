"""Pure ASR affine centered-peer subspace residual transform."""
from __future__ import annotations

import numpy as np

from .common import aggregate_components, centered_eigensystem, decode_gram, validate_common


def transform(c: np.ndarray, G: np.ndarray, valid_reference: np.ndarray, config: dict) -> dict[str, np.ndarray | dict]:
    validate_common(c, G, valid_reference)
    c = np.asarray(c, dtype=np.float64)
    grams = decode_gram(G)
    stages = np.zeros((3, 1369), dtype=np.float64)
    rank_counts = []
    for stage in range(3):
        for patch in range(1369):
            tol, eigenvalues, eigenvectors, total_energy, rank = centered_eigensystem(grams[stage, patch], c[stage, patch], config["rank_policy"])
            z_dot = c[stage, patch] - float(c[stage, patch].mean())
            projection = 0.0
            for component in range(rank):
                if eigenvalues[component] > tol:
                    projection += float((z_dot @ eigenvectors[:, component]) ** 2 / eigenvalues[component])
            outside = max(total_energy - projection, 0.0)
            stages[stage, patch] = outside / max(total_energy + np.finfo(np.float32).eps, np.finfo(np.float32).eps)
            rank_counts.append(rank)
    out = aggregate_components(stages, valid_reference, config["stage_aggregation"])
    out["diagnostics"] = {"family": "ASR", "config_id": config["config_id"], "rank_min": int(min(rank_counts)), "rank_max": int(max(rank_counts)), "valid_coverage": float(np.mean(valid_reference))}
    return out
