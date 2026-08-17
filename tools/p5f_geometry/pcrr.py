"""Pure PCRR peer-relative witness-rank transform."""
from __future__ import annotations

import numpy as np

from .common import aggregate_components, decode_gram, validate_common


def transform(c: np.ndarray, G: np.ndarray, valid_reference: np.ndarray, config: dict) -> dict[str, np.ndarray | dict]:
    validate_common(c, G, valid_reference)
    c = np.asarray(c, dtype=np.float64)
    grams = decode_gram(G)
    stages = np.zeros((3, 1369), dtype=np.float64)
    for stage in range(3):
        query_distance = 1.0 - c[stage]
        peer_distance = 1.0 - grams[stage]
        if config["witness_pool"] == "witness_local":
            values = np.zeros((1369, 8), dtype=np.float64)
            for peer in range(8):
                comparisons = peer_distance[:, peer, :] <= query_distance[:, peer, None]
                comparisons[:, peer] = False
                values[:, peer] = (1.0 + comparisons.sum(axis=1)) / 8.0
        elif config["witness_pool"] == "pooled_peer_pairs":
            pair_distance = peer_distance[:, np.triu_indices(8, 1)[0], np.triu_indices(8, 1)[1]]
            values = (1.0 + (pair_distance[:, None, :] <= query_distance[:, :, None]).sum(axis=2)) / 29.0
        else:
            raise ValueError("unknown PCRR witness pool")
        if config["witness_aggregation"] == "mean":
            stages[stage] = values.mean(axis=1)
        elif config["witness_aggregation"] == "median":
            stages[stage] = np.median(values, axis=1)
        else:
            raise ValueError("unknown PCRR witness aggregation")
    out = aggregate_components(stages, valid_reference, config["stage_aggregation"])
    out["diagnostics"] = {"family": "PCRR", "config_id": config["config_id"], "valid_coverage": float(np.mean(valid_reference))}
    return out
