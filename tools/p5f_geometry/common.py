"""Pure compact-geometry utilities shared by P5-F family transforms."""
from __future__ import annotations

import numpy as np

STAGES = 3
PATCHES = 1369
PEERS = 8
PAIR_COUNT = 36
PAIR_I, PAIR_J = np.triu_indices(PEERS, 1)
EPS_FLOAT32 = np.finfo(np.float32).eps


def validate_common(c: np.ndarray, G: np.ndarray, valid_reference: np.ndarray) -> None:
    c = np.asarray(c)
    G = np.asarray(G)
    valid_reference = np.asarray(valid_reference)
    if c.shape != (STAGES, PATCHES, PEERS):
        raise ValueError(f"c shape must be {(STAGES, PATCHES, PEERS)}, got {c.shape}")
    if G.shape != (STAGES, PATCHES, PAIR_COUNT):
        raise ValueError(f"G shape must be {(STAGES, PATCHES, PAIR_COUNT)}, got {G.shape}")
    if valid_reference.shape != (PATCHES,) or valid_reference.dtype != np.bool_:
        raise ValueError("valid_reference must be boolean [1369]")
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(G))):
        raise ValueError("common geometry must be finite")


def decode_gram(G: np.ndarray) -> np.ndarray:
    """Decode upper-triangle peer Gram vectors to symmetric [G,P,8,8]."""
    G = np.asarray(G, dtype=np.float64)
    if G.shape[-1] != PAIR_COUNT:
        raise ValueError("expected 36 upper-triangle values")
    full = np.zeros(G.shape[:-1] + (PEERS, PEERS), dtype=np.float64)
    diag = np.arange(PEERS)
    full[..., diag, diag] = G[..., :PEERS]
    # The compact common-cache writer stores the same fixed upper-triangle
    # order as np.triu_indices(8, 1), after the eight diagonal entries.
    full[..., PAIR_I, PAIR_J] = G[..., PEERS:]
    full[..., PAIR_J, PAIR_I] = G[..., PEERS:]
    return full


def average_tie_percentile(values: np.ndarray) -> np.ndarray:
    """Authoritative average-tie percentile rank in [0,1]."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("percentile input must be one-dimensional")
    n = values.size
    if n == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    ranks_sorted = np.empty(n, dtype=np.float64)
    denom = max(n - 1, 1)
    for start, end in zip(starts, ends):
        ranks_sorted[start:end] = ((start + end - 1) / 2.0) / denom
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def aggregate_components(component_raw: np.ndarray, valid_reference: np.ndarray, aggregation: str) -> dict[str, np.ndarray]:
    """Percentile-normalize each component, then aggregate deterministically."""
    component_raw = np.asarray(component_raw, dtype=np.float64)
    valid_reference = np.asarray(valid_reference, dtype=bool)
    if component_raw.ndim != 2 or component_raw.shape[1] != PATCHES:
        raise ValueError("component_raw must be [components,1369]")
    if aggregation not in {"mean", "median", "max"}:
        raise ValueError(f"unsupported aggregation {aggregation}")
    if not np.all(np.isfinite(component_raw)):
        raise ValueError("component_raw must be finite")
    ranks = np.stack([average_tie_percentile(row) for row in component_raw], axis=0)
    if aggregation == "mean":
        final = ranks.mean(axis=0)
    elif aggregation == "median":
        final = np.median(ranks, axis=0)
    else:
        final = ranks.max(axis=0)
    ranks[:, ~valid_reference] = 0.0
    final[~valid_reference] = 0.0
    return {
        "component_raw": component_raw.astype(np.float32),
        "component_percentile": ranks.astype(np.float32),
        "final": final.astype(np.float32),
    }


def centered_eigensystem(gram: np.ndarray, c: np.ndarray, rank_policy: str) -> tuple[float, np.ndarray, np.ndarray, float, int]:
    """Return centered peer eigensystem and query-centered energy quantities."""
    gram = np.asarray(gram, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    H = np.eye(PEERS, dtype=np.float64) - np.full((PEERS, PEERS), 1.0 / PEERS, dtype=np.float64)
    centered = H @ gram @ H
    centered = (centered + centered.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    max_eigen = max(float(eigenvalues[0]) if eigenvalues.size else 0.0, 0.0)
    tol = float(EPS_FLOAT32 * max(1.0, max_eigen) * PEERS)
    positive = eigenvalues > tol
    if rank_policy == "machine_rank":
        rank = int(positive.sum())
    elif rank_policy in {"energy_95", "energy_99"}:
        target = 0.95 if rank_policy == "energy_95" else 0.99
        total = float(eigenvalues[positive].sum())
        rank = 0
        if total > 0.0:
            cumulative = 0.0
            for value in eigenvalues:
                if value <= tol:
                    continue
                rank += 1
                cumulative += float(value)
                if cumulative / total >= target:
                    break
    else:
        raise ValueError(f"unsupported rank policy {rank_policy}")
    rank = min(rank, PEERS - 1)
    z_dot = c - float(c.mean())
    centered_query_energy = float(1.0 - 2.0 * c.mean() + gram.mean())
    return tol, eigenvalues, eigenvectors, max(centered_query_energy, 0.0), rank
