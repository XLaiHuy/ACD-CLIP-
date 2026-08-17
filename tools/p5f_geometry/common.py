"""Pure compact geometry utilities for the P5-FR1 contract."""
from __future__ import annotations

import numpy as np

# Frozen production geometry constants. Keep these as the sole source for the
# writer, decoder, validators, and synthetic tests.
STAGES = 3
PATCHES = 1369
PEERS = 8
GRAM_LAYOUT = "diag8_then_offdiag28"
DIAG_COUNT = PEERS
OFFDIAG_COUNT = PEERS * (PEERS - 1) // 2
PACKED_GRAM_COUNT = DIAG_COUNT + OFFDIAG_COUNT
PAIR_COUNT = PACKED_GRAM_COUNT  # compatibility name; includes diagonal
PAIR_I, PAIR_J = np.triu_indices(PEERS, 1)
EPS_FLOAT32 = np.finfo(np.float32).eps
GRAM_DIAG_TOL = 1.0e-5


def validate_common(c: np.ndarray, G: np.ndarray, valid_reference: np.ndarray) -> None:
    c = np.asarray(c)
    G = np.asarray(G)
    valid_reference = np.asarray(valid_reference)
    if c.shape != (STAGES, PATCHES, PEERS):
        raise ValueError(f"c shape must be {(STAGES, PATCHES, PEERS)}, got {c.shape}")
    if G.shape != (STAGES, PATCHES, PACKED_GRAM_COUNT):
        raise ValueError(f"G shape must be {(STAGES, PATCHES, PACKED_GRAM_COUNT)}, got {G.shape}")
    if valid_reference.shape != (PATCHES,) or valid_reference.dtype != np.bool_:
        raise ValueError(f"valid_reference must be boolean [{PATCHES}]")
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(G))):
        raise ValueError("common geometry must be finite")


def pack_gram(full_gram: np.ndarray) -> np.ndarray:
    """Pack [...,8,8] as eight diagonal then 28 upper off-diagonal values."""
    full = np.asarray(full_gram)
    if full.shape[-2:] != (PEERS, PEERS):
        raise ValueError(f"full Gram must end in {(PEERS, PEERS)}, got {full.shape}")
    if not np.all(np.isfinite(full)):
        raise ValueError("full Gram must be finite")
    diag = np.diagonal(full, axis1=-2, axis2=-1)
    offdiag = full[..., PAIR_I, PAIR_J]
    return np.concatenate([diag, offdiag], axis=-1).astype(np.float32, copy=False)


def decode_gram(packed: np.ndarray) -> np.ndarray:
    """Decode diag8_then_offdiag28 to symmetric [...,8,8]."""
    packed = np.asarray(packed, dtype=np.float64)
    if packed.shape[-1] != PACKED_GRAM_COUNT:
        raise ValueError(f"expected {PACKED_GRAM_COUNT} packed Gram values")
    if not np.all(np.isfinite(packed)):
        raise ValueError("packed Gram must be finite")
    full = np.zeros(packed.shape[:-1] + (PEERS, PEERS), dtype=np.float64)
    diag = np.arange(PEERS)
    full[..., diag, diag] = packed[..., :DIAG_COUNT]
    full[..., PAIR_I, PAIR_J] = packed[..., DIAG_COUNT:]
    full[..., PAIR_J, PAIR_I] = packed[..., DIAG_COUNT:]
    return full


def average_tie_percentile(values: np.ndarray) -> np.ndarray:
    """Reuse the authoritative Phase5 average-tie percentile implementation."""
    from audit_phase5_hsir import percentile_rank
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("percentile input must be one-dimensional")
    return np.asarray(percentile_rank(values), dtype=np.float64)


def aggregate_components(component_raw: np.ndarray, valid_reference: np.ndarray, aggregation: str) -> dict[str, np.ndarray]:
    """Percentile-normalize each component, then aggregate deterministically."""
    # Frozen production evidence is canonical FP32 before percentile ranking.
    # Quantizing here makes peer-slot permutations exactly reproducible.
    component_raw = np.asarray(component_raw, dtype=np.float32).astype(np.float64)
    valid_reference = np.asarray(valid_reference, dtype=bool)
    if component_raw.ndim != 2 or component_raw.shape[1] != PATCHES:
        raise ValueError(f"component_raw must be [components,{PATCHES}]")
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


def centered_eigensystem(gram: np.ndarray, c: np.ndarray, rank_policy: str) -> tuple[float, np.ndarray, np.ndarray, float, int, np.ndarray]:
    """Return C eigensystem plus exact centered query energy t and b=Qz."""
    gram = np.asarray(gram, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if gram.shape != (PEERS, PEERS) or c.shape != (PEERS,):
        raise ValueError("centered geometry shape mismatch")
    H = np.eye(PEERS, dtype=np.float64) - np.full((PEERS, PEERS), 1.0 / PEERS, dtype=np.float64)
    w = np.full(PEERS, 1.0 / PEERS, dtype=np.float64)
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
        total_positive = float(eigenvalues[positive].sum())
        rank = 0
        if total_positive > 0.0:
            cumulative = 0.0
            for value in eigenvalues:
                if value <= tol:
                    continue
                rank += 1
                cumulative += float(value)
                if cumulative / total_positive >= target:
                    break
    else:
        raise ValueError(f"unsupported rank policy {rank_policy}")
    rank = min(rank, PEERS - 1)
    b = H @ (c - gram @ w)
    total_energy = max(float(1.0 - 2.0 * (w @ c) + w @ gram @ w), 0.0)
    return tol, eigenvalues, eigenvectors, total_energy, rank, b
