"""GT-free SABRA relational construction."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

STAGES = 3
PATCH_GRID = (37, 37)
PATCHES = PATCH_GRID[0] * PATCH_GRID[1]
PEERS = 8
FEATURE_ORDER = (
    "E",
    "peer_coherence",
    "query_support_mean",
    "peer_eigen_entropy",
    "stage_query_profile_disagreement",
)
NEED_ORDER = (
    "margin_within_image_rank",
    "robust_margin_normalization",
    "D_rank",
    "deployment_sensitivity",
)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    result_sorted = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        result_sorted[start:end] = ((start + end - 1) / 2.0) / max(values.size - 1, 1)
        start = end
    result = np.empty(values.size, dtype=np.float64)
    result[order] = result_sorted
    return result.astype(np.float32)


def _pack_gram(gram: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(PEERS)
    return np.asarray(gram)[..., upper[0], upper[1]].astype(np.float32)


def _unpack_gram(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.float32)
    result = np.zeros(packed.shape[:-1] + (PEERS, PEERS), dtype=np.float32)
    upper = np.triu_indices(PEERS)
    result[..., upper[0], upper[1]] = packed
    result[..., upper[1], upper[0]] = packed
    return result


def _pgm_raw(query_peer: np.ndarray, packed_gram: np.ndarray) -> np.ndarray:
    c = np.asarray(query_peer, dtype=np.float64)
    gram = _unpack_gram(packed_gram).astype(np.float64)
    if c.shape[-1] != PEERS or gram.shape[-2:] != (PEERS, PEERS):
        raise ValueError("invalid PGM geometry")
    center = np.eye(PEERS) - np.full((PEERS, PEERS), 1.0 / PEERS)
    centered = np.einsum("ij,...jk,kl->...il", center, gram, center)
    centered = 0.5 * (centered + np.swapaxes(centered, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    eigenvalues = eigenvalues[..., ::-1]
    eigenvectors = eigenvectors[..., :, ::-1]
    maximum = np.maximum(eigenvalues[..., 0], 0.0)
    tolerance = np.finfo(np.float32).eps * np.maximum(1.0, maximum) * PEERS
    positive = eigenvalues > tolerance[..., None]
    weights = np.full((PEERS,), 1.0 / PEERS)
    centered_query = c - np.einsum("...ij,j->...i", gram, weights)
    projection = np.einsum("ij,...j->...i", center, centered_query)
    projection = np.einsum("...i,...ij->...j", projection, eigenvectors)
    terms = np.where(positive, 7.0 * projection * projection / np.maximum(eigenvalues * eigenvalues, 1e-30), 0.0)
    return np.sum(terms, axis=-1, dtype=np.float64).astype(np.float32)


def _fixed_cdf(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    result = np.zeros_like(raw, dtype=np.float32)
    for stage in range(raw.shape[0]):
        indices = np.flatnonzero(valid)
        if indices.size:
            result[stage, indices] = percentile_rank(raw[stage, indices])
    return result


def _candidate_geometry(features: np.ndarray, d_rank: np.ndarray, margins: np.ndarray) -> dict[str, np.ndarray]:
    stage_rank = np.stack([percentile_rank(margins[stage]) for stage in range(STAGES)], axis=0)
    pool = (d_rank < np.median(d_rank)) & np.all(stage_rank < 0.5, axis=0)
    pool_indices = np.flatnonzero(pool).astype(np.int64)
    # Audited B1: average stages, then L2-normalize the shared feature.
    shared = features.mean(axis=0)
    shared /= np.maximum(np.linalg.norm(shared, axis=-1, keepdims=True), np.finfo(np.float32).tiny)
    yy, xx = np.divmod(np.arange(PATCHES), PATCH_GRID[1])
    peers = np.full((PATCHES, PEERS), -1, dtype=np.int64)
    valid = np.zeros(PATCHES, dtype=bool)
    candidate_count = np.zeros(PATCHES, dtype=np.int32)
    for query in range(PATCHES):
        if not pool_indices.size:
            continue
        spatial_ok = np.maximum(np.abs(yy[pool_indices] - yy[query]), np.abs(xx[pool_indices] - xx[query])) > 3
        candidates = pool_indices[spatial_ok]
        candidate_count[query] = int(candidates.size)
        if candidates.size < PEERS:
            continue
        similarities = shared[query] @ shared[candidates].T
        order = np.lexsort((candidates, -similarities))
        peers[query] = candidates[order[:PEERS]]
        valid[query] = True
    safe_peers = np.maximum(peers, 0)
    references = features[:, safe_peers]
    query_peer = np.sum(features[:, :, None, :] * references, axis=-1, dtype=np.float32)
    gram = np.einsum("spkd,spld->spkl", references, references, dtype=np.float32)
    query_peer[:, ~valid] = 0.0
    gram[:, ~valid] = 0.0
    return {
        "stage_margin_rank": stage_rank.astype(np.float32),
        "pool": pool,
        "peer_indices": peers,
        "valid": valid,
        "candidate_count": candidate_count,
        "query_peer_cos": query_peer,
        "peer_gram_upper": _pack_gram(gram),
    }


def _credibility(geometry: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    c = np.asarray(geometry["query_peer_cos"], dtype=np.float64)
    gram = _unpack_gram(geometry["peer_gram_upper"]).astype(np.float64)
    valid = np.asarray(geometry["valid"], dtype=bool)
    upper = np.triu_indices(PEERS, 1)
    peer_coherence = gram[:, :, upper[0], upper[1]].mean(axis=(0, 2)).astype(np.float32)
    query_support_mean = c.mean(axis=(0, 2)).astype(np.float32)
    center = np.eye(PEERS) - np.full((PEERS, PEERS), 1.0 / PEERS)
    centered = np.einsum("ij,spjk,kl->spil", center, gram, center)
    centered = 0.5 * (centered + np.swapaxes(centered, -1, -2))
    eigenvalues = np.linalg.eigvalsh(centered)[..., ::-1]
    positive = eigenvalues > (np.finfo(np.float32).eps * np.maximum(1.0, np.maximum(eigenvalues[..., 0], 0.0))[..., None] * PEERS)
    values = np.where(positive, np.maximum(eigenvalues, 0.0), 0.0)
    total = np.maximum(values.sum(axis=-1, keepdims=True), 1e-30)
    probabilities = values / total
    entropy = -np.sum(np.where(positive, probabilities * np.log(np.maximum(probabilities, 1e-30)), 0.0), axis=-1)
    entropy = np.clip(entropy / np.log(PEERS - 1), 0.0, 1.0).mean(axis=0).astype(np.float32)
    disagreements = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        denominator = np.maximum(np.linalg.norm(c[left], axis=-1) * np.linalg.norm(c[right], axis=-1), 1e-12)
        disagreements.append(1.0 - np.sum(c[left] * c[right], axis=-1) / denominator)
    disagreement = np.mean(disagreements, axis=0).astype(np.float32)
    for value in (peer_coherence, query_support_mean, entropy, disagreement):
        value[~valid] = 0.0
    return {
        "peer_coherence": peer_coherence,
        "query_support_mean": query_support_mean,
        "peer_eigen_entropy": entropy,
        "stage_query_profile_disagreement": disagreement,
    }


def build_relational_record(
    stage_features: np.ndarray,
    native_margins: np.ndarray,
    deployment_sensitivity: np.ndarray | None = None,
    image_path: str | None = None,
) -> dict[str, Any]:
    """Build one image record without accepting any GT-bearing input."""
    features = np.asarray(stage_features, dtype=np.float32)
    margins = np.asarray(native_margins, dtype=np.float32)
    if features.shape != (STAGES, PATCHES, 768):
        raise ValueError(f"expected features [3,1369,768], got {features.shape}")
    if margins.shape != (STAGES, PATCHES):
        raise ValueError(f"expected margins [3,1369], got {margins.shape}")
    stage_rank = np.stack([percentile_rank(margins[stage]) for stage in range(STAGES)], axis=0)
    d_rank = np.std(stage_rank.astype(np.float64), axis=0, ddof=0).astype(np.float32)
    mean_margin = margins.mean(axis=0)
    median = float(np.median(mean_margin))
    mad = float(np.median(np.abs(mean_margin - median)))
    robust = ((mean_margin - median) / (mad + 1e-6)).astype(np.float32)
    margin_rank = percentile_rank(mean_margin)
    geometry = _candidate_geometry(features, d_rank, margins)
    pgm_raw = _pgm_raw(geometry["query_peer_cos"], geometry["peer_gram_upper"])
    pgm_rank = _fixed_cdf(pgm_raw, geometry["valid"])
    evidence = pgm_rank.mean(axis=0).astype(np.float32)
    credibility = _credibility(geometry)
    sensitivity = np.zeros(PATCHES, dtype=np.float32) if deployment_sensitivity is None else np.asarray(deployment_sensitivity, dtype=np.float32).reshape(-1)
    if sensitivity.shape != (PATCHES,):
        raise ValueError("deployment sensitivity must have one value per patch")
    return {
        "E": evidence,
        "peer_indices": geometry["peer_indices"],
        "valid_peers": geometry["valid"],
        "candidate_count": geometry["candidate_count"],
        "query_peer_cos": geometry["query_peer_cos"].astype(np.float32),
        "peer_gram_upper": geometry["peer_gram_upper"],
        "pgm_raw": pgm_raw,
        "stage_margin_percentile_rank": stage_rank,
        "margin_within_image_rank": margin_rank,
        "robust_margin_normalization": robust,
        "D_rank": d_rank,
        "deployment_sensitivity": sensitivity,
        "peer_coherence": credibility["peer_coherence"],
        "query_support_mean": credibility["query_support_mean"],
        "peer_eigen_entropy": credibility["peer_eigen_entropy"],
        "stage_query_profile_disagreement": credibility["stage_query_profile_disagreement"],
        "class_name": None,
        "image_path": image_path,
    }


def trust_features(record: Mapping[str, Any]) -> np.ndarray:
    return np.stack([np.asarray(record[name], dtype=np.float64) for name in FEATURE_ORDER], axis=-1)


def need_features(record: Mapping[str, Any]) -> np.ndarray:
    return np.stack([np.asarray(record[name], dtype=np.float64) for name in NEED_ORDER], axis=-1)


def assert_gt_free_payload(payload: Mapping[str, Any]) -> None:
    forbidden = {"mask", "label", "mask_path", "pixel_gt", "image_label"}
    found = sorted(forbidden.intersection(payload))
    if found:
        raise AssertionError(f"GT-bearing fields reached relational construction: {found}")
