"""Ground-truth-free numerical core for the frozen SABRA audit."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from p5f_geometry.common import decode_gram, pack_gram

ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "runs/phase5/sabra/PRETRAIN_LOGIC_AUDIT"
CACHE_ROOT = AUDIT_ROOT / "cache"
PATCH_GRID = (37, 37)
PATCHES = 1369
STAGES = 3
PEERS = 8
IMAGE_SIZE = 518
PATCH_STRIDE = 14


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def percentile_rank(values: np.ndarray) -> np.ndarray:
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
    for start, end in zip(starts, ends):
        ranks_sorted[start:end] = ((start + end - 1) / 2.0) / max(n - 1, 1)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def percentile_against_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    reference = np.sort(np.asarray(reference, dtype=np.float32))
    if reference.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    result = (left + right - 1.0) / max(reference.size - 1, 1)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def fixed_cdf_components(raw: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    raw = np.asarray(raw, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if raw.shape != (STAGES, PATCHES) or valid.shape != (PATCHES,):
        raise ValueError("invalid baseline component shapes")
    ranks = np.zeros_like(raw, dtype=np.float32)
    supports: list[np.ndarray] = []
    for stage in range(STAGES):
        support = np.sort(raw[stage, valid])
        supports.append(support)
        if support.size:
            ranks[stage, valid] = percentile_rank(raw[stage, valid]).astype(np.float32)
    return ranks, supports


def map_fixed_cdf(raw: np.ndarray, valid: np.ndarray, supports: list[np.ndarray]) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if raw.shape[-2:] != (STAGES, PATCHES):
        raise ValueError(f"expected [...,3,1369], got {raw.shape}")
    result = np.zeros(raw.shape, dtype=np.float32)
    for stage in range(STAGES):
        mapped = percentile_against_reference(raw[..., stage, :], supports[stage])
        result[..., stage, valid] = mapped[..., valid]
    return result


def construct_b1(
    stage_features: np.ndarray,
    d_rank: np.ndarray,
    native_margins: np.ndarray,
) -> dict[str, np.ndarray]:
    """Exact frozen candidate pool, top-8 ordering, and p9 reservation."""
    import torch
    import torch.nn.functional as F

    features = np.asarray(stage_features, dtype=np.float32)
    d_rank = np.asarray(d_rank, dtype=np.float64)
    native_margins = np.asarray(native_margins, dtype=np.float64)
    if features.shape != (STAGES, PATCHES, 768):
        raise ValueError(f"unexpected feature shape {features.shape}")
    tensor = torch.from_numpy(features)
    shared = F.normalize(tensor.mean(dim=0), dim=-1)
    stage_rank = np.stack([percentile_rank(native_margins[s]) for s in range(STAGES)], axis=0)
    pool = (d_rank < np.median(d_rank)) & np.all(stage_rank < 0.5, axis=0)
    pool_indices = np.flatnonzero(pool).astype(np.int64)
    yy, xx = np.divmod(np.arange(PATCHES), PATCH_GRID[1])
    peers = np.full((PATCHES, PEERS), -1, dtype=np.int64)
    reserve = np.full(PATCHES, -1, dtype=np.int64)
    valid_b1 = np.zeros(PATCHES, dtype=bool)
    valid_stability = np.zeros(PATCHES, dtype=bool)
    candidate_count = np.zeros(PATCHES, dtype=np.int32)
    centroid = np.zeros(PATCHES, dtype=np.float32)
    gap = np.zeros(PATCHES, dtype=np.float32)
    pool_features = shared.float()[pool_indices]
    for query in range(PATCHES):
        if not pool_indices.size:
            continue
        spatial_ok = np.maximum(
            np.abs(yy[pool_indices] - yy[query]),
            np.abs(xx[pool_indices] - xx[query]),
        ) > 3
        candidates = pool_indices[spatial_ok]
        candidate_count[query] = int(candidates.size)
        if not candidates.size:
            continue
        columns = np.flatnonzero(spatial_ok)
        similarities = (shared[query] @ pool_features[columns].T).numpy()
        order = np.lexsort((candidates, -similarities))
        ordered = candidates[order]
        ordered_similarity = similarities[order]
        if ordered.size >= PEERS:
            peers[query] = ordered[:PEERS]
            valid_b1[query] = True
            refs = F.normalize(tensor[:, peers[query]].mean(dim=1), dim=-1)
            centroid[query] = float((1.0 - (tensor[:, query] * refs).sum(dim=-1)).mean())
        if ordered.size >= PEERS + 1:
            reserve[query] = int(ordered[PEERS])
            valid_stability[query] = True
            gap[query] = float(ordered_similarity[PEERS - 1] - ordered_similarity[PEERS])
    return {
        "peer_indices": peers,
        "reserve_peer_index": reserve,
        "valid_b1": valid_b1,
        "valid_stability": valid_stability,
        "candidate_count": candidate_count,
        "b1_centroid_evidence": centroid,
        "p8_p9_similarity_gap": gap,
        "stage_margin_rank": stage_rank.astype(np.float32),
        "pool": pool,
    }


def compact_geometry(stage_features: np.ndarray, b1: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    features = np.asarray(stage_features, dtype=np.float32)
    peers = np.maximum(np.asarray(b1["peer_indices"], dtype=np.int64), 0)
    reserve = np.maximum(np.asarray(b1["reserve_peer_index"], dtype=np.int64), 0)
    valid = np.asarray(b1["valid_b1"], dtype=bool)
    stable = np.asarray(b1["valid_stability"], dtype=bool)
    references = features[:, peers]
    c = np.sum(features[:, :, None, :] * references, axis=-1, dtype=np.float32)
    gram = np.einsum("spkd,spld->spkl", references, references, dtype=np.float32)
    reserve_features = features[:, reserve]
    query_reserve = np.sum(features * reserve_features, axis=-1, dtype=np.float32)
    reserve_to_peer = np.einsum("spd,spkd->spk", reserve_features, references, dtype=np.float32)
    c[:, ~valid] = 0.0
    gram[:, ~valid] = 0.0
    query_reserve[:, ~stable] = 0.0
    reserve_to_peer[:, ~stable] = 0.0
    return {
        "query_peer_cos": c.astype(np.float32),
        "peer_gram_upper": pack_gram(gram),
        "query_reserve_cos": query_reserve.astype(np.float32),
        "reserve_to_peer_cos": reserve_to_peer.astype(np.float32),
    }


def pgm_raw(c: np.ndarray, packed_gram: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized canonical machine-rank PGM raw component."""
    c = np.asarray(c, dtype=np.float64)
    grams = decode_gram(np.asarray(packed_gram, dtype=np.float64))
    if c.shape[-2:] != (STAGES, PEERS) or grams.shape[-2:] != (PEERS, PEERS):
        raise ValueError(f"invalid PGM geometry shapes {c.shape}, {grams.shape}")
    H = np.eye(PEERS) - np.full((PEERS, PEERS), 1.0 / PEERS)
    centered = np.einsum("ij,...jk,kl->...il", H, grams, H)
    centered = (centered + np.swapaxes(centered, -1, -2)) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    eigenvalues = eigenvalues[..., ::-1]
    eigenvectors = eigenvectors[..., :, ::-1]
    max_eigen = np.maximum(eigenvalues[..., 0], 0.0)
    tolerance = np.finfo(np.float32).eps * np.maximum(1.0, max_eigen) * PEERS
    positive = eigenvalues > tolerance[..., None]
    weights = np.full(PEERS, 1.0 / PEERS)
    centered_query = c - np.einsum("...ij,j->...i", grams, weights)
    b = np.einsum("ij,...j->...i", H, centered_query)
    projection = np.einsum("...i,...ij->...j", b, eigenvectors)
    terms = np.where(
        positive,
        7.0 * projection * projection / np.maximum(eigenvalues * eigenvalues, 1e-30),
        0.0,
    )
    return {
        "raw": np.sum(terms, axis=-1, dtype=np.float64).astype(np.float32),
        "rank": positive.sum(axis=-1).astype(np.int16),
        "tol": tolerance.astype(np.float32),
        "max_eigen": max_eigen.astype(np.float32),
    }


def pcrr_raw(c: np.ndarray, packed_gram: np.ndarray) -> dict[str, np.ndarray]:
    c = np.asarray(c, dtype=np.float32)
    grams = decode_gram(np.asarray(packed_gram, dtype=np.float64)).astype(np.float32)
    query_distance = 1.0 - c
    peer_distance = 1.0 - grams
    values = np.zeros(c.shape[:-1] + (PEERS,), dtype=np.float32)
    comparisons = np.zeros(c.shape[:-1] + (PEERS,), dtype=np.int16)
    for peer in range(PEERS):
        compared = peer_distance[..., peer, :] <= query_distance[..., peer, None]
        compared[..., peer] = False
        comparisons[..., peer] = compared.sum(axis=-1).astype(np.int16)
        values[..., peer] = (1.0 + comparisons[..., peer]) / PEERS
    values.sort(axis=-1)
    return {
        "raw": values.mean(axis=-1, dtype=np.float32).astype(np.float32),
        "comparison_count": comparisons,
    }


def replacement_geometry(geometry: dict[str, np.ndarray], b1: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(geometry["query_peer_cos"], dtype=np.float32)
    packed = np.asarray(geometry["peer_gram_upper"], dtype=np.float32)
    q_reserve = np.asarray(geometry["query_reserve_cos"], dtype=np.float32)
    reserve_to_peer = np.asarray(geometry["reserve_to_peer_cos"], dtype=np.float32)
    full = decode_gram(packed).astype(np.float32)
    replacement_c = np.repeat(c[None], PEERS, axis=0)
    replacement_g = np.repeat(full[None], PEERS, axis=0)
    for slot in range(PEERS):
        replacement_c[slot, :, :, slot] = q_reserve
        replacement_g[slot, :, :, :, slot] = reserve_to_peer
        replacement_g[slot, :, :, slot, :] = reserve_to_peer
        replacement_g[slot, :, :, slot, slot] = 1.0
    invalid = ~np.asarray(b1["valid_stability"], dtype=bool)
    replacement_c[:, :, invalid] = 0.0
    replacement_g[:, :, invalid] = 0.0
    return replacement_c, pack_gram(replacement_g)


def compute_relational_scores(geometry: dict[str, np.ndarray], b1: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    c = geometry["query_peer_cos"]
    packed = geometry["peer_gram_upper"]
    valid = np.asarray(b1["valid_b1"], dtype=bool)
    baseline_pgm = pgm_raw(c, packed)
    baseline_pcrr = pcrr_raw(c, packed)
    pgm_ranks, pgm_support = fixed_cdf_components(baseline_pgm["raw"], valid)
    pcrr_ranks, pcrr_support = fixed_cdf_components(baseline_pcrr["raw"], valid)
    replacement_c, replacement_g = replacement_geometry(geometry, b1)
    replacement_c_flat = replacement_c.transpose(0, 2, 1, 3).reshape(-1, STAGES, PEERS)
    replacement_g_flat = replacement_g.transpose(0, 2, 1, 3).reshape(-1, STAGES, 36)
    replacement_pgm = pgm_raw(replacement_c_flat, replacement_g_flat)
    replacement_pcrr = pcrr_raw(replacement_c_flat, replacement_g_flat)
    replacement_pgm_raw = replacement_pgm["raw"].reshape(PEERS, PATCHES, STAGES).transpose(0, 2, 1)
    replacement_pcrr_raw = replacement_pcrr["raw"].reshape(PEERS, PATCHES, STAGES).transpose(0, 2, 1)
    stable = np.asarray(b1["valid_stability"], dtype=bool)
    replacement_pgm_rank = map_fixed_cdf(replacement_pgm_raw, stable, pgm_support).mean(axis=-2)
    replacement_pcrr_rank = map_fixed_cdf(replacement_pcrr_raw, stable, pcrr_support).mean(axis=-2)
    replacement_pgm_rank[:, ~stable] = 0.0
    replacement_pcrr_rank[:, ~stable] = 0.0
    return {
        "baseline_pgm": pgm_ranks.mean(axis=0).astype(np.float32),
        "baseline_pcrr": pcrr_ranks.mean(axis=0).astype(np.float32),
        "replacement_pgm": replacement_pgm_rank.astype(np.float32),
        "replacement_pcrr": replacement_pcrr_rank.astype(np.float32),
        "pgm_raw": baseline_pgm["raw"],
        "pcrr_raw": baseline_pcrr["raw"],
        "pgm_component_rank": pgm_ranks,
        "pcrr_component_rank": pcrr_ranks,
        "pgm_rank": baseline_pgm["rank"],
        "pgm_tol": baseline_pgm["tol"],
        "pgm_max_eigen": baseline_pgm["max_eigen"],
        "pcrr_comparison_count": baseline_pcrr["comparison_count"],
    }


def structural_trust(relational: dict[str, np.ndarray], valid_stability: np.ndarray) -> dict[str, np.ndarray]:
    valid = np.asarray(valid_stability, dtype=bool)
    pgm = np.asarray(relational["baseline_pgm"], dtype=np.float32)
    replacements = np.asarray(relational["replacement_pgm"], dtype=np.float32)
    all_ranks = np.concatenate([pgm[None], replacements], axis=0)
    robust = np.min(all_ranks, axis=0)
    boundary = 1.0 - np.abs(replacements[7] - pgm)
    influence = 1.0 - np.max(np.abs(replacements - pgm[None]), axis=0)
    stability = np.minimum(boundary, influence)
    trust = np.minimum(robust, stability)
    for value in (boundary, influence, robust, stability, trust):
        np.clip(value, 0.0, 1.0, out=value)
        value[~valid] = 0.0
    return {
        "pgm_boundary": boundary.astype(np.float32),
        "pgm_influence": influence.astype(np.float32),
        "pgm_robust": robust.astype(np.float32),
        "pgm_stability": stability.astype(np.float32),
        "trust": trust.astype(np.float32),
    }
