"""Additive FP32 FAST backend for frozen Trust-v2 geometry.

The exact implementation remains authoritative. This module only changes
execution strategy: candidate similarities and compact geometry are batched,
while PGM/PCRR/CDF and feature semantics are delegated to the frozen
numerical implementation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from p5f_geometry.common import decode_gram, pack_gram

from sabra.trust_v2 import numerical as exact

PATCHES = exact.PATCHES
STAGES = exact.STAGES
PEERS = exact.PEERS
PATCH_GRID = exact.PATCH_GRID


def construct_b1_fast(
    stage_features: np.ndarray,
    d_rank: np.ndarray,
    native_margins: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorized candidate pool and deterministic p1..p8/p9/p16 selection."""
    features = np.asarray(stage_features, dtype=np.float32)
    d_rank = np.asarray(d_rank, dtype=np.float64)
    native_margins = np.asarray(native_margins, dtype=np.float64)
    if features.shape[:2] != (STAGES, PATCHES):
        raise ValueError(f"unexpected feature shape {features.shape}")

    stage_rank = np.stack([exact.percentile_rank(native_margins[s]) for s in range(STAGES)])
    pool = (d_rank < np.median(d_rank)) & np.all(stage_rank < 0.5, axis=0)
    indices = np.arange(PATCHES, dtype=np.int64)
    yy, xx = np.divmod(indices, PATCH_GRID[1])
    spatial = np.maximum(
        np.abs(yy[:, None] - yy[None, :]),
        np.abs(xx[:, None] - xx[None, :]),
    ) > 3
    candidate_mask = spatial & pool[None, :]
    counts = candidate_mask.sum(axis=1).astype(np.int32)
    shared_tensor = F.normalize(torch.from_numpy(features).mean(dim=0), dim=-1)

    # Match EXACT's per-query FP32 torch dot products in one padded batch.
    # Padding is masked before sorting, so invalid columns cannot affect IDs.
    max_count = int(counts.max()) if counts.size else 0
    if max_count:
        candidate_indices = np.zeros((PATCHES, max_count), dtype=np.int64)
        candidate_valid = np.zeros((PATCHES, max_count), dtype=bool)
        for query in range(PATCHES):
            candidates = np.flatnonzero(candidate_mask[query])
            candidate_indices[query, :candidates.size] = candidates
            candidate_valid[query, :candidates.size] = True
        candidate_tensor = shared_tensor[torch.from_numpy(candidate_indices)]
        query_tensor = shared_tensor[:, None, :]
        batched_scores = torch.bmm(query_tensor, candidate_tensor.transpose(1, 2)).squeeze(1).numpy()
        batched_scores[~candidate_valid] = -np.inf
        sortable_ids = np.where(candidate_valid, candidate_indices, PATCHES + candidate_indices)
        order = np.lexsort((sortable_ids, -batched_scores), axis=1)
        ordered_ids = np.take_along_axis(candidate_indices, order, axis=1)
        ordered_scores = np.take_along_axis(batched_scores, order, axis=1)

        # A full NumPy matrix is only a disagreement detector.  Rows whose
        # ordering differs from the batched torch result are repaired with
        # EXACT's original per-query torch dot and lexsort semantics.
        full_scores = shared_tensor.numpy() @ shared_tensor.numpy().T
        full_candidate_scores = np.take_along_axis(full_scores, candidate_indices, axis=1)
        full_candidate_scores[~candidate_valid] = -np.inf
        full_order = np.lexsort((sortable_ids, -full_candidate_scores), axis=1)
        full_ordered_ids = np.take_along_axis(candidate_indices, full_order, axis=1)
        valid_positions = np.arange(max_count)[None, :] < counts[:, None]
        disagreement = np.any(
            valid_positions & (ordered_ids != full_ordered_ids),
            axis=1,
        )
        for query in np.flatnonzero(disagreement):
            candidates = np.flatnonzero(candidate_mask[query])
            exact_scores = (shared_tensor[query] @ shared_tensor[candidates].T).numpy()
            exact_order = np.lexsort((candidates, -exact_scores))
            length = candidates.size
            ordered_ids[query, :length] = candidates[exact_order]
            ordered_scores[query, :length] = exact_scores[exact_order]
    else:
        ordered_ids = np.empty((PATCHES, 0), dtype=np.int64)
        ordered_scores = np.empty((PATCHES, 0), dtype=np.float32)

    peers = np.full((PATCHES, PEERS), -1, dtype=np.int64)
    valid_b1 = counts >= PEERS
    peers[valid_b1] = ordered_ids[valid_b1, :PEERS]
    reserves = np.full((2, PATCHES), -1, dtype=np.int64)
    valid_p9 = counts >= 9
    valid_p16 = counts >= 16
    reserves[0, valid_p9] = ordered_ids[valid_p9, 8]
    reserves[1, valid_p16] = ordered_ids[valid_p16, 15]
    gap9 = np.zeros(PATCHES, dtype=np.float32)
    gap16 = np.zeros(PATCHES, dtype=np.float32)
    gap9[valid_p9] = (ordered_scores[valid_p9, 7] - ordered_scores[valid_p9, 8]).astype(np.float32)
    gap16[valid_p16] = (ordered_scores[valid_p16, 7] - ordered_scores[valid_p16, 15]).astype(np.float32)
    return {
        "peer_indices": peers,
        "reserve_p9_index": reserves[0],
        "reserve_p16_index": reserves[1],
        "valid_b1": valid_b1,
        "valid_p9": valid_p9,
        "valid_p16": valid_p16,
        "candidate_count": counts,
        "p8_p9_similarity_gap": gap9,
        "p8_p16_similarity_gap": gap16,
        "stage_margin_rank": stage_rank.astype(np.float32),
        "pool": pool,
    }


def compact_geometry_fast(stage_features: np.ndarray, b1: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Batch compact query-peer and peer-Gram geometry in FP32."""
    features = np.asarray(stage_features, dtype=np.float32)
    peers = np.maximum(np.asarray(b1["peer_indices"], dtype=np.int64), 0)
    reserves = np.maximum(np.stack([b1["reserve_p9_index"], b1["reserve_p16_index"]]), 0)
    valid = np.asarray(b1["valid_b1"], dtype=bool)
    valid_reserves = np.stack([b1["valid_p9"], b1["valid_p16"]])
    references = features[:, peers]
    query_peer = np.sum(features[:, :, None, :] * references, axis=-1, dtype=np.float32)
    gram = np.einsum("spkd,spld->spkl", references, references, dtype=np.float32)
    reserve_features = features[:, reserves]
    query_reserve = np.sum(features[:, None] * reserve_features, axis=-1, dtype=np.float32).transpose(1, 0, 2)
    reserve_to_peer = np.einsum("srpd,spkd->srpk", reserve_features, references, dtype=np.float32).transpose(1, 0, 2, 3)
    query_peer[:, ~valid] = 0.0
    gram[:, ~valid] = 0.0
    for reserve_index in range(2):
        query_reserve[reserve_index, :, ~valid_reserves[reserve_index]] = 0.0
        reserve_to_peer[reserve_index, :, ~valid_reserves[reserve_index]] = 0.0
    return {
        "query_peer_cos": query_peer.astype(np.float32),
        "peer_gram_upper": pack_gram(gram),
        "query_reserve_cos": query_reserve.astype(np.float32),
        "reserve_to_peer_cos": reserve_to_peer.astype(np.float32),
    }


def _relational_for_reserve(
    geometry: dict[str, np.ndarray],
    b1: dict[str, np.ndarray],
    reserve_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch one reserve's replacement geometry with deterministic slot identity."""
    c = np.asarray(geometry["query_peer_cos"], dtype=np.float32)
    full = decode_gram(np.asarray(geometry["peer_gram_upper"], dtype=np.float32)).astype(np.float32)
    q_reserve = geometry["query_reserve_cos"][reserve_index]
    reserve_to_peer = geometry["reserve_to_peer_cos"][reserve_index]
    slots = np.arange(PEERS)
    replacement_c = np.broadcast_to(c[None], (PEERS,) + c.shape).copy()
    replacement_c[slots, :, :, slots] = q_reserve
    replacement_g = np.broadcast_to(full[None], (PEERS,) + full.shape).copy()
    replacement_g[slots, :, :, slots, :] = reserve_to_peer
    replacement_g[slots, :, :, :, slots] = reserve_to_peer
    replacement_g[slots, :, :, slots, slots] = 1.0
    valid = np.asarray(b1["valid_p9" if reserve_index == 0 else "valid_p16"], dtype=bool)
    replacement_c[:, :, ~valid] = 0.0
    replacement_g[:, :, ~valid] = 0.0
    return replacement_c, pack_gram(replacement_g)


def _batched_pgm_pcrr(c: np.ndarray, packed: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Run frozen PGM/PCRR on one combined batch, using FP64 PGM on CUDA."""
    c = np.asarray(c, dtype=np.float32)
    packed = np.asarray(packed, dtype=np.float32)
    if not torch.cuda.is_available():
        pgm = exact.base.pgm_raw(c, packed)
        pcrr = exact.base.pcrr_raw(c, packed)
        return pgm, pcrr
    device = torch.device("cuda")
    c64 = torch.from_numpy(c.astype(np.float64, copy=False)).to(device)
    grams64 = torch.from_numpy(decode_gram(packed.astype(np.float64, copy=False))).to(device)
    H = torch.eye(PEERS, dtype=torch.float64, device=device) - (1.0 / PEERS)
    centered = torch.einsum("ij,...jk,kl->...il", H, grams64, H)
    centered = (centered + centered.transpose(-1, -2)) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(centered)
    eigenvalues = eigenvalues.flip(-1)
    eigenvectors = eigenvectors.flip(-1)
    max_eigen = torch.clamp(eigenvalues[..., 0], min=0.0)
    tolerance = torch.finfo(torch.float32).eps * torch.maximum(torch.ones_like(max_eigen), max_eigen) * PEERS
    positive = eigenvalues > tolerance[..., None]
    weights = torch.full((PEERS,), 1.0 / PEERS, dtype=torch.float64, device=device)
    centered_query = c64 - torch.einsum("...ij,j->...i", grams64, weights)
    b = torch.einsum("ij,...j->...i", H, centered_query)
    projection = torch.einsum("...i,...ij->...j", b, eigenvectors)
    terms = torch.where(positive, 7.0 * projection * projection / torch.clamp(eigenvalues * eigenvalues, min=1e-30), torch.zeros_like(projection))
    pgm_raw = terms.sum(dim=-1, dtype=torch.float64).to(torch.float32)
    pgm = {
        "raw": pgm_raw.cpu().numpy(),
        "rank": positive.sum(dim=-1).to(torch.int16).cpu().numpy(),
        "tol": tolerance.to(torch.float32).cpu().numpy(),
        "max_eigen": max_eigen.to(torch.float32).cpu().numpy(),
    }
    c32 = torch.from_numpy(c).to(device)
    grams32 = torch.from_numpy(decode_gram(packed.astype(np.float64, copy=False)).astype(np.float32)).to(device)
    query_distance = 1.0 - c32
    peer_distance = 1.0 - grams32
    compared = peer_distance <= query_distance[..., None]
    diagonal = torch.eye(PEERS, dtype=torch.bool, device=device)
    compared = compared.masked_fill(diagonal, False)
    comparisons = compared.sum(dim=-1).to(torch.int16)
    values = (1.0 + comparisons.to(torch.float32)) / PEERS
    values = torch.sort(values, dim=-1).values
    pcrr = {
        "raw": values.mean(dim=-1, dtype=torch.float32).cpu().numpy(),
        "comparison_count": comparisons.cpu().numpy(),
    }
    torch.cuda.synchronize(device)
    return pgm, pcrr


def relational_v2_fast(geometry: dict[str, np.ndarray], b1: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Use frozen PGM/PCRR/CDF semantics over batched replacement geometry."""
    c = geometry["query_peer_cos"]
    packed = geometry["peer_gram_upper"]
    valid = np.asarray(b1["valid_b1"], dtype=bool)
    replacement_batches = [_relational_for_reserve(geometry, b1, reserve_index) for reserve_index in range(2)]
    replacement_c = np.stack([item[0] for item in replacement_batches])
    replacement_g = np.stack([item[1] for item in replacement_batches])
    c_flat = replacement_c.transpose(0, 1, 3, 2, 4).reshape(-1, STAGES, PEERS)
    g_flat = replacement_g.transpose(0, 1, 3, 2, 4).reshape(-1, STAGES, 36)
    c_all = np.concatenate([c.transpose(1, 0, 2), c_flat], axis=0)
    g_all = np.concatenate([packed.transpose(1, 0, 2), g_flat], axis=0)
    all_pgm, all_pcrr = _batched_pgm_pcrr(c_all, g_all)
    baseline_pgm = {
        key: value[:PATCHES].T for key, value in all_pgm.items()
    }
    baseline_pcrr = {
        key: value[:PATCHES].T for key, value in all_pcrr.items()
    }
    pgm_ranks, pgm_support = exact.base.fixed_cdf_components(baseline_pgm["raw"], valid)
    pcrr_ranks, pcrr_support = exact.base.fixed_cdf_components(baseline_pcrr["raw"], valid)
    reserve_pgm_raw_array = all_pgm["raw"][PATCHES:].reshape(2, PEERS, PATCHES, STAGES).transpose(0, 1, 3, 2)
    reserve_pcrr_raw_array = all_pcrr["raw"][PATCHES:].reshape(2, PEERS, PATCHES, STAGES).transpose(0, 1, 3, 2)
    reserve_pgm_rank = np.stack([
        exact._map_fixed(reserve_pgm_raw_array[r], b1["valid_p9" if r == 0 else "valid_p16"], pgm_support).mean(axis=-2)
        for r in range(2)
    ])
    reserve_pcrr_rank = np.stack([
        exact._map_fixed(reserve_pcrr_raw_array[r], b1["valid_p9" if r == 0 else "valid_p16"], pcrr_support).mean(axis=-2)
        for r in range(2)
    ])
    for r, key in enumerate(("valid_p9", "valid_p16")):
        reserve_pgm_rank[r, :, ~b1[key]] = 0.0
        reserve_pcrr_rank[r, :, ~b1[key]] = 0.0
    baseline = pgm_ranks.mean(axis=0).astype(np.float32)
    pcrr = pcrr_ranks.mean(axis=0).astype(np.float32)
    return {
        "baseline_pgm": baseline,
        "baseline_pcrr": pcrr,
        "d_rel": np.abs(baseline - pcrr).astype(np.float32),
        "pgm_raw": baseline_pgm["raw"],
        "pcrr_raw": baseline_pcrr["raw"],
        "pgm_component_rank": pgm_ranks,
        "pcrr_component_rank": pcrr_ranks,
        "reserve_pgm_raw": reserve_pgm_raw_array,
        "reserve_pcrr_raw": reserve_pcrr_raw_array,
        "reserve_pgm_rank": reserve_pgm_rank.astype(np.float32),
        "reserve_pcrr_rank": reserve_pcrr_rank.astype(np.float32),
        "pgm_rank": baseline_pgm["rank"],
    }


def build_compact_record_fast(
    features: np.ndarray,
    native_margins: np.ndarray,
    image_path: str = "",
) -> tuple[dict[str, np.ndarray | str], dict[str, Any]]:
    """Build the same compact record as exact using batched geometry."""
    stage_rank = np.stack([exact.percentile_rank(native_margins[s]) for s in range(STAGES)]).astype(np.float32)
    d_rank = np.std(stage_rank.astype(np.float64), axis=0, ddof=0).astype(np.float32)
    b1 = construct_b1_fast(features, d_rank, native_margins)
    geometry = compact_geometry_fast(features, b1)
    relational = relational_v2_fast(geometry, b1)
    credibility = exact._feature_credibility(geometry, b1)
    stability = exact.trust_stability(relational, b1)
    record = {
        "image_path": image_path,
        "D_rank": d_rank,
        "stage_margin_percentile_rank": stage_rank,
        "peer_indices": b1["peer_indices"],
        "reserve_p9_index": b1["reserve_p9_index"],
        "reserve_p16_index": b1["reserve_p16_index"],
        "valid_b1": b1["valid_b1"],
        "valid_p9": b1["valid_p9"],
        "valid_p16": b1["valid_p16"],
        "candidate_count": b1["candidate_count"],
        "p8_p9_similarity_gap": b1["p8_p9_similarity_gap"],
        "p8_p16_similarity_gap": b1["p8_p16_similarity_gap"],
        "baseline_pgm": relational["baseline_pgm"],
        "baseline_pcrr": relational["baseline_pcrr"],
        "D_rel": relational["d_rel"],
        **credibility,
        **stability,
    }
    return record, {"b1": b1, "geometry": geometry, "relational": relational}
