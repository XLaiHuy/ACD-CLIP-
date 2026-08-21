"""Pure numerical Trust-v2 sidecar operations.

The module deliberately has no dataset, mask, MVTec, or model imports.  It
extends the frozen B1/p9 geometry with the preregistered exact p16 reserve
and computes only compact GT-free evidence.
"""
from __future__ import annotations

from typing import Any

import numpy as np
try:
    from p5f_geometry.common import decode_gram, pack_gram
except ModuleNotFoundError:  # package import from the repository root
    from tools.p5f_geometry.common import decode_gram, pack_gram

try:
    from sabra import logic_core as base
    from sabra import logic_core_fixed as fixed
except ModuleNotFoundError:  # package import from the repository root
    from tools.sabra import logic_core as base
    from tools.sabra import logic_core_fixed as fixed

PATCH_GRID = (37, 37)
PATCHES = 1369
STAGES = 3
PEERS = 8
RESERVES = (9, 16)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    ranks_sorted = np.empty(values.size, dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks_sorted[start:end] = ((start + end - 1) / 2.0) / max(values.size - 1, 1)
    result = np.empty(values.size, dtype=np.float64)
    result[order] = ranks_sorted
    return result


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).tiny)


def construct_b1_v2(
    stage_features: np.ndarray,
    d_rank: np.ndarray,
    native_margins: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the frozen B1 pool, p1..p8, exact p9 and exact p16.

    Ordering and tie handling are explicit so reserve identity is never
    fabricated or independently reranked.
    """
    features = np.asarray(stage_features, dtype=np.float32)
    d_rank = np.asarray(d_rank, dtype=np.float64)
    native_margins = np.asarray(native_margins, dtype=np.float64)
    if features.shape[:2] != (STAGES, PATCHES):
        raise ValueError(f"unexpected feature shape {features.shape}")
    stage_rank = np.stack([percentile_rank(native_margins[s]) for s in range(STAGES)])
    pool = (d_rank < np.median(d_rank)) & np.all(stage_rank < 0.5, axis=0)
    pool_indices = np.flatnonzero(pool).astype(np.int64)
    yy, xx = np.divmod(np.arange(PATCHES), PATCH_GRID[1])
    peers = np.full((PATCHES, PEERS), -1, dtype=np.int64)
    reserves = np.full((2, PATCHES), -1, dtype=np.int64)
    valid_b1 = np.zeros(PATCHES, dtype=bool)
    valid_p9 = np.zeros(PATCHES, dtype=bool)
    valid_p16 = np.zeros(PATCHES, dtype=bool)
    candidate_count = np.zeros(PATCHES, dtype=np.int32)
    gap9 = np.zeros(PATCHES, dtype=np.float32)
    gap16 = np.zeros(PATCHES, dtype=np.float32)
    import torch
    import torch.nn.functional as F
    tensor = torch.from_numpy(features)
    shared = F.normalize(tensor.mean(dim=0), dim=-1)
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
        if ordered.size >= 9:
            reserves[0, query] = ordered[8]
            valid_p9[query] = True
            gap9[query] = float(ordered_similarity[7] - ordered_similarity[8])
        if ordered.size >= 16:
            reserves[1, query] = ordered[15]
            valid_p16[query] = True
            gap16[query] = float(ordered_similarity[7] - ordered_similarity[15])
    return {
        "peer_indices": peers,
        "reserve_p9_index": reserves[0],
        "reserve_p16_index": reserves[1],
        "valid_b1": valid_b1,
        "valid_p9": valid_p9,
        "valid_p16": valid_p16,
        "candidate_count": candidate_count,
        "p8_p9_similarity_gap": gap9,
        "p8_p16_similarity_gap": gap16,
        "stage_margin_rank": stage_rank.astype(np.float32),
        "pool": pool,
    }


def compact_geometry_v2(stage_features: np.ndarray, b1: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    features = np.asarray(stage_features, dtype=np.float32)
    peers = np.maximum(np.asarray(b1["peer_indices"], dtype=np.int64), 0)
    reserves = np.maximum(
        np.stack([b1["reserve_p9_index"], b1["reserve_p16_index"]]), 0
    )
    valid = np.asarray(b1["valid_b1"], dtype=bool)
    valid_reserves = np.stack([b1["valid_p9"], b1["valid_p16"]])
    references = features[:, peers]
    c = np.sum(features[:, :, None, :] * references, axis=-1, dtype=np.float32)
    gram = np.einsum("spkd,spld->spkl", references, references, dtype=np.float32)
    reserve_features = features[:, reserves]
    query_reserve = np.sum(features[:, None] * reserve_features, axis=-1, dtype=np.float32).transpose(1, 0, 2)
    reserve_to_peer = np.einsum("srpd,spkd->srpk", reserve_features, references, dtype=np.float32).transpose(1, 0, 2, 3)
    c[:, ~valid] = 0.0
    gram[:, ~valid] = 0.0
    for reserve_index in range(2):
        query_reserve[reserve_index, :, ~valid_reserves[reserve_index]] = 0.0
        reserve_to_peer[reserve_index, :, ~valid_reserves[reserve_index]] = 0.0
    return {
        "query_peer_cos": c.astype(np.float32),
        "peer_gram_upper": pack_gram(gram),
        "query_reserve_cos": query_reserve.astype(np.float32),
        "reserve_to_peer_cos": reserve_to_peer.astype(np.float32),
    }


def _replacement_geometry_v2(
    geometry: dict[str, np.ndarray],
    b1: dict[str, np.ndarray],
    reserve_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(geometry["query_peer_cos"], dtype=np.float32)
    full = decode_gram(np.asarray(geometry["peer_gram_upper"], dtype=np.float32)).astype(np.float32)
    q_reserve = geometry["query_reserve_cos"][reserve_index]
    reserve_to_peer = geometry["reserve_to_peer_cos"][reserve_index]
    replacement_c = np.repeat(c[None], PEERS, axis=0)
    replacement_g = np.repeat(full[None], PEERS, axis=0)
    for slot in range(PEERS):
        replacement_c[slot, :, :, slot] = q_reserve
        replacement_g[slot, :, :, :, slot] = reserve_to_peer
        replacement_g[slot, :, :, slot, :] = reserve_to_peer
        replacement_g[slot, :, :, slot, slot] = 1.0
    invalid = ~np.asarray(b1["valid_p9" if reserve_index == 0 else "valid_p16"], dtype=bool)
    replacement_c[:, :, invalid] = 0.0
    replacement_g[:, :, invalid] = 0.0
    return replacement_c, pack_gram(replacement_g)


def _map_fixed(raw: np.ndarray, valid: np.ndarray, supports: list[np.ndarray]) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    result = np.zeros_like(raw, dtype=np.float32)
    for stage in range(STAGES):
        support = supports[stage]
        if support.size:
            left = np.searchsorted(support, raw[..., stage, :], side="left")
            right = np.searchsorted(support, raw[..., stage, :], side="right")
            mapped = np.clip((left + right - 1.0) / max(support.size - 1, 1), 0.0, 1.0)
            result[..., stage, valid] = mapped[..., valid]
    return result


def relational_v2(geometry: dict[str, np.ndarray], b1: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    c = geometry["query_peer_cos"]
    packed = geometry["peer_gram_upper"]
    valid = np.asarray(b1["valid_b1"], dtype=bool)
    baseline_pgm = fixed.pgm_raw(c, packed)
    baseline_pcrr = base.pcrr_raw(c, packed)
    pgm_ranks, pgm_support = base.fixed_cdf_components(baseline_pgm["raw"], valid)
    pcrr_ranks, pcrr_support = base.fixed_cdf_components(baseline_pcrr["raw"], valid)
    reserve_pgm_raw = []
    reserve_pcrr_raw = []
    for reserve_index in range(2):
        replacement_c, replacement_g = _replacement_geometry_v2(geometry, b1, reserve_index)
        c_flat = replacement_c.transpose(0, 2, 1, 3).reshape(-1, STAGES, PEERS)
        g_flat = replacement_g.transpose(0, 2, 1, 3).reshape(-1, STAGES, 36)
        reserve_pgm = fixed.pgm_raw(c_flat, g_flat)["raw"].reshape(PEERS, PATCHES, STAGES).transpose(0, 2, 1)
        reserve_pcrr = base.pcrr_raw(c_flat, g_flat)["raw"].reshape(PEERS, PATCHES, STAGES).transpose(0, 2, 1)
        reserve_pgm_raw.append(reserve_pgm)
        reserve_pcrr_raw.append(reserve_pcrr)
    reserve_pgm_raw_array = np.stack(reserve_pgm_raw)
    reserve_pcrr_raw_array = np.stack(reserve_pcrr_raw)
    reserve_pgm_rank = np.stack([
        _map_fixed(reserve_pgm_raw_array[r], b1["valid_p9" if r == 0 else "valid_p16"], pgm_support).mean(axis=-2)
        for r in range(2)
    ])
    reserve_pcrr_rank = np.stack([
        _map_fixed(reserve_pcrr_raw_array[r], b1["valid_p9" if r == 0 else "valid_p16"], pcrr_support).mean(axis=-2)
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


def _feature_credibility(geometry: dict[str, np.ndarray], b1: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    c = np.asarray(geometry["query_peer_cos"], dtype=np.float64)
    gram = decode_gram(np.asarray(geometry["peer_gram_upper"], dtype=np.float64))
    valid = np.asarray(b1["valid_b1"], dtype=bool)
    upper = np.triu_indices(PEERS, 1)
    peer_coherence = gram[:, :, upper[0], upper[1]].mean(axis=(0, 2)).astype(np.float32)
    query_support_mean = c.mean(axis=(0, 2)).astype(np.float32)
    H = np.eye(PEERS) - np.full((PEERS, PEERS), 1.0 / PEERS)
    centered = np.einsum("ij,spjk,kl->spil", H, gram, H)
    centered = (centered + np.swapaxes(centered, -1, -2)) * 0.5
    eigenvalues = np.linalg.eigvalsh(centered)[..., ::-1]
    maximum = np.maximum(eigenvalues[..., 0], 0.0)
    tolerance = np.finfo(np.float32).eps * np.maximum(1.0, maximum) * PEERS
    positive = eigenvalues > tolerance[..., None]
    positive_values = np.where(positive, eigenvalues, 0.0)
    total = positive_values.sum(axis=-1, keepdims=True)
    probabilities = np.divide(positive_values, np.maximum(total, 1e-30), where=np.ones_like(positive_values, dtype=bool))
    entropy_stage = -np.sum(np.where(positive, probabilities * np.log(np.maximum(probabilities, 1e-30)), 0.0), axis=-1) / np.log(PEERS - 1)
    entropy = np.clip(entropy_stage.mean(axis=0), 0.0, 1.0).astype(np.float32)
    profile_values = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        denominator = np.maximum(np.linalg.norm(c[left], axis=-1) * np.linalg.norm(c[right], axis=-1), 1e-12)
        profile_values.append(1.0 - np.sum(c[left] * c[right], axis=-1) / denominator)
    profile_disagreement = np.mean(profile_values, axis=0).astype(np.float32)
    for value in (peer_coherence, query_support_mean, entropy, profile_disagreement):
        value[~valid] = 0.0
    return {
        "peer_coherence": peer_coherence,
        "query_support_mean": query_support_mean,
        "peer_eigen_entropy": entropy,
        "stage_query_profile_disagreement": profile_disagreement,
    }

def trust_stability(relational: dict[str, np.ndarray], b1: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    baseline = relational["baseline_pgm"]
    reserve_ranks = relational["reserve_pgm_rank"]
    result: dict[str, np.ndarray] = {}
    for index, suffix in enumerate(("9", "16")):
        valid = b1["valid_p9" if index == 0 else "valid_p16"]
        replacements = reserve_ranks[index]
        boundary = 1.0 - np.abs(replacements[7] - baseline)
        influence = 1.0 - np.max(np.abs(replacements - baseline[None]), axis=0)
        robust = np.minimum.reduce(np.concatenate([baseline[None], replacements], axis=0))
        for value in (boundary, influence, robust):
            np.clip(value, 0.0, 1.0, out=value)
            value[~valid] = 0.0
        result[f"S{suffix}"] = influence.astype(np.float32)
        result[f"R{suffix}"] = robust.astype(np.float32)
        result[f"S_boundary{suffix}"] = boundary.astype(np.float32)
    return result


def build_compact_record(
    features: np.ndarray,
    native_margins: np.ndarray,
    image_path: str,
) -> tuple[dict[str, np.ndarray | str], dict[str, Any]]:
    stage_rank = np.stack([percentile_rank(native_margins[s]) for s in range(STAGES)]).astype(np.float32)
    d_rank = np.std(stage_rank.astype(np.float64), axis=0, ddof=0).astype(np.float32)
    b1 = construct_b1_v2(features, d_rank, native_margins)
    geometry = compact_geometry_v2(features, b1)
    relational = relational_v2(geometry, b1)
    credibility = _feature_credibility(geometry, b1)
    stability = trust_stability(relational, b1)
    record: dict[str, np.ndarray | str] = {
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
    transient = {"b1": b1, "geometry": geometry, "relational": relational}
    return record, transient

