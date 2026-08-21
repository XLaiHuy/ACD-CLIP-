"""Compatibility wrapper correcting axis normalization for the shared core."""
from __future__ import annotations

import numpy as np

try:
    from sabra import logic_core as _base
    from sabra.logic_core import *  # noqa: F401,F403
except ModuleNotFoundError:  # package import from the repository root
    from tools.sabra import logic_core as _base
    from tools.sabra.logic_core import *  # noqa: F401,F403


def pgm_raw(c: np.ndarray, packed_gram: np.ndarray) -> dict[str, np.ndarray]:
    c = np.asarray(c)
    packed_gram = np.asarray(packed_gram)
    stage_first = c.ndim == 3 and c.shape[:2] == (_base.STAGES, _base.PATCHES)
    if not stage_first:
        return _base.pgm_raw(c, packed_gram)
    result = _base.pgm_raw(c.transpose(1, 0, 2), packed_gram.transpose(1, 0, 2))
    return {
        key: value.transpose(1, 0) if value.ndim == 2 else value.transpose(1, 0)
        for key, value in result.items()
    }


def compute_relational_scores(geometry: dict[str, np.ndarray], b1: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    c = geometry["query_peer_cos"]
    packed = geometry["peer_gram_upper"]
    valid = np.asarray(b1["valid_b1"], dtype=bool)
    baseline_pgm = pgm_raw(c, packed)
    baseline_pcrr = _base.pcrr_raw(c, packed)
    pgm_ranks, pgm_support = _base.fixed_cdf_components(baseline_pgm["raw"], valid)
    pcrr_ranks, pcrr_support = _base.fixed_cdf_components(baseline_pcrr["raw"], valid)
    replacement_c, replacement_g = _base.replacement_geometry(geometry, b1)
    replacement_c_flat = replacement_c.transpose(0, 2, 1, 3).reshape(-1, _base.STAGES, _base.PEERS)
    replacement_g_flat = replacement_g.transpose(0, 2, 1, 3).reshape(-1, _base.STAGES, 36)
    replacement_pgm = _base.pgm_raw(replacement_c_flat, replacement_g_flat)
    replacement_pcrr = _base.pcrr_raw(replacement_c_flat, replacement_g_flat)
    replacement_pgm_raw = replacement_pgm["raw"].reshape(_base.PEERS, _base.PATCHES, _base.STAGES).transpose(0, 2, 1)
    replacement_pcrr_raw = replacement_pcrr["raw"].reshape(_base.PEERS, _base.PATCHES, _base.STAGES).transpose(0, 2, 1)
    stable = np.asarray(b1["valid_stability"], dtype=bool)
    replacement_pgm_rank = _base.map_fixed_cdf(replacement_pgm_raw, stable, pgm_support).mean(axis=-2)
    replacement_pcrr_rank = _base.map_fixed_cdf(replacement_pcrr_raw, stable, pcrr_support).mean(axis=-2)
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
