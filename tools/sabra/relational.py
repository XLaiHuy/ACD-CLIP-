"""Thin canonical wrapper around the audited Trust-v2 geometry backends.

Relational construction is deliberately GT-free.  The exact backend in
``tools.sabra.trust_v2.numerical`` is authoritative; the certified FAST
backend delegates all numerical semantics to that implementation.
"""
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
BACKEND_VERSION = "trust-v2-numerical-p16-v2"


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Tie-aware mid-rank used by the audited backend and margin features."""
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


def _backend_builder(backend: str):
    normalized = str(backend).lower()
    if normalized == "exact":
        from .trust_v2.numerical import build_compact_record
        return normalized, build_compact_record
    if normalized == "fast":
        from .trust_v2.fast_geometry import build_compact_record_fast
        return normalized, build_compact_record_fast
    raise ValueError(f"unknown SABRA backend {backend!r}; expected exact or fast")


def build_relational_record(
    stage_features: np.ndarray,
    native_margins: np.ndarray,
    deployment_sensitivity: np.ndarray | None = None,
    image_path: str | None = None,
    backend: str = "fast",
) -> dict[str, Any]:
    """Build one GT-free record using the authoritative exact/FAST backend."""
    features = np.asarray(stage_features, dtype=np.float32)
    margins = np.asarray(native_margins, dtype=np.float32)
    if features.ndim != 3 or features.shape[:2] != (STAGES, PATCHES):
        raise ValueError(f"expected features [3,1369,D], got {features.shape}")
    if margins.shape != (STAGES, PATCHES):
        raise ValueError(f"expected margins [3,1369], got {margins.shape}")
    if deployment_sensitivity is None:
        raise ValueError("deployment_sensitivity must be computed; zero placeholders are forbidden")
    sensitivity = np.asarray(deployment_sensitivity, dtype=np.float32).reshape(-1)
    if sensitivity.shape != (PATCHES,) or not np.isfinite(sensitivity).all():
        raise ValueError("deployment sensitivity must be finite [1369]")
    normalized, builder = _backend_builder(backend)
    compact, transient = builder(features, margins, str(image_path or ""))
    b1 = transient["b1"]
    geometry = transient["geometry"]
    relational = transient["relational"]
    mean_margin = margins.mean(axis=0)
    median_margin = float(np.median(mean_margin))
    mad_margin = float(np.median(np.abs(mean_margin - median_margin)))
    record: dict[str, Any] = {
        **compact,
        "E": np.asarray(compact["baseline_pgm"], dtype=np.float32),
        "pgm_raw": np.asarray(relational["pgm_raw"], dtype=np.float32),
        "peer_indices": np.asarray(compact["peer_indices"], dtype=np.int64),
        "valid_peers": np.asarray(compact["valid_b1"], dtype=bool),
        "candidate_count": np.asarray(compact["candidate_count"], dtype=np.int32),
        "query_peer_cos": np.asarray(geometry["query_peer_cos"], dtype=np.float32),
        "peer_gram_upper": np.asarray(geometry["peer_gram_upper"], dtype=np.float32),
        "valid_b1": np.asarray(compact["valid_b1"], dtype=bool),
        "margin_within_image_rank": percentile_rank(mean_margin),
        "robust_margin_normalization": ((mean_margin - median_margin) / (mad_margin + 1e-6)).astype(np.float32),
        "deployment_sensitivity": sensitivity,
        "backend": normalized,
        "backend_version": BACKEND_VERSION,
        "class_name": None,
        "image_path": image_path,
    }
    # ``compact`` already carries the exact D_rank/stability/credibility
    # fields.  Keep the transient geometry out of persisted records except for
    # the compact arrays explicitly required by the audit contract.
    if not np.isfinite(np.asarray(record["E"], dtype=np.float32)).all():
        raise FloatingPointError("SABRA evidence is non-finite")
    return record


def trust_features(record: Mapping[str, Any]) -> np.ndarray:
    return np.stack([np.asarray(record[name], dtype=np.float64) for name in FEATURE_ORDER], axis=-1)


def need_features(record: Mapping[str, Any]) -> np.ndarray:
    return np.stack([np.asarray(record[name], dtype=np.float64) for name in NEED_ORDER], axis=-1)


def assert_gt_free_payload(payload: Mapping[str, Any]) -> None:
    forbidden = {"mask", "label", "mask_path", "pixel_gt", "image_label", "trust_target", "need_target"}
    found = sorted(forbidden.intersection(payload))
    if found:
        raise AssertionError(f"GT-bearing fields reached relational construction: {found}")
