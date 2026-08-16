#!/usr/bin/env python3
"""Frozen, GT-free P5B positive-only minimum projection.

This module is deliberately independent of the Phase2B predictor.  It accepts
already-produced native quantities, proposes relations from the frozen R0
protocol, and applies one bounded native-logit action per selected relation.
No labels, masks, scores, AP, or learned state are accepted or consulted.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np


PATCH_COUNT = 37 * 37
STAGES = 3
K = 8
BINS = 10
RISK_FRACTION = 0.20
IMAGE_SIZE = 518
PATCH_GRID = (37, 37)


def _array(name: str, value: np.ndarray, shape: tuple[int, ...], *, finite: bool = True) -> np.ndarray:
    out = np.asarray(value)
    if out.shape != shape:
        raise ValueError(f"{name} shape={out.shape}, expected={shape}")
    if finite and not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _vector(name: str, value: np.ndarray, *, dtype: np.dtype | None = None) -> np.ndarray:
    out = np.asarray(value if dtype is None else np.asarray(value, dtype=dtype)).reshape(-1)
    if out.size != PATCH_COUNT:
        raise ValueError(f"{name} size={out.size}, expected={PATCH_COUNT}")
    if np.issubdtype(out.dtype, np.floating) and not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def stable_desc_order(values: np.ndarray) -> np.ndarray:
    """Exact audited descending-value/ascending-patch-ID order."""
    values = _vector("values", values, dtype=np.float64)
    return np.lexsort((np.arange(PATCH_COUNT, dtype=np.int64), -values))


def quantile_bins(values: np.ndarray, bins: int = BINS) -> np.ndarray:
    """Exact audited stable rank quantile bins, including ties."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    values = _vector("quantile_values", values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    out = np.empty(PATCH_COUNT, dtype=np.int64)
    out[order] = np.minimum((np.arange(PATCH_COUNT) * bins) // PATCH_COUNT, bins - 1)
    return out


def frozen_cells(
    m_bar: np.ndarray,
    d_rank: np.ndarray,
    valid_reference: np.ndarray,
    score_bin: np.ndarray | None = None,
    d_rank_bin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Reconstruct the exact R0 risk/cell population without GT."""
    m_bar = _vector("m_bar", m_bar, dtype=np.float64)
    d_rank = _vector("D_rank", d_rank, dtype=np.float64)
    valid_reference = _vector("valid_reference", valid_reference).astype(bool)
    expected_score = quantile_bins(m_bar)
    expected_rank = quantile_bins(d_rank)
    if score_bin is not None and not np.array_equal(_vector("score_bin", score_bin).astype(np.int64), expected_score):
        raise ValueError("score_bin does not match exact audited quantile bins")
    if d_rank_bin is not None and not np.array_equal(_vector("d_rank_bin", d_rank_bin).astype(np.int64), expected_rank):
        raise ValueError("d_rank_bin does not match exact audited quantile bins")
    risk_count = int(math.ceil(RISK_FRACTION * PATCH_COUNT))
    risk = np.zeros(PATCH_COUNT, dtype=bool)
    risk[stable_desc_order(d_rank)[:risk_count]] = True
    eligible = risk & valid_reference
    cells: list[np.ndarray] = []
    for score_cell in range(BINS):
        for rank_cell in range(BINS):
            cells.append(np.flatnonzero(eligible & (expected_score == score_cell) & (expected_rank == rank_cell)))
    return expected_score, expected_rank, eligible, cells


def raw_relations(m_bar: np.ndarray, evidence: np.ndarray, cells: Iterable[np.ndarray]) -> list[tuple[int, int, float]]:
    """Return strict base inversions for which E prefers the lower-base patch."""
    # Preserve the predictor's persisted dtype for exact R0 cost parity.
    m_bar = _vector("m_bar", m_bar)
    evidence = _vector("E_nonlocal", evidence)
    out: list[tuple[int, int, float]] = []
    for members in cells:
        members = np.asarray(members, dtype=np.int64).reshape(-1)
        if members.size and (members.min() < 0 or members.max() >= PATCH_COUNT):
            raise ValueError("cell patch ID out of range")
        for a in range(int(members.size)):
            for b in range(a + 1, int(members.size)):
                p, q = int(members[a]), int(members[b])
                if m_bar[p] < m_bar[q] and evidence[p] > evidence[q]:
                    out.append((p, q, float(m_bar[q] - m_bar[p])))
                elif m_bar[q] < m_bar[p] and evidence[q] > evidence[p]:
                    out.append((q, p, float(m_bar[p] - m_bar[q])))
    return out


def certified_relations(
    raw: Iterable[tuple[int, int, float]],
    e_stage: np.ndarray,
    e_loo: np.ndarray,
) -> list[tuple[int, int, float]]:
    """Keep only strict 3-stage and 8-view unanimous relations."""
    e_stage = _array("E_stage", e_stage, (STAGES, PATCH_COUNT))
    e_loo = _array("E_LOO", e_loo, (K, PATCH_COUNT))
    out: list[tuple[int, int, float]] = []
    for i, j, cost in raw:
        if not (0 <= i < PATCH_COUNT and 0 <= j < PATCH_COUNT and i != j):
            raise ValueError("relation patch ID out of range")
        if not np.isfinite(cost) or cost <= 0:
            raise ValueError("relation cost must be finite and positive")
        if bool(np.all(e_stage[:, i] > e_stage[:, j])) and bool(np.all(e_loo[:, i] > e_loo[:, j])):
            out.append((int(i), int(j), float(cost)))
    return out


def select_disjoint(certified: Iterable[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """Select one deterministic, disjoint, minimum-cost pass."""
    ordered = sorted(
        ((int(i), int(j), float(cost)) for i, j, cost in certified),
        key=lambda item: (item[2], min(item[0], item[1]), max(item[0], item[1])),
    )
    used: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for i, j, cost in ordered:
        if i in used or j in used:
            continue
        out.append((i, j, cost))
        used.update((i, j))
    return out


def select_gt_free(
    m_bar: np.ndarray,
    d_rank: np.ndarray,
    valid_reference: np.ndarray,
    e_nonlocal: np.ndarray,
    e_stage: np.ndarray,
    e_loo: np.ndarray,
    score_bin: np.ndarray | None = None,
    d_rank_bin: np.ndarray | None = None,
) -> dict[str, list[tuple[int, int, float]] | np.ndarray]:
    """Run the frozen R0 raw/certified/selected pipeline with no GT input."""
    # Keep float32 m_bar/E values through raw relation arithmetic; the audited
    # R0 driver persisted and compared those arrays at their native dtype.
    m_bar = _vector("m_bar", m_bar)
    d_rank = _vector("D_rank", d_rank)
    e_nonlocal = _vector("E_nonlocal", e_nonlocal)
    e_stage = _array("E_stage", e_stage, (STAGES, PATCH_COUNT))
    e_loo = _array("E_LOO", e_loo, (K, PATCH_COUNT))
    _, _, _, cells = frozen_cells(m_bar, d_rank, valid_reference, score_bin, d_rank_bin)
    raw = raw_relations(m_bar, e_nonlocal, cells)
    certified = certified_relations(raw, e_stage, e_loo)
    selected = select_disjoint(certified)
    return {"raw": raw, "certified": certified, "selected": selected}


def apply_positive_only_projection(
    native_logits: np.ndarray,
    m_bar: np.ndarray,
    selected: Iterable[tuple[int, int, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one positive anomaly-logit uplift for each selected relation."""
    native_logits = _array("native_logits", native_logits, (STAGES, PATCH_COUNT, 2)).astype(np.float32, copy=True)
    # The action gap must use the same native dtype as the selected trace.
    m_bar = _vector("m_bar", m_bar)
    delta = np.zeros(PATCH_COUNT, dtype=np.float32)
    used: set[int] = set()
    for i, j, cost in selected:
        i, j = int(i), int(j)
        if i in used or j in used:
            raise ValueError("selected relation repeats a patch")
        if not (m_bar[i] < m_bar[j]):
            raise ValueError("selected relation is not a strict base inversion")
        gap = float(m_bar[j] - m_bar[i])
        if not np.isfinite(gap) or gap <= 0:
            raise ValueError("selected relation has invalid positive-only gap")
        if not np.isfinite(cost) or float(cost) != gap:
            raise ValueError("selected relation cost does not equal minimum projection gap")
        delta[i] = np.float32(gap)
        used.update((i, j))
    corrected = native_logits.copy()
    corrected[:, :, 1] += delta[None, :]
    if not np.all(np.isfinite(corrected)):
        raise ValueError("corrected native logits are non-finite")
    return corrected, delta


def deploy_pre_softmax(native_logits: np.ndarray, image_size: int = IMAGE_SIZE):
    """Exact native -> blur -> resize -> stage mean operator."""
    import torch
    import torch.nn.functional as F
    from model.adapter import gaussian_blur2d

    native = _array("native_logits", native_logits, (STAGES, PATCH_COUNT, 2)).astype(np.float32, copy=False)
    tensor = torch.from_numpy(native).float()
    outputs = []
    for stage in range(STAGES):
        logits = tensor[stage].permute(1, 0).reshape(1, 2, *PATCH_GRID)
        logits = gaussian_blur2d(logits, (7, 7), (1, 1))
        outputs.append(F.interpolate(logits, size=(image_size, image_size), mode="bilinear", align_corners=True))
    return torch.stack(outputs).mean(dim=0)


def deploy_native_logits(native_logits: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Exact native -> blur -> resize -> stage mean -> softmax deployment."""
    import torch.nn.functional as F

    return F.softmax(deploy_pre_softmax(native_logits, image_size), dim=1).numpy()


__all__ = [
    "apply_positive_only_projection",
    "certified_relations",
    "deploy_pre_softmax",
    "deploy_native_logits",
    "frozen_cells",
    "quantile_bins",
    "raw_relations",
    "select_disjoint",
    "select_gt_free",
    "stable_desc_order",
]
