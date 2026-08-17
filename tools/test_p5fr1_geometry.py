#!/usr/bin/env python3
"""Synthetic and structural tests for the P5-FR1 geometry contract.

No MVTec image, mask, label, dataset object, model forward, or GT evaluator is
used here. The direct-reference tests operate only on synthetic normalized
vectors.
"""
from __future__ import annotations

import ast
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from audit_p5fr1_mvtec import atomic_npz, compact_geometry, validate_record
from p5f_geometry import PACKED_GRAM_COUNT, GRAM_LAYOUT, decode_gram, pack_gram
from p5f_geometry.common import (
    DIAG_COUNT, OFFDIAG_COUNT, PAIR_I, PAIR_J, aggregate_components,
    average_tie_percentile,
)
from p5f_geometry import asr, csrc, pcrr, pgm

ROOT = Path(__file__).resolve().parents[1]
FAMILY_MODULES = {"PCRR": pcrr, "CSRC": csrc, "ASR": asr, "PGM": pgm}


def cfg(family: str, config_id: str, **kwargs):
    result = {"family": family, "config_id": config_id}
    result.update(kwargs)
    return result


def synthetic(seed: int = 17, dim: int = 13):
    rng = np.random.default_rng(seed)
    peers = rng.normal(size=(3, 1369, 8, dim)).astype(np.float64)
    peers /= np.linalg.norm(peers, axis=-1, keepdims=True)
    query = rng.normal(size=(3, 1369, dim)).astype(np.float64)
    query /= np.linalg.norm(query, axis=-1, keepdims=True)
    c = np.einsum("gpd,gpkd->gpk", query, peers).astype(np.float32)
    full = np.einsum("gpkd,gpld->gpkl", peers, peers).astype(np.float64)
    G = pack_gram(full)
    valid = np.ones(1369, dtype=bool)
    return query, peers, c, G, valid


def finite(result):
    return bool(np.all(np.isfinite(result["final"]))) and result["final"].shape == (1369,)


def brute_kendall(x, y):
    x, y = np.asarray(x), np.asarray(y)
    concordant = discordant = tie_x_only = tie_y_only = both = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                both += 1
            elif dx == 0:
                tie_x_only += 1
            elif dy == 0:
                tie_y_only += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    del both
    if np.all(x == x[0]) and np.all(y == y[0]): return 1.0
    if np.all(x == x[0]) or np.all(y == y[0]): return 0.0
    denom = math.sqrt((concordant + discordant + tie_x_only) * (concordant + discordant + tie_y_only))
    return 0.0 if denom == 0 else (concordant - discordant) / denom


def direct_centered(query, peer, policy):
    k = peer.shape[0]
    w = np.full(k, 1.0 / k)
    H = np.eye(k) - np.full((k, k), 1.0 / k)
    G = peer @ peer.T
    c = peer @ query
    b = H @ (c - G @ w)
    t = max(float(1.0 - 2.0 * w @ c + w @ G @ w), 0.0)
    C = (H @ G @ H); C = (C + C.T) / 2.0
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]; vals, vecs = vals[order], vecs[:, order]
    tol = np.finfo(np.float32).eps * max(1.0, max(float(vals[0]), 0.0)) * k
    positive = vals > tol
    if policy == "machine_rank":
        rank = int(positive.sum())
    else:
        target = 0.95 if policy == "energy_95" else 0.99
        total = float(vals[positive].sum()); rank = 0; cumulative = 0.0
        if total > 0:
            for value in vals:
                if value <= tol: continue
                rank += 1; cumulative += float(value)
                if cumulative / total >= target: break
    rank = min(rank, k - 1)
    proj = sum(float((b @ vecs[:, j]) ** 2 / vals[j]) for j in range(rank) if vals[j] > tol)
    out = max(t - proj, 0.0)
    return float(np.clip(0.0 if t <= np.finfo(np.float32).eps else out / t, 0.0, 1.0)), vals, vecs, b, t, tol, rank


def run():
    query, peer_bank, c, G, valid = synthetic()
    tests = {}

    # T01: shared schema constants.
    tests["T01_packed_count_shape"] = (GRAM_LAYOUT == "diag8_then_offdiag28" and DIAG_COUNT == 8 and OFFDIAG_COUNT == 28 and PACKED_GRAM_COUNT == 36 and G.shape == (3, 1369, PACKED_GRAM_COUNT))

    # T02: multiple-seed direct Gram round trips, including non-unit diagonals.
    roundtrip = True
    for seed in range(5):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=(3, 7, 8, 8)); full = (a + np.swapaxes(a, -1, -2)) / 2
        roundtrip &= np.allclose(decode_gram(pack_gram(full)), full, atol=1e-7, rtol=1e-7)
    tests["T02_gram_roundtrip"] = bool(roundtrip)

    # T03: production compact_geometry emits the exact production shapes.
    stage_features = [peer_bank[g, :, 0, :].astype(np.float32) for g in range(3)]
    peer_ids = np.tile(np.arange(8, dtype=np.int64), (1369, 1))
    compact_c, compact_G, centroid, diag_max = compact_geometry(stage_features, peer_ids, np.zeros((3, 1369), np.float32), np.zeros(1369, np.float32), valid)
    tests["T03_production_compact_shapes"] = (compact_c.shape == (3, 1369, 8) and compact_G.shape == (3, 1369, PACKED_GRAM_COUNT) and centroid.shape == (1369,) and np.isfinite(diag_max))

    # T04: production NPZ write and validator round trip.
    identity = {"canonical_index": 0, "class_name": "synthetic", "image_path": "test/synthetic.png"}
    with tempfile.TemporaryDirectory(prefix="p5fr1_geometry_") as td:
        path = Path(td) / "record.npz"
        arrays = {
            "peer_indices": peer_ids,
            "valid_reference": valid,
            "query_peer_cos": compact_c,
            "peer_gram_upper": compact_G,
            "b1_centroid_patch": centroid,
            "native_stage_logits": np.zeros((3, 1369, 2), np.float32),
            "native_stage_margins": np.zeros((3, 1369), np.float32),
            "d_rank_patch": np.zeros(1369, np.float32),
            "deployed_score_patch": np.zeros(518 * 518, np.float32),
            "deployed_margin_patch": np.zeros(518 * 518, np.float32),
            "gram_diag_max_abs": np.asarray(diag_max, np.float32),
            "class_name": np.asarray("synthetic"),
        }
        atomic_npz(path, arrays)
        validated = validate_record(path, identity, "protocol-test")
        tests["T04_production_record_roundtrip"] = validated["protocol_sha"] == "protocol-test"

    # T05: direct full Gram parity, no synthesized diagonal.
    decoded = decode_gram(G)
    tests["T05_direct_full_gram_parity"] = bool(np.allclose(decoded, np.einsum("gpkd,gpld->gpkl", peer_bank, peer_bank), atol=1e-6))

    configs = {
        "PCRR": cfg("PCRR", "p", witness_pool="pooled_peer_pairs", witness_aggregation="median", stage_aggregation="median"),
        "CSRC": cfg("CSRC", "c", association="spearman_average_tie", pair_scope="all_three", pair_aggregation="max"),
        "ASR": cfg("ASR", "a", rank_policy="energy_95", stage_aggregation="mean"),
        "PGM": cfg("PGM", "g", whitened_aggregation="sum_whitened", stage_aggregation="median"),
    }

    # T06: PCRR compact result equals an independent direct peer reference.
    direct_pcrr = np.zeros((3, 1369))
    for g in range(3):
        dist = 1.0 - np.einsum("gpkd,gpld->gpkl", peer_bank[g:g+1], peer_bank[g:g+1])[0]
        qdist = 1.0 - c[g]
        pairdist = dist[:, PAIR_I, PAIR_J]
        values = (1.0 + (pairdist[:, None, :] <= qdist[:, :, None]).sum(axis=2)) / 29.0
        direct_pcrr[g] = np.median(values, axis=1)
    direct_pcrr_out = aggregate_components(direct_pcrr, valid, "median")
    tests["T06_pcrr_direct_parity"] = bool(np.allclose(pcrr.transform(c, G, valid, configs["PCRR"])["final"], direct_pcrr_out["final"], atol=1e-7))

    # T07: ASR all rank policies equal direct feature-space SVD formula.
    asr_ok = True
    for policy in ("machine_rank", "energy_95", "energy_99"):
        direct_raw = np.zeros((3, 1369))
        for g in range(3):
            for p in range(1369):
                direct_raw[g, p] = direct_centered(query[g, p], peer_bank[g, p], policy)[0]
        expected = aggregate_components(direct_raw, valid, "mean")["final"]
        got = asr.transform(c, G, valid, cfg("ASR", policy, rank_policy=policy, stage_aggregation="mean"))["final"]
        asr_ok &= np.allclose(got, expected, atol=2e-6, rtol=2e-6)
    tests["T07_asr_direct_feature_parity"] = bool(asr_ok)

    # T08: PGM direct PCA whitened coordinates, sum and max.
    pgm_ok = True
    for aggregation in ("sum_whitened", "max_whitened"):
        direct_raw = np.zeros((3, 1369))
        for g in range(3):
            for p in range(1369):
                _, vals, vecs, b, _, tol, rank = direct_centered(query[g, p], peer_bank[g, p], "machine_rank")
                coordinates = [(7.0 * (b @ vecs[:, j]) ** 2) / (vals[j] ** 2) for j in range(rank) if vals[j] > tol]
                direct_raw[g, p] = (sum(coordinates) if aggregation == "sum_whitened" else max(coordinates, default=0.0))
        expected = aggregate_components(direct_raw, valid, "median")["final"]
        got = pgm.transform(c, G, valid, cfg("PGM", aggregation, whitened_aggregation=aggregation, stage_aggregation="median"))["final"]
        pgm_ok &= np.allclose(got, expected, atol=2e-6, rtol=2e-6)
    tests["T08_pgm_direct_feature_parity"] = bool(pgm_ok)

    # T09: exact Kendall tau-b against an independent brute-force implementation.
    tie_cases = [
        ([1, 1, 2, 3], [1, 2, 2, 3]),
        ([1, 1, 1], [2, 2, 2]),
        ([1, 2, 3], [3, 2, 1]),
        ([1, 1, 2], [1, 1, 3]),
    ]
    tests["T09_kendall_exact_ties"] = all(abs(csrc._kendall_tau_b(np.asarray(x), np.asarray(y)) - brute_kendall(x, y)) < 1e-12 for x, y in tie_cases)

    # T10: Spearman parity and frozen constant rules.
    tests["T10_spearman_degenerate_and_percentile"] = (
        csrc._spearman(np.ones(8), np.ones(8)) == 1.0 and
        csrc._spearman(np.ones(8), np.arange(8)) == 0.0 and
        np.isclose(csrc._spearman(np.arange(8), np.arange(8)[::-1]), -1.0) and
        np.allclose(average_tie_percentile(np.array([3., 1., 1., 2.])), np.array([1., 0.16666666666666666, 0.16666666666666666, 0.6666666666666666]))
    )

    # T11: every frozen configuration is finite on synthetic compact geometry.
    canonical = json.loads((ROOT / "runs/phase5/hsir/P5FR1_MVTEC_FOUR_FAMILY_V2/CANONICAL_CONFIGS.json").read_text())
    finite_all = True
    config_ids = []
    for family, rows in canonical["families"].items():
        for row in rows:
            config_ids.append(row["config_id"])
            finite_all &= finite(FAMILY_MODULES[family].transform(c, G, valid, {"family": family, **row}))
    tests["T11_all_26_finite"] = bool(len(config_ids) == 26 and len(set(config_ids)) == 26 and finite_all)

    # T12: all-config production serialization has exact IDs and shape.
    with tempfile.TemporaryDirectory(prefix="p5fr1_all_config_") as td:
        out = Path(td) / "all.npz"
        ids = np.asarray(config_ids)
        evidence = np.vstack([FAMILY_MODULES[family].transform(c, G, valid, {"family": family, **row})["final"] for family, rows in canonical["families"].items() for row in rows]).astype(np.float32)
        atomic_npz(out, {"config_ids": ids, "evidence": evidence, "valid_reference": valid})
        with np.load(out, allow_pickle=False) as data:
            tests["T12_all_config_serialization"] = (data["evidence"].shape == (26, 1369) and list(data["config_ids"].astype(str)) == config_ids and data["valid_reference"].dtype == np.bool_)

    # T13: invalid reference gives zero final evidence in every family.
    zero_valid = np.zeros(1369, dtype=bool)
    tests["T13_invalid_reference_all_families"] = all(np.all(FAMILY_MODULES[family].transform(c, G, zero_valid, {"family": family, **row})["final"] == 0.0) for family, rows in canonical["families"].items() for row in rows)

    # T14: permuting peer slots leaves every family invariant.
    perm = np.array([3, 0, 7, 2, 5, 1, 6, 4])
    c_perm = c[:, :, perm]
    full_perm = np.take(np.take(decoded, perm, axis=2), perm, axis=3)
    G_perm = pack_gram(full_perm)
    perm_ok = True
    for family, rows in canonical["families"].items():
        for row in rows:
            base = FAMILY_MODULES[family].transform(c, G, valid, {"family": family, **row})["final"]
            other = FAMILY_MODULES[family].transform(c_perm, G_perm, valid, {"family": family, **row})["final"]
            perm_ok &= np.allclose(base, other, atol=2e-6, rtol=2e-6)
    tests["T14_peer_permutation_invariant"] = bool(perm_ok)

    # T15: GT-free runner has no dataset/mask/label performance path.
    source = (ROOT / "tools/audit_p5fr1_mvtec.py").read_text()
    tests["T15_gt_free_barrier_static"] = all(token not in source for token in ("BaseSingleClassDataset", "mask_path", "target_occupancy", "raw[\"mask\"]", "label")) and "ground_truth" in source

    # T16: no real MVTec forward or cache state is created by this suite.
    cache_root = Path("/tmp/p5fr1_mvtec_common")
    tests["T16_no_official_forward"] = not (cache_root / "RUN_STATUS.json").exists() and not any((cache_root / "records").glob("*.npz")) if (cache_root / "records").exists() else not (cache_root / "RUN_STATUS.json").exists()

    tests = {name: bool(value) for name, value in tests.items()}
    failed = [name for name, value in tests.items() if not value]
    report = {"status": "PASS" if not failed else "FAIL", "tests": tests, "failed": failed, "official_model_forwards": 0, "gt_accessed": False, "gram_layout": GRAM_LAYOUT, "packed_count": PACKED_GRAM_COUNT}
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
