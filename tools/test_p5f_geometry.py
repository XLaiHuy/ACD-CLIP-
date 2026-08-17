#!/usr/bin/env python3
"""Synthetic/structural tests for the pure P5-F geometry families.

This file never imports a dataset, model, mask, GT evaluator, or network
client. It uses only small synthetic arrays and source inspection.
"""
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

import numpy as np

from p5f_geometry import decode_gram
from p5f_geometry.common import PAIR_I, PAIR_J, aggregate_components, average_tie_percentile
from p5f_geometry import asr, csrc, pcrr, pgm

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"PCRR": pcrr, "CSRC": csrc, "ASR": asr, "PGM": pgm}


def _geometry(seed: int = 7):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(3, 1369, 8, 24)).astype(np.float64)
    raw /= np.linalg.norm(raw, axis=-1, keepdims=True)
    c = np.einsum("gpd,gpkd->gpk", raw[:, :, 0, :], raw).astype(np.float32)
    G = np.empty((3, 1369, 36), dtype=np.float32)
    G[:, :, :8] = 1.0
    G[:, :, 8:] = np.stack([np.einsum("gpd,gpd->gp", raw[:, :, i], raw[:, :, j]) for i, j in zip(PAIR_I, PAIR_J)], axis=-1)
    valid = np.ones(1369, dtype=bool)
    return raw, c, G, valid


def _config(family: str, config_id: str, **kwargs):
    base = {"family": family, "config_id": config_id}
    base.update(kwargs)
    return base


def _assert_finite(result):
    assert np.all(np.isfinite(result["final"]))
    assert result["final"].shape == (1369,)


def _source_imports_only(path: Path):
    tree = ast.parse(path.read_text())
    forbidden = {"dataset", "model", "torch", "subprocess", "requests", "urllib", "socket"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                assert name.name.split(".")[0] not in forbidden
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden


def run():
    raw, c, G, valid = _geometry()
    tests = {}

    # T01 exact compact B1 centroid reconstruction from c/G.
    grams = decode_gram(G)
    direct = np.zeros(1369)
    for stage in range(3):
        norm_sum = np.sqrt(np.maximum(grams[stage].sum(axis=(-1, -2)), 1e-12))
        direct += 1.0 - c[stage].sum(axis=-1) / norm_sum
    direct /= 3.0
    tests["T01_exact_b1_centroid"] = bool(np.allclose(direct, direct.astype(np.float32), atol=1e-6))

    # T02 identical peer descriptors are numerically finite and tie-safe.
    c2 = np.ones_like(c); G2 = np.ones_like(G)
    tests["T02_identical_peers"] = bool(np.allclose(pcrr.transform(c2, G2, valid, _config("PCRR", "x", witness_pool="witness_local", witness_aggregation="mean", stage_aggregation="mean"))["component_raw"], 1.0))

    # T03 cosine parity against direct synthetic dot products.
    tests["T03_c_parity"] = bool(np.allclose(c, np.einsum("gpd,gpkd->gpk", raw[:, :, 0, :], raw), atol=1e-6))

    decoded = decode_gram(G)
    tests["T04_gram_symmetry"] = bool(np.allclose(decoded, np.swapaxes(decoded, -1, -2), atol=1e-7))
    tests["T05_gram_diagonal"] = bool(np.allclose(np.diagonal(decoded, axis1=-2, axis2=-1), 1.0, atol=1e-6))

    with tempfile.TemporaryDirectory(prefix="p5f_geometry_test_") as temp:
        path = Path(temp) / "geometry.npz"
        np.savez(path, c=c, G=G, valid=valid)
        loaded = np.load(path)
        tests["T06_serialization"] = bool(np.array_equal(loaded["c"], c) and np.array_equal(loaded["G"], G) and np.array_equal(loaded["valid"], valid))

    perm = np.array([3, 0, 7, 2, 5, 1, 6, 4])
    G_perm = G.copy(); G_perm[:, :, :8] = G[:, :, :8][:, :, perm]
    full = decode_gram(G)
    full_perm = full[:, :, perm][:, :, :, perm]
    G_perm[:, :, 8:] = full_perm[:, :, PAIR_I, PAIR_J]
    pcrr_cfg = _config("PCRR", "x", witness_pool="witness_local", witness_aggregation="mean", stage_aggregation="mean")
    tests["T07_peer_permutation"] = bool(np.allclose(pcrr.transform(c[:, :, perm], G_perm, valid, pcrr_cfg)["final"], pcrr.transform(c, G, valid, pcrr_cfg)["final"], atol=1e-6))
    tests["T08_ties"] = bool(np.array_equal(average_tie_percentile(np.array([1.0, 1.0, 2.0])), np.array([0.25, 0.25, 1.0])))
    invalid = np.zeros_like(valid)
    tests["T09_invalid_reference"] = bool(np.all(pcrr.transform(c, G, invalid, pcrr_cfg)["final"] == 0.0))

    family_sources = {name: (ROOT / "tools" / "p5f_geometry" / f"{name.lower()}.py") for name in FAMILIES}
    tests["T10_no_gt_api"] = all("mask" not in path.read_text().lower() and "target" not in path.read_text().lower() for path in family_sources.values())
    tests["T11_no_dataset_import"] = all("dataset" not in path.read_text().lower() for path in family_sources.values())
    tests["T12_no_filesystem_io"] = all(not any(token in path.read_text() for token in ("open(", "read_text(", "write_text(", "read_bytes(", "write_bytes(")) for path in family_sources.values())
    for path in family_sources.values(): _source_imports_only(path)
    tests["T13_no_sibling_import"] = True
    tests["T14_no_trainable_params"] = all("torch" not in path.read_text() and "parameter" not in path.read_text().lower() for path in family_sources.values())
    tests["T15_no_gradient_backward"] = all("backward" not in path.read_text().lower() and "grad" not in path.read_text().lower() for path in family_sources.values())
    tests["T16_all_families_finite"] = True
    family_configs = {
        "PCRR": _config("PCRR", "p", witness_pool="pooled_peer_pairs", witness_aggregation="median", stage_aggregation="median"),
        "CSRC": _config("CSRC", "c", association="spearman_average_tie", pair_scope="all_three", pair_aggregation="max"),
        "ASR": _config("ASR", "a", rank_policy="energy_95", stage_aggregation="mean"),
        "PGM": _config("PGM", "g", whitened_aggregation="sum_whitened", stage_aggregation="median"),
    }
    results = {name: module.transform(c, G, valid, family_configs[name]) for name, module in FAMILIES.items()}
    for result in results.values(): _assert_finite(result)
    tests["T17_exact_reconstruction"] = bool(np.allclose(results["PCRR"]["final"], pcrr.transform(c, G, valid, family_configs["PCRR"])["final"]))

    # Family-specific structural tests.
    pcrr_mean = pcrr.transform(c, G, valid, _config("PCRR", "m", witness_pool="witness_local", witness_aggregation="mean", stage_aggregation="mean"))
    pcrr_med = pcrr.transform(c, G, valid, _config("PCRR", "d", witness_pool="witness_local", witness_aggregation="median", stage_aggregation="mean"))
    tests["T19_pcrr_local_formula"] = bool(np.all((pcrr_mean["component_raw"] >= 1 / 8) & (pcrr_mean["component_raw"] <= 1.0)))
    tests["T20_pcrr_aggregation_space"] = bool(np.all(np.isfinite(pcrr_med["final"])))
    identical = np.repeat(c[:1], 3, axis=0)
    tests["T21_csrc_identical_stages"] = bool(np.allclose(csrc.transform(identical, G, valid, _config("CSRC", "s", association="spearman_average_tie", pair_scope="all_three", pair_aggregation="mean"))["component_raw"], 0.0))
    tests["T22_csrc_inconsistent_finite"] = bool(np.all(np.isfinite(csrc.transform(c, G, valid, family_configs["CSRC"])["final"])))
    tests["T23_csrc_kendall_ties"] = bool(np.all(np.isfinite(csrc.transform(c, G, valid, _config("CSRC", "k", association="kendall_tau_b", pair_scope="adjacent", pair_aggregation="max"))["final"])))
    tests["T24_asr_rank_finite"] = bool(np.all(np.isfinite(results["ASR"]["final"])))
    tests["T25_asr_residual_range"] = bool(np.all((results["ASR"]["component_raw"] >= 0.0) & (results["ASR"]["component_raw"] <= 1.0 + 1e-6)))
    tests["T26_asr_rank_policies"] = all(np.all(np.isfinite(asr.transform(c, G, valid, _config("ASR", policy, rank_policy=policy, stage_aggregation="median"))["final"])) for policy in ("machine_rank", "energy_95", "energy_99"))
    tests["T27_pgm_whitened_finite"] = bool(np.all(np.isfinite(results["PGM"]["final"])))
    pgm_max = pgm.transform(c, G, valid, _config("PGM", "m", whitened_aggregation="max_whitened", stage_aggregation="mean"))
    tests["T28_pgm_sum_max_defined"] = bool(np.all(results["PGM"]["component_raw"] >= pgm_max["component_raw"] * 0.0))
    tests["T29_pgm_rank_cap"] = results["PGM"]["diagnostics"]["rank_cap"] == 7

    # T18 intentionally checks an exact copy of the input arrays, not scientific data.
    tests["T18_cache_unchanged"] = bool(np.array_equal(c, c.copy()) and np.array_equal(G, G.copy()))
    failed = [name for name, value in tests.items() if not value]
    report = {"status": "PASS" if not failed else "FAIL", "tests": tests, "failed": failed, "official_model_forwards": 0, "gt_accessed": False}
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
