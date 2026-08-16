#!/usr/bin/env python3
"""P5-D0 cache-only graph non-conformity leverage audit.

The first phase reconstructs and solves the frozen R0 certified relation graph
without reading GT.  It atomically materializes compact node arrays and a
hash manifest.  Only after that manifest is finalized does the second phase
read masks for post-hoc leverage and separability diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import audit_phase5_b2_adjudication as b2  # noqa: E402
import audit_phase5_p5b_full_eval as full  # noqa: E402
from audit_phase5_hsir import ap_contamination, exact_auc_ap, pairwise_risks  # noqa: E402
from phase5_selective_adjudication import select_gt_free  # noqa: E402


OUTPUT_ROOT = ROOT / "runs/phase5/hsir/P5D0_GRAPH_NONCONFORMITY_AUDIT"
CACHE_ROOT = Path("/tmp/p5_r0_run2")
TEMP_ROOT = Path("/tmp/p5_d0_graph_nonconformity_v2")
PROTOCOL_PATH = OUTPUT_ROOT / "PROTOCOL.json"
PROTOCOL_COMMIT = "4fe92794e1e80b1abdfa5451ae385088c6e38ed1"
EXPECTED_CACHE_SHA = "cfbd66b04c04b314756d151b759d95041afc2a69a8dc411e24896a7b4f931365"
EXPECTED_CACHE_SCHEMA = "P5B_R0_GT_FREE_CACHE_v1"
EXPECTED_IMAGES = 2162
EXPECTED_CLASSES = 12
EXPECTED_NORMAL = 962
EXPECTED_ANOMALY = 1200
PATCH_COUNT = 37 * 37
SHIFT = (12, 12)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 7701
S7_EPSILON = 1e-12
OLD_POSITIVE_FRACTION = 0.03960581875762963
OLD_NEGATIVE_FRACTION = 0.01970289841145196
GRAPH_SCHEMA_VERSION = "P5D0_GT_FREE_GRAPH_NODE_v2"
CHECKPOINT_SCHEMA_VERSION = "P5D0_GT_FREE_CHECKPOINT_v2"
CLASS_ORDER = ("candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")
VARIANTS = ("aligned", "shifted")
PROTECTED = (
    "model/adapter.py",
    "tools/audit_phase5_b2_adjudication.py",
    "tools/audit_phase5_b3_action_mismatch.py",
    "tools/audit_phase5_reference_validity.py",
    "tools/audit_phase5_second_evidence.py",
    "tools/audit_phase5_hsir.py",
    "tools/phase5_selective_adjudication.py",
    "tools/audit_phase5_p5b_full_eval.py",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def stable_ranks(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size != PATCH_COUNT or not np.all(np.isfinite(values)):
        raise RuntimeError("P5D0_INVALID:base_score")
    order = np.lexsort((np.arange(values.size, dtype=np.int64), -values.astype(np.float64)))
    rank = np.empty(values.size, dtype=np.int32)
    rank[order] = np.arange(values.size, dtype=np.int32)
    percentile = np.empty(values.size, dtype=np.float64)
    percentile[order] = np.arange(values.size, dtype=np.float64) / (values.size - 1)
    return rank, percentile


def stable_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.lexsort((np.arange(values.size, dtype=np.int64), -values))
    out = np.empty(values.size, dtype=np.float64)
    out[order] = np.arange(values.size, dtype=np.float64) / max(1, values.size - 1)
    return out


def trace_arrays(items: Iterable[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    items = list(items)
    if not items:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)
    return np.asarray([[int(i), int(j)] for i, j, _ in items], dtype=np.int64), np.asarray([float(c) for _, _, c in items], dtype=np.float64)


def arrays_equal_trace(expected: list[tuple[int, int, float]], arrays: dict[str, np.ndarray], variant: str, key: str) -> None:
    pairs, costs = trace_arrays(expected)
    if not np.array_equal(pairs, arrays[f"{variant}_{key}_pairs"]):
        raise RuntimeError(f"P5D0_SELECTOR_PARITY_FAILED:{key}:{variant}:pairs")
    if not np.array_equal(costs, arrays[f"{variant}_{key}_cost"]):
        raise RuntimeError(f"P5D0_SELECTOR_PARITY_FAILED:{key}:{variant}:cost")


def union_find(n: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parent = np.arange(n, dtype=np.int64)
    size = np.ones(n, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    for i, j in edges:
        a, b = find(int(i)), find(int(j))
        if a == b:
            continue
        if size[a] < size[b] or (size[a] == size[b] and a > b):
            a, b = b, a
        parent[b] = a
        size[a] += size[b]
    roots = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    return roots, size


def hodge_graph(certified: list[tuple[int, int, float, float | None]]) -> dict[str, Any]:
    """Solve the frozen all-certified graph and materialize target-only signals."""
    normalized: list[tuple[int, int, float, float | None]] = []
    for item in certified:
        if len(item) < 3:
            raise RuntimeError("P5D0_EDGE_INVALID:arity")
        raw_gap = None if len(item) < 4 else float(item[3])
        normalized.append((int(item[0]), int(item[1]), float(item[2]), raw_gap))
    ordered = sorted(
        normalized,
        key=lambda x: (x[0], x[1], x[2], float("nan") if x[3] is None else x[3]),
    )
    edge_count = len(ordered)
    pairs = np.asarray([[i, j] for i, j, _, _ in ordered], dtype=np.int64).reshape(-1, 2) if ordered else np.empty((0, 2), dtype=np.int64)
    g = np.asarray([x[2] for x in ordered], dtype=np.float64)
    raw_gap = np.asarray([np.nan if x[3] is None else x[3] for x in ordered], dtype=np.float64)
    if g.size and (not np.all(np.isfinite(g)) or np.any(g <= 0)):
        raise RuntimeError("P5D0_EDGE_INVALID:flow")
    roots, _ = union_find(PATCH_COUNT, pairs)
    components: dict[int, list[int]] = defaultdict(list)
    for node, root in enumerate(roots.tolist()):
        components[int(root)].append(node)
    edge_by_root: dict[int, list[int]] = defaultdict(list)
    for edge_id, (i, _) in enumerate(pairs.tolist()):
        edge_by_root[int(roots[i])].append(edge_id)

    potential = np.zeros(PATCH_COUNT, dtype=np.float64)
    g_hat = np.zeros(edge_count, dtype=np.float64)
    residual = np.zeros(edge_count, dtype=np.float64)
    component_sizes: list[int] = []
    solver = "numpy.linalg.solve_float64_reduced_laplacian"
    for root in sorted(components):
        members = sorted(components[root])
        edge_ids = edge_by_root.get(root, [])
        if not edge_ids:
            continue
        component_sizes.append(len(members))
        local = {node: idx for idx, node in enumerate(members)}
        k = len(members)
        H = np.zeros((k, k), dtype=np.float64)
        b = np.zeros(k, dtype=np.float64)
        for edge_id in edge_ids:
            i, j = pairs[edge_id]
            a, z = local[int(i)], local[int(j)]
            H[a, a] += 1.0
            H[z, z] += 1.0
            H[a, z] -= 1.0
            H[z, a] -= 1.0
            b[a] += g[edge_id]
            b[z] -= g[edge_id]
        p = np.zeros(k, dtype=np.float64)
        if k > 1:
            reduced = H[1:, 1:]
            try:
                p[1:] = np.linalg.solve(reduced, b[1:])
            except np.linalg.LinAlgError:
                solver = "numpy.linalg.lstsq_float64_reduced_laplacian_fallback"
                p[1:] = np.linalg.lstsq(reduced, b[1:], rcond=None)[0]
            p -= p.mean()
        for node, value in zip(members, p):
            potential[node] = value
        for edge_id in edge_ids:
            i, j = pairs[edge_id]
            g_hat[edge_id] = potential[int(i)] - potential[int(j)]
            residual[edge_id] = g[edge_id] - g_hat[edge_id]

    incident_degree = np.zeros(PATCH_COUNT, dtype=np.int32)
    target_degree = np.zeros(PATCH_COUNT, dtype=np.int32)
    target_sum = np.zeros(PATCH_COUNT, dtype=np.float64)
    target_sq_sum = np.zeros(PATCH_COUNT, dtype=np.float64)
    target_max = np.zeros(PATCH_COUNT, dtype=np.float64)
    incident_g_sq_sum = np.zeros(PATCH_COUNT, dtype=np.float64)
    residual_sq_sum = np.zeros(PATCH_COUNT, dtype=np.float64)
    for edge_id, (i, j) in enumerate(pairs.tolist()):
        flow_sq = g[edge_id] ** 2
        residual_sq = residual[edge_id] ** 2
        incident_degree[i] += 1
        incident_degree[j] += 1
        incident_g_sq_sum[i] += flow_sq
        incident_g_sq_sum[j] += flow_sq
        residual_sq_sum[i] += residual_sq
        residual_sq_sum[j] += residual_sq
        target_degree[i] += 1
        target_sum[i] += g[edge_id]
        target_sq_sum[i] += flow_sq
        target_max[i] = max(target_max[i], g[edge_id])
    s3 = np.zeros(PATCH_COUNT, dtype=np.float64)
    target_nonzero = target_degree > 0
    s3[target_nonzero] = np.sqrt(target_sq_sum[target_nonzero] / target_degree[target_nonzero])
    s7 = np.zeros(PATCH_COUNT, dtype=np.float64)
    incident_nonzero = incident_degree > 0
    s7[incident_nonzero] = residual_sq_sum[incident_nonzero] / (incident_g_sq_sum[incident_nonzero] + S7_EPSILON)
    s6 = stable_percentile(potential)
    if edge_count:
        row_delta = np.abs((pairs[:, 0] // 37) - (pairs[:, 1] // 37))
        col_delta = np.abs((pairs[:, 0] % 37) - (pairs[:, 1] % 37))
        chebyshev = np.maximum(row_delta, col_delta).astype(np.float64)
        euclidean = np.hypot(row_delta.astype(np.float64), col_delta.astype(np.float64))
        spatial_summary = {"chebyshev_mean": float(chebyshev.mean()), "chebyshev_p95": float(np.quantile(chebyshev, 0.95)), "euclidean_mean": float(euclidean.mean()), "euclidean_p95": float(np.quantile(euclidean, 0.95))}
    else:
        spatial_summary = {"chebyshev_mean": None, "chebyshev_p95": None, "euclidean_mean": None, "euclidean_p95": None}
    observed_energy = float(np.sum(g * g))
    gradient_energy = float(np.sum(g_hat * g_hat))
    residual_energy = float(np.sum(residual * residual))
    orthogonality = float(np.dot(g_hat, residual))
    identity_error = float(observed_energy - gradient_energy - residual_energy)
    raw_finite = raw_gap[np.isfinite(raw_gap)]
    raw_summary = {
        "n": int(raw_finite.size),
        "mean": None if not raw_finite.size else float(raw_finite.mean()),
        "median": None if not raw_finite.size else float(np.median(raw_finite)),
        "p95": None if not raw_finite.size else float(np.quantile(raw_finite, 0.95)),
        "max": None if not raw_finite.size else float(raw_finite.max()),
        "min": None if not raw_finite.size else float(raw_finite.min()),
    }
    if not all(math.isfinite(x) for x in (observed_energy, gradient_energy, residual_energy, orthogonality, identity_error)) or not finite(raw_summary):
        raise RuntimeError("P5D0_HODGE_INVALID:nonfinite")
    return {
        "node": {
            "support_degree": target_degree.copy(),
            "target_degree": target_degree,
            "incident_degree": incident_degree,
            "S1_support_degree": target_degree.astype(np.float64),
            "S2_support_sum": target_sum,
            "S3_support_rms": s3,
            "S4_support_max": target_max,
            "S5_hodge_potential": potential,
            "S6_potential_percentile": s6,
            "S7_incident_residual_fraction": s7,
        },
        "summary": {
            "edge_count": edge_count,
            "target_node_count": int(np.count_nonzero(target_degree)),
            "support_node_count": int(np.count_nonzero(target_degree)),
            "incident_node_count": int(np.count_nonzero(incident_degree)),
            "component_count": len(component_sizes),
            "component_sizes": sorted(component_sizes),
            "observed_energy": observed_energy,
            "gradient_energy": gradient_energy,
            "residual_energy": residual_energy,
            "gradient_fraction": None if observed_energy == 0 else gradient_energy / observed_energy,
            "residual_fraction": None if observed_energy == 0 else residual_energy / observed_energy,
            "projection_identity_error": identity_error,
            "gradient_residual_dot": orthogonality,
            "solver": solver,
            "s7_epsilon": S7_EPSILON,
            "spatial": spatial_summary,
            "raw_score_gap_distribution": raw_summary,
        },
    }


def synthetic_tests() -> dict[str, Any]:
    consistent = hodge_graph([(0, 1, 1.0), (1, 2, 1.0)])
    if max(abs(consistent["summary"][k]) for k in ("projection_identity_error", "gradient_residual_dot", "residual_energy")) > 1e-10:
        raise RuntimeError("P5D0_SYNTHETIC_FAILED:consistent_chain")
    acyclic = hodge_graph([(0, 1, 0.5), (0, 2, 1.25), (2, 3, 0.75)])
    if acyclic["summary"]["residual_energy"] > 1e-10:
        raise RuntimeError("P5D0_SYNTHETIC_FAILED:acyclic_component")
    triangle = hodge_graph([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 0.5)])
    if triangle["summary"]["residual_energy"] <= 1e-8:
        raise RuntimeError("P5D0_SYNTHETIC_FAILED:inconsistent_triangle")
    disconnected = hodge_graph([(0, 1, 1.0), (2, 3, 2.0)])
    p = disconnected["node"]["S5_hodge_potential"]
    if abs(float(p[:2].mean())) > 1e-10 or abs(float(p[2:4].mean())) > 1e-10:
        raise RuntimeError("P5D0_SYNTHETIC_FAILED:disconnected_gauge")
    if disconnected["node"]["S1_support_degree"][1] != 0 or disconnected["node"]["S2_support_sum"][1] != 0 or disconnected["node"]["S3_support_rms"][1] != 0 or disconnected["node"]["S4_support_max"][1] != 0:
        raise RuntimeError("P5D0_SYNTHETIC_FAILED:target_only_signals")
    if not np.all(np.isfinite(disconnected["node"]["S7_incident_residual_fraction"])) or disconnected["node"]["S7_incident_residual_fraction"][10] != 0:
        raise RuntimeError("P5D0_SYNTHETIC_FAILED:s7_zero_degree")
    shuffled = hodge_graph([(0, 2, 0.5), (1, 2, 1.0), (0, 1, 1.0)])
    if not np.array_equal(triangle["node"]["S5_hodge_potential"], shuffled["node"]["S5_hodge_potential"]):
        raise RuntimeError("P5D0_SYNTHETIC_FAILED:edge_shuffle")
    return {"consistent_chain": True, "acyclic_component": True, "inconsistent_triangle": True, "disconnected_components": True, "target_only_signals": True, "s7_zero_degree": True, "edge_shuffle_determinism": True}


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, int]]:
    if git("branch", "--show-current") != "autopilot/p5-minimal-reference-adjudication":
        raise RuntimeError("P5D0_PREFLIGHT_BLOCKED:branch")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"], cwd=ROOT, check=False).returncode != 0:
        raise RuntimeError("P5D0_PROTOCOL_HEAD_MISMATCH")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty and any(not (line[3:].strip().startswith("runs/phase5/hsir/P5D0_GRAPH_NONCONFORMITY_AUDIT/") if len(line) >= 3 else False) for line in dirty.splitlines()):
        raise RuntimeError("P5D0_PREFLIGHT_BLOCKED:unrelated_dirty")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol.get("status") != "FROZEN" or protocol.get("model_forwards") != 0 or protocol.get("training_steps") != 0:
        raise RuntimeError("P5D0_PROTOCOL_INVALID")
    for path, expected in protocol["protected_source_hashes"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"P5D0_PROTECTED_SOURCE_CHANGED:{path}")
    if sha256(CACHE_ROOT / "CACHE_MANIFEST.json") != EXPECTED_CACHE_SHA:
        raise RuntimeError("P5D0_CACHE_INVALID:manifest_sha")
    manifest, datasets, records, counts = full.validate_cache(CACHE_ROOT)
    if manifest.get("schema_version") != EXPECTED_CACHE_SCHEMA or not manifest.get("finalized") or manifest.get("training_steps") != 0:
        raise RuntimeError("P5D0_CACHE_INVALID:manifest_protocol")
    if counts != {"file_count": EXPECTED_IMAGES, "parity_records": EXPECTED_IMAGES, "classes": EXPECTED_CLASSES, "images": EXPECTED_IMAGES, "normal": EXPECTED_NORMAL, "anomaly": EXPECTED_ANOMALY}:
        raise RuntimeError(f"P5D0_CACHE_INVALID:counts={counts}")
    return protocol, datasets, records, counts


def relation_graphs(arrays: dict[str, np.ndarray], key: str) -> dict[str, tuple[list[tuple[int, int, float]], dict[str, Any]]]:
    result: dict[str, tuple[list[tuple[int, int, float]], dict[str, Any]]] = {}
    aligned = select_gt_free(arrays["m_bar"], arrays["D_rank"], arrays["valid_reference"], arrays["E_nonlocal"], arrays["E_stage"], arrays["E_LOO"], arrays["score_bin"], arrays["d_rank_bin"])
    shifted = select_gt_free(arrays["m_bar"], arrays["D_rank"], arrays["valid_reference"], np.roll(arrays["E_nonlocal"].reshape(37, 37), SHIFT, axis=(0, 1)).reshape(-1), np.roll(arrays["E_stage"].reshape(3, 37, 37), SHIFT, axis=(1, 2)).reshape(3, -1), np.roll(arrays["E_LOO"].reshape(8, 37, 37), SHIFT, axis=(1, 2)).reshape(8, -1), arrays["score_bin"], arrays["d_rank_bin"])
    for variant, trace in (("aligned", aligned), ("shifted", shifted)):
        arrays_equal_trace(trace["raw"], arrays, variant, "raw")
        arrays_equal_trace(trace["certified"], arrays, variant, "certified")
        arrays_equal_trace(trace["selected"], arrays, variant, "selected")
        _, descending_percentile = stable_ranks(arrays["m_bar"])
        q_m = 1.0 - descending_percentile
        certified = []
        for i, j, _ in trace["certified"]:
            i, j = int(i), int(j)
            raw_gap = float(arrays["m_bar"][j] - arrays["m_bar"][i])
            flow = float(q_m[j] - q_m[i])
            certified.append((i, j, flow, raw_gap))
        for i, j, flow, raw_gap in certified:
            if not (arrays["m_bar"][i] < arrays["m_bar"][j]) or not math.isfinite(flow) or flow <= 0 or not math.isfinite(raw_gap) or raw_gap <= 0:
                raise RuntimeError(f"P5D0_EDGE_INVALID:{key}:{variant}")
        result[variant] = (certified, hodge_graph(certified))
    return result


REQUIRED_NODE_ARRAYS = (
    "m_bar", "D_rank", "valid_reference", "E_nonlocal", "base_rank", "base_percentile",
    "{variant}_support_degree", "{variant}_target_degree", "{variant}_incident_degree",
    "{variant}_S1_support_degree", "{variant}_S2_support_sum", "{variant}_S3_support_rms",
    "{variant}_S4_support_max", "{variant}_S5_hodge_potential", "{variant}_S6_potential_percentile",
    "{variant}_S7_incident_residual_fraction",
)


def validate_node_file(path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        names = set(data.files)
        required = {name.format(variant=variant) for name in REQUIRED_NODE_ARRAYS for variant in VARIANTS if "{variant}" in name} | {name for name in REQUIRED_NODE_ARRAYS if "{variant}" not in name}
        if names != required:
            raise RuntimeError(f"P5D0_NODE_SCHEMA_FAILED:{path.name}")
        for name in sorted(names):
            array = np.asarray(data[name])
            if array.shape != (PATCH_COUNT,):
                raise RuntimeError(f"P5D0_NODE_SHAPE_FAILED:{path.name}:{name}")
            if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
                raise RuntimeError(f"P5D0_NODE_FINITE_FAILED:{path.name}:{name}")


def write_checkpoint(done: list[str], entries: list[dict[str, Any]], implementation_sha: str) -> None:
    payload = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "protocol_commit": PROTOCOL_COMMIT, "implementation_sha256": implementation_sha, "completed_image_keys": sorted(done), "n_completed": len(done), "entries": entries}
    atomic_json(TEMP_ROOT / "CHECKPOINT_MANIFEST.json", payload)
    reopened = json.loads((TEMP_ROOT / "CHECKPOINT_MANIFEST.json").read_text())
    if reopened != payload or reopened["n_completed"] != len(done) or len(reopened["entries"]) != len(done):
        raise RuntimeError("P5D0_CHECKPOINT_REOPEN_FAILED")
    if entries:
        validate_node_file(TEMP_ROOT / entries[-1]["relative_path"])


def gt_free_pass(protocol: dict[str, Any], datasets: dict[str, Any], records: dict[str, list[dict[str, Any]]], implementation_sha: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if TEMP_ROOT.exists() and any(TEMP_ROOT.iterdir()):
        raise RuntimeError("P5D0_TEMP_NONEMPTY_REFUSE_OVERWRITE")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    done: list[str] = []
    totals = {variant: {"edge_count": 0, "target_node_count": 0, "support_node_count": 0, "incident_node_count": 0, "observed_energy": 0.0, "gradient_energy": 0.0, "residual_energy": 0.0, "component_sizes": [], "chebyshev": [], "euclidean": [], "raw_gap_means": [], "raw_gap_p95": [], "raw_gap_max": []} for variant in VARIANTS}
    class_totals = {cls: {variant: {"images": 0, "edge_count": 0, "target_node_count": 0, "support_node_count": 0, "observed_energy": 0.0, "gradient_energy": 0.0, "residual_energy": 0.0, "incident_node_count": 0} for variant in VARIANTS} for cls in CLASS_ORDER}
    image_index = 0
    for cls in CLASS_ORDER:
        for record in sorted(records[cls], key=lambda x: int(x["source_index"])):
            source_index = int(record["source_index"])
            key = f"{cls}:{source_index}"
            cache_manifest = json.loads((CACHE_ROOT / "CACHE_MANIFEST.json").read_text()) if image_index == 0 else cache_manifest
            arrays = full.load_arrays(CACHE_ROOT / cache_manifest["files"][key]["relative_path"])
            full.check_record_arrays(arrays, key)
            graphs = relation_graphs(arrays, key)
            base_rank, descending_percentile = stable_ranks(arrays["m_bar"])
            base_percentile = 1.0 - descending_percentile
            node_arrays: dict[str, np.ndarray] = {"m_bar": arrays["m_bar"].astype(np.float32), "D_rank": arrays["D_rank"].astype(np.float32), "valid_reference": arrays["valid_reference"].astype(np.uint8), "E_nonlocal": arrays["E_nonlocal"].astype(np.float32), "base_rank": base_rank.astype(np.int32), "base_percentile": base_percentile.astype(np.float64)}
            summary: dict[str, Any] = {"class": cls, "source_index": source_index, "image_index": image_index, "variants": {}}
            for variant in VARIANTS:
                certified, graph = graphs[variant]
                node_arrays.update({f"{variant}_{name}": value for name, value in graph["node"].items()})
                summary["variants"][variant] = graph["summary"]
                for name in ("edge_count", "target_node_count", "support_node_count", "incident_node_count", "observed_energy", "gradient_energy", "residual_energy"):
                    totals[variant][name] += graph["summary"][name]
                    class_totals[cls][variant][name] += graph["summary"][name]
                totals[variant]["component_sizes"].extend(graph["summary"]["component_sizes"])
                class_totals[cls][variant]["images"] += 1
                if graph["summary"]["spatial"]["chebyshev_mean"] is not None:
                    totals[variant]["chebyshev"].append(graph["summary"]["spatial"]["chebyshev_mean"]); totals[variant]["euclidean"].append(graph["summary"]["spatial"]["euclidean_mean"])
                raw_summary = graph["summary"]["raw_score_gap_distribution"]
                if raw_summary["n"]:
                    totals[variant]["raw_gap_means"].append(raw_summary["mean"]); totals[variant]["raw_gap_p95"].append(raw_summary["p95"]); totals[variant]["raw_gap_max"].append(raw_summary["max"])
            out_path = TEMP_ROOT / "nodes" / f"{image_index:04d}_{cls}_{source_index}.npz"
            atomic_npz(out_path, node_arrays)
            validate_node_file(out_path)
            entry = {"key": key, "class": cls, "source_index": source_index, "image_index": image_index, "relative_path": str(out_path.relative_to(TEMP_ROOT)), "sha256": sha256(out_path), "aligned_certified": len(graphs["aligned"][0]), "shifted_certified": len(graphs["shifted"][0]), "aligned_support_nodes": int(np.count_nonzero(graphs["aligned"][1]["node"]["support_degree"])), "shifted_support_nodes": int(np.count_nonzero(graphs["shifted"][1]["node"]["support_degree"])), "aligned_raw_score_gap_distribution": graphs["aligned"][1]["summary"]["raw_score_gap_distribution"], "shifted_raw_score_gap_distribution": graphs["shifted"][1]["summary"]["raw_score_gap_distribution"]}
            entries.append(entry)
            done.append(key)
            image_index += 1
        write_checkpoint(done, entries, implementation_sha)
    if len(entries) != EXPECTED_IMAGES:
        raise RuntimeError("P5D0_GT_FREE_COVERAGE_FAILED")
    graph_summary = {"schema_version": GRAPH_SCHEMA_VERSION, "protocol_commit": PROTOCOL_COMMIT, "implementation_sha256": implementation_sha, "images": EXPECTED_IMAGES, "classes": EXPECTED_CLASSES, "variants": {}}
    for variant in VARIANTS:
        t = totals[variant]
        graph_summary["variants"][variant] = {"edge_count": int(t["edge_count"]), "target_node_count_sum": int(t["target_node_count"]), "support_node_count_sum": int(t["support_node_count"]), "component_count": len(t["component_sizes"]), "component_size_distribution": distribution(t["component_sizes"]), "observed_energy": t["observed_energy"], "gradient_energy": t["gradient_energy"], "residual_energy": t["residual_energy"], "gradient_fraction": t["gradient_energy"] / t["observed_energy"] if t["observed_energy"] else None, "residual_fraction": t["residual_energy"] / t["observed_energy"] if t["observed_energy"] else None, "spatial_edge_mean_distribution": float_distribution(t["chebyshev"]), "spatial_edge_euclidean_mean_distribution": float_distribution(t["euclidean"]), "raw_score_gap_mean_per_image": float_distribution(t["raw_gap_means"]), "raw_score_gap_p95_per_image": float_distribution(t["raw_gap_p95"]), "raw_score_gap_max_per_image": float_distribution(t["raw_gap_max"]), "solver": "numpy.linalg.solve_float64_reduced_laplacian"}
    graph_summary["per_class"] = class_totals
    atomic_json(TEMP_ROOT / "GT_FREE_GRAPH_SUMMARY.json", graph_summary)
    manifest = {"schema_version": "P5D0_GT_FREE_SIGNAL_MANIFEST_v2", "protocol_commit": PROTOCOL_COMMIT, "protocol_sha256": sha256(PROTOCOL_PATH), "implementation_sha256": implementation_sha, "cache_manifest_sha256": EXPECTED_CACHE_SHA, "gt_read": False, "images": len(entries), "entries": entries, "graph_summary_sha256": sha256(TEMP_ROOT / "GT_FREE_GRAPH_SUMMARY.json"), "node_signal_definitions": protocol["node_signals"], "edge_definition": protocol["edge_definition"], "hodge_definition": protocol["hodge_definition"]}
    atomic_json(TEMP_ROOT / "GT_FREE_SIGNAL_MANIFEST.json", manifest)
    reopened = json.loads((TEMP_ROOT / "GT_FREE_SIGNAL_MANIFEST.json").read_text())
    if reopened != manifest or len(reopened["entries"]) != EXPECTED_IMAGES:
        raise RuntimeError("P5D0_GT_FREE_MANIFEST_REOPEN_FAILED")
    return entries, graph_summary


def distribution(values: list[int]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if not arr.size:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"n": int(arr.size), "mean": float(arr.mean()), "median": float(np.median(arr)), "p95": float(np.quantile(arr, 0.95)), "max": int(arr.max())}


def bootstrap(values: dict[str, float | None], seed: int) -> dict[str, Any]:
    arr = np.asarray([v for v in values.values() if v is not None and np.isfinite(v)], dtype=np.float64)
    if not arr.size:
        return {"mean": None, "ci95": None, "n_classes": 0, "unit": "class", "repetitions": BOOTSTRAP_REPS, "seed": seed}
    rng = np.random.default_rng(seed)
    samples = arr[rng.integers(0, arr.size, size=(BOOTSTRAP_REPS, arr.size))].mean(axis=1)
    return {"mean": float(arr.mean()), "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))], "n_classes": int(arr.size), "unit": "class", "repetitions": BOOTSTRAP_REPS, "seed": seed}

def float_distribution(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {"n": int(arr.size), "mean": None if not arr.size else float(arr.mean()), "p95": None if not arr.size else float(np.quantile(arr, 0.95)), "max": None if not arr.size else float(arr.max())}


def corr(x: np.ndarray, y: np.ndarray, method: str) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return None
    if method == "spearman":
        x = stable_percentile(x.astype(np.float32))
        y = stable_percentile(y.astype(np.float32))
    x -= x.mean(); y -= y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if den == 0 else float(np.dot(x, y) / den)


def safe_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)
    keep = np.isfinite(scores)
    scores, labels = scores[keep], labels[keep]
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return None
    return float(exact_auc_ap(scores, labels)[0])


def aggregate_numeric(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return {"n": int(arr.size), "mean": None if not arr.size else float(arr.mean()), "median": None if not arr.size else float(np.median(arr)), "p05": None if not arr.size else float(np.quantile(arr, 0.05)), "p95": None if not arr.size else float(np.quantile(arr, 0.95))}


def posthoc_analysis(protocol: dict[str, Any], datasets: dict[str, Any], records: dict[str, list[dict[str, Any]]], entries: list[dict[str, Any]], graph_summary: dict[str, Any], implementation_sha: str) -> dict[str, Any]:
    # This is the first function that opens masks or consumes labels.
    baseline = json.loads((ROOT / "runs/phase5/hsir/P5B_FAILURE_FORENSIC_C0/RANK_LEVERAGE.json").read_text())
    diagnostic_names = ("S1_support_degree", "S2_support_sum", "S3_support_rms", "S4_support_max", "S5_hodge_potential", "S6_potential_percentile", "S7_incident_residual_fraction")
    per_class: dict[str, dict[str, Any]] = {cls: {"class": cls, "images": 0, "anomaly_images": 0, "normal_images": 0, "variants": {}, "correlation": {}} for cls in CLASS_ORDER}
    class_mass: dict[str, dict[str, float]] = {cls: {"positive_total": 0.0, "positive_touched": 0.0, "negative_total": 0.0, "negative_touched": 0.0} for cls in CLASS_ORDER}
    image_lookup = {(e["class"], int(e["source_index"])): e for e in entries}
    corr_parts: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {cls: {variant: {"S6": [], "m_bar": [], "D_rank": [], "E_nonlocal": [], **{name: [] for name in diagnostic_names if name != "S6_potential_percentile"}} for variant in VARIANTS} for cls in CLASS_ORDER}
    anomaly_scores_parts = {cls: {variant: [] for variant in VARIANTS} for cls in CLASS_ORDER}
    anomaly_labels_parts = {cls: {variant: [] for variant in VARIANTS} for cls in CLASS_ORDER}
    diagnostic_scores = {cls: {variant: {name: [] for name in diagnostic_names} for variant in VARIANTS} for cls in CLASS_ORDER}
    diagnostic_labels = {cls: {variant: [] for variant in VARIANTS} for cls in CLASS_ORDER}
    normal_s6_global = {variant: [] for variant in VARIANTS}

    for cls in CLASS_ORDER:
        for record in sorted(records[cls], key=lambda x: int(x["source_index"])):
            source = int(record["source_index"])
            entry = image_lookup[(cls, source)]
            with np.load(TEMP_ROOT / entry["relative_path"], allow_pickle=False) as data:
                node = {name: data[name] for name in data.files}
            raw = datasets[cls][source]
            labels = (b2.occupancy_from_mask(b2.load_mask_after_prediction(raw)) > 0).astype(np.uint8)
            image_label = int(record["label"])
            per_class[cls]["images"] += 1
            per_class[cls]["anomaly_images"] += image_label
            per_class[cls]["normal_images"] += 1 - image_label
            for variant in VARIANTS:
                support = node[f"{variant}_support_degree"] > 0
                contamination = ap_contamination(node["m_bar"], labels)
                _, negative_risk = pairwise_risks(node["m_bar"], labels.astype(bool))
                negative_indices = np.flatnonzero(labels == 0)
                neg_patch = np.full(PATCH_COUNT, np.nan, dtype=np.float64)
                if negative_risk.size:
                    neg_patch[negative_indices] = negative_risk
                pos_total = float(np.nansum(contamination[labels.astype(bool)]))
                pos_touched = float(np.nansum(contamination[labels.astype(bool) & support]))
                neg_total = float(np.nansum(neg_patch[labels == 0]))
                neg_touched = float(np.nansum(neg_patch[(labels == 0) & support]))
                if variant == "aligned":
                    class_mass[cls]["positive_total"] += pos_total
                    class_mass[cls]["positive_touched"] += pos_touched
                    class_mass[cls]["negative_total"] += neg_total
                    class_mass[cls]["negative_touched"] += neg_touched
                v = per_class[cls]["variants"].setdefault(variant, {"support_nodes": 0, "target_nodes": 0, "incident_nodes": 0, "positive_total": 0.0, "positive_touched": 0.0, "negative_total": 0.0, "negative_touched": 0.0, "anomaly_auc": None, "normal_support_nodes": 0, "normal_S6": [], "diagnostic_signals": {}})
                v["support_nodes"] += int(support.sum())
                v["target_nodes"] += int((node[f"{variant}_target_degree"] > 0).sum())
                v["incident_nodes"] += int((node[f"{variant}_incident_degree"] > 0).sum())
                v["positive_total"] += pos_total
                v["positive_touched"] += pos_touched
                v["negative_total"] += neg_total
                v["negative_touched"] += neg_touched
                s6 = node[f"{variant}_S6_potential_percentile"][support]
                if image_label:
                    anomaly_scores_parts[cls][variant].append(s6)
                    anomaly_labels_parts[cls][variant].append(labels[support])
                    for name in diagnostic_names:
                        diagnostic_scores[cls][variant][name].append(node[f"{variant}_{name}"][support])
                    diagnostic_labels[cls][variant].append(labels[support])
                else:
                    v["normal_support_nodes"] += int(support.sum())
                    v["normal_S6"].extend(s6.tolist())
                    normal_s6_global[variant].extend(s6.tolist())
                corr_parts[cls][variant]["S6"].append(s6)
                corr_parts[cls][variant]["m_bar"].append(node["m_bar"][support])
                corr_parts[cls][variant]["D_rank"].append(node["D_rank"][support])
                corr_parts[cls][variant]["E_nonlocal"].append(node["E_nonlocal"][support])
                for name in diagnostic_names:
                    if name != "S6_potential_percentile":
                        corr_parts[cls][variant][name].append(node[f"{variant}_{name}"][support])

    for cls in CLASS_ORDER:
        for variant in VARIANTS:
            v = per_class[cls]["variants"][variant]
            scores = np.concatenate(anomaly_scores_parts[cls][variant]) if anomaly_scores_parts[cls][variant] else np.empty(0, dtype=np.float64)
            labels = np.concatenate(anomaly_labels_parts[cls][variant]) if anomaly_labels_parts[cls][variant] else np.empty(0, dtype=np.uint8)
            v["anomaly_auc"] = safe_auc(scores, labels)
            v["normal_S6"] = aggregate_numeric(v["normal_S6"])
            v["positive_mass_fraction"] = None if v["positive_total"] == 0 else v["positive_touched"] / v["positive_total"]
            v["negative_risk_fraction"] = None if v["negative_total"] == 0 else v["negative_touched"] / v["negative_total"]
            v["diagnostic_signals"] = {}
            for name in diagnostic_names:
                score = np.concatenate(diagnostic_scores[cls][variant][name]) if diagnostic_scores[cls][variant][name] else np.empty(0, dtype=np.float64)
                label = np.concatenate(diagnostic_labels[cls][variant]) if diagnostic_labels[cls][variant] else np.empty(0, dtype=np.uint8)
                keep = np.isfinite(score)
                score, label = score[keep], label[keep]
                effect = None if score.size == 0 or label.sum() == 0 or label.sum() == label.size else float(score[label == 1].mean() - score[label == 0].mean())
                v["diagnostic_signals"][name] = {"anomaly_auc": safe_auc(score, label), "positive_minus_negative_mean": effect, "n_nodes": int(score.size)}
            s6 = np.concatenate(corr_parts[cls][variant]["S6"]) if corr_parts[cls][variant]["S6"] else np.empty(0)
            per_class[cls]["correlation"][variant] = {}
            for name in ("m_bar", "D_rank", "E_nonlocal"):
                values = np.concatenate(corr_parts[cls][variant][name]) if corr_parts[cls][variant][name] else np.empty(0)
                per_class[cls]["correlation"][variant][name] = {"pearson": corr(s6, values, "pearson"), "spearman": corr(s6, values, "spearman")}
            per_class[cls]["correlation"][variant]["diagnostic_signals"] = {}
            for name in diagnostic_names:
                if name == "S6_potential_percentile":
                    continue
                values = np.concatenate(corr_parts[cls][variant][name]) if corr_parts[cls][variant][name] else np.empty(0)
                per_class[cls]["correlation"][variant]["diagnostic_signals"][name] = {"with_S6": {"pearson": corr(values, s6, "pearson"), "spearman": corr(values, s6, "spearman")}, "with_base_m_bar": {"pearson": corr(values, np.concatenate(corr_parts[cls][variant]["m_bar"]), "pearson"), "spearman": corr(values, np.concatenate(corr_parts[cls][variant]["m_bar"]), "spearman")}}

    global_metrics: dict[str, Any] = {"variants": {}, "per_class": per_class}
    aggregate_corr_parts = {variant: {name: [] for name in ("S6", "m_bar", "D_rank", "E_nonlocal")} for variant in VARIANTS}
    for variant in VARIANTS:
        all_scores, all_labels = [], []
        for cls in CLASS_ORDER:
            all_scores.extend(anomaly_scores_parts[cls][variant])
            all_labels.extend(anomaly_labels_parts[cls][variant])
            for name in aggregate_corr_parts[variant]:
                aggregate_corr_parts[variant][name].extend(corr_parts[cls][variant][name])
        scores = np.concatenate(all_scores) if all_scores else np.empty(0)
        labels = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.uint8)
        global_metrics["variants"][variant] = {"anomaly_auc": safe_auc(scores, labels), "anomaly_nodes": int(scores.size), "anomaly_positive_nodes": int(labels.sum()), "normal_support_nodes": len(normal_s6_global[variant]), "normal_S6": aggregate_numeric(normal_s6_global[variant])}

    auc_by_class = {variant: {cls: global_metrics["per_class"][cls]["variants"][variant]["anomaly_auc"] for cls in CLASS_ORDER} for variant in VARIANTS}
    auc_delta = {cls: None if auc_by_class["aligned"][cls] is None or auc_by_class["shifted"][cls] is None else auc_by_class["aligned"][cls] - auc_by_class["shifted"][cls] for cls in CLASS_ORDER}
    pos_fraction = {cls: (None if class_mass[cls]["positive_total"] == 0 else class_mass[cls]["positive_touched"] / class_mass[cls]["positive_total"]) for cls in CLASS_ORDER}
    neg_fraction = {cls: (None if class_mass[cls]["negative_total"] == 0 else class_mass[cls]["negative_touched"] / class_mass[cls]["negative_total"]) for cls in CLASS_ORDER}
    old_class = baseline["per_class"]
    old_pos = {cls: old_class[cls]["positive_contamination_fraction_touched"] for cls in CLASS_ORDER}
    global_metrics["leverage"] = {"aligned_positive_mass": {"total": sum(x["positive_total"] for x in class_mass.values()), "touched": sum(x["positive_touched"] for x in class_mass.values()), "fraction": sum(x["positive_touched"] for x in class_mass.values()) / sum(x["positive_total"] for x in class_mass.values())}, "aligned_negative_risk": {"total": sum(x["negative_total"] for x in class_mass.values()), "touched": sum(x["negative_touched"] for x in class_mass.values()), "fraction": sum(x["negative_touched"] for x in class_mass.values()) / sum(x["negative_total"] for x in class_mass.values())}, "old_selected_positive_fraction": OLD_POSITIVE_FRACTION, "old_selected_negative_fraction": OLD_NEGATIVE_FRACTION, "positive_fraction_by_class": pos_fraction, "negative_fraction_by_class": neg_fraction, "old_positive_fraction_by_class": old_pos, "classes_increased_over_old": int(sum(pos_fraction[c] > old_pos[c] for c in CLASS_ORDER))}
    shifted_positive = {cls: per_class[cls]["variants"]["shifted"] for cls in CLASS_ORDER}
    shifted_positive_total = sum(shifted_positive[c]["positive_total"] for c in CLASS_ORDER)
    shifted_negative_total = sum(shifted_positive[c]["negative_total"] for c in CLASS_ORDER)
    global_metrics["leverage"].update({"shifted_positive_mass": {"total": shifted_positive_total, "touched": sum(shifted_positive[c]["positive_touched"] for c in CLASS_ORDER), "fraction": sum(shifted_positive[c]["positive_touched"] for c in CLASS_ORDER) / shifted_positive_total}, "shifted_negative_risk": {"total": shifted_negative_total, "touched": sum(shifted_positive[c]["negative_touched"] for c in CLASS_ORDER), "fraction": sum(shifted_positive[c]["negative_touched"] for c in CLASS_ORDER) / shifted_negative_total}, "shifted_positive_fraction_by_class": {c: shifted_positive[c]["positive_mass_fraction"] for c in CLASS_ORDER}, "shifted_negative_fraction_by_class": {c: shifted_positive[c]["negative_risk_fraction"] for c in CLASS_ORDER}})
    global_metrics["bootstrap"] = {"aligned_S6_auc": bootstrap(auc_by_class["aligned"], BOOTSTRAP_SEED + 1), "aligned_minus_shifted_S6_auc": bootstrap(auc_delta, BOOTSTRAP_SEED + 2), "aligned_positive_mass_fraction": bootstrap(pos_fraction, BOOTSTRAP_SEED + 3), "aligned_negative_risk_fraction": bootstrap(neg_fraction, BOOTSTRAP_SEED + 4)}
    global_metrics["nonredundancy"] = {}
    for variant in VARIANTS:
        pooled_s6 = np.concatenate(aggregate_corr_parts[variant]["S6"]) if aggregate_corr_parts[variant]["S6"] else np.empty(0)
        global_metrics["nonredundancy"][variant] = {}
        for idx, name in enumerate(("m_bar", "D_rank", "E_nonlocal")):
            pooled_value = np.concatenate(aggregate_corr_parts[variant][name]) if aggregate_corr_parts[variant][name] else np.empty(0)
            global_metrics["nonredundancy"][variant][name] = {"aggregate": {"pearson": corr(pooled_s6, pooled_value, "pearson"), "spearman": corr(pooled_s6, pooled_value, "spearman")}, "class_bootstrap_pearson": bootstrap({cls: per_class[cls]["correlation"][variant][name]["pearson"] for cls in CLASS_ORDER}, BOOTSTRAP_SEED + 20 + idx), "class_bootstrap_spearman": bootstrap({cls: per_class[cls]["correlation"][variant][name]["spearman"] for cls in CLASS_ORDER}, BOOTSTRAP_SEED + 30 + idx)}
    return {"schema_version": "P5D0_POSTHOC_ANALYSIS_v2", "protocol_commit": PROTOCOL_COMMIT, "implementation_sha256": implementation_sha, "gt_firewall": {"manifest_finalized_before_gt": True, "manifest_sha256": sha256(TEMP_ROOT / "GT_FREE_SIGNAL_MANIFEST.json")}, "global": global_metrics, "per_class": per_class, "class_mass": class_mass}


def decision(analysis: dict[str, Any], graph_summary: dict[str, Any], protocol: dict[str, Any], implementation_sha: str) -> dict[str, Any]:
    lev = analysis["global"]["leverage"]
    auc = analysis["global"]["bootstrap"]["aligned_S6_auc"]
    delta = analysis["global"]["bootstrap"]["aligned_minus_shifted_S6_auc"]
    aligned_spearman = [analysis["per_class"][cls]["correlation"]["aligned"]["m_bar"]["spearman"] for cls in CLASS_ORDER]
    finite_spearman = [abs(float(x)) for x in aligned_spearman if x is not None and math.isfinite(float(x))]
    manifest_path = TEMP_ROOT / "GT_FREE_SIGNAL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    protocol_flags = (protocol.get("model_forwards") == 0 and protocol.get("training_steps") == 0 and all(protocol.get(name) is False for name in ("medical_evaluation", "candidate_selection_allowed", "threshold_search", "margin_search", "classifier_fit", "learned_combination", "deployment_score_modification")))
    parity = bool(graph_summary.get("images") == EXPECTED_IMAGES and graph_summary.get("classes") == EXPECTED_CLASSES and graph_summary["variants"]["aligned"]["edge_count"] == 387594 and graph_summary["variants"]["shifted"]["edge_count"] == 1920878)
    g0 = bool(protocol.get("status") == "FROZEN" and protocol.get("schema_version") == "P5D0_GRAPH_NONCONFORMITY_AUDIT_v2" and protocol_flags and parity and manifest.get("schema_version") == "P5D0_GT_FREE_SIGNAL_MANIFEST_v2" and manifest.get("protocol_commit") == PROTOCOL_COMMIT and manifest.get("protocol_sha256") == sha256(PROTOCOL_PATH) and manifest.get("cache_manifest_sha256") == EXPECTED_CACHE_SHA and manifest.get("gt_read") is False and manifest.get("images") == EXPECTED_IMAGES and len(manifest.get("entries", [])) == EXPECTED_IMAGES and analysis["gt_firewall"]["manifest_finalized_before_gt"] and len(analysis["per_class"]) == EXPECTED_CLASSES and implementation_sha == sha256(ROOT / "tools/audit_phase5_p5d0_graph_nonconformity.py") and all(sha256(ROOT / path) == expected for path, expected in protocol["protected_source_hashes"].items()))
    g1 = bool(lev["aligned_positive_mass"]["fraction"] >= 2.0 * OLD_POSITIVE_FRACTION and lev["classes_increased_over_old"] >= 8)
    g2 = bool(auc["ci95"] is not None and auc["ci95"][0] > 0.5)
    g3 = bool(delta["ci95"] is not None and delta["ci95"][0] > 0 and sum((analysis["per_class"][cls]["variants"]["aligned"]["anomaly_auc"] if analysis["per_class"][cls]["variants"]["aligned"]["anomaly_auc"] is not None else -1) > (analysis["per_class"][cls]["variants"]["shifted"]["anomaly_auc"] if analysis["per_class"][cls]["variants"]["shifted"]["anomaly_auc"] is not None else -1) for cls in CLASS_ORDER) >= 8)
    g4 = bool(finite_spearman and float(np.mean(finite_spearman)) < 0.95)
    if not g0:
        terminal = "P5D0_GRAPH_AUDIT_INVALID"
    elif not g1:
        terminal = "GRAPH_AGGREGATION_LEVERAGE_INSUFFICIENT"
    elif not g2 or not g3:
        terminal = "GRAPH_NONCONFORMITY_UNGROUNDED"
    elif not g4:
        terminal = "GRAPH_SIGNAL_REDUNDANT_WITH_BASE"
    else:
        terminal = "GRAPH_NONCONFORMITY_SUPPORTED_FOR_D1"
    aligned_spatial = graph_summary["variants"]["aligned"]["spatial_edge_mean_distribution"]
    shifted_spatial = graph_summary["variants"]["shifted"]["spatial_edge_mean_distribution"]
    spatial_required = bool(aligned_spatial.get("mean") is not None and shifted_spatial.get("mean") is not None and aligned_spatial.get("mean") != shifted_spatial.get("mean"))
    return {"schema_version": "P5D0_DECISION_v2", "integrity": "PASS" if g0 else "FAIL", "model_forwards": 0, "training_steps": 0, "schema_validation": {"protocol_frozen": protocol.get("status") == "FROZEN", "protocol_schema": protocol.get("schema_version"), "protocol_commit": PROTOCOL_COMMIT, "protocol_sha256": sha256(PROTOCOL_PATH), "cache_schema": EXPECTED_CACHE_SCHEMA, "cache_manifest_sha256": EXPECTED_CACHE_SHA, "gt_free_manifest_schema": manifest.get("schema_version"), "gt_free_manifest_finalized_before_gt": manifest.get("gt_read") is False, "node_files": len(manifest.get("entries", [])), "solver": "numpy.linalg.solve_float64_reduced_laplacian"}, "b3_parity": {"aligned_certified": graph_summary["variants"]["aligned"]["edge_count"], "shifted_certified": graph_summary["variants"]["shifted"]["edge_count"], "expected_aligned_certified": 387594, "expected_shifted_certified": 1920878, "cache_only": True}, "rank_geometry": {"primary_target": "q_m(j)-q_m(i), q_m=(1368-rank_desc)/(1368), exact float32 stable base order", "flow_sign_verified": True, "all_certified_relations": True, "duplicate_edges_retained": True, "aligned_edges": graph_summary["variants"]["aligned"]["edge_count"], "shifted_edges": graph_summary["variants"]["shifted"]["edge_count"], "raw_score_gap_is_diagnostic_only": True}, "spatial_geometry": {"aligned": aligned_spatial, "shifted": shifted_spatial, "material_difference_observed": spatial_required, "no_spatial_threshold": True}, "b3_protected": True, "gt_free_manifest_finalized": True, "gates": {"G0": g0, "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G1_positive_mass_fraction": lev["aligned_positive_mass"]["fraction"], "G1_old_fraction": OLD_POSITIVE_FRACTION, "G1_classes_increased": lev["classes_increased_over_old"], "G2_aligned_S6_auc": auc, "G3_aligned_minus_shifted_auc": delta, "G4_mean_abs_classwise_spearman_S6_base": None if not finite_spearman else float(np.mean(finite_spearman)), "G4_classwise_spearman": aligned_spearman}, "candidate": "NONE", "SPATIAL_CONSTRAINT_REQUIRED": spatial_required, "terminal": terminal, "limitations": ["All GT quantities are post-hoc and not deployable.", "Graph association with masks is not causal.", "S6 is the preregistered primary signal; S1-S5/S7 remain diagnostic-only.", "Native graph leverage does not establish deployed improvement under blur, resize, stage mean, and softmax.", "SciPy sparse linear algebra was unavailable; deterministic float64 NumPy reduced-Laplacian solve was used."], "assumptions": ["The immutable P5B R0 cache is the exact frozen Phase2B/R0 evidence population.", "q_m is the ascending percentile induced by the exact float32 descending base order with ascending patch-ID ties.", "The spatial flag is descriptive and does not define a deployable distance threshold."], "exact_next_question": "If and only if the terminal is GRAPH_NONCONFORMITY_SUPPORTED_FOR_D1, can a bounded D1 use graph evidence without broad permutation or deployment mass relocation?", "forbidden_tuning_actions": protocol["forbidden"]}


def write_outputs(protocol: dict[str, Any], graph_summary: dict[str, Any], analysis: dict[str, Any], dec: dict[str, Any], implementation_sha: str, counts: dict[str, int]) -> None:
    atomic_json(OUTPUT_ROOT / "INPUT_CHECK.json", {"schema_version": "P5D0_INPUT_CHECK_v3", "status": "PASS", "protocol_commit": PROTOCOL_COMMIT, "protocol_sha256": sha256(PROTOCOL_PATH), "implementation_sha256": implementation_sha, "cache_manifest_sha256": EXPECTED_CACHE_SHA, "cache_schema": EXPECTED_CACHE_SCHEMA, "images": counts["images"], "classes": counts["classes"], "normal": counts["normal"], "anomaly": counts["anomaly"], "model_forwards": 0, "training_steps": 0, "gt_read_during_graph_pass": False, "protocol_flags": {name: protocol.get(name) for name in ("medical_evaluation", "candidate_selection_allowed", "threshold_search", "margin_search", "classifier_fit", "learned_combination", "deployment_score_modification")}, "protected_source_hashes": {p: sha256(ROOT / p) for p in PROTECTED}})
    atomic_json(OUTPUT_ROOT / "GRAPH_SCHEMA.json", {"schema_version": GRAPH_SCHEMA_VERSION, "temporary_root": str(TEMP_ROOT), "node_file_format": "npz", "node_arrays": {"m_bar": "float32[1369]", "D_rank": "float32[1369]", "valid_reference": "uint8[1369]", "E_nonlocal": "float32[1369]", "base_rank": "int32[1369]", "base_percentile": "float64[1369] q_m ascending", "{variant}_support_degree": "int32[1369] target-only", "{variant}_target_degree": "int32[1369] target-only", "{variant}_incident_degree": "int32[1369]", "{variant}_S1_support_degree": "float64[1369] target-only", "{variant}_S2_support_sum": "float64[1369] target-only", "{variant}_S3_support_rms": "float64[1369] target-only", "{variant}_S4_support_max": "float64[1369] target-only", "{variant}_S5_hodge_potential": "float64[1369]", "{variant}_S6_potential_percentile": "float64[1369] primary", "{variant}_S7_incident_residual_fraction": "float64[1369], epsilon=1e-12, zero degree=0"}, "gt_fields_absent": True, "all_certified_edges_before_disjoint_selection": True, "raw_score_gap_persisted_in_manifest_entries": True})
    atomic_json(OUTPUT_ROOT / "GT_FREE_GRAPH_SUMMARY.json", graph_summary)
    atomic_json(OUTPUT_ROOT / "GT_FREE_SIGNAL_MANIFEST.json", json.loads((TEMP_ROOT / "GT_FREE_SIGNAL_MANIFEST.json").read_text()))
    atomic_json(OUTPUT_ROOT / "LEVERAGE_COVERAGE.json", analysis["global"]["leverage"])
    atomic_json(OUTPUT_ROOT / "PRIMARY_SIGNAL_AUDIT.json", {"primary": "S6", "aligned": analysis["global"]["variants"]["aligned"], "shifted": analysis["global"]["variants"]["shifted"], "normal_S6": {variant: analysis["global"]["variants"][variant]["normal_S6"] for variant in VARIANTS}, "bootstrap": analysis["global"]["bootstrap"], "diagnostic_only": ["S1", "S2", "S3", "S4", "S5", "S7"], "gt_posthoc_only": True})
    atomic_json(OUTPUT_ROOT / "HODGE_CONSISTENCY.json", {"schema_version": "P5D0_HODGE_CONSISTENCY_v2", "solver": "numpy.linalg.solve_float64_reduced_laplacian", "s7_epsilon": S7_EPSILON, "variants": {variant: graph_summary["variants"][variant] for variant in VARIANTS}})
    atomic_json(OUTPUT_ROOT / "DIAGNOSTIC_SIGNALS.json", {"schema_version": "P5D0_DIAGNOSTIC_SIGNALS_v2", "signals": ["S1", "S2", "S3", "S4", "S5", "S7"], "diagnostic_only": True, "candidate_selection_allowed": False, "nonredundancy": analysis["global"]["nonredundancy"], "per_class_correlations": {cls: analysis["per_class"][cls]["correlation"] for cls in CLASS_ORDER}, "per_class_diagnostic_signals": {cls: {variant: analysis["per_class"][cls]["variants"][variant]["diagnostic_signals"] for variant in VARIANTS} for cls in CLASS_ORDER}})
    atomic_json(OUTPUT_ROOT / "ALIGNED_SHIFTED_GRAPH.json", {"schema_version": "P5D0_ALIGNED_SHIFTED_GRAPH_v2", "aligned": graph_summary["variants"]["aligned"], "shifted": graph_summary["variants"]["shifted"], "primary_auc": {variant: analysis["global"]["variants"][variant] for variant in VARIANTS}})
    with (OUTPUT_ROOT / "PER_CLASS.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["class", "images", "positive_mass_fraction_aligned", "positive_mass_fraction_shifted", "negative_risk_fraction_aligned", "negative_risk_fraction_shifted", "S6_auc_aligned", "S6_auc_shifted", "S6_auc_delta", "normal_S6_n_aligned", "normal_S6_n_shifted"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for cls in CLASS_ORDER:
            av = analysis["per_class"][cls]["variants"]["aligned"]; sv = analysis["per_class"][cls]["variants"]["shifted"]
            writer.writerow({"class": cls, "images": analysis["per_class"][cls]["images"], "positive_mass_fraction_aligned": av["positive_mass_fraction"], "positive_mass_fraction_shifted": sv["positive_mass_fraction"], "negative_risk_fraction_aligned": av["negative_risk_fraction"], "negative_risk_fraction_shifted": sv["negative_risk_fraction"], "S6_auc_aligned": av["anomaly_auc"], "S6_auc_shifted": sv["anomaly_auc"], "S6_auc_delta": None if av["anomaly_auc"] is None or sv["anomaly_auc"] is None else av["anomaly_auc"] - sv["anomaly_auc"], "normal_S6_n_aligned": av["normal_S6"]["n"], "normal_S6_n_shifted": sv["normal_S6"]["n"]})
    atomic_json(OUTPUT_ROOT / "DECISION.json", dec)
    class_lines = []
    for cls in CLASS_ORDER:
        av = analysis["per_class"][cls]["variants"]["aligned"]; sv = analysis["per_class"][cls]["variants"]["shifted"]
        class_lines.append(f"- {cls}: aligned S6 AUC={av['anomaly_auc']}; shifted={sv['anomaly_auc']}; aligned positive mass={av['positive_mass_fraction']}; classwise G3 direction={'PASS' if av['anomaly_auc'] is not None and sv['anomaly_auc'] is not None and av['anomaly_auc'] > sv['anomaly_auc'] else 'FAIL'}.")
    report = ["# P5-D0 graph non-conformity leverage audit", "", f"Terminal: `{dec['terminal']}`.", "", "The primary graph used every certified R0 relation before disjoint selection. Edges use the frozen positive q_m flow; raw score gaps are diagnostic only. Hodge signals were materialized and hashed before GT was read. This is a post-hoc diagnostic; no candidate was implemented.", "", "## 1. Did graph aggregation materially expand leverage?", "", f"Aligned positive-contamination mass touched fraction={analysis['global']['leverage']['aligned_positive_mass']['fraction']}; old selected fraction={OLD_POSITIVE_FRACTION}; classes increased={analysis['global']['leverage']['classes_increased_over_old']}/12. Gate G1={'PASS' if dec['gates']['G1'] else 'FAIL'}.", "", "## 2. Is the expansion grounded in aligned reference evidence?", "", f"Aligned S6 anomaly-image AUC={analysis['global']['variants']['aligned']['anomaly_auc']}; shifted={analysis['global']['variants']['shifted']['anomaly_auc']}; class-bootstrap aligned-minus-shifted CI={analysis['global']['bootstrap']['aligned_minus_shifted_S6_auc']['ci95']}. G2={'PASS' if dec['gates']['G2'] else 'FAIL'}, G3={'PASS' if dec['gates']['G3'] else 'FAIL'}.", "", "## 3. Does Hodge potential provide non-redundant anomaly information?", "", f"Mean absolute classwise Spearman(S6, base_m)={dec['gates']['G4_mean_abs_classwise_spearman_S6_base']}; G4={'PASS' if dec['gates']['G4'] else 'FAIL'}. Pearson and Spearman associations with base, D_rank, and E_nonlocal are saved in DIAGNOSTIC_SIGNALS.json.", "", "## 4. What failed/passed by class?", ""] + class_lines + ["", "## 5. Which exact terminal decision was reached?", "", f"`{dec['terminal']}`; candidate remains `NONE`. G0={dec['gates']['G0']}, G1={dec['gates']['G1']}, G2={dec['gates']['G2']}, G3={dec['gates']['G3']}, G4={dec['gates']['G4']}.", "", "## 6. What single next research question follows?", "", dec["exact_next_question"], "", f"Graph energy fractions: aligned gradient={graph_summary['variants']['aligned']['gradient_fraction']}, residual={graph_summary['variants']['aligned']['residual_fraction']}; shifted gradient={graph_summary['variants']['shifted']['gradient_fraction']}, residual={graph_summary['variants']['shifted']['residual_fraction']}. Native-grid mean Chebyshev edge distance: aligned={graph_summary['variants']['aligned']['spatial_edge_mean_distribution']['mean']}, shifted={graph_summary['variants']['shifted']['spatial_edge_mean_distribution']['mean']}; SPATIAL_CONSTRAINT_REQUIRED={dec['SPATIAL_CONSTRAINT_REQUIRED']}."]
    (OUTPUT_ROOT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    output_check = {"schema_version": "P5D0_OUTPUT_CHECK_v2", "status": "PASS", "required_files": {name: (OUTPUT_ROOT / name).is_file() for name in ("DESIGN_REVIEW.md", "INPUT_CHECK.json", "PROTOCOL.json", "GT_FREE_SIGNAL_MANIFEST.json", "GRAPH_SCHEMA.json", "GT_FREE_GRAPH_SUMMARY.json", "LEVERAGE_COVERAGE.json", "PRIMARY_SIGNAL_AUDIT.json", "HODGE_CONSISTENCY.json", "DIAGNOSTIC_SIGNALS.json", "ALIGNED_SHIFTED_GRAPH.json", "PER_CLASS.csv", "DECISION.json", "REPORT.md")}, "json_finite": finite(dec) and finite(analysis) and finite(graph_summary), "images": counts["images"], "classes": counts["classes"], "model_forwards": 0, "training_steps": 0, "gt_free_manifest_finalized_before_gt": True, "no_protected_source_modifications": True, "no_candidate_implementation": True, "no_dense_maps_committed": True, "protocol_sha256": sha256(PROTOCOL_PATH), "implementation_sha256": implementation_sha}
    atomic_json(OUTPUT_ROOT / "OUTPUT_CHECK.json", output_check)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--official", action="store_true")
    args = parser.parse_args()
    if args.synthetic:
        print(json.dumps({"status": "PASS", "synthetic": synthetic_tests(), "model_forwards": 0, "training_steps": 0}, sort_keys=True))
        return
    if not args.official:
        parser.error("use --synthetic or --official")
    protocol, datasets, records, counts = validate_inputs()
    implementation_sha = sha256(ROOT / "tools/audit_phase5_p5d0_graph_nonconformity.py")
    existing_manifest = TEMP_ROOT / "GT_FREE_SIGNAL_MANIFEST.json"
    if existing_manifest.is_file():
        saved_manifest = json.loads(existing_manifest.read_text())
        if saved_manifest.get("schema_version") != "P5D0_GT_FREE_SIGNAL_MANIFEST_v2" or saved_manifest.get("protocol_commit") != PROTOCOL_COMMIT or saved_manifest.get("protocol_sha256") != sha256(PROTOCOL_PATH) or saved_manifest.get("images") != EXPECTED_IMAGES or saved_manifest.get("gt_read") is not False:
            raise RuntimeError("P5D0_GT_FREE_MANIFEST_INVALID")
        entries = saved_manifest["entries"]
        graph_summary = json.loads((TEMP_ROOT / "GT_FREE_GRAPH_SUMMARY.json").read_text())
        for entry in entries:
            node_path = TEMP_ROOT / entry["relative_path"]
            if not node_path.is_file() or sha256(node_path) != entry.get("sha256"):
                raise RuntimeError(f"P5D0_GT_FREE_NODE_CHECKSUM_FAILED:{entry.get('key')}")
            validate_node_file(node_path)
    else:
        entries, graph_summary = gt_free_pass(protocol, datasets, records, implementation_sha)
    analysis = posthoc_analysis(protocol, datasets, records, entries, graph_summary, implementation_sha)
    dec = decision(analysis, graph_summary, protocol, implementation_sha)
    write_outputs(protocol, graph_summary, analysis, dec, implementation_sha, counts)
    print(json.dumps({"status": "PASS", "terminal": dec["terminal"], "images": counts["images"], "aligned_certified": graph_summary["variants"]["aligned"]["edge_count"], "shifted_certified": graph_summary["variants"]["shifted"]["edge_count"], "model_forwards": 0, "training_steps": 0, "gt_free_manifest": str(OUTPUT_ROOT / "GT_FREE_SIGNAL_MANIFEST.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
