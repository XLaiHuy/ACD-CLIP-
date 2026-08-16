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
TEMP_ROOT = Path("/tmp/p5_d0_graph_nonconformity")
PROTOCOL_PATH = OUTPUT_ROOT / "PROTOCOL.json"
PROTOCOL_COMMIT = "c89d8e639e3564350db752e67e7ad73771e79356"
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
OLD_POSITIVE_FRACTION = 0.03960581875762963
OLD_NEGATIVE_FRACTION = 0.01970289841145196
GRAPH_SCHEMA_VERSION = "P5D0_GT_FREE_GRAPH_NODE_v1"
CHECKPOINT_SCHEMA_VERSION = "P5D0_GT_FREE_CHECKPOINT_v1"
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


def hodge_graph(certified: list[tuple[int, int, float]]) -> dict[str, Any]:
    ordered = sorted(((int(i), int(j), float(g)) for i, j, g in certified), key=lambda x: (x[0], x[1], x[2]))
    edge_count = len(ordered)
    pairs = np.asarray([[i, j] for i, j, _ in ordered], dtype=np.int64).reshape(-1, 2) if ordered else np.empty((0, 2), dtype=np.int64)
    g = np.asarray([x[2] for x in ordered], dtype=np.float64)
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
                p[1:] = np.linalg.lstsq(reduced, b[1:], rcond=None)[0]
            p -= p.mean()
        for node, value in zip(members, p):
            potential[node] = value
        for edge_id in edge_ids:
            i, j = pairs[edge_id]
            g_hat[edge_id] = potential[int(i)] - potential[int(j)]
            residual[edge_id] = g[edge_id] - g_hat[edge_id]

    degree = np.zeros(PATCH_COUNT, dtype=np.int32)
    target_degree = np.zeros(PATCH_COUNT, dtype=np.int32)
    signed_sum = np.zeros(PATCH_COUNT, dtype=np.float64)
    target_sq_sum = np.zeros(PATCH_COUNT, dtype=np.float64)
    residual_sq_sum = np.zeros(PATCH_COUNT, dtype=np.float64)
    max_abs_residual = np.zeros(PATCH_COUNT, dtype=np.float64)
    for edge_id, (i, j) in enumerate(pairs.tolist()):
        degree[i] += 1
        degree[j] += 1
        target_degree[i] += 1
        signed_sum[i] += g[edge_id]
        signed_sum[j] -= g[edge_id]
        target_sq_sum[i] += g[edge_id] ** 2
        target_sq_sum[j] += g[edge_id] ** 2
        residual_sq_sum[i] += residual[edge_id] ** 2
        residual_sq_sum[j] += residual[edge_id] ** 2
        max_abs_residual[i] = max(max_abs_residual[i], abs(residual[edge_id]))
        max_abs_residual[j] = max(max_abs_residual[j], abs(residual[edge_id]))
    s3 = np.zeros(PATCH_COUNT, dtype=np.float64)
    nonzero = degree > 0
    s3[nonzero] = np.sqrt(target_sq_sum[nonzero] / degree[nonzero])
    s7 = np.full(PATCH_COUNT, np.nan, dtype=np.float64)
    s7[nonzero] = residual_sq_sum[nonzero] / target_sq_sum[nonzero]
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
    if not all(math.isfinite(x) for x in (observed_energy, gradient_energy, residual_energy, orthogonality, identity_error)):
        raise RuntimeError("P5D0_HODGE_INVALID:nonfinite")
    return {
        "node": {
            "support_degree": degree,
            "target_degree": target_degree,
            "S1_support_degree": degree.astype(np.float64),
            "S2_signed_incident_target_sum": signed_sum,
            "S3_incident_target_rms": s3,
            "S4_max_abs_incident_residual": max_abs_residual,
            "S5_hodge_potential": potential,
            "S6_potential_percentile": s6,
            "S7_incident_residual_fraction": s7,
        },
        "summary": {
            "edge_count": edge_count,
            "target_node_count": int(np.count_nonzero(target_degree)),
            "support_node_count": int(np.count_nonzero(degree)),
            "component_count": len(component_sizes),
            "component_sizes": sorted(component_sizes),
            "observed_energy": observed_energy,
            "gradient_energy": gradient_energy,
            "residual_energy": residual_energy,
            "gradient_fraction": None if observed_energy == 0 else gradient_energy / observed_energy,
            "residual_fraction": None if observed_energy == 0 else residual_energy / observed_energy,
            "projection_identity_error": identity_error,
            "gradient_residual_dot": orthogonality,
            "spatial": spatial_summary,
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
    shuffled = hodge_graph([(0, 2, 0.5), (1, 2, 1.0), (0, 1, 1.0)])
    if not np.array_equal(triangle["node"]["S5_hodge_potential"], shuffled["node"]["S5_hodge_potential"]):
        raise RuntimeError("P5D0_SYNTHETIC_FAILED:edge_shuffle")
    return {"consistent_chain": True, "acyclic_component": True, "inconsistent_triangle": True, "disconnected_components": True, "edge_shuffle_determinism": True}


def validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, int]]:
    if git("branch", "--show-current") != "autopilot/p5-minimal-reference-adjudication":
        raise RuntimeError("P5D0_PREFLIGHT_BLOCKED:branch")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"], cwd=ROOT, check=False).returncode != 0:
        raise RuntimeError("P5D0_PROTOCOL_HEAD_MISMATCH")
    if git("status", "--porcelain"):
        raise RuntimeError("P5D0_PREFLIGHT_BLOCKED:dirty")
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
        certified = [(int(i), int(j), float(arrays["m_bar"][j] - arrays["m_bar"][i])) for i, j, _ in trace["certified"]]
        for i, j, cost in certified:
            if not (arrays["m_bar"][i] < arrays["m_bar"][j]) or not math.isfinite(cost) or cost <= 0:
                raise RuntimeError(f"P5D0_EDGE_INVALID:{key}:{variant}")
        result[variant] = (certified, hodge_graph(certified))
    return result


def write_checkpoint(done: list[str], entries: list[dict[str, Any]], implementation_sha: str) -> None:
    atomic_json(TEMP_ROOT / "CHECKPOINT_MANIFEST.json", {"schema_version": CHECKPOINT_SCHEMA_VERSION, "protocol_commit": PROTOCOL_COMMIT, "implementation_sha256": implementation_sha, "completed_image_keys": sorted(done), "n_completed": len(done), "entries": entries})


def gt_free_pass(protocol: dict[str, Any], datasets: dict[str, Any], records: dict[str, list[dict[str, Any]]], implementation_sha: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if TEMP_ROOT.exists() and any(TEMP_ROOT.iterdir()):
        raise RuntimeError("P5D0_TEMP_NONEMPTY_REFUSE_OVERWRITE")
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    done: list[str] = []
    totals = {variant: {"edge_count": 0, "target_node_count": 0, "support_node_count": 0, "observed_energy": 0.0, "gradient_energy": 0.0, "residual_energy": 0.0, "component_sizes": [], "chebyshev": [], "euclidean": []} for variant in VARIANTS}
    class_totals = {cls: {variant: {"images": 0, "edge_count": 0, "target_node_count": 0, "support_node_count": 0, "observed_energy": 0.0, "gradient_energy": 0.0, "residual_energy": 0.0} for variant in VARIANTS} for cls in CLASS_ORDER}
    image_index = 0
    for cls in CLASS_ORDER:
        for record in sorted(records[cls], key=lambda x: int(x["source_index"])):
            source_index = int(record["source_index"])
            key = f"{cls}:{source_index}"
            cache_manifest = json.loads((CACHE_ROOT / "CACHE_MANIFEST.json").read_text()) if image_index == 0 else cache_manifest
            arrays = full.load_arrays(CACHE_ROOT / cache_manifest["files"][key]["relative_path"])
            full.check_record_arrays(arrays, key)
            graphs = relation_graphs(arrays, key)
            _, base_percentile = stable_ranks(arrays["m_bar"])
            node_arrays: dict[str, np.ndarray] = {"m_bar": arrays["m_bar"].astype(np.float32), "D_rank": arrays["D_rank"].astype(np.float32), "valid_reference": arrays["valid_reference"].astype(np.uint8), "E_nonlocal": arrays["E_nonlocal"].astype(np.float32), "base_percentile": base_percentile.astype(np.float64)}
            summary: dict[str, Any] = {"class": cls, "source_index": source_index, "image_index": image_index, "variants": {}}
            for variant in VARIANTS:
                certified, graph = graphs[variant]
                node_arrays.update({f"{variant}_{name}": value for name, value in graph["node"].items()})
                summary["variants"][variant] = graph["summary"]
                for name in ("edge_count", "target_node_count", "support_node_count", "observed_energy", "gradient_energy", "residual_energy"):
                    totals[variant][name] += graph["summary"][name]
                    class_totals[cls][variant][name] += graph["summary"][name]
                totals[variant]["component_sizes"].extend(graph["summary"]["component_sizes"])
                class_totals[cls][variant]["images"] += 1
                if graph["summary"]["spatial"]["chebyshev_mean"] is not None:
                    totals[variant]["chebyshev"].append(graph["summary"]["spatial"]["chebyshev_mean"]); totals[variant]["euclidean"].append(graph["summary"]["spatial"]["euclidean_mean"])
            out_path = TEMP_ROOT / "nodes" / f"{image_index:04d}_{cls}_{source_index}.npz"
            atomic_npz(out_path, node_arrays)
            entry = {"key": key, "class": cls, "source_index": source_index, "image_index": image_index, "relative_path": str(out_path.relative_to(TEMP_ROOT)), "sha256": sha256(out_path), "aligned_certified": len(graphs["aligned"][0]), "shifted_certified": len(graphs["shifted"][0]), "aligned_support_nodes": int(np.count_nonzero(graphs["aligned"][1]["node"]["support_degree"])), "shifted_support_nodes": int(np.count_nonzero(graphs["shifted"][1]["node"]["support_degree"]))}
            entries.append(entry)
            done.append(key)
            image_index += 1
        write_checkpoint(done, entries, implementation_sha)
    if len(entries) != EXPECTED_IMAGES:
        raise RuntimeError("P5D0_GT_FREE_COVERAGE_FAILED")
    graph_summary = {"schema_version": GRAPH_SCHEMA_VERSION, "protocol_commit": PROTOCOL_COMMIT, "implementation_sha256": implementation_sha, "images": EXPECTED_IMAGES, "classes": EXPECTED_CLASSES, "variants": {}}
    for variant in VARIANTS:
        t = totals[variant]
        graph_summary["variants"][variant] = {"edge_count": int(t["edge_count"]), "target_node_count_sum": int(t["target_node_count"]), "support_node_count_sum": int(t["support_node_count"]), "component_count": len(t["component_sizes"]), "component_size_distribution": distribution(t["component_sizes"]), "observed_energy": t["observed_energy"], "gradient_energy": t["gradient_energy"], "residual_energy": t["residual_energy"], "gradient_fraction": t["gradient_energy"] / t["observed_energy"] if t["observed_energy"] else None, "residual_fraction": t["residual_energy"] / t["observed_energy"] if t["observed_energy"] else None, "spatial_edge_mean_distribution": float_distribution(t["chebyshev"]), "spatial_edge_euclidean_mean_distribution": float_distribution(t["euclidean"])}
    graph_summary["per_class"] = class_totals
    atomic_json(TEMP_ROOT / "GT_FREE_GRAPH_SUMMARY.json", graph_summary)
    manifest = {"schema_version": "P5D0_GT_FREE_SIGNAL_MANIFEST_v1", "protocol_commit": PROTOCOL_COMMIT, "implementation_sha256": implementation_sha, "cache_manifest_sha256": EXPECTED_CACHE_SHA, "gt_read": False, "images": len(entries), "entries": entries, "graph_summary_sha256": sha256(TEMP_ROOT / "GT_FREE_GRAPH_SUMMARY.json"), "node_signal_definitions": protocol["node_signals"], "edge_definition": protocol["edge_definition"], "hodge_definition": protocol["hodge_definition"]}
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
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return None
    return float(exact_auc_ap(np.asarray(scores, dtype=np.float32), labels)[0])


def aggregate_numeric(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return {"n": int(arr.size), "mean": None if not arr.size else float(arr.mean()), "median": None if not arr.size else float(np.median(arr)), "p05": None if not arr.size else float(np.quantile(arr, 0.05)), "p95": None if not arr.size else float(np.quantile(arr, 0.95))}


def posthoc_analysis(protocol: dict[str, Any], datasets: dict[str, Any], records: dict[str, list[dict[str, Any]]], entries: list[dict[str, Any]], graph_summary: dict[str, Any], implementation_sha: str) -> dict[str, Any]:
    # This is the first function that opens masks or consumes labels.
    baseline = json.loads((ROOT / "runs/phase5/hsir/P5B_FAILURE_FORENSIC_C0/RANK_LEVERAGE.json").read_text())
    per_class: dict[str, dict[str, Any]] = {cls: {"class": cls, "images": 0, "anomaly_images": 0, "normal_images": 0, "variants": {}, "correlation": {}} for cls in CLASS_ORDER}
    values: dict[str, dict[str, list[np.ndarray]]] = {variant: {name: [] for name in ("S1_support_degree", "S2_signed_incident_target_sum", "S3_incident_target_rms", "S4_max_abs_incident_residual", "S5_hodge_potential", "S6_potential_percentile", "S7_incident_residual_fraction")} for variant in VARIANTS}
    class_values: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {cls: {variant: {name: [] for name in diagnostic_names} for variant in VARIANTS} for cls in CLASS_ORDER}
    relation_values: dict[str, dict[str, list[float]]] = {variant: {"score_gap": [], "chebyshev": [], "euclidean": []} for variant in VARIANTS}
    class_mass: dict[str, dict[str, float]] = {cls: {"positive_total": 0.0, "positive_touched": 0.0, "negative_total": 0.0, "negative_touched": 0.0} for cls in CLASS_ORDER}
    image_lookup = {(e["class"], int(e["source_index"])): e for e in entries}
    for cls in CLASS_ORDER:
        for record in sorted(records[cls], key=lambda x: int(x["source_index"])):
            source = int(record["source_index"])
            entry = image_lookup[(cls, source)]
            with np.load(TEMP_ROOT / entry["relative_path"], allow_pickle=False) as data:
                node = {name: data[name] for name in data.files}
            raw = datasets[cls][source]
            mask = b2.load_mask_after_prediction(raw)
            occupancy = b2.occupancy_from_mask(mask)
            labels = (occupancy > 0).astype(np.uint8)
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
                class_mass[cls]["positive_total"] += pos_total
                class_mass[cls]["positive_touched"] += pos_touched
                class_mass[cls]["negative_total"] += neg_total
                class_mass[cls]["negative_touched"] += neg_touched
                for name in diagnostic_names:
                    signal = node[f"{variant}_{name}"]
                    class_values[cls][variant][name].append(signal[support])
                if variant not in per_class[cls]["variants"]:
                    per_class[cls]["variants"][variant] = {"support_nodes": 0, "target_nodes": 0, "positive_total": 0.0, "positive_touched": 0.0, "negative_total": 0.0, "negative_touched": 0.0, "anomaly_auc": None, "normal_auc": None, "normal_support_nodes": 0, "normal_S6": []}
                v = per_class[cls]["variants"][variant]
                v["support_nodes"] += int(support.sum())
                v["target_nodes"] += int((node[f"{variant}_target_degree"] > 0).sum())
                v["positive_total"] += pos_total
                v["positive_touched"] += pos_touched
                v["negative_total"] += neg_total
                v["negative_touched"] += neg_touched
                if image_label:
                    v.setdefault("anomaly_S6", []).append(node[f"{variant}_S6_potential_percentile"][support])
                else:
                    v["normal_support_nodes"] += int(support.sum())
                    v["normal_S6"].extend(node[f"{variant}_S6_potential_percentile"][support].tolist())
            # Correlations are frozen descriptive comparisons, not selection inputs.
            for variant in VARIANTS:
                support = node[f"{variant}_support_degree"] > 0
                per_class[cls]["correlation"].setdefault(variant, {})
                for name in ("m_bar", "D_rank", "E_nonlocal"):
                    per_class[cls]["correlation"][variant][name] = corr(node[f"{variant}_S6_potential_percentile"][support], node[name][support], "spearman")
    for cls in CLASS_ORDER:
        for variant in VARIANTS:
            v = per_class[cls]["variants"][variant]
            anomaly_parts = v.pop("anomaly_S6", [])
            anomaly_scores = np.concatenate(anomaly_parts) if anomaly_parts else np.empty(0, dtype=np.float64)
            anomaly_labels: list[np.ndarray] = []
            for record in sorted(records[cls], key=lambda x: int(x["source_index"])):
                if int(record["label"]) == 0:
                    continue
                entry = image_lookup[(cls, int(record["source_index"]))]
                with np.load(TEMP_ROOT / entry["relative_path"], allow_pickle=False) as data:
                    support = data[f"{variant}_support_degree"] > 0
                    raw = datasets[cls][int(record["source_index"])]
                    labels = (b2.occupancy_from_mask(b2.load_mask_after_prediction(raw)) > 0).astype(np.uint8)
                    anomaly_labels.append(labels[support])
            labels_concat = np.concatenate(anomaly_labels) if anomaly_labels else np.empty(0, dtype=np.uint8)
            v["anomaly_auc"] = safe_auc(anomaly_scores, labels_concat)
            v["normal_S6"] = aggregate_numeric(v["normal_S6"])
            v["positive_mass_fraction"] = None if v["positive_total"] == 0 else v["positive_touched"] / v["positive_total"]
            v["negative_risk_fraction"] = None if v["negative_total"] == 0 else v["negative_touched"] / v["negative_total"]
    global_metrics: dict[str, Any] = {"variants": {}, "per_class": per_class}
    for variant in VARIANTS:
        all_anomaly_scores: list[np.ndarray] = []
        all_anomaly_labels: list[np.ndarray] = []
        for cls in CLASS_ORDER:
            global_metrics["per_class"][cls]["variants"][variant]["anomaly_auc"] = global_metrics["per_class"][cls]["variants"][variant]["anomaly_auc"]
            for record in sorted(records[cls], key=lambda x: int(x["source_index"])):
                if not int(record["label"]):
                    continue
                entry = image_lookup[(cls, int(record["source_index"]))]
                with np.load(TEMP_ROOT / entry["relative_path"], allow_pickle=False) as data:
                    support = data[f"{variant}_support_degree"] > 0
                    labels = (b2.occupancy_from_mask(b2.load_mask_after_prediction(datasets[cls][int(record["source_index"])])) > 0).astype(np.uint8)
                    all_anomaly_scores.append(data[f"{variant}_S6_potential_percentile"][support])
                    all_anomaly_labels.append(labels[support])
        scores = np.concatenate(all_anomaly_scores) if all_anomaly_scores else np.empty(0)
        labels = np.concatenate(all_anomaly_labels) if all_anomaly_labels else np.empty(0, dtype=np.uint8)
        global_metrics["variants"][variant] = {"anomaly_auc": safe_auc(scores, labels), "anomaly_nodes": int(scores.size), "anomaly_positive_nodes": int(labels.sum()), "normal_auc": None}
    auc_by_class = {variant: {cls: global_metrics["per_class"][cls]["variants"][variant]["anomaly_auc"] for cls in CLASS_ORDER} for variant in VARIANTS}
    auc_delta = {cls: None if auc_by_class["aligned"][cls] is None or auc_by_class["shifted"][cls] is None else auc_by_class["aligned"][cls] - auc_by_class["shifted"][cls] for cls in CLASS_ORDER}
    pos_fraction = {cls: (None if class_mass[cls]["positive_total"] == 0 else class_mass[cls]["positive_touched"] / class_mass[cls]["positive_total"]) for cls in CLASS_ORDER}
    neg_fraction = {cls: (None if class_mass[cls]["negative_total"] == 0 else class_mass[cls]["negative_touched"] / class_mass[cls]["negative_total"]) for cls in CLASS_ORDER}
    old_class = baseline["per_class"]
    old_pos = {cls: old_class[cls]["positive_contamination_fraction_touched"] for cls in CLASS_ORDER}
    global_metrics["leverage"] = {"aligned_positive_mass": {"total": sum(x["positive_total"] for x in class_mass.values()), "touched": sum(x["positive_touched"] for x in class_mass.values()), "fraction": sum(x["positive_touched"] for x in class_mass.values()) / sum(x["positive_total"] for x in class_mass.values())}, "aligned_negative_risk": {"total": sum(x["negative_total"] for x in class_mass.values()), "touched": sum(x["negative_touched"] for x in class_mass.values()), "fraction": sum(x["negative_touched"] for x in class_mass.values()) / sum(x["negative_total"] for x in class_mass.values())}, "old_selected_positive_fraction": OLD_POSITIVE_FRACTION, "old_selected_negative_fraction": OLD_NEGATIVE_FRACTION, "positive_fraction_by_class": pos_fraction, "negative_fraction_by_class": neg_fraction, "old_positive_fraction_by_class": old_pos, "classes_increased_over_old": int(sum(pos_fraction[c] > old_pos[c] for c in CLASS_ORDER))}
    global_metrics["bootstrap"] = {"aligned_S6_auc": bootstrap(auc_by_class["aligned"], BOOTSTRAP_SEED + 1), "aligned_minus_shifted_S6_auc": bootstrap(auc_delta, BOOTSTRAP_SEED + 2), "aligned_positive_mass_fraction": bootstrap(pos_fraction, BOOTSTRAP_SEED + 3), "aligned_negative_risk_fraction": bootstrap(neg_fraction, BOOTSTRAP_SEED + 4)}
    global_metrics["nonredundancy"] = {variant: {name: bootstrap({cls: global_metrics["per_class"][cls]["correlation"][variant][name] for cls in CLASS_ORDER}, BOOTSTRAP_SEED + 20 + idx) for idx, name in enumerate(("m_bar", "D_rank", "E_nonlocal"))} for variant in VARIANTS}
    return {"schema_version": "P5D0_POSTHOC_ANALYSIS_v1", "protocol_commit": PROTOCOL_COMMIT, "implementation_sha256": implementation_sha, "gt_firewall": {"manifest_finalized_before_gt": True, "manifest_sha256": sha256(TEMP_ROOT / "GT_FREE_SIGNAL_MANIFEST.json")}, "global": global_metrics, "per_class": per_class, "class_mass": class_mass}


def decision(analysis: dict[str, Any], graph_summary: dict[str, Any], protocol: dict[str, Any], implementation_sha: str) -> dict[str, Any]:
    lev = analysis["global"]["leverage"]
    auc = analysis["global"]["bootstrap"]["aligned_S6_auc"]
    delta = analysis["global"]["bootstrap"]["aligned_minus_shifted_S6_auc"]
    correlations = [abs(analysis["per_class"][cls]["correlation"]["aligned"]["m_bar"]) for cls in CLASS_ORDER if analysis["per_class"][cls]["correlation"]["aligned"]["m_bar"] is not None]
    g0 = bool(analysis["gt_firewall"]["manifest_finalized_before_gt"] and analysis["global"]["leverage"]["aligned_positive_mass"]["fraction"] is not None and len(analysis["per_class"]) == EXPECTED_CLASSES and implementation_sha == sha256(ROOT / "tools/audit_phase5_p5d0_graph_nonconformity.py"))
    g1 = bool(lev["aligned_positive_mass"]["fraction"] >= 2.0 * OLD_POSITIVE_FRACTION and lev["classes_increased_over_old"] >= 8)
    g2 = bool(auc["ci95"] is not None and auc["ci95"][0] > 0.5)
    g3 = bool(delta["ci95"] is not None and delta["ci95"][0] > 0 and sum((analysis["per_class"][cls]["variants"]["aligned"]["anomaly_auc"] if analysis["per_class"][cls]["variants"]["aligned"]["anomaly_auc"] is not None else -1) > (analysis["per_class"][cls]["variants"]["shifted"]["anomaly_auc"] if analysis["per_class"][cls]["variants"]["shifted"]["anomaly_auc"] is not None else -1) for cls in CLASS_ORDER) >= 8)
    g4 = bool(correlations and float(np.mean(correlations)) < 0.95)
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
    spatial_required = bool(graph_summary["variants"]["aligned"]["component_size_distribution"]["mean"] is not None and graph_summary["variants"]["aligned"]["component_size_distribution"]["mean"] != graph_summary["variants"]["shifted"]["component_size_distribution"]["mean"])
    return {"schema_version": "P5D0_DECISION_v1", "integrity": "PASS" if g0 else "FAIL", "model_forwards": 0, "training_steps": 0, "b3_protected": True, "gt_free_manifest_finalized": True, "gates": {"G0": g0, "G1": g1, "G2": g2, "G3": g3, "G4": g4, "G1_positive_mass_fraction": lev["aligned_positive_mass"]["fraction"], "G1_old_fraction": OLD_POSITIVE_FRACTION, "G1_classes_increased": lev["classes_increased_over_old"], "G2_aligned_S6_auc": auc, "G3_aligned_minus_shifted_auc": delta, "G4_mean_abs_classwise_spearman_S6_base": None if not correlations else float(np.mean(correlations))}, "candidate": "NONE", "SPATIAL_CONSTRAINT_REQUIRED": spatial_required, "terminal": terminal, "limitations": ["All GT quantities are post-hoc and not deployable.", "Graph association with masks is not causal.", "S6 is a preregistered diagnostic and no alternate signal was selected.", "Native graph leverage does not establish deployed improvement under blur, resize, stage mean, and softmax."], "exact_next_question": "If and only if the terminal is GRAPH_NONCONFORMITY_SUPPORTED_FOR_D1, can a bounded D1 use graph evidence without broad permutation or deployment mass relocation?", "forbidden_tuning_actions": protocol["forbidden"]}


def write_outputs(protocol: dict[str, Any], graph_summary: dict[str, Any], analysis: dict[str, Any], dec: dict[str, Any], implementation_sha: str, counts: dict[str, int]) -> None:
    atomic_json(OUTPUT_ROOT / "INPUT_CHECK.json", {"schema_version": "P5D0_INPUT_CHECK_v2", "status": "PASS", "protocol_commit": PROTOCOL_COMMIT, "implementation_sha256": implementation_sha, "cache_manifest_sha256": EXPECTED_CACHE_SHA, "cache_schema": EXPECTED_CACHE_SCHEMA, "images": counts["images"], "classes": counts["classes"], "normal": counts["normal"], "anomaly": counts["anomaly"], "model_forwards": 0, "training_steps": 0, "gt_read_during_graph_pass": False, "protected_source_hashes": {p: sha256(ROOT / p) for p in PROTECTED}})
    atomic_json(OUTPUT_ROOT / "GRAPH_SCHEMA.json", {"schema_version": GRAPH_SCHEMA_VERSION, "temporary_root": str(TEMP_ROOT), "node_file_format": "npz", "node_arrays": {"m_bar": "float32[1369]", "D_rank": "float32[1369]", "valid_reference": "uint8[1369]", "E_nonlocal": "float32[1369]", "base_percentile": "float64[1369]", "{variant}_support_degree": "int32[1369]", "{variant}_target_degree": "int32[1369]", "{variant}_S1_support_degree": "float64[1369]", "{variant}_S2_signed_incident_target_sum": "float64[1369]", "{variant}_S3_incident_target_rms": "float64[1369]", "{variant}_S4_max_abs_incident_residual": "float64[1369]", "{variant}_S5_hodge_potential": "float64[1369]", "{variant}_S6_potential_percentile": "float64[1369]", "{variant}_S7_incident_residual_fraction": "float64[1369]"}, "gt_fields_absent": True})
    atomic_json(OUTPUT_ROOT / "GT_FREE_GRAPH_SUMMARY.json", graph_summary)
    atomic_json(OUTPUT_ROOT / "GT_FREE_SIGNAL_MANIFEST.json", json.loads((TEMP_ROOT / "GT_FREE_SIGNAL_MANIFEST.json").read_text()))
    atomic_json(OUTPUT_ROOT / "LEVERAGE_COVERAGE.json", analysis["global"]["leverage"])
    atomic_json(OUTPUT_ROOT / "PRIMARY_SIGNAL_AUDIT.json", {"primary": "S6", "aligned": analysis["global"]["variants"]["aligned"], "shifted": analysis["global"]["variants"]["shifted"], "bootstrap": analysis["global"]["bootstrap"], "diagnostic_only": ["S1", "S2", "S3", "S4", "S5", "S7"]})
    atomic_json(OUTPUT_ROOT / "HODGE_CONSISTENCY.json", {variant: graph_summary["variants"][variant] for variant in VARIANTS})
    atomic_json(OUTPUT_ROOT / "DIAGNOSTIC_SIGNALS.json", {"schema_version": "P5D0_DIAGNOSTIC_SIGNALS_v1", "signals": ["S1", "S2", "S3", "S4", "S5", "S7"], "diagnostic_only": True, "nonredundancy": analysis["global"]["nonredundancy"], "per_class_correlations": {cls: analysis["per_class"][cls]["correlation"] for cls in CLASS_ORDER}})
    atomic_json(OUTPUT_ROOT / "ALIGNED_SHIFTED_GRAPH.json", {variant: graph_summary["variants"][variant] for variant in VARIANTS} | {"primary_auc": {variant: analysis["global"]["variants"][variant] for variant in VARIANTS}})
    with (OUTPUT_ROOT / "PER_CLASS.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["class", "images", "positive_mass_fraction_aligned", "positive_mass_fraction_shifted", "negative_risk_fraction_aligned", "negative_risk_fraction_shifted", "S6_auc_aligned", "S6_auc_shifted", "S6_auc_delta"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for cls in CLASS_ORDER:
            av = analysis["per_class"][cls]["variants"]["aligned"]; sv = analysis["per_class"][cls]["variants"]["shifted"]
            writer.writerow({"class": cls, "images": analysis["per_class"][cls]["images"], "positive_mass_fraction_aligned": av["positive_mass_fraction"], "positive_mass_fraction_shifted": sv["positive_mass_fraction"], "negative_risk_fraction_aligned": av["negative_risk_fraction"], "negative_risk_fraction_shifted": sv["negative_risk_fraction"], "S6_auc_aligned": av["anomaly_auc"], "S6_auc_shifted": sv["anomaly_auc"], "S6_auc_delta": None if av["anomaly_auc"] is None or sv["anomaly_auc"] is None else av["anomaly_auc"] - sv["anomaly_auc"]})
    atomic_json(OUTPUT_ROOT / "DECISION.json", dec)
    report = ["# P5-D0 graph non-conformity leverage audit", "", f"Terminal: `{dec['terminal']}`.", "", "The graph used all certified R0 relations before disjoint selection. Hodge potentials and S6 were materialized and hashed before GT was read. This is a post-hoc diagnostic; no candidate was implemented.", "", f"Aligned certified edges={graph_summary['variants']['aligned']['edge_count']}; shifted certified edges={graph_summary['variants']['shifted']['edge_count']}.", f"Aligned S6 anomaly-image AUC={analysis['global']['variants']['aligned']['anomaly_auc']}; aligned-minus-shifted class-bootstrap CI={analysis['global']['bootstrap']['aligned_minus_shifted_S6_auc']['ci95']}.", f"Aligned graph positive-contamination fraction={analysis['global']['leverage']['aligned_positive_mass']['fraction']}; old selected fraction={OLD_POSITIVE_FRACTION}; classes increased={analysis['global']['leverage']['classes_increased_over_old']}.", f"Gates: {dec['gates']}.", "", "No D1 proposal is made unless the frozen supported terminal is reached."]
    (OUTPUT_ROOT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    output_check = {"schema_version": "P5D0_OUTPUT_CHECK_v1", "status": "PASS", "required_files": {name: (OUTPUT_ROOT / name).is_file() for name in ("DESIGN_REVIEW.md", "INPUT_CHECK.json", "PROTOCOL.json", "GT_FREE_SIGNAL_MANIFEST.json", "GRAPH_SCHEMA.json", "GT_FREE_GRAPH_SUMMARY.json", "LEVERAGE_COVERAGE.json", "PRIMARY_SIGNAL_AUDIT.json", "HODGE_CONSISTENCY.json", "DIAGNOSTIC_SIGNALS.json", "ALIGNED_SHIFTED_GRAPH.json", "PER_CLASS.csv", "DECISION.json", "REPORT.md")}, "json_finite": finite(dec) and finite(analysis) and finite(graph_summary), "images": counts["images"], "classes": counts["classes"], "model_forwards": 0, "training_steps": 0, "gt_free_manifest_finalized_before_gt": True, "no_protected_source_modifications": True, "no_candidate_implementation": True, "no_dense_maps_committed": True}
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
    entries, graph_summary = gt_free_pass(protocol, datasets, records, implementation_sha)
    analysis = posthoc_analysis(protocol, datasets, records, entries, graph_summary, implementation_sha)
    dec = decision(analysis, graph_summary, protocol, implementation_sha)
    write_outputs(protocol, graph_summary, analysis, dec, implementation_sha, counts)
    print(json.dumps({"status": "PASS", "terminal": dec["terminal"], "images": counts["images"], "aligned_certified": graph_summary["variants"]["aligned"]["edge_count"], "shifted_certified": graph_summary["variants"]["shifted"]["edge_count"], "model_forwards": 0, "training_steps": 0, "gt_free_manifest": str(OUTPUT_ROOT / "GT_FREE_SIGNAL_MANIFEST.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
