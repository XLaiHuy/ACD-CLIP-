#!/usr/bin/env python3
"""One preregistered, cache-only full evaluation of frozen P5B.

The evaluator constructs C0/P5/P5_SHIFT from the finalized GT-free R0 cache,
freezes those outputs, and only then loads each image mask for post-hoc
metrics.  It is measurement code; candidate selection and projection remain
in :mod:`phase5_selective_adjudication`.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from phase5_selective_adjudication import (  # noqa: E402
    apply_positive_only_projection,
    deploy_native_logits,
    select_gt_free,
)
import audit_phase5_b2_adjudication as b2  # noqa: E402
from audit_phase5_hsir import aggregate_values, exact_auc_ap, pairwise_risks  # noqa: E402


CACHE_ROOT = Path("/tmp/p5_r0_run2")
OUTPUT_ROOT = ROOT / "runs/phase5/hsir/P5B_FULL_EVAL"
SCRATCH_ROOT_BASE = Path("/tmp")
EXPECTED_CACHE_SHA = "cfbd66b04c04b314756d151b759d95041afc2a69a8dc411e24896a7b4f931365"
EXPECTED_CACHE_SCHEMA = "P5B_R0_GT_FREE_CACHE_v1"
EXPECTED_CANDIDATE_COMMIT = "d2c8ef0c75d80cd40a31c5042b104f39c267661a"
EXPECTED_R0_COMMIT = "8865489a7aad490d218886e8ec534187f9f70e12"
EXPECTED_START_HEAD = EXPECTED_CANDIDATE_COMMIT
EXPECTED_IMAGES = 2162
EXPECTED_CLASSES = 12
EXPECTED_NORMAL = 962
EXPECTED_ANOMALY = 1200
PATCH_COUNT = 37 * 37
IMAGE_SIZE = 518
STAGES = 3
K = 8
SHIFT = (12, 12)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 5501
REQUIRED_ARRAYS = {
    "m_bar",
    "D_rank",
    "valid_reference",
    "score_bin",
    "d_rank_bin",
    "E_nonlocal",
    "E_stage",
    "E_LOO",
    "native_logits",
    "aligned_selected_pairs",
    "aligned_selected_cost",
    "shifted_selected_pairs",
    "shifted_selected_cost",
}
PROTECTED_SOURCE_FILES = {
    "model/adapter.py",
    "tools/audit_phase5_b2_adjudication.py",
    "tools/audit_phase5_b3_action_mismatch.py",
    "tools/audit_phase5_reference_validity.py",
    "tools/audit_phase5_second_evidence.py",
}
FINAL_FILES = {
    "INPUT_CHECK.json",
    "PROTOCOL.json",
    "CACHE_CHECK.json",
    "PER_CLASS.csv",
    "SUMMARY.json",
    "NORMAL_SAFETY.json",
    "ACTION_DIAGNOSTICS.json",
    "DEPLOYMENT_ANALYSIS.json",
    "DECISION.json",
    "OUTPUT_CHECK.json",
    "REPORT.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    return True


def current_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def protected_hashes() -> dict[str, str]:
    return {path: sha256(ROOT / path) for path in sorted(PROTECTED_SOURCE_FILES)}


def shift_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return np.roll(values.reshape(37, 37), SHIFT, axis=(0, 1)).reshape(-1).astype(values.dtype)
    if values.ndim == 2 and values.shape[1] == PATCH_COUNT:
        return np.stack([shift_map(row) for row in values], axis=0).astype(values.dtype)
    raise ValueError(f"unsupported shift shape={values.shape}")


def trace_arrays(items: list[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    if not items:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)
    return (
        np.asarray([[i, j] for i, j, _ in items], dtype=np.int64),
        np.asarray([cost for _, _, cost in items], dtype=np.float64),
    )


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def check_record_arrays(arrays: dict[str, np.ndarray], key: str) -> None:
    missing = REQUIRED_ARRAYS.difference(arrays)
    if missing:
        raise RuntimeError(f"P5B_FULL_EVAL_CACHE_INVALID:{key}:missing={sorted(missing)}")
    expected = {
        "m_bar": (PATCH_COUNT,),
        "D_rank": (PATCH_COUNT,),
        "valid_reference": (PATCH_COUNT,),
        "score_bin": (PATCH_COUNT,),
        "d_rank_bin": (PATCH_COUNT,),
        "E_nonlocal": (PATCH_COUNT,),
        "E_stage": (STAGES, PATCH_COUNT),
        "E_LOO": (K, PATCH_COUNT),
        "native_logits": (STAGES, PATCH_COUNT, 2),
    }
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise RuntimeError(f"P5B_FULL_EVAL_CACHE_INVALID:{key}:{name}:shape={arrays[name].shape}")
        if name not in {"valid_reference", "score_bin", "d_rank_bin"} and not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"P5B_FULL_EVAL_CACHE_INVALID:{key}:{name}:nonfinite")
    for variant in ("aligned", "shifted"):
        pairs = arrays[f"{variant}_selected_pairs"]
        costs = arrays[f"{variant}_selected_cost"]
        if pairs.ndim != 2 or pairs.shape[1] != 2 or costs.shape != (pairs.shape[0],):
            raise RuntimeError(f"P5B_FULL_EVAL_CACHE_INVALID:{key}:{variant}:trace_shape")
        if pairs.size and (pairs.min() < 0 or pairs.max() >= PATCH_COUNT or np.unique(pairs).size != pairs.size):
            raise RuntimeError(f"P5B_FULL_EVAL_CACHE_INVALID:{key}:{variant}:trace_ids")
        if not np.all(np.isfinite(costs)):
            raise RuntimeError(f"P5B_FULL_EVAL_CACHE_INVALID:{key}:{variant}:trace_costs")


def selector_traces(arrays: dict[str, np.ndarray], key: str) -> dict[str, list[tuple[int, int, float]]]:
    aligned = select_gt_free(
        arrays["m_bar"], arrays["D_rank"], arrays["valid_reference"],
        arrays["E_nonlocal"], arrays["E_stage"], arrays["E_LOO"],
        arrays["score_bin"], arrays["d_rank_bin"],
    )
    shifted = select_gt_free(
        arrays["m_bar"], arrays["D_rank"], arrays["valid_reference"],
        shift_map(arrays["E_nonlocal"]), shift_map(arrays["E_stage"]),
        shift_map(arrays["E_LOO"]), arrays["score_bin"], arrays["d_rank_bin"],
    )
    for variant, trace in (("aligned", aligned["selected"]), ("shifted", shifted["selected"])):
        expected_pairs, expected_cost = trace_arrays(trace)
        actual_pairs = arrays[f"{variant}_selected_pairs"]
        actual_cost = arrays[f"{variant}_selected_cost"]
        if not np.array_equal(expected_pairs, actual_pairs) or not np.array_equal(expected_cost, actual_cost):
            raise RuntimeError(f"P5B_FULL_EVAL_SELECTOR_PARITY_FAILED:{key}:{variant}")
    return {"aligned": aligned["selected"], "shifted": shifted["selected"]}


def validate_cache(cache_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, int]]:
    manifest_path = cache_root / "CACHE_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("P5B_FULL_EVAL_CACHE_INVALID:manifest_missing")
    manifest = json.loads(manifest_path.read_text())
    manifest_digest = sha256(manifest_path)
    if manifest_digest != EXPECTED_CACHE_SHA:
        raise RuntimeError(f"P5B_FULL_EVAL_CACHE_INVALID:manifest_sha={manifest_digest}")
    if manifest.get("schema_version") != EXPECTED_CACHE_SCHEMA or not manifest.get("finalized"):
        raise RuntimeError("P5B_FULL_EVAL_CACHE_INVALID:manifest_protocol")
    if manifest.get("scientific_unique_image_forwards") != EXPECTED_IMAGES or manifest.get("training_steps") != 0:
        raise RuntimeError("P5B_FULL_EVAL_CACHE_INVALID:manifest_counts")
    if set(manifest.get("processed_image_keys", [])) != set(manifest.get("files", {})):
        raise RuntimeError("P5B_FULL_EVAL_CACHE_INVALID:manifest_key_set")
    datasets, records, counts = b2.canonical_records(IMAGE_SIZE)
    if counts != {"classes": EXPECTED_CLASSES, "images": EXPECTED_IMAGES, "normal": EXPECTED_NORMAL, "anomaly": EXPECTED_ANOMALY}:
        raise RuntimeError(f"P5B_FULL_EVAL_CACHE_INVALID:canonical_counts={counts}")
    expected_names = {
        f"{class_name}:{int(record['source_index'])}": str(record["file_name"])
        for class_name in sorted(records)
        for record in records[class_name]
    }
    if set(expected_names) != set(manifest["files"]):
        raise RuntimeError("P5B_FULL_EVAL_CACHE_INVALID:canonical_identity_set")
    checked = 0
    parity_checked = 0
    for key in sorted(manifest["files"]):
        entry = manifest["files"][key]
        if entry.get("image_identity") != expected_names[key]:
            raise RuntimeError(f"P5B_FULL_EVAL_CACHE_INVALID:identity:{key}")
        path = cache_root / entry["relative_path"]
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            raise RuntimeError(f"P5B_FULL_EVAL_CACHE_INVALID:checksum:{key}")
        arrays = load_arrays(path)
        check_record_arrays(arrays, key)
        selector_traces(arrays, key)
        checked += 1
        parity_checked += 1
    return manifest, datasets, records, {"file_count": checked, "parity_records": parity_checked, **counts}


def run_synthetic_tests() -> dict[str, Any]:
    rng = np.random.default_rng(5501)
    native = rng.normal(size=(STAGES, PATCH_COUNT, 2)).astype(np.float32)
    empty, empty_delta = apply_positive_only_projection(native, np.arange(PATCH_COUNT, dtype=np.float32), [])
    if not np.array_equal(empty, native) or np.any(empty_delta != 0):
        raise RuntimeError("P5B_FULL_EVAL_SYNTHETIC_FAILED:C0_identity")
    m = np.arange(PATCH_COUNT, dtype=np.float32)
    selected = [(0, 1, 1.0)]
    corrected, delta = apply_positive_only_projection(native, m, selected)
    if np.any(delta < 0) or delta[0] != 1.0 or delta[1] != 0.0:
        raise RuntimeError("P5B_FULL_EVAL_SYNTHETIC_FAILED:P5_delta")
    if not np.array_equal(corrected[:, :, 0], native[:, :, 0]) or not np.all(corrected[:, 0, 1] > native[:, 0, 1]):
        raise RuntimeError("P5B_FULL_EVAL_SYNTHETIC_FAILED:logit_invariance")
    p_candidate = deploy_native_logits(native)
    import torch
    p_b2, _ = b2.deploy_native(torch.from_numpy(native[:, None, :, :]))
    if not np.allclose(p_candidate, p_b2.detach().cpu().numpy(), atol=1e-6, rtol=0):
        raise RuntimeError("P5B_FULL_EVAL_SYNTHETIC_FAILED:deployment_parity")
    auc, ap = exact_auc_ap(np.asarray([0.0, 1.0], dtype=np.float32), np.asarray([0, 1], dtype=np.uint8))
    if auc != 1.0 or ap != 1.0:
        raise RuntimeError("P5B_FULL_EVAL_SYNTHETIC_FAILED:metric_helper")
    _, r_neg = pairwise_risks(np.asarray([0.0, 1.0], dtype=np.float32), np.asarray([0, 1], dtype=bool))
    if r_neg.size != 1 or not np.all(np.isfinite(r_neg)):
        raise RuntimeError("P5B_FULL_EVAL_SYNTHETIC_FAILED:pairwise_helper")
    if set(inspect.signature(select_gt_free).parameters).intersection({"gt", "mask", "label", "labels"}):
        raise RuntimeError("P5B_FULL_EVAL_SYNTHETIC_FAILED:selector_gt_interface")
    if set(inspect.signature(apply_positive_only_projection).parameters).intersection({"gt", "mask", "label", "labels"}):
        raise RuntimeError("P5B_FULL_EVAL_SYNTHETIC_FAILED:action_gt_interface")
    dense_extensions = {".npy", ".npz", ".pt", ".pth", ".bin"}
    if OUTPUT_ROOT.exists() and any(path.suffix in dense_extensions for path in OUTPUT_ROOT.iterdir()):
        raise RuntimeError("P5B_FULL_EVAL_SYNTHETIC_FAILED:dense_output")
    return {
        "status": "PASS",
        "model_forwards": 0,
        "training_steps": 0,
        "performance_metrics_computed": False,
        "C0_identity": True,
        "P5_delta_nonnegative": True,
        "deployment_parity": True,
        "metric_helper_plumbing": True,
        "selector_action_gt_free": True,
        "no_dense_full_dataset_cache_written": True,
    }


def class_aggregate(values: list[float | None], seed_offset: int) -> dict[str, Any]:
    result = aggregate_values(values, BOOTSTRAP_SEED + seed_offset)
    result["unit"] = "class"
    result["bootstrap_reps"] = BOOTSTRAP_REPS
    result["bootstrap_seed"] = BOOTSTRAP_SEED + seed_offset
    return result


def mean_negative_risk(scores: np.ndarray, labels: np.ndarray) -> float | None:
    _, r_neg = pairwise_risks(scores, labels)
    finite = r_neg[np.isfinite(r_neg)]
    return None if finite.size == 0 else float(np.mean(finite))


def metric_triplet(rows: list[dict[str, Any]], stem: str, seed: int) -> dict[str, Any]:
    fields = {"C0": f"{stem}_C0", "P5": f"{stem}_P5", "P5_SHIFT": f"{stem}_P5_SHIFT"}
    values = {name: class_aggregate([row[field] for row in rows], seed + index) for index, (name, field) in enumerate(fields.items())}
    deltas = {
        "P5_minus_C0": class_aggregate([row[f"{stem}_delta_P5_C0"] for row in rows], seed + 10),
        "P5_SHIFT_minus_C0": class_aggregate([row[f"{stem}_delta_SHIFT_C0"] for row in rows], seed + 11),
        "P5_minus_P5_SHIFT": class_aggregate([row[f"{stem}_delta_P5_SHIFT"] for row in rows], seed + 12),
    }
    return {**values, **deltas}


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "p95": float(np.quantile(arr, 0.95)), "max": float(arr.max())}


def action_diagnostics(selected_records: list[list[tuple[int, int, float]]], deltas: list[np.ndarray], spatial: list[dict[str, float]], selector_totals: dict[str, int]) -> dict[str, Any]:
    positive = [float(value) for delta in deltas for value in delta[delta > 0]]
    acted = int(sum(int(np.sum(delta > 0)) for delta in deltas))
    participating = int(sum(2 * len(items) for items in selected_records))
    support_rows = [item["row_i"] for item in spatial]
    support_cols = [item["col_i"] for item in spatial]
    return {
        "selected_relations": int(sum(len(items) for items in selected_records)),
        "participating_patches": participating,
        "acted_patches": acted,
        "images_with_actions": int(sum(bool(items) for items in selected_records)),
        "delta": _stats(positive),
        "total_positive_native_score_mass_added": float(sum(positive)),
        "spatial_support": {
            "acted_patch_mean_row": None if not support_rows else float(np.mean(support_rows)),
            "acted_patch_mean_col": None if not support_cols else float(np.mean(support_cols)),
            "acted_patch_min_row": None if not support_rows else int(min(support_rows)),
            "acted_patch_max_row": None if not support_rows else int(max(support_rows)),
            "acted_patch_min_col": None if not support_cols else int(min(support_cols)),
            "acted_patch_max_col": None if not support_cols else int(max(support_cols)),
            "pair_chebyshev": _stats([item["chebyshev"] for item in spatial]),
            "pair_euclidean": _stats([item["euclidean"] for item in spatial]),
        },
        "selector_labels": selector_totals,
    }


def evaluate_class(
    class_name: str,
    dataset: Any,
    records: list[dict[str, Any]],
    cache_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    pixel_scores = {"C0": [], "P5": [], "P5_SHIFT": []}
    pixel_labels: list[np.ndarray] = []
    patch_scores = {"C0": [], "P5": [], "P5_SHIFT": []}
    patch_labels: list[np.ndarray] = []
    normal_scores = {"C0": [], "P5": [], "P5_SHIFT": []}
    selected_by_variant = {"aligned": [], "shifted": []}
    deltas_by_variant = {"aligned": [], "shifted": []}
    spatial_by_variant = {"aligned": [], "shifted": []}
    selector_totals = {
        "aligned": {"mixed_label": 0, "same_label": 0, "rescue_opportunity": 0, "damage_risk": 0, "inversion_neutralized": 0},
        "shifted": {"mixed_label": 0, "same_label": 0, "rescue_opportunity": 0, "damage_risk": 0, "inversion_neutralized": 0},
    }
    source_indices = {int(record["source_index"]): record for record in records}
    if len(source_indices) != len(records):
        raise RuntimeError(f"P5B_FULL_EVAL_INPUT_INVALID:duplicate_source_index:{class_name}")
    for source_index in sorted(source_indices):
        record = source_indices[source_index]
        key = f"{class_name}:{source_index}"
        entry = manifest["files"][key]
        arrays = load_arrays(cache_root / entry["relative_path"])
        check_record_arrays(arrays, key)
        traces = selector_traces(arrays, key)
        native = arrays["native_logits"].astype(np.float32, copy=False)
        # Construct and freeze every method output before any mask is read.
        c0_native = native.copy()
        p5_native, p5_delta = apply_positive_only_projection(native, arrays["m_bar"], traces["aligned"])
        shift_native, shift_delta = apply_positive_only_projection(native, arrays["m_bar"], traces["shifted"])
        if not np.array_equal(c0_native, native) or np.any(p5_delta < 0) or np.any(shift_delta < 0):
            raise RuntimeError(f"P5B_FULL_EVAL_ACTION_INVALID:{key}")
        c0_prob = deploy_native_logits(c0_native)
        p5_prob = deploy_native_logits(p5_native)
        shift_prob = deploy_native_logits(shift_native)
        if not (np.all(np.isfinite(c0_prob)) and np.all(np.isfinite(p5_prob)) and np.all(np.isfinite(shift_prob))):
            raise RuntimeError(f"P5B_FULL_EVAL_NUMERIC_INVALID:{key}")

        # GT firewall: only now load the mask/label for post-hoc evaluation.
        raw = dataset[source_index]
        mask = b2.load_mask_after_prediction(raw)
        occupancy = b2.occupancy_from_mask(mask)
        pixel_target = mask.reshape(-1).astype(np.uint8)
        patch_target = (occupancy > 0).astype(np.uint8)
        labels_bool = pixel_target.astype(bool)
        pixel_scores["C0"].append(c0_prob[0, 1].reshape(-1).astype(np.float32))
        pixel_scores["P5"].append(p5_prob[0, 1].reshape(-1).astype(np.float32))
        pixel_scores["P5_SHIFT"].append(shift_prob[0, 1].reshape(-1).astype(np.float32))
        pixel_labels.append(pixel_target)
        patch_scores["C0"].append(arrays["m_bar"].astype(np.float32))
        patch_scores["P5"].append((arrays["m_bar"] + p5_delta).astype(np.float32))
        patch_scores["P5_SHIFT"].append((arrays["m_bar"] + shift_delta).astype(np.float32))
        patch_labels.append(patch_target)
        if int(record["label"]) == 0:
            normal_scores["C0"].append(pixel_scores["C0"][-1])
            normal_scores["P5"].append(pixel_scores["P5"][-1])
            normal_scores["P5_SHIFT"].append(pixel_scores["P5_SHIFT"][-1])
        for variant, selected, delta in (("aligned", traces["aligned"], p5_delta), ("shifted", traces["shifted"], shift_delta)):
            selected_by_variant[variant].append(selected)
            deltas_by_variant[variant].append(delta[delta > 0].copy())
            for i, j, _ in selected:
                row_i, col_i = divmod(int(i), 37)
                row_j, col_j = divmod(int(j), 37)
                spatial_by_variant[variant].append({
                    "row_i": float(row_i),
                    "col_i": float(col_i),
                    "chebyshev": float(max(abs(row_i - row_j), abs(col_i - col_j))),
                    "euclidean": float(math.hypot(row_i - row_j, col_i - col_j)),
                })
                li, lj = bool(patch_target[i]), bool(patch_target[j])
                if li == lj:
                    selector_totals[variant]["same_label"] += 1
                else:
                    selector_totals[variant]["mixed_label"] += 1
                    if li and not lj:
                        selector_totals[variant]["rescue_opportunity"] += 1
                    if (not li) and lj:
                        selector_totals[variant]["damage_risk"] += 1
            selector_totals[variant]["inversion_neutralized"] += len(selected)
        del raw, mask, c0_prob, p5_prob, shift_prob, c0_native, p5_native, shift_native
        gc.collect()

    pixels = {name: np.concatenate(values) for name, values in pixel_scores.items()}
    labels = np.concatenate(pixel_labels)
    patches = {name: np.concatenate(values) for name, values in patch_scores.items()}
    patch_target = np.concatenate(patch_labels)
    pixel_metrics = {name: exact_auc_ap(values, labels) for name, values in pixels.items()}
    patch_metrics = {name: exact_auc_ap(values, patch_target) for name, values in patches.items()}
    normal_c0 = np.concatenate(normal_scores["C0"])
    normal_p5 = np.concatenate(normal_scores["P5"])
    normal_shift = np.concatenate(normal_scores["P5_SHIFT"])
    normal = b2.normal_metrics(normal_c0, normal_p5, normal_shift)
    risk = {name: mean_negative_risk(values, labels) for name, values in pixels.items()}
    aligned_action = action_diagnostics(
        selected_by_variant["aligned"], deltas_by_variant["aligned"], spatial_by_variant["aligned"], selector_totals["aligned"]
    )
    shifted_action = action_diagnostics(
        selected_by_variant["shifted"], deltas_by_variant["shifted"], spatial_by_variant["shifted"], selector_totals["shifted"]
    )
    row: dict[str, Any] = {"class": class_name, "n_images": len(records)}
    for metric_name, metric_values in (("pixel_ap", {name: value[1] for name, value in pixel_metrics.items()}), ("pixel_auroc", {name: value[0] for name, value in pixel_metrics.items()}), ("native_patch_ap", {name: value[1] for name, value in patch_metrics.items()}), ("native_patch_auroc", {name: value[0] for name, value in patch_metrics.items()})):
        row[f"{metric_name}_C0"] = float(metric_values["C0"])
        row[f"{metric_name}_P5"] = float(metric_values["P5"])
        row[f"{metric_name}_P5_SHIFT"] = float(metric_values["P5_SHIFT"])
        row[f"{metric_name}_delta_P5_C0"] = float(metric_values["P5"] - metric_values["C0"])
        row[f"{metric_name}_delta_SHIFT_C0"] = float(metric_values["P5_SHIFT"] - metric_values["C0"])
        row[f"{metric_name}_delta_P5_SHIFT"] = float(metric_values["P5"] - metric_values["P5_SHIFT"])
    row.update({
        "negative_risk_C0": risk["C0"],
        "negative_risk_P5": risk["P5"],
        "negative_risk_P5_SHIFT": risk["P5_SHIFT"],
        "negative_risk_delta_P5_C0": None if risk["C0"] is None or risk["P5"] is None else float(risk["P5"] - risk["C0"]),
        "negative_risk_delta_SHIFT_C0": None if risk["C0"] is None or risk["P5_SHIFT"] is None else float(risk["P5_SHIFT"] - risk["C0"]),
        "negative_risk_delta_P5_SHIFT": None if risk["P5"] is None or risk["P5_SHIFT"] is None else float(risk["P5"] - risk["P5_SHIFT"]),
        "normal_tau95": float(normal["tau95"]),
        "normal_tau99": float(normal["tau99"]),
    })
    for metric in ("fpr_at_tau95", "fpr_at_tau99", "mean_anomaly_probability", "p99_anomaly_probability", "maximum_anomaly_probability"):
        for variant, source in (("C0", "C0"), ("P5", "C1"), ("P5_SHIFT", "C1_SHIFT")):
            row[f"normal_{metric}_{variant}"] = float(normal[source][metric])
        row[f"normal_{metric}_delta_P5_C0"] = float(normal["delta_C1"][metric])
        row[f"normal_{metric}_delta_SHIFT_C0"] = float(normal["delta_C1_SHIFT"][metric])
    for variant, action in (("aligned", aligned_action), ("shifted", shifted_action)):
        prefix = f"action_{variant}"
        row[f"{prefix}_selected_relations"] = action["selected_relations"]
        row[f"{prefix}_participating_patches"] = action["participating_patches"]
        row[f"{prefix}_acted_patches"] = action["acted_patches"]
        row[f"{prefix}_images_with_actions"] = action["images_with_actions"]
        row[f"{prefix}_delta_mean"] = action["delta"]["mean"]
        row[f"{prefix}_delta_median"] = action["delta"]["median"]
        row[f"{prefix}_delta_p95"] = action["delta"]["p95"]
        row[f"{prefix}_delta_max"] = action["delta"]["max"]
        row[f"{prefix}_positive_native_score_mass_added"] = action["total_positive_native_score_mass_added"]
        row[f"{prefix}_spatial_json"] = json.dumps(action["spatial_support"], sort_keys=True)
        for label_name, label_value in action["selector_labels"].items():
            row[f"selector_{variant}_{label_name}"] = label_value
    row["action_aligned_minus_shifted_relations"] = row["action_aligned_selected_relations"] - row["action_shifted_selected_relations"]
    row["action_aligned_minus_shifted_acted_patches"] = row["action_aligned_acted_patches"] - row["action_shifted_acted_patches"]
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "pixel_ap": metric_triplet(rows, "pixel_ap", 100),
        "pixel_auroc": metric_triplet(rows, "pixel_auroc", 120),
        "native_patch_ap": metric_triplet(rows, "native_patch_ap", 140),
        "native_patch_auroc": metric_triplet(rows, "native_patch_auroc", 160),
    }
    for variant, field in (("C0", "negative_risk_C0"), ("P5", "negative_risk_P5"), ("P5_SHIFT", "negative_risk_P5_SHIFT")):
        metrics.setdefault("negative_pairwise_risk", {})[variant] = class_aggregate([row[field] for row in rows], 180 + len(variant))
    metrics["negative_pairwise_risk"].update({
        "P5_minus_C0": class_aggregate([row["negative_risk_delta_P5_C0"] for row in rows], 190),
        "P5_SHIFT_minus_C0": class_aggregate([row["negative_risk_delta_SHIFT_C0"] for row in rows], 191),
        "P5_minus_P5_SHIFT": class_aggregate([row["negative_risk_delta_P5_SHIFT"] for row in rows], 192),
    })
    normal = {}
    for metric in ("fpr_at_tau95", "fpr_at_tau99", "mean_anomaly_probability", "p99_anomaly_probability", "maximum_anomaly_probability"):
        normal[metric] = {
            "C0": class_aggregate([row[f"normal_{metric}_C0"] for row in rows], 200 + len(metric)),
            "P5": class_aggregate([row[f"normal_{metric}_P5"] for row in rows], 220 + len(metric)),
            "P5_SHIFT": class_aggregate([row[f"normal_{metric}_P5_SHIFT"] for row in rows], 240 + len(metric)),
            "P5_minus_C0": class_aggregate([row[f"normal_{metric}_delta_P5_C0"] for row in rows], 260 + len(metric)),
            "P5_SHIFT_minus_C0": class_aggregate([row[f"normal_{metric}_delta_SHIFT_C0"] for row in rows], 280 + len(metric)),
        }
    normal["tau95_C0_class_macro"] = class_aggregate([row["normal_tau95"] for row in rows], 300)
    normal["tau99_C0_class_macro"] = class_aggregate([row["normal_tau99"] for row in rows], 301)
    metrics["normal_safety"] = normal
    return metrics


def aggregate_action(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    prefix = f"action_{variant}"
    return {
        "selected_relations_total": int(sum(row[f"{prefix}_selected_relations"] for row in rows)),
        "participating_patches_total": int(sum(row[f"{prefix}_participating_patches"] for row in rows)),
        "acted_patches_total": int(sum(row[f"{prefix}_acted_patches"] for row in rows)),
        "images_with_actions_total": int(sum(row[f"{prefix}_images_with_actions"] for row in rows)),
        "delta_mean_macro": class_aggregate([row[f"{prefix}_delta_mean"] for row in rows], 320),
        "delta_median_macro": class_aggregate([row[f"{prefix}_delta_median"] for row in rows], 321),
        "delta_p95_macro": class_aggregate([row[f"{prefix}_delta_p95"] for row in rows], 322),
        "delta_max_macro": class_aggregate([row[f"{prefix}_delta_max"] for row in rows], 323),
        "total_positive_native_score_mass_added": float(sum(row[f"{prefix}_positive_native_score_mass_added"] for row in rows)),
        "selector_labels_total": {name: int(sum(row[f"selector_{variant}_{name}"] for row in rows)) for name in ("mixed_label", "same_label", "rescue_opportunity", "damage_risk", "inversion_neutralized")},
    }


def build_decision(rows: list[dict[str, Any]], metrics: dict[str, Any], cache_check: dict[str, Any], protocol_commit: str, evaluator_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    ap_delta = metrics["pixel_ap"]["P5_minus_C0"]
    ap_shift = metrics["pixel_ap"]["P5_minus_P5_SHIFT"]
    auc_delta = metrics["pixel_auroc"]["P5_minus_C0"]
    e1 = {
        "mean_pixel_AP_delta_positive": ap_delta["mean"] is not None and ap_delta["mean"] > 0,
        "pixel_AP_bootstrap_lower_positive": ap_delta["bootstrap95_ci"] is not None and ap_delta["bootstrap95_ci"][0] > 0,
        "at_least_8_classes_pixel_AP_positive": sum(row["pixel_ap_delta_P5_C0"] > 0 for row in rows) >= 8,
        "macro_pixel_AUROC_delta_nonnegative": auc_delta["mean"] is not None and auc_delta["mean"] >= 0,
    }
    e2 = {
        "mean_pixel_AP_P5_minus_SHIFT_positive": ap_shift["mean"] is not None and ap_shift["mean"] > 0,
        "pixel_AP_P5_minus_SHIFT_bootstrap_lower_positive": ap_shift["bootstrap95_ci"] is not None and ap_shift["bootstrap95_ci"][0] > 0,
    }
    normal = metrics["normal_safety"]
    risk = metrics["negative_pairwise_risk"]["P5_minus_C0"]
    e3 = {
        "upper_CI_delta_FPR_tau95_nonpositive": normal["fpr_at_tau95"]["P5_minus_C0"]["bootstrap95_ci"] is not None and normal["fpr_at_tau95"]["P5_minus_C0"]["bootstrap95_ci"][1] <= 0,
        "upper_CI_delta_FPR_tau99_nonpositive": normal["fpr_at_tau99"]["P5_minus_C0"]["bootstrap95_ci"] is not None and normal["fpr_at_tau99"]["P5_minus_C0"]["bootstrap95_ci"][1] <= 0,
        "upper_CI_delta_negative_pairwise_risk_nonpositive": risk["bootstrap95_ci"] is not None and risk["bootstrap95_ci"][1] <= 0,
    }
    e0 = {
        "cache_integrity": cache_check["status"] == "PASS",
        "numeric_outputs_finite": True,
        "selector_trace_parity": cache_check["parity_records"] == EXPECTED_IMAGES,
        "gt_firewall": True,
        "zero_model_forwards": True,
        "zero_training": True,
        "class_count_12": len(rows) == EXPECTED_CLASSES,
        "image_count_2162": sum(int(row["n_images"]) for row in rows) == EXPECTED_IMAGES,
    }
    e0_pass = bool(all(e0.values()))
    e1_pass, e2_pass, e3_pass = bool(all(e1.values())), bool(all(e2.values())), bool(all(e3.values()))
    native = metrics["native_patch_ap"]["P5_minus_C0"]
    dilution = bool(native["mean"] is not None and native["mean"] > 0 and native["bootstrap95_ci"] is not None and native["bootstrap95_ci"][0] > 0 and not e1_pass)
    if not e0_pass:
        terminal = "P5B_FULL_EVAL_INVALID"
    elif e1_pass and e2_pass and e3_pass:
        terminal = "P5B_SELECTIVE_ADJUDICATION_SUPPORTED"
    elif e1_pass and e2_pass and not e3_pass:
        terminal = "P5B_EFFICACY_GROUNDED_SAFETY_NOT_DEMONSTRATED"
    else:
        terminal = "P5B_SELECTIVE_ADJUDICATION_UNSUPPORTED"
    decision = {
        "schema_version": "P5B_FULL_EVAL_v1",
        "integrity": "PASS" if e0_pass else "FAIL",
        "protocol_commit_sha": protocol_commit,
        "evaluator_sha256": evaluator_sha,
        "candidate_commit_sha": EXPECTED_CANDIDATE_COMMIT,
        "r0_evidence_commit_sha": EXPECTED_R0_COMMIT,
        "model_forwards": 0,
        "scientific_model_forwards": 0,
        "physical_model_forward_calls": 0,
        "training_steps": 0,
        "gt_firewall": {"status": "PASS", "gt_loaded_only_after_method_outputs_frozen": True, "gt_changes_selector_or_action": False},
        "E0": e0,
        "E1": e1,
        "E2": e2,
        "E3": e3,
        "E0_pass": e0_pass,
        "E1_pass": e1_pass,
        "E2_pass": e2_pass,
        "E3_pass": e3_pass,
        "primary_macro_deltas": {"pixel_AP_P5_minus_C0": ap_delta, "pixel_AP_P5_minus_SHIFT": ap_shift, "pixel_AUROC_P5_minus_C0": auc_delta, "native_patch_AP_P5_minus_C0": native},
        "class_consistency": {"pixel_AP_positive_P5_minus_C0": int(sum(row["pixel_ap_delta_P5_C0"] > 0 for row in rows)), "pixel_AP_positive_P5_minus_SHIFT": int(sum(row["pixel_ap_delta_P5_SHIFT"] > 0 for row in rows)), "classes": EXPECTED_CLASSES},
        "normal_safety_result": e3,
        "deployment_dilution": dilution,
        "terminal": terminal,
        "exact_next_scientific_question": "Can the frozen positive-only selector improve deployed pixel AP while satisfying the preregistered normal-safety guardrail?",
        "forbidden_posthoc_actions": ["Do not change selector, K=8, risk fraction, cells, unanimity, shift, projection, deployment, or action count.", "Do not tune thresholds or rerun another candidate automatically.", "Do not train, use medical data, or modify Phase2B/model predictor code.", "Do not relabel selector opportunities as strict post-action rescues/breaks."],
    }
    normal_output = {"schema_version": "P5B_FULL_EVAL_v1", "metrics": metrics["normal_safety"], "negative_pairwise_risk": metrics["negative_pairwise_risk"], "gates": e3}
    action_output = {"schema_version": "P5B_FULL_EVAL_v1", "aligned": aggregate_action(rows, "aligned"), "shifted": aggregate_action(rows, "shifted"), "aligned_minus_shifted": {"selected_relations": int(sum(row["action_aligned_minus_shifted_relations"] for row in rows)), "acted_patches": int(sum(row["action_aligned_minus_shifted_acted_patches"] for row in rows))}}
    deployment_output = {"schema_version": "P5B_FULL_EVAL_v1", "pixel_AP": metrics["pixel_ap"], "pixel_AUROC": metrics["pixel_auroc"], "native_patch_AP": metrics["native_patch_ap"], "native_patch_AUROC": metrics["native_patch_auroc"], "deployment_dilution": dilution}
    return decision, normal_output, action_output, deployment_output


def validate_outputs(rows: list[dict[str, Any]], protocol_commit: str, evaluator_sha: str) -> dict[str, Any]:
    expected_before_output_check = FINAL_FILES - {"OUTPUT_CHECK.json"}
    if {path.name for path in OUTPUT_ROOT.iterdir()} != expected_before_output_check:
        raise RuntimeError("P5B_FULL_EVAL_OUTPUT_INVALID:file_set")
    if len(rows) != EXPECTED_CLASSES or len({row["class"] for row in rows}) != EXPECTED_CLASSES:
        raise RuntimeError("P5B_FULL_EVAL_OUTPUT_INVALID:classes")
    with (OUTPUT_ROOT / "PER_CLASS.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != EXPECTED_CLASSES or sum(int(row["n_images"]) for row in csv_rows) != EXPECTED_IMAGES:
        raise RuntimeError("P5B_FULL_EVAL_OUTPUT_INVALID:csv_counts")
    for path in OUTPUT_ROOT.glob("*.json"):
        value = json.loads(path.read_text())
        if not finite_json(value):
            raise RuntimeError(f"P5B_FULL_EVAL_OUTPUT_INVALID:nonfinite:{path.name}")
    forbidden = [path.name for path in OUTPUT_ROOT.iterdir() if path.suffix in {".npy", ".npz", ".pt", ".pth", ".bin"}]
    if forbidden:
        raise RuntimeError(f"P5B_FULL_EVAL_OUTPUT_INVALID:dense={forbidden}")
    return {"status": "PASS", "json_finite": True, "csv_classes": EXPECTED_CLASSES, "csv_images": EXPECTED_IMAGES, "zero_model_forwards": True, "zero_training": True, "gt_free_selector_action": True, "protocol_commit_sha": protocol_commit, "evaluator_sha256": evaluator_sha, "no_dense_artifacts": True}


def write_report(decision: dict[str, Any], metrics: dict[str, Any], action: dict[str, Any]) -> None:
    ap = metrics["pixel_ap"]["P5_minus_C0"]
    shift = metrics["pixel_ap"]["P5_minus_P5_SHIFT"]
    native = metrics["native_patch_ap"]["P5_minus_C0"]
    lines = [
        "# P5B frozen full evaluation",
        "",
        f"Terminal: `{decision['terminal']}`.",
        "",
        f"Pixel AP P5-C0: mean={ap['mean']}, class-bootstrap CI={ap['bootstrap95_ci']}; P5-P5_SHIFT: mean={shift['mean']}, CI={shift['bootstrap95_ci']}.",
        f"Native patch AP P5-C0: mean={native['mean']}, CI={native['bootstrap95_ci']}; deployment_dilution={decision['deployment_dilution']}.",
        f"Aligned selected relations={action['aligned']['selected_relations_total']}; shifted={action['shifted']['selected_relations_total']}; aligned acted patches={action['aligned']['acted_patches_total']}; shifted acted patches={action['shifted']['acted_patches_total']}.",
        "",
        "All selector/action fields were frozen before GT masks were loaded. No model forward, training, tuning, medical data, or candidate modification was used.",
    ]
    atomic_write(OUTPUT_ROOT / "REPORT.md", "\n".join(lines) + "\n")


def run_validation(cache_root: Path) -> None:
    manifest, _, _, counts = validate_cache(cache_root)
    synthetic = run_synthetic_tests()
    result = {"status": "PASS", "cache_manifest_sha256": sha256(cache_root / "CACHE_MANIFEST.json"), "cache_records": counts["file_count"], "selector_parity_records": counts["parity_records"], "synthetic": synthetic, "performance_metrics_computed": False, "model_forwards": 0, "training_steps": 0}
    print(json.dumps(result, sort_keys=True))


def run_evaluation(cache_root: Path, protocol_commit: str) -> None:
    if current_head() != protocol_commit:
        raise RuntimeError(f"P5B_FULL_EVAL_PROTOCOL_HEAD_MISMATCH:{current_head()}:{protocol_commit}")
    protocol_path = OUTPUT_ROOT / "PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    evaluator_sha = sha256(ROOT / "tools/audit_phase5_p5b_full_eval.py")
    if protocol.get("evaluator_sha256") != evaluator_sha or protocol.get("performance_metrics_computed") is not False:
        raise RuntimeError("P5B_FULL_EVAL_PROTOCOL_INVALID")
    manifest, datasets, records, counts = validate_cache(cache_root)
    synthetic = run_synthetic_tests()
    cache_check = {"schema_version": "P5B_FULL_EVAL_v1", "status": "PASS", "manifest": str(cache_root / "CACHE_MANIFEST.json"), "manifest_sha256": sha256(cache_root / "CACHE_MANIFEST.json"), "schema_version_cache": manifest["schema_version"], "finalized": True, "file_count": counts["file_count"], "unique_canonical_images": counts["images"], "parity_records": counts["parity_records"], "training_steps": 0, "scientific_model_forwards": 0, "physical_model_forward_calls": 0, "gt_free_cache_finalized_before_gt": True, "all_referenced_checksums_valid": True, "synthetic_validation": synthetic, "performance_metrics_computed": False}
    atomic_json(OUTPUT_ROOT / "CACHE_CHECK.json", cache_check)
    scratch = SCRATCH_ROOT_BASE / f"p5b_full_eval_class_results_{protocol_commit[:12]}"
    scratch.mkdir(parents=True, exist_ok=True)
    class_rows: list[dict[str, Any]] = []
    metrics_started = False
    for class_name in sorted(records):
        class_path = scratch / f"{class_name}.json"
        if class_path.is_file():
            cached_row = json.loads(class_path.read_text())
            if cached_row.get("protocol_commit_sha") == protocol_commit and cached_row.get("evaluator_sha256") == evaluator_sha:
                class_rows.append(cached_row["row"])
                continue
        metrics_started = True
        row = evaluate_class(class_name, datasets[class_name], records[class_name], cache_root, manifest)
        if not finite_json(row):
            raise RuntimeError(f"P5B_FULL_EVAL_MEASUREMENT_INVALID:nonfinite_class:{class_name}")
        atomic_json(class_path, {"protocol_commit_sha": protocol_commit, "evaluator_sha256": evaluator_sha, "row": row})
        reopened = json.loads(class_path.read_text())
        if reopened["row"]["class"] != class_name or not finite_json(reopened):
            raise RuntimeError(f"P5B_FULL_EVAL_MEASUREMENT_INVALID:checkpoint:{class_name}")
        class_rows.append(row)
        print(json.dumps({"event": "class_complete", "class": class_name, "images": row["n_images"], "metrics_started": metrics_started}), flush=True)
    class_rows.sort(key=lambda row: row["class"])
    if len(class_rows) != EXPECTED_CLASSES or sum(row["n_images"] for row in class_rows) != EXPECTED_IMAGES:
        raise RuntimeError("P5B_FULL_EVAL_MEASUREMENT_INVALID:class_coverage")
    metrics = build_metrics(class_rows)
    integrity = {"status": "PASS", "cache": cache_check, "class_rows": len(class_rows), "images": sum(row["n_images"] for row in class_rows), "gt_firewall": True, "numeric_finite": True, "selector_parity": True, "model_forwards": 0, "training_steps": 0}
    decision, normal_output, action_output, deployment_output = build_decision(class_rows, metrics, {**cache_check, "status": "PASS"}, protocol_commit, evaluator_sha)
    summary = {"schema_version": "P5B_FULL_EVAL_v1", "protocol_commit_sha": protocol_commit, "candidate_commit_sha": EXPECTED_CANDIDATE_COMMIT, "r0_evidence_commit_sha": EXPECTED_R0_COMMIT, "model_forwards": 0, "scientific_model_forwards": 0, "physical_model_forward_calls": 0, "training_steps": 0, "performance_metrics_computed": True, "gt_firewall": decision["gt_firewall"], "counts": {"classes": EXPECTED_CLASSES, "images": EXPECTED_IMAGES, "normal": EXPECTED_NORMAL, "anomaly": EXPECTED_ANOMALY}, "metrics": metrics, "gates": {"E0": decision["E0"], "E1": decision["E1"], "E2": decision["E2"], "E3": decision["E3"], "E0_pass": decision["E0_pass"], "E1_pass": decision["E1_pass"], "E2_pass": decision["E2_pass"], "E3_pass": decision["E3_pass"]}, "decision": decision["terminal"], "deployment_dilution": decision["deployment_dilution"]}
    atomic_json(OUTPUT_ROOT / "SUMMARY.json", summary)
    atomic_json(OUTPUT_ROOT / "NORMAL_SAFETY.json", normal_output)
    atomic_json(OUTPUT_ROOT / "ACTION_DIAGNOSTICS.json", action_output)
    atomic_json(OUTPUT_ROOT / "DEPLOYMENT_ANALYSIS.json", deployment_output)
    atomic_json(OUTPUT_ROOT / "DECISION.json", decision)
    write_csv(OUTPUT_ROOT / "PER_CLASS.csv", class_rows)
    write_report(decision, metrics, action_output)
    output_check = validate_outputs(class_rows, protocol_commit, evaluator_sha)
    atomic_json(OUTPUT_ROOT / "OUTPUT_CHECK.json", output_check)
    # Reopen the final output check after its atomic write.
    if json.loads((OUTPUT_ROOT / "OUTPUT_CHECK.json").read_text())["status"] != "PASS":
        raise RuntimeError("P5B_FULL_EVAL_OUTPUT_INVALID:output_check")
    print(json.dumps({"event": "full_evaluation_complete", "terminal": decision["terminal"], "model_forwards": 0, "training_steps": 0, "performance_metrics_computed": True}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "evaluate"), required=True)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--protocol-commit", type=str)
    args = parser.parse_args()
    if args.mode == "validate":
        run_validation(args.cache_root)
    else:
        if not args.protocol_commit:
            raise SystemExit("--protocol-commit is required in evaluate mode")
        run_evaluation(args.cache_root, args.protocol_commit)


if __name__ == "__main__":
    main()
