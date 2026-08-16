#!/usr/bin/env python3
"""Zero-GT operational replay for the frozen P5B candidate.

This replay consumes only the finalized R0 cache.  It checks selector parity,
positive-only action invariants, and finite deployment outputs.  It does not
load dataset records, masks, labels, or any performance metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phase5_selective_adjudication import (
    apply_positive_only_projection,
    deploy_native_logits,
    select_gt_free,
)


EXPECTED_SCHEMA = "P5B_R0_GT_FREE_CACHE_v1"
PATCH_COUNT = 37 * 37
STAGES = 3
K = 8
SHIFT = (12, 12)


def _shift_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return np.roll(values.reshape(37, 37), SHIFT, axis=(0, 1)).reshape(-1).astype(values.dtype)
    if values.ndim == 2 and values.shape[1] == PATCH_COUNT:
        return np.stack([_shift_map(row) for row in values], axis=0).astype(values.dtype)
    raise ValueError(f"unsupported shift shape={values.shape}")


def _trace_array(items: list[tuple[int, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    if not items:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float64)
    return (
        np.asarray([[i, j] for i, j, _ in items], dtype=np.int64),
        np.asarray([cost for _, _, cost in items], dtype=np.float64),
    )


def _check_trace(arrays: dict[str, np.ndarray], prefix: str, trace: list[tuple[int, int, float]]) -> None:
    expected_pairs, expected_cost = _trace_array(trace)
    actual_pairs = arrays[f"{prefix}_selected_pairs"]
    actual_cost = arrays[f"{prefix}_selected_cost"]
    if not np.array_equal(actual_pairs, expected_pairs):
        raise RuntimeError(f"selector parity failed for {prefix} pairs")
    if not np.array_equal(actual_cost, expected_cost):
        raise RuntimeError(f"selector parity failed for {prefix} costs")
    if actual_pairs.size and np.unique(actual_pairs).size != actual_pairs.size:
        raise RuntimeError(f"repeated patch in {prefix} selected trace")


def _check_action(arrays: dict[str, np.ndarray], selected: list[tuple[int, int, float]]) -> dict[str, int | float | bool]:
    native = arrays["native_logits"]
    m_bar = arrays["m_bar"]
    corrected, delta = apply_positive_only_projection(native, m_bar, selected)
    if corrected.shape != native.shape or not np.all(np.isfinite(corrected)):
        raise RuntimeError("candidate output shape or finiteness failure")
    if np.any(delta < 0) or np.any(~np.isfinite(delta)):
        raise RuntimeError("positive-only delta invariant failed")
    acted = np.flatnonzero(delta > 0)
    selected_ids = np.asarray([p for i, j, _ in selected for p in (i, j)], dtype=np.int64)
    if selected_ids.size and np.unique(selected_ids).size != selected_ids.size:
        raise RuntimeError("one-patch-one-action invariant failed")
    if selected_ids.size != 2 * len(selected):
        raise RuntimeError("action count invariant failed")
    normal_unchanged = np.array_equal(corrected[:, :, 0], native[:, :, 0])
    unrelated_unchanged = np.array_equal(corrected[:, ~np.isin(np.arange(PATCH_COUNT), acted), 1], native[:, ~np.isin(np.arange(PATCH_COUNT), acted), 1])
    if not normal_unchanged or not unrelated_unchanged:
        raise RuntimeError("unrelated/native-normal invariants failed")
    for i, j, cost in selected:
        expected = float(m_bar[j] - m_bar[i])
        if not np.isclose(delta[i], expected, atol=0.0, rtol=0.0):
            raise RuntimeError("positive-only gap mismatch")
        if delta[j] != 0.0 or not np.isclose(cost, expected, atol=0.0, rtol=0.0):
            raise RuntimeError("selected relation cost mismatch")
        if not np.all(corrected[:, i, 1] > native[:, i, 1]):
            raise RuntimeError("anomaly logit uplift missing")
    deployed = deploy_native_logits(corrected)
    if deployed.shape != (1, 2, 518, 518) or not np.all(np.isfinite(deployed)):
        raise RuntimeError("deployment finiteness/shape failure")
    return {
        "finite": True,
        "selected_relations": len(selected),
        "acted_patches": int(acted.size),
        "normal_logits_unchanged": normal_unchanged,
        "unrelated_anomaly_logits_unchanged": unrelated_unchanged,
        "positive_only_delta": True,
        "deployment_finite": True,
    }


def run(cache_root: Path, output: Path) -> dict[str, object]:
    manifest_path = cache_root / "CACHE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError("cache schema mismatch")
    if not manifest.get("finalized") or manifest.get("scientific_unique_image_forwards") != 2162:
        raise RuntimeError("cache is not finalized at 2162 unique forwards")
    if manifest.get("training_steps") != 0:
        raise RuntimeError("cache reports training steps")
    records = []
    action_counts = []
    for key in sorted(manifest["processed_image_keys"]):
        path = cache_root / manifest["files"][key]["relative_path"]
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        required = {
            "m_bar", "D_rank", "valid_reference", "score_bin", "d_rank_bin",
            "E_nonlocal", "E_stage", "E_LOO", "native_logits",
            "aligned_selected_pairs", "aligned_selected_cost",
            "shifted_selected_pairs", "shifted_selected_cost",
        }
        if not required.issubset(arrays):
            raise RuntimeError(f"missing cache arrays for {key}")
        aligned = select_gt_free(
            arrays["m_bar"], arrays["D_rank"], arrays["valid_reference"],
            arrays["E_nonlocal"], arrays["E_stage"], arrays["E_LOO"],
            arrays["score_bin"], arrays["d_rank_bin"],
        )
        _check_trace(arrays, "aligned", aligned["selected"])
        shifted = select_gt_free(
            arrays["m_bar"], arrays["D_rank"], arrays["valid_reference"],
            _shift_map(arrays["E_nonlocal"]), _shift_map(arrays["E_stage"]),
            _shift_map(arrays["E_LOO"]), arrays["score_bin"], arrays["d_rank_bin"],
        )
        _check_trace(arrays, "shifted", shifted["selected"])
        action_counts.append(_check_action(arrays, aligned["selected"]))
        records.append(key)
    result = {
        "schema_version": "P5B_CANDIDATE_REPLAY_v1",
        "status": "PASS",
        "gt_loaded": False,
        "performance_metrics_computed": False,
        "unique_cache_records": len(records),
        "physical_model_forwards": 0,
        "training_steps": 0,
        "aligned_selector_trace_parity": True,
        "shifted_selector_trace_parity": True,
        "operational_invariants": {
            "all_finite": all(x["finite"] for x in action_counts),
            "normal_logits_unchanged": all(x["normal_logits_unchanged"] for x in action_counts),
            "unrelated_anomaly_logits_unchanged": all(x["unrelated_anomaly_logits_unchanged"] for x in action_counts),
            "positive_only_delta": all(x["positive_only_delta"] for x in action_counts),
            "deployment_finite": all(x["deployment_finite"] for x in action_counts),
            "no_repeated_patches": True,
        },
        "action_summary": {
            "images_with_actions": sum(x["selected_relations"] > 0 for x in action_counts),
            "selected_relations": sum(int(x["selected_relations"]) for x in action_counts),
            "acted_patches": sum(int(x["acted_patches"]) for x in action_counts),
        },
        "cache_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cache_root, args.output)


if __name__ == "__main__":
    main()
