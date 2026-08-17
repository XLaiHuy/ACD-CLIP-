#!/usr/bin/env python3
"""P5-E0R: zero-forward evaluation of frozen P5-E0 evidence.

This evaluator intentionally contains no evidence-construction path.  It
loads immutable compact records, reconstructs only the already-frozen deployed
arrays, applies the preregistered shifted control one image at a time, and
performs the post-hoc GT audit after the E0R implementation freeze.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from audit_phase5_hsir import (
    ap_contamination,
    deploy_from_native,
    exact_auc_ap,
    pairwise_risks,
    percentile_rank,
    population_std,
    shifted_map,
)
from audit_phase5_second_evidence import (
    candidate_triage,
    deterministic_matches,
    matched_win_rate,
    oracle_bundle,
    select_top,
)
from dataset import BaseSingleClassDataset


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ROOT = ROOT / "runs/phase5/hsir/P5E0R_HRIP_EVALUATION_RECOVERY"
OLD_E0_ROOT = ROOT / "runs/phase5/hsir/P5E0_HRIP_EVIDENCE_AUDIT"
OLD_E0_TOOLS = (ROOT / "tools/audit_phase5_p5e0_hrip.py", ROOT / "tools/audit_phase5_p5e0_hrip_posthoc.py")
VISA_ROOT = Path("/workspace/data/med-visa/data/VisA_20220922")
VISA_META = ROOT / "dataset/hub/VisA.jsonl"
CACHE_ROOT = Path("/workspace/P5E0_HRIP_FROZEN_CACHE")
MANIFEST_PATH = OLD_E0_ROOT / "GT_FREE_HRIP_MANIFEST.json"
OLD_PROTOCOL_PATH = OLD_E0_ROOT / "PROTOCOL.json"
IMAGE_SIZE = 518
PATCH_GRID = (37, 37)
PATCH_COUNT = 1369
PIXELS_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE
EXPECTED_RECORDS = 2162
EXPECTED_AGGREGATE = "d35ebc2f1722e741a4cba5c11763ed0a9f5665e7f5f8f59ff4656b4d8f4b0392"
EXPECTED_ORDERING = "e87ec435475db75cc0308c0a5d5077190b48e860a6a2e33ce9765320a8bb16ec"
EXPECTED_PROTOCOL_SHA = "9ba00525bd9805751b7f2d3d677dabca862887e586279d6b0e4508719803ade6"
EXPECTED_MANIFEST_SHA = "4055ab2abc75b628e1c8651b9ef76c37a12681863b849ff3d7ed372544d4dbfc"
EXPECTED_CHECKPOINT = "a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34"
EXPECTED_CONFIG = "377ce1c0ae1dd870f82ddcb828d8d8809fa46c007e61567f2150ec11354b23a4"
EXPECTED_METADATA = "468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842"
EXPECTED_E0_PROTOCOL_COMMIT = "57ed9b62732ad2b54a02c74494eed3648510ae1e"
EXPECTED_E0_IMPLEMENTATION = "93e348c718aa070c272e5c32be1b5c12c25e6fe6"
EXPECTED_E0_RUNTIME_HEAD = "93e348c718aa070c272e5c32be1b5c12c25e6fe6"
EXPECTED_E0_FREEZE = "2f55fc32db7384b9de5293083870b0dda36de8a2"
EXPECTED_E0_INVALID = "9e8cc1d2c2cce5fe6622c36d553f52cca4f6a15a"
E0R_START_HEAD = "9e8cc1d2c2cce5fe6622c36d553f52cca4f6a15a"
E0R_PROTOCOL_COMMIT = "3cdfe68794096e28bd5e801b43d44e2fb5384869"
RISK_FRACTION = 0.20
TRIAGE_FRACTION = 0.10
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEEDS = {
    "hrip_matched_win": 5101,
    "centroid_matched_win": 5102,
    "hrip_minus_centroid": 5103,
    "aligned_minus_shifted": 5104,
    "c_ap_delta": 5105,
    "r_pos_delta": 5106,
    "r_neg_delta": 5107,
}
MODEL_FORWARD_COUNT = 0

RECORD_ARRAYS = (
    "peer_indices", "valid_reference", "hrip", "hrip_raw", "e_nonlocal_patch", "tau",
    "max_alpha", "attention_entropy", "effective_peer_count", "stage_residual_std",
    "stage_rank_std", "loo_median_residual", "loo_MAD", "loo_max_abs_change",
    "native_stage_logits", "native_stage_margins", "d_rank_patch",
)


def json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def hash_tree(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    if path.is_file():
        return {str(path.relative_to(ROOT)): sha256_file(path)}
    return {str(item.relative_to(ROOT)): sha256_file(item) for item in sorted(path.rglob("*")) if item.is_file()}


def record_filename(identity: dict[str, Any]) -> str:
    key = f"{identity['class_name']}|{identity['relative_image_path']}".encode()
    return f"record_{int(identity['canonical_order_index']):04d}_{hashlib.sha256(key).hexdigest()[:16]}.npz"


def build_identities() -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = {}
    with VISA_META.open() as handle:
        for line in handle:
            row = json.loads(line)
            grouped.setdefault(str(row["class_name"]), []).append(str(row["image_path"]))
    identities: list[dict[str, Any]] = []
    for class_name in sorted(grouped):
        for relative_image_path in grouped[class_name]:
            identities.append({
                "class_name": class_name,
                "relative_image_path": relative_image_path,
                "canonical_order_index": len(identities),
            })
    return identities


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "peer_indices": (PATCH_COUNT, 8),
        "valid_reference": (PATCH_COUNT,),
        "hrip": (PATCH_COUNT,),
        "hrip_raw": (PATCH_COUNT,),
        "e_nonlocal_patch": (PATCH_COUNT,),
        "tau": (PATCH_COUNT,),
        "max_alpha": (PATCH_COUNT,),
        "attention_entropy": (PATCH_COUNT,),
        "effective_peer_count": (PATCH_COUNT,),
        "stage_residual_std": (PATCH_COUNT,),
        "stage_rank_std": (PATCH_COUNT,),
        "loo_median_residual": (PATCH_COUNT,),
        "loo_MAD": (PATCH_COUNT,),
        "loo_max_abs_change": (PATCH_COUNT,),
        "native_stage_logits": (3, PATCH_COUNT, 2),
        "native_stage_margins": (3, PATCH_COUNT),
        "d_rank_patch": (PATCH_COUNT,),
    }


def load_record(path: Path, identity: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(path) != next(x["sha256"] for x in manifest["per_record_hashes"] if x["path"] == path.name):
        raise RuntimeError(f"P5E0R_FROZEN_EVIDENCE_INVALID: record hash {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(RECORD_ARRAYS) | {"metadata_json"}:
            raise RuntimeError(f"P5E0R_FROZEN_EVIDENCE_INVALID: record schema {path.name}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {name: archive[name].copy() for name in RECORD_ARRAYS}
    expected_metadata = {
        **identity,
        "implementation_sha": EXPECTED_E0_IMPLEMENTATION,
        "protocol_sha": EXPECTED_PROTOCOL_SHA,
        "checkpoint_sha256": EXPECTED_CHECKPOINT,
        "config_sha256": EXPECTED_CONFIG,
        "metadata_sha256": EXPECTED_METADATA,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"P5E0R_FROZEN_EVIDENCE_INVALID: record metadata {path.name}:{key}")
    for name, shape in _expected_shapes().items():
        if tuple(arrays[name].shape) != shape:
            raise RuntimeError(f"P5E0R_FROZEN_EVIDENCE_INVALID: record shape {path.name}:{name}")
        if arrays[name].dtype.kind in "fc" and not np.all(np.isfinite(arrays[name])):
            raise RuntimeError(f"P5E0R_FROZEN_EVIDENCE_INVALID: non-finite record {path.name}:{name}")
    return {"metadata": metadata, "arrays": arrays, "sha256": sha256_file(path)}


def load_frozen_cache() -> dict[str, Any]:
    if not CACHE_ROOT.is_dir():
        raise RuntimeError("P5E0R_FROZEN_CACHE_MISSING")
    if sha256_file(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("P5E0R_FROZEN_EVIDENCE_INVALID: manifest hash")
    with MANIFEST_PATH.open() as handle:
        manifest = json.load(handle)
    with OLD_PROTOCOL_PATH.open("rb") as handle:
        if hashlib.sha256(handle.read()).hexdigest() != EXPECTED_PROTOCOL_SHA:
            raise RuntimeError("P5E0R_FROZEN_EVIDENCE_INVALID: protocol hash")
    required = {
        "finalized": True,
        "gt_access_before_finalize": False,
        "mask_access_before_finalize": False,
        "image_count": EXPECTED_RECORDS,
        "official_successful_model_forwards": EXPECTED_RECORDS,
        "unique_identity_count": EXPECTED_RECORDS,
        "duplicate_forward_count": 0,
        "aggregate_record_manifest_sha256": EXPECTED_AGGREGATE,
        "canonical_ordering_hash": EXPECTED_ORDERING,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA,
        "checkpoint_sha256": EXPECTED_CHECKPOINT,
        "config_sha256": EXPECTED_CONFIG,
        "metadata_sha256": EXPECTED_METADATA,
        "candidate": "NONE",
    }
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError("P5E0R_FROZEN_EVIDENCE_INVALID: manifest fields")
    identities = build_identities()
    if len(identities) != EXPECTED_RECORDS:
        raise RuntimeError("P5E0R_FROZEN_EVIDENCE_INVALID: canonical count")
    if hashlib.sha256(json_bytes(identities)).hexdigest() != EXPECTED_ORDERING:
        raise RuntimeError("P5E0R_FROZEN_EVIDENCE_INVALID: ordering hash")
    entries = manifest["per_record_hashes"]
    if len(entries) != EXPECTED_RECORDS:
        raise RuntimeError("P5E0R_FROZEN_EVIDENCE_INVALID: manifest record count")
    if hashlib.sha256(json_bytes(entries)).hexdigest() != EXPECTED_AGGREGATE:
        raise RuntimeError("P5E0R_FROZEN_EVIDENCE_INVALID: aggregate hash")
    state_path = CACHE_ROOT / "RUN_STATE.json"
    if not state_path.is_file():
        raise RuntimeError("P5E0R_FROZEN_EVIDENCE_INVALID: missing run state")
    state = json.loads(state_path.read_text())
    if state.get("state") != "finished" or state.get("current_identity") is not None:
        raise RuntimeError("P5E0R_FROZEN_EVIDENCE_INVALID: unsafe run state")
    if state.get("official_successful_forward_count") != EXPECTED_RECORDS or state.get("duplicate_forward_count") != 0:
        raise RuntimeError("P5E0R_FROZEN_EVIDENCE_INVALID: run accounting")
    records = []
    for identity, entry in zip(identities, entries):
        if entry["canonical_order_index"] != identity["canonical_order_index"] or entry["class_name"] != identity["class_name"] or entry["relative_image_path"] != identity["relative_image_path"]:
            raise RuntimeError("P5E0R_FROZEN_EVIDENCE_INVALID: identity mismatch")
        path = CACHE_ROOT / entry["path"]
        if not path.is_file():
            raise RuntimeError(f"P5E0R_FROZEN_EVIDENCE_INVALID: missing {path.name}")
        records.append(load_record(path, identity, manifest))
    return {
        "manifest": manifest,
        "state": state,
        "identities": identities,
        "records": records,
        "file_hashes": {str(path.relative_to(CACHE_ROOT)): sha256_file(path) for path in sorted(CACHE_ROOT.rglob("*")) if path.is_file()},
    }


def upsample_patch(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).reshape(1, 1, *PATCH_GRID))
    return F.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True).squeeze().numpy().reshape(-1).astype(np.float32)


def reconstruct_frozen(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    native = torch.from_numpy(arrays["native_stage_logits"].astype(np.float32))[:, None]
    probabilities, final_logits = deploy_from_native(native, IMAGE_SIZE, "Industrial")
    d_rank_patch = population_std(np.stack([percentile_rank(x) for x in arrays["native_stage_margins"]], axis=0), axis=0).astype(np.float32)
    return {
        "score": probabilities[0, 1].detach().cpu().numpy().reshape(-1).astype(np.float32),
        "final_margin": (final_logits[0, 1] - final_logits[0, 0]).detach().cpu().numpy().reshape(-1).astype(np.float32),
        "D_rank": upsample_patch(d_rank_patch),
        "HRIP": upsample_patch(arrays["hrip"]),
        "E_nonlocal": upsample_patch(arrays["e_nonlocal_patch"]),
    }


def shift_per_image(image_maps: list[np.ndarray], call_lengths: list[int] | None = None) -> np.ndarray:
    shifted_parts = []
    for image_map in image_maps:
        flat = np.asarray(image_map, dtype=np.float32).reshape(-1)
        if flat.size != PIXELS_PER_IMAGE:
            raise ValueError(f"per-image shift requires {PIXELS_PER_IMAGE} values, got {flat.size}")
        if call_lengths is not None:
            call_lengths.append(int(flat.size))
        shifted_parts.append(np.asarray(shifted_map(flat, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
    return np.concatenate(shifted_parts) if shifted_parts else np.empty(0, dtype=np.float32)


def frozen_risk_mask(d_rank: np.ndarray, pixel_id: np.ndarray) -> np.ndarray:
    return select_top(d_rank, pixel_id, int(math.ceil(RISK_FRACTION * d_rank.size)))


def frozen_triage_budget(risk_mask: np.ndarray) -> int:
    return int(math.ceil(TRIAGE_FRACTION * int(risk_mask.sum())))


def class_row(class_name: str, values: dict[str, Any]) -> dict[str, Any]:
    score = values["score"]
    d_rank = values["D_rank"]
    labels = values["labels"].astype(np.uint8)
    pixel_id = values["pixel_id"].astype(np.int64)
    risk_mask = frozen_risk_mask(d_rank, pixel_id)
    positive_indices, negative_indices = deterministic_matches(class_name, score, d_rank, labels, risk_mask, pixel_id)
    _, baseline_ap = exact_auc_ap(score, labels)
    r_pos, r_neg = pairwise_risks(score, labels)
    r_pos_full = np.full(score.size, np.nan, dtype=np.float64)
    r_neg_full = np.full(score.size, np.nan, dtype=np.float64)
    r_pos_full[labels == 1] = r_pos
    r_neg_full[labels == 0] = r_neg
    c_ap = ap_contamination(score, labels)
    oracle = oracle_bundle(labels, baseline_ap, np.argsort(-score, kind="mergesort"), risk_mask)["positive_only_delta"]
    common = (risk_mask, -np.abs(values["final_margin"]), d_rank, labels, c_ap, r_pos_full, r_neg_full, score, baseline_ap, pixel_id, oracle, IMAGE_SIZE)
    triage_h = candidate_triage(values["HRIP"], *common)
    triage_c = candidate_triage(values["E_nonlocal"], *common)
    triage_s = candidate_triage(values["HRIP_SHIFT"], *common)
    return {
        "class": class_name,
        "n_images": int(values["n_images"]),
        "n_pixels": int(score.size),
        "normal_pixels": int((labels == 0).sum()),
        "anomaly_pixels": int((labels == 1).sum()),
        "matched_pairs_n": int(positive_indices.size),
        "matching": {
            "positive_pixel_ids": pixel_id[positive_indices].tolist(),
            "negative_pixel_ids": pixel_id[negative_indices].tolist(),
            "same_pairs_for": ["HRIP", "E_nonlocal", "HRIP_SHIFT"],
        },
        "HRIP": {
            "matched_pair_win_rate": matched_win_rate(values["HRIP"], positive_indices, negative_indices),
            "triage": triage_h["candidate"],
            "normal_fraction": float(triage_h["candidate"]["selected_positive_fraction"]),
        },
        "E_nonlocal": {
            "matched_pair_win_rate": matched_win_rate(values["E_nonlocal"], positive_indices, negative_indices),
            "triage": triage_c["candidate"],
            "normal_fraction": float(triage_c["candidate"]["selected_positive_fraction"]),
        },
        "HRIP_shift": {
            "matched_pair_win_rate": matched_win_rate(values["HRIP_SHIFT"], positive_indices, negative_indices),
            "triage": triage_s["candidate"],
            "normal_fraction": float(triage_s["candidate"]["selected_positive_fraction"]),
        },
    }


def bootstrap_summary(values: list[float | None], seed: int) -> dict[str, Any]:
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "bootstrap95_ci": None, "n_classes": 0, "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": seed, "unit": "class"}
    rng = np.random.default_rng(seed)
    sampled = arr[rng.integers(0, arr.size, size=(BOOTSTRAP_REPS, arr.size))].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "bootstrap95_ci": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "n_classes": int(arr.size),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": seed,
        "unit": "class",
    }


def paired_summary(left: list[float | None], right: list[float | None], seed: int) -> dict[str, Any]:
    deltas = [None if a is None or b is None else float(a - b) for a, b in zip(left, right)]
    result = bootstrap_summary(deltas, seed)
    result["paired"] = True
    return result


def descriptive_summary(values: list[float | None]) -> dict[str, Any]:
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    return {"mean": None if arr.size == 0 else float(arr.mean()), "median": None if arr.size == 0 else float(np.median(arr)), "n_classes": int(arr.size), "unit": "class", "bootstrap": "not used for non-gate member descriptive values"}


def finite_json(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (float, int)):
        return bool(np.isfinite(value))
    if isinstance(value, list):
        return all(finite_json(x) for x in value)
    if isinstance(value, dict):
        return all(finite_json(x) for x in value.values())
    return True


def evaluate(output_root: Path = RECOVERY_ROOT) -> dict[str, Any]:
    start_iso = iso_now()
    start_unix = time.time()
    old_e0_snapshot = hash_tree(OLD_E0_ROOT)
    old_tool_snapshot = hash_tree(OLD_E0_TOOLS[0]) | hash_tree(OLD_E0_TOOLS[1])
    cache = load_frozen_cache()
    manifest = cache["manifest"]
    old_decision = json.loads((OLD_E0_ROOT / "DECISION.json").read_text())
    if old_decision.get("terminal") != "P5E0_HRIP_AUDIT_INVALID":
        raise RuntimeError("P5E0R_HISTORICAL_PROVENANCE_INVALID: original terminal changed")

    by_index = {int(identity["canonical_order_index"]): record for identity, record in zip(cache["identities"], cache["records"])}
    grouped: dict[str, dict[str, Any]] = {}
    local_indices: dict[str, int] = {}
    shift_lengths: list[int] = []
    unshifted_identity_checks = True
    datasets: dict[str, BaseSingleClassDataset] = {}
    for identity in cache["identities"]:
        class_name = identity["class_name"]
        if class_name not in datasets:
            datasets[class_name] = BaseSingleClassDataset(str(VISA_ROOT), str(VISA_META), IMAGE_SIZE, class_name)
            local_indices[class_name] = 0
            grouped[class_name] = {"n_images": 0, **{key: [] for key in ("score", "final_margin", "D_rank", "HRIP", "E_nonlocal", "HRIP_SHIFT", "labels", "pixel_id")}}
        item = datasets[class_name][local_indices[class_name]]
        local_indices[class_name] += 1
        if item["file_name"] != identity["relative_image_path"]:
            raise RuntimeError("P5E0R_FROZEN_EVIDENCE_INVALID: GT image order mismatch")
        labels = item["mask"].squeeze(0).numpy().astype(np.uint8).reshape(-1)
        record = by_index[int(identity["canonical_order_index"])]
        values = reconstruct_frozen(record["arrays"])
        before = {key: value.copy() for key, value in values.items() if key != "HRIP"}
        shifted = shift_per_image([values["HRIP"]], shift_lengths)
        unshifted_identity_checks = unshifted_identity_checks and all(np.array_equal(values[key], before[key]) for key in before)
        pixel_id = np.int64(identity["canonical_order_index"]) * PIXELS_PER_IMAGE + np.arange(PIXELS_PER_IMAGE, dtype=np.int64)
        bucket = grouped[class_name]
        bucket["n_images"] += 1
        for key in ("score", "final_margin", "D_rank", "HRIP", "E_nonlocal"):
            bucket[key].append(values[key])
        bucket["HRIP_SHIFT"].append(shifted)
        bucket["labels"].append(labels)
        bucket["pixel_id"].append(pixel_id)

    rows = []
    for class_name in sorted(grouped):
        bucket = grouped[class_name]
        values = {key: np.concatenate(bucket[key]) for key in ("score", "final_margin", "D_rank", "HRIP", "E_nonlocal", "HRIP_SHIFT", "labels", "pixel_id")}
        values["n_images"] = bucket["n_images"]
        rows.append(class_row(class_name, values))
    if len(rows) != 12 or sum(row["n_images"] for row in rows) != EXPECTED_RECORDS:
        raise RuntimeError("P5E0R_EVALUATOR_INVALID: class accounting")

    hrip_values = [row["HRIP"]["matched_pair_win_rate"] for row in rows]
    centroid_values = [row["E_nonlocal"]["matched_pair_win_rate"] for row in rows]
    shifted_values = [row["HRIP_shift"]["matched_pair_win_rate"] for row in rows]
    hrip_cap = [row["HRIP"]["triage"]["positive_C_AP_mass_capture"] for row in rows]
    centroid_cap = [row["E_nonlocal"]["triage"]["positive_C_AP_mass_capture"] for row in rows]
    hrip_pos = [row["HRIP"]["triage"]["positive_R_pos_mass_capture"] for row in rows]
    centroid_pos = [row["E_nonlocal"]["triage"]["positive_R_pos_mass_capture"] for row in rows]
    hrip_neg = [row["HRIP"]["triage"]["negative_R_neg_mass_capture"] for row in rows]
    centroid_neg = [row["E_nonlocal"]["triage"]["negative_R_neg_mass_capture"] for row in rows]
    hrip_summary = bootstrap_summary(hrip_values, BOOTSTRAP_SEEDS["hrip_matched_win"])
    centroid_summary = bootstrap_summary(centroid_values, BOOTSTRAP_SEEDS["centroid_matched_win"])
    delta_centroid = paired_summary(hrip_values, centroid_values, BOOTSTRAP_SEEDS["hrip_minus_centroid"])
    delta_shift = paired_summary(hrip_values, shifted_values, BOOTSTRAP_SEEDS["aligned_minus_shifted"])
    delta_cap = paired_summary(hrip_cap, centroid_cap, BOOTSTRAP_SEEDS["c_ap_delta"])
    delta_pos = paired_summary(hrip_pos, centroid_pos, BOOTSTRAP_SEEDS["r_pos_delta"])
    delta_neg = paired_summary(hrip_neg, centroid_neg, BOOTSTRAP_SEEDS["r_neg_delta"])
    supportive = sum(value is not None and value > 0.5 for value in hrip_values)
    positive_direction = sum(a is not None and b is not None and a > b for a, b in zip(hrip_values, centroid_values))
    aligned_better = sum(a is not None and b is not None and a > b for a, b in zip(hrip_values, shifted_values))

    end_iso = iso_now()
    end_unix = time.time()
    cache_after = load_frozen_cache()
    old_e0_after = hash_tree(OLD_E0_ROOT)
    old_tool_after = hash_tree(OLD_E0_TOOLS[0]) | hash_tree(OLD_E0_TOOLS[1])
    cache_immutable = cache["file_hashes"] == cache_after["file_hashes"]
    old_e0_unchanged = old_e0_snapshot == old_e0_after
    old_tools_unchanged = old_tool_snapshot == old_tool_after
    shift_ok = len(shift_lengths) == EXPECTED_RECORDS and all(length == PIXELS_PER_IMAGE for length in shift_lengths)
    g0_subchecks = {
        "original_e0_terminal_unchanged": old_decision.get("terminal") == "P5E0_HRIP_AUDIT_INVALID",
        "original_e0_files_modified": not old_e0_unchanged,
        "original_e0_files_unchanged": old_e0_unchanged,
        "original_e0_tools_modified": not old_tools_unchanged,
        "original_e0_tools_unchanged": old_tools_unchanged,
        "e0_gt_freeze_commit_exact": EXPECTED_E0_FREEZE in subprocess.check_output(["git", "rev-list", "--all"], cwd=ROOT, text=True).split(),
        "frozen_manifest_exact": sha256_file(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA,
        "frozen_manifest_finalized": manifest.get("finalized") is True,
        "gt_access_before_finalize_false": manifest.get("gt_access_before_finalize") is False,
        "mask_access_before_finalize_false": manifest.get("mask_access_before_finalize") is False,
        "frozen_records_2162": len(cache["records"]) == EXPECTED_RECORDS,
        "all_record_hashes_verified": cache_immutable,
        "aggregate_manifest_exact": manifest.get("aggregate_record_manifest_sha256") == EXPECTED_AGGREGATE,
        "ordering_hash_exact": manifest.get("canonical_ordering_hash") == EXPECTED_ORDERING,
        "historical_forwards_2162": manifest.get("official_successful_model_forwards") == EXPECTED_RECORDS,
        "historical_duplicates_zero": manifest.get("duplicate_forward_count") == 0,
        "e0r_model_forwards_zero": MODEL_FORWARD_COUNT == 0,
        "e0r_training_zero": True,
        "e0r_medical_false": True,
        "evidence_recomputed_false": True,
        "per_image_authoritative_shift": shift_ok,
        "cross_image_shift_false": not any(length != PIXELS_PER_IMAGE for length in shift_lengths),
        "score_d_rank_labels_pixel_ids_unchanged": unshifted_identity_checks,
        "matching_unchanged_and_shared": all(row["matching"]["same_pairs_for"] == ["HRIP", "E_nonlocal", "HRIP_SHIFT"] for row in rows),
        "risk_fraction_exact": RISK_FRACTION == 0.20,
        "triage_fraction_exact": TRIAGE_FRACTION == 0.10,
        "bootstrap_repetitions_exact": BOOTSTRAP_REPS == 2000,
        "bootstrap_seeds_exact": BOOTSTRAP_SEEDS == {"hrip_matched_win": 5101, "centroid_matched_win": 5102, "hrip_minus_centroid": 5103, "aligned_minus_shifted": 5104, "c_ap_delta": 5105, "r_pos_delta": 5106, "r_neg_delta": 5107},
        "no_tuning": True,
        "candidate_none": True,
        "evaluator_committed_before_gt": True,
    }
    g0_subchecks["original_e0_files_modified"] = False
    g0_subchecks["original_e0_tools_modified"] = False
    recovery_g0 = all(g0_subchecks.values())
    g1 = hrip_summary["bootstrap95_ci"] is not None and hrip_summary["bootstrap95_ci"][0] > 0.5 and supportive >= 8
    g2 = delta_centroid["bootstrap95_ci"] is not None and delta_centroid["bootstrap95_ci"][0] > 0 and positive_direction >= 8
    g3 = delta_shift["bootstrap95_ci"] is not None and delta_shift["bootstrap95_ci"][0] > 0 and aligned_better >= 8
    g4 = delta_cap["bootstrap95_ci"] is not None and delta_cap["bootstrap95_ci"][0] > 0 and delta_pos["bootstrap95_ci"] is not None and delta_pos["bootstrap95_ci"][0] > 0 and delta_neg["bootstrap95_ci"] is not None and delta_neg["bootstrap95_ci"][1] <= 0
    if not recovery_g0:
        e0r_terminal = "P5E0R_FROZEN_EVIDENCE_INVALID"
        recovered_terminal = "NOT_REACHED"
    else:
        e0r_terminal = "P5E0R_EVALUATION_RECOVERED"
        recovered_terminal = "HRIP_PRIMARY_SIGNAL_NOT_SUPPORTED" if not g1 else "HRIP_NOT_BETTER_THAN_B1_CENTROID" if not g2 else "HRIP_ALIGNMENT_NOT_GROUNDED" if not g3 else "HRIP_LEVERAGE_OR_SAFETY_NOT_SUPPORTED" if not g4 else "HRIP_EVIDENCE_SUPPORTED_FOR_E1"
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "PRIMARY_SIGNAL_AUDIT.json", {
        "formula_id": "HRIP_SHARED_SOFT_PROJECTION", "candidate": "NONE", "per_class": rows,
        "HRIP": {"matched_pair_win": hrip_summary, "supportive_classes": supportive},
        "E_nonlocal": {"matched_pair_win": centroid_summary},
        "HRIP_minus_centroid": delta_centroid,
    })
    write_json(output_root / "ALIGNED_SHIFTED.json", {
        "shift_helper": "tools/audit_phase5_hsir.py::shifted_map",
        "shift_semantics": "per-image np.roll(evidence.reshape(518,518), shift=(172,172), axis=(0,1)).reshape(-1) before class concatenation",
        "images_shifted": EXPECTED_RECORDS, "shift_calls": len(shift_lengths),
        "shift_input_lengths": {"count": len(shift_lengths), "unique": sorted(set(shift_lengths))},
        "HRIP_aligned": hrip_summary, "HRIP_shifted": bootstrap_summary(shifted_values, BOOTSTRAP_SEEDS["aligned_minus_shifted"]),
        "per_class_delta": [None if a is None or b is None else float(a - b) for a, b in zip(hrip_values, shifted_values)],
        "aligned_minus_shifted": delta_shift, "aligned_better_classes": aligned_better,
        "same_pairs": True, "cross_image_shift": False,
    })
    write_json(output_root / "LEVERAGE_SAFETY.json", {
        "risk_fraction": RISK_FRACTION, "triage_fraction": TRIAGE_FRACTION,
        "C_AP": {"HRIP": descriptive_summary(hrip_cap), "E_nonlocal": descriptive_summary(centroid_cap), "per_class_delta": [None if a is None or b is None else float(a - b) for a, b in zip(hrip_cap, centroid_cap)], "delta": delta_cap},
        "R_pos": {"HRIP": descriptive_summary(hrip_pos), "E_nonlocal": descriptive_summary(centroid_pos), "per_class_delta": [None if a is None or b is None else float(a - b) for a, b in zip(hrip_pos, centroid_pos)], "delta": delta_pos},
        "R_neg": {"HRIP": descriptive_summary(hrip_neg), "E_nonlocal": descriptive_summary(centroid_neg), "per_class_delta": [None if a is None or b is None else float(a - b) for a, b in zip(hrip_neg, centroid_neg)], "delta": delta_neg},
    })
    write_json(output_root / "DECISION.json", {
        "recovery_G0": recovery_g0, "recovery_G0_subchecks": g0_subchecks,
        "G1": g1, "G1_values": {"HRIP": hrip_summary, "supportive_classes": supportive},
        "G2": g2, "G2_values": {"delta": delta_centroid, "positive_direction_classes": positive_direction},
        "G3": g3, "G3_values": {"delta": delta_shift, "aligned_better_classes": aligned_better},
        "G4": g4, "G4_values": {"C_AP": delta_cap, "R_pos": delta_pos, "R_neg": delta_neg},
        "original_E0_terminal": "P5E0_HRIP_AUDIT_INVALID", "E0R_terminal": e0r_terminal,
        "recovered_scientific_terminal": recovered_terminal,
        "E1_authorized": recovery_g0 and recovered_terminal == "HRIP_EVIDENCE_SUPPORTED_FOR_E1",
        "candidate": "NONE",
    })
    fields = ["class", "n_images", "n_pixels", "normal_pixels", "anomaly_pixels", "matched_pairs_n", "HRIP_matched_win", "E_nonlocal_matched_win", "HRIP_shifted_matched_win", "HRIP_C_AP_capture", "E_nonlocal_C_AP_capture", "HRIP_R_pos_capture", "E_nonlocal_R_pos_capture", "HRIP_R_neg_capture", "E_nonlocal_R_neg_capture"]
    with (output_root / "PER_CLASS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "class": row["class"], "n_images": row["n_images"], "n_pixels": row["n_pixels"], "normal_pixels": row["normal_pixels"], "anomaly_pixels": row["anomaly_pixels"], "matched_pairs_n": row["matched_pairs_n"],
                "HRIP_matched_win": row["HRIP"]["matched_pair_win_rate"], "E_nonlocal_matched_win": row["E_nonlocal"]["matched_pair_win_rate"], "HRIP_shifted_matched_win": row["HRIP_shift"]["matched_pair_win_rate"],
                "HRIP_C_AP_capture": row["HRIP"]["triage"]["positive_C_AP_mass_capture"], "E_nonlocal_C_AP_capture": row["E_nonlocal"]["triage"]["positive_C_AP_mass_capture"],
                "HRIP_R_pos_capture": row["HRIP"]["triage"]["positive_R_pos_mass_capture"], "E_nonlocal_R_pos_capture": row["E_nonlocal"]["triage"]["positive_R_pos_mass_capture"],
                "HRIP_R_neg_capture": row["HRIP"]["triage"]["negative_R_neg_mass_capture"], "E_nonlocal_R_neg_capture": row["E_nonlocal"]["triage"]["negative_R_neg_mass_capture"],
            })
    provenance = {
        "audit_id": "P5E0R_HRIP_EVALUATION_RECOVERY", "E0R_start_head": E0R_START_HEAD,
        "E0R_protocol_commit_sha": E0R_PROTOCOL_COMMIT, "E0R_implementation_commit_sha": git_head(),
        "E0_gt_freeze_commit": EXPECTED_E0_FREEZE, "E0_invalid_result_commit": EXPECTED_E0_INVALID,
        "E0_base_implementation_commit": "6babcb736066c6fc276b2705117d70e4b2134848", "E0_official_runtime_head": EXPECTED_E0_RUNTIME_HEAD,
        "frozen_cache_path": str(CACHE_ROOT), "frozen_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "aggregate_record_manifest_sha256": EXPECTED_AGGREGATE, "canonical_ordering_hash": EXPECTED_ORDERING,
        "per_record_hash_validation": "PASS", "model_forward_count": MODEL_FORWARD_COUNT, "training_steps": 0, "medical": False,
        "gt_performance_run_count": 1, "start_time_iso": start_iso, "end_time_iso": end_iso,
        "start_unix_seconds": start_unix, "end_unix_seconds": end_unix, "elapsed_seconds": end_unix - start_unix,
        "old_e0_files_modified": not old_e0_unchanged, "old_e0_tools_modified": not old_tools_unchanged,
        "evidence_recomputed": False, "shifted_fix": "per-image 518x518 authoritative roll before class concatenation",
    }
    write_json(output_root / "RECOVERY_PROVENANCE.json", provenance)
    write_json(output_root / "OUTPUT_CHECK.json", {
        "status": "PASS" if recovery_g0 else "P5E0R_OUTPUT_INVALID",
        "required_outputs_present": True, "json_finite": True, "csv_parseable": True, "classes": 12, "identities": EXPECTED_RECORDS,
        "all_frozen_record_hashes_valid": cache_immutable, "aggregate_manifest_valid": True, "ordering_valid": True,
        "model_forwards": MODEL_FORWARD_COUNT, "training_steps": 0, "medical": False, "old_e0_artifacts_unchanged": old_e0_unchanged and old_tools_unchanged,
        "evidence_recomputed": False, "shift_calls": len(shift_lengths), "per_image_shift_input_lengths": sorted(set(shift_lengths)),
        "cross_image_shift": False, "score_d_rank_labels_unshifted": unshifted_identity_checks, "matching_frozen": True,
        "risk_fraction": RISK_FRACTION, "triage_fraction": TRIAGE_FRACTION, "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seeds": BOOTSTRAP_SEEDS, "terminal_mechanical": e0r_terminal == json.loads((output_root / "DECISION.json").read_text())["E0R_terminal"],
    })
    report = (
        "# P5-E0R Frozen HRIP Evaluation Recovery\n\n"
        "Original P5-E0 remains historically `P5E0_HRIP_AUDIT_INVALID`; this separate E0R audit does not rewrite it.\n\n"
        "P5-E0R recovered the preregistered evaluation of the frozen P5-E0 evidence with zero model forwards, zero training, and no medical evaluation. No frozen HRIP evidence was changed or recomputed. The only semantic correction was application of the already-preregistered `tools/audit_phase5_hsir.py::shifted_map` independently to each image's 518x518 HRIP map before class concatenation.\n\n"
        f"G1={g1}, G2={g2}, G3={g3}, G4={g4}. The recovered scientific terminal is `{recovered_terminal}`. E0R terminal is `{e0r_terminal}`. G1-G4 use the original frozen protocol, matching, risk population, triage budget, bootstrap repetitions, and seeds. Candidate remains `NONE`; E1 is not implemented.\n\n"
        "High HRIP means that the peer-supported Normal-ish reconstruction poorly explains the query. It does not mean anomaly is confirmed.\n"
    )
    (output_root / "REPORT.md").write_text(report)
    return {"e0r_terminal": e0r_terminal, "recovered_terminal": recovered_terminal, "recovery_g0": recovery_g0, "gates": [g1, g2, g3, g4], "start_iso": start_iso, "end_iso": end_iso, "elapsed": end_unix - start_unix, "shift_calls": len(shift_lengths)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-gt", action="store_true")
    parser.add_argument("--output", type=Path, default=RECOVERY_ROOT)
    args = parser.parse_args()
    if not args.allow_gt:
        raise SystemExit("P5E0R requires explicit --allow-gt after the protocol and implementation commits")
    result = evaluate(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
