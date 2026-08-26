"""Execute the one-shot, zero-training P29R1 fast objective forensic."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from model.phase2b_runtime import deploy_native_logits
from tools.sabra.data import read_visa_metadata
from tools.sabra_car.r0_direction import classify_actions
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p28_mechanism_diagnostic import (
    _load_masks,
    enforce_data_firewall,
    patch_correction_from_actions,
)
from tools.sabra_v2.p29_objective import p29_sign_guarded_loss
from tools.sabra_v2.p29r1_forensic import (
    CLASS_NAMES,
    estimate_forensic_runtime,
    forensic_utility_for_batch,
    gradient_summary,
    normal_guard_conflict,
    select_probe_source_classes,
    sign_alignment,
    vectorized_pixel_shifts,
)
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import sha256_file, stable_sample_id
from tools.sabra_v2.region_pool import pool_patch_map, symmetric_margin_delta, upsample_region_map
from utils import calculate_seg_loss


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SIZE = 518
PATCH_GRID = (37, 37)
REGION_GRID = (9, 9)
STAGES = 3
PATCH_COUNT = PATCH_GRID[0] * PATCH_GRID[1]
P29_ROOT = Path("/workspace/p29_science_v1")
P27_ROOT = Path("/workspace/p27r1_science_v1")
OUTPUT_ROOT = ROOT / "research/sabra_v2/region_distill"
P29_TERMINAL_SHA = "7eeee454538cb997496f8cd1107f66fa73a9c876"
P29_EXECUTION_BASE_SHA = "24135127c246d024636ec752c656e9bb828f8cdf"
P29_SCHEMA = "P29_IMMUTABLE_HELD_PREDICTIONS_V1"
P27_SCHEMA = "P27_IMMUTABLE_HELD_PREDICTIONS_V1"
P29_ADAPTER_SCHEMA = "P29_REGION_ADAPTER_CHECKPOINT_V1"
P27_ADAPTER_SCHEMA = "P27_REGION_ADAPTER_CHECKPOINT_V1"
PREDICTION_PARITY_TOLERANCE = 2e-5
P27_SIGN_AGREEMENT = 0.5228332297
FORENSIC_OUTPUTS = (
    "P29R1_FORENSIC_ATTEMPT.json",
    "P29R1_FORENSIC_METRICS.json",
    "P29R1_GRADIENT_DIAGNOSTIC.json",
    "P29R1_SIGN_ALIGNMENT.csv",
    "P29R1_NORMALITY_DIAGNOSTIC.json",
    "P29R1_DECISION_TREE.json",
    "P29R1_POST_RUN_AUDIT.json",
    "P29R1_FINAL_REPORT.md",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(_jsonify(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _jsonify(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _remote_head(branch: str) -> str:
    output = _git("ls-remote", "--heads", "origin", branch)
    fields = output.split()
    if len(fields) != 2 or fields[1] != f"refs/heads/{branch}":
        raise RuntimeError(f"remote branch is missing or ambiguous: {branch}")
    return fields[0]


def _protocol_payload(path: Path) -> dict[str, Any]:
    protocol = _read_json(path)
    if protocol.get("schema_version") != "P29R1_FAST_OBJECTIVE_TRANSFER_FORENSIC_V1":
        raise RuntimeError("P29R1 protocol schema drift")
    if protocol.get("status") != "P29R1_PREREGISTERED":
        raise RuntimeError("P29R1 protocol is not preregistered")
    identity = protocol.get("identity", {})
    if identity.get("parent_terminal_sha") != P29_TERMINAL_SHA:
        raise RuntimeError("P29R1 parent terminal SHA drift")
    if identity.get("parent_execution_base_sha") != P29_EXECUTION_BASE_SHA:
        raise RuntimeError("P29R1 parent execution-base SHA drift")
    if identity.get("branch") != "research/p29r1-fast-objective-forensic-v1":
        raise RuntimeError("P29R1 branch identity drift")
    prohibitions = protocol.get("hard_prohibitions", {})
    required_prohibitions = (
        "training", "optimizer_step", "p29_or_p27_rerun", "new_clip_forward",
        "new_phase2b_forward", "mvtec_reads", "medical_reads",
        "normal_anomaly_pair_matrix", "p30_implementation",
    )
    if any(prohibitions.get(key) is not True for key in required_prohibitions):
        raise RuntimeError("P29R1 hard-prohibition contract drift")
    if protocol.get("execution", {}).get("exactly_one_attempt_marker") != (
        "P29R1_FORENSIC_ATTEMPT.json immediately before first real forensic computation"
    ):
        raise RuntimeError("P29R1 attempt-marker contract drift")
    return protocol


def validate_runner_execution_contract() -> None:
    """Fail closed if this runner grows a training or model-forward path."""
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = (
        "torch" + ".optim",
        "optimizer" + ".step",
        "optimizer" + ".zero_grad",
        "forward" + "_phase2b",
        "build" + "_phase2b",
        "clip" + ".load",
        "MV" + "Tec",
        "Med" + "ical",
    )
    found = [token for token in forbidden if token in source]
    if found:
        raise RuntimeError(f"forbidden P29R1 runner path: {found}")


def _validate_cache_manifest(
    path: Path,
    *,
    tier: str,
    expected_class: str,
    expected_ids: Sequence[str],
) -> dict[str, Any]:
    manifest = _read_json(path)
    expected_schema = (
        "P27_TIER_A_FROZEN_FEATURES_V1" if tier == "A" else "P27_TIER_B_SOURCE_SUPERVISION_V1"
    )
    if manifest.get("schema") != expected_schema or manifest.get("completion_status") != "COMPLETE":
        raise RuntimeError(f"cache manifest is incomplete or has the wrong schema: {path}")
    if tier == "A":
        if manifest.get("class") != expected_class:
            raise RuntimeError(f"Tier-A class mismatch: {path}")
        if manifest.get("contains_gt") or manifest.get("contains_masks") or manifest.get("contains_teacher_targets"):
            raise RuntimeError(f"Tier-A is not GT-free: {path}")
    else:
        if manifest.get("held_class") != expected_class or manifest.get("held_mask_reads") != 0:
            raise RuntimeError(f"Tier-B held-class or mask firewall mismatch: {path}")
        if set(manifest.get("source_classes", [])) != set(CLASS_NAMES) - {expected_class}:
            raise RuntimeError(f"Tier-B source-class inventory mismatch: {path}")
    observed_ids = manifest.get("sample_ids")
    if observed_ids != list(expected_ids) or len(set(observed_ids or [])) != len(expected_ids):
        raise RuntimeError(f"cache sample identity/order mismatch: {path}")
    if int(manifest.get("sample_count", -1)) != len(expected_ids):
        raise RuntimeError(f"cache sample count mismatch: {path}")
    expected_shapes = {
        "seg_features": (STAGES, PATCH_COUNT, 768),
        "native_logits": (STAGES, PATCH_COUNT, 2),
    } if tier == "A" else {
        "source_mask": (1, IMAGE_SIZE, IMAGE_SIZE),
        "teacher_region": REGION_GRID,
    }
    tensors = manifest.get("tensors", {})
    for name, sample_shape in expected_shapes.items():
        spec = tensors.get(name, {})
        array_path = path.parent / f"{name}.npy"
        if not array_path.is_file() or spec.get("sample_shape") != list(sample_shape) or spec.get("dtype") != "float32":
            raise RuntimeError(f"cache tensor contract mismatch: {array_path}")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != (len(expected_ids), *sample_shape):
            raise RuntimeError(f"cache tensor shape mismatch: {array_path}")
    return manifest


def _fold_map(audit: Mapping[str, Any], path: Path, key: str) -> dict[str, Mapping[str, Any]]:
    rows = audit.get(key)
    if not isinstance(rows, list):
        raise RuntimeError(f"fold inventory missing from {path}")
    result = {str(row.get("held_class")): row for row in rows if isinstance(row, Mapping)}
    if set(result) != set(CLASS_NAMES):
        raise RuntimeError(f"12-class fold inventory mismatch: {path}")
    return result


def _artifact_inventory(cache_root: Path, metadata: Path) -> dict[str, Any]:
    """Validate and hash all reusable frozen inputs without opening images/masks."""
    rows = read_visa_metadata(metadata)
    p29_audit_path = OUTPUT_ROOT / "P29_POST_RUN_AUDIT.json"
    p27_audit_path = OUTPUT_ROOT / "P27R1_POST_RUN_AUDIT.json"
    p29_audit = _read_json(p29_audit_path)
    p27_audit = _read_json(p27_audit_path)
    p29_folds = _fold_map(p29_audit.get("scientific_runner_audit", {}), p29_audit_path, "folds")
    p27_folds = _fold_map(p27_audit, p27_audit_path, "folds")
    result: dict[str, Any] = {
        "metadata_sha256": sha256_file(metadata),
        "p29_predictions": {},
        "p29_checkpoints": {},
        "p29_scores": {},
        "p27_predictions": {},
        "p27_checkpoints": {},
        "tier_a_manifests": {},
        "tier_b_manifests": {},
    }
    for name in CLASS_NAMES:
        inventory = loco_inventory(rows, name)
        held_ids = [stable_sample_id(row) for row in inventory.held_rows]
        fit_ids = [stable_sample_id(row) for row in inventory.fit_rows]
        tier_a = cache_root / "tier_a" / name / "manifest.json"
        tier_b = cache_root / "tier_b" / name / "manifest.json"
        _validate_cache_manifest(tier_a, tier="A", expected_class=name, expected_ids=held_ids)
        _validate_cache_manifest(tier_b, tier="B", expected_class=name, expected_ids=fit_ids)
        p29_prediction = P29_ROOT / name / "predictions/p29_held_predictions.pt"
        p29_checkpoint = P29_ROOT / name / "training/p29_region_adapter.pt"
        p29_score = P29_ROOT / name / "metrics/p29_held_metrics.json"
        p27_prediction = P27_ROOT / name / "predictions/p27_held_predictions.pt"
        p27_checkpoint = P27_ROOT / name / "training/p27_region_adapter.pt"
        paths = (p29_prediction, p29_checkpoint, p29_score, p27_prediction, p27_checkpoint)
        if any(not path.is_file() for path in paths):
            raise RuntimeError(f"required frozen P27/P29 input is missing for {name}")
        p29_prediction_sha = sha256_file(p29_prediction)
        p27_prediction_sha = sha256_file(p27_prediction)
        p29_checkpoint_sha = sha256_file(p29_checkpoint)
        p27_checkpoint_sha = sha256_file(p27_checkpoint)
        if p29_prediction_sha != p29_folds[name].get("prediction_sha256"):
            raise RuntimeError(f"P29 immutable prediction hash mismatch: {name}")
        if p27_prediction_sha != p27_folds[name].get("prediction_sha256"):
            raise RuntimeError(f"P27 immutable prediction hash mismatch: {name}")
        if p29_checkpoint_sha != p29_folds[name].get("checkpoint_sha256"):
            raise RuntimeError(f"P29 checkpoint hash mismatch: {name}")
        if p27_checkpoint_sha != p27_folds[name].get("checkpoint_sha256"):
            raise RuntimeError(f"P27 checkpoint hash mismatch: {name}")
        result["p29_predictions"][name] = {"sha256": p29_prediction_sha, "expected": p29_folds[name]["prediction_sha256"]}
        result["p29_checkpoints"][name] = {"sha256": p29_checkpoint_sha, "expected": p29_folds[name]["checkpoint_sha256"]}
        result["p29_scores"][name] = sha256_file(p29_score)
        result["p27_predictions"][name] = {"sha256": p27_prediction_sha, "expected": p27_folds[name]["prediction_sha256"]}
        result["p27_checkpoints"][name] = {"sha256": p27_checkpoint_sha, "expected": p27_folds[name]["checkpoint_sha256"]}
        result["tier_a_manifests"][name] = sha256_file(tier_a)
        result["tier_b_manifests"][name] = sha256_file(tier_b)
    for relative in ("P29_PROTOCOL.json", "P29_TERMINAL_EVIDENCE.json", "P28R1_DIAGNOSTIC/P28R1_METRICS.json"):
        path = OUTPUT_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"frozen reference is missing: {path}")
        result[relative.replace("/", "_").replace(".", "_") + "sha256"] = sha256_file(path)
    return result


def _prediction_records(
    path: Path,
    held_class: str,
    expected_hash: str,
    expected_checkpoint_hash: str,
    schema: str,
    student_key: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    observed = sha256_file(path)
    if observed != expected_hash:
        raise RuntimeError(f"frozen prediction hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != schema or payload.get("held_class") != held_class:
        raise RuntimeError(f"immutable prediction schema/class mismatch: {path}")
    if payload.get("gt_used") is not False or int(payload.get("mask_reads", -1)) != 0:
        raise RuntimeError(f"immutable prediction firewall mismatch: {path}")
    if payload.get("adapter_checkpoint_sha256") != expected_checkpoint_hash:
        raise RuntimeError(f"immutable prediction/checkpoint linkage mismatch: {path}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != expected_count:
        raise RuntimeError(f"immutable prediction count mismatch: {path}")
    identities = [str(record.get("image_path", "")) for record in records]
    if not all(identities) or len(identities) != len(set(identities)):
        raise RuntimeError(f"immutable prediction identities invalid: {path}")
    for record in records:
        native = record.get("native_abnormal_probability")
        student = record.get(student_key)
        if not isinstance(native, torch.Tensor) or not isinstance(student, torch.Tensor):
            raise RuntimeError(f"immutable prediction tensors missing: {path}")
        if tuple(native.shape) != (IMAGE_SIZE, IMAGE_SIZE) or tuple(student.shape) != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"immutable prediction map shape mismatch: {path}")
        if not torch.isfinite(native).all() or not torch.isfinite(student).all():
            raise RuntimeError(f"immutable prediction contains non-finite values: {path}")
    return records


def _load_adapter(path: Path, schema: str, held_class: str, device: torch.device) -> RegionResidualAdapter:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != schema or payload.get("held_class") != held_class:
        raise RuntimeError(f"adapter schema/class mismatch: {path}")
    if payload.get("status") != "FOLD_TRAINING_COMPLETE":
        raise RuntimeError(f"adapter is not a frozen scientific checkpoint: {path}")
    if int(payload.get("phase2b_optimization_steps", -1)) != 0 or int(payload.get("clip_optimization_steps", -1)) != 0:
        raise RuntimeError(f"forbidden optimization recorded in adapter: {path}")
    adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    adapter.load_state_dict(payload["state_dict"], strict=True)
    adapter.eval()
    return adapter


def _ordered_arrays(records: Sequence[Mapping[str, Any]], paths: Sequence[str], key: str) -> np.ndarray:
    by_path = {str(record["image_path"]): record for record in records}
    if len(by_path) != len(records) or set(by_path) != set(paths):
        raise RuntimeError("immutable prediction identities do not match metadata inventory")
    return np.stack([
        by_path[path][key].detach().cpu().numpy().astype(np.float32, copy=False)
        for path in paths
    ])


def _tier_a_indices(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], class_name: str) -> np.ndarray:
    sample_ids = list(manifest.get("sample_ids", []))
    expected = [stable_sample_id(row) for row in rows]
    if sample_ids != expected:
        raise RuntimeError(f"Tier-A sample identity mismatch for {class_name}")
    return np.arange(len(expected), dtype=np.int64)


def _student_regions(adapter: RegionResidualAdapter, seg_cache: np.memmap, indices: np.ndarray, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.no_grad():
        for index in indices:
            seg = torch.from_numpy(np.array(seg_cache[int(index)], copy=True)).unsqueeze(1).to(device=device, dtype=torch.float32)
            values.append(adapter(seg).detach().cpu().numpy())
    if not values:
        raise RuntimeError("empty student region inventory")
    return np.concatenate(values, axis=1)


def _teacher_regions(native_cache: np.memmap, indices: np.ndarray, masks: np.ndarray, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    for row_index, cache_index in enumerate(indices):
        native = torch.from_numpy(np.array(native_cache[int(cache_index)], copy=True)).unsqueeze(1).to(device=device, dtype=torch.float32)
        mask = torch.from_numpy(masks[row_index : row_index + 1, None].astype(np.float32, copy=False)).to(device=device)
        utility, _ = forensic_utility_for_batch(native, mask)
        correction = patch_correction_from_actions(classify_actions(utility))
        values.append(pool_patch_map(correction).detach().cpu().numpy())
    if not values:
        raise RuntimeError("empty teacher region inventory")
    return np.concatenate(values, axis=0)


def _native_cache_probability(native_cache: np.memmap, indices: np.ndarray, device: torch.device) -> np.ndarray:
    native = torch.from_numpy(np.array(native_cache[indices], copy=True)).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        probability, _ = deploy_native_logits(native, domain="Industrial")
    return probability[:, 1].detach().cpu().numpy()


def _held_class_result(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    p27_records: Sequence[Mapping[str, Any]],
    p29_records: Sequence[Mapping[str, Any]],
    cache_root: Path,
    visa_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    tier_a_root = cache_root / "tier_a" / name
    manifest = _read_json(tier_a_root / "manifest.json")
    native_cache = np.load(tier_a_root / "native_logits.npy", mmap_mode="r", allow_pickle=False)
    seg_cache = np.load(tier_a_root / "seg_features.npy", mmap_mode="r", allow_pickle=False)
    indices = _tier_a_indices(manifest, rows, name)
    paths = [str(row["image_path"]) for row in rows]
    native = _ordered_arrays(p29_records, paths, "native_abnormal_probability")
    p27 = _ordered_arrays(p27_records, paths, "p27_abnormal_probability")
    p29 = _ordered_arrays(p29_records, paths, "p29_abnormal_probability")
    if not np.array_equal(native, _ordered_arrays(p27_records, paths, "native_abnormal_probability")):
        raise RuntimeError(f"native frozen prediction disagreement: {name}")
    cache_native = _native_cache_probability(native_cache, indices, device)
    native_cache_parity = float(np.max(np.abs(cache_native - native)))
    if native_cache_parity > PREDICTION_PARITY_TOLERANCE:
        raise RuntimeError(f"Tier-A/native immutable parity failure for {name}: {native_cache_parity}")
    masks, mask_reads = _load_masks(rows, visa_root)
    p27_adapter = _load_adapter(P27_ROOT / name / "training/p27_region_adapter.pt", P27_ADAPTER_SCHEMA, name, device)
    p29_adapter = _load_adapter(P29_ROOT / name / "training/p29_region_adapter.pt", P29_ADAPTER_SCHEMA, name, device)
    teacher = _teacher_regions(native_cache, indices, masks, device)
    p27_region = _student_regions(p27_adapter, seg_cache, indices, device)
    p29_region = _student_regions(p29_adapter, seg_cache, indices, device)
    teacher_staged = np.broadcast_to(teacher[None, ...], p29_region.shape)
    p27_alignment = sign_alignment(teacher_staged, p27_region)
    p29_alignment = sign_alignment(teacher_staged, p29_region)
    return {
        "class": name,
        "sample_count": len(rows),
        "held_mask_reads": int(mask_reads),
        "native_cache_max_abs_error": native_cache_parity,
        "teacher_magnitude": residual_magnitude_summary(teacher),
        "p27_alignment": p27_alignment,
        "p29_alignment": p29_alignment,
        "alignment_delta": {
            key: (p29_alignment[key] - p27_alignment[key]) if p29_alignment[key] is not None and p27_alignment[key] is not None else None
            for key in p29_alignment
        },
        "p27_residual_magnitude": residual_magnitude_summary(p27_region),
        "p29_residual_magnitude": residual_magnitude_summary(p29_region),
        "normality": {
            "p27_minus_native": vectorized_pixel_shifts(native, p27, masks),
            "p29_minus_native": vectorized_pixel_shifts(native, p29, masks),
            "p29_minus_p27": vectorized_pixel_shifts(p27, p29, masks),
        },
    }


def _source_batch(
    held_class: str,
    source_class: str,
    fit_rows: Sequence[Mapping[str, Any]],
    cache_root: Path,
    maximum_samples: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    if maximum_samples <= 0:
        raise ValueError("maximum_samples must be positive")
    source_rows = {stable_sample_id(row): row for row in fit_rows if str(row["class_name"]) == source_class}
    tier_a_root = cache_root / "tier_a" / source_class
    tier_a_manifest = _read_json(tier_a_root / "manifest.json")
    tier_a_ids = list(tier_a_manifest.get("sample_ids", []))
    selected_ids = [sample_id for sample_id in tier_a_ids if sample_id in source_rows][:maximum_samples]
    if not selected_ids or any(str(source_rows[sample_id]["class_name"]) == held_class for sample_id in selected_ids):
        raise RuntimeError("invalid deterministic source probe inventory")
    tier_b_manifest = _read_json(cache_root / "tier_b" / held_class / "manifest.json")
    tier_b_ids = list(tier_b_manifest.get("sample_ids", []))
    a_index = {sample_id: index for index, sample_id in enumerate(tier_a_ids)}
    b_index = {sample_id: index for index, sample_id in enumerate(tier_b_ids)}
    if any(sample_id not in b_index for sample_id in selected_ids):
        raise RuntimeError("source cache sample identity missing")
    seg = np.load(tier_a_root / "seg_features.npy", mmap_mode="r", allow_pickle=False)
    native = np.load(tier_a_root / "native_logits.npy", mmap_mode="r", allow_pickle=False)
    tier_b_root = cache_root / "tier_b" / held_class
    source_mask = np.load(tier_b_root / "source_mask.npy", mmap_mode="r", allow_pickle=False)
    teacher = np.load(tier_b_root / "teacher_region.npy", mmap_mode="r", allow_pickle=False)
    a_indices = [a_index[sample_id] for sample_id in selected_ids]
    b_indices = [b_index[sample_id] for sample_id in selected_ids]
    return {
        "seg": torch.from_numpy(np.array(seg[a_indices], copy=True)).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32),
        "native": torch.from_numpy(np.array(native[a_indices], copy=True)).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32),
        "mask": torch.from_numpy(np.array(source_mask[b_indices], copy=True)).to(device=device, dtype=torch.float32),
        "teacher": torch.from_numpy(np.array(teacher[b_indices], copy=True)).to(device=device, dtype=torch.float32),
    }, selected_ids


def _gradient_vector(loss: torch.Tensor, adapter: RegionResidualAdapter) -> list[torch.Tensor]:
    gradients = torch.autograd.grad(loss, tuple(adapter.parameters()), allow_unused=False)
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("non-finite forensic gradient")
    return [gradient.detach() for gradient in gradients]


def _component_gradients(adapter: RegionResidualAdapter, batch: Mapping[str, torch.Tensor]) -> dict[str, list[torch.Tensor]]:
    teacher = batch["teacher"].unsqueeze(0).expand(STAGES, -1, -1, -1)

    def p29_component(name: str) -> list[torch.Tensor]:
        terms = p29_sign_guarded_loss(adapter(batch["seg"]), teacher, batch["mask"])
        return _gradient_vector(getattr(terms, name), adapter)

    values = {name: p29_component(name) for name in ("value", "sign", "normal", "total")}
    values["p27_distill"] = _gradient_vector(F.smooth_l1_loss(adapter(batch["seg"]), teacher), adapter)
    residual = adapter(batch["seg"])
    corrected = symmetric_margin_delta(batch["native"], upsample_region_map(residual))
    probability, _ = deploy_native_logits(corrected, domain="Industrial")
    values["seg"] = _gradient_vector(calculate_seg_loss(probability, batch["mask"]), adapter)
    return {f"g_{name}": value for name, value in values.items()}


def _cosine(summary: Mapping[str, Any], left: str, right: str) -> float | None:
    return summary["cosines"].get("__".join(sorted((left, right))))


def _gradient_probe(held_class: str, fit_rows: Sequence[Mapping[str, Any]], cache_root: Path, device: torch.device, maximum_samples: int = 8) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    checkpoint = P29_ROOT / held_class / "training/p29_region_adapter.pt"
    for state in ("zero_init", "p29_checkpoint"):
        if state == "zero_init":
            torch.manual_seed(0)
            adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
            adapter.eval()
        else:
            adapter = _load_adapter(checkpoint, P29_ADAPTER_SCHEMA, held_class, device)
        for source_class in select_probe_source_classes(held_class):
            batch, sample_ids = _source_batch(held_class, source_class, fit_rows, cache_root, maximum_samples, device)
            gradients = _component_gradients(adapter, batch)
            summary = gradient_summary(gradients)
            summary["value_sign_cosine"] = _cosine(summary, "g_value", "g_sign")
            summary["value_normal_cosine"] = _cosine(summary, "g_value", "g_normal")
            summary["sign_normal_cosine"] = _cosine(summary, "g_sign", "g_normal")
            summary["seg_p27_distill_cosine"] = _cosine(summary, "g_seg", "g_p27_distill")
            summary["seg_p29_value_cosine"] = _cosine(summary, "g_seg", "g_value")
            summary["seg_p29_total_cosine"] = _cosine(summary, "g_seg", "g_total")
            p27_norm = summary["norms"]["g_p27_distill"]
            total_norm = summary["norms"]["g_total"]
            summary["p29_value_over_p27_distill"] = summary["norms"]["g_value"] / p27_norm if p27_norm else None
            summary["seg_over_p27_distill"] = summary["norms"]["g_seg"] / p27_norm if p27_norm else None
            summary["seg_over_p29_total"] = summary["norms"]["g_seg"] / total_norm if total_norm else None
            records.append({
                "held_class": held_class,
                "state": state,
                "source_class": source_class,
                "sample_ids": sample_ids,
                "sample_count": len(sample_ids),
                "gradients": summary,
                "normal_guard_conflict": normal_guard_conflict(batch["mask"], batch["teacher"]),
            })
    return records


def _mean(values: Sequence[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def _gradient_aggregate(records: Sequence[Mapping[str, Any]], state: str) -> dict[str, Any]:
    selected = [record for record in records if record["state"] == state]
    if not selected:
        raise RuntimeError(f"no gradient records for state {state}")
    names = ("g_value", "g_sign", "g_normal", "g_total", "g_p27_distill", "g_seg")
    output: dict[str, Any] = {
        "record_count": len(selected),
        "norms": {name: _mean([record["gradients"]["norms"][name] for record in selected]) for name in names},
    }
    for name in (
        "value_sign_cosine", "value_normal_cosine", "sign_normal_cosine",
        "seg_p27_distill_cosine", "seg_p29_value_cosine", "seg_p29_total_cosine",
        "p29_value_over_p27_distill", "seg_over_p27_distill", "seg_over_p29_total",
    ):
        output[name] = _mean([record["gradients"][name] for record in selected])
    count = sum(int(record["normal_guard_conflict"]["pure_normal_region_count"]) for record in selected)
    output["normal_guard"] = {
        "pure_normal_region_count": count,
        "pure_normal_region_fraction": _mean([record["normal_guard_conflict"]["pure_normal_region_fraction"] for record in selected]),
        "teacher_positive_fraction": _mean([record["normal_guard_conflict"]["teacher_positive_fraction"] for record in selected]),
        "teacher_zero_fraction": _mean([record["normal_guard_conflict"]["teacher_zero_fraction"] for record in selected]),
        "teacher_negative_fraction": _mean([record["normal_guard_conflict"]["teacher_negative_fraction"] for record in selected]),
        "positive_strength_mass": _mean([record["normal_guard_conflict"]["positive_strength_mass"] for record in selected]),
    }
    if state == "zero_init":
        output["zero_gradient_classification"] = {
            name: "ZERO" if output["norms"][name] == 0.0 else "NONZERO"
            for name in ("g_value", "g_sign", "g_normal")
        }
    return output


def _score_rows() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    p29_rows = {
        row["class"]: {key: float(value) for key, value in row.items() if key != "class"}
        for row in csv.DictReader((OUTPUT_ROOT / "P29_CLASS_TABLE.csv").open(encoding="utf-8"))
    }
    p28 = _read_json(OUTPUT_ROOT / "P28R1_DIAGNOSTIC/P28R1_METRICS.json")
    p27_rows = {
        row["class"]: {
            "native_pAP": float(row["state_metrics"]["N"]["pAP"]),
            "native_pAUROC": float(row["state_metrics"]["N"]["pAUROC"]),
            "p27_pAP": float(row["state_metrics"]["S"]["pAP"]),
            "p27_pAUROC": float(row["state_metrics"]["S"]["pAUROC"]),
            "or_pAP": float(row["state_metrics"]["OR"]["pAP"]),
            "or_pAUROC": float(row["state_metrics"]["OR"]["pAUROC"]),
        }
        for row in p28["classes"]
    }
    if set(p29_rows) != set(CLASS_NAMES) or set(p27_rows) != set(CLASS_NAMES):
        raise RuntimeError("frozen score class inventory mismatch")
    return p29_rows, p27_rows


def _recovery_ratios() -> dict[str, Any]:
    p29_rows, p27_rows = _score_rows()
    classes: dict[str, Any] = {}

    def ratio(value: float, native: float, oracle: float) -> float | None:
        denominator = oracle - native
        return None if denominator == 0.0 else (value - native) / denominator

    for name in CLASS_NAMES:
        native_ap, native_auc = p27_rows[name]["native_pAP"], p27_rows[name]["native_pAUROC"]
        oracle_ap, oracle_auc = p27_rows[name]["or_pAP"], p27_rows[name]["or_pAUROC"]
        classes[name] = {
            "p27_pAP_recovery": ratio(p27_rows[name]["p27_pAP"], native_ap, oracle_ap),
            "p29_pAP_recovery": ratio(p29_rows[name]["p29_pAP"], native_ap, oracle_ap),
            "p27_pAUROC_recovery": ratio(p27_rows[name]["p27_pAUROC"], native_auc, oracle_auc),
            "p29_pAUROC_recovery": ratio(p29_rows[name]["p29_pAUROC"], native_auc, oracle_auc),
        }
    keys = tuple(next(iter(classes.values())))
    return {"classes": classes, "macro_mean": {key: _mean([classes[name][key] for name in CLASS_NAMES]) for key in keys}}


def _decision(class_results: Sequence[Mapping[str, Any]], gradients: Mapping[str, Any]) -> dict[str, Any]:
    p29_rows, p27_rows = _score_rows()
    p29_sign = _mean([result["p29_alignment"]["sign_agreement"] for result in class_results])
    p27_sign = _mean([result["p27_alignment"]["sign_agreement"] for result in class_results])
    p27_abs = _mean([result["p27_residual_magnitude"]["mean_abs"] for result in class_results])
    p29_abs = _mean([result["p29_residual_magnitude"]["mean_abs"] for result in class_results])
    residual_ratio = p29_abs / p27_abs if p27_abs else None
    zero = gradients["zero_init"]
    checkpoint = gradients["p29_checkpoint"]
    value_ratio = zero["p29_value_over_p27_distill"]
    sign_zero = zero["norms"]["g_sign"] == 0.0
    normal_zero = zero["norms"]["g_normal"] == 0.0
    p29_normal_q95 = _mean([result["normality"]["p29_minus_native"]["normal"]["q95"] for result in class_results])
    p27_normal_q95 = _mean([result["normality"]["p27_minus_native"]["normal"]["q95"] for result in class_results])
    p29_normal_q99 = _mean([result["normality"]["p29_minus_native"]["normal"]["q99"] for result in class_results])
    p27_normal_q99 = _mean([result["normality"]["p27_minus_native"]["normal"]["q99"] for result in class_results])
    p29_pap = float(np.mean([p29_rows[name]["p29_pAP"] for name in CLASS_NAMES]))
    p27_pap = float(np.mean([p27_rows[name]["p27_pAP"] for name in CLASS_NAMES]))
    p29_auc = float(np.mean([p29_rows[name]["p29_pAUROC"] for name in CLASS_NAMES]))
    p27_auc = float(np.mean([p27_rows[name]["p27_pAUROC"] for name in CLASS_NAMES]))
    native_auc = float(np.mean([p27_rows[name]["native_pAUROC"] for name in CLASS_NAMES]))
    auc_deltas = [p29_rows[name]["p29_pAUROC"] - p27_rows[name]["p27_pAUROC"] for name in CLASS_NAMES]
    pap_deltas = [p29_rows[name]["p29_pAP"] - p27_rows[name]["p27_pAP"] for name in CLASS_NAMES]
    auc_breadth_loss = sum(delta < 0.0 for delta in auc_deltas) > sum(delta >= 0.0 for delta in auc_deltas)
    pap_breadth_loss = sum(delta < 0.0 for delta in pap_deltas) > sum(delta >= 0.0 for delta in pap_deltas)
    poor_auc = p29_auc < p27_auc and p29_auc < native_auc
    sign_delta = p29_sign - P27_SIGN_AGREEMENT if p29_sign is not None else None
    positive_fraction = zero["normal_guard"]["teacher_positive_fraction"]
    positive_mass = zero["normal_guard"]["positive_strength_mass"]
    guard_conflict = (
        (positive_fraction is not None and positive_fraction > 1e-3)
        or (positive_mass is not None and positive_mass > 1e-2)
        or (checkpoint["sign_normal_cosine"] is not None and checkpoint["sign_normal_cosine"] < -0.1)
    )
    gradient_starvation = sign_zero and normal_zero and value_ratio is not None and value_ratio < 0.9 and ((residual_ratio is not None and residual_ratio < 1.0) or (sign_delta is not None and sign_delta <= 0.01))
    gradient_plausible = sign_zero and normal_zero and value_ratio is not None and value_ratio < 1.0
    seg_evidence = zero["norms"]["g_seg"] > 0.0 and zero["seg_p27_distill_cosine"] is not None and zero["seg_p27_distill_cosine"] >= 0.0 and zero["seg_p29_value_cosine"] is not None and zero["seg_p29_value_cosine"] >= 0.0
    seg_anchor = seg_evidence and poor_auc and (auc_breadth_loss or pap_breadth_loss)
    labels = {
        "GRADIENT_STARVATION": "SUPPORTED" if gradient_starvation else "PLAUSIBLE" if gradient_plausible else "NOT_SUPPORTED",
        "SEGMENTATION_ANCHOR_REMOVAL": "SUPPORTED" if seg_anchor else "PLAUSIBLE" if seg_evidence else "NOT_SUPPORTED",
        "NORMAL_GUARD_CONFLICT": "SUPPORTED" if guard_conflict else "NOT_SUPPORTED",
        "SIGN_FIX_INSUFFICIENT": "SUPPORTED" if sign_delta is not None and sign_delta > 0.01 and poor_auc and auc_breadth_loss else "NOT_SUPPORTED",
        "NORMALITY_FIX_INSUFFICIENT": "SUPPORTED" if p29_normal_q95 is not None and p27_normal_q95 is not None and p29_normal_q99 is not None and p27_normal_q99 is not None and p29_normal_q95 < p27_normal_q95 and p29_normal_q99 < p27_normal_q99 and poor_auc else "NOT_SUPPORTED",
        "STUDENT_CAPACITY_LIMIT": "NOT_SUPPORTED",
        "REGION_REPRESENTATION_LIMIT": "NOT_SUPPORTED",
    }
    supported_mechanisms = [name for name in ("GRADIENT_STARVATION", "SEGMENTATION_ANCHOR_REMOVAL", "NORMAL_GUARD_CONFLICT", "SIGN_FIX_INSUFFICIENT", "NORMALITY_FIX_INSUFFICIENT") if labels[name] == "SUPPORTED"]
    labels["MIXED_OBJECTIVE_CONFLICT"] = "SUPPORTED" if len(supported_mechanisms) >= 2 else "PLAUSIBLE" if len(supported_mechanisms) == 1 else "NOT_SUPPORTED"
    if len(supported_mechanisms) >= 2:
        primary, secondary = "MIXED_OBJECTIVE_CONFLICT", supported_mechanisms[0]
    elif len(supported_mechanisms) == 1:
        primary, secondary = supported_mechanisms[0], "NONE"
    else:
        primary, secondary = "INSUFFICIENT", "NONE"
    recommendation_map = {
        "GRADIENT_STARVATION": "SINGLE_OBJECTIVE_DIRECTIONAL_DISTILLATION",
        "SEGMENTATION_ANCHOR_REMOVAL": "TEACHER_DISTILLATION_WITH_LIGHT_LOCALIZATION_ANCHOR",
        "NORMAL_GUARD_CONFLICT": "TEACHER_CONSISTENT_NORMALITY_PRESERVATION",
        "SIGN_FIX_INSUFFICIENT": "RELATIONAL_OR_RANKING_DISTILLATION",
        "NORMALITY_FIX_INSUFFICIENT": "TEACHER_CONSISTENT_NORMALITY_PRESERVATION",
        "MIXED_OBJECTIVE_CONFLICT": "SINGLE_OBJECTIVE_DIRECTIONAL_DISTILLATION",
        "INSUFFICIENT": "STOP_REGION_DISTILLATION_LINEAGE",
    }
    candidates = ("SINGLE_OBJECTIVE_DIRECTIONAL_DISTILLATION", "TEACHER_DISTILLATION_WITH_LIGHT_LOCALIZATION_ANCHOR", "TEACHER_CONSISTENT_NORMALITY_PRESERVATION", "RELATIONAL_OR_RANKING_DISTILLATION", "FEATURE_LEVEL_CONSISTENCY", "STOP_REGION_DISTILLATION_LINEAGE")
    base = {
        "SINGLE_OBJECTIVE_DIRECTIONAL_DISTILLATION": (8, 9, 9, 8, 8, 10, 9),
        "TEACHER_DISTILLATION_WITH_LIGHT_LOCALIZATION_ANCHOR": (7, 7, 7, 8, 6, 10, 8),
        "TEACHER_CONSISTENT_NORMALITY_PRESERVATION": (7, 8, 7, 8, 7, 10, 9),
        "RELATIONAL_OR_RANKING_DISTILLATION": (8, 5, 8, 9, 8, 9, 8),
        "FEATURE_LEVEL_CONSISTENCY": (2, 4, 6, 7, 6, 8, 6),
        "STOP_REGION_DISTILLATION_LINEAGE": (1, 10, 1, 1, 1, 10, 10),
    }
    weights = (0.30, 0.15, 0.15, 0.15, 0.10, 0.05, 0.05, 0.05)
    recommendation = recommendation_map[primary]
    scores = []
    dimension_names = ("evidence_fit", "simplicity", "scientific_upside", "cross_category_adaptability", "novelty_potential", "zero_inference_overhead", "auditability", "expected_training_cost")
    for candidate in candidates:
        dimensions = ((10 if candidate == recommendation else base[candidate][0]), *base[candidate][1:])
        scores.append({"direction": candidate, "score_0_to_10": float(sum(weight * value for weight, value in zip(weights, dimensions))), "dimensions": dict(zip(dimension_names, dimensions))})
    return {
        "candidate_classifications": labels,
        "primary_root_cause": primary,
        "secondary_root_cause": secondary,
        "sign_macro": p29_sign,
        "p27_sign_macro": p27_sign,
        "sign_delta_vs_frozen_p27": sign_delta,
        "p29_over_p27_residual_magnitude_ratio": residual_ratio,
        "p27_normal_q95_shift": p27_normal_q95,
        "p29_normal_q95_shift": p29_normal_q95,
        "p27_normal_q99_shift": p27_normal_q99,
        "p29_normal_q99_shift": p29_normal_q99,
        "score_summary": {"p27_pAP": p27_pap, "p29_pAP": p29_pap, "p27_pAUROC": p27_auc, "p29_pAUROC": p29_auc, "native_pAUROC": native_auc, "p29_vs_p27_AUROC_regressing_classes": sum(delta < 0.0 for delta in auc_deltas), "p29_vs_p27_AUROC_non_regressing_classes": sum(delta >= 0.0 for delta in auc_deltas)},
        "rule_evidence": {"zero_sign_gradient": zero["norms"]["g_sign"], "zero_normal_gradient": zero["norms"]["g_normal"], "zero_value_over_p27_distill": value_ratio, "teacher_positive_fraction_in_pure_normal": positive_fraction, "teacher_positive_strength_mass_in_pure_normal": positive_mass, "poor_pAUROC": poor_auc, "AUROC_breadth_loss": auc_breadth_loss, "pAP_breadth_loss": pap_breadth_loss},
        "decision_thresholds": {"material_value_gradient_ratio_max": 0.9, "material_sign_delta_min": 0.01, "meaningful_positive_fraction_min": 0.001, "meaningful_positive_strength_mass_min": 0.01},
        "ranked_next_direction_scores": sorted(scores, key=lambda row: row["score_0_to_10"], reverse=True),
        "optimal_next_research_direction": recommendation,
        "one_recommendation_only": True,
    }


def _write_report(metrics: Mapping[str, Any], gradient: Mapping[str, Any], decision: Mapping[str, Any], runtime: Mapping[str, Any]) -> None:
    zero = gradient["zero_init"]
    recovery = metrics["output_recovery"]["macro_mean"]
    lines = [
        "# P29R1 FAST FORENSIC FINAL REPORT", "", "## IDENTITY", "",
        f"- P29 terminal SHA: `{metrics['identity']['p29_terminal_sha']}`",
        f"- P29R1 prereg SHA: `{metrics['identity']['p29r1_prereg_sha']}`",
        f"- P29R1 execution-base SHA: `{metrics['identity']['p29r1_execution_base_sha']}`",
        f"- Forensic UUID: `{metrics['identity']['forensic_uuid']}`", "",
        "## PERFORMANCE", "",
        f"- Runtime: `{runtime['wall_seconds']:.3f}` seconds for `{runtime['classes_processed']}/12` classes.",
        f"- Peak RSS: `{runtime['peak_rss_kib']}` KiB; peak GPU allocated: `{runtime['peak_gpu_allocated_bytes']}` bytes.",
        "- New CLIP forwards: `0`; new Phase2B forwards: `0`; training steps: `0`; optimizer steps: `0`.", "",
        "## SIGN / MAGNITUDE", "",
        f"- Frozen P27 sign agreement: `{P27_SIGN_AGREEMENT}`; P29 sign agreement: `{decision['sign_macro']}`; delta: `{decision['sign_delta_vs_frozen_p27']}`.",
        f"- P29/P27 mean residual-magnitude ratio: `{decision['p29_over_p27_residual_magnitude_ratio']}`.", "",
        "## ZERO-INIT GRADIENTS", "",
        f"- L_value: `{zero['norms']['g_value']}`; L_sign: `{zero['norms']['g_sign']}`; L_normal: `{zero['norms']['g_normal']}`.",
        f"- P27 raw distillation: `{zero['norms']['g_p27_distill']}`; P29/P27 value ratio: `{zero['p29_value_over_p27_distill']}`.",
        f"- Zero-init classification: `{zero['zero_gradient_classification']}`.", "",
        "## NORMALITY / RECOVERY", "",
        f"- Pure-normal teacher-positive fraction: `{zero['normal_guard']['teacher_positive_fraction']}`; positive strength mass: `{zero['normal_guard']['positive_strength_mass']}`.",
        f"- P27/P29 normal q99 shift: `{decision['p27_normal_q99_shift']}` / `{decision['p29_normal_q99_shift']}`.",
        f"- P27/P29 OR pAP recovery: `{recovery['p27_pAP_recovery']}` / `{recovery['p29_pAP_recovery']}`.",
        f"- P27/P29 OR pAUROC recovery: `{recovery['p27_pAUROC_recovery']}` / `{recovery['p29_pAUROC_recovery']}`.", "",
        "## ROOT CAUSE DECISION", "",
    ]
    for name, label in decision["candidate_classifications"].items():
        lines.append(f"- `{name}`: `{label}`")
    lines.extend([
        "", f"Primary root cause: `{decision['primary_root_cause']}`.",
        f"Secondary root cause: `{decision['secondary_root_cause']}`.",
        f"One recommended next research direction: `{decision['optimal_next_research_direction']}`.",
        "", "All reported quantities use frozen P27/P29 artifacts, the P28R1-compatible held teacher definition, and bounded source-only gradient probes. No model was trained or updated.",
        "", "## STATUS", "", "`P29R1_FORENSIC_COMPLETE`",
    ])
    _atomic_text(OUTPUT_ROOT / "P29R1_FINAL_REPORT.md", "\n".join(lines) + "\n")


def pre_audit(args: argparse.Namespace) -> dict[str, Any]:
    validate_runner_execution_contract()
    _protocol_payload(args.protocol)
    enforce_data_firewall(args.visa_root, [args.visa_root])
    output = args.output_dir
    if any((output / name).exists() for name in FORENSIC_OUTPUTS):
        raise RuntimeError("P29R1 forensic output already exists before pre-audit")
    local = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    if branch != "research/p29r1-fast-objective-forensic-v1":
        raise RuntimeError("P29R1 pre-audit is on the wrong branch")
    if args.prereg_sha:
        subprocess.run(["git", "merge-base", "--is-ancestor", args.prereg_sha, "HEAD"], cwd=ROOT, check=True)
    remote = _remote_head(branch)
    inventory = _artifact_inventory(args.cache_root, args.metadata)
    result = {
        "schema_version": "P29R1_PRE_AUDIT_V1",
        "status": "PASS",
        "branch": branch,
        "local_sha": local,
        "remote_sha": remote,
        "remote_equals_local": remote == local,
        "prereg_sha": args.prereg_sha,
        "worktree_before_execution_base": _git("status", "--porcelain"),
        "protocol_sha256": sha256_file(args.protocol),
        "input_inventory": inventory,
        "attempt_marker_absent": not (output / "P29R1_FORENSIC_ATTEMPT.json").exists(),
        "training_steps": 0,
        "optimizer_steps": 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "mvtec_reads": 0,
        "medical_reads": 0,
    }
    if not result["remote_equals_local"]:
        raise RuntimeError("P29R1 preregistration is not synchronized with origin")
    _atomic_json(args.preaudit_output, result)
    return result


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    validate_runner_execution_contract()
    _protocol_payload(args.protocol)
    enforce_data_firewall(args.visa_root, [args.visa_root])
    output = args.output_dir
    if (output / "P29R1_FORENSIC_ATTEMPT.json").exists():
        raise RuntimeError("P29R1 attempt marker exists; preflight cannot run afterward")
    rows = read_visa_metadata(args.metadata)
    held = CLASS_NAMES[0]
    inventory = loco_inventory(rows, held)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = select_probe_source_classes(held)[0]
    load_started = time.perf_counter()
    batch, sample_ids = _source_batch(held, source, inventory.fit_rows, args.cache_root, 2, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    io_bytes = sum(int(value.numel() * value.element_size()) for value in batch.values())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timing_adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32).eval()
    held_started = time.perf_counter()
    forensic_utility_for_batch(batch["native"][:, :1], batch["mask"][:1])
    with torch.no_grad():
        timing_adapter(batch["seg"][:, :1])
        timing_adapter(batch["seg"][:, :1])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    held_seconds = time.perf_counter() - held_started
    probe_started = time.perf_counter()
    _component_gradients(timing_adapter, batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    probe_seconds = time.perf_counter() - probe_started
    held_counts = {name: int(_read_json(args.cache_root / "tier_a" / name / "manifest.json")["sample_count"]) for name in CLASS_NAMES}
    probe_exposures = 0
    for held_class in CLASS_NAMES:
        fold = loco_inventory(rows, held_class)
        for source_class in select_probe_source_classes(held_class):
            probe_exposures += 2 * min(8, sum(str(row["class_name"]) == source_class for row in fold.fit_rows))
    held_per_sample = held_seconds
    probe_per_sample = probe_seconds / max(1, len(sample_ids))
    projected_work = sum(held_counts.values()) * held_per_sample + probe_exposures * probe_per_sample
    estimate = estimate_forensic_runtime(seconds_per_class=projected_work / len(CLASS_NAMES), classes=len(CLASS_NAMES), fixed_seconds=30.0)
    payload = {
        "schema_version": "P29R1_ENGINEERING_PREFLIGHT_V1",
        "status": "PASS" if estimate["decision"] == "PROCEED" else "OPTIMIZE_REQUIRED" if estimate["decision"] == "OPTIMIZE_ONCE" else "STOP",
        "held_class_for_timing_only": held,
        "source_class_for_timing_only": source,
        "source_sample_ids": sample_ids,
        "source_sample_count": len(sample_ids),
        "cache_load_seconds": load_seconds,
        "cache_bytes_sample": io_bytes,
        "cache_io_throughput_mib_per_second": (io_bytes / (1024.0 ** 2)) / load_seconds if load_seconds else None,
        "held_like_seconds_per_sample": held_per_sample,
        "probe_seconds_per_sample": probe_per_sample,
        "held_samples_projected": sum(held_counts.values()),
        "source_probe_exposures_projected": probe_exposures,
        "projected_work_seconds_without_fixed": projected_work,
        "estimate": estimate,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "held_gt_or_mask_reads": 0,
        "scientific_outputs_written": [],
    }
    _atomic_json(args.preflight_output, payload)
    return payload


def _normality_macro(classes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for state in ("p27_minus_native", "p29_minus_native", "p29_minus_p27"):
        result[state] = {}
        for stratum in ("normal", "anomaly"):
            metrics = classes[0]["normality"][state][stratum]
            result[state][stratum] = {metric: _mean([item["normality"][state][stratum][metric] for item in classes]) for metric in metrics}
    return result


def _post_run_audit(
    *,
    args: argparse.Namespace,
    marker: Mapping[str, Any],
    input_inventory: Mapping[str, Any],
    post_inventory: Mapping[str, Any],
    classes: Sequence[Mapping[str, Any]],
    probe_records: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    required = [args.output_dir / name for name in FORENSIC_OUTPUTS[1:7]]
    if any(not path.is_file() for path in required):
        raise RuntimeError("P29R1 output set is incomplete before post-run audit")
    if len(classes) != len(CLASS_NAMES) or {item["class"] for item in classes} != set(CLASS_NAMES):
        raise RuntimeError("P29R1 class output inventory is incomplete")
    expected_records = len(CLASS_NAMES) * 2 * 4
    if len(probe_records) != expected_records:
        raise RuntimeError(f"P29R1 source probe count mismatch: {len(probe_records)} != {expected_records}")
    rows = read_visa_metadata(args.metadata)
    for record in probe_records:
        held_class = str(record["held_class"])
        source_class = str(record["source_class"])
        fold = loco_inventory(rows, held_class)
        expected_count = min(8, sum(str(row["class_name"]) == source_class for row in fold.fit_rows))
        if source_class not in select_probe_source_classes(held_class) or int(record["sample_count"]) != expected_count:
            raise RuntimeError("bounded source probe inventory mismatch")
    if post_inventory != input_inventory:
        raise RuntimeError("frozen P27/P29/cache artifact changed during forensic")
    if marker.get("status") != "CONSUMED" or marker.get("forensic_attempt_count") != 1:
        raise RuntimeError("forensic attempt marker is not singular and consumed")
    if _git("rev-parse", "HEAD") != args.execution_base_sha:
        raise RuntimeError("execution base changed after forensic marker")
    if sha256_file(args.protocol) != marker.get("protocol_sha256"):
        raise RuntimeError("protocol changed after forensic marker")
    csv_rows = list(csv.DictReader((args.output_dir / "P29R1_SIGN_ALIGNMENT.csv").open(encoding="utf-8")))
    if len(csv_rows) != len(CLASS_NAMES) or {row.get("class") for row in csv_rows} != set(CLASS_NAMES):
        raise RuntimeError("sign-alignment CSV is incomplete")
    audit = {
        "schema_version": "P29R1_POST_RUN_AUDIT_V1",
        "status": "PASS",
        "terminal_status": "P29R1_FORENSIC_COMPLETE",
        "forensic_uuid": marker["forensic_uuid"],
        "forensic_attempt_count": 1,
        "classes_processed": len(classes),
        "source_probe_record_count": len(probe_records),
        "training_steps": 0,
        "optimizer_steps": 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "mvtec_reads": 0,
        "medical_reads": 0,
        "held_gt_or_mask_reads_posthoc": sum(int(item["held_mask_reads"]) for item in classes),
        "p29_predictions_unchanged": True,
        "p29_checkpoints_unchanged": True,
        "p29_scores_unchanged": True,
        "bounded_source_inventory_exact": True,
        "no_pairwise_explosion": True,
        "protocol_unchanged_after_execution_base": True,
        "execution_base_unchanged_since_marker": True,
        "input_artifacts_unchanged": True,
        "decision_primary_root_cause": decision["primary_root_cause"],
        "decision_recommendation": decision["optimal_next_research_direction"],
        "input_inventory": input_inventory,
        "metrics_schema": metrics["schema_version"],
    }
    _atomic_json(args.output_dir / "P29R1_POST_RUN_AUDIT.json", audit)
    return audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_runner_execution_contract()
    _protocol_payload(args.protocol)
    if not args.execution_base_sha or not args.prereg_sha or not args.forensic_uuid or not args.utc_started:
        raise RuntimeError("run mode requires execution-base SHA, prereg SHA, UUID, and UTC timestamp")
    if _git("rev-parse", "HEAD") != args.execution_base_sha:
        raise RuntimeError("P29R1 execution-base SHA mismatch")
    branch = _git("branch", "--show-current")
    if branch != "research/p29r1-fast-objective-forensic-v1":
        raise RuntimeError("P29R1 run is on the wrong branch")
    if _remote_head(branch) != args.execution_base_sha:
        raise RuntimeError("P29R1 execution base is not synchronized with origin")
    if _git("status", "--porcelain"):
        raise RuntimeError("P29R1 execution base must have a clean worktree before the attempt marker")
    output = args.output_dir
    marker_path = output / "P29R1_FORENSIC_ATTEMPT.json"
    if marker_path.exists():
        raise RuntimeError("P29R1 forensic attempt marker already exists; refusing a second attempt")
    preflight_payload = _read_json(args.preflight_output)
    if preflight_payload.get("status") != "PASS" or preflight_payload.get("estimate", {}).get("decision") != "PROCEED":
        raise RuntimeError("qualified P29R1 performance preflight is required before execution")
    enforce_data_firewall(args.visa_root, [args.visa_root])
    input_inventory = _artifact_inventory(args.cache_root, args.metadata)
    attempt = {
        "schema_version": "P29R1_FORENSIC_ATTEMPT_V1",
        "status": "CONSUMED",
        "forensic_attempt_count": 1,
        "forensic_uuid": args.forensic_uuid,
        "utc_started": args.utc_started,
        "p29_terminal_sha": P29_TERMINAL_SHA,
        "p29_execution_base_sha": P29_EXECUTION_BASE_SHA,
        "p29r1_prereg_sha": args.prereg_sha,
        "p29r1_execution_base_sha": args.execution_base_sha,
        "protocol_sha256": sha256_file(args.protocol),
        "preflight_sha256": sha256_file(args.preflight_output),
        "preaudit_sha256": sha256_file(args.preaudit_output),
        "input_artifacts": input_inventory,
        "training_steps": 0,
        "optimizer_steps": 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "mvtec_reads": 0,
        "medical_reads": 0,
    }
    _atomic_json(marker_path, attempt)
    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        rows = read_visa_metadata(args.metadata)
        p29_hashes = {name: input_inventory["p29_predictions"][name]["expected"] for name in CLASS_NAMES}
        p27_hashes = {name: input_inventory["p27_predictions"][name]["expected"] for name in CLASS_NAMES}
        p29_checkpoint_hashes = {name: input_inventory["p29_checkpoints"][name]["expected"] for name in CLASS_NAMES}
        p27_checkpoint_hashes = {name: input_inventory["p27_checkpoints"][name]["expected"] for name in CLASS_NAMES}
        class_results: list[dict[str, Any]] = []
        probe_records: list[dict[str, Any]] = []
        for name in CLASS_NAMES:
            fold = loco_inventory(rows, name)
            p27_records = _prediction_records(P27_ROOT / name / "predictions/p27_held_predictions.pt", name, p27_hashes[name], p27_checkpoint_hashes[name], P27_SCHEMA, "p27_abnormal_probability", len(fold.held_rows))
            p29_records = _prediction_records(P29_ROOT / name / "predictions/p29_held_predictions.pt", name, p29_hashes[name], p29_checkpoint_hashes[name], P29_SCHEMA, "p29_abnormal_probability", len(fold.held_rows))
            class_results.append(_held_class_result(name, fold.held_rows, p27_records, p29_records, args.cache_root, args.visa_root, device))
            probe_records.extend(_gradient_probe(name, fold.fit_rows, args.cache_root, device))
        gradient = {"schema_version": "P29R1_GRADIENT_DIAGNOSTIC_V1", "records": probe_records, "zero_init": _gradient_aggregate(probe_records, "zero_init"), "p29_checkpoint": _gradient_aggregate(probe_records, "p29_checkpoint")}
        normality = {"schema_version": "P29R1_NORMALITY_DIAGNOSTIC_V1", "classes": {result["class"]: result["normality"] for result in class_results}, "macro": _normality_macro(class_results)}
        recovery = _recovery_ratios()
        decision = _decision(class_results, gradient)
        runtime = {"wall_seconds": time.perf_counter() - started, "device": str(device), "classes_processed": len(class_results), "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0, "cache_mode": "sequential Tier-A/Tier-B memmap; one held class at a time; bounded source probes"}
        metrics = {"schema_version": "P29R1_FORENSIC_METRICS_V1", "identity": {"p29_terminal_sha": attempt["p29_terminal_sha"], "p29r1_prereg_sha": args.prereg_sha, "p29r1_execution_base_sha": args.execution_base_sha, "forensic_uuid": args.forensic_uuid}, "classes": class_results, "macro": {"p27_sign_agreement": decision["p27_sign_macro"], "p29_sign_agreement": decision["sign_macro"], "p29_sign_delta_vs_p27": decision["sign_delta_vs_frozen_p27"], "p27_residual_magnitude_mean": _mean([row["p27_residual_magnitude"]["mean_abs"] for row in class_results]), "p29_residual_magnitude_mean": _mean([row["p29_residual_magnitude"]["mean_abs"] for row in class_results]), "p29_over_p27_residual_magnitude_ratio": decision["p29_over_p27_residual_magnitude_ratio"]}, "output_recovery": recovery, "runtime": runtime, "input_inventory": input_inventory}
        _atomic_json(output / "P29R1_FORENSIC_METRICS.json", metrics)
        _atomic_json(output / "P29R1_GRADIENT_DIAGNOSTIC.json", gradient)
        _atomic_json(output / "P29R1_NORMALITY_DIAGNOSTIC.json", normality)
        _atomic_csv(output / "P29R1_SIGN_ALIGNMENT.csv", ("class", "p27_sign_agreement", "p29_sign_agreement", "p29_minus_p27_sign_agreement", "p29_pearson", "p29_spearman", "p29_mae", "p29_robust_relative_magnitude_ratio"), [{"class": result["class"], "p27_sign_agreement": result["p27_alignment"]["sign_agreement"], "p29_sign_agreement": result["p29_alignment"]["sign_agreement"], "p29_minus_p27_sign_agreement": result["alignment_delta"]["sign_agreement"], "p29_pearson": result["p29_alignment"]["pearson"], "p29_spearman": result["p29_alignment"]["spearman"], "p29_mae": result["p29_alignment"]["mae"], "p29_robust_relative_magnitude_ratio": result["p29_alignment"]["robust_relative_magnitude_ratio"]} for result in class_results])
        _atomic_json(output / "P29R1_DECISION_TREE.json", decision)
        post_inventory = _artifact_inventory(args.cache_root, args.metadata)
        audit = _post_run_audit(args=args, marker=attempt, input_inventory=input_inventory, post_inventory=post_inventory, classes=class_results, probe_records=probe_records, metrics=metrics, decision=decision)
        _write_report(metrics, gradient, decision, runtime)
        return {"status": "P29R1_FORENSIC_COMPLETE", "metrics": metrics, "audit": audit}
    except Exception as exc:
        _atomic_json(output / "P29R1_FORENSIC_FAILURE.json", {"schema_version": "P29R1_FORENSIC_FAILURE_V1", "forensic_uuid": args.forensic_uuid, "error_type": type(exc).__name__, "error": str(exc), "status": "P29R1_ENGINEERING_STOP", "rerun_forbidden": True})
        raise


def audit_completed_run(args: argparse.Namespace) -> dict[str, Any]:
    """Complete deterministic terminalization if execution stopped after outputs."""
    validate_runner_execution_contract()
    _protocol_payload(args.protocol)
    marker = _read_json(args.output_dir / "P29R1_FORENSIC_ATTEMPT.json")
    if marker.get("status") != "CONSUMED" or marker.get("forensic_attempt_count") != 1:
        raise RuntimeError("invalid or absent P29R1 attempt marker")
    if _git("rev-parse", "HEAD") != marker.get("p29r1_execution_base_sha"):
        raise RuntimeError("audit must run on the frozen execution base")
    input_inventory = marker["input_artifacts"]
    post_inventory = _artifact_inventory(args.cache_root, args.metadata)
    metrics = _read_json(args.output_dir / "P29R1_FORENSIC_METRICS.json")
    gradient = _read_json(args.output_dir / "P29R1_GRADIENT_DIAGNOSTIC.json")
    decision = _read_json(args.output_dir / "P29R1_DECISION_TREE.json")
    audit = _post_run_audit(args=args, marker=marker, input_inventory=input_inventory, post_inventory=post_inventory, classes=metrics["classes"], probe_records=gradient["records"], metrics=metrics, decision=decision)
    _write_report(metrics, gradient, decision, metrics["runtime"])
    return {"status": audit["terminal_status"], "audit": audit}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "preaudit", "preflight", "run"), required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--protocol", type=Path, default=OUTPUT_ROOT / "P29R1_FORENSIC_PROTOCOL.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--preaudit-output", type=Path, default=OUTPUT_ROOT / "P29R1_PRE_AUDIT.json")
    parser.add_argument("--preflight-output", type=Path, default=OUTPUT_ROOT / "P29R1_ENGINEERING_PREFLIGHT.json")
    parser.add_argument("--execution-base-sha")
    parser.add_argument("--prereg-sha")
    parser.add_argument("--forensic-uuid")
    parser.add_argument("--utc-started")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.mode == "preaudit":
        result = pre_audit(args)
    elif args.mode == "preflight":
        result = preflight(args)
    elif args.mode == "audit":
        result = audit_completed_run(args)
    else:
        result = run(args)
    print(json.dumps(_jsonify(result), sort_keys=True))


if __name__ == "__main__":
    main()
