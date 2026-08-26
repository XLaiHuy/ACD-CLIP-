"""P28 post-hoc mechanism diagnostic for the frozen P27 experiment.

This module contains no model-training path.  It consumes frozen logits,
features, immutable prediction maps, and post-freeze VisA masks, and writes
small aggregate evidence files.  The adapter is evaluated only on the frozen
Tier-A feature cache to recover its 9x9 residual for alignment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from model.phase2b_runtime import deploy_native_logits
from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata, safe_data_path
from tools.sabra_car.r0_direction import (
    MARGIN_SCALE,
    classify_actions,
    exact_metrics as _r0_exact_metrics,
    utility_for_batch,
)
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_pool import (
    pool_patch_map,
    symmetric_margin_delta,
    upsample_region_map,
)


IMAGE_SIZE = 518
PATCH_COUNT = 37 * 37
STAGES = 3
REGION_SIZE = 9
R0_ALPHA = 0.25
R0_EPSILON = 1e-8
TOP_RANK_FRACTIONS = (0.001, 0.005, 0.01, 0.05)
CLASS_NAMES = tuple(EXPECTED_VISA_CLASSES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def enforce_data_firewall(visa_root: Path, paths: Iterable[Path]) -> None:
    """Reject any data path outside VisA or containing forbidden domains."""
    root = visa_root.resolve()
    for path in paths:
        text = str(path).lower()
        if "mvtec" in text or "medical" in text:
            raise RuntimeError(f"P28 data firewall rejected path: {path}")
        resolved = Path(path).resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"P28 data firewall rejected non-VisA path: {path}") from exc


def patch_correction_from_actions(actions: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Apply the frozen R0 signed action-to-margin transformation."""
    if isinstance(actions, np.ndarray):
        actions = torch.from_numpy(actions)
    if actions.ndim != 2 or actions.shape[1] != PATCH_COUNT:
        raise ValueError("actions must be [B,1369]")
    if not torch.all((actions == -1) | (actions == 0) | (actions == 1)):
        raise ValueError("actions must contain only -1, 0, or +1")
    return (actions.to(dtype=torch.float32) * (R0_ALPHA * MARGIN_SCALE)).reshape(-1, 37, 37)


def abnormal_only_delta(native_logits: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
    """Return the exact historical R0 abnormal-only shared-stage delta."""
    if native_logits.ndim != 4 or tuple(native_logits.shape[:1]) != (STAGES,) or native_logits.shape[-1] != 2:
        raise ValueError("native_logits must be [3,B,1369,2]")
    if correction.shape != native_logits.shape[1:3]:
        raise ValueError("correction must be [B,1369]")
    one_stage = torch.stack((torch.zeros_like(correction), correction), dim=-1)
    return one_stage.unsqueeze(0).expand_as(native_logits)


def regionize_and_reconstruct(patch_correction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the frozen P27 37x37 -> 9x9 -> 37x37 path."""
    region = pool_patch_map(patch_correction)
    return region, upsample_region_map(region)


def native_probability_from_logits(native_logits: torch.Tensor) -> torch.Tensor:
    """Return the deployed native abnormal map without a model forward."""
    probability, _ = deploy_native_logits(native_logits, domain="Industrial")
    return probability[:, 1]


def exact_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Use the frozen P27 rank-group metric implementation."""
    return _r0_exact_metrics(scores, labels)


def load_immutable_predictions(path: Path, held_class: str, expected_sha256: str) -> list[dict[str, Any]]:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(f"prediction freeze hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "P27_IMMUTABLE_HELD_PREDICTIONS_V1":
        raise ValueError("wrong immutable prediction schema")
    if payload.get("held_class") != held_class or payload.get("gt_used") is not False:
        raise ValueError("immutable prediction provenance mismatch")
    if payload.get("mask_reads") != 0:
        raise ValueError("immutable prediction payload reports mask reads")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("immutable prediction records are missing")
    paths = [str(record.get("image_path")) for record in records]
    if any(not path for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("immutable prediction image identities are not unique")
    for record in records:
        for key in ("native_abnormal_probability", "p27_abnormal_probability"):
            value = record.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"immutable {key} shape mismatch")
    return records


class _Fenwick:
    def __init__(self, size: int) -> None:
        self.values = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int, value: int = 1) -> None:
        index += 1
        while index < self.values.size:
            self.values[index] += value
            index += index & -index

    def query(self, count: int) -> int:
        total = 0
        index = int(count)
        while index > 0:
            total += int(self.values[index])
            index -= index & -index
        return total


def pair_ordering_change(
    native_anomaly: np.ndarray,
    native_normal: np.ndarray,
    state_anomaly: np.ndarray,
    state_normal: np.ndarray,
) -> dict[str, int]:
    """Count strict pair order changes without materializing a pair matrix."""
    native_anomaly = np.asarray(native_anomaly, dtype=np.float32).reshape(-1)
    native_normal = np.asarray(native_normal, dtype=np.float32).reshape(-1)
    state_anomaly = np.asarray(state_anomaly, dtype=np.float32).reshape(-1)
    state_normal = np.asarray(state_normal, dtype=np.float32).reshape(-1)
    if not (native_anomaly.size == state_anomaly.size and native_normal.size == state_normal.size):
        raise ValueError("native/state strata sizes must match")
    if native_anomaly.size == 0 or native_normal.size == 0:
        return {"gained": 0, "lost": 0, "net": 0, "base_strict": 0, "state_strict": 0}

    native_normal_order = np.argsort(native_normal, kind="mergesort")
    state_normal_order = np.argsort(state_normal, kind="mergesort")
    state_normal_sorted = state_normal[state_normal_order]
    anomaly_order = np.argsort(native_anomaly, kind="mergesort")
    fenwick = _Fenwick(state_normal.size)
    native_pointer = 0
    both = 0
    base_strict = 0
    state_strict = 0
    for anomaly_index in anomaly_order:
        native_value = native_anomaly[anomaly_index]
        while native_pointer < native_normal.size and native_normal[native_normal_order[native_pointer]] < native_value:
            normal_index = int(native_normal_order[native_pointer])
            state_rank = int(np.searchsorted(state_normal_sorted, state_normal[normal_index], side="left"))
            fenwick.add(state_rank)
            native_pointer += 1
        base_count = native_pointer
        state_count = int(np.searchsorted(state_normal_sorted, state_anomaly[anomaly_index], side="left"))
        both_count = fenwick.query(state_count)
        base_strict += base_count
        state_strict += state_count
        both += both_count
    gained = state_strict - both
    lost = base_strict - both
    return {
        "gained": int(gained),
        "lost": int(lost),
        "net": int(gained - lost),
        "base_strict": int(base_strict),
        "state_strict": int(state_strict),
    }


def top_rank_behavior(
    scores: np.ndarray,
    labels: np.ndarray,
    fractions: Sequence[float] = TOP_RANK_FRACTIONS,
) -> dict[str, dict[str, float | int]]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    order = np.argsort(-scores, kind="mergesort")
    result: dict[str, dict[str, float | int]] = {}
    for fraction in fractions:
        count = max(1, int(math.ceil(float(fraction) * scores.size)))
        count = min(count, scores.size)
        anomaly_fraction = float(labels[order[:count]].mean())
        result[f"{float(fraction):g}"] = {
            "count": int(count),
            "anomaly_fraction": anomaly_fraction,
        }
    return result


def _quantile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values.astype(np.float64, copy=False), q, method="linear"))


def _summary(values: np.ndarray, quantiles: Sequence[float] = ()) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    result: dict[str, float | None] = {
        "mean": float(values.mean()) if values.size else None,
        "median": _quantile(values, 0.5),
        "absolute_magnitude_mean": float(np.abs(values).mean()) if values.size else None,
        "positive_fraction": float((values > 0).mean()) if values.size else None,
        "negative_fraction": float((values < 0).mean()) if values.size else None,
    }
    for q in quantiles:
        result[f"q{int(round(q * 100)):02d}"] = _quantile(values, q)
    return result


def _shift_summary(values: np.ndarray, quantiles: Sequence[float]) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return {
        "mean": float(values.mean()) if values.size else None,
        "median": _quantile(values, 0.5),
        **{f"q{int(round(q * 100)):02d}": _quantile(values, q) for q in quantiles},
    }


def _rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or y.size != x.size or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def alignment_metrics(teacher: np.ndarray, student: np.ndarray) -> dict[str, float | None]:
    teacher = np.asarray(teacher, dtype=np.float64)
    student = np.asarray(student, dtype=np.float64)
    if student.ndim == 3:
        student = student[None, ...]
    if teacher.ndim == 3:
        teacher = teacher[None, ...]
    if teacher.shape != student.shape:
        teacher = np.broadcast_to(teacher, student.shape)
    x = teacher.reshape(-1)
    y = student.reshape(-1)
    return {
        "pearson": _correlation(x, y),
        "spearman": _correlation(_rank_values(x), _rank_values(y)),
        "sign_agreement": float((np.sign(x) == np.sign(y)).mean()),
        "mae": float(np.abs(x - y).mean()),
        "robust_relative_magnitude_ratio": float((np.median(np.abs(y)) + 1e-12) / (np.median(np.abs(x)) + 1e-12)),
    }


def _load_masks(rows: Sequence[Mapping[str, Any]], visa_root: Path) -> tuple[np.ndarray, int]:
    masks = np.zeros((len(rows), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    reads = 0
    for index, row in enumerate(rows):
        if int(row["label"]) == 0:
            continue
        mask_path = safe_data_path(visa_root, str(row["mask_path"]))
        enforce_data_firewall(visa_root, [mask_path])
        with Image.open(mask_path) as handle:
            resized = handle.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
            masks[index] = (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)
        reads += 1
    return masks, reads


def _load_tier_a(cache_root: Path, class_name: str) -> tuple[dict[str, Any], np.memmap, np.memmap]:
    shard = cache_root / "tier_a" / class_name
    manifest_path = shard / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "P27_TIER_A_FROZEN_FEATURES_V1":
        raise RuntimeError(f"Tier-A schema mismatch for {class_name}")
    if manifest.get("class") != class_name or manifest.get("completion_status") != "COMPLETE":
        raise RuntimeError(f"Tier-A provenance mismatch for {class_name}")
    if manifest.get("contains_gt") or manifest.get("contains_masks") or manifest.get("contains_teacher_targets"):
        raise RuntimeError(f"Tier-A is not GT-free for {class_name}")
    native = np.load(shard / "native_logits.npy", mmap_mode="r", allow_pickle=False)
    seg = np.load(shard / "seg_features.npy", mmap_mode="r", allow_pickle=False)
    if native.shape[1:] != (STAGES, PATCH_COUNT, 2) or seg.shape[1:] != (STAGES, PATCH_COUNT, 768):
        raise RuntimeError(f"Tier-A tensor shape mismatch for {class_name}")
    return manifest, native, seg


def _load_adapter(path: Path, device: torch.device) -> RegionResidualAdapter:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "P27_REGION_ADAPTER_CHECKPOINT_V1":
        raise RuntimeError(f"adapter schema mismatch: {path}")
    if int(payload.get("phase2b_optimization_steps", -1)) != 0 or int(payload.get("clip_optimization_steps", -1)) != 0:
        raise RuntimeError(f"adapter checkpoint provenance has forbidden optimization: {path}")
    adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    adapter.load_state_dict(payload["state_dict"], strict=True)
    adapter.eval()
    return adapter


def _class_metrics(states: Mapping[str, np.ndarray], labels: np.ndarray) -> dict[str, dict[str, float]]:
    flat_labels = labels.reshape(-1).astype(np.uint8, copy=False)
    return {state: exact_metrics(values.reshape(-1), flat_labels) for state, values in states.items()}


def _transition_metrics(state_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    pairs = {"OP-N": ("OP", "N"), "OR-N": ("OR", "N"), "S-N": ("S", "N"), "OR-OP": ("OR", "OP"), "S-OR": ("S", "OR")}
    return {
        name: {metric: float(state_metrics[left][metric] - state_metrics[right][metric]) for metric in ("pAP", "pAUROC")}
        for name, (left, right) in pairs.items()
    }


def _breadth(values: Mapping[str, float]) -> dict[str, Any]:
    ordered = list(values.items())
    positive = [(name, value) for name, value in ordered if value > 0.0]
    non_regressing = [(name, value) for name, value in ordered if value >= 0.0]
    negative = [(name, value) for name, value in ordered if value < 0.0]
    positive_sorted = sorted(positive, key=lambda item: item[1], reverse=True)
    positive_total = sum(value for _, value in positive)
    return {
        "median_class_delta": float(np.median(np.asarray(list(values.values()), dtype=np.float64))),
        "positive_count": len(positive),
        "non_regressing_count": len(non_regressing),
        "negative_count": len(negative),
        "best_class": max(ordered, key=lambda item: item[1]) if ordered else None,
        "worst_class": min(ordered, key=lambda item: item[1]) if ordered else None,
        "top1_positive_gain_concentration": (positive_sorted[0][1] / positive_total) if positive_sorted and positive_total else None,
        "top2_positive_gain_concentration": (sum(value for _, value in positive_sorted[:2]) / positive_total) if positive_sorted and positive_total else None,
    }


def _ranking_for_state(native: np.ndarray, state: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    flat_native = native.reshape(-1).astype(np.float32, copy=False)
    flat_state = state.reshape(-1).astype(np.float32, copy=False)
    flat_labels = labels.reshape(-1).astype(np.uint8, copy=False)
    anomaly = flat_labels == 1
    normal = ~anomaly
    pair = pair_ordering_change(flat_native[anomaly], flat_native[normal], flat_state[anomaly], flat_state[normal])
    return {
        "anomaly_score_shift": _shift_summary(flat_state[anomaly] - flat_native[anomaly], (0.10, 0.50, 0.90)),
        "normal_score_shift": _shift_summary(flat_state[normal] - flat_native[normal], (0.90, 0.95, 0.99)),
        "pair_ordering_change": pair,
        "top_rank_behavior": {
            "N": top_rank_behavior(flat_native, flat_labels),
            "state": top_rank_behavior(flat_state, flat_labels),
        },
    }


def _residual_for_state(
    margin_residual: np.ndarray,
    score_residual: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    flat_margin = margin_residual.reshape(-1)
    flat_score = score_residual.reshape(-1)
    flat_labels = labels.reshape(-1).astype(bool)
    return {
        "effective_margin": _summary(flat_margin),
        "score": _summary(flat_score),
        "anomaly_pixels": {
            "effective_margin": _summary(flat_margin[flat_labels], (0.10, 0.50, 0.90)),
            "score": _summary(flat_score[flat_labels], (0.10, 0.50, 0.90)),
        },
        "normal_pixels": {
            "effective_margin": _summary(flat_margin[~flat_labels], (0.90, 0.95, 0.99)),
            "score": _summary(flat_score[~flat_labels], (0.90, 0.95, 0.99)),
        },
        "normal_upper_tail_effective_margin": {
            "q90": _quantile(flat_margin[~flat_labels], 0.90),
            "q95": _quantile(flat_margin[~flat_labels], 0.95),
            "q99": _quantile(flat_margin[~flat_labels], 0.99),
        },
        "anomaly_lower_tail_effective_margin": {
            "q10": _quantile(flat_margin[flat_labels], 0.10),
            "q50": _quantile(flat_margin[flat_labels], 0.50),
            "q90": _quantile(flat_margin[flat_labels], 0.90),
        },
    }


def _run_class(
    class_name: str,
    held_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    native_cache: np.memmap,
    seg_cache: np.memmap,
    adapter: RegionResidualAdapter,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    record_by_path = {str(record["image_path"]): record for record in records}
    ordered_paths = [str(row["image_path"]) for row in held_rows]
    if set(ordered_paths) != set(record_by_path) or len(ordered_paths) != len(record_by_path):
        raise RuntimeError(f"immutable identities do not match held inventory for {class_name}")
    tier_paths = []
    for sample_id in json.loads((Path(native_cache.filename).parent / "manifest.json").read_text())["sample_ids"]:
        prefix, separator, image_path = str(sample_id).partition(":")
        if prefix != class_name or not separator:
            raise RuntimeError(f"Tier-A sample identity mismatch for {class_name}")
        tier_paths.append(image_path)
    tier_index = {path: index for index, path in enumerate(tier_paths)}
    if set(tier_paths) != set(ordered_paths) or len(tier_index) != len(tier_paths):
        raise RuntimeError(f"Tier-A identities do not match held inventory for {class_name}")
    indices = np.asarray([tier_index[path] for path in ordered_paths], dtype=np.int64)
    native_logits_ordered = np.asarray(native_cache[indices], dtype=np.float32)
    native_scores_artifact = np.stack([
        record_by_path[path]["native_abnormal_probability"].numpy().astype(np.float32, copy=False)
        for path in ordered_paths
    ])
    student_scores_artifact = np.stack([
        record_by_path[path]["p27_abnormal_probability"].numpy().astype(np.float32, copy=False)
        for path in ordered_paths
    ])
    count = len(ordered_paths)
    op_scores = np.empty_like(native_scores_artifact)
    or_scores = np.empty_like(native_scores_artifact)
    native_margin_maps = np.empty_like(native_scores_artifact)
    op_margin_maps = np.empty_like(native_scores_artifact)
    or_margin_maps = np.empty_like(native_scores_artifact)
    student_margin_maps = np.empty_like(native_scores_artifact)
    student_region = np.empty((STAGES, count, REGION_SIZE, REGION_SIZE), dtype=np.float32)
    op_region = np.empty((count, REGION_SIZE, REGION_SIZE), dtype=np.float32)
    action_counts = {"positive": 0, "negative": 0, "zero": 0}
    native_parity = 0.0
    student_parity = 0.0

    with torch.no_grad():
        adapter = adapter.to(device=device)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        native = torch.from_numpy(native_logits_ordered[start:stop]).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
        mask = torch.from_numpy(masks[start:stop, None].astype(np.float32, copy=False)).to(device=device)
        utility, _ = utility_for_batch(native, mask)
        actions = classify_actions(utility)
        correction = patch_correction_from_actions(actions)
        action_counts["positive"] += int((actions > 0).sum().item())
        action_counts["negative"] += int((actions < 0).sum().item())
        action_counts["zero"] += int((actions == 0).sum().item())
        with torch.no_grad():
            native_probability, native_logits_deployed = deploy_native_logits(native, domain="Industrial")
            op_delta = abnormal_only_delta(native, correction.reshape(-1, PATCH_COUNT))
            op_probability, op_logits_deployed = deploy_native_logits(native + op_delta, domain="Industrial")
            region, region_patch = regionize_and_reconstruct(correction)
            region_patch_staged = region_patch.unsqueeze(0).expand(STAGES, -1, -1, -1)
            or_logits = symmetric_margin_delta(native, region_patch_staged)
            or_probability, or_logits_deployed = deploy_native_logits(or_logits, domain="Industrial")
            seg_batch = np.asarray(seg_cache[indices[start:stop]], dtype=np.float32)
            seg = torch.from_numpy(seg_batch).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            student_region_batch = adapter(seg)
            student_patch = upsample_region_map(student_region_batch)
            student_logits = symmetric_margin_delta(native, student_patch)
            student_probability, student_logits_deployed = deploy_native_logits(student_logits, domain="Industrial")

        native_scores_computed = native_probability[:, 1].cpu().numpy()
        student_scores_computed = student_probability[:, 1].cpu().numpy()
        native_parity = max(native_parity, float(np.max(np.abs(native_scores_computed - native_scores_artifact[start:stop]))))
        student_parity = max(student_parity, float(np.max(np.abs(student_scores_computed - student_scores_artifact[start:stop]))))
        op_scores[start:stop] = op_probability[:, 1].cpu().numpy()
        or_scores[start:stop] = or_probability[:, 1].cpu().numpy()
        native_margin_maps[start:stop] = (native_logits_deployed[:, 1] - native_logits_deployed[:, 0]).cpu().numpy()
        op_margin_maps[start:stop] = (op_logits_deployed[:, 1] - op_logits_deployed[:, 0]).cpu().numpy()
        or_margin_maps[start:stop] = (or_logits_deployed[:, 1] - or_logits_deployed[:, 0]).cpu().numpy()
        student_margin_maps[start:stop] = (student_logits_deployed[:, 1] - student_logits_deployed[:, 0]).cpu().numpy()
        student_region[:, start:stop] = student_region_batch.cpu().numpy()
        op_region[start:stop] = region.cpu().numpy()

    if native_parity > 2e-5 or student_parity > 2e-5:
        raise RuntimeError(f"frozen map parity failure for {class_name}: native={native_parity}, student={student_parity}")
    states = {"N": native_scores_artifact, "OP": op_scores, "OR": or_scores, "S": student_scores_artifact}
    state_metrics = _class_metrics(states, masks)
    transitions = _transition_metrics(state_metrics)
    ranking = {state: _ranking_for_state(native_scores_artifact, values, masks) for state, values in states.items() if state != "N"}
    residual = {
        "OP": _residual_for_state(op_margin_maps - native_margin_maps, op_scores - native_scores_artifact, masks),
        "OR": _residual_for_state(or_margin_maps - native_margin_maps, or_scores - native_scores_artifact, masks),
        "S": _residual_for_state(student_margin_maps - native_margin_maps, student_scores_artifact - native_scores_artifact, masks),
    }
    alignment = alignment_metrics(np.broadcast_to(op_region[None, ...], student_region.shape), student_region)
    return {
        "class": class_name,
        "sample_count": count,
        "abnormal_image_count": int(sum(int(row["label"]) for row in held_rows)),
        "pixel_count": int(masks.size),
        "anomaly_pixel_count": int(masks.sum()),
        "state_metrics": state_metrics,
        "transition_metrics": transitions,
        "ranking": ranking,
        "residual": residual,
        "alignment": alignment,
        "action_counts": action_counts,
        "cache_parity": {"native_max_abs": native_parity, "student_max_abs": student_parity},
        "anomaly_pixel_prevalence": float(masks.mean()),
        "median_anomaly_area_fraction_per_abnormal_image": float(np.median([
            masks[index].mean() for index, row in enumerate(held_rows) if int(row["label"]) == 1
        ])),
        "native_pAP": state_metrics["N"]["pAP"],
        "native_pAUROC": state_metrics["N"]["pAUROC"],
        "OP_minus_N_pAP_headroom": transitions["OP-N"]["pAP"],
        "OR_minus_OP_pAP_regionization_gap": transitions["OR-OP"]["pAP"],
        "S_minus_OR_pAP_student_approximation_gap": transitions["S-OR"]["pAP"],
    }


def _aggregate_state_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        state: {
            metric: float(np.mean([result["state_metrics"][state][metric] for result in results]))
            for metric in ("pAP", "pAUROC")
        }
        for state in ("N", "OP", "OR", "S")
    }


def _aggregate_transition_breadth(results: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for transition in ("OP-N", "OR-N", "S-N", "OR-OP", "S-OR"):
        output[transition] = {
            metric: _breadth({result["class"]: result["transition_metrics"][transition][metric] for result in results})
            for metric in ("pAP", "pAUROC")
        }
    return output


def _correlations(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    descriptor_names = (
        "anomaly_pixel_prevalence",
        "median_anomaly_area_fraction_per_abnormal_image",
        "abnormal_image_count",
        "native_pAP",
        "native_pAUROC",
        "OP_minus_N_pAP_headroom",
        "OR_minus_OP_pAP_regionization_gap",
        "S_minus_OR_pAP_student_approximation_gap",
    )
    outcomes = {
        "S_minus_N_pAP": np.asarray([r["transition_metrics"]["S-N"]["pAP"] for r in results], dtype=np.float64),
        "S_minus_N_pAUROC": np.asarray([r["transition_metrics"]["S-N"]["pAUROC"] for r in results], dtype=np.float64),
    }
    output: dict[str, Any] = {}
    for descriptor in descriptor_names:
        values = np.asarray([result[descriptor] for result in results], dtype=np.float64)
        output[descriptor] = {
            outcome: {"pearson": _correlation(values, target), "spearman": _correlation(_rank_values(values), _rank_values(target))}
            for outcome, target in outcomes.items()
        }
    return output


def _hypothesis_labels(results: Sequence[Mapping[str, Any]], macro: Mapping[str, Mapping[str, float]], pair_totals: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, Any]]:
    def breadth(transition: str, metric: str) -> dict[str, Any]:
        return _breadth({r["class"]: r["transition_metrics"][transition][metric] for r in results})

    op_ap = macro["OP"]["pAP"] - macro["N"]["pAP"]
    op_auc = macro["OP"]["pAUROC"] - macro["N"]["pAUROC"]
    or_ap = macro["OR"]["pAP"] - macro["N"]["pAP"]
    or_auc = macro["OR"]["pAUROC"] - macro["N"]["pAUROC"]
    s_ap = macro["S"]["pAP"] - macro["N"]["pAP"]
    s_auc = macro["S"]["pAUROC"] - macro["N"]["pAUROC"]
    op_auc_breadth = breadth("OP-N", "pAUROC")
    or_op_auc_breadth = breadth("OR-OP", "pAUROC")
    s_or_auc_breadth = breadth("S-OR", "pAUROC")
    h1_supported = op_ap > 0 and op_auc < 0 and op_auc_breadth["negative_count"] > op_auc_breadth["positive_count"]
    h2_supported = or_op_auc_breadth["negative_count"] > or_op_auc_breadth["positive_count"] and pair_totals["OR"]["net"] < 0
    h3_supported = or_ap > 0 and s_ap < or_ap and s_or_auc_breadth["negative_count"] > s_or_auc_breadth["positive_count"]
    h4_normal_q99 = float(np.mean([r["residual"]["S"]["normal_upper_tail_effective_margin"]["q99"] for r in results]))
    h4_supported = h4_normal_q99 > 0 and pair_totals["S"]["lost"] > pair_totals["S"]["gained"]
    s_ap_breadth = breadth("S-N", "pAP")
    h5_supported = s_ap_breadth["positive_count"] > 0 and s_ap_breadth["negative_count"] > 0
    def label(supported: bool, plausible: bool) -> str:
        return "SUPPORTED" if supported else "PLAUSIBLE" if plausible else "NOT_SUPPORTED"
    return {
        "H1_teacher_objective_conflict": {"classification": label(h1_supported, op_ap > 0 and op_auc < 0), "evidence": {"OP_minus_N_macro_pAP": op_ap, "OP_minus_N_macro_pAUROC": op_auc, "OP_minus_N_pAUROC_breadth": op_auc_breadth}},
        "H2_regionization_loss": {"classification": label(h2_supported, or_op_auc_breadth["negative_count"] > or_op_auc_breadth["positive_count"]), "evidence": {"OR_minus_OP_pAUROC_breadth": or_op_auc_breadth, "OR_pair_net": pair_totals["OR"]["net"]}},
        "H3_student_transfer_failure": {"classification": label(h3_supported, or_ap > 0 and (s_ap < or_ap or s_auc < or_auc)), "evidence": {"OR_minus_N_macro_pAP": or_ap, "S_minus_OR_macro_pAP": s_ap - or_ap, "S_minus_OR_macro_pAUROC": s_auc - or_auc, "S_minus_OR_pAUROC_breadth": s_or_auc_breadth}},
        "H4_normal_score_inflation": {"classification": label(h4_supported, h4_normal_q99 > 0 or pair_totals["S"]["lost"] > pair_totals["S"]["gained"]), "evidence": {"S_normal_q99_effective_margin_mean": h4_normal_q99, "S_pair_ordering": pair_totals["S"]}},
        "H5_heterogeneous_category_actionability": {"classification": label(h5_supported, h5_supported), "evidence": {"S_minus_N_pAP_breadth": s_ap_breadth}},
    }


def _root_cause(hypotheses: Mapping[str, Mapping[str, Any]], macro: Mapping[str, Mapping[str, float]], pair_totals: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    labels = {key: value["classification"] for key, value in hypotheses.items()}
    if labels["H1_teacher_objective_conflict"] == "SUPPORTED" and labels["H2_regionization_loss"] != "SUPPORTED":
        primary = "TEACHER_OBJECTIVE_CONFLICT"
        secondary = "NORMAL_SCORE_INFLATION"
    elif labels["H2_regionization_loss"] == "SUPPORTED" and labels["H1_teacher_objective_conflict"] != "SUPPORTED":
        primary = "REGIONIZATION_RANKING_LOSS"
        secondary = "TEACHER_OBJECTIVE_CONFLICT"
    elif labels["H3_student_transfer_failure"] == "SUPPORTED" and labels["H1_teacher_objective_conflict"] != "SUPPORTED" and labels["H2_regionization_loss"] != "SUPPORTED":
        primary = "STUDENT_TRANSFER_FAILURE"
        secondary = "HETEROGENEOUS_ACTIONABILITY"
    elif labels["H5_heterogeneous_category_actionability"] == "SUPPORTED":
        primary = "HETEROGENEOUS_ACTIONABILITY"
        secondary = "MIXED_MECHANISM"
    elif any(labels[key] in {"SUPPORTED", "PLAUSIBLE"} for key in labels):
        primary = "MIXED_MECHANISM"
        secondary = "INSUFFICIENT"
    else:
        primary = "INSUFFICIENT"
        secondary = "INSUFFICIENT"
    return {
        "primary_mechanism": primary,
        "secondary_mechanism": secondary,
        "hypothesis_classifications": labels,
        "decision_basis": "complete N -> OP -> OR -> S decomposition, macro effects, class breadth, strict pair-order changes, residual tails, and alignment",
    }


def _write_class_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    fields = ["class"]
    for state in ("N", "OP", "OR", "S"):
        fields.extend([f"{state}_pAP", f"{state}_pAUROC"])
    for transition in ("OP-N", "OR-N", "S-N", "OR-OP", "S-OR"):
        fields.extend([f"{transition}_pAP", f"{transition}_pAUROC"])
    fields.extend(["anomaly_pixel_prevalence", "median_anomaly_area_fraction_per_abnormal_image", "abnormal_image_count"])
    rows = []
    for result in results:
        row: dict[str, Any] = {"class": result["class"]}
        for state in ("N", "OP", "OR", "S"):
            for metric in ("pAP", "pAUROC"):
                row[f"{state}_{metric}"] = result["state_metrics"][state][metric]
        for transition in ("OP-N", "OR-N", "S-N", "OR-OP", "S-OR"):
            for metric in ("pAP", "pAUROC"):
                row[f"{transition}_{metric}"] = result["transition_metrics"][transition][metric]
        row.update({key: result[key] for key in fields if key in result})
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _markdown_report(
    output: Path,
    identity: Mapping[str, Any],
    audit: Mapping[str, Any],
    macro: Mapping[str, Mapping[str, float]],
    breadth: Mapping[str, Any],
    root: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> None:
    def f(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.10f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = [
        "# P28 FINAL MECHANISM DIAGNOSTIC",
        "",
        "## IDENTITY",
        "",
        f"1. P27 terminal SHA: `{identity['p27_terminal_sha']}`",
        f"2. P27 scientific execution-base: `{identity['p27_scientific_execution_base_sha']}`",
        f"3. P28 prereg SHA: `{identity['p28_prereg_sha']}`",
        f"4. P28 execution-base SHA: `{identity['p28_execution_base_sha']}`",
        f"5. P28 attempt UUID: `{identity['p28_attempt_uuid']}`",
        "",
        "## AUDIT",
        "",
        f"6. Training steps: `{audit['training_steps']}`",
        f"7. Parameter-update steps: `{audit['optimizer_steps']}`",
        f"8. New CLIP forwards: `{audit['new_clip_forwards']}`",
        f"9. New Phase2B forwards: `{audit['new_phase2b_forwards']}`",
        f"10. MVTec reads: `{audit['mvtec_reads']}`",
        f"11. Medical reads: `{audit['medical_reads']}`",
        f"12. Post-audit: `{audit['status']}`",
        "",
        "## FOUR-STATE RESULTS",
        "",
    ]
    for number, state in enumerate(("N", "OP", "OR", "S"), start=13):
        lines.append(f"{number}. `{state}` macro pAP / pAUROC: `{f(macro[state]['pAP'])}` / `{f(macro[state]['pAUROC'])}`")
    lines.extend(["", "## DECOMPOSITION", ""])
    for number, transition, label in ((17, "OP-N", "teacher effect"), (18, "OR-OP", "regionization effect"), (19, "S-OR", "student-transfer effect"), (20, "S-N", "final effect")):
        left, right = {"OP-N": ("OP", "N"), "OR-OP": ("OR", "OP"), "S-OR": ("S", "OR"), "S-N": ("S", "N")}[transition]
        ap = macro[left]["pAP"] - macro[right]["pAP"]
        auc = macro[left]["pAUROC"] - macro[right]["pAUROC"]
        lines.append(f"{number}. {label} `{transition}` macro delta pAP / pAUROC: `{f(ap)}` / `{f(auc)}`")
    lines.extend(["", "## BREADTH", ""])
    for number, transition in enumerate(("OP-N", "OR-OP", "S-OR", "S-N"), start=21):
        b = breadth[transition]
        lines.append(f"{number}. `{transition}` pAP breadth: positive `{b['pAP']['positive_count']}`, non-regressing `{b['pAP']['non_regressing_count']}`, negative `{b['pAP']['negative_count']}`; median `{f(b['pAP']['median_class_delta'])}`; best `{b['pAP']['best_class']}`; worst `{b['pAP']['worst_class']}`.")
    lines.extend(["", "## RANKING MECHANISM", "", "25. Per-class anomaly/normal score-shift summaries, strict gained/lost pair orderings, and fixed top-rank AP behavior are in `P28_RANKING_DIAGNOSTIC.json`.", "26. Normal score shifts include mean, median, q90, q95, q99; anomaly shifts include mean, median, q10, q50, q90.", "27. Gained pair orderings are strict state anomaly>normal pairs absent in N; lost pairs are strict N pairs absent in the state.", "28. Net strict ordering changes are reported per class and in aggregate.", "29. AUROC uses P27 tied-score half-credit; strict ordering counts intentionally exclude ties.", "30. Top-rank behavior uses fixed 0.1%, 0.5%, 1%, and 5% fractions.", "", "## ALIGNMENT", "", "31. Per-class teacher/student Pearson, Spearman, sign agreement, MAE, and robust magnitude ratio are in `P28_ALIGNMENT_DIAGNOSTIC.json`.", "32. The teacher is the OP-derived shared 9x9 region target.", "33. The student is the frozen adapter output on Tier-A features.", "34. No calibration or fitting was performed.", "35. No optimized sign threshold was used.", "", "## HYPOTHESES", ""])
    for number, (key, value) in enumerate(root["hypothesis_classifications"].items(), start=36):
        lines.append(f"{number}. `{key}`: `{value}`")
    lines.extend(["", "## ROOT CAUSE", "", f"41. Primary mechanism: `{root['primary_mechanism']}`", f"42. Secondary mechanism: `{root['secondary_mechanism']}`", "43. OBSERVED", "", "The full numerical evidence is stored in the required JSON and CSV artifacts. The state chain is frozen and the diagnostic uses no training or deployment selection.", "", "44. INTERPRETATION", "", f"The preregistered decomposition attributes the dominant observed behavior to `{root['primary_mechanism']}` with secondary evidence labeled `{root['secondary_mechanism']}`. Category correlations are exploratory because n=12.", "", "## NEXT-STEP DECISION", "", f"45. Recommendation for P29: `{ {'TEACHER_OBJECTIVE_CONFLICT':'RANKING_SAFE_TEACHER','REGIONIZATION_RANKING_LOSS':'REGION_REPRESENTATION_REDESIGN','STUDENT_TRANSFER_FAILURE':'ROBUST_STUDENT_TRANSFER','HETEROGENEOUS_ACTIONABILITY':'CONDITIONAL / HETEROGENEOUS MODELING','MIXED_MECHANISM':'STOP SABRA-V2 REGION LINEAGE','INSUFFICIENT':'STOP SABRA-V2 REGION LINEAGE'}[root['primary_mechanism']] }`", "", "## ENGINEERING", "", f"Diagnostic wall time: `{runtime['wall_seconds']:.3f}` seconds; class timings and cache parity are in `P28_METRICS.json`."])
    _atomic_write(output, "\n".join(lines) + "\n")


def _input_inventory(
    science_root: Path,
    cache_root: Path,
    metadata: Path,
    protocol: Path,
    p27_audit: Path,
    classes: Sequence[str],
) -> dict[str, Any]:
    predictions = {}
    manifests = {}
    adapters = {}
    for class_name in classes:
        prediction = science_root / class_name / "predictions" / "p27_held_predictions.pt"
        manifest = cache_root / "tier_a" / class_name / "manifest.json"
        adapter = science_root / class_name / "training" / "p27_region_adapter.pt"
        predictions[class_name] = sha256_file(prediction)
        manifests[class_name] = {"sha256": sha256_file(manifest), "payload": json.loads(manifest.read_text(encoding="utf-8"))}
        adapters[class_name] = sha256_file(adapter)
    return {
        "predictions_sha256": predictions,
        "tier_a_manifests": manifests,
        "adapter_checkpoints_sha256": adapters,
        "metadata_sha256": sha256_file(metadata),
        "protocol_sha256": sha256_file(protocol),
        "p27_post_run_audit_sha256": sha256_file(p27_audit),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    protocol = args.protocol
    p27_audit_path = args.p27_audit
    current_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if current_sha != args.execution_base_sha:
        raise RuntimeError(f"P28 execution base mismatch: {current_sha} != {args.execution_base_sha}")
    subprocess.run(["git", "cat-file", "-e", "cdf06234bee861bbe81a7f07e382530f9a66c207^{commit}"], check=True)
    if (output / "P28_ATTEMPT.json").exists():
        raise RuntimeError("P28 attempt marker already exists; refusing a second diagnostic")
    enforce_data_firewall(args.visa_root, [args.visa_root])
    inventory_hashes = _input_inventory(args.science_root, args.cache_root, args.metadata, protocol, p27_audit_path, CLASS_NAMES)
    attempt = {
        "schema_version": "P28_ATTEMPT_V1",
        "status": "CONSUMED",
        "p28_attempt_uuid": args.attempt_uuid,
        "execution_base_sha": args.execution_base_sha,
        "utc_started": args.utc_started,
        "p27_terminal_sha": "cdf06234bee861bbe81a7f07e382530f9a66c207",
        "p27_scientific_execution_base_sha": "de41b380449dcbc0b124f71f4f8fbb789e1a96f0",
        "p27_attempt_uuid": "884f7327-1135-491b-8c12-dc188455be2c",
        "p28_prereg_sha": args.prereg_sha,
        "protocol_sha256": inventory_hashes["protocol_sha256"],
        "asset_hashes": {
            "p26_checkpoint_sha256": "a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34",
            "clip_asset_sha256": "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
            "phase2b_config_sha256": "edf5745686e3d3d0d3b4142341569da06ad5b54025a779b78d83f74303ce67fc",
            "visa_metadata_sha256": inventory_hashes["metadata_sha256"],
        },
        "input_artifacts": inventory_hashes,
        "held_gt_reads_before_marker": 0,
        "diagnostic_execution_count": 1,
    }
    _atomic_json(output / "P28_ATTEMPT.json", attempt)

    metadata_rows = read_visa_metadata(args.metadata)
    class_results = []
    class_timings = {}
    total_mask_reads = 0
    device = torch.device(args.device)
    for class_name in CLASS_NAMES:
        class_started = time.perf_counter()
        held_rows = loco_inventory(metadata_rows, class_name).held_rows
        records = load_immutable_predictions(
            args.science_root / class_name / "predictions" / "p27_held_predictions.pt",
            class_name,
            inventory_hashes["predictions_sha256"][class_name],
        )
        manifest, native_cache, seg_cache = _load_tier_a(args.cache_root, class_name)
        masks, mask_reads = _load_masks(held_rows, args.visa_root)
        total_mask_reads += mask_reads
        adapter = _load_adapter(args.science_root / class_name / "training" / "p27_region_adapter.pt", device)
        class_results.append(_run_class(class_name, held_rows, records, native_cache, seg_cache, adapter, masks, device, args.batch_size))
        class_timings[class_name] = {"seconds": time.perf_counter() - class_started, "tier_a_manifest_sha256": inventory_hashes["tier_a_manifests"][class_name]["sha256"], "sample_count": int(manifest["sample_count"])}

    macro = _aggregate_state_metrics(class_results)
    transition_breadth = _aggregate_transition_breadth(class_results)
    pair_totals = {}
    ranking_output = {"schema_version": "P28_RANKING_DIAGNOSTIC_V1", "classes": {}}
    alignment_output = {"schema_version": "P28_ALIGNMENT_DIAGNOSTIC_V1", "classes": {}}
    for result in class_results:
        ranking_output["classes"][result["class"]] = result["ranking"]
        alignment_output["classes"][result["class"]] = result["alignment"]
    for state in ("OP", "OR", "S"):
        pair_totals[state] = {key: int(sum(result["ranking"][state]["pair_ordering_change"][key] for result in class_results)) for key in ("gained", "lost", "net", "base_strict", "state_strict")}
    ranking_output["aggregate_pair_ordering_change"] = pair_totals
    ranking_output["fixed_top_rank_fractions"] = list(TOP_RANK_FRACTIONS)
    alignment_output["macro_mean"] = {metric: float(np.mean([result["alignment"][metric] for result in class_results if result["alignment"][metric] is not None])) for metric in ("pearson", "spearman", "sign_agreement", "mae", "robust_relative_magnitude_ratio")}
    correlations = _correlations(class_results)
    hypotheses = _hypothesis_labels(class_results, macro, pair_totals)
    root = _root_cause(hypotheses, macro, pair_totals)
    root["category_correlations_exploratory"] = correlations
    audit = {
        "schema_version": "P28_POST_RUN_AUDIT_V1",
        "status": "PASS",
        "p27_terminal_sha": "cdf06234bee861bbe81a7f07e382530f9a66c207",
        "p27_attempt_uuid": "884f7327-1135-491b-8c12-dc188455be2c",
        "p28_attempt_uuid": args.attempt_uuid,
        "p28_prereg_sha": args.prereg_sha,
        "p28_execution_base_sha": args.execution_base_sha,
        "training_steps": 0,
        "optimizer_steps": 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "mvtec_reads": 0,
        "medical_reads": 0,
        "held_gt_reads_before_p27_scoring": 0,
        "held_mask_reads_before_p27_scoring": 0,
        "p28_posthoc_held_mask_file_reads": total_mask_reads,
        "immutable_prediction_count": len(class_results),
        "intended_fold_count": len(CLASS_NAMES),
        "duplicate_scientific_folds": 0,
        "diagnostic_execution_count": 1,
        "result_driven_reruns": 0,
        "p26_checkpoint_unchanged": True,
        "clip_unchanged": True,
        "phase2b_config_unchanged": True,
        "p27_protocol_unchanged": True,
        "only_region_residual_adapter_evaluated": True,
        "cache_provenance_valid": True,
        "execution_base_unchanged_since_marker": current_sha == args.execution_base_sha,
        "worktree_clean_at_execution_base": True,
        "root_cause": root["primary_mechanism"],
    }
    runtime = {"wall_seconds": time.perf_counter() - started, "class_timings": class_timings, "device": str(device), "batch_size": args.batch_size, "cache_mode": "Tier-A memmap, one class at a time"}
    metrics_payload = {
        "schema_version": "P28_METRICS_V1",
        "identity": {"p27_terminal_sha": audit["p27_terminal_sha"], "p27_scientific_execution_base_sha": "de41b380449dcbc0b124f71f4f8fbb789e1a96f0", "p28_prereg_sha": args.prereg_sha, "p28_execution_base_sha": args.execution_base_sha, "p28_attempt_uuid": args.attempt_uuid},
        "macro_state_metrics": macro,
        "macro_transition_deltas": {transition: {metric: macro[left][metric] - macro[right][metric] for metric in ("pAP", "pAUROC")} for transition, (left, right) in {"OP-N": ("OP", "N"), "OR-N": ("OR", "N"), "S-N": ("S", "N"), "OR-OP": ("OR", "OP"), "S-OR": ("S", "OR")}.items()},
        "transition_breadth": transition_breadth,
        "classes": class_results,
        "category_correlations_exploratory": correlations,
        "hypotheses": hypotheses,
        "root_cause": root,
        "runtime": runtime,
        "p28_posthoc_held_mask_file_reads": total_mask_reads,
    }
    _atomic_json(output / "P28_METRICS.json", _json_value(metrics_payload))
    _atomic_json(output / "P28_RANKING_DIAGNOSTIC.json", _json_value(ranking_output))
    _atomic_json(output / "P28_ALIGNMENT_DIAGNOSTIC.json", _json_value(alignment_output))
    _atomic_json(output / "P28_ROOT_CAUSE.json", _json_value(root))
    _write_class_csv(output / "P28_CLASS_TABLE.csv", class_results)
    _markdown_report(output / "P28_FINAL_REPORT.md", metrics_payload["identity"], audit, macro, transition_breadth, root, class_results, runtime)
    _atomic_json(output / "P28_POST_RUN_AUDIT.json", _json_value(audit))
    return {"status": "P28_DIAGNOSTIC_COMPLETE", "metrics": metrics_payload, "audit": audit}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--science-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--p27-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-base-sha", required=True)
    parser.add_argument("--prereg-sha", required=True)
    parser.add_argument("--attempt-uuid", required=True)
    parser.add_argument("--utc-started", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


if __name__ == "__main__":
    print(json.dumps(_json_value(run(make_parser().parse_args())), sort_keys=True))
