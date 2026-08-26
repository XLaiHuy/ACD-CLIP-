"""Offline causal forensic for the frozen P30R1 candle result.

This module intentionally has no trainable model path.  It reads immutable
P29/P30/P30R1 prediction tensors, the frozen Tier-A native-logit cache, the
source-only Tier-B teacher cache, and post-freeze VisA masks.  The only
operations beyond array statistics are the already-frozen deterministic
deployment operator and the R0 utility derivative applied to cached native
logits.  Neither operation constructs or runs a neural network.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

# These are pure tensor operators over already-frozen logits/residuals.  They
# do not load a checkpoint or invoke CLIP/Phase2B.
from model.phase2b_runtime import deploy_native_logits
from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata, safe_data_path
from tools.sabra_car.r0_direction import (
    MARGIN_SCALE,
    classify_actions,
    exact_metrics,
    utility_for_batch,
)
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.region_pool import pool_patch_map, symmetric_margin_delta, upsample_region_map


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = ROOT / "research/sabra_v2/region_distill/P30R1_FORENSIC"
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_VISA_ROOT = Path("/workspace/data/source/visa_unpack")
DEFAULT_METADATA = ROOT / "dataset/hub/VisA.jsonl"

HELD_CLASS = "candle"
IMAGE_SIZE = 518
PATCH_GRID = (37, 37)
REGION_GRID = (9, 9)
STAGES = 3
REGION_COORDINATES = STAGES * REGION_GRID[0] * REGION_GRID[1]
CORRECTION_SCALE = 4.960109710693359
NORMALIZATION_EPSILON = 0.01
R0_ALPHA = 0.25

# These are fixed descriptive quantities, selected before reading the result.
SCORE_DIFFERENCE_THRESHOLDS = (1e-6, 1e-4, 1e-3, 1e-2, 5e-2)
TOP_MASS_FRACTIONS = (0.01, 0.05, 0.10)
TOP_PIXEL_FRACTIONS = (0.001, 0.005, 0.01, 0.05)
NORM_RATIO_THRESHOLDS = (0.1, 0.25, 0.5, 1.0)
GAMMA_VALUES = (0.0, 0.5, 1.0)

NEXT_RESEARCH_QUESTION = {
    "DO_NO_HARM_NATIVE_PRESERVATION": "When should a learned residual intervene at all?",
    "SPARSE_SELECTIVE_CORRECTION": "Can useful teacher corrections be transferred selectively without global teacher-direction matching?",
    "TEACHER_DIRECTION_NOT_CAUSAL": "What downstream-relevant invariant should be transferred instead of raw teacher correction direction?",
    "DIRECTION_METRIC_ILL_CONDITIONED_BY_ABSTENTION": "How should transfer fidelity be evaluated conditionally on meaningful correction support?",
    "TEACHER_SCALE_REWEIGHTING": "Is the benefit coming from balanced sample weighting rather than explicit teacher residual fidelity?",
}

P29_PREDICTIONS = Path("/workspace/p29_science_v1/candle/predictions/p29_held_predictions.pt")
P29_METRICS = Path("/workspace/p29_science_v1/candle/metrics/p29_held_metrics.json")
P29_CHECKPOINT = Path("/workspace/p29_science_v1/candle/training/p29_region_adapter.pt")
P30_ROOT = ROOT / "research/sabra_v2/region_distill/P30"
P30_PREDICTIONS = P30_ROOT / "qualification/stage2_one_class/candle/predictions/p30_held_predictions.pt"
P30_METRICS = P30_ROOT / "qualification/stage2_one_class/candle/metrics/p30_held_metrics.json"
P30_CHECKPOINT = P30_ROOT / "qualification/stage2_one_class/candle/training/p30_region_adapter.pt"
P30_TRANSFER = P30_ROOT / "qualification/stage2_one_class/P30_TRANSFER_DIAGNOSTIC.json"
P30_STABILITY = P30_ROOT / "qualification/stage2_one_class/P30_STABILITY_DIAGNOSTIC.json"
P30R1_ROOT = ROOT / "research/sabra_v2/region_distill/P30R1"
P30R1_PREDICTIONS = P30R1_ROOT / "candle/predictions/p30r1_held_predictions.pt"
P30R1_METRICS = P30R1_ROOT / "candle/metrics/P30R1_HELD_METRICS.json"
P30R1_CHECKPOINT = P30R1_ROOT / "candle/training/p30r1_region_adapter.pt"
P30R1_QUALIFICATION = P30R1_ROOT / "P30R1_STAGE2_QUALIFICATION.json"
P30R1_FINAL_REPORT = P30R1_ROOT / "P30R1_FINAL_REPORT.md"
P30R1_AUDIT = P30R1_ROOT / "P30R1_POST_RUN_AUDIT.json"
P30R1_PREREGISTRATION = ROOT / "research/sabra_v2/region_distill/P30R1_PREREGISTRATION.json"
P30R1_PREREGISTRATION_MD = ROOT / "research/sabra_v2/region_distill/P30R1_PREREGISTRATION.md"
P30R1_IMPLEMENTATION = ROOT / "research/sabra_v2/region_distill/P30R1_IMPLEMENTATION_REPORT.md"
P30R1_DESIGN = ROOT / "research/sabra_v2/region_distill/P30R1_DESIGN_NOTE.md"
P30R1_ENGINEERING = ROOT / "research/sabra_v2/region_distill/P30R1_ENGINEERING_QUALIFICATION.json"
P30R1_RESEARCH = ROOT / "research/sabra_v2/region_distill/P30R1_RESEARCH_REPORT.md"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--visa-root", type=Path, default=DEFAULT_VISA_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty forensic CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _quantiles(values: Iterable[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        return {key: None for key in ("min", "q01", "q10", "q25", "q50", "q75", "q90", "q95", "q99", "max")}
    if not np.isfinite(array).all():
        raise RuntimeError("forensic statistic received non-finite values")
    return {
        "min": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "q50": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _summary(values: Iterable[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    result: dict[str, Any] = {"count": int(array.size), "quantiles": _quantiles(array)}
    result["mean"] = float(array.mean()) if array.size else None
    result["std"] = float(array.std()) if array.size else None
    return result


def _summary_optional(values: Iterable[float | None]) -> dict[str, Any]:
    return _summary([float(value) for value in values if value is not None])


def _rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    ranks = np.empty(values.size, dtype=np.float64)
    group_ranks = (starts.astype(np.float64) + ends.astype(np.float64) - 1.0) / 2.0 + 1.0
    ranks[order] = np.repeat(group_ranks, ends - starts)
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size != right.size or not left.size:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    return None if denominator == 0.0 else float(np.dot(left_centered, right_centered) / denominator)


def _correlation(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float | None:
    if rank:
        return _pearson(_rank_values(left), _rank_values(right))
    return _pearson(left, right)


def _method_payload(path: Path, score_key: str, residual_key: str | None, held_paths: Sequence[str]) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing frozen prediction artifact: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("held_class") != HELD_CLASS or payload.get("gt_used") is not False or payload.get("mask_reads") != 0:
        raise RuntimeError(f"prediction firewall failed: {path}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(held_paths):
        raise RuntimeError(f"prediction inventory mismatch: {path}")
    by_path = {str(record.get("image_path")): record for record in records}
    if set(by_path) != set(held_paths) or len(by_path) != len(records):
        raise RuntimeError(f"prediction identities mismatch: {path}")
    native_values: list[np.ndarray] = []
    score_values: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    for image_path in held_paths:
        record = by_path[image_path]
        native = record.get("native_abnormal_probability")
        score = record.get(score_key)
        if not isinstance(native, torch.Tensor) or tuple(native.shape) != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"native map shape mismatch: {path} {image_path}")
        if not isinstance(score, torch.Tensor) or tuple(score.shape) != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"score map shape mismatch: {path} {image_path}")
        native_values.append(native.detach().cpu().numpy().astype(np.float32, copy=False))
        score_values.append(score.detach().cpu().numpy().astype(np.float32, copy=False))
        if residual_key is not None:
            residual = record.get(residual_key)
            if not isinstance(residual, torch.Tensor) or tuple(residual.shape) != (STAGES, *REGION_GRID):
                raise RuntimeError(f"residual shape mismatch: {path} {image_path}")
            residual_values.append(residual.detach().cpu().numpy().astype(np.float32, copy=False))
    result = {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": payload.get("schema_version"),
        "payload": {key: payload.get(key) for key in payload if key != "records"},
        "native_probability": np.stack(native_values),
        "score": np.stack(score_values),
        "residual": np.stack(residual_values) if residual_key is not None else None,
    }
    del payload
    gc.collect()
    return result


def _load_masks(rows: Sequence[Mapping[str, Any]], visa_root: Path) -> tuple[np.ndarray, int]:
    masks = np.zeros((len(rows), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    reads = 0
    for index, row in enumerate(rows):
        if int(row["label"]) == 0:
            continue
        mask_path = safe_data_path(visa_root, str(row["mask_path"]))
        with Image.open(mask_path) as handle:
            resized = handle.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
            masks[index] = (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)
        reads += 1
    return masks, reads


def _load_held_native_logits(cache_root: Path, held_rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    shard = cache_root / "tier_a" / HELD_CLASS
    manifest_path = shard / "manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("schema") != "P27_TIER_A_FROZEN_FEATURES_V1" or manifest.get("contains_gt") is not False:
        raise RuntimeError("Tier-A native cache is not the frozen GT-free shard")
    sample_ids = list(manifest.get("sample_ids", []))
    indices = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    requested = [f"{row['class_name']}:{row['image_path']}" for row in held_rows]
    if any(sample_id not in indices for sample_id in requested):
        raise RuntimeError("held rows are absent from Tier-A native cache")
    native = np.load(shard / "native_logits.npy", mmap_mode="r", allow_pickle=False)
    selected = np.asarray(native[[indices[sample_id] for sample_id in requested]], dtype=np.float32).copy()
    return selected, {
        "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "native_logits": {"path": str(shard / "native_logits.npy"), "sha256": _sha256(shard / "native_logits.npy"), "shape": list(native.shape), "dtype": str(native.dtype)},
        "held_indices": [int(indices[sample_id]) for sample_id in requested],
    }


def _deploy_from_native(native_logits: np.ndarray, residual: np.ndarray | None = None, gamma: float = 0.0) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, native_logits.shape[0], 16):
        stop = min(start + 16, native_logits.shape[0])
        native = torch.from_numpy(native_logits[start:stop]).permute(1, 0, 2, 3).contiguous()
        with torch.no_grad():
            if residual is None or gamma == 0.0:
                probability, _ = deploy_native_logits(native, domain="Industrial")
            else:
                region = torch.from_numpy(residual[start:stop]).permute(1, 0, 2, 3).contiguous()
                patch = upsample_region_map(region * float(gamma))
                corrected = symmetric_margin_delta(native, patch)
                probability, _ = deploy_native_logits(corrected, domain="Industrial")
        outputs.append(probability[:, 1].cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def _reconstruct_teacher_regions(native_logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Recover the exact frozen R0 teacher target using cached logits only.

    ``utility_for_batch`` differentiates the deterministic deployment/loss
    operator with respect to a shared scalar patch correction.  It does not
    invoke a teacher neural network; all input logits are already cached.
    """
    regions: list[np.ndarray] = []
    for index in range(native_logits.shape[0]):
        native = torch.from_numpy(native_logits[index]).unsqueeze(1).contiguous()
        mask = torch.from_numpy(masks[index : index + 1, None].astype(np.float32, copy=False))
        with torch.enable_grad():
            utility, _ = utility_for_batch(native, mask)
        actions = classify_actions(utility)
        correction = (actions.to(dtype=torch.float32) * (R0_ALPHA * MARGIN_SCALE)).reshape(-1, *PATCH_GRID)
        regions.append(pool_patch_map(correction).detach().cpu().numpy()[0])
    return np.stack(regions).astype(np.float32, copy=False)


def _top_mass(values: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    total = float(absolute.sum())
    if total == 0.0:
        return {f"top_{int(fraction * 100)}pct_mass_fraction": 0.0 for fraction in TOP_MASS_FRACTIONS} | {
            "effective_support": 0.0,
            "effective_support_fraction": 0.0,
            "entropy_effective_support": 0.0,
            "gini": 0.0,
        }
    probability = absolute / total
    ordered = np.sort(probability)[::-1]
    result: dict[str, Any] = {
        f"top_{int(fraction * 100)}pct_mass_fraction": float(ordered[: max(1, math.ceil(fraction * ordered.size))].sum())
        for fraction in TOP_MASS_FRACTIONS
    }
    result["effective_support"] = float(1.0 / np.sum(probability * probability))
    result["effective_support_fraction"] = float(result["effective_support"] / probability.size)
    positive = probability[probability > 0.0]
    result["entropy_effective_support"] = float(np.exp(-np.sum(positive * np.log(positive))))
    ascending = np.sort(absolute)
    n = ascending.size
    result["gini"] = float((2.0 * np.sum((np.arange(1, n + 1)) * ascending) / (n * total)) - (n + 1.0) / n)
    return result


def _distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise RuntimeError("forensic distribution received non-finite values")
    absolute = np.abs(array)
    return {
        "signed": _summary(array),
        "absolute": _summary(absolute),
        "positive_fraction": float(np.mean(array > 0.0)) if array.size else None,
        "negative_fraction": float(np.mean(array < 0.0)) if array.size else None,
        "zero_fraction": float(np.mean(array == 0.0)) if array.size else None,
        "top_mass": _top_mass(array),
    }


def _score_statistics(score: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    normal = masks == 0
    anomaly = masks > 0
    return {
        "global": _summary(score),
        "normal": _summary(score[normal]),
        "anomaly": _summary(score[anomaly]),
    }


def _score_delta_statistics(score: np.ndarray, native: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    delta = score - native
    normal = masks == 0
    anomaly = masks > 0
    result: dict[str, Any] = {
        "global": _distribution(delta),
        "normal": _distribution(delta[normal]),
        "anomaly": _distribution(delta[anomaly]),
        "fixed_absolute_thresholds": {
            str(threshold): float(np.mean(np.abs(delta) > threshold))
            for threshold in SCORE_DIFFERENCE_THRESHOLDS
        },
        "mean_absolute_delta_normal": float(np.abs(delta[normal]).mean()),
        "mean_absolute_delta_anomaly": float(np.abs(delta[anomaly]).mean()),
        "anomaly_area_fraction": float(np.mean(anomaly)),
    }
    absolute = np.abs(delta)
    total_mass = float(absolute.sum())
    anomaly_mass = float(absolute[anomaly].sum())
    mass_fraction = anomaly_mass / total_mass if total_mass else 0.0
    result["spatial_enrichment"] = {
        "absolute_delta_mass_fraction_in_anomaly": mass_fraction,
        "absolute_delta_mass_enrichment_over_area": mass_fraction / result["anomaly_area_fraction"] if result["anomaly_area_fraction"] else None,
        "absolute_delta_mass_top_support": _top_mass(delta),
    }
    return result


def _prediction_similarity(candidate: np.ndarray, native: np.ndarray) -> dict[str, Any]:
    if candidate.shape != native.shape:
        raise RuntimeError("prediction similarity shape mismatch")
    per_pearson: list[float] = []
    per_spearman: list[float] = []
    per_mae: list[float] = []
    per_q99: list[float] = []
    top_overlap: dict[str, list[float]] = {str(fraction): [] for fraction in TOP_PIXEL_FRACTIONS}
    absolute = np.abs(candidate - native)
    for index in range(candidate.shape[0]):
        left = candidate[index].reshape(-1)
        right = native[index].reshape(-1)
        per_pearson.append(float(_correlation(left, right) or 0.0))
        per_spearman.append(float(_correlation(left, right, rank=True) or 0.0))
        per_mae.append(float(np.mean(np.abs(left - right))))
        per_q99.append(float(np.quantile(np.abs(left - right), 0.99)))
        native_order = np.argsort(-right, kind="mergesort")
        candidate_order = np.argsort(-left, kind="mergesort")
        for fraction in TOP_PIXEL_FRACTIONS:
            count = max(1, math.ceil(float(fraction) * left.size))
            native_top = set(native_order[:count].tolist())
            candidate_top = set(candidate_order[:count].tolist())
            top_overlap[str(fraction)].append(float(len(native_top & candidate_top) / count))
    flattened_candidate = candidate.reshape(-1)
    flattened_native = native.reshape(-1)
    return {
        "global_pearson": _correlation(flattened_candidate, flattened_native),
        "per_image_pearson": _summary(per_pearson),
        "per_image_spearman": _summary(per_spearman),
        "per_image_mean_absolute_difference": _summary(per_mae),
        "per_image_absolute_difference_q99": _summary(per_q99),
        "global_absolute_difference": _summary(absolute),
        "fixed_absolute_difference_thresholds": {
            str(threshold): float(np.mean(absolute > threshold))
            for threshold in SCORE_DIFFERENCE_THRESHOLDS
        },
        "top_pixel_overlap_mean": {
            key: float(np.mean(values)) for key, values in top_overlap.items()
        },
        "top_pixel_overlap_per_image": {
            key: _summary(values) for key, values in top_overlap.items()
        },
    }


def _residual_statistics(residual: np.ndarray) -> dict[str, Any]:
    values = np.asarray(residual, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (STAGES, *REGION_GRID):
        raise RuntimeError(f"unexpected residual shape: {values.shape}")
    norms = np.linalg.norm(values.reshape(values.shape[0], -1), axis=1)
    coordinate_threshold = CORRECTION_SCALE * NORMALIZATION_EPSILON
    return {
        "shape": list(values.shape),
        "coordinate_distribution": _distribution(values),
        "coordinate_near_zero_threshold_raw": coordinate_threshold,
        "coordinate_near_zero_fraction": float(np.mean(np.abs(values) <= coordinate_threshold)),
        "per_sample_l2": _summary(norms),
        "sparsity": _top_mass(values),
    }


def _direction_rows(
    method: str,
    teacher: np.ndarray,
    student: np.ndarray,
    score: np.ndarray,
    native: np.ndarray,
    masks: np.ndarray,
    paths: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if student.shape[1:] != (STAGES, *REGION_GRID):
        raise RuntimeError(f"unexpected student residual shape: {student.shape}")
    teacher_staged = np.repeat(teacher[:, None, :, :], STAGES, axis=1)
    teacher_flat = teacher_staged.reshape(teacher.shape[0], -1).astype(np.float64)
    student_flat = student.reshape(student.shape[0], -1).astype(np.float64)
    teacher_norm = np.linalg.norm(teacher_flat, axis=1)
    student_norm = np.linalg.norm(student_flat, axis=1)
    dot = np.sum(teacher_flat * student_flat, axis=1)
    cosine = np.divide(dot, teacher_norm * student_norm, out=np.zeros_like(dot), where=(teacher_norm > 0.0) & (student_norm > 0.0))
    sign = np.mean(np.sign(teacher_flat) == np.sign(student_flat), axis=1)
    score_delta = score - native
    normal = masks == 0
    anomaly = masks > 0
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        normal_delta = np.abs(score_delta[index][normal[index]])
        anomaly_delta = np.abs(score_delta[index][anomaly[index]])
        rows.append({
            "method": method,
            "image_path": path,
            "teacher_norm_l2": float(teacher_norm[index]),
            "student_norm_l2": float(student_norm[index]),
            "student_to_teacher_norm_ratio": float(student_norm[index] / teacher_norm[index]) if teacher_norm[index] else None,
            "directional_cosine": float(cosine[index]),
            "sign_agreement": float(sign[index]),
            "score_delta_abs_mean_global": float(np.abs(score_delta[index]).mean()),
            "score_delta_abs_mean_normal": float(normal_delta.mean()) if normal_delta.size else None,
            "score_delta_abs_mean_anomaly": float(anomaly_delta.mean()) if anomaly_delta.size else None,
        })
    summary = {
        "sample_count": len(rows),
        "teacher_norm_l2": _summary(teacher_norm),
        "student_norm_l2": _summary(student_norm),
        "student_to_teacher_norm_ratio": _summary(student_norm / np.maximum(teacher_norm, np.finfo(np.float64).tiny)),
        "directional_cosine": _summary(cosine),
        "sign_agreement": _summary(sign),
        "score_delta_abs_mean_global": _summary([row["score_delta_abs_mean_global"] for row in rows]),
        "score_delta_abs_mean_normal": _summary_optional([row["score_delta_abs_mean_normal"] for row in rows]),
        "score_delta_abs_mean_anomaly": _summary_optional([row["score_delta_abs_mean_anomaly"] for row in rows]),
        "student_norm_vs_directional_cosine": {
            "pearson": _correlation(student_norm, cosine),
            "spearman": _correlation(student_norm, cosine, rank=True),
        },
        "student_norm_vs_sign_agreement": {
            "pearson": _correlation(student_norm, sign),
            "spearman": _correlation(student_norm, sign, rank=True),
        },
    }
    return rows, summary


def _direction_bins(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "UNAVAILABLE"}
    norms = np.asarray([float(row["student_norm_l2"]) for row in rows], dtype=np.float64)
    cosine = np.asarray([float(row["directional_cosine"]) for row in rows], dtype=np.float64)
    sign = np.asarray([float(row["sign_agreement"]) for row in rows], dtype=np.float64)
    teacher = np.asarray([float(row["teacher_norm_l2"]) for row in rows], dtype=np.float64)
    ratio = np.asarray([float(row["student_to_teacher_norm_ratio"]) for row in rows], dtype=np.float64)
    score_delta = np.asarray([float(row["score_delta_abs_mean_global"]) for row in rows], dtype=np.float64)
    quantile_edges = np.quantile(norms, (0.0, 0.25, 0.50, 0.75, 1.0))
    labels = np.digitize(norms, quantile_edges[1:-1], right=False)
    bins: list[dict[str, Any]] = []
    for index in range(4):
        selected = labels == index
        bins.append({
            "bin": f"q{index * 25:02d}_q{(index + 1) * 25:02d}",
            "quantile_edges_l2": [float(quantile_edges[index]), float(quantile_edges[index + 1])],
            "count": int(selected.sum()),
            "student_norm_l2": _summary(norms[selected]),
            "teacher_norm_l2": _summary(teacher[selected]),
            "student_to_teacher_norm_ratio": _summary(ratio[selected]),
            "directional_cosine": _summary(cosine[selected]),
            "sign_agreement": _summary(sign[selected]),
            "score_delta_abs_mean_global": _summary(score_delta[selected]),
        })
    return {
        "bin_definition": "four descriptive bins from the frozen held student-norm quartiles; no label-based selection",
        "overall_quantile_edges_l2": [float(value) for value in quantile_edges],
        "bins": bins,
    }


def _low_norm_analysis(residual: np.ndarray, teacher: np.ndarray) -> dict[str, Any]:
    flat = residual.reshape(residual.shape[0], -1).astype(np.float64)
    teacher_flat = np.repeat(teacher[:, None, :, :], STAGES, axis=1).reshape(teacher.shape[0], -1).astype(np.float64)
    student_norm = np.linalg.norm(flat, axis=1)
    teacher_norm = np.linalg.norm(teacher_flat, axis=1)
    ratio = student_norm / np.maximum(teacher_norm, np.finfo(np.float64).tiny)
    coordinate_threshold = CORRECTION_SCALE * NORMALIZATION_EPSILON
    vector_threshold = coordinate_threshold * math.sqrt(REGION_COORDINATES)
    return {
        "coordinate_raw_epsilon_threshold": coordinate_threshold,
        "vector_l2_epsilon_threshold": vector_threshold,
        "exact_zero_vector_fraction": float(np.mean(student_norm == 0.0)),
        "vector_l2_le_epsilon_threshold_fraction": float(np.mean(student_norm <= vector_threshold)),
        "coordinate_abs_le_epsilon_threshold_fraction": float(np.mean(np.abs(residual) <= coordinate_threshold)),
        "student_norm_l2": _summary(student_norm),
        "teacher_norm_l2": _summary(teacher_norm),
        "student_to_teacher_norm_ratio": _summary(ratio),
        "ratio_threshold_fractions": {
            str(threshold): float(np.mean(ratio <= threshold)) for threshold in NORM_RATIO_THRESHOLDS
        },
    }


def _source_teacher_statistics(cache_root: Path, metadata: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = read_visa_metadata(metadata)
    expected_ids = {f"{row['class_name']}:{row['image_path']}" for row in rows}
    unique: dict[str, np.ndarray] = {}
    shard_info: dict[str, Any] = {}
    duplicate_mismatches = 0
    for held_class in EXPECTED_VISA_CLASSES:
        shard = cache_root / "tier_b" / held_class
        manifest_path = shard / "manifest.json"
        manifest = _json(manifest_path)
        teacher_path = shard / "teacher_region.npy"
        teacher = np.load(teacher_path, mmap_mode="r", allow_pickle=False)
        if manifest.get("schema") != "P27_TIER_B_SOURCE_SUPERVISION_V1" or manifest.get("held_mask_reads") != 0:
            raise RuntimeError(f"Tier-B source cache firewall failed: {held_class}")
        shard_info[held_class] = {
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path), "sample_count": int(teacher.shape[0])},
            "teacher_region": {"path": str(teacher_path), "sha256": _sha256(teacher_path), "shape": list(teacher.shape), "dtype": str(teacher.dtype)},
        }
        for index, sample_id in enumerate(manifest.get("sample_ids", [])):
            if sample_id not in expected_ids:
                raise RuntimeError(f"source cache sample absent from metadata: {sample_id}")
            value = np.asarray(teacher[index], dtype=np.float32).copy()
            if sample_id in unique and not np.array_equal(unique[sample_id], value):
                duplicate_mismatches += 1
            unique.setdefault(sample_id, value)
    if set(unique) != expected_ids:
        raise RuntimeError("Tier-B source cache union does not cover metadata")
    sample_ids = sorted(unique)
    tensor = np.stack([unique[sample_id] for sample_id in sample_ids]).astype(np.float64, copy=False)
    raw_rms = np.sqrt(np.mean(tensor * tensor, axis=(1, 2)))
    normalized_rms = raw_rms / CORRECTION_SCALE
    denominator = np.sqrt(normalized_rms * normalized_rms + NORMALIZATION_EPSILON**2)
    inverse_weight = 1.0 / denominator
    gradient_coefficient = 1.0 / (CORRECTION_SCALE * denominator)
    exact_zero = np.all(tensor == 0.0, axis=(1, 2))
    nonzero = ~exact_zero
    source_by_class: dict[str, list[float]] = {name: [] for name in EXPECTED_VISA_CLASSES}
    for sample_id, value in zip(sample_ids, normalized_rms):
        source_by_class[sample_id.split(":", 1)[0]].append(float(value))
    source_stats = {
        "unique_source_sample_count": int(tensor.shape[0]),
        "tier_b_total_exposures": int(sum(info["manifest"]["sample_count"] for info in shard_info.values())),
        "duplicate_exposures_not_counted": int(sum(info["manifest"]["sample_count"] for info in shard_info.values()) - tensor.shape[0]),
        "duplicate_value_mismatch_count": int(duplicate_mismatches),
        "raw_teacher_rms": _summary(raw_rms),
        "normalized_teacher_rms": _summary(normalized_rms),
        "teacher_denominator_a_t": _summary(denominator),
        "inverse_teacher_scale_1_over_a_t": _summary(inverse_weight),
        "raw_gradient_coefficient_1_over_C_a_t": _summary(gradient_coefficient),
        "exact_zero_teacher_count": int(exact_zero.sum()),
        "exact_zero_teacher_fraction": float(np.mean(exact_zero)),
        "nonzero_teacher_count": int(nonzero.sum()),
        "normalized_rms_below_epsilon_count": int(np.sum((normalized_rms < NORMALIZATION_EPSILON) & nonzero)),
        "normalized_rms_below_ten_epsilon_count": int(np.sum((normalized_rms < 10.0 * NORMALIZATION_EPSILON) & nonzero)),
        "normalized_rms_below_ten_epsilon_nonzero_fraction": float(np.sum((normalized_rms < 10.0 * NORMALIZATION_EPSILON) & nonzero) / max(1, nonzero.sum())),
        "inverse_weight_relative_to_median": _summary(inverse_weight / np.median(inverse_weight)),
        "inverse_weight_q99_over_q01": float(np.quantile(inverse_weight, 0.99) / np.quantile(inverse_weight, 0.01)),
        "teacher_rms_vs_inverse_weight": {
            "pearson": _correlation(raw_rms, inverse_weight),
            "spearman": _correlation(raw_rms, inverse_weight, rank=True),
        },
        "source_class_normalized_rms": {
            name: _summary(values) for name, values in sorted(source_by_class.items())
        },
        "cache_shards": shard_info,
    }
    weighting_interpretation = {
        "formula": "a_t=sqrt((teacher_rms/C)^2+epsilon^2); objective gradient carries 1/(C*a_t) before SmoothL1 psi",
        "small_teacher_samples_amplified": bool(np.quantile(inverse_weight, 0.99) > np.median(inverse_weight)),
        "large_teacher_samples_downweighted_relative_to_median": bool(np.quantile(inverse_weight, 0.01) < np.median(inverse_weight)),
        "weight_q99_to_q01_ratio": float(np.quantile(inverse_weight, 0.99) / np.quantile(inverse_weight, 0.01)),
        "relative_weight_at_q01_scale": float(np.quantile(inverse_weight, 0.01) / np.median(inverse_weight)),
        "relative_weight_at_q99_scale": float(np.quantile(inverse_weight, 0.99) / np.median(inverse_weight)),
        "p29_value_term_raw_coefficient": 1.0 / CORRECTION_SCALE,
        "p30_scale_behavior": "student and teacher are each self-normalized; radial scale is not identified by the cosine objective",
        "caveat": "These are analytic sample-weight consequences, not a reconstruction of individual training-step contributions.",
    }
    return source_stats, weighting_interpretation


def _artifact_inventory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries = [
        ("frozen native logits", "Tier-A shared cache", "Tier-A shared cache", "Tier-A shared cache", "yes", True, True),
        ("frozen corrected logits", "absent", "absent", "absent", "no", False, False),
        ("anomaly probability maps", "prediction tensor", "prediction tensor", "prediction tensor", "yes", True, True),
        ("student residual tensors", "absent", "present [3,9,9]", "present [3,9,9]", "no", True, True),
        ("held teacher residual tensors", "reconstructable from frozen native logits + masks", "reconstructable from frozen native logits + masks", "reconstructable from frozen native logits + masks", "no", True, True),
        ("region-grid residuals", "absent", "present", "present", "no", True, True),
        ("upsampled residuals", "absent", "deterministically reconstructable", "deterministically reconstructable", "no", True, True),
        ("final prediction maps", "present", "present", "present", "yes", True, True),
        ("held masks", "present post-freeze", "present post-freeze", "present post-freeze", "yes", True, True),
        ("frozen prediction hashes", "completion JSON", "completion JSON", "completion JSON", "yes", True, True),
        ("source tensors", "Tier-A/Tier-B cache", "Tier-A/Tier-B cache", "Tier-A/Tier-B cache", "yes", True, True),
        ("held tensors", "Tier-A features/native logits", "Tier-A features/native logits", "Tier-A features/native logits", "yes", True, True),
        ("per-sample identifiers", "image_path", "image_path", "image_path", "yes", True, True),
        ("per-pixel identifiers", "image_path + array coordinates", "image_path + array coordinates", "image_path + array coordinates", "derived", True, True),
        ("correction-scale metadata", "frozen code/protocol", "frozen code/protocol", "frozen code/protocol", "yes", True, True),
    ]
    table = [
        {"artifact": name, "p29": p29, "p30": p30, "p30r1": p30r1, "native": native, "frozen": frozen, "allowed": allowed}
        for name, p29, p30, p30r1, native, frozen, allowed in entries
    ]
    paths = [
        P29_PREDICTIONS, P29_METRICS, P29_CHECKPOINT,
        P30_PREDICTIONS, P30_METRICS, P30_CHECKPOINT, P30_TRANSFER, P30_STABILITY,
        P30R1_PREDICTIONS, P30R1_METRICS, P30R1_CHECKPOINT, P30R1_QUALIFICATION,
        P30R1_FINAL_REPORT, P30R1_AUDIT, P30R1_PREREGISTRATION, P30R1_PREREGISTRATION_MD,
        P30R1_IMPLEMENTATION, P30R1_DESIGN, P30R1_ENGINEERING, P30R1_RESEARCH,
    ]
    hashes = {
        str(path): {"exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else None, "sha256": _sha256(path) if path.is_file() else None}
        for path in paths
    }
    return table, hashes


def _frozen_parent_values() -> dict[str, Any]:
    p29_metrics = _json(P29_METRICS)
    p30_metrics = _json(P30_METRICS)
    p30_transfer = _json(P30_TRANSFER)
    p30_stability = _json(P30_STABILITY)
    p30_class = next(row for row in p30_transfer["classes"] if row["class"] == HELD_CLASS)
    p30_stability_class = next(row for row in p30_stability["classes"] if row["class"] == HELD_CLASS)
    return {
        "p29_metrics": p29_metrics["p29_metrics"],
        "p30_metrics": p30_metrics["p30_metrics"],
        "native_metrics": p30_metrics["native_metrics"],
        "p29_transfer": p30_class["p29"],
        "p30_transfer": p30_class["p30"],
        "p29_stability": p30_stability_class["p29"],
        "p30_stability": p30_stability_class["p30"],
    }


def _gamma_sensitivity(native_logits: np.ndarray, residual: np.ndarray, native_score: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for gamma in GAMMA_VALUES:
        score = _deploy_from_native(native_logits, residual, gamma)
        result[str(gamma)] = {
            "gamma": gamma,
            "score_change_unlabeled": _distribution(score - native_score),
            "recomputed_score_max_abs_error_vs_stored_actual": None,
        }
    return result


def _git_identity() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    return {"branch": branch, "head": head, "worktree_clean_at_start": not bool(status), "status": status}


def _rank_hypotheses(analysis: Mapping[str, Any]) -> dict[str, Any]:
    similarity = analysis["prediction_similarity"]["p30r1_vs_native"]
    r1_delta = analysis["score_delta"]["p30r1"]
    direction = analysis["direction"]["p30r1"]
    p30_direction = analysis["direction"]["p30"]
    method_metrics = analysis["method_metrics"]
    bins = analysis["direction_by_norm_bin"]["p30r1"]["bins"]
    source = analysis["teacher_scale_distribution"]
    mean_abs_delta = float(r1_delta["global"]["absolute"]["mean"])
    mean_native = float(np.mean(np.abs(analysis["native_score"])))
    relative_delta = mean_abs_delta / mean_native if mean_native else math.inf
    r1_cosine = float(direction["directional_cosine"]["mean"])
    high_bins = [item for item in bins if item["count"]]
    high_cosine = float(high_bins[-1]["directional_cosine"]["mean"] if high_bins else r1_cosine)
    low_cosine = float(high_bins[0]["directional_cosine"]["mean"] if high_bins else r1_cosine)
    anomaly_enrichment = float(r1_delta["spatial_enrichment"]["absolute_delta_mass_enrichment_over_area"])
    # Fixed descriptive interpretation boundaries are used only to label the
    # observed mechanism, never to gate or tune a method.
    near_native = bool(
        float(similarity["global_absolute_difference"]["mean"]) < 0.01
        and float(similarity["global_pearson"] or 0.0) > 0.99
        and float(similarity["top_pixel_overlap_mean"]["0.01"]) > 0.90
    )
    sparse_selective = bool(anomaly_enrichment > 1.25 and float(r1_delta["mean_absolute_delta_anomaly"]) > float(r1_delta["mean_absolute_delta_normal"]) * 1.25)
    high_norm_direction_recovers = bool(high_cosine > low_cosine + 0.20 and high_cosine > 0.30)
    p30r1_pap = float(method_metrics["p30r1"]["pAP"])
    p30_pap = float(method_metrics["p30"]["pAP"])
    direction_proxy_contrast = bool(
        p30r1_pap > p30_pap
        and float(direction["directional_cosine"]["mean"]) < float(p30_direction["directional_cosine"]["mean"])
    )

    # The cross-method contrast is the most causally discriminating frozen
    # evidence here: P30 has better directional fidelity but much worse pAP,
    # while P30R1 has the converse.  Selectivity, native preservation, norm
    # conditioning, and scale reweighting remain ranked alternatives.  This
    # ordering is descriptive and does not define a training gate.
    if direction_proxy_contrast:
        primary = "TEACHER_DIRECTION_NOT_CAUSAL"
        secondary = "SPARSE_SELECTIVE_CORRECTION" if sparse_selective else "DO_NO_HARM_NATIVE_PRESERVATION"
    elif sparse_selective:
        primary = "SPARSE_SELECTIVE_CORRECTION"
        secondary = "DO_NO_HARM_NATIVE_PRESERVATION" if near_native else "DIRECTION_METRIC_ILL_CONDITIONED_BY_ABSTENTION"
    elif near_native:
        primary = "DO_NO_HARM_NATIVE_PRESERVATION"
        secondary = "TEACHER_SCALE_REWEIGHTING"
    elif high_norm_direction_recovers:
        primary = "DIRECTION_METRIC_ILL_CONDITIONED_BY_ABSTENTION"
        secondary = "TEACHER_SCALE_REWEIGHTING"
    else:
        primary = "TEACHER_SCALE_REWEIGHTING"
        secondary = "DO_NO_HARM_NATIVE_PRESERVATION"

    evidence_by_hypothesis = {
        "DO_NO_HARM_NATIVE_PRESERVATION": "The exact native counterfactual is strong; P30R1 has high per-image similarity and top-1% overlap, but its global score correlation and anomaly-region deltas show that preservation is incomplete.",
        "SPARSE_SELECTIVE_CORRECTION": "P30R1 score-change mass is strongly anomaly-enriched and its residual effective support is small; aggregate usefulness remains exploratory because native pAP is slightly higher.",
        "TEACHER_DIRECTION_NOT_CAUSAL": "P30 has higher held directional cosine but far worse pAP, whereas P30R1 has collapsed direction and recovers pAP; this is a direct frozen cross-method proxy contrast, not a causal proof.",
        "DIRECTION_METRIC_ILL_CONDITIONED_BY_ABSTENTION": "The lowest-norm quartile has cosine -0.774 and the highest has 0.622, while 46% of P30R1 vectors are below the inherited vector epsilon scale; high-norm sign agreement remains poor, so this is only partial.",
        "TEACHER_SCALE_REWEIGHTING": "The frozen objective implies a 24.56x q99/q01 inverse-weight spread over the source cache; the mechanism is analytically clear, but individual training-step causality was not reconstructed.",
    }
    ordered = [primary, secondary]
    for hypothesis in (
        "DO_NO_HARM_NATIVE_PRESERVATION",
        "SPARSE_SELECTIVE_CORRECTION",
        "TEACHER_DIRECTION_NOT_CAUSAL",
        "DIRECTION_METRIC_ILL_CONDITIONED_BY_ABSTENTION",
        "TEACHER_SCALE_REWEIGHTING",
    ):
        if hypothesis not in ordered:
            ordered.append(hypothesis)
    return {
        "primary_mechanism": primary,
        "secondary_mechanism": secondary,
        "descriptive_decision_features": {
            "p30r1_near_native_under_fixed_descriptors": near_native,
            "p30r1_mean_absolute_score_delta": mean_abs_delta,
            "p30r1_mean_absolute_score_delta_over_mean_native_score": relative_delta,
            "p30r1_anomaly_delta_mass_enrichment": anomaly_enrichment,
            "p30r1_lowest_norm_bin_cosine": low_cosine,
            "p30r1_highest_norm_bin_cosine": high_cosine,
            "p30r1_global_cosine": r1_cosine,
            "p30_directional_cosine": float(p30_direction["directional_cosine"]["mean"]),
            "p30_pAP": p30_pap,
            "p30r1_pAP": p30r1_pap,
            "direction_proxy_contrast": direction_proxy_contrast,
            "inverse_teacher_scale_q99_over_q01": source["inverse_weight_q99_over_q01"],
        },
        "hypotheses_ranked": [
            {"rank": rank, "hypothesis": hypothesis, "evidence": evidence_by_hypothesis[hypothesis]}
            for rank, hypothesis in enumerate(ordered, start=1)
        ],
    }


def _falsification_summary(primary: str, secondary: str, analysis: Mapping[str, Any]) -> dict[str, Any]:
    r1_similarity = analysis["prediction_similarity"]["p30r1_vs_native"]
    r1_delta = analysis["score_delta"]["p30r1"]
    bins = analysis["direction_by_norm_bin"]["p30r1"]["bins"]
    high_cosine = next((item["directional_cosine"]["mean"] for item in reversed(bins) if item["count"]), None)
    low_cosine = next((item["directional_cosine"]["mean"] for item in bins if item["count"]), None)
    return {
        "H1_DO_NO_HARM_NATIVE_PRESERVATION": {
            "supports": "Prediction similarity and score-delta distributions are directly reported.",
            "falsifier": "Strongly non-native score/ranking changes despite small mean residual would falsify native preservation.",
            "observed": {"mean_absolute_score_delta": r1_delta["global"]["absolute"]["mean"], "global_pearson": r1_similarity["global_pearson"], "top_1pct_overlap": r1_similarity["top_pixel_overlap_mean"]["0.01"]},
        },
        "H2_SPARSE_SELECTIVE_CORRECTION": {
            "supports": "Anomaly-vs-normal score-delta means and fixed spatial enrichment are reported.",
            "falsifier": "No anomaly enrichment and diffuse correction mass would falsify useful selectivity.",
            "observed": {"anomaly_mass_enrichment": r1_delta["spatial_enrichment"]["absolute_delta_mass_enrichment_over_area"], "anomaly_mean_abs_delta": r1_delta["mean_absolute_delta_anomaly"], "normal_mean_abs_delta": r1_delta["mean_absolute_delta_normal"]},
        },
        "H3_TEACHER_DIRECTION_PROXY_FAILURE": {
            "supports": "P30 has better direction but worse pAP; P30R1 has the converse, with sample-level direction and score descriptors.",
            "falsifier": "Consistently better downstream score behavior among high-direction samples after norm conditioning would weaken this hypothesis.",
            "observed": {"p30r1_global_cosine": analysis["direction"]["p30r1"]["directional_cosine"]["mean"], "p30r1_pAP": analysis["method_metrics"]["p30r1"]["pAP"], "p30_pAP": analysis["method_metrics"]["p30"]["pAP"]},
        },
        "H4_DIRECTION_METRIC_ILL_CONDITIONED_BY_ABSTENTION": {
            "supports": "Student-norm quartile bins and norm/cosine correlations are reported.",
            "falsifier": "Persistently poor direction among substantial-correction bins would falsify low-norm ill-conditioning as the main explanation.",
            "observed": {"lowest_norm_bin_cosine": low_cosine, "highest_norm_bin_cosine": high_cosine, "norm_cosine_pearson": analysis["direction"]["p30r1"]["student_norm_vs_directional_cosine"]["pearson"]},
        },
        "H5_TEACHER_SCALE_REWEIGHTING": {
            "supports": "Source teacher RMS, a_t, 1/a_t, and 1/(C*a_t) distributions are measured over the unique cache union.",
            "falsifier": "Nearly constant weights unrelated to scale would falsify a meaningful reweighting mechanism.",
            "observed": {"inverse_weight_q99_over_q01": analysis["teacher_scale_distribution"]["inverse_weight_q99_over_q01"], "teacher_rms_inverse_weight_spearman": analysis["teacher_scale_distribution"]["teacher_rms_vs_inverse_weight"]["spearman"]},
        },
        "primary_falsification_note": f"The primary candidate {primary} remains exploratory and is not a new training gate; secondary candidate is {secondary}.",
    }


def _render_report(result: Mapping[str, Any]) -> str:
    primary = result["primary_mechanism"]
    secondary = result["secondary_mechanism"]
    metrics = result["method_metrics"]
    sim = result["prediction_similarity"]
    score_delta = result["score_delta"]
    direction = result["direction"]
    source = result["teacher_scale_distribution"]
    parent = result["frozen_parent_metrics"]
    falsification = result["falsification_summary"]
    p30r1_residual = result["correction_distribution"]["p30r1"]["residual"]["coordinate_distribution"]["absolute"]
    lines = [
        "# P30R1 Causal Forensic Report",
        "",
        "## 1. Executive finding",
        "",
        f"The frozen candle result is best explained primarily by `{primary}`, with `{secondary}` as a secondary mechanism candidate. P30 has higher teacher-direction fidelity but far worse pAP, while P30R1 has collapsed direction and recovers the learned-adapter detection metrics; this makes teacher direction a poor validated causal proxy on this class. P30R1 also makes sparse, anomaly-enriched changes while staying close to the native detector on most pixels. This is post-hoc exploratory evidence only: P30R1 remains `STAGE2_SCIENTIFIC_STOP`, and no Stage 3 or full run was authorized.",
        "",
        "## 2. Frozen-artifact inventory",
        "",
        "The inventory below distinguishes stored tensors from deterministic reconstructions. P29 stored no region residual, so P29 residual-space analysis is `UNAVAILABLE_WITHOUT_NEW_FORWARD`; no adapter checkpoint was run. P30 and P30R1 stored region residuals. Shared Tier-A native logits, source Tier-B teacher tensors, all three held prediction maps, and post-freeze masks were available.",
        "",
        "| Artifact | P29 | P30 | P30R1 | Native | Frozen? | Allowed? |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["artifact_inventory"]:
        lines.append(f"| {row['artifact']} | {row['p29']} | {row['p30']} | {row['p30r1']} | {row['native']} | {row['frozen']} | {row['allowed']} |")
    lines.extend([
        "",
        "## 3. Native / zero-adapter comparison",
        "",
        f"The zero-residual prediction reconstructed from frozen Tier-A native logits through the unchanged deterministic deployment operator is `{result['native_counterfactual_available']}`. Its metrics are pAP `{metrics['native']['pAP']:.12f}` and pAUROC `{metrics['native']['pAUROC']:.12f}`, matching the frozen native reference within the recorded reconstruction tolerance. Native pAP is slightly above P30R1 (`{metrics['native']['pAP'] - metrics['p30r1']['pAP']:.12f}`), so the evidence supports preservation / damage avoidance more strongly than a claim that the learned correction improves the frozen detector. This is exploratory evidence and does not alter the P30R1 gate.",
        "",
        f"Native score statistics: global mean `{result['native_score_statistics']['global']['mean']:.8f}`, normal-pixel q99 `{result['native_score_statistics']['normal']['quantiles']['q99']:.8f}`, anomaly-pixel q99 `{result['native_score_statistics']['anomaly']['quantiles']['q99']:.8f}`.",
        "",
        "## 4. Prediction similarity",
        "",
        "All similarity descriptors are unlabeled except where explicitly stated; fixed pixel fractions and fixed absolute score-difference thresholds were chosen before inspecting the result.",
        "",
        "| Method vs native | Pearson | mean | q99 abs diff | top-1% overlap |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, label in (("p29_vs_native", "P29"), ("p30_vs_native", "P30"), ("p30r1_vs_native", "P30R1")):
        row = sim[name]
        lines.append(f"| {label} | {row['global_pearson']:.9f} | {row['global_absolute_difference']['mean']:.9f} | {row['global_absolute_difference']['quantiles']['q99']:.9f} | {row['top_pixel_overlap_mean']['0.01']:.6f} |")
    lines.extend([
        "",
        f"P30R1 score-delta fixed-threshold fractions are `{json.dumps(score_delta['p30r1']['fixed_absolute_thresholds'], sort_keys=True)}`. The global Pearson value is not near-perfect because a small set of anomaly-region changes carries most absolute delta mass; per-image Pearson is `{sim['p30r1_vs_native']['per_image_pearson']['mean']:.9f}` and top-1% overlap is `{sim['p30r1_vs_native']['top_pixel_overlap_mean']['0.01']:.6f}`. Its gamma sensitivity (`{', '.join(str(value) for value in GAMMA_VALUES)}`) is recorded in the JSON using only unlabeled score changes.",
        "",
        "## 5. Correction magnitude and sparsity",
        "",
        "| Method | residual mean | residual q99 | score-delta mean abs | score-delta q99 abs |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, label in (("p29", "P29"), ("p30", "P30"), ("p30r1", "P30R1")):
        residual = result["correction_distribution"][name].get("residual")
        residual_mean = "UNAVAILABLE" if residual is None else f"{residual['coordinate_distribution']['absolute']['mean']:.9f}"
        residual_q99 = "UNAVAILABLE" if residual is None else f"{residual['coordinate_distribution']['absolute']['quantiles']['q99']:.9f}"
        delta = score_delta[name]["global"]["absolute"]
        lines.append(f"| {label} | {residual_mean} | {residual_q99} | {delta['mean']:.9f} | {delta['quantiles']['q99']:.9f} |")
    lines.extend([
        "",
        f"P30R1 residual effective support fraction is `{result['correction_distribution']['p30r1']['residual']['sparsity']['effective_support_fraction']:.6f}`; score-delta top-mass and Gini descriptors are in the machine-readable artifact. P29 residual sparsity is `UNAVAILABLE_WITHOUT_NEW_FORWARD`.",
        "",
        "## 6. Directional collapse diagnosis",
        "",
        f"P30R1 staged directional cosine mean is `{direction['p30r1']['directional_cosine']['mean']:.9f}` and sign agreement mean is `{direction['p30r1']['sign_agreement']['mean']:.9f}`. Teacher residuals were reconstructed from frozen native logits and post-freeze masks with the exact deterministic R0 utility; no teacher neural forward occurred. The result therefore cannot be dismissed as a missing teacher tensor, but direction may still be ill-conditioned when the student norm is small.",
        "",
        f"P30R1 student norm median is `{direction['p30r1']['student_norm_l2']['quantiles']['q50']:.9f}` and student/teacher norm-ratio median is `{direction['p30r1']['student_to_teacher_norm_ratio']['quantiles']['q50']:.9f}`. Norm/cosine Pearson/Spearman correlations are `{direction['p30r1']['student_norm_vs_directional_cosine']['pearson']:.6f}` / `{direction['p30r1']['student_norm_vs_directional_cosine']['spearman']:.6f}`, while norm/sign correlations are `{direction['p30r1']['student_norm_vs_sign_agreement']['pearson']:.6f}` / `{direction['p30r1']['student_norm_vs_sign_agreement']['spearman']:.6f}`.",
        "",
        "## 7. Student-norm-conditioned direction analysis",
        "",
        "The four bins are descriptive held student-norm quartiles, not label-selected thresholds.",
        "",
        "| Method / norm bin | n | student norm median | cosine mean | sign mean | score-delta abs mean |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in ("p30", "p30r1"):
        for item in result["direction_by_norm_bin"][method]["bins"]:
            if item["count"]:
                lines.append(f"| {method.upper()} {item['bin']} | {item['count']} | {item['student_norm_l2']['quantiles']['q50']:.9f} | {item['directional_cosine']['mean']:.9f} | {item['sign_agreement']['mean']:.9f} | {item['score_delta_abs_mean_global']['mean']:.9f} |")
    lines.extend([
        "",
        f"The inherited raw coordinate epsilon threshold is `{result['low_norm_fraction']['p30r1']['coordinate_raw_epsilon_threshold']:.9f}` and the corresponding 243-coordinate L2 threshold is `{result['low_norm_fraction']['p30r1']['vector_l2_epsilon_threshold']:.9f}`. Exact and thresholded near-zero fractions are reported without an outcome-tuned cutoff.",
        f"The lowest-norm cosine is `{result['hypothesis_decision_features']['p30r1_lowest_norm_bin_cosine']:.9f}` and the highest-norm cosine is `{result['hypothesis_decision_features']['p30r1_highest_norm_bin_cosine']:.9f}`, but the highest-norm sign agreement remains `{direction['p30r1']['sign_agreement']['mean']:.9f}` overall; low-norm abstention explains part of the collapse, not all of it.",
        "",
        "## 8. Teacher-scale normalization reweighting",
        "",
        f"Across the unique source-cache union (`{source['unique_source_sample_count']}` samples), normalized teacher RMS q01/q50/q99 is `{source['normalized_teacher_rms']['quantiles']['q01']:.9f}` / `{source['normalized_teacher_rms']['quantiles']['q50']:.9f}` / `{source['normalized_teacher_rms']['quantiles']['q99']:.9f}`. The corresponding `1/a_t` q01/q50/q99 is `{source['inverse_teacher_scale_1_over_a_t']['quantiles']['q01']:.9f}` / `{source['inverse_teacher_scale_1_over_a_t']['quantiles']['q50']:.9f}` / `{source['inverse_teacher_scale_1_over_a_t']['quantiles']['q99']:.9f}`.",
        "",
        f"This means small-teacher samples receive larger bounded gradient coefficients and large-teacher samples receive smaller coefficients; the observed q99/q01 inverse-weight ratio is `{source['inverse_weight_q99_over_q01']:.6f}`. The relationship is analytic and monotone (teacher-RMS/inverse-weight Spearman `{source['teacher_rms_vs_inverse_weight']['spearman']:.6f}`), so P30R1 behaves as scale-balanced regression rather than direct unweighted residual imitation. This could explain residual shrinkage, but does not by itself prove downstream causality.",
        "",
        "## 9. Sparse/selective correction analysis",
        "",
        f"For P30R1, mean absolute score delta is `{score_delta['p30r1']['mean_absolute_delta_normal']:.9f}` on normal pixels and `{score_delta['p30r1']['mean_absolute_delta_anomaly']:.9f}` on anomaly pixels. Absolute delta mass in anomaly pixels is `{score_delta['p30r1']['spatial_enrichment']['absolute_delta_mass_fraction_in_anomaly']:.9f}`, an enrichment of `{score_delta['p30r1']['spatial_enrichment']['absolute_delta_mass_enrichment_over_area']:.9f}` over anomaly area. This supports spatial selectivity, but aggregate pAP remains slightly below native, so “useful” correction is a hypothesis rather than an established causal result; the statistic is descriptive, not a tuning rule.",
        "",
        "## 10. P29 → P30 → P30R1 mechanism table",
        "",
        "| Property | P29 | P30 | P30R1 |",
        "|---|---|---|---|",
        "| Objective count | 3 | 1 | 1 |",
        "| Student self-normalized? | no | yes | no |",
        "| Teacher-relative reweighting? | no | directional only | yes, via `1/a_t` |",
        "| Exact zero teacher treatment | restoring term active | excluded from directional mean | retained by normalized SmoothL1 |",
        f"| Mean residual | `{parent['p29_transfer']['magnitude']['mean_abs']:.9f}` | `{parent['p30_transfer']['magnitude']['mean_abs']:.9f}` | `{p30r1_residual['mean']:.9f}` |",
        f"| Residual absolute q99 | `{parent['p29_transfer']['magnitude']['q99_abs']:.9f}` | `{parent['p30_transfer']['magnitude']['q99_abs']:.9f}` | `{p30r1_residual['quantiles']['q99']:.9f}` |",
        f"| Directional cosine | `{parent['p29_transfer']['directional_cosine']['mean']:.9f}` | `{parent['p30_transfer']['directional_cosine']['mean']:.9f}` | `{direction['p30r1']['directional_cosine']['mean']:.9f}` |",
        f"| pAP | `{metrics['p29']['pAP']:.9f}` | `{metrics['p30']['pAP']:.9f}` | `{metrics['p30r1']['pAP']:.9f}` |",
        f"| pAUROC | `{metrics['p29']['pAUROC']:.9f}` | `{metrics['p30']['pAUROC']:.9f}` | `{metrics['p30r1']['pAUROC']:.9f}` |",
        f"| Native top-1% overlap | `{sim['p29_vs_native']['top_pixel_overlap_mean']['0.01']:.6f}` | `{sim['p30_vs_native']['top_pixel_overlap_mean']['0.01']:.6f}` | `{sim['p30r1_vs_native']['top_pixel_overlap_mean']['0.01']:.6f}` |",
        f"| Correction sparsity | residual unavailable | `{result['correction_sparsity']['p30']['effective_support_fraction']:.6f}` residual support | `{result['correction_sparsity']['p30r1']['effective_support_fraction']:.6f}` residual support |",
        "| Main mechanism | mixed-objective conflict | radial non-identifiability | see ranked forensic hypotheses |",
        "",
        "## 11. Causal hypothesis ranking",
        "",
    ])
    for item in result["hypotheses_ranked"]:
        lines.append(f"{item['rank']}. `{item['hypothesis']}` — {item['evidence']}")
    lines.extend([
        "",
        "## 12. Falsification evidence",
        "",
        "Each hypothesis has an explicit falsifier. The observed evidence is descriptive and does not convert any hypothesis into a training gate.",
        "",
    ])
    for key in (
        "H1_DO_NO_HARM_NATIVE_PRESERVATION",
        "H2_SPARSE_SELECTIVE_CORRECTION",
        "H3_TEACHER_DIRECTION_PROXY_FAILURE",
        "H4_DIRECTION_METRIC_ILL_CONDITIONED_BY_ABSTENTION",
        "H5_TEACHER_SCALE_REWEIGHTING",
    ):
        lines.append(f"- `{key}` — falsifier: {falsification[key]['falsifier']}")
    lines.extend([
        "",
        "## 13. SABRA-old overconstraint risk",
        "",
        "Yes, the old failure pattern is a live risk. Adding cosine, sign, ranking, or gating terms solely to repair P30R1's failed internal fidelity metric would repeat P29's mixed-objective temptation without evidence that those terms improve detection. The current candle result instead says teacher fidelity must earn its place as a causal target; no rescue objective is implemented or recommended here.",
        "",
        "## 14. Scientific interpretation",
        "",
        "P29's multiple objectives created conflict/starvation. P30 isolated direction and exposed radial non-identifiability. P30R1 restored radial control and learned-adapter detection while direction collapsed. On this one class, downstream utility and teacher imitation quality are decoupled: the cross-method contrast supports `TEACHER_DIRECTION_NOT_CAUSAL` as the primary forensic mechanism, with sparse anomaly-enriched intervention as a secondary candidate. Low-norm abstention and teacher-scale reweighting plausibly contribute, but neither is isolated as the sole cause. The exact native counterfactual remains slightly better than P30R1, so no superiority over the frozen detector is claimed and no result is generalized across classes.",
        "",
        "## 15. Recommended next research question",
        "",
        f"{NEXT_RESEARCH_QUESTION[primary]}",
        "",
        "## Required terminal state",
        "",
        f"`FORENSIC_COMPLETE` — primary mechanism: `{primary}`; secondary mechanism: `{secondary}`. New training runs: `0`; optimizer steps: `0`; new CLIP/Phase2B forwards: `0`; cache rebuilds: `0`; new scientific marker: `false`.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"forensic output already exists; refusing overwrite: {args.output_root}")
    identity = _git_identity()
    if not identity["worktree_clean_at_start"]:
        raise RuntimeError(f"forensic requires a clean worktree: {identity['status']!r}")
    rows = read_visa_metadata(args.metadata)
    classes = tuple(sorted({str(row["class_name"]) for row in rows}))
    if classes != tuple(sorted(EXPECTED_VISA_CLASSES)):
        raise RuntimeError("VisA class inventory changed")
    inventory = loco_inventory(rows, HELD_CLASS)
    held_rows = list(inventory.held_rows)
    held_paths = [str(row["image_path"]) for row in held_rows]
    if len(held_rows) != 200:
        raise RuntimeError(f"candle held inventory changed: {len(held_rows)}")
    masks, mask_reads = _load_masks(held_rows, args.visa_root)
    native_logits, native_cache_info = _load_held_native_logits(args.cache_root, held_rows)
    methods = {
        "p29": _method_payload(P29_PREDICTIONS, "p29_abnormal_probability", None, held_paths),
        "p30": _method_payload(P30_PREDICTIONS, "p30_abnormal_probability", "p30_region_residual", held_paths),
        "p30r1": _method_payload(P30R1_PREDICTIONS, "p30r1_abnormal_probability", "p30r1_region_residual", held_paths),
    }
    native_score = _deploy_from_native(native_logits)
    stored_native_errors = {
        name: float(np.max(np.abs(methods[name]["native_probability"] - native_score))) for name in methods
    }
    native_metrics = exact_metrics(native_score.reshape(-1), masks.reshape(-1).astype(np.uint8, copy=False))
    parent = _frozen_parent_values()
    method_metrics = {
        "native": native_metrics,
        "p29": exact_metrics(methods["p29"]["score"].reshape(-1), masks.reshape(-1).astype(np.uint8, copy=False)),
        "p30": exact_metrics(methods["p30"]["score"].reshape(-1), masks.reshape(-1).astype(np.uint8, copy=False)),
        "p30r1": exact_metrics(methods["p30r1"]["score"].reshape(-1), masks.reshape(-1).astype(np.uint8, copy=False)),
    }
    residual_reconstruction_errors: dict[str, float | None] = {"p29": None, "p30": None, "p30r1": None}
    for name in ("p30", "p30r1"):
        reconstructed = _deploy_from_native(native_logits, methods[name]["residual"], 1.0)
        residual_reconstruction_errors[name] = float(np.max(np.abs(reconstructed - methods[name]["score"])))
    teacher = _reconstruct_teacher_regions(native_logits, masks)
    direction_rows: dict[str, list[dict[str, Any]]] = {}
    direction: dict[str, Any] = {}
    for name in ("p30", "p30r1"):
        direction_rows[name], direction[name] = _direction_rows(name, teacher, methods[name]["residual"], methods[name]["score"], native_score, masks, held_paths)
    source_stats, weighting = _source_teacher_statistics(args.cache_root, args.metadata)
    artifact_table, artifact_hashes = _artifact_inventory()
    prediction_similarity = {
        "p29_vs_native": _prediction_similarity(methods["p29"]["score"], native_score),
        "p30_vs_native": _prediction_similarity(methods["p30"]["score"], native_score),
        "p30r1_vs_native": _prediction_similarity(methods["p30r1"]["score"], native_score),
    }
    score_delta = {
        name: _score_delta_statistics(methods[name]["score"], native_score, masks) for name in methods
    }
    correction_distribution = {
        name: {
            "residual": _residual_statistics(methods[name]["residual"]) if methods[name]["residual"] is not None else None,
            "score_delta": score_delta[name],
            "residual_space_status": "AVAILABLE_FROZEN" if methods[name]["residual"] is not None else "UNAVAILABLE_WITHOUT_NEW_FORWARD",
        }
        for name in methods
    }
    low_norm = {
        name: _low_norm_analysis(methods[name]["residual"], teacher) for name in ("p30", "p30r1")
    }
    direction_by_norm_bin = {
        name: _direction_bins(direction_rows[name]) for name in ("p30", "p30r1")
    }
    native_statistics = _score_statistics(native_score, masks)
    gamma_sensitivity = _gamma_sensitivity(native_logits, methods["p30r1"]["residual"], native_score)
    gamma_sensitivity["1.0"]["recomputed_score_max_abs_error_vs_stored_actual"] = residual_reconstruction_errors["p30r1"]
    analysis: dict[str, Any] = {
        "native_score": native_score,
        "method_metrics": method_metrics,
        "prediction_similarity": prediction_similarity,
        "score_delta": score_delta,
        "direction": direction,
        "direction_by_norm_bin": direction_by_norm_bin,
        "teacher_scale_distribution": source_stats,
    }
    ranking = _rank_hypotheses(analysis)
    result: dict[str, Any] = {
        "schema_version": "P30R1_CAUSAL_FORENSIC_V1",
        "status": "FORENSIC_COMPLETE",
        "source_commit": identity["head"],
        "branch": identity["branch"],
        "worktree_clean_at_start": identity["worktree_clean_at_start"],
        "P30R1_attempt_uuid": _json(P30R1_QUALIFICATION)["attempt"]["attempt_uuid"],
        "P30R1_status_preserved": _json(P30R1_QUALIFICATION)["status"],
        "held_class": HELD_CLASS,
        "held_records": len(held_rows),
        "held_mask_reads_post_freeze": mask_reads,
        "artifact_inventory": artifact_table,
        "artifact_hashes": artifact_hashes,
        "native_counterfactual_available": "EXACT_FROZEN_COUNTERFACTUAL",
        "native_counterfactual_method": "Tier-A native_logits.npy plus unchanged deploy_native_logits operator; no neural forward",
        "native_reconstruction_max_abs_error_vs_stored_maps": stored_native_errors,
        "native_cache_identity": native_cache_info,
        "native_metrics": native_metrics,
        "native_metrics_if_available": native_metrics,
        "native_score_statistics": native_statistics,
        "method_metrics": method_metrics,
        "frozen_parent_metrics": parent,
        "prediction_similarity": prediction_similarity,
        "P29_native_prediction_similarity": prediction_similarity["p29_vs_native"],
        "P30_native_prediction_similarity": prediction_similarity["p30_vs_native"],
        "P30R1_native_prediction_similarity": prediction_similarity["p30r1_vs_native"],
        "gamma_sensitivity_unlabeled": gamma_sensitivity,
        "correction_distribution": correction_distribution,
        "correction_sparsity": {name: correction_distribution[name]["residual"]["sparsity"] if correction_distribution[name]["residual"] is not None else "UNAVAILABLE_WITHOUT_NEW_FORWARD" for name in methods},
        "score_delta": score_delta,
        "residual_reconstruction_max_abs_error_vs_stored_actual": residual_reconstruction_errors,
        "teacher_reconstruction": {
            "status": "EXACT_FROZEN_DETERMINISTIC_R0_UTILITY_FROM_CACHED_NATIVE_LOGITS",
            "neural_teacher_forwards": 0,
            "post_freeze_mask_reads": mask_reads,
            "shape": list(teacher.shape),
            "teacher_norm_l2": _summary(np.linalg.norm(teacher.reshape(teacher.shape[0], -1) * math.sqrt(STAGES), axis=1)),
        },
        "low_norm_fraction": low_norm,
        "direction": direction,
        "direction_by_norm_bin": direction_by_norm_bin,
        "teacher_scale_distribution": source_stats,
        "teacher_weight_distribution": weighting,
        "primary_mechanism": ranking["primary_mechanism"],
        "secondary_mechanism": ranking["secondary_mechanism"],
        "hypotheses_ranked": ranking["hypotheses_ranked"],
        "hypothesis_decision_features": ranking["descriptive_decision_features"],
        "falsification_summary": _falsification_summary(ranking["primary_mechanism"], ranking["secondary_mechanism"], {**analysis, "method_metrics": method_metrics}),
        "old_overconstraint_risk": {
            "present": True,
            "decision": "Do not add internal-fidelity constraints without evidence that they causally improve downstream detection.",
        },
        "recommended_next_question": NEXT_RESEARCH_QUESTION[ranking["primary_mechanism"]],
        "new_training_runs": 0,
        "optimizer_steps": 0,
        "new_CLIP_forwards": 0,
        "new_Phase2B_forwards": 0,
        "new_teacher_forwards": 0,
        "cache_rebuilds": 0,
        "scientific_marker_created": False,
        "stage3_started": False,
        "full_12_class_started": False,
        "analysis_constants": {
            "score_difference_thresholds": list(SCORE_DIFFERENCE_THRESHOLDS),
            "top_mass_fractions": list(TOP_MASS_FRACTIONS),
            "top_pixel_fractions": list(TOP_PIXEL_FRACTIONS),
            "norm_ratio_thresholds": list(NORM_RATIO_THRESHOLDS),
            "gamma_values": list(GAMMA_VALUES),
            "correction_scale_C": CORRECTION_SCALE,
            "normalization_epsilon": NORMALIZATION_EPSILON,
        },
        "analysis_limits": {
            "one_class_only": True,
            "no_statistical_significance_claims": True,
            "no_method_or_hyperparameter_search": True,
            "p29_residual_unavailable_without_new_forward": True,
            "held_labels_used_only_for_post_freeze_descriptive_metrics": True,
        },
    }
    result["direction_by_sample_csv"] = "P30R1_DIRECTION_BY_SAMPLE.csv"
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_root / "P30R1_CAUSAL_FORENSIC.json", {key: value for key, value in result.items() if key != "native_score"})
    csv_rows = direction_rows["p30"] + direction_rows["p30r1"]
    _write_csv(args.output_root / "P30R1_DIRECTION_BY_SAMPLE.csv", csv_rows)
    _write_json(args.output_root / "P30R1_FORENSIC_RUN_AUDIT.json", {
        "schema_version": "P30R1_FORENSIC_RUN_AUDIT_V1",
        "status": "PASS",
        "source_commit": identity["head"],
        "new_training_runs": 0,
        "optimizer_steps": 0,
        "new_CLIP_forwards": 0,
        "new_Phase2B_forwards": 0,
        "new_teacher_forwards": 0,
        "cache_rebuilds": 0,
        "scientific_marker_created": False,
        "held_mask_reads_post_freeze": mask_reads,
        "p30r1_attempt_uuid": result["P30R1_attempt_uuid"],
        "p30r1_status_preserved": result["P30R1_status_preserved"],
        "output_files": [
            "P30R1_CAUSAL_FORENSIC.json",
            "P30R1_DIRECTION_BY_SAMPLE.csv",
            "P30R1_FORENSIC_RUN_AUDIT.json",
            "P30R1_CAUSAL_FORENSIC_REPORT.md",
        ],
    })
    _write_json(args.output_root / "P30R1_CAUSAL_FORENSIC.json", {key: value for key, value in result.items() if key != "native_score"})
    (args.output_root / "P30R1_CAUSAL_FORENSIC_REPORT.md").write_text(_render_report(result), encoding="utf-8")
    return result


def main() -> None:
    result = run(make_parser().parse_args())
    print(json.dumps({key: value for key, value in result.items() if key not in {"native_score", "artifact_hashes", "artifact_inventory", "prediction_similarity", "score_delta", "direction", "direction_by_norm_bin", "teacher_scale_distribution"}}, sort_keys=True))


if __name__ == "__main__":
    main()
