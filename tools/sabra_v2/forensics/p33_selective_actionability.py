"""Offline P32 selective-actionability forensic.

This reads only frozen P32/P30R1 prediction tensors, the frozen candle native
maps, post-freeze descriptive masks, and source Tier-B teacher tensors.  It
does not load a checkpoint, invoke a neural model, rebuild a cache, or score a
new prediction.  The deployment transform used for source descriptors is a
fixed tensor operator over already-cached teacher regions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.sabra_v2.p32_objective import deployed_margin_effect


ROOT = Path("/workspace/ACD-CLIP-sabra")
CACHE = Path("/workspace/p27r1_cache_v1")
VISA = Path("/workspace/data/source/visa_unpack")
META = ROOT / "dataset/hub/VisA.jsonl"
P32 = ROOT / "research/sabra_v2/region_distill/P32/candle/predictions/p32_held_predictions.pt"
P30R1 = ROOT / "research/sabra_v2/region_distill/P30R1/candle/predictions/p30r1_held_predictions.pt"
RAW_EPSILON = 0.0496010971069336  # inherited P30R1 forensic coordinate epsilon
SUMMARY_Q = (0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
TOP_FRACTIONS = (0.01, 0.05, 0.10)
SCORE_THRESHOLDS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2)


def summary(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    absolute = np.abs(values)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "mean_abs": float(absolute.mean()),
        "median_abs": float(np.quantile(absolute, 0.5)),
        "q_abs": {str(q): float(np.quantile(absolute, q)) for q in SUMMARY_Q},
        "min": float(values.min()),
        "max": float(values.max()),
        "finite": bool(np.isfinite(values).all()),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
        "zero_fraction": float(np.mean(values == 0)),
    }


def concentration(values: np.ndarray, stride: int | None = None) -> dict[str, object]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    source_n = absolute.size
    selected = absolute if stride is None else absolute[::stride]
    total = float(selected.sum(dtype=np.float64))
    result: dict[str, object] = {"n": int(selected.size), "source_n": int(source_n)}
    if stride is not None:
        result["sample_stride"] = stride
    if total == 0.0:
        result.update({
            "effective_support": 0.0,
            "effective_support_fraction": 0.0,
            "gini": 0.0,
            "top_mass": {str(f): 0.0 for f in TOP_FRACTIONS},
        })
        return result
    ordered = np.sort(selected)[::-1]
    squared = float(np.sum(selected * selected, dtype=np.float64))
    result["effective_support"] = float(total * total / squared)
    result["effective_support_fraction"] = float(result["effective_support"] / selected.size)
    result["top_mass"] = {
        str(f): float(ordered[: max(1, math.ceil(f * selected.size))].sum(dtype=np.float64) / total)
        for f in TOP_FRACTIONS
    }
    ascending = ordered[::-1]
    n = ascending.size
    weighted = float(np.sum(np.arange(1, n + 1, dtype=np.float64) * ascending, dtype=np.float64))
    result["gini"] = float(2.0 * weighted / (n * total) - (n + 1.0) / n)
    return result


def top_fraction(values: np.ndarray, fraction: float) -> float:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    total = float(absolute.sum(dtype=np.float64))
    count = max(1, math.ceil(fraction * absolute.size))
    return float(np.partition(absolute, -count)[-count:].sum(dtype=np.float64) / total) if total else 0.0


def load_predictions(path: Path, score_key: str, residual_key: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    records = payload["records"]
    paths = [str(row["image_path"]) for row in records]
    native = np.stack([row["native_abnormal_probability"].numpy() for row in records]).astype(np.float32)
    score = np.stack([row[score_key].numpy() for row in records]).astype(np.float32)
    residual = np.stack([row[residual_key].numpy() for row in records]).astype(np.float32)
    metadata = {key: payload.get(key) for key in ("status", "protocol_id", "attempt_uuid", "held_class", "gt_used", "mask_reads", "held_gt_reads", "preregistration_sha256")}
    return paths, native, score, residual, metadata


def load_masks(paths: list[str]) -> tuple[np.ndarray, int]:
    metadata: dict[tuple[str, str], dict[str, object]] = {}
    for line in META.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        metadata[(str(row["class_name"]), str(row["image_path"]))] = row
    masks = np.zeros((len(paths), 518, 518), dtype=bool)
    reads = 0
    for index, path in enumerate(paths):
        row = metadata[("candle", path)]
        if int(row["label"]) == 0:
            continue
        with Image.open(VISA / str(row["mask_path"])) as handle:
            resized = handle.convert("L").resize((518, 518), Image.Resampling.NEAREST)
            masks[index] = np.asarray(resized, dtype=np.uint8) > 0
        reads += 1
    return masks, reads


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def rank_for(candidate: np.ndarray, native: np.ndarray) -> dict[str, object]:
    per_image_spearman: list[float] = []
    top_overlap = {str(f): [] for f in (0.001, 0.005, 0.01, 0.05)}
    displacement_sample: list[np.ndarray] = []
    gaps_sample: list[np.ndarray] = []
    flips_sample: list[np.ndarray] = []
    nonadjacent_flip: list[np.ndarray] = []
    nonadjacent_gap: list[np.ndarray] = []
    adjacent_flip_count = 0
    adjacent_positive_count = 0
    adjacent_flip_positive_count = 0
    adjacent_flipped_gaps: list[np.ndarray] = []
    for image_index in range(native.shape[0]):
        base = native[image_index].reshape(-1)
        value = candidate[image_index].reshape(-1)
        count = base.size
        native_order = np.argsort(-base, kind="mergesort")
        candidate_order = np.argsort(-value, kind="mergesort")
        native_rank = np.empty(count, dtype=np.int64)
        candidate_rank = np.empty(count, dtype=np.int64)
        native_rank[native_order] = np.arange(count)
        candidate_rank[candidate_order] = np.arange(count)
        per_image_spearman.append(correlation(native_rank, candidate_rank))
        displacement_sample.append(np.abs(native_rank - candidate_rank)[::17])
        for fraction in top_overlap:
            top_count = max(1, math.ceil(float(fraction) * count))
            top_overlap[fraction].append(float(len(set(native_order[:top_count]) & set(candidate_order[:top_count])) / top_count))
        gaps = base[native_order[:-1]] - base[native_order[1:]]
        flips = value[native_order[:-1]] < value[native_order[1:]]
        adjacent_flip_count += int(flips.sum())
        adjacent_positive_count += int((gaps > 0).sum())
        adjacent_flip_positive_count += int((flips & (gaps > 0)).sum())
        if np.any(flips):
            adjacent_flipped_gaps.append(gaps[flips])
        gaps_sample.append(gaps[::17])
        flips_sample.append(flips[::17])
        left = native_order[:-31:17]
        right = native_order[31::17]
        nonadjacent_flip.append(value[left] < value[right])
        nonadjacent_gap.append(base[left] - base[right])
    sampled_gaps = np.concatenate(gaps_sample)
    sampled_flips = np.concatenate(flips_sample)
    pair_flips = np.concatenate(nonadjacent_flip)
    pair_gaps = np.concatenate(nonadjacent_gap)
    quantile_levels = np.array([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    gap_edges = np.quantile(sampled_gaps, quantile_levels)
    bins = []
    for index in range(len(gap_edges) - 1):
        lower, upper = gap_edges[index], gap_edges[index + 1]
        selected = (sampled_gaps >= lower) & ((sampled_gaps <= upper) if index == len(gap_edges) - 2 else (sampled_gaps < upper))
        bins.append({
            "q_range": [float(quantile_levels[index]), float(quantile_levels[index + 1])],
            "native_gap_range": [float(lower), float(upper)],
            "n": int(selected.sum()),
            "flip_fraction": float(sampled_flips[selected].mean()) if np.any(selected) else None,
        })
    flipped_gaps = np.concatenate(adjacent_flipped_gaps) if adjacent_flipped_gaps else np.zeros(0)
    total_adjacent = native.shape[0] * (native.shape[1] * native.shape[2] - 1)
    return {
        "spearman_per_image": summary(np.asarray(per_image_spearman)),
        "top_overlap_mean": {key: float(np.mean(value)) for key, value in top_overlap.items()},
        "top_overlap_q10_q90": {key: [float(np.quantile(value, 0.10)), float(np.quantile(value, 0.90))] for key, value in top_overlap.items()},
        "rank_displacement_sample_abs_pixels": summary(np.concatenate(displacement_sample)),
        "rank_displacement_sample_fraction_of_pixels": summary(np.concatenate(displacement_sample) / float(native.shape[1] * native.shape[2])),
        "adjacent_native_order": {
            "sample_count": int(sampled_gaps.size),
            "flip_fraction_all_sample": float(sampled_flips.mean()),
            "flip_fraction_positive_gap_sample": float(sampled_flips[sampled_gaps > 0].mean()) if np.any(sampled_gaps > 0) else None,
            "exact_flip_count_all_images": int(adjacent_flip_count),
            "exact_pair_count_all_images": int(total_adjacent),
            "exact_flip_fraction_all_images": float(adjacent_flip_count / total_adjacent),
            "exact_positive_gap_pair_count": int(adjacent_positive_count),
            "exact_flip_fraction_positive_gap": float(adjacent_flip_positive_count / adjacent_positive_count) if adjacent_positive_count else None,
            "flipped_gap_summary": summary(flipped_gaps),
        },
        "nonadjacent_fixed_pairs": {
            "pair_count": int(pair_flips.size),
            "offset": 31,
            "stride": 17,
            "flip_fraction": float(pair_flips.mean()),
            "positive_gap_flip_fraction": float(pair_flips[pair_gaps > 0].mean()) if np.any(pair_gaps > 0) else None,
            "gap_summary": summary(pair_gaps),
        },
        "native_gap_quantile_bins": bins,
        "native_adjacent_gap_summary": summary(sampled_gaps),
    }


def enrichment(delta: np.ndarray, masks: np.ndarray) -> dict[str, object]:
    absolute = np.abs(delta)
    area = float(masks.mean())
    total_mass = float(absolute.sum(dtype=np.float64))
    anomaly_mass = float(absolute[masks].sum(dtype=np.float64))
    active = {}
    for threshold in (1e-6, 1e-4, 1e-3, 1e-2):
        selected = absolute > threshold
        active[str(threshold)] = {
            "fraction": float(selected.mean()),
            "anomaly_fraction_of_active": float(masks[selected].mean()) if np.any(selected) else 0.0,
            "anomaly_enrichment": float(masks[selected].mean() / area) if np.any(selected) and area else None,
        }
    top_anomaly = {}
    flat_absolute = absolute.reshape(-1)
    flat_masks = masks.reshape(-1)
    for fraction in (0.001, 0.005, 0.01, 0.05):
        count = max(1, math.ceil(fraction * flat_absolute.size))
        indexes = np.argpartition(flat_absolute, -count)[-count:]
        top_anomaly[str(fraction)] = float(flat_masks[indexes].mean())
    return {
        "mean_abs_normal": float(absolute[~masks].mean()),
        "mean_abs_anomaly": float(absolute[masks].mean()),
        "mass_fraction_in_anomaly": anomaly_mass / total_mass if total_mass else 0.0,
        "mass_enrichment_over_area": (anomaly_mass / total_mass) / area if total_mass and area else None,
        "active_fixed": active,
        "top_effect_pixel_anomaly_fraction": top_anomaly,
    }


def source_descriptors() -> dict[str, object]:
    exposures: list[tuple[str, np.ndarray]] = []
    for shard in sorted((CACHE / "tier_b").iterdir()):
        if not shard.is_dir() or not (shard / "manifest.json").exists():
            continue
        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        teacher = np.load(shard / "teacher_region.npy", mmap_mode="r")
        for index, sample_id in enumerate(manifest["sample_ids"]):
            exposures.append((str(sample_id), np.array(teacher[index], dtype=np.float32, copy=True)))
    unique: dict[str, np.ndarray] = {}
    for sample_id, value in exposures:
        if sample_id in unique and not np.array_equal(unique[sample_id], value):
            raise RuntimeError("duplicate source cache values disagree")
        unique.setdefault(sample_id, value)
    sample_ids = sorted(unique)
    teacher = np.stack([unique[sample_id] for sample_id in sample_ids]).astype(np.float32)
    absolute = np.abs(teacher)
    region_rms = np.sqrt(np.mean(teacher.astype(np.float64) ** 2, axis=(1, 2)))
    region_max = absolute.reshape(len(teacher), -1).max(axis=1)
    region_support = np.mean(absolute > RAW_EPSILON, axis=(1, 2))
    flat = absolute.reshape(len(teacher), -1)
    top_count = max(1, math.ceil(flat.shape[1] * 0.10))
    region_top10 = np.sort(flat, axis=1)[:, -top_count:].sum(axis=1) / np.maximum(flat.sum(axis=1), np.finfo(np.float64).tiny)
    effect_rms: list[float] = []
    effect_max: list[float] = []
    effect_top10: list[float] = []
    for start in range(0, len(teacher), 64):
        with torch.no_grad():
            effect = deployed_margin_effect(torch.from_numpy(teacher[start : start + 64])).numpy().astype(np.float32)
        effect_absolute = np.abs(effect)
        effect_rms.extend(np.sqrt(np.mean(effect.astype(np.float64) ** 2, axis=(1, 2))))
        effect_max.extend(effect_absolute.reshape(effect.shape[0], -1).max(axis=1))
        effect_flat = effect_absolute.reshape(effect.shape[0], -1)
        effect_top10.extend(np.sort(effect_flat, axis=1)[:, -max(1, math.ceil(effect_flat.shape[1] * 0.10)) :].sum(axis=1) / np.maximum(effect_flat.sum(axis=1), np.finfo(np.float64).tiny))
    functional_rms = np.asarray(effect_rms)
    descriptors = {
        "teacher_region_rms": region_rms,
        "teacher_functional_effect_rms": functional_rms,
        "teacher_region_max_abs": region_max,
        "teacher_region_support_gt_inherited_epsilon": region_support,
        "teacher_region_top10_abs_mass": region_top10,
        "teacher_functional_effect_top10_abs_mass": np.asarray(effect_top10),
        "teacher_functional_effect_max_abs": np.asarray(effect_max),
    }

    def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
        left_order = np.argsort(left, kind="mergesort")
        right_order = np.argsort(right, kind="mergesort")
        left_rank = np.empty_like(left_order, dtype=np.float64)
        right_rank = np.empty_like(right_order, dtype=np.float64)
        left_rank[left_order] = np.arange(len(left))
        right_rank[right_order] = np.arange(len(right))
        return correlation(left_rank, right_rank)

    output: dict[str, object] = {
        "sample_count": len(sample_ids),
        "exposure_count": len(exposures),
        "duplicate_exposure_count": len(exposures) - len(sample_ids),
        "classes": sorted({sample_id.split(":", 1)[0] for sample_id in sample_ids}),
        "descriptors": {},
    }
    descriptor_output = output["descriptors"]
    assert isinstance(descriptor_output, dict)
    for name, values in descriptors.items():
        by_class: dict[str, list[float]] = {}
        for sample_id, value in zip(sample_ids, values):
            by_class.setdefault(sample_id.split(":", 1)[0], []).append(float(value))
        medians = {name: float(np.median(value)) for name, value in by_class.items()}
        positive_medians = [value for value in medians.values() if value > 0]
        descriptor_output[name] = {
            "summary": summary(values),
            "category_medians": medians,
            "category_median_ratio_max_min": float(max(positive_medians) / min(positive_medians)) if positive_medians else None,
            "spearman_with_functional_effect_rms": rank_correlation(values, functional_rms),
        }
    return output


def main() -> None:
    torch.set_num_threads(4)
    p32_paths, native32, p32_score, p32_residual, p32_metadata = load_predictions(P32, "p32_abnormal_probability", "p32_region_residual")
    p30_paths, native30, p30_score, p30_residual, p30_metadata = load_predictions(P30R1, "p30r1_abnormal_probability", "p30r1_region_residual")
    if p32_paths != p30_paths or not np.array_equal(native32, native30):
        raise RuntimeError("frozen P32/P30R1 prediction alignment failed")
    masks, mask_reads = load_masks(p32_paths)
    native = native32
    p32_delta = p32_score - native
    p30_delta = p30_score - native
    output: dict[str, object] = {
        "status": "P33_OFFLINE_FORENSIC_COMPLETE",
        "execution": {
            "new_scientific_stage2": 0,
            "new_stage3": 0,
            "new_full_runs": 0,
            "new_clip_forwards": 0,
            "new_phase2b_forwards": 0,
            "new_teacher_forwards": 0,
            "cache_rebuilds": 0,
            "held_result_tuning_iterations": 0,
            "held_mask_reads_descriptive": mask_reads,
        },
        "payloads": {"p32": p32_metadata, "p30r1": p30_metadata, "path_order_equal": True, "native_maps_equal": True, "shape_native": list(native.shape), "shape_residual": list(p32_residual.shape)},
        "held_descriptive_masks": {"shape": list(masks.shape), "anomaly_pixel_fraction": float(masks.mean()), "anomaly_image_count": int(np.any(masks, axis=(1, 2)).sum())},
        "inherited_support_threshold": {"coordinate_abs_threshold": RAW_EPSILON, "source": "P30R1 forensic fixed coordinate epsilon; not held-tuned"},
        "residuals": {},
        "score_effects": {},
        "enrichment": {},
        "rank_forensic": {},
        "support_overlap": {},
        "source_actionability": source_descriptors(),
    }
    for name, residual in (("P30R1", p30_residual), ("P32", p32_residual)):
        coordinate = np.abs(residual)
        stage_mean = np.mean(residual, axis=1)
        output["residuals"][name] = {
            "summary": summary(residual),
            "absolute_quantiles": {str(q): float(np.quantile(coordinate, q)) for q in SUMMARY_Q},
            "concentration": concentration(residual),
            "fraction_abs_gt_inherited_threshold": float(np.mean(coordinate > RAW_EPSILON)),
            "stage_mean_fraction_abs_gt_inherited_threshold": float(np.mean(np.abs(stage_mean) > RAW_EPSILON)),
            "any_stage_fraction_abs_gt_inherited_threshold": float(np.mean(np.any(coordinate > RAW_EPSILON, axis=1))),
            "per_sample_l2": summary(np.linalg.norm(residual.reshape(residual.shape[0], -1), axis=1)),
        }
    for name, delta in (("P30R1", p30_delta), ("P32", p32_delta)):
        output["score_effects"][name] = {
            "summary": summary(delta),
            "concentration_full": concentration(delta),
            "concentration_fixed_stride_17": concentration(delta, 17),
            "near_zero_fractions": {str(t): float(np.mean(np.abs(delta) <= t)) for t in SCORE_THRESHOLDS},
            "top_mass_full": {str(f): top_fraction(delta, f) for f in TOP_FRACTIONS},
        }
        output["enrichment"][name] = enrichment(delta, masks)
        output["rank_forensic"][name] = rank_for(delta + native, native)
    for name, left, right in (
        ("coordinate", np.abs(p30_residual) > RAW_EPSILON, np.abs(p32_residual) > RAW_EPSILON),
        ("any_stage", np.any(np.abs(p30_residual) > RAW_EPSILON, axis=1), np.any(np.abs(p32_residual) > RAW_EPSILON, axis=1)),
        ("stage_mean", np.abs(np.mean(p30_residual, axis=1)) > RAW_EPSILON, np.abs(np.mean(p32_residual, axis=1)) > RAW_EPSILON),
    ):
        left = left.reshape(-1)
        right = right.reshape(-1)
        intersection = left & right
        union = left | right
        output["support_overlap"][name] = {
            "p30r1_active_fraction": float(left.mean()),
            "p32_active_fraction": float(right.mean()),
            "intersection_fraction": float(intersection.mean()),
            "union_fraction": float(union.mean()),
            "jaccard": float(intersection.sum() / union.sum()) if union.any() else 1.0,
            "p30r1_active_p32_active_containment": float(intersection.sum() / left.sum()) if left.any() else None,
            "p32_active_p30r1_active_containment": float(intersection.sum() / right.sum()) if right.any() else None,
            "p30r1_active_p32_inactive_fraction": float((left & ~right).mean()),
            "p30r1_inactive_p32_active_fraction": float((~left & right).mean()),
        }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
