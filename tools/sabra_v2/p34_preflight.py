"""Frozen/source-only P34 formulation falsification.

This module deliberately has no scientific runner and never opens the held
fold.  It combines an effect-space gradient check with deterministic toy
cases and a compact audit of the existing candle source cache.  The source
audit uses the already frozen P32/P33 deployment map; it does not run CLIP,
Phase2B, or a teacher model.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tools.sabra_v2.p29_contract import CORRECTION_SCALE
from tools.sabra_v2.p32_objective import deployed_margin_effect
from tools.sabra_v2.region_cache import atomic_write_json


ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
SOURCE_SHARD = CACHE_ROOT / "tier_b" / "candle"
OUTPUT = ROOT / "research/sabra_v2/region_distill/P34_PREFLIGHT_FALSIFICATION.json"
SMOOTH_L1_BETA = 1.0
SEED = 3400


def _json_number(value: Any) -> Any:
    """Convert numpy/scalar values while keeping non-finite values explicit."""
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, (float, int)):
        return value if math.isfinite(float(value)) else None
    return value


def _tensor_stats(values: torch.Tensor) -> dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    abs_flat = flat.abs()
    qs = torch.quantile(abs_flat, torch.tensor([0.50, 0.90, 0.95, 0.99]))
    return {
        "mean_abs": _json_number(abs_flat.mean().item()),
        "q50_abs": _json_number(qs[0].item()),
        "q90_abs": _json_number(qs[1].item()),
        "q95_abs": _json_number(qs[2].item()),
        "q99_abs": _json_number(qs[3].item()),
        "max_abs": _json_number(abs_flat.max().item()),
        "finite": bool(torch.isfinite(flat).all()),
    }


def _gini(values: np.ndarray) -> float:
    flat = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    if flat.size == 0 or float(flat.sum()) == 0.0:
        return 0.0
    ordered = np.sort(flat)
    index = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * ordered) / (ordered.size * ordered.sum())) - (ordered.size + 1.0) / ordered.size)


def _concentration(values: np.ndarray) -> dict[str, Any]:
    flat = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    total = float(flat.sum())
    ordered = np.sort(flat)[::-1]
    return {
        "effective_support": float((total * total) / max(float(np.square(flat).sum()), 1e-30)),
        "effective_support_fraction": float((total * total) / max(float(np.square(flat).sum()), 1e-30) / flat.size),
        "gini": _gini(flat),
        "top_1_mass": float(ordered[: max(1, flat.size // 100)].sum() / max(total, 1e-30)),
        "top_5_mass": float(ordered[: max(1, flat.size // 20)].sum() / max(total, 1e-30)),
        "top_10_mass": float(ordered[: max(1, flat.size // 10)].sum() / max(total, 1e-30)),
    }


def _smooth_l1(error: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(error, torch.zeros_like(error), beta=SMOOTH_L1_BETA, reduction="none")


def _effect_objectives(
    student: torch.Tensor,
    teacher: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate P33 and P34 in effect space with independently supplied w.

    Supplying w independently is intentional for the algebraic counterexample
    w=0, teacher!=0.  The actual source rule is audited separately below and
    always derives w from the teacher effect.
    """
    student_p33 = student.detach().clone().requires_grad_(True)
    student_p34 = student.detach().clone().requires_grad_(True)
    teacher = teacher.detach()
    weight = weight.detach()
    p33_pointwise = weight * _smooth_l1(student_p33 - teacher)
    p34_target = (weight * teacher).detach()
    p34_pointwise = _smooth_l1(student_p34 - p34_target)
    p33_loss = p33_pointwise.mean()
    p34_loss = p34_pointwise.mean()
    p33_grad = torch.autograd.grad(p33_loss, student_p33)[0]
    p34_grad = torch.autograd.grad(p34_loss, student_p34)[0]

    def per_sample_norm(gradient: torch.Tensor) -> torch.Tensor:
        return gradient.reshape(gradient.shape[0], -1).norm(dim=1)

    p33_sample_norm = per_sample_norm(p33_grad)
    p34_sample_norm = per_sample_norm(p34_grad)
    positive = p34_sample_norm[p34_sample_norm > 0]
    dominance = float(positive.max().item() / max(float(torch.median(positive).item()), 1e-30)) if positive.numel() else 0.0
    restoring_dot = float((p34_grad * student_p34.detach()).sum().item())
    return {
        "p33": {
            "loss": _json_number(p33_loss.item()),
            "gradient_l2": _json_number(p33_grad.norm().item()),
            "max_gradient": _json_number(p33_grad.abs().max().item()),
            "nonzero_gradient_fraction": float((p33_grad.abs() > 0).float().mean().item()),
            "finite": bool(torch.isfinite(p33_grad).all() and torch.isfinite(p33_loss)),
            "sample_gradient_norms": [_json_number(x) for x in p33_sample_norm.tolist()],
        },
        "p34": {
            "loss": _json_number(p34_loss.item()),
            "gradient_l2": _json_number(p34_grad.norm().item()),
            "max_gradient": _json_number(p34_grad.abs().max().item()),
            "nonzero_gradient_fraction": float((p34_grad.abs() > 0).float().mean().item()),
            "finite": bool(torch.isfinite(p34_grad).all() and torch.isfinite(p34_loss)),
            "sample_gradient_norms": [_json_number(x) for x in p34_sample_norm.tolist()],
            "max_to_median_positive_sample_gradient": _json_number(dominance),
            "gradient_student_dot": _json_number(restoring_dot),
        },
        "target": _tensor_stats(p34_target),
        "target_exact_zero_fraction": float((p34_target == 0).float().mean().item()),
    }


def _synthetic_suite() -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    cases: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    shape = (4, 5, 7)
    cases.append(("zero_restoring_teacher_nonzero", torch.ones(shape), torch.full(shape, 2.0), torch.zeros(shape)))
    cases.append(("zero_optimum", torch.zeros(shape), torch.zeros(shape), torch.zeros(shape)))
    cases.append(("all_actionable_w1", torch.randn(shape, generator=generator), torch.full(shape, 2.0), torch.ones(shape)))
    cases.append(("intermediate_w05", torch.randn(shape, generator=generator), torch.full(shape, 2.0), torch.full(shape, 0.5)))
    cases.append(("tiny_teacher_effect", torch.randn(shape, generator=generator), torch.full(shape, 0.01), torch.full(shape, 0.01)))
    cases.append(("huge_teacher_effect", torch.randn(shape, generator=generator), torch.full(shape, 100.0), torch.ones(shape)))
    heavy_teacher = torch.ones(shape)
    heavy_teacher[3] = 10_000.0
    heavy_student = torch.randn(shape, generator=generator)
    cases.append(("heavy_tail", heavy_student, heavy_teacher, torch.tensor([[[0.0] * 7] * 5, [[0.0] * 7] * 5, [[0.0] * 7] * 5, [[1.0] * 7] * 5])))
    mixed_weight = torch.zeros((10, 5, 7))
    mixed_weight[9] = 1.0
    cases.append(("mixed_90pct_abstain_10pct_active", torch.randn((10, 5, 7), generator=generator), torch.ones((10, 5, 7)), mixed_weight))
    continuous_weight = torch.linspace(0.0, 1.0, steps=4).reshape(4, 1, 1).expand(shape)
    cases.append(("mixed_continuous_weights", torch.randn(shape, generator=generator), torch.full(shape, 2.0), continuous_weight))
    cases.append(("all_abstain", torch.randn(shape, generator=generator), torch.full(shape, 3.0), torch.zeros(shape)))
    cases.append(("all_active", torch.randn(shape, generator=generator), torch.randn(shape, generator=generator), torch.ones(shape)))
    outlier_student = torch.randn(shape, generator=generator)
    outlier_student[3] = 1_000.0
    cases.append(("one_extreme_student_outlier", outlier_student, torch.zeros(shape), torch.zeros(shape)))
    cases.append(("sign_reversed_student", -torch.ones(shape), torch.ones(shape), torch.ones(shape)))
    cases.append(("near_zero_student", torch.full(shape, 1e-9), torch.zeros(shape), torch.zeros(shape)))
    cases.append(("teacher_effect_exactly_zero", torch.randn(shape, generator=generator), torch.zeros(shape), torch.zeros(shape)))

    result: dict[str, Any] = {}
    for name, student, teacher, weight in cases:
        result[name] = {
            "student": _tensor_stats(student),
            "teacher": _tensor_stats(teacher),
            "weight": _tensor_stats(weight),
            "weight_zero_fraction": float((weight == 0).float().mean().item()),
            "weight_one_fraction": float((weight == 1).float().mean().item()),
            **_effect_objectives(student, teacher, weight),
        }

    zero = result["zero_restoring_teacher_nonzero"]
    zero_opt = result["zero_optimum"]
    w1 = result["all_actionable_w1"]
    mid = result["intermediate_w05"]
    # These are deliberately algebraic gates, not performance thresholds.
    result["gates"] = {
        "zero_restoring_gradient": bool(zero["p33"]["gradient_l2"] == 0.0 and zero["p34"]["gradient_l2"] > 0.0 and zero["p34"]["gradient_student_dot"] > 0.0),
        "zero_optimum": bool(zero_opt["p34"]["gradient_l2"] == 0.0 and zero_opt["p34"]["loss"] == 0.0),
        "w1_functional_behavior_finite": bool(w1["p34"]["finite"]),
        "intermediate_target_has_signal": bool(mid["target"]["mean_abs"] > 0.0 and mid["p34"]["finite"]),
        "all_synthetic_finite": bool(all(item.get("p34", {}).get("finite", False) for key, item in result.items() if key != "gates")),
        "mixed_batch_not_unbounded": bool(result["mixed_90pct_abstain_10pct_active"]["p34"]["max_to_median_positive_sample_gradient"] < 100.0),
    }
    return result


def _source_summary(values: np.ndarray, threshold: float) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    abs_flat = np.abs(flat)
    quantiles = np.quantile(abs_flat, [0.5, 0.9, 0.95, 0.99])
    concentration = _concentration(flat)
    return {
        "mean_abs": float(abs_flat.mean()),
        "q50_abs": float(quantiles[0]),
        "q90_abs": float(quantiles[1]),
        "q95_abs": float(quantiles[2]),
        "q99_abs": float(quantiles[3]),
        "max_abs": float(abs_flat.max()),
        "exact_zero_fraction": float(np.mean(flat == 0)),
        "near_zero_le_1e-6_fraction": float(np.mean(abs_flat <= 1e-6)),
        "meaningful_gt_C_over_100_fraction": float(np.mean(abs_flat > threshold)),
        "finite": bool(np.isfinite(flat).all()),
        **concentration,
    }


def _source_audit() -> dict[str, Any]:
    manifest_path = SOURCE_SHARD / "manifest.json"
    teacher_path = SOURCE_SHARD / "teacher_region.npy"
    manifest = json.loads(manifest_path.read_text())
    teacher = np.load(teacher_path, mmap_mode="r", allow_pickle=False)
    if manifest.get("completion_status") != "COMPLETE":
        raise RuntimeError("source cache is not complete")
    if manifest.get("sample_count") != int(teacher.shape[0]):
        raise RuntimeError("source manifest/tensor count mismatch")
    # Older P27 manifests omit ``contains_gt`` and record source-mask
    # production reads from cache construction.  The forensic process only
    # opens teacher_region.npy; it requires no source mask and no held mask.
    if manifest.get("contains_gt", False) is True or manifest.get("held_mask_reads") != 0:
        raise RuntimeError("source cache is not an allowed GT-free fit cache")

    rng = np.random.default_rng(SEED)
    total = int(teacher.shape[0] * 518 * 518)
    sample_target = 300_000
    # Deterministic systematic sampling preserves category order while keeping
    # the JSON compact.  All exact counts below are streamed over all source
    # pixels; no held shard is touched.
    sample_stride = max(1, total // sample_target)
    sample_indices = np.arange(0, total, sample_stride, dtype=np.int64)
    if sample_indices.size > sample_target:
        sample_indices = sample_indices[:sample_target]
    raw_sample: list[np.ndarray] = []
    target_sample: list[np.ndarray] = []
    weight_sample: list[np.ndarray] = []
    category_values: dict[str, list[np.ndarray]] = {}
    exact = {
        "pixels": 0,
        "raw_zero": 0,
        "raw_near_zero": 0,
        "raw_gt_threshold": 0,
        "target_zero": 0,
        "target_near_zero": 0,
        "target_gt_threshold": 0,
        "weight_zero": 0,
        "weight_one": 0,
        "weight_gt_075": 0,
        "weight_gt_09": 0,
        "ratio_ge_1": 0,
        "raw_abs_sum": 0.0,
        "target_abs_sum": 0.0,
        "raw_sq_sum": 0.0,
        "target_sq_sum": 0.0,
    }
    threshold = float(CORRECTION_SCALE) / 100.0
    cursor = 0
    for start in range(0, teacher.shape[0], 8):
        raw_input = np.asarray(teacher[start : start + 8], dtype=np.float32).copy()
        raw = deployed_margin_effect(torch.from_numpy(raw_input)).numpy()
        weight = np.clip(np.abs(raw) / float(CORRECTION_SCALE), 0.0, 1.0)
        target = weight * raw
        flat_raw = raw.reshape(-1)
        flat_target = target.reshape(-1)
        flat_weight = weight.reshape(-1)
        end = cursor + flat_raw.size
        selected = sample_indices[(sample_indices >= cursor) & (sample_indices < end)] - cursor
        if selected.size:
            raw_sample.append(flat_raw[selected])
            target_sample.append(flat_target[selected])
            weight_sample.append(flat_weight[selected])
        exact["pixels"] += int(flat_raw.size)
        raw_abs = np.abs(flat_raw)
        target_abs = np.abs(flat_target)
        exact["raw_zero"] += int(np.count_nonzero(flat_raw == 0))
        exact["raw_near_zero"] += int(np.count_nonzero(raw_abs <= 1e-6))
        exact["raw_gt_threshold"] += int(np.count_nonzero(raw_abs > threshold))
        exact["target_zero"] += int(np.count_nonzero(flat_target == 0))
        exact["target_near_zero"] += int(np.count_nonzero(target_abs <= 1e-6))
        exact["target_gt_threshold"] += int(np.count_nonzero(target_abs > threshold))
        exact["weight_zero"] += int(np.count_nonzero(flat_weight == 0))
        exact["weight_one"] += int(np.count_nonzero(flat_weight == 1))
        exact["weight_gt_075"] += int(np.count_nonzero(flat_weight > 0.75))
        exact["weight_gt_09"] += int(np.count_nonzero(flat_weight > 0.9))
        exact["ratio_ge_1"] += int(np.count_nonzero(np.abs(flat_raw) / float(CORRECTION_SCALE) >= 1.0))
        exact["raw_abs_sum"] += float(raw_abs.sum())
        exact["target_abs_sum"] += float(target_abs.sum())
        exact["raw_sq_sum"] += float(np.square(flat_raw).sum())
        exact["target_sq_sum"] += float(np.square(flat_target).sum())
        cursor = end
    raw_values = np.concatenate(raw_sample)
    target_values = np.concatenate(target_sample)
    weight_values = np.concatenate(weight_sample)
    ratio_values = np.abs(raw_values) / float(CORRECTION_SCALE)
    alternative_weight_values = ratio_values / (1.0 + ratio_values)
    alternative_target_values = alternative_weight_values * raw_values
    raw_full_count = exact["pixels"]
    sample_stats = {
        "sample_pixels": int(raw_values.size),
        "raw_Et": _source_summary(raw_values, threshold),
        "P34_target_wEt": _source_summary(target_values, threshold),
        "alternative_unsaturated_target": {
            "weight": _source_summary(alternative_weight_values, 0.0),
            "target": _source_summary(alternative_target_values, threshold),
            "target_rms_ratio_to_raw": float(np.sqrt(np.mean(np.square(alternative_target_values)) / max(np.mean(np.square(raw_values)), 1e-30))),
            "target_abs_mass_ratio_to_raw": float(np.sum(np.abs(alternative_target_values)) / max(np.sum(np.abs(raw_values)), 1e-30)),
            "reason_not_selected": "Changes the actionability transform and target simultaneously; saturation was observed but not causally isolated, so it is a confounded alternative rather than the clean P34 test.",
        },
        "weight": {
            "zero_fraction": exact["weight_zero"] / raw_full_count,
            "0_lt_w_lt_025_fraction": float(np.mean((weight_values > 0) & (weight_values < 0.25))),
            "025_le_w_lt_05_fraction": float(np.mean((weight_values >= 0.25) & (weight_values < 0.5))),
            "05_le_w_lt_075_fraction": float(np.mean((weight_values >= 0.5) & (weight_values < 0.75))),
            "075_le_w_lt_1_fraction": float(np.mean((weight_values >= 0.75) & (weight_values < 1.0))),
            "one_fraction": exact["weight_one"] / raw_full_count,
            "gt_075_fraction": exact["weight_gt_075"] / raw_full_count,
            "gt_09_fraction": exact["weight_gt_09"] / raw_full_count,
            "preclamp_ratio_ge_1_fraction": exact["ratio_ge_1"] / raw_full_count,
            "sample_mean": float(weight_values.mean()),
            "sample_median": float(np.median(weight_values)),
            "sample_q90": float(np.quantile(weight_values, 0.90)),
            "sample_q95": float(np.quantile(weight_values, 0.95)),
            "sample_q99": float(np.quantile(weight_values, 0.99)),
        },
        "mass_ratio_target_to_raw": exact["target_abs_sum"] / max(exact["raw_abs_sum"], 1e-30),
        "rms_ratio_target_to_raw": math.sqrt(exact["target_sq_sum"] / max(exact["raw_sq_sum"], 1e-30)),
    }

    rows = manifest.get("rows", manifest.get("records", []))
    if not rows and manifest.get("sample_ids"):
        rows = [{"class_name": str(sample_id).split(":", 1)[0]} for sample_id in manifest["sample_ids"]]
    # The canonical manifest stores source records in a companion metadata
    # file in some cache versions.  Sample IDs provide the category identity
    # without opening any image or label file.
    metadata_path = SOURCE_SHARD / "records.json"
    if metadata_path.is_file():
        rows = json.loads(metadata_path.read_text())
    if rows and len(rows) == teacher.shape[0]:
        for index, row in enumerate(rows):
            category_values.setdefault(str(row.get("class_name", "unknown")), []).append(teacher[index])
        category_summary: dict[str, Any] = {}
        for category, arrays in sorted(category_values.items()):
            cat_raw = np.concatenate([deployed_margin_effect(torch.from_numpy(np.asarray(x, dtype=np.float32).copy())[None]).numpy().reshape(-1) for x in arrays])
            cat_w = np.clip(np.abs(cat_raw) / float(CORRECTION_SCALE), 0.0, 1.0)
            cat_t = cat_w * cat_raw
            category_summary[category] = {
                "records": len(arrays),
                "raw_rms": float(np.sqrt(np.mean(np.square(cat_raw)))),
                "target_rms": float(np.sqrt(np.mean(np.square(cat_t)))),
                "weight_mean": float(cat_w.mean()),
                "weight_zero_fraction": float(np.mean(cat_w == 0)),
                "weight_one_fraction": float(np.mean(cat_w == 1)),
                "raw_meaningful_support": float(np.mean(np.abs(cat_raw) > threshold)),
                "target_meaningful_support": float(np.mean(np.abs(cat_t) > threshold)),
            }
        sample_stats["category_summary"] = category_summary

    exact_summary = {
        "pixels": raw_full_count,
        "raw_exact_zero_fraction": exact["raw_zero"] / raw_full_count,
        "raw_near_zero_fraction": exact["raw_near_zero"] / raw_full_count,
        "raw_meaningful_support_fraction": exact["raw_gt_threshold"] / raw_full_count,
        "target_exact_zero_fraction": exact["target_zero"] / raw_full_count,
        "target_near_zero_fraction": exact["target_near_zero"] / raw_full_count,
        "target_meaningful_support_fraction": exact["target_gt_threshold"] / raw_full_count,
        "weight_zero_fraction": exact["weight_zero"] / raw_full_count,
        "weight_one_fraction": exact["weight_one"] / raw_full_count,
        "weight_gt_075_fraction": exact["weight_gt_075"] / raw_full_count,
        "weight_gt_09_fraction": exact["weight_gt_09"] / raw_full_count,
        "preclamp_ratio_ge_1_fraction": exact["ratio_ge_1"] / raw_full_count,
        "raw_abs_sum": exact["raw_abs_sum"],
        "target_abs_sum": exact["target_abs_sum"],
        "raw_sq_sum": exact["raw_sq_sum"],
        "target_sq_sum": exact["target_sq_sum"],
    }
    return {
        "cache_root": str(CACHE_ROOT),
        "source_shard": str(SOURCE_SHARD),
        "manifest_sha256": None,
        "records": int(teacher.shape[0]),
        "teacher_region_shape": list(teacher.shape),
        "cache_contains_gt": manifest.get("contains_gt"),
        "held_mask_reads": manifest.get("held_mask_reads"),
        "correction_scale_C": float(CORRECTION_SCALE),
        "meaningful_threshold": "C/100",
        "meaningful_threshold_value": threshold,
        "exact_counts": exact_summary,
        "sample_stats": sample_stats,
        "sampling": {
            "method": "deterministic_systematic_flat_index",
            "seed": SEED,
            "sample_stride": sample_stride,
            "sample_pixels": int(raw_values.size),
        },
    }


def main() -> None:
    started = time.perf_counter()
    synthetic = _synthetic_suite()
    source = _source_audit()
    gates = synthetic["gates"]
    # The source gate is deliberately a structural change, not a performance
    # target: the same frozen C/100 threshold is applied to raw and shaped
    # source targets, and target mass must remain nontrivial.
    raw_support = source["exact_counts"]["raw_meaningful_support_fraction"]
    target_support = source["exact_counts"]["target_meaningful_support_fraction"]
    target_zero = source["exact_counts"]["target_exact_zero_fraction"]
    source_gate = {
        "meaningful_support_reduced": bool(target_support < raw_support),
        "nonzero_target_not_collapsed": bool(target_zero < 0.5),
        "target_rms_preserved": bool(source["sample_stats"]["rms_ratio_target_to_raw"] > 0.5),
        "target_abs_mass_preserved": bool(source["sample_stats"]["mass_ratio_target_to_raw"] > 0.5),
    }
    all_gates = {**gates, **{f"source_{key}": value for key, value in source_gate.items()}}
    payload = {
        "schema": "P34_PREFLIGHT_FALSIFICATION_V1",
        "protocol": "P34",
        "status": "P34_PREFLIGHT_PASS" if all(all_gates.values()) else "P34_PREFLIGHT_FAIL",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_execution_uuid": None,
        "new_scientific_stage2_attempts": 0,
        "new_stage3_attempts": 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
        "cache_rebuilds": 0,
        "held_reads": 0,
        "selected_hypothesis": "EXPLICIT_ACTIONABILITY_TARGET_FUNCTIONAL_TRANSFER",
        "equations": {
            "p33": "mean(w * SmoothL1(E_s - E_t; beta=1))",
            "p34": "mean(SmoothL1(E_s - stop_gradient(w * E_t); beta=1))",
            "weight": "stop_gradient(clamp(abs(E_t)/C, 0, 1))",
            "smooth_l1_derivative": "psi_beta(u)=sign(u)*min(abs(u)/beta,1)",
            "p33_gradient": "dL/dE_s = w * psi_beta(E_s-E_t)/N",
            "p34_gradient": "dL/dE_s = psi_beta(E_s-w*E_t)/N",
        },
        "gradient_conclusion": {
            "w_zero_removes_p33_gradient": True,
            "w_zero_restores_p34_gradient_to_zero": True,
            "actual_rule_note": "In the actual source rule w=0 implies E_t=0; the synthetic decoupled w=0,E_t!=0 case isolates the operator algebra.",
        },
        "synthetic": synthetic,
        "source_only": source,
        "gates": all_gates,
        "runtime_seconds": time.perf_counter() - started,
        "forbidden_evidence_used": {
            "held_gt": False,
            "held_masks": False,
            "held_metrics": False,
            "held_threshold_tuning": False,
            "new_neural_forwards": False,
        },
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps({"path": str(OUTPUT), "status": payload["status"], "gates": all_gates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
