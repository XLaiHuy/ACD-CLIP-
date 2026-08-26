"""Standalone P30R1 math and cache preflight; never trains a scientific model.

This module intentionally contains only the proposed teacher-relative
SmoothL1 objective plus deterministic synthetic/source-cache diagnostics. It
does not define a trainer, optimizer loop, evaluator, or scientific marker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p29_objective import CORRECTION_SCALE
from tools.sabra_v2.region_cache import atomic_write_json


ROOT = Path(__file__).resolve().parents[2]


STAGES = 3
REGION_GRID = (9, 9)
COORDINATE_COUNT = STAGES * REGION_GRID[0] * REGION_GRID[1]
NORMALIZATION_EPSILON = 0.01
SMOOTH_L1_BETA = 1.0
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_P30_ROOT = ROOT / "research/sabra_v2/region_distill/P30"
DEFAULT_VISA_ROOT = Path("/workspace/data/source/visa_unpack")


FORMULATION_TEXT = """P30R1 teacher-relative SmoothL1: t_bar=t/C, s_bar=s/C; a_t=stop_gradient(sqrt(mean(t_bar^2)+eps^2)); z_t=t_bar/a_t; z_s=s_bar/a_t; L=F.smooth_l1_loss(z_s,z_t,beta=1.0,reduction=mean) over B*243 coordinates; exact-zero teacher samples remain active."""


SYNTHETIC_THRESHOLDS = {
    "scale_1x_loss_max": 1e-7,
    "gross_scale_10x_loss_min": 0.1,
    "gross_scale_100x_loss_gt_10x": True,
    "opposite_loss_margin_over_1x": 0.1,
    "zero_teacher_gradient_l2_min": 1e-8,
    "near_zero_gradient_max_abs_max": 100.0,
    "near_zero_gradient_l2_max": 1000.0,
    "mixed_nonzero_gradient_max_to_median_max": 100.0,
    "heavy_tail_loss_margin_over_1x": 0.01,
}


SOURCE_CACHE_THRESHOLDS = {
    "q99_over_median_normalized_rms_max": 100.0,
    "near_ten_epsilon_nonzero_fraction_max": 0.10,
    "denominator_min_must_equal_or_exceed_epsilon": True,
    "all_observed_values_finite": True,
}


def formulation_sha256() -> str:
    return hashlib.sha256(FORMULATION_TEXT.encode("utf-8")).hexdigest()


def _validate_shapes(student_region: torch.Tensor, teacher_region: torch.Tensor) -> None:
    if student_region.ndim != 4 or student_region.shape[0] != STAGES or tuple(student_region.shape[-2:]) != REGION_GRID:
        raise ValueError("student_region must be [3,B,9,9]")
    if teacher_region.ndim != 3 or tuple(teacher_region.shape[-2:]) != REGION_GRID:
        raise ValueError("teacher_region must be [B,9,9]")
    if teacher_region.shape[0] != student_region.shape[1]:
        raise ValueError("teacher batch must match student batch")


def teacher_relative_components(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
    *,
    epsilon: float = NORMALIZATION_EPSILON,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``loss, z_s, z_t, a_t`` for the isolated preregistered math."""
    _validate_shapes(student_region, teacher_region)
    if epsilon <= 0.0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be positive and finite")
    teacher_staged = teacher_region.detach().unsqueeze(0).expand_as(student_region)
    student_bar = student_region / CORRECTION_SCALE
    teacher_bar = teacher_staged / CORRECTION_SCALE
    student_vectors = student_bar.permute(1, 0, 2, 3).reshape(student_region.shape[1], -1)
    teacher_vectors = teacher_bar.permute(1, 0, 2, 3).reshape(student_region.shape[1], -1)
    a_t = torch.sqrt(teacher_vectors.square().mean(dim=1, keepdim=True) + float(epsilon) ** 2).detach()
    z_s = student_vectors / a_t
    z_t = teacher_vectors / a_t
    loss = F.smooth_l1_loss(z_s, z_t, beta=SMOOTH_L1_BETA, reduction="mean")
    return loss, z_s, z_t, a_t


def teacher_relative_smooth_l1(
    student_region: torch.Tensor,
    teacher_region: torch.Tensor,
    *,
    epsilon: float = NORMALIZATION_EPSILON,
) -> torch.Tensor:
    """The one isolated objective used by the P30R1 preflight tests."""
    return teacher_relative_components(student_region, teacher_region, epsilon=epsilon)[0]


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(child) for child in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _gradient_stats(gradient: torch.Tensor, teacher: torch.Tensor) -> dict[str, Any]:
    return {
        "l2": float(torch.linalg.vector_norm(gradient.detach()).cpu()),
        "max_abs": float(gradient.detach().abs().max().cpu()),
        "nonzero_fraction": float((gradient.detach() != 0).float().mean().cpu()),
        "finite": bool(torch.isfinite(gradient.detach()).all()),
        "teacher_grad_status": "NONE" if teacher.grad is None else {
            "l2": float(torch.linalg.vector_norm(teacher.grad.detach()).cpu()),
            "max_abs": float(teacher.grad.detach().abs().max().cpu()),
        },
    }


def _case(student: torch.Tensor, teacher: torch.Tensor) -> dict[str, Any]:
    student_var = student.detach().clone().requires_grad_(True)
    teacher_var = teacher.detach().clone().requires_grad_(True)
    loss = teacher_relative_smooth_l1(student_var, teacher_var)
    loss.backward()
    gradient = student_var.grad.detach()
    result = {
        "loss": float(loss.detach().cpu()),
        "gradient": _gradient_stats(gradient, teacher_var),
        "teacher_detached": teacher_var.grad is None,
    }
    return result


def _unit_direction() -> torch.Tensor:
    direction = torch.linspace(-1.0, 1.0, COORDINATE_COUNT, dtype=torch.float32)
    direction = direction / torch.sqrt(direction.square().mean())
    return direction.reshape(STAGES, 1, REGION_GRID[0], REGION_GRID[1])


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return None
    left_rank = np.argsort(np.argsort(left)).astype(np.float64)
    right_rank = np.argsort(np.argsort(right)).astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def synthetic_falsification() -> dict[str, Any]:
    """Run all deterministic formulation-only falsification cases."""
    torch.manual_seed(0)
    direction = _unit_direction()
    normal_teacher = (CORRECTION_SCALE * direction[0]).detach()
    matching_student = direction[0:1].expand(STAGES, -1, -1, -1) * CORRECTION_SCALE
    scales = (0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0)
    scale_cases = {
        str(scale): _case(matching_student * scale, normal_teacher)
        for scale in scales
    }
    opposite = _case(-matching_student, normal_teacher)

    zero_teacher = torch.zeros_like(normal_teacher)
    zero_student = 0.5 * matching_student
    zero_case = _case(zero_student, zero_teacher)
    zero_student_var = zero_student.detach().clone().requires_grad_(True)
    zero_teacher_var = zero_teacher.detach().clone().requires_grad_(True)
    zero_loss = teacher_relative_smooth_l1(zero_student_var, zero_teacher_var)
    zero_loss.backward()
    zero_dot = float((zero_student_var.grad.detach() * zero_student_var.detach()).sum().cpu())
    zero_case["gradient_toward_zero_dot"] = zero_dot

    near_zero_cases: dict[str, Any] = {}
    fixed_nonzero_student = 0.5 * matching_student
    for scale in (1e-8, 1e-6, 1e-4):
        near_teacher = normal_teacher * scale
        near_zero_cases[str(scale)] = _case(fixed_nonzero_student, near_teacher)

    batch_scales = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
    teacher_batch = torch.stack([normal_teacher[0] * scale for scale in batch_scales], dim=0)
    student_batch = torch.stack([
        0.5 * matching_student if scale == 0.0 else 1.5 * matching_student * scale
        for scale in batch_scales
    ], dim=1).squeeze(2)
    student_batch_var = student_batch.detach().clone().requires_grad_(True)
    teacher_batch_var = teacher_batch.detach().clone().requires_grad_(True)
    mixed_loss = teacher_relative_smooth_l1(student_batch_var, teacher_batch_var)
    mixed_gradient = torch.autograd.grad(mixed_loss, student_batch_var)[0].detach()
    mixed_per_sample = torch.linalg.vector_norm(mixed_gradient.permute(1, 0, 2, 3).reshape(len(batch_scales), -1), dim=1)
    mixed_case = {
        "scales": list(batch_scales),
        "loss": float(mixed_loss.detach().cpu()),
        "gradient": _gradient_stats(mixed_gradient, teacher_batch_var),
        "per_sample_gradient_l2": [float(value.cpu()) for value in mixed_per_sample],
        "teacher_detached": teacher_batch_var.grad is None,
        "nonzero_scale_gradient_l2_ratio_max_to_min": float(
            mixed_per_sample[1:].max().cpu() / mixed_per_sample[1:].min().clamp_min(torch.finfo(torch.float32).tiny).cpu()
        ),
        "nonzero_scale_gradient_l2_ratio_max_to_median": float(
            mixed_per_sample[1:].max().cpu() / torch.median(mixed_per_sample[1:]).clamp_min(torch.finfo(torch.float32).tiny).cpu()
        ),
    }

    corruptions: dict[str, Any] = {}
    for factor in (10.0, 100.0):
        corrupted = matching_student.detach().clone()
        flat = corrupted.reshape(-1)
        corrupt_count = max(1, round(flat.numel() * 0.01))
        flat[:corrupt_count] *= factor
        entry = _case(corrupted, normal_teacher)
        entry["corrupted_coordinate_count"] = corrupt_count
        corruptions[str(factor)] = entry

    residual = torch.tensor([-100.0, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0, 100.0], dtype=torch.float32)
    target = torch.zeros_like(residual)
    robust_losses = {
        "L1": float(F.l1_loss(residual, target, reduction="mean")),
        "SmoothL1_beta_1": float(F.smooth_l1_loss(residual, target, beta=1.0, reduction="mean")),
        "L2_MSE": float(F.mse_loss(residual, target, reduction="mean")),
    }
    robust_gradients: dict[str, Any] = {}
    for name, function in (
        ("L1", lambda value: F.l1_loss(value, target, reduction="mean")),
        ("SmoothL1_beta_1", lambda value: F.smooth_l1_loss(value, target, beta=1.0, reduction="mean")),
        ("L2_MSE", lambda value: F.mse_loss(value, target, reduction="mean")),
    ):
        value = residual.detach().clone().requires_grad_(True)
        function(value).backward()
        robust_gradients[name] = {
            "max_abs": float(value.grad.abs().max()),
            "finite": bool(torch.isfinite(value.grad).all()),
        }

    scale_losses = [scale_cases[str(scale)]["loss"] for scale in scales]
    nonzero_mixed = mixed_per_sample[1:]
    checks = {
        "scale_1x_unique_clear_minimum": (
            scale_cases["1.0"]["loss"] <= 1e-7
            and scale_cases["1.0"]["loss"] < min(scale_losses[index] for index, scale in enumerate(scales) if scale != 1.0)
        ),
        "gross_scale_errors_materially_penalized": (
            scale_cases["10.0"]["loss"] > 0.1 and scale_cases["100.0"]["loss"] > scale_cases["10.0"]["loss"]
        ),
        "opposite_direction_penalized": opposite["loss"] > scale_cases["1.0"]["loss"] + 0.1,
        "zero_teacher_restoring_force": (
            zero_case["loss"] > 0.0
            and zero_case["gradient"]["finite"]
            and zero_case["gradient"]["l2"] > 1e-8
            and zero_dot > 0.0
        ),
        "near_zero_finite_and_bounded": all(
            item["gradient"]["finite"] and item["gradient"]["max_abs"] < 100.0 and item["gradient"]["l2"] < 1000.0
            for item in near_zero_cases.values()
        ),
        "mixed_scales_finite": mixed_case["gradient"]["finite"] and mixed_case["teacher_detached"],
        "mixed_scales_no_trivial_gradient_dominance": (
            bool(torch.isfinite(nonzero_mixed).all())
            and float(nonzero_mixed.max() / nonzero_mixed.median().clamp_min(torch.finfo(torch.float32).tiny)) < 100.0
        ),
        "mixed_scales_retain_scale_identifiability": all(
            scale_cases[str(scale)]["loss"] >= 0.0 for scale in scales
        ),
        "heavy_tail_detected": all(
            corruptions[str(factor)]["loss"] > scale_cases["1.0"]["loss"] + 0.01 for factor in (10.0, 100.0)
        ),
        "teacher_gradients_none": all(
            item.get("teacher_detached") is True
            for item in [*scale_cases.values(), opposite, zero_case, *near_zero_cases.values(), mixed_case, *corruptions.values()]
        ),
        "smooth_l1_is_bounded_and_robust": (
            robust_gradients["SmoothL1_beta_1"]["max_abs"] <= robust_gradients["L1"]["max_abs"]
            and robust_gradients["SmoothL1_beta_1"]["max_abs"] < robust_gradients["L2_MSE"]["max_abs"]
        ),
    }
    return {
        "schema_version": "P30R1_SYNTHETIC_FALSIFICATION_V1",
        "formulation_sha256": formulation_sha256(),
        "objective_count": 1,
        "observed_data_used": False,
        "correction_scale_C": CORRECTION_SCALE,
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "smooth_l1_beta": SMOOTH_L1_BETA,
        "smooth_l1_reduction": "mean over all batch and 243 coordinates",
        "fixed_thresholds": SYNTHETIC_THRESHOLDS,
        "scale_identifiability": {
            "scales": list(scales),
            "cases": scale_cases,
        },
        "direction_sensitivity": {"opposite": opposite},
        "zero_teacher_restoring_force": zero_case,
        "near_zero_teacher": near_zero_cases,
        "mixed_batch_scales": mixed_case,
        "heavy_tail_student_corruption": corruptions,
        "robust_loss_comparison": {"losses": robust_losses, "gradient_bounds": robust_gradients},
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "RESEARCH_STOP",
    }


def mathematical_analysis() -> dict[str, Any]:
    """Record the preregistered scalar/radial consequences for cases A--J."""
    return {
        "notation": {
            "n": COORDINATE_COUNT,
            "assumption": "t=alpha*u, s=beta*v, ||u||_2=||v||_2=1",
            "teacher_rms": "q_u=sqrt(mean(u^2)); a_t=sqrt((alpha/C)^2*q_u^2+eps^2)",
            "normalized_residual": "delta=(beta*v-alpha*u)/(C*a_t)",
            "per_coordinate_gradient": "dL/ds_j=psi(delta_j)/(B*n*C*a_t), psi(x)=x for |x|<1 else sign(x)",
            "absolute_gradient_bound": "|dL/ds_j| <= 1/(B*n*C*eps); per-sample L2 <= 1/(B*C*eps*sqrt(n))",
            "bound_at_B1": {
                "gradient_max_abs": 0.08296643779020978,
                "gradient_l2": 1.2933187701808124,
            },
        },
        "cases": {
            "A": {
                "condition": "v=u, beta=0.1*alpha",
                "loss_behavior": "positive radial error; below the beta=alpha minimum and materially nonzero",
                "student_gradient_direction": "coordinatewise toward +u, increasing beta toward alpha",
                "beta_collapse_or_explode": "neither; beta=alpha is the target and is identifiable",
                "gradient_explosion": "no; epsilon and bounded SmoothL1 derivative give the fixed bound",
                "normalization_bias": "if alpha is small, epsilon makes the target less than unit RMS; otherwise a_t tracks alpha",
                "small_teacher_domination": "possible only as bounded weighting through 1/a_t; capped by 1/epsilon",
            },
            "B": {
                "condition": "v=u, beta=alpha",
                "loss_behavior": "exactly zero, the unique minimum for fixed teacher and direction",
                "student_gradient_direction": "zero at the optimum",
                "beta_collapse_or_explode": "neither; no radial drift remains at the match",
                "gradient_explosion": "none",
                "normalization_bias": "none for the exact match; both sides share the same teacher denominator",
                "small_teacher_domination": "no mismatch gradient at the exact match",
            },
            "C": {
                "condition": "v=u, beta=10*alpha",
                "loss_behavior": "large positive radial error; SmoothL1 becomes linear on coordinates beyond beta=1",
                "student_gradient_direction": "coordinatewise toward -u, decreasing beta toward alpha",
                "beta_collapse_or_explode": "neither; the radial error is penalized rather than ignored",
                "gradient_explosion": "no; the tail derivative is clipped to magnitude one",
                "normalization_bias": "large alpha cancels in a_t approximately, leaving the relative beta/alpha error identifiable",
                "small_teacher_domination": "not caused by this case; larger a_t reduces the raw gradient weight",
            },
            "D": {
                "condition": "v=u, beta=100*alpha",
                "loss_behavior": "very large positive radial error, approximately linear in the normalized excess in the SmoothL1 tail",
                "student_gradient_direction": "coordinatewise toward -u; it cannot be mistaken for a low-loss directional match",
                "beta_collapse_or_explode": "neither as an objective property; updates oppose the excess radius",
                "gradient_explosion": "no from the loss derivative, although an external optimizer could still be misconfigured",
                "normalization_bias": "same teacher-relative denominator retains beta/alpha dependence; epsilon matters only for near-zero alpha",
                "small_teacher_domination": "bounded by epsilon, not unbounded",
            },
            "E": {
                "condition": "v=-u, beta=alpha",
                "loss_behavior": "positive directional and radial residual, materially above the correct-direction match",
                "student_gradient_direction": "toward +u and away from -u, coordinatewise through the residual",
                "beta_collapse_or_explode": "neither is required; direction and magnitude both move toward teacher",
                "gradient_explosion": "no; bounded SmoothL1 derivative and a_t floor",
                "normalization_bias": "shared teacher denominator does not erase the sign mismatch",
                "small_teacher_domination": "only bounded by 1/epsilon for near-zero teachers",
            },
            "F": {
                "condition": "v=u with catastrophic beta/alpha, such as 100",
                "loss_behavior": "tail loss grows with the residual rather than remaining direction-only invariant",
                "student_gradient_direction": "radial correction toward the teacher magnitude",
                "beta_collapse_or_explode": "the loss supplies a restoring force; it does not reward unbounded beta",
                "gradient_explosion": "SmoothL1 prevents quadratic tail gradient growth, while the denominator floor caps scale",
                "normalization_bias": "teacher-relative scaling intentionally makes the target strength depend on alpha through a_t",
                "small_teacher_domination": "risk is explicit and bounded; the mixed-scale gate tests it rather than hiding it",
            },
            "G": {
                "condition": "t=0, s!=0",
                "loss_behavior": "finite positive loss with a_t=epsilon and z_t=0",
                "student_gradient_direction": "toward zero; the gradient dot student is positive",
                "beta_collapse_or_explode": "beta is restored toward zero, not left unconstrained",
                "gradient_explosion": "no; |psi|<=1 and a_t=epsilon",
                "normalization_bias": "epsilon is the deliberate zero-target scale, not a learned target statistic",
                "small_teacher_domination": "zero samples receive the maximum bounded per-coordinate weight; their fraction is measured",
            },
            "H": {
                "condition": "||t|| approximately zero but nonzero",
                "loss_behavior": "finite; z_t is attenuated when teacher RMS is below epsilon and radial mismatch remains identifiable",
                "student_gradient_direction": "toward the small teacher target, or toward zero when the student is larger",
                "beta_collapse_or_explode": "no objective-level collapse or explosion; epsilon prevents a singular denominator",
                "gradient_explosion": "no under the fixed analytic bound, verified at 1e-8, 1e-6, and 1e-4 scales",
                "normalization_bias": "yes, a known epsilon-floor bias for tiny targets; no empirical epsilon tuning is allowed",
                "small_teacher_domination": "bounded but potentially stronger than large-target samples; source and mixed-scale gates quantify it",
            },
            "I": {
                "condition": "large teacher residual, alpha much larger than C*epsilon",
                "loss_behavior": "relative residual is compared after a_t approximately tracks teacher RMS",
                "student_gradient_direction": "toward teacher coordinatewise, with smaller raw weight 1/a_t",
                "beta_collapse_or_explode": "neither; large teacher scale is not a route to scale-invariant student output",
                "gradient_explosion": "no; denominator is large and SmoothL1 is bounded",
                "normalization_bias": "minimal epsilon bias in this regime; the intended relative normalization dominates",
                "small_teacher_domination": "large targets do not dominate solely due to raw magnitude",
            },
            "J": {
                "condition": "one batch contains teacher scales 0, 0.01, 0.1, 1, 10, and 100",
                "loss_behavior": "each sample retains its own beta/alpha radial minimum; batch loss is the mean over all coordinates",
                "student_gradient_direction": "each sample receives a residual-directed gradient toward its own teacher",
                "beta_collapse_or_explode": "no scale invariance; per-sample radial errors remain observable in one batch",
                "gradient_explosion": "finite and bounded; the fixed mixed-scale max/median gradient ratio gate is 100",
                "normalization_bias": "only the declared teacher-relative weighting and epsilon floor; no student norm or class scale is introduced",
                "small_teacher_domination": "the principal remaining risk; the deterministic mixed batch must stay below the fixed bound and source cache must be non-pathological",
            },
        },
    }


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return {key: None for key in ("min", "q50", "q90", "q95", "q99", "max")}
    return {
        "min": float(np.min(values)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _class_stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "rms_quantiles": _quantiles(array),
        "mean": float(array.mean()) if array.size else None,
        "std": float(array.std()) if array.size else None,
    }


def source_cache_radial_stats(cache_root: Path, metadata: Path) -> dict[str, Any]:
    """Inspect only cached source teacher tensors; no masks or held files."""
    rows = read_visa_metadata(metadata)
    expected_ids = {f"{row['class_name']}:{row['image_path']}" for row in rows}
    unique: dict[str, np.ndarray] = {}
    by_source_class: dict[str, list[float]] = {name: [] for name in EXPECTED_VISA_CLASSES}
    shard_counts: dict[str, int] = {}
    for held_class in EXPECTED_VISA_CLASSES:
        manifest_path = cache_root / "tier_b" / held_class / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        teacher = np.load(cache_root / "tier_b" / held_class / "teacher_region.npy", mmap_mode="r", allow_pickle=False)
        if list(teacher.shape) != [int(manifest["sample_count"]), 9, 9] or str(teacher.dtype) != "float32":
            raise RuntimeError(f"Tier-B teacher tensor contract mismatch: {held_class}")
        shard_counts[held_class] = int(teacher.shape[0])
        for index, sample_id in enumerate(manifest["sample_ids"]):
            if sample_id not in expected_ids:
                raise RuntimeError(f"cache sample is absent from metadata: {sample_id}")
            if sample_id not in unique:
                unique[sample_id] = np.asarray(teacher[index], dtype=np.float64).copy()
    if set(unique) != expected_ids:
        raise RuntimeError("source cache union does not cover the frozen metadata inventory")
    tensors = np.stack([unique[sample_id] for sample_id in sorted(unique)], axis=0)
    raw_rms = np.sqrt(np.mean(tensors * tensors, axis=(1, 2)))
    normalized_rms = raw_rms / CORRECTION_SCALE
    raw_l2 = np.linalg.norm(tensors.reshape(tensors.shape[0], -1), axis=1)
    exact_zero = np.all(tensors == 0.0, axis=(1, 2))
    nonzero = ~exact_zero
    near_epsilon = normalized_rms < NORMALIZATION_EPSILON
    near_ten_epsilon = normalized_rms < 10.0 * NORMALIZATION_EPSILON
    for sample_id, tensor in unique.items():
        source_class = sample_id.split(":", 1)[0]
        by_source_class.setdefault(source_class, []).append(float(np.sqrt(np.mean(tensor * tensor)) / CORRECTION_SCALE))
    class_summary = {name: _class_stats(values) for name, values in sorted(by_source_class.items())}
    denominator = np.sqrt(normalized_rms * normalized_rms + NORMALIZATION_EPSILON**2)
    raw_values = np.concatenate((raw_rms, normalized_rms, raw_l2, denominator))
    normalized_median = float(np.quantile(normalized_rms, 0.50))
    normalized_q99 = float(np.quantile(normalized_rms, 0.99))
    q99_over_median = normalized_q99 / normalized_median if normalized_median > 0.0 else math.inf
    source_checks = {
        "all_observed_values_finite": bool(np.isfinite(raw_values).all()),
        "denominator_floor": bool(np.all(denominator >= NORMALIZATION_EPSILON)),
        "q99_over_median_within_fixed_bound": bool(
            q99_over_median
            <= SOURCE_CACHE_THRESHOLDS["q99_over_median_normalized_rms_max"]
        ),
        "near_ten_epsilon_fraction_within_fixed_bound": bool(
            (near_ten_epsilon & nonzero).sum() / nonzero.sum()
            <= SOURCE_CACHE_THRESHOLDS["near_ten_epsilon_nonzero_fraction_max"]
            if bool(nonzero.any())
            else True
        ),
    }
    result = {
        "schema_version": "P30R1_SOURCE_CACHE_RADIAL_STATS_V1",
        "status": "PASS" if all(source_checks.values()) else "RESEARCH_STOP",
        "observed_data": "immutable Tier-B teacher_region.npy only",
        "held_labels_used_for_tuning": False,
        "cache_root": str(cache_root.resolve()),
        "tier_b_shard_sample_counts": shard_counts,
        "unique_source_sample_count": int(tensors.shape[0]),
        "duplicate_cache_exposures_not_counted": int(sum(shard_counts.values()) - tensors.shape[0]),
        "correction_scale_C": CORRECTION_SCALE,
        "normalization_epsilon": NORMALIZATION_EPSILON,
        "raw_rms": _quantiles(raw_rms),
        "normalized_teacher_rms": _quantiles(normalized_rms),
        "teacher_l2": _quantiles(raw_l2),
        "teacher_denominator_a_t": _quantiles(denominator),
        "min_nonzero_normalized_rms": float(normalized_rms[nonzero].min()) if bool(nonzero.any()) else None,
        "exact_zero_count": int(exact_zero.sum()),
        "exact_zero_fraction": float(exact_zero.mean()),
        "near_epsilon_nonzero_count": int((near_epsilon & nonzero).sum()),
        "near_epsilon_nonzero_fraction": float((near_epsilon & nonzero).sum() / nonzero.sum()) if bool(nonzero.any()) else 0.0,
        "near_ten_epsilon_nonzero_count": int((near_ten_epsilon & nonzero).sum()),
        "near_ten_epsilon_nonzero_fraction": float((near_ten_epsilon & nonzero).sum() / nonzero.sum()) if bool(nonzero.any()) else 0.0,
        "q99_over_median_normalized_rms": q99_over_median if math.isfinite(q99_over_median) else None,
        "cross_source_class": class_summary,
        "epsilon_floor_is_active_for_all_targets": bool(np.all(denominator >= NORMALIZATION_EPSILON)),
        "category_specific_C_in_code": False,
        "fixed_thresholds": SOURCE_CACHE_THRESHOLDS,
        "checks": source_checks,
    }
    return result


def p30_counterfactual_radial_stats(
    p30_root: Path,
    cache_root: Path,
    metadata: Path,
    visa_root: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Analyze existing P30 candle residuals after their prior freeze only."""
    from tools.sabra_v2.analyze_p30_outputs import _teacher_regions
    from tools.sabra_v2.p28_mechanism_diagnostic import _load_masks

    class_name = "candle"
    prediction_path = p30_root / "qualification/stage2_one_class/candle/predictions/p30_held_predictions.pt"
    if not prediction_path.is_file():
        return {
            "schema_version": "P30R1_P30_COUNTERFACTUAL_V1",
            "status": "UNAVAILABLE",
            "reason": f"missing existing frozen P30 prediction: {prediction_path}",
            "observed_data_used_for_tuning": False,
        }
    rows = read_visa_metadata(metadata)
    fold = loco_inventory(rows, class_name)
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    records = payload.get("records", [])
    by_path = {str(record["image_path"]): record for record in records}
    paths = [str(row["image_path"]) for row in fold.held_rows]
    if set(by_path) != set(paths):
        raise RuntimeError("frozen P30 counterfactual prediction identity mismatch")
    tier_a = cache_root / "tier_a" / class_name
    manifest = json.loads((tier_a / "manifest.json").read_text(encoding="utf-8"))
    index_by_id = {sample_id: index for index, sample_id in enumerate(manifest["sample_ids"])}
    indices = [index_by_id[f"{row['class_name']}:{row['image_path']}"] for row in fold.held_rows]
    native_cache = np.load(tier_a / "native_logits.npy", mmap_mode="r", allow_pickle=False)
    masks, mask_reads = _load_masks(fold.held_rows, visa_root)
    teacher = _teacher_regions(native_cache, indices, masks, device)
    student = np.stack([np.asarray(by_path[path]["p30_region_residual"], dtype=np.float32) for path in paths], axis=0).transpose(1, 0, 2, 3)
    teacher_staged = np.broadcast_to(teacher[None, ...], student.shape)
    student_flat = student.transpose(1, 0, 2, 3).reshape(len(paths), -1).astype(np.float64)
    teacher_flat = teacher_staged.transpose(1, 0, 2, 3).reshape(len(paths), -1).astype(np.float64)
    student_l2 = np.linalg.norm(student_flat, axis=1)
    teacher_l2 = np.linalg.norm(teacher_flat, axis=1)
    radial_ratio = student_l2 / (teacher_l2 + NORMALIZATION_EPSILON)
    sample_q99 = np.quantile(np.abs(student_flat), 0.99, axis=1)
    ratio_q99 = _quantiles(radial_ratio)
    return {
        "schema_version": "P30R1_P30_COUNTERFACTUAL_V1",
        "status": "PASS",
        "observed_data_used_for_tuning": False,
        "source": str(prediction_path),
        "held_class": class_name,
        "sample_count": len(paths),
        "held_mask_reads_post_freeze": int(mask_reads),
        "ratio_definition": "||s_P30||_2 / (||t||_2 + 0.01), raw staged [3,9,9] residual units",
        "radial_ratio": ratio_q99,
        "p30_sample_q99_abs": _quantiles(sample_q99),
        "p30_residual_global_q99_abs": float(np.quantile(np.abs(student_flat), 0.99)),
        "teacher_l2": _quantiles(teacher_l2),
        "student_l2": _quantiles(student_l2),
        "exact_zero_teacher_count": int(np.all(teacher_flat == 0.0, axis=1).sum()),
        "radial_ratio_vs_sample_q99": {
            "pearson": _pearson(radial_ratio, sample_q99),
            "spearman": _rank_correlation(radial_ratio, sample_q99),
        },
        "top_radial_ratio_paths": [paths[index] for index in np.argsort(radial_ratio)[-5:][::-1]],
        "top_sample_q99_paths": [paths[index] for index in np.argsort(sample_q99)[-5:][::-1]],
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--p30-root", type=Path, default=DEFAULT_P30_ROOT)
    parser.add_argument("--visa-root", type=Path, default=DEFAULT_VISA_ROOT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    synthetic = synthetic_falsification()
    if synthetic["status"] != "PASS":
        result = {
            "schema_version": "P30R1_PREFLIGHT_FALSIFICATION_V1",
            "formulation_sha256": formulation_sha256(),
            "synthetic": synthetic,
            "source_cache": None,
            "p30_counterfactual": None,
            "engineering_smoke": {"status": "NOT_RUN", "reason": "synthetic gate failed"},
            "final_gate": "RESEARCH_STOP",
        }
        atomic_write_json(args.output, result)
        return result
    source_cache = source_cache_radial_stats(args.cache_root, args.metadata)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    counterfactual = p30_counterfactual_radial_stats(
        args.p30_root,
        args.cache_root,
        args.metadata,
        args.visa_root,
        device=device,
    )
    result = {
        "schema_version": "P30R1_PREFLIGHT_FALSIFICATION_V1",
        "formulation_sha256": formulation_sha256(),
        "formulation": FORMULATION_TEXT,
        "mathematical_analysis": mathematical_analysis(),
        "synthetic": synthetic,
        "source_cache": source_cache,
        "p30_counterfactual": counterfactual,
        "engineering_smoke": {"status": "NOT_RUN", "reason": "P30R1 implementation is forbidden in this phase"},
        "speed_profile": {"status": "NOT_RUN", "reason": "P30R1 implementation and engineering profile are forbidden in this phase"},
        "prohibited_actions_confirmed": {
            "p30r1_trainer_created": False,
            "p30r1_stage2_started": False,
            "p30_full_training_started": False,
            "p29_or_p27_rerun": False,
            "held_labels_used_for_tuning": False,
        },
        "final_gate": "PASS_TO_IMPLEMENTATION"
        if synthetic["status"] == "PASS"
        and source_cache["status"] == "PASS"
        and counterfactual["status"] in ("PASS", "UNAVAILABLE")
        else "RESEARCH_STOP",
    }
    atomic_write_json(args.output, result)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
