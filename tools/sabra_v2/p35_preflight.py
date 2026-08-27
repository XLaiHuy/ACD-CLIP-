"""Deterministic non-held P35 candidate and objective preflight.

The preflight is effect-space algebra only.  It deliberately keeps the full
teacher effect as the target and varies only the detached source-example
importance map.  No dataset labels, held masks, neural-model forwards, or
cache writes are used.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tools.sabra_v2.p29_contract import CORRECTION_SCALE
from tools.sabra_v2.forensics.p35_soft_actionability import analyze_source, actionability_map


MAP_NAMES = ("clamp", "tanh", "rational")
SMOOTH_L1_BETA = 1.0


def _gradient_summary(gradient: torch.Tensor) -> dict[str, Any]:
    value = gradient.detach().float().reshape(-1)
    return {
        "l2": float(torch.linalg.vector_norm(value)),
        "max_abs": float(value.abs().max()),
        "q99_abs": float(torch.quantile(value.abs(), torch.tensor(0.99))),
        "finite": bool(torch.isfinite(value).all()),
        "nonzero_fraction": float((value != 0).float().mean()),
    }


def _case(name: str, teacher: torch.Tensor, student: torch.Tensor, x: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {"teacher_shape": list(teacher.shape), "student_shape": list(student.shape), "candidates": {}}
    for map_name in MAP_NAMES:
        weight = torch.as_tensor(actionability_map(x.detach().cpu().numpy(), map_name), dtype=torch.float32)
        student_value = student.detach().clone().requires_grad_(True)
        target = teacher.detach().clone()
        loss = (weight * F.smooth_l1_loss(student_value, target, beta=SMOOTH_L1_BETA, reduction="none")).mean()
        gradient = torch.autograd.grad(loss, student_value)[0]
        result["candidates"][map_name] = {
            "weight_min": float(weight.min()),
            "weight_max": float(weight.max()),
            "weight_mean": float(weight.mean()),
            "loss": float(loss),
            "gradient": _gradient_summary(gradient),
            "target_equals_full_teacher": bool(torch.equal(target, teacher.detach())),
            "finite": bool(torch.isfinite(loss) and torch.isfinite(gradient).all()),
            "target_was_not_weighted": True,
        }
    return {"name": name, **result}


def _synthetic_suite() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    # Decoupled zero actionability isolates importance semantics.  P35 is
    # intentionally allowed to provide no restoring gradient at w=0.
    teacher = torch.full((1, 8), 2.0, dtype=torch.float32)
    student = torch.ones_like(teacher)
    cases.append(_case("zero_weight_decoupled", teacher, student, torch.zeros_like(teacher)))

    for x_value in (0.01, 0.1, 0.5, 1.0, 2.0, 10.0):
        x = torch.full((1, 8), x_value, dtype=torch.float32)
        teacher = x * CORRECTION_SCALE
        student = torch.full_like(teacher, -0.5)
        cases.append(_case(f"x_{x_value:g}", teacher, student, x))

    x = torch.tensor([[0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]], dtype=torch.float32)
    teacher = torch.tensor([[0.0, 0.01, -0.2, 1.0, -2.0, 4.0, -20.0, 100.0]], dtype=torch.float32)
    student = -teacher
    cases.append(_case("mixed_effect_sign_reversal", teacher, student, x))

    x = torch.cat((torch.zeros(90, 1), torch.ones(10, 1)), dim=0)
    teacher = torch.cat((torch.zeros(90, 1), torch.full((10, 1), CORRECTION_SCALE)), dim=0)
    student = torch.full_like(teacher, 0.25)
    cases.append(_case("mixed_90_low_10_high", teacher, student, x))

    x = torch.tensor([[0.0, 0.01, 0.1, 0.5, 1.0, 10.0]], dtype=torch.float32)
    teacher = torch.tensor([[0.0, 0.01, 0.1, 0.5, 1.0, 1.0e6]], dtype=torch.float32)
    student = torch.zeros_like(teacher)
    cases.append(_case("heavy_tail_one_outlier", teacher, student, x))

    x = torch.zeros((2, 4), dtype=torch.float32)
    teacher = torch.zeros_like(x)
    student = torch.full_like(x, 0.5)
    cases.append(_case("all_zero_actionability", teacher, student, x))

    x = torch.ones((2, 4), dtype=torch.float32)
    teacher = torch.full_like(x, CORRECTION_SCALE)
    student = torch.zeros_like(x)
    cases.append(_case("all_high_actionability", teacher, student, x))

    return {"cases": cases, "all_finite": all(item["candidates"][name]["finite"] for item in cases for name in MAP_NAMES)}


def _radial_identifiability() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(35001)
    direction = torch.randn((1, 4, 4), generator=generator, dtype=torch.float32)
    teacher = torch.randn((1, 4, 4), generator=generator, dtype=torch.float32)
    x = teacher.abs() / CORRECTION_SCALE
    outputs: dict[str, Any] = {}
    for name in MAP_NAMES:
        weight = torch.as_tensor(actionability_map(x.numpy(), name), dtype=torch.float32)
        values = []
        gradients = []
        for beta in (0.0, 0.25, 0.5, 1.0, 2.0):
            student = (beta * direction).requires_grad_(True)
            loss = (weight * F.smooth_l1_loss(student, teacher, beta=1.0, reduction="none")).mean()
            gradient = torch.autograd.grad(loss, student)[0]
            values.append(float(loss))
            gradients.append(float(torch.linalg.vector_norm(gradient)))
        outputs[name] = {
            "loss_by_beta": values,
            "gradient_l2_by_beta": gradients,
            "loss_range": max(values) - min(values),
            "identifiable": max(values) - min(values) > 1e-6,
        }
    return outputs


def run_preflight() -> dict[str, Any]:
    source = analyze_source()
    synthetic = _synthetic_suite()
    radial = _radial_identifiability()
    target_preserved = all(
        case["candidates"][name]["target_equals_full_teacher"]
        for case in synthetic["cases"]
        for name in MAP_NAMES
    )
    selected = source["candidates"]["tanh"]
    source_selected = selected["weight"]
    source_gate = (
        source_selected["full_n"] > 0
        and source_selected["full_mean"] > 0.0
        and source_selected["full_mean"] < 1.0
        and source_selected["exact_one_fraction"] == 0.0
        and source_selected["finite"]
        and source_selected["full_counts"]["w_eq_0"] > 0
    )
    return {
        "schema_version": "P35_PREFLIGHT_FALSIFICATION_V1",
        "protocol_id": "P35",
        "status": "P35_PREFLIGHT_PASS" if synthetic["all_finite"] and target_preserved and source_gate and all(item["identifiable"] for item in radial.values()) else "P35_PREFLIGHT_FAIL",
        "selected_candidate": "SOFT_TANH_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER",
        "equations": {
            "normalized_effect": "x=abs(E_t)/C",
            "selected_weight": "stop_gradient(tanh(x))",
            "target": "stop_gradient(E_t) (full target; never multiplied by weight)",
            "objective": "mean(weight * SmoothL1(E_s, stop_gradient(E_t), beta=1.0, reduction=none))",
        },
        "synthetic": synthetic,
        "radial_identifiability": radial,
        "source_only": source,
        "gates": {
            "all_synthetic_finite": synthetic["all_finite"],
            "full_target_preserved": target_preserved,
            "source_weight_is_finite_bounded_and_nontrivial": source_gate,
            "selected_map_has_no_source_hard_one_saturation": source_selected["exact_one_fraction"] == 0.0,
            "radial_identifiable_for_all_candidates": all(item["identifiable"] for item in radial.values()),
            "one_objective": True,
            "new_tuned_hyperparameters": 0,
            "held_reads": 0,
            "new_clip_forwards": 0,
            "new_phase2b_forwards": 0,
            "new_teacher_forwards": 0,
            "cache_rebuilds": 0,
        },
        "interpretation": {
            "zero_actionability_semantics": "importance zero means no direct P35 gradient; this is intentional and differs from P34 zero-target shaping",
            "actionability_signal": "source-only absolute deployed teacher effect",
            "target_shrinkage": False,
            "category_specific_rule": False,
        },
    }


def main() -> None:
    result = run_preflight()
    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)


if __name__ == "__main__":
    main()
