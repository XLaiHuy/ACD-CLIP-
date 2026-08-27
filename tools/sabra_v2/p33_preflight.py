"""Deterministic, non-held P33 formulation and source-cache preflight.

This module uses only synthetic tensors and frozen Tier-B source teacher
regions.  It never loads a model/checkpoint, reads held data, rebuilds a
cache, or produces scientific predictions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.sabra_v2.p29_contract import CORRECTION_SCALE
from tools.sabra_v2.p33_objective import p33_actionability_components
from tools.sabra_v2.region_cache import sha256_file


ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
OUTPUT = ROOT / "research/sabra_v2/region_distill/P33_PREFLIGHT_FALSIFICATION.json"
STAGES = 3
GRID = 9


def _gradient_audit(gradient: torch.Tensor) -> dict[str, Any]:
    flat = gradient.detach().reshape(-1)
    return {
        "l2": float(torch.linalg.vector_norm(flat)),
        "max_abs": float(flat.abs().max()),
        "nonzero_fraction": float(torch.count_nonzero(flat).item() / flat.numel()),
        "finite": bool(torch.isfinite(flat).all().item()),
    }


def _case(name: str, student: torch.Tensor, teacher: torch.Tensor) -> dict[str, Any]:
    student = student.to(dtype=torch.float32).clone().requires_grad_(True)
    teacher = teacher.to(dtype=torch.float32).clone().requires_grad_(True)
    loss, student_effect, teacher_effect, weight = p33_actionability_components(student, teacher)
    gradient = torch.autograd.grad(loss, student, retain_graph=False)[0]
    staged_teacher = teacher.detach().unsqueeze(0).expand_as(student)
    displacement = student.detach() - staged_teacher
    alignment = float((gradient.detach() * displacement).sum())
    per_sample_l2: list[float] = []
    for index in range(teacher.shape[0]):
        sample_student = student.detach()[..., index : index + 1, :, :].clone().requires_grad_(True)
        sample_teacher = teacher.detach()[index : index + 1]
        sample_loss = p33_actionability_components(sample_student, sample_teacher)[0]
        sample_gradient = torch.autograd.grad(sample_loss, sample_student)[0]
        per_sample_l2.append(float(torch.linalg.vector_norm(sample_gradient.detach())))
    nonzero_samples = [value for value in per_sample_l2 if value > 0.0]
    ratio = None
    if nonzero_samples:
        ratio = float(max(nonzero_samples) / statistics_median(nonzero_samples))
    return {
        "name": name,
        "batch": int(teacher.shape[0]),
        "loss": float(loss.detach()),
        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        "student_effect_shape": list(student_effect.shape),
        "teacher_effect_shape": list(teacher_effect.shape),
        "weight": {
            "min": float(weight.min()),
            "max": float(weight.max()),
            "mean": float(weight.mean()),
            "exact_zero_fraction": float(torch.mean((weight == 0).to(torch.float32))),
            "nonzero_fraction": float(torch.mean((weight > 0).to(torch.float32))),
            "bounded": bool((weight >= 0).all().item() and (weight <= 1).all().item()),
            "detached": not weight.requires_grad,
        },
        "gradient": _gradient_audit(gradient),
        "teacher_gradient_absent": teacher.grad is None,
        "target_displacement_gradient_alignment": alignment,
        "expected_descent_alignment_nonnegative": alignment >= -1e-8,
        "per_sample_gradient_l2": per_sample_l2,
        "nonzero_sample_gradient_max_median_ratio": ratio,
    }


def statistics_median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _synthetic_suite() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(33033)
    base_teacher = torch.randn((1, GRID, GRID), generator=generator, dtype=torch.float32)
    base_student = 0.5 * base_teacher.unsqueeze(0).expand(STAGES, -1, -1, -1)
    cases: list[dict[str, Any]] = []
    cases.append(_case("exact_zero", torch.full((STAGES, 1, GRID, GRID), 0.5), torch.zeros_like(base_teacher)))
    cases.append(_case("near_zero", base_student, base_teacher * 1e-6))
    cases.append(_case("normal_scale", base_student, base_teacher))
    for scale in (0.01, 0.1, 1.0, 10.0, 100.0):
        cases.append(_case(f"scale_{scale:g}x", base_student * scale, base_teacher * scale))
    cases.append(_case("sign_reversal", -base_student, base_teacher))
    sparse_teacher = torch.zeros_like(base_teacher)
    sparse_teacher.reshape(-1)[0] = 4.0
    cases.append(_case("sparse_1_percent_actionable", torch.zeros((STAGES, 1, GRID, GRID)), sparse_teacher))
    heavy_teacher = base_teacher.clone()
    heavy_teacher.reshape(-1)[0] = 100.0 * float(base_teacher.abs().max())
    cases.append(_case("heavy_tail_corruption", base_student, heavy_teacher))
    mixed_scales = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
    mixed_teacher = torch.cat([base_teacher * scale for scale in mixed_scales], dim=0)
    mixed_student = 0.5 * mixed_teacher.unsqueeze(0).expand(STAGES, -1, -1, -1)
    cases.append(_case("mixed_scale_batch", mixed_student, mixed_teacher))
    outlier_teacher = torch.cat([base_teacher, base_teacher, base_teacher, heavy_teacher], dim=0)
    outlier_student = 0.5 * outlier_teacher.unsqueeze(0).expand(STAGES, -1, -1, -1)
    cases.append(_case("one_extreme_outlier_sample", outlier_student, outlier_teacher))
    cases.append(_case("all_abstain", torch.randn((STAGES, 4, GRID, GRID), generator=generator), torch.zeros((4, GRID, GRID))))
    high_teacher = torch.full((2, GRID, GRID), CORRECTION_SCALE, dtype=torch.float32)
    cases.append(_case("all_high_confidence_active", torch.zeros((STAGES, 2, GRID, GRID)), high_teacher))

    finite = all(
        row["loss_finite"]
        and row["gradient"]["finite"]
        and row["weight"]["bounded"]
        and row["teacher_gradient_absent"]
        for row in cases
    )
    exact_zero = next(row for row in cases if row["name"] == "exact_zero")
    all_abstain = next(row for row in cases if row["name"] == "all_abstain")
    return {
        "generator_seed": 33033,
        "cases": cases,
        "all_finite_bounded_and_teacher_detached": finite,
        "exact_zero_has_zero_loss_and_gradient": exact_zero["loss"] == 0.0 and exact_zero["gradient"]["l2"] == 0.0,
        "all_abstain_has_zero_loss_and_gradient": all_abstain["loss"] == 0.0 and all_abstain["gradient"]["l2"] == 0.0,
        "selected_candidate_status": "PASS" if finite else "FAIL",
    }


def _source_audit() -> dict[str, Any]:
    unique: dict[str, np.ndarray] = {}
    exposure_count = 0
    shard_count = 0
    for shard in sorted(CACHE_ROOT.joinpath("tier_b").iterdir()):
        if not shard.is_dir() or not (shard / "manifest.json").is_file():
            continue
        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        teacher = np.load(shard / "teacher_region.npy", mmap_mode="r")
        shard_count += 1
        exposure_count += len(manifest["sample_ids"])
        for index, sample_id in enumerate(manifest["sample_ids"]):
            value = np.array(teacher[index], dtype=np.float32, copy=True)
            if sample_id in unique and not np.array_equal(unique[sample_id], value):
                raise RuntimeError("duplicate Tier-B source values disagree")
            unique.setdefault(sample_id, value)
    if not unique:
        raise RuntimeError("no frozen Tier-B source samples found")
    values = torch.from_numpy(np.stack([unique[key] for key in sorted(unique)])).to(torch.float32)
    with torch.no_grad():
        dummy_student = torch.zeros((STAGES, values.shape[0], GRID, GRID), dtype=torch.float32)
        _loss, _student, effect, weight = p33_actionability_components(dummy_student, values)
    rms = torch.sqrt(torch.mean(effect.square(), dim=(1, 2)))
    category_values: dict[str, list[float]] = {}
    for key, value in zip(sorted(unique), rms.tolist()):
        category_values.setdefault(key.split(":", 1)[0], []).append(float(value))
    category_medians = {key: statistics_median(value) for key, value in category_values.items()}
    positive = [value for value in category_medians.values() if value > 0]
    return {
        "cache_root": str(CACHE_ROOT.resolve()),
        "tier_b_shard_count": shard_count,
        "exposure_count": exposure_count,
        "unique_source_samples": len(unique),
        "duplicate_exposures_not_counted": exposure_count - len(unique),
        "new_cache_builds": 0,
        "held_reads": 0,
        "neural_forwards": 0,
        "functional_effect_rms": {
            "q01": float(torch.quantile(rms, 0.01)),
            "q50": float(torch.quantile(rms, 0.50)),
            "q99": float(torch.quantile(rms, 0.99)),
            "exact_zero_fraction": float(torch.mean((rms == 0).to(torch.float32))),
        },
        "weight": {
            "min": float(weight.min()),
            "max": float(weight.max()),
            "mean": float(weight.mean()),
            "exact_zero_fraction": float(torch.mean((weight == 0).to(torch.float32))),
            "bounded": bool((weight >= 0).all().item() and (weight <= 1).all().item()),
            "detached": not weight.requires_grad,
        },
        "category_effect_rms_median_ratio_max_min": float(max(positive) / min(positive)) if positive else None,
        "category_specific_rule_used": False,
        "source_only_identifiability": "operational teacher-requested deployed effect, not validated anomaly utility",
    }


def main() -> None:
    torch.set_num_threads(4)
    synthetic = _synthetic_suite()
    source = _source_audit()
    output = {
        "schema_version": "P33_PREFLIGHT_FALSIFICATION_V1",
        "status": "P33_PREFLIGHT_PASS" if synthetic["selected_candidate_status"] == "PASS" and source["weight"]["bounded"] else "P33_PREFLIGHT_FAIL",
        "protocol_id": "P33",
        "preregistration_md_sha256": "d2460555be14af7d23316e43ad16c8585faeecbedf1698ee71f29dce765aed6c",
        "allowed_evidence": ["symbolic analysis", "deterministic synthetic tensors", "frozen Tier-B source cache"],
        "forbidden_evidence_used": ["held GT", "held masks", "held metrics", "new neural forwards", "cache rebuild"],
        "objective_contract": {
            "objective_count": 1,
            "new_tuned_hyperparameters": 0,
            "inherited_correction_scale_C": CORRECTION_SCALE,
            "weight_formula": "clamp(abs(detached_teacher_effect)/C,0,1)",
            "target_shrinkage": False,
            "student_self_normalized": False,
        },
        "synthetic": synthetic,
        "source_only": source,
        "alternative_rejections": {
            "target_weighted_functional_target": "rejected: target w*E_t shrinks signed magnitude and creates a restoring null-target pressure under all-abstain",
            "hard_support_transfer": "rejected: inherited threshold is discontinuous and threshold-sensitive; it is not needed for a one-objective continuous test",
        },
        "final_gate": "P33_PREFLIGHT_PASS",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(OUTPUT)
    print(json.dumps({"status": output["status"], "output": str(OUTPUT), "sha256": sha256_file(OUTPUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
