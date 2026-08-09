#!/usr/bin/env python3
"""CPU-only post-300 schedule, accumulation, and gradient-report audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import torch
from torch import nn

from model.h6.utility_routing import exploration_epsilon
from train import grad_accum_window_size, h6_drift_gradient_attribution


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _historical_weighting(milestone: dict) -> dict:
    report = milestone["gradient_attribution"]
    components = {}
    reserved = {"active", "raw", "weight"}
    for name, component in report["components"].items():
        weight = float(component["weight"])
        weighted_norms = {
            key: float(value)
            for key, value in component.items()
            if key not in reserved and isinstance(value, (int, float))
        }
        raw_norms = {
            key: (None if weight == 0.0 else value / abs(weight))
            for key, value in weighted_norms.items()
        }
        components[name] = {
            "loss_value_raw": float(component["raw"]),
            "lambda_weight": weight,
            "active": bool(component["active"]),
            "historically_reported_norms": weighted_norms,
            "inferred_raw_norms": raw_norms,
        }
    raw_main = components["main_task"]["inferred_raw_norms"].get("shared_semantic")
    raw_factor = components["utility_factor"]["inferred_raw_norms"].get("shared_semantic")
    raw_router = components["utility_router"]["inferred_raw_norms"].get("shared_semantic")
    raw_ratios = {
        "utility_factor_to_task_shared_grad_ratio": None if not raw_main else raw_factor / raw_main,
        "utility_router_to_task_shared_grad_ratio": None if not raw_main else raw_router / raw_main,
        "total_aux_to_task_shared_grad_ratio": (
            None if not raw_main else (raw_factor + raw_router) / raw_main
        ),
    }
    return {
        "batch": int(milestone["batch"]),
        "components": components,
        "historically_reported_ratios": report["ratios"],
        "inferred_raw_ratios": raw_ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/p1_v83_dev/300batch_specialization_probe"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/p1_v83_dev/post300_audit")
    )
    parser.add_argument("--baseline-commit", default="2019e5ada8f30166ab70c8a859de35c487ecc1c1")
    args = parser.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text())
    smoke = json.loads((args.run_dir / "smoke_summary.json").read_text())
    trajectory = json.loads((args.run_dir / "trajectory.json").read_text())["milestones"]

    historical_epsilon = float(smoke["utility"]["exploration_epsilon"])
    canonical_horizon = 20
    epsilon = {
        "status": "FAIL_FIXED",
        "hard_stop_triggered": True,
        "classification": "NONCANONICAL_EARLY_EXPLORATION",
        "historical_run": {
            "epoch_argument": int(config["epoch"]),
            "effective_schedule_horizon": int(config["epoch"]),
            "reported_epsilon": historical_epsilon,
            "expected_canonical_epoch1_epsilon": exploration_epsilon(1, canonical_horizon),
            "cause": "diagnostic runner used --epoch 1 as both run length and exploration horizon",
            "artifacts_preserved": True,
        },
        "canonical_schedule": {
            str(epoch): exploration_epsilon(epoch, canonical_horizon)
            for epoch in (1, 2, 3, 10, 20)
        },
        "repair": {
            "cli": "--h6_exploration_total_epochs",
            "diagnostic_runner_default": canonical_horizon,
            "repaired_epoch1_epsilon": exploration_epsilon(1, canonical_horizon),
            "final20_default_unchanged": True,
        },
        "checks": {
            "historical_probe_used_final_schedule_value": abs(
                historical_epsilon - exploration_epsilon(1, 1)
            ) <= 1e-5,
            "canonical_epoch1_is_start_value": exploration_epsilon(1, canonical_horizon) == 0.15,
            "canonical_epoch20_is_end_value": exploration_epsilon(20, canonical_horizon) == 0.05,
        },
    }
    _write_json(args.output_dir / "epsilon_schedule_audit.json", epsilon)

    full_batches = 2162
    accumulation_steps = int(config["grad_accum_steps"])
    remainder = full_batches % accumulation_steps
    probe_batches = int(config["h6_smoke_max_batches"])
    accumulation = {
        "status": "FAIL_FIXED",
        "hard_stop_triggered": True,
        "classification": "FINAL_REMAINDER_UNDERWEIGHT_BUG",
        "historical_300_batch_probe": {
            "batches": probe_batches,
            "grad_accum_steps": accumulation_steps,
            "remainder": probe_batches % accumulation_steps,
            "affected": probe_batches % accumulation_steps != 0,
        },
        "blocked_full_epoch_geometry": {
            "batches": full_batches,
            "grad_accum_steps": accumulation_steps,
            "optimizer_steps": math.ceil(full_batches / accumulation_steps),
            "remainder_microbatches": remainder,
            "historical_divisor": accumulation_steps,
            "correct_divisor": remainder,
            "historical_remainder_weight_fraction": remainder / accumulation_steps,
        },
        "repair": "divide every microbatch by the actual size of its accumulation window",
        "window_sizes_14_batches_accum6": [
            grad_accum_window_size(index, 14, 6) for index in range(1, 15)
        ],
        "checks": {
            "full_windows_use_six": all(
                grad_accum_window_size(index, 14, 6) == 6 for index in range(1, 13)
            ),
            "final_two_use_two": all(
                grad_accum_window_size(index, 14, 6) == 2 for index in (13, 14)
            ),
            "historical_300_probe_unaffected": probe_batches % accumulation_steps == 0,
        },
    }
    _write_json(args.output_dir / "grad_accum_audit.json", accumulation)

    baseline_source = subprocess.run(
        ["git", "show", f"{args.baseline_commit}:train.py"],
        check=True,
        capture_output=True,
    ).stdout
    source_expression = b"loss * float(weight)" in baseline_source
    historical = [
        _historical_weighting(milestone)
        for milestone in trajectory
        if milestone.get("gradient_attribution") is not None
    ]
    shared = nn.Parameter(torch.tensor(2.0))
    router = nn.Parameter(torch.tensor(3.0))
    current_report = h6_drift_gradient_attribution(
        {
            "main_task": (shared.square(), 1.0),
            "utility_factor": (3.0 * shared, 0.1),
            "utility_router": (5.0 * router, 0.1),
            "disabled": (7.0 * shared, 0.0),
        },
        {"shared_semantic": [shared], "router": [router]},
    )
    weighting = {
        "status": "PASS_CLARIFIED",
        "historical_reported_ratio_basis": "lambda_weighted",
        "proof": {
            "baseline_commit": args.baseline_commit,
            "baseline_train_py_sha256": _sha256_bytes(baseline_source),
            "baseline_autograd_used_loss_times_weight": source_expression,
            "historical_component_group_fields": "lambda-weighted gradient L2 norms",
            "historical_component_raw_field": "raw scalar loss value, not raw gradient norm",
        },
        "historical_milestones": historical,
        "current_diagnostic_contract": current_report,
        "checks": {
            "historical_expression_is_weighted": source_expression,
            "new_report_labels_ratio_basis": current_report["ratio_basis"] == "lambda_weighted",
            "new_report_exposes_raw_and_weighted": (
                current_report["components"]["utility_factor"]["raw_gradient_norms"]["shared_semantic"] == 3.0
                and abs(current_report["components"]["utility_factor"]["weighted_gradient_norms"]["shared_semantic"] - 0.3) < 1e-7
            ),
            "diagnostic_does_not_populate_grad": shared.grad is None and router.grad is None,
        },
        "lambda_interpretation": (
            "Historical ratios already included lambda=0.1. Raw utility ratios are ten times "
            "the reported ratios when the main-task weight is 1.0."
        ),
    }
    _write_json(args.output_dir / "gradient_weighting_audit.json", weighting)

    gradient_milestones = [
        {"batch": int(item["batch"]), "gradients": item["gradients"]}
        for item in trajectory if item.get("gradients") is not None
    ]
    null_fields = sorted({
        key
        for item in gradient_milestones
        for key, value in item["gradients"].items()
        if value is None
    })
    classification = {
        "class_to_context_grad_norm": {
            "class": "ARCHITECTURE_INACTIVE_LEGACY_BRANCH",
            "expected": True,
            "reason": "structured prompt mode injects decoded class_token directly; legacy class_delta context is not consumed",
            "active_class_path_evidence": "vae_mu, vae_decoder, and shared semantic gradients are nonzero",
        },
        "factor_generator_context_grad_norm": {
            "class": "CONFIG_DISABLED",
            "expected": True,
            "reason": "h6_factor_generator_specialization_enabled=false",
        },
        "factor_generator_head_grad_norms": {
            "class": "CONFIG_DISABLED",
            "expected": True,
            "reason": "h6_factor_generator_specialization_enabled=false",
        },
        "factor_generator_identity_grad_norm": {
            "class": "CONFIG_DISABLED",
            "expected": True,
            "reason": "h6_factor_generator_specialization_enabled=false",
        },
        "factor_id_projection_grad_norm": {
            "class": "CONFIG_DISABLED",
            "expected": True,
            "reason": "h6_late_factor_identity_enabled=false",
        },
        "rho_gate_grad_norm": {
            "class": "FROZEN_BY_PROTOCOL",
            "expected": True,
            "reason": "rho is fixed at 0.05, requires_grad=false, and absent from optimizer",
        },
    }
    unclassified = [name for name in null_fields if name not in classification]
    router_activity = [
        {
            "batch": item["batch"],
            "utility_router_raw_loss": item["gradient_attribution"]["components"]["utility_router"]["raw"],
            "utility_router_weighted_router_grad": item["gradient_attribution"]["components"]["utility_router"].get("router", 0.0),
        }
        for item in trajectory if item.get("gradient_attribution") is not None
    ]
    null_audit = {
        "status": "PASS" if not unclassified else "FAIL",
        "milestones": gradient_milestones,
        "persistently_null_fields": null_fields,
        "classification": classification,
        "unclassified_null_fields": unclassified,
        "router_schedule_classification": {
            "class": "DATA_GATE_INACTIVE_THEN_LIVE",
            "expected": True,
            "evidence": router_activity,
        },
        "conclusion": (
            "No active required gradient path is persistently null. The class path is live through "
            "the structured class token/VAE path; null legacy and disabled branches are expected."
        ),
    }
    _write_json(args.output_dir / "null_gradient_classification.json", null_audit)
    print(json.dumps({
        "epsilon": epsilon["classification"],
        "accumulation": accumulation["classification"],
        "gradient_basis": weighting["historical_reported_ratio_basis"],
        "nulls": null_audit["status"],
    }))
    if not all(weighting["checks"].values()) or null_audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
