#!/usr/bin/env python3
"""Validate the bounded source-only H_short/A_active_short Anchor trajectory."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from h2_clean.contract import (
    ANCHOR_FAMILY_NAMES,
    ANCHOR_LAMBDA_ACTIVE,
    ANCHOR_LAMBDA_OLD,
    ANCHOR_R_MED,
    ANCHOR_TARGET_EFFECTIVE_RATIO,
    PRIMARY_HORIZON,
    RESUME_BRANCH_KEYS,
    SECONDARY_HORIZON,
    TRAINING_HORIZON,
    sha256_file,
    anchor_parameter_family,
)


RHO = 0.10
EPOCHS = (2, 3, 4, 5)
MEANINGFUL_ACTIVE_RATIO_FLOOR = 0.02
RATIO_TOLERANCE = 1.0e-6
STEP_RE = re.compile(r"anchor_family_step epoch=(\d+) batch=(\d+) metrics=(\{.*\})$")
SKIP_RE = re.compile(r"skip_counts epoch=(\d+) non_finite_loss=(\d+) non_finite_grad=(\d+)")


def load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_step_metrics(log_path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    parsed: dict[tuple[int, int], dict[str, Any]] = {}
    for line in log_path.read_text().splitlines():
        match = STEP_RE.search(line)
        if match is None:
            continue
        epoch = int(match.group(1))
        batch = int(match.group(2))
        parsed[(epoch, batch)] = json.loads(match.group(3))
    return parsed


def parse_skip_counts(log_path: Path) -> dict[int, tuple[int, int]]:
    parsed: dict[int, tuple[int, int]] = {}
    for line in log_path.read_text().splitlines():
        match = SKIP_RE.search(line)
        if match is not None:
            parsed[int(match.group(1))] = (int(match.group(2)), int(match.group(3)))
    return parsed


def parse_batch_identities(log_path: Path, epoch: int) -> list[str]:
    values = []
    for line in log_path.read_text().splitlines():
        if f"batch_identity epoch={epoch} " in line and "batch_identity " in line:
            values.append(line.split("batch_identity ", 1)[1])
    return values


def finite(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def assert_finite_metrics(metrics: dict[str, Any], label: str) -> None:
    for key in (
        "rho",
        "anchor_lambda",
        "task_floor",
        "global_task_grad_norm",
        "global_effective_anchor_grad_norm",
        "global_effective_ratio",
        "max_effective_active_family_ratio",
    ):
        if not finite(metrics.get(key)):
            raise AssertionError(f"{label} non-finite metric {key}: {metrics.get(key)!r}")
    if not metrics.get("family_partition_complete", False):
        raise AssertionError(f"{label} family partition is incomplete")
    families = metrics.get("families", {})
    if set(families) != set(ANCHOR_FAMILY_NAMES):
        raise AssertionError(f"{label} family names are not the complete fixed partition")
    for family in ANCHOR_FAMILY_NAMES:
        row = families[family]
        for key in (
            "task_floor",
            "task_grad_norm",
            "anchor_grad_raw_norm",
            "raw_gradient_ratio",
            "cosine_task_anchor_raw",
            "lambda_anchor_grad_norm",
            "effective_anchor_grad_norm",
            "effective_ratio",
            "scale",
        ):
            if not finite(row.get(key)):
                raise AssertionError(f"{label}/{family} non-finite metric {key}")


def summarize_steps(
    steps: dict[tuple[int, int], dict[str, Any]],
    expected_batches: int,
    *,
    label: str,
    enforce_budget: bool,
) -> tuple[dict[str, Any], float, float, bool, bool]:
    summary: dict[str, Any] = {}
    max_effective_ratio = 0.0
    max_raw_ratio = 0.0
    near_zero_ok = True
    budget_ok = True
    for epoch in EPOCHS:
        rows = []
        for batch in range(expected_batches):
            key = (epoch, batch)
            if key not in steps:
                raise AssertionError(f"{label} missing Anchor-family telemetry for {key}")
            metrics = steps[key]
            assert_finite_metrics(metrics, f"{label} epoch={epoch} batch={batch}")
            rows.append(metrics)
        family_summary: dict[str, Any] = {}
        global_ratios = []
        for metrics in rows:
            global_ratios.append(float(metrics["global_effective_ratio"]))
            for family in ANCHOR_FAMILY_NAMES:
                row = metrics["families"][family]
                effective_ratio = float(row["effective_ratio"])
                max_effective_ratio = max(max_effective_ratio, effective_ratio)
                raw_ratio = row.get("raw_gradient_ratio")
                if raw_ratio is not None:
                    max_raw_ratio = max(max_raw_ratio, float(raw_ratio))
                if row["status"] == "TASK_NEAR_ZERO" and float(row["effective_anchor_grad_norm"]) != 0.0:
                    near_zero_ok = False
                if enforce_budget and float(row["task_grad_norm"]) > float(row["task_floor"]):
                    if effective_ratio > RHO + 1.0e-6:
                        budget_ok = False
        summary[str(epoch)] = {
            "steps": len(rows),
            "global_effective_ratio_min": min(global_ratios),
            "global_effective_ratio_max": max(global_ratios),
            "family_effective_ratio_max": max(
                float(metrics["max_effective_active_family_ratio"]) for metrics in rows
            ),
            "task_floor_min": min(float(metrics["task_floor"]) for metrics in rows),
            "task_floor_max": max(float(metrics["task_floor"]) for metrics in rows),
        }
    return summary, max_effective_ratio, max_raw_ratio, near_zero_ok, budget_ok


def meaningful_active_ratio_max(steps: dict[tuple[int, int], dict[str, Any]]) -> float:
    values = [
        float(row["effective_ratio"])
        for metrics in steps.values()
        for row in metrics.get("families", {}).values()
        if row.get("status") == "TASK_ACTIVE"
    ]
    return max(values, default=0.0)


def family_drift(reference: dict[str, torch.Tensor], state: dict[str, torch.Tensor]) -> dict[str, float]:
    sums = {family: 0.0 for family in ANCHOR_FAMILY_NAMES}
    for name in sorted(reference):
        if name not in state:
            raise AssertionError(f"missing image-adapter state {name}")
        family = anchor_parameter_family(name)
        difference = state[name].float() - reference[name].float()
        sums[family] += float(difference.square().sum().item())
    return {family: math.sqrt(value) for family, value in sums.items()}




def module_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for module in sorted(set(left) | set(right)):
        left_values = left.get(module, {})
        right_values = right.get(module, {})
        total = 0.0
        for name in sorted(set(left_values) | set(right_values)):
            if name not in left_values or name not in right_values:
                raise AssertionError(f"state key mismatch {module}/{name}")
            total += float((right_values[name].float() - left_values[name].float()).square().sum().item())
        result[module] = math.sqrt(total)
    return result


def validate_configs(shared: dict[str, Any], h: dict[str, Any], a: dict[str, Any]) -> dict[str, Any]:
    if h["parent_scientific_config"] != a["parent_scientific_config"]:
        raise AssertionError("H_short and A_active_short parent scientific configs differ")
    non_branch = sorted((set(h["resolved_scientific_config"]) | set(a["resolved_scientific_config"])) - set(RESUME_BRANCH_KEYS))
    mismatches = [
        key for key in non_branch
        if h["resolved_scientific_config"].get(key) != a["resolved_scientific_config"].get(key)
    ]
    if mismatches:
        raise AssertionError(f"H/A differ outside Anchor/CIR branch keys: {mismatches}")
    h_config = h["resolved_scientific_config"]
    a_config = a["resolved_scientific_config"]
    expected_h = {
        "use_safe_anchor": False,
        "anchor_lambda": 0.0,
        "anchor_gradient_budget": False,
        "anchor_family_budget": RHO,
        "use_cir_training": False,
        "cir_alpha": 0.0,
        "training_horizon": TRAINING_HORIZON,
        "primary_horizon": PRIMARY_HORIZON,
        "secondary_horizon": SECONDARY_HORIZON,
        "anchor_target_effective_ratio": ANCHOR_TARGET_EFFECTIVE_RATIO,
        "anchor_r_med": ANCHOR_R_MED,
        "anchor_lambda_old": ANCHOR_LAMBDA_OLD,
    }
    expected_a = {
        "use_safe_anchor": True,
        "anchor_lambda": ANCHOR_LAMBDA_ACTIVE,
        "anchor_gradient_budget": True,
        "anchor_family_budget": RHO,
        "use_cir_training": False,
        "cir_alpha": 0.0,
    }
    for key, value in expected_h.items():
        if h_config.get(key) != value:
            raise AssertionError(f"H_short {key}={h_config.get(key)!r}, expected {value!r}")
    for key, value in expected_a.items():
        if a_config.get(key) != value:
            raise AssertionError(f"A_active_short {key}={a_config.get(key)!r}, expected {value!r}")
    if h_config.get("anchor_family_budget") != RHO:
        raise AssertionError("H_short did not carry the fixed family rho in scientific identity")
    if not a_config.get("anchor_reference_sha256"):
        raise AssertionError("A_active_short has no E1 Anchor reference identity")
    if shared["parent_scientific_config"] != h["parent_scientific_config"]:
        raise AssertionError("shared E1 and H_short parent scientific configs differ")
    for key in ("epoch", "training_horizon", "primary_horizon", "secondary_horizon"):
        if shared["resolved_scientific_config"].get(key) != h_config.get(key):
            raise AssertionError(f"shared E1 and H_short horizon identity differs for {key}")
    if shared["epoch"] != 1:
        raise AssertionError(f"shared checkpoint epoch is {shared['epoch']}, expected 1")
    return {"parent_equal": True, "non_branch_mismatches": [], "H_native": True, "A_anchor_budget": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-batches", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    shared_path = root / "shared_e1" / "adapter_1.pth"
    h_path = root / "H_short" / f"adapter_{EPOCHS[-1]}.pth"
    a_path = root / "A_active_short" / f"adapter_{EPOCHS[-1]}.pth"
    for path in (shared_path, h_path, a_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    shared = load(shared_path)
    h = load(h_path)
    a = load(a_path)
    config_result = validate_configs(shared, h, a)
    h_steps = parse_step_metrics(root / "H_short" / "train.log")
    a_steps = parse_step_metrics(root / "A_active_short" / "train.log")
    h_summary, h_max_effective, h_max_raw, h_near_zero_ok, h_budget_ok = summarize_steps(
        h_steps, args.expected_batches, label="H_short", enforce_budget=False
    )
    a_summary, a_max_effective, a_max_raw, a_near_zero_ok, a_budget_ok = summarize_steps(
        a_steps, args.expected_batches, label="A_active_short", enforce_budget=True
    )
    for label, log_path in (("H_short", root / "H_short" / "train.log"), ("A_active_short", root / "A_active_short" / "train.log")):
        skips = parse_skip_counts(log_path)
        for epoch in EPOCHS:
            if skips.get(epoch) != (0, 0):
                raise AssertionError(f"{label} non-finite skips at epoch {epoch}: {skips.get(epoch)}")
    identity_equal = True
    identity_counts: dict[str, int] = {}
    for epoch in EPOCHS:
        h_ids = parse_batch_identities(root / "H_short" / "train.log", epoch)
        a_ids = parse_batch_identities(root / "A_active_short" / "train.log", epoch)
        identity_counts[f"epoch_{epoch}"] = len(h_ids)
        if len(h_ids) != args.expected_batches or len(a_ids) != args.expected_batches or h_ids != a_ids:
            identity_equal = False
    if not identity_equal:
        raise AssertionError(f"H/A batch identities differ: {identity_counts}")
    shared_image = shared["model_state"]["image_adapter"]
    h_image = h["model_state"]["image_adapter"]
    a_image = a["model_state"]["image_adapter"]
    h_drift = family_drift(shared_image, h_image)
    a_drift = family_drift(shared_image, a_image)
    h_a_deltas = {}
    for epoch in EPOCHS:
        h_a_deltas[f"E{epoch}"] = module_delta(
            load(root / "H_short" / f"adapter_{epoch}.pth")["model_state"],
            load(root / "A_active_short" / f"adapter_{epoch}.pth")["model_state"],
        )
    image_delta = max(row.get("image_adapter", 0.0) for row in h_a_deltas.values())
    global_effective_ratios = [
        float(metrics["global_effective_ratio"])
        for metrics in a_steps.values()
        if metrics.get("global_effective_ratio") is not None
    ]
    global_effective_nonzero = any(value > 1.0e-12 for value in global_effective_ratios)
    global_effective_not_tiny = max(global_effective_ratios, default=0.0) > 1.0e-6
    meaningful_active_ratio = meaningful_active_ratio_max(a_steps)
    h_native = config_result["H_native"] and h_max_effective <= 1.0e-10 and h_budget_ok
    a_safe = a_budget_ok and a_max_effective <= RHO + RATIO_TOLERANCE and a_near_zero_ok
    no_40000x_pathology = a_max_effective < 40000.0
    expected_difference = identity_equal and a_safe and image_delta > 0.0
    hard_checks = {
        "cap": a_safe,
        "near_zero": h_near_zero_ok and a_near_zero_ok,
        "finite": True,
        "H_native": h_native,
        "A_only": expected_difference,
        "meaningful_active": meaningful_active_ratio >= MEANINGFUL_ACTIVE_RATIO_FLOOR - RATIO_TOLERANCE,
        "anchor_after_drift": global_effective_nonzero and a_max_effective > 1.0e-12,
        "no_40000x_pathology": no_40000x_pathology,
    }
    safety_gate = all(hard_checks[key] for key in ("cap", "near_zero", "finite", "H_native", "A_only", "no_40000x_pathology"))
    activation_gate = safety_gate and hard_checks["meaningful_active"] and hard_checks["anchor_after_drift"]
    if not safety_gate:
        anchor_status = "FAMILY_UNSAFE"
    elif not activation_gate:
        anchor_status = "FAMILY_SAFE_BUT_NEGLIGIBLE"
    else:
        anchor_status = "FAMILY_SAFE_ACTIVE"
    result = {
        "SHARED_E1_CHECKPOINT_SHA256": sha256_file(shared_path),
        "ACTIVATION_CODE_SHA": shared.get("implementation_git_sha"),
        "ACTIVATION_WORKING_TREE_DIFF_SHA256": shared.get("working_tree_diff_sha256"),
        "ACTIVATION_CONFIG_SHA256": shared.get("config_sha256"),
        "ACTIVATION_CLIP_SHA256": shared.get("clip_sha256"),
        "ACTIVATION_DATASET_MANIFEST_SHA256": shared.get("dataset_manifest_sha256"),
        "ACTIVATION_BASE_H2_COMMIT": shared.get("base_h2_commit"),
        "scope": "VisA train source-only bounded mechanism validation; no Medical/MVTec/target labels; no full training",
        "shared_e1": str(shared_path),
        "TRAINING_HORIZON": TRAINING_HORIZON,
        "PRIMARY_HORIZON": PRIMARY_HORIZON,
        "SECONDARY_HORIZON": SECONDARY_HORIZON,
        "ANCHOR_R_MED": ANCHOR_R_MED,
        "ANCHOR_TARGET_EFFECTIVE_RATIO": ANCHOR_TARGET_EFFECTIVE_RATIO,
        "MEANINGFUL_ACTIVE_RATIO_FLOOR": MEANINGFUL_ACTIVE_RATIO_FLOOR,
        "HARD_CHECKS": hard_checks,
        "ACTIVATION_GATE": "PASS" if activation_gate else "FAIL",
        "H_short": {
            "passed": h_native,
            "epochs": list(EPOCHS),
            "summary": h_summary,
            "max_effective_active_family_ratio": h_max_effective,
            "max_raw_gradient_ratio": h_max_raw,
        },
        "A_active_short": {
            "passed": a_safe,
            "epochs": list(EPOCHS),
            "summary": a_summary,
            "max_effective_active_family_ratio": a_max_effective,
            "max_raw_gradient_ratio": a_max_raw,
            "global_effective_nonzero_after_drift": global_effective_nonzero,
            "global_effective_not_tiny_after_drift": global_effective_not_tiny,
            "no_40000x_pathology": no_40000x_pathology,
        },
        "H_A_ONLY_EXPECTED_DIFFERENCE": expected_difference,
        "same_batch_identity": identity_equal,
        "batch_identity_counts": identity_counts,
        "drift_from_shared_e1": {"H_short": h_drift, "A_active_short": a_drift},
        "H_vs_A_delta": h_a_deltas,
        "FAMILY_PARTITION_COMPLETE": True,
        "ANCHOR_FAMILY_BUDGET_RHO": RHO,
        "ANCHOR_LAMBDA": ANCHOR_LAMBDA_ACTIVE,
        "MEANINGFUL_TASK_ACTIVE_RATIO_MAX": meaningful_active_ratio,
        "FINITE_SKIPS": "PASS",
        "ANCHOR_STATUS": anchor_status,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not activation_gate:
        failed = [key for key, passed in hard_checks.items() if not passed]
        raise AssertionError(f"Anchor activation gate failed: {failed}")


if __name__ == "__main__":
    main()
