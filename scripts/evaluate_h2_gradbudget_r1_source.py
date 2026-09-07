#!/usr/bin/env python3
"""Run the H2 GradBudget R1 frozen VisA source endpoint audit only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from h2_clean.contract import sha256_file  # noqa: E402
from h2_clean.precision import PrecisionPolicy  # noqa: E402
from scripts.evaluate_h2_fixed_fusion_r2_source import (  # noqa: E402
    CLASS_NAMES,
    ENDPOINT_SUBSET,
    H2_EQUAL_STAGE_FUSION_WEIGHTS,
    SPLIT_IDENTITY,
    START_CHECKPOINT,
    endpoint_state,
    evaluate_arm,
    load_endpoint_selection,
    make_model,
)


PROTOCOL_ID = "H2_NEXT_MECHANISM_BOUNDED_R1_LATE_CONVLORA_ABNORMAL_DICE"
ARM_CONTROL = "A_GRADBUDGET_SHORT_R1_CONTROL"
ARM_CANDIDATE = "A_GRADBUDGET_SHORT_R1_CANDIDATE"
ARM_ORDER = ("E1", ARM_CONTROL, ARM_CANDIDATE)
IDENTITY_KEYS = ("attempt_index", "epoch", "batch", "file_names", "image_sha256", "mask_sha256", "labels")
CONTROL_CSV = REPO / "audit/H2_GRADBUDGET_R1_CONTROL.csv"
CANDIDATE_CSV = REPO / "audit/H2_GRADBUDGET_R1_CANDIDATE.csv"
SCOPE_CSV = REPO / "audit/H2_GRADBUDGET_R1_PARAMETER_SCOPE.csv"


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def result_value(result: dict, region: str, stat: str) -> float:
    return float(result["coverage"][region][stat])


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def exact_batch_match(control: list[dict[str, str]], candidate: list[dict[str, str]]) -> bool:
    return (
        len(control) == len(candidate) == 500
        and all(
            all(left[key] == right[key] for key in IDENTITY_KEYS)
            for left, right in zip(control, candidate)
        )
    )


def finite_float_rows(rows: list[dict[str, str]], keys: tuple[str, ...]) -> bool:
    return all(all(np.isfinite(float(row[key])) for key in keys) for row in rows)


def telemetry_summary(rows: list[dict[str, str]], arm: str) -> dict:
    usable = [row for row in rows if row.get("status") == "success" and row.get("alpha") not in (None, "")]
    if not usable:
        return {"arm": arm, "rows": 0, "finite": False, "activity_fraction": 0.0}
    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in usable], dtype=np.float64)
    alpha = values("alpha")
    raw = values("raw_ratio")
    effective = values("effective_ratio")
    active = np.asarray([row.get("budget_active") == "True" for row in usable], dtype=bool)
    active_rows = [row for row in usable if row.get("budget_active") == "True"]
    effective_reduced_when_active = all(
        float(row["effective_ratio"]) < float(row["raw_ratio"]) + 1.0e-12
        for row in active_rows
    )
    return {
        "arm": arm,
        "rows": len(usable),
        "finite": finite_float_rows(usable, ("alpha", "raw_ratio", "effective_ratio", "g_abn_norm", "g_rest_norm")),
        "alpha": {
            "median": float(np.median(alpha)),
            "p10": float(np.quantile(alpha, .10)),
            "p90": float(np.quantile(alpha, .90)),
            "min": float(alpha.min()),
            "max": float(alpha.max()),
        },
        "raw_ratio": {
            "median": float(np.median(raw)),
            "p10": float(np.quantile(raw, .10)),
            "p90": float(np.quantile(raw, .90)),
        },
        "effective_ratio": {
            "median": float(np.median(effective)),
            "p10": float(np.quantile(effective, .10)),
            "p90": float(np.quantile(effective, .90)),
        },
        "activity_fraction": float(active.mean()),
        "active_rows": int(active.sum()),
        "effective_ratio_lower_than_raw_when_active": effective_reduced_when_active,
        "gradients_unscaled": all(row.get("gradients_unscaled") in ("1", "True") for row in usable),
        "safe_anchor_once_each_success": all(row.get("safe_anchor_calls") == "1" for row in usable),
    }


def numerical_summary(control_summary: dict, candidate_summary: dict, batch_match: bool, endpoint_finite: bool, control_rows, candidate_rows) -> dict:
    equal_success = control_summary["successful_steps"] == candidate_summary["successful_steps"]
    arms_ok = []
    for summary in (control_summary, candidate_summary):
        arms_ok.append(
            summary["attempted_steps"] == 500
            and summary["nonfinite_loss_skips"] == 0
            and summary["nonfinite_grad_skips"] <= 1
            and summary["max_consecutive_nonfinite_grad_skips"] <= 1
            and summary["optimizer_state_failures"] == 0
            and summary["parameter_corruption"] == 0
            and summary["batch_match_gate"] == "PASS"
            and summary["numerical_failure"] is None
        )
    return {
        "control_arm_contract": bool(arms_ok[0]),
        "candidate_arm_contract": bool(arms_ok[1]),
        "equal_successful_count": bool(equal_success),
        "exact_attempt_identity": bool(batch_match),
        "endpoint_finite": bool(endpoint_finite),
        "control_successful_steps": int(control_summary["successful_steps"]),
        "candidate_successful_steps": int(candidate_summary["successful_steps"]),
        "control_nonfinite_grad_skips": int(control_summary["nonfinite_grad_skips"]),
        "candidate_nonfinite_grad_skips": int(candidate_summary["nonfinite_grad_skips"]),
        "valid": bool(all(arms_ok) and equal_success and batch_match and endpoint_finite),
    }


def drift_value(result: dict, key: str) -> float:
    return float(result["parameter_drift"]["requested_families"][key]["difference_l2"])


def build_gates(results: dict, numerical: dict, scope: dict, telemetry: dict) -> dict:
    control = results[ARM_CONTROL]
    candidate = results[ARM_CANDIDATE]
    activity = telemetry[ARM_CANDIDATE]["activity_fraction"] > 0.0
    effective_reduction = telemetry[ARM_CANDIDATE]["effective_ratio_lower_than_raw_when_active"]
    late_drift_reduced = (
        drift_value(candidate, "Conv-LoRA_stage2") + drift_value(candidate, "Conv-LoRA_stage3")
        < drift_value(control, "Conv-LoRA_stage2") + drift_value(control, "Conv-LoRA_stage3")
    )
    stage1_outside = (
        "stage1" not in scope["selected_stages"]
        and all(int(row["selected"]) == 0 for row in csv.DictReader(SCOPE_CSV.open(newline="")) if row["stage"] == "stage1")
    )
    direct = {
        "activity_fraction_gt_zero": bool(activity),
        "effective_abnormal_rest_lower_than_raw_when_alpha_lt_one": bool(effective_reduction),
        "late_stage2_plus_stage3_convlora_drift_lower_than_control": bool(late_drift_reduced),
        "stage1_outside_selected_scope": bool(stage1_outside),
    }
    scientific = {
        "numerical_validity": bool(numerical["valid"]),
        "batch_identity": bool(numerical["exact_attempt_identity"]),
        "mechanism_activity": bool(activity and effective_reduction),
        "late_drift_reduced": bool(late_drift_reduced),
        "final_ap_not_below_control": candidate["ranking"]["final"]["ap"] >= control["ranking"]["final"]["ap"] - 1.0e-6,
        "final_auroc_not_below_control": candidate["ranking"]["final"]["auroc"] >= control["ranking"]["final"]["auroc"] - 1.0e-6,
        "positive_mean_not_below_control": result_value(candidate, "positive", "mean") >= result_value(control, "positive", "mean"),
        "positive_median_not_below_control": result_value(candidate, "positive", "median") >= result_value(control, "positive", "median"),
        "interior_mean_not_below_control": result_value(candidate, "interior", "mean") >= result_value(control, "interior", "mean"),
        "interior_median_not_below_control": result_value(candidate, "interior", "median") >= result_value(control, "interior", "median"),
        "near_background_p95_not_above_control": result_value(candidate, "near_background", "p95") <= result_value(control, "near_background", "p95"),
        "near_background_p99_not_above_control": result_value(candidate, "near_background", "p99") <= result_value(control, "near_background", "p99"),
        "stage1_ap_not_below_control": candidate["ranking"]["stage_1"]["ap"] >= control["ranking"]["stage_1"]["ap"] - 1.0e-6,
        "stage1_auroc_not_below_control": candidate["ranking"]["stage_1"]["auroc"] >= control["ranking"]["stage_1"]["auroc"] - 1.0e-6,
    }
    all_scientific = all(scientific.values())
    bounded = "PASS" if all_scientific and all(direct.values()) else "FAIL"
    if not numerical["valid"]:
        bounded = "INVALID_NUMERICAL"
    red_team = {
        "DRIFT_REDUCTION_ALONE_IS_NOT_SUFFICIENT": bool(late_drift_reduced and not all_scientific),
        "MECHANISM_NOT_ESTABLISHED": bool(not activity),
        "OVER_REGULARIZATION": bool(
            result_value(candidate, "near_background", "p95") < result_value(control, "near_background", "p95")
            and result_value(candidate, "positive", "mean") < result_value(control, "positive", "mean")
        ),
    }
    return {
        "direct": direct,
        "scientific": scientific,
        "all_direct_gates": bool(all(direct.values())),
        "all_scientific_gates": bool(all_scientific),
        "BOUNDED_SCREEN": bounded,
        "red_team_interpretations": red_team,
        "confirmatory_support": bool(bounded == "PASS"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/h2_late_convlora_gradbudget_bounded_r1")
    args = parser.parse_args()
    root = Path(args.root)
    selected_rows, selected_by_category = load_endpoint_selection()
    split = json.loads(SPLIT_IDENTITY.read_text())
    if split["intersection_count"] != 0 or split["endpoint_eval_count"] != 96:
        raise RuntimeError("frozen endpoint split identity mismatch")
    paths = {
        "E1": START_CHECKPOINT,
        ARM_CONTROL: root / ARM_CONTROL / "final.pth",
        ARM_CANDIDATE: root / ARM_CANDIDATE / "final.pth",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError([str(path) for path in paths.values() if not path.is_file()])
    control_rows = read_rows(CONTROL_CSV)
    candidate_rows = read_rows(CANDIDATE_CSV)
    batch_match = exact_batch_match(control_rows, candidate_rows)
    control_summary = json.loads((root / ARM_CONTROL / "summary.json").read_text())
    candidate_summary = json.loads((root / ARM_CANDIDATE / "summary.json").read_text())
    telemetry = {
        ARM_CONTROL: telemetry_summary(control_rows, ARM_CONTROL),
        ARM_CANDIDATE: telemetry_summary(candidate_rows, ARM_CANDIDATE),
    }
    scope_rows = list(csv.DictReader(SCOPE_CSV.open(newline="")))
    scope = {
        "selected_stages": ["stage2", "stage3"],
        "selected_module_names": ["image_adapter.lora_adapters.1", "image_adapter.lora_adapters.2"],
        "selected_parameter_count": sum(int(row["selected"]) for row in scope_rows),
    }

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    e1_payload = torch.load(START_CHECKPOINT, map_location="cpu", weights_only=False)
    e1_state = endpoint_state(e1_payload)["image_adapter"]
    model = make_model(device, H2_EQUAL_STAGE_FUSION_WEIGHTS)
    results = {}
    e1_pooled = None
    for arm in ARM_ORDER:
        model.stage_fusion_weights = tuple(H2_EQUAL_STAGE_FUSION_WEIGHTS)
        result, pooled = evaluate_arm(
            model,
            paths[arm],
            selected_rows,
            selected_by_category,
            device,
            policy,
            e1_pooled,
            e1_state,
        )
        if arm == "E1":
            e1_pooled = pooled
            result["geometry"] = [
                {"stage": stage + 1, "pooled_feature_cosine_to_e1_mean": 1.0,
                 "pooled_feature_cosine_to_e1_median": 1.0, "linear_cka_to_e1": 1.0}
                for stage in range(3)
            ]
        results[arm] = result
        del pooled
        torch.cuda.empty_cache()

    endpoint_finite = all(bool(results[arm]["endpoint_finite"]) for arm in ARM_ORDER)
    numerical = numerical_summary(control_summary, candidate_summary, batch_match, endpoint_finite, control_rows, candidate_rows)
    gates = build_gates(results, numerical, scope, telemetry)
    control = results[ARM_CONTROL]
    candidate = results[ARM_CANDIDATE]
    deltas = {
        "final_ap": candidate["ranking"]["final"]["ap"] - control["ranking"]["final"]["ap"],
        "final_auroc": candidate["ranking"]["final"]["auroc"] - control["ranking"]["final"]["auroc"],
        "positive_mean": result_value(candidate, "positive", "mean") - result_value(control, "positive", "mean"),
        "positive_median": result_value(candidate, "positive", "median") - result_value(control, "positive", "median"),
        "interior_mean": result_value(candidate, "interior", "mean") - result_value(control, "interior", "mean"),
        "interior_median": result_value(candidate, "interior", "median") - result_value(control, "interior", "median"),
        "near_background_p95": result_value(candidate, "near_background", "p95") - result_value(control, "near_background", "p95"),
        "near_background_p99": result_value(candidate, "near_background", "p99") - result_value(control, "near_background", "p99"),
        "near_background_gt_positive_inversion": candidate["inversion_rates"]["near_background_gt_positive"] - control["inversion_rates"]["near_background_gt_positive"],
        "near_background_gt_interior_inversion": candidate["inversion_rates"]["near_background_gt_interior"] - control["inversion_rates"]["near_background_gt_interior"],
        "stage1_ap": candidate["ranking"]["stage_1"]["ap"] - control["ranking"]["stage_1"]["ap"],
        "stage1_auroc": candidate["ranking"]["stage_1"]["auroc"] - control["ranking"]["stage_1"]["auroc"],
        "stage2_ap": candidate["ranking"]["stage_2"]["ap"] - control["ranking"]["stage_2"]["ap"],
        "stage2_auroc": candidate["ranking"]["stage_2"]["auroc"] - control["ranking"]["stage_2"]["auroc"],
        "stage3_ap": candidate["ranking"]["stage_3"]["ap"] - control["ranking"]["stage_3"]["ap"],
        "stage3_auroc": candidate["ranking"]["stage_3"]["auroc"] - control["ranking"]["stage_3"]["auroc"],
        "stage2_convlora_drift": drift_value(candidate, "Conv-LoRA_stage2") - drift_value(control, "Conv-LoRA_stage2"),
        "stage3_convlora_drift": drift_value(candidate, "Conv-LoRA_stage3") - drift_value(control, "Conv-LoRA_stage3"),
    }
    output = {
        "protocol_id": PROTOCOL_ID,
        "evaluation_scope": "VisA-only frozen source endpoint; no Medical/MVTec/target inference",
        "manifest": "dataset/hub/VisA.jsonl",
        "manifest_sha256": split["manifest_sha256"],
        "endpoint_subset_sha256": sha256_file(ENDPOINT_SUBSET),
        "endpoint_sample_count": len(selected_rows),
        "calibration_subset_evaluated": False,
        "endpoint_fusion_weights": {arm: list(H2_EQUAL_STAGE_FUSION_WEIGHTS) for arm in ARM_ORDER},
        "parameter_scope": scope,
        "arms": results,
        "training_summaries": {ARM_CONTROL: control_summary, ARM_CANDIDATE: candidate_summary},
        "exact_batch_identity_match": batch_match,
        "numerical_validity": numerical,
        "candidate_gradient_telemetry": telemetry,
        "candidate_minus_control": deltas,
        "gates": gates,
    }
    json_dump(REPO / "audit/H2_GRADBUDGET_R1_ENDPOINT.json", output)
    rows = []
    for arm in ARM_ORDER:
        result = results[arm]
        for metric_name, metric in result["ranking"].items():
            rows.append({"arm": arm, "record": "ranking", "metric": metric_name, **metric})
        for region, metric in result["coverage"].items():
            rows.append({"arm": arm, "record": "coverage", "metric": region, **metric})
        for metric_name, value in result["inversion_rates"].items():
            if metric_name != "definition":
                rows.append({"arm": arm, "record": "inversion_rate", "metric": metric_name, "value": value})
        for metric_name, metric in result["parameter_drift"]["requested_families"].items():
            rows.append({"arm": arm, "record": "parameter_drift", "metric": metric_name, **metric})
        for geometry in result["geometry"]:
            rows.append({"arm": arm, "record": "geometry", "metric": f"stage_{geometry['stage']}", **geometry})
    for arm, summary in telemetry.items():
        rows.append({"arm": arm, "record": "gradient_telemetry", "metric": "activity_fraction", "value": summary["activity_fraction"]})
        rows.append({"arm": arm, "record": "gradient_telemetry", "metric": "alpha_median", "value": summary["alpha"]["median"]})
        rows.append({"arm": arm, "record": "gradient_telemetry", "metric": "raw_ratio_median", "value": summary["raw_ratio"]["median"]})
    for name, value in gates["direct"].items():
        rows.append({"arm": "GATES", "record": "direct_gate", "metric": name, "value": value})
    for name, value in gates["scientific"].items():
        rows.append({"arm": "GATES", "record": "scientific_gate", "metric": name, "value": value})
    fields = sorted({key for row in rows for key in row})
    with (REPO / "audit/H2_GRADBUDGET_R1_ENDPOINT.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "status": gates["BOUNDED_SCREEN"],
        "endpoint": str(REPO / "audit/H2_GRADBUDGET_R1_ENDPOINT.json"),
        "csv": str(REPO / "audit/H2_GRADBUDGET_R1_ENDPOINT.csv"),
        "numerical_validity": numerical["valid"],
        "exact_batch_identity_match": batch_match,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
