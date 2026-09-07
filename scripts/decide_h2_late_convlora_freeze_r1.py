#!/usr/bin/env python3
"""Apply the preregistered H2 late Conv-LoRA freeze decision gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "H2_LATE_CONVLORA_FREEZE_R1"
ARM_CONTROL = "A_LATE_FREEZE_R1_CONTROL"
ARM_CANDIDATE = "A_LATE_FREEZE_R1_STAGE23_CONVLORA"
ENDPOINT_JSON = REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_ENDPOINT.json"
PARITY_JSON = REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_IMPLEMENTATION_PARITY.json"
CONTROL_SUMMARY = Path("/workspace/h2_late_convlora_freeze_r1") / ARM_CONTROL / "summary.json"
CANDIDATE_SUMMARY = Path("/workspace/h2_late_convlora_freeze_r1") / ARM_CANDIDATE / "summary.json"
SCOPE_CSV = REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_PARAMETER_SCOPE.csv"


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def v(result: dict, region: str, stat: str) -> float:
    return float(result["coverage"][region][stat])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/workspace/h2_late_convlora_freeze_r1")
    args = parser.parse_args()
    root = Path(args.root)
    endpoint = json.loads(ENDPOINT_JSON.read_text())
    parity = json.loads(PARITY_JSON.read_text())
    control_summary = json.loads((root / ARM_CONTROL / "summary.json").read_text())
    candidate_summary = json.loads((root / ARM_CANDIDATE / "summary.json").read_text())
    control = endpoint["arms"][ARM_CONTROL]
    candidate = endpoint["arms"][ARM_CANDIDATE]
    cdrift = control_summary["parameter_drift_from_e10"]["image_adapter"]
    kdrift = candidate_summary["parameter_drift_from_e10"]["image_adapter"]
    candidate_nonf = candidate_summary["parameter_drift_from_e10"]["non_frozen_trainable"]
    direct = {
        "implementation_parity_pass": bool(parity["true_freeze_implementation"]),
        "candidate_stage2_convlora_drift_exactly_zero": bool(kdrift["stage2_convlora"]["difference_l2"] == 0.0 and candidate_summary["candidate_frozen_param_byte_identity"]),
        "candidate_stage3_convlora_drift_exactly_zero": bool(kdrift["stage3_convlora"]["difference_l2"] == 0.0 and candidate_summary["candidate_frozen_param_byte_identity"]),
        "control_stage2_convlora_drift_gt_zero": bool(cdrift["stage2_convlora"]["difference_l2"] > 0.0),
        "control_stage3_convlora_drift_gt_zero": bool(cdrift["stage3_convlora"]["difference_l2"] > 0.0),
        "candidate_stage2_pre_suppression_gradient_meaningful": bool(candidate_summary["pre_suppression_stage2_gradient_nonzero_fraction"] > 0.0),
        "candidate_stage3_pre_suppression_gradient_meaningful": bool(candidate_summary["pre_suppression_stage3_gradient_nonzero_fraction"] > 0.0),
        "non_selected_training_continues": bool(
            candidate_nonf["image_non_frozen"]["difference_l2"] > 0.0
            and candidate_nonf["text_adapter"]["difference_l2"] > 0.0
            and candidate_nonf["soft_prompt"]["difference_l2"] > 0.0
        ),
    }
    mechanism_test = "SUPPORTED" if all(direct.values()) else "NOT_INTERPRETABLE"
    numerical_validity = bool(
        endpoint["numerical_validity"]
        and control_summary["attempted"] == candidate_summary["attempted"] == 500
        and control_summary["successful_updates"] == candidate_summary["successful_updates"]
        and control_summary["natural_nonfinite_loss_skips"] == 0
        and control_summary["natural_nonfinite_grad_skips"] == 0
        and candidate_summary["natural_nonfinite_loss_skips"] == 0
        and candidate_summary["natural_nonfinite_grad_skips"] == 0
        and candidate_summary["forced_parity_skips"] == control_summary["natural_nonfinite_loss_skips"] + control_summary["natural_nonfinite_grad_skips"]
        and control_summary["batch_match_gate"] == candidate_summary["batch_match_gate"] == "PASS"
    )
    exact_batch = bool(endpoint["exact_batch_identity_match"])
    count_match = bool(control_summary["successful_updates"] == candidate_summary["successful_updates"])
    freeze_gate = bool(direct["candidate_stage2_convlora_drift_exactly_zero"] and direct["candidate_stage3_convlora_drift_exactly_zero"])
    scientific = {
        "numerical_validity": numerical_validity,
        "exact_500_attempted_batch_stream": bool(control_summary["attempted"] == candidate_summary["attempted"] == 500),
        "successful_count_match": count_match,
        "true_freeze": freeze_gate,
        "final_ap_not_below_control_minus_1e6": bool(candidate["ranking"]["final"]["ap"] >= control["ranking"]["final"]["ap"] - 1.0e-6),
        "final_auroc_not_below_control_minus_1e6": bool(candidate["ranking"]["final"]["auroc"] >= control["ranking"]["final"]["auroc"] - 1.0e-6),
        "positive_mean_not_below_control": bool(v(candidate, "positive", "mean") >= v(control, "positive", "mean")),
        "positive_median_not_below_control": bool(v(candidate, "positive", "median") >= v(control, "positive", "median")),
        "interior_mean_not_below_control": bool(v(candidate, "interior", "mean") >= v(control, "interior", "mean")),
        "interior_median_not_below_control": bool(v(candidate, "interior", "median") >= v(control, "interior", "median")),
        "near_background_p95_not_above_control": bool(v(candidate, "near_background", "p95") <= v(control, "near_background", "p95")),
        "near_background_p99_not_above_control": bool(v(candidate, "near_background", "p99") <= v(control, "near_background", "p99")),
        "near_background_gt_positive_inversion_not_above_control": bool(candidate["inversion_rates"]["near_background_gt_positive"] <= control["inversion_rates"]["near_background_gt_positive"]),
        "near_background_gt_interior_inversion_not_above_control": bool(candidate["inversion_rates"]["near_background_gt_interior"] <= control["inversion_rates"]["near_background_gt_interior"]),
    }
    all_scientific = bool(all(scientific.values()))
    if not numerical_validity:
        bounded = "INVALID_NUMERICAL"
    elif all_scientific:
        bounded = "PASS"
    else:
        bounded = "FAIL"
    if mechanism_test == "NOT_INTERPRETABLE":
        hypothesis = "NOT_INTERPRETABLE"
        interpretation = "MECHANISM_TEST=NOT_INTERPRETABLE; at least one direct mechanism gate failed."
        governor = "NO"
    elif not numerical_validity:
        hypothesis = "NOT_INTERPRETABLE"
        interpretation = "INVALID_NUMERICAL; no causal interpretation is authorized."
        governor = "NO"
    elif all_scientific:
        hypothesis = "SUPPORTED"
        interpretation = "LATE_CONVLORA_CONTINUED_ADAPTATION=CAUSALLY_SUPPORTED_AS_HARMFUL_ON_SOURCE_LATE_PHASE"
        governor = "YES_PENDING_USER_APPROVAL"
    elif not scientific["final_ap_not_below_control_minus_1e6"] or not scientific["final_auroc_not_below_control_minus_1e6"]:
        hypothesis = "NOT_SUPPORTED"
        interpretation = "SIMPLE_LATE_STAGE23_FREEZE_NOT_SUPPORTED"
        governor = "NO"
    elif (
        scientific["near_background_p95_not_above_control"]
        and scientific["near_background_p99_not_above_control"]
        and (not scientific["positive_mean_not_below_control"] or not scientific["positive_median_not_below_control"] or not scientific["interior_mean_not_below_control"] or not scientific["interior_median_not_below_control"])
    ):
        hypothesis = "NOT_SUPPORTED"
        interpretation = "STAGE23_CONVLORA_ADAPTATION_HAS_NECESSARY_ANOMALY_FUNCTION"
        governor = "NO"
    else:
        hypothesis = "NOT_SUPPORTED"
        interpretation = "SIMPLE_LATE_STAGE23_FREEZE_NOT_SUPPORTED"
        governor = "NO"
    result = {
        "protocol_id": PROTOCOL_ID,
        "branch": "research/h2-late-convlora-freeze-r1",
        "parent_head": "47158bd1a64d23f5f4752fb066e5dd4b91ef07d9",
        "start_epoch": 10,
        "start_global_step": 3607,
        "start_checkpoint_sha256": "64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7",
        "parameter_scope_identity": "image_adapter.lora_adapters.stage2+stage3.all_parameters",
        "stage1_convlora_frozen": False,
        "stage2_convlora_frozen": True,
        "stage3_convlora_frozen": True,
        "true_freeze_implementation": bool(parity["true_freeze_implementation"]),
        "control": {
            "attempted": control_summary["attempted"],
            "natural_skips": control_summary["natural_nonfinite_loss_skips"] + control_summary["natural_nonfinite_grad_skips"],
            "natural_loss_skips": control_summary["natural_nonfinite_loss_skips"],
            "natural_grad_skips": control_summary["natural_nonfinite_grad_skips"],
            "successful": control_summary["successful_updates"],
            "final_global_step": control_summary["final_global_step"],
            "final_scaler": control_summary["final_scaler_value"],
        },
        "candidate": {
            "attempted": candidate_summary["attempted"],
            "natural_skips": candidate_summary["natural_nonfinite_loss_skips"] + candidate_summary["natural_nonfinite_grad_skips"],
            "natural_loss_skips": candidate_summary["natural_nonfinite_loss_skips"],
            "natural_grad_skips": candidate_summary["natural_nonfinite_grad_skips"],
            "forced_parity_skips": candidate_summary["forced_parity_skips"],
            "successful": candidate_summary["successful_updates"],
            "final_global_step": candidate_summary["final_global_step"],
            "final_scaler": candidate_summary["final_scaler_value"],
        },
        "exact_batch_match": exact_batch,
        "successful_count_match": count_match,
        "control_stage2_drift": cdrift["stage2_convlora"]["difference_l2"],
        "control_stage3_drift": cdrift["stage3_convlora"]["difference_l2"],
        "candidate_stage2_drift": kdrift["stage2_convlora"]["difference_l2"],
        "candidate_stage3_drift": kdrift["stage3_convlora"]["difference_l2"],
        "candidate_frozen_param_byte_identity": bool(candidate_summary["candidate_frozen_param_byte_identity"]),
        "candidate_pre_suppression_stage2_gradient_nonzero_fraction": candidate_summary["pre_suppression_stage2_gradient_nonzero_fraction"],
        "candidate_pre_suppression_stage3_gradient_nonzero_fraction": candidate_summary["pre_suppression_stage3_gradient_nonzero_fraction"],
        "final_auroc": {"control": control["ranking"]["final"]["auroc"], "candidate": candidate["ranking"]["final"]["auroc"], "delta": endpoint["candidate_minus_control"]["final_auroc"]},
        "final_ap": {"control": control["ranking"]["final"]["ap"], "candidate": candidate["ranking"]["final"]["ap"], "delta": endpoint["candidate_minus_control"]["final_ap"]},
        "positive_deltas": {"mean": endpoint["candidate_minus_control"]["positive_mean"], "median": endpoint["candidate_minus_control"]["positive_median"]},
        "interior_deltas": {"mean": endpoint["candidate_minus_control"]["interior_mean"], "median": endpoint["candidate_minus_control"]["interior_median"]},
        "near_background_deltas": {"p95": endpoint["candidate_minus_control"]["near_background_p95"], "p99": endpoint["candidate_minus_control"]["near_background_p99"]},
        "inversion_deltas": {"near_background_gt_positive": endpoint["candidate_minus_control"]["near_background_gt_positive_inversion"], "near_background_gt_interior": endpoint["candidate_minus_control"]["near_background_gt_interior_inversion"]},
        "stage_metric_deltas": {key: value for key, value in endpoint["candidate_minus_control"].items() if key.startswith("stage")},
        "numerical_validity": "PASS" if numerical_validity else "FAIL",
        "mechanism_test": mechanism_test,
        "bounded_screen": bounded,
        "late_convlora_causal_hypothesis": hypothesis,
        "interpretation": interpretation,
        "soft_drift_governor_justified": governor,
        "direct_gates": direct,
        "scientific_gates": scientific,
        "stage_metrics_and_geometry_are_reporting_only": True,
        "no_target_inference": True,
        "waiting_for_user_approval": "YES",
    }
    json_dump(REPO / "audit/H2_LATE_CONVLORA_FREEZE_R1_DECISION.json", result)
    lines = [
        f"# {PROTOCOL_ID} decision",
        "",
        "## Decision",
        "",
        f"- `MECHANISM_TEST={mechanism_test}`",
        f"- `BOUNDED_SCREEN={bounded}`",
        f"- `LATE_CONVLORA_CAUSAL_HYPOTHESIS={hypothesis}`",
        f"- `{interpretation}`",
        f"- `SOFT_DRIFT_GOVERNOR_JUSTIFIED={governor}`",
        "",
        "The screen used the exact E10 Safe-Anchor full state, one matched 500-attempt VisA source stream, and equal inference fusion on the frozen 96-image endpoint cohort. Stage metrics and geometry are reported diagnostics, not hidden gates.",
        "",
        "## Direct mechanism gates",
        "",
    ]
    lines.extend(f"- `{key}={value}`" for key, value in direct.items())
    lines.extend(["", "## Scientific gates", ""])
    lines.extend(f"- `{key}={value}`" for key, value in scientific.items())
    lines.extend([
        "",
        "## Endpoint deltas (candidate minus control)",
        "",
        f"- AUROC: `{endpoint['candidate_minus_control']['final_auroc']}`",
        f"- AP: `{endpoint['candidate_minus_control']['final_ap']}`",
        f"- Positive mean / median: `{endpoint['candidate_minus_control']['positive_mean']}` / `{endpoint['candidate_minus_control']['positive_median']}`",
        f"- Interior mean / median: `{endpoint['candidate_minus_control']['interior_mean']}` / `{endpoint['candidate_minus_control']['interior_median']}`",
        f"- Near-background p95 / p99: `{endpoint['candidate_minus_control']['near_background_p95']}` / `{endpoint['candidate_minus_control']['near_background_p99']}`",
        f"- Inversion near-background > positive / interior: `{endpoint['candidate_minus_control']['near_background_gt_positive_inversion']}` / `{endpoint['candidate_minus_control']['near_background_gt_interior_inversion']}`",
        "",
        "No follow-up trust-region/governor run or target evaluation is authorized by this artifact. `WAITING_FOR_USER_APPROVAL=YES`.",
        "",
    ])
    (REPO / "results/H2_LATE_CONVLORA_FREEZE_R1_DECISION.md").parent.mkdir(parents=True, exist_ok=True)
    (REPO / "results/H2_LATE_CONVLORA_FREEZE_R1_DECISION.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
