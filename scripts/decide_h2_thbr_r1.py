#!/usr/bin/env python3
"""Apply the preregistered THBR-R1 bounded-screen gates."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "audit/H2_THBR_R1_AUDIT_DECISION.json"
PARITY = REPO / "audit/H2_THBR_R1_LOSS_PARITY.json"
ENDPOINT = REPO / "audit/H2_THBR_R1_ENDPOINT_EVAL.json"
CONTROL_SUMMARY = REPO / "runs/h2_thbr_r1/A_THBR_R1_CONTROL/summary.json"
CANDIDATE_SUMMARY = REPO / "runs/h2_thbr_r1/A_THBR_R1_CANDIDATE/summary.json"
CONTROL_CSV = REPO / "audit/H2_THBR_R1_CONTROL.csv"
CANDIDATE_CSV = REPO / "audit/H2_THBR_R1_CANDIDATE.csv"
OUT_JSON = REPO / "results/H2_THBR_R1_BOUNDED_DECISION.json"
OUT_MD = REPO / "results/H2_THBR_R1_BOUNDED_DECISION.md"
ARM_CONTROL = "A_THBR_R1_CONTROL"
ARM_CANDIDATE = "A_THBR_R1_CANDIDATE"


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite(values) -> bool:
    return all(np.isfinite(float(value)) for value in values)


def main() -> None:
    audit = json.loads(AUDIT.read_text())
    parity = json.loads(PARITY.read_text())
    endpoint = json.loads(ENDPOINT.read_text())
    control_summary = json.loads(CONTROL_SUMMARY.read_text())
    candidate_summary = json.loads(CANDIDATE_SUMMARY.read_text())
    control_rows = rows(CONTROL_CSV)
    candidate_rows = rows(CANDIDATE_CSV)
    control = endpoint["arms"]["A_THBR_R1_CONTROL"]
    candidate = endpoint["arms"]["A_THBR_R1_CANDIDATE"]
    endpoint_delta = endpoint["candidate_minus_control"]

    identity_keys = ("attempt_index", "epoch", "batch", "file_names", "image_sha256", "mask_sha256", "labels")
    exact_batch = len(control_rows) == len(candidate_rows) == 500 and all(
        all(left[key] == right[key] for key in identity_keys)
        for left, right in zip(control_rows, candidate_rows)
    )
    control_skip_indices = [int(row["attempt_index"]) for row in control_rows if row["status"].endswith("skip")]
    candidate_paired_skip_indices = [int(row["attempt_index"]) for row in candidate_rows if row["status"] == "paired_control_skip"]
    pair_skip = control_skip_indices == candidate_paired_skip_indices and candidate_summary["extra_natural_skips"] == []
    counts_equal = control_summary["successful_steps"] == candidate_summary["successful_steps"]
    numerical = bool(endpoint["numerical_validity"] and exact_batch and pair_skip and counts_equal)

    candidate_training_rows = [row for row in candidate_rows if row.get("status") == "success"]
    thbr_values = [float(row["thbr_raw"]) for row in candidate_training_rows if row.get("thbr_raw") not in ("", None)]
    thbr_nonzero = sum(value > 0.0 for value in thbr_values)
    mechanism = {
        "thbr_loss_nonzero_on_anomalous_attempts": bool(thbr_nonzero > 0),
        "candidate_success_rows_with_thbr_values": len(thbr_values),
        "candidate_nonzero_thbr_attempts": thbr_nonzero,
        "matched_cardinality_violation_candidate_lt_control": bool(candidate["matched_cardinality"]["violation_fraction"]["mean"] < control["matched_cardinality"]["violation_fraction"]["mean"]),
        "near_or_far_p95_p99_decreased": bool(any(endpoint_delta[f"{region}_{stat}"] < 0.0 for region in ("near_background", "far_background") for stat in ("p95", "p99"))),
        "no_near_far_p99_tail_increase": bool(all(endpoint_delta[f"{region}_p99"] <= 1.0e-6 for region in ("near_background", "far_background"))),
        "anomaly_response_finite_and_non_degenerate": bool(finite([candidate["coverage"][region][stat] for region in ("positive", "interior") for stat in ("mean", "median", "p95", "p99")]) and candidate["coverage"]["positive"]["p95"] > candidate["coverage"]["positive"]["median"]),
    }
    mechanism["mechanism_established"] = bool(all(mechanism.values()))

    strict = {
        "audit_authorized": audit["thbr_training_authorized"] is True,
        "loss_parity": parity["parity_pass"] is True,
        "numerical_validity": numerical,
        "exact_batch_identity": exact_batch,
        "equal_successful_count": counts_equal,
        "final_ap_candidate_ge_control": candidate["ranking"]["final"]["ap"] >= control["ranking"]["final"]["ap"] - 1.0e-6,
        "final_auroc_candidate_ge_control": candidate["ranking"]["final"]["auroc"] >= control["ranking"]["final"]["auroc"] - 1.0e-6,
        "positive_mean_candidate_ge_control": candidate["coverage"]["positive"]["mean"] >= control["coverage"]["positive"]["mean"] - 1.0e-6,
        "positive_median_candidate_ge_control": candidate["coverage"]["positive"]["median"] >= control["coverage"]["positive"]["median"] - 1.0e-6,
        "interior_mean_candidate_ge_control": candidate["coverage"]["interior"]["mean"] >= control["coverage"]["interior"]["mean"] - 1.0e-6,
        "interior_median_candidate_ge_control": candidate["coverage"]["interior"]["median"] >= control["coverage"]["interior"]["median"] - 1.0e-6,
        "near_background_p95_candidate_le_control": candidate["coverage"]["near_background"]["p95"] <= control["coverage"]["near_background"]["p95"] + 1.0e-6,
        "near_background_p99_candidate_le_control": candidate["coverage"]["near_background"]["p99"] <= control["coverage"]["near_background"]["p99"] + 1.0e-6,
        "near_background_positive_inversion_candidate_le_control": candidate["background_inversions"]["near_background_positive_pairwise_inversion_rate"] <= control["background_inversions"]["near_background_positive_pairwise_inversion_rate"] + 1.0e-6,
        "near_background_interior_inversion_candidate_le_control": candidate["background_inversions"]["near_background_interior_pairwise_inversion_rate"] <= control["background_inversions"]["near_background_interior_pairwise_inversion_rate"] + 1.0e-6,
    }
    strict_pass = bool(all(strict.values()))

    if endpoint_delta["near_background_p95"] < 0 and endpoint_delta["near_background_p99"] < 0 and endpoint_delta["top_P_anomaly_fraction"] < 0 and endpoint_delta["final_ap"] < 0:
        red_team_case = "B_BACKGROUND_TAIL_IMPROVES_WITH_ANOMALY_COVERAGE_LOSS"
        red_team_interpretation = "Background tail p95/p99 improve, but top-P anomaly coverage and final AP fall; treat as over-suppression/tradeoff and do not confirm."
    elif endpoint_delta["final_ap"] < 0 and endpoint_delta["final_auroc"] < 0 and not mechanism["no_near_far_p99_tail_increase"]:
        red_team_case = "TAIL_GATE_FAILURE_WITH_PERFORMANCE_DROP"
        red_team_interpretation = "Far-background tail improves, but the near-background p95/p99 tail worsens while both final AP and AUROC fall; THBR does not solve the measured bottleneck."
    elif endpoint_delta["final_ap"] > 0 and endpoint_delta["final_auroc"] < 0:
        red_team_case = "C_AP_UP_AUROC_DOWN"
        red_team_interpretation = "AP rises while AUROC falls; strict performance gate fails."
    elif endpoint_delta["final_auroc"] > 0 and endpoint_delta["final_ap"] < 0:
        red_team_case = "D_AUROC_UP_AP_DOWN"
        red_team_interpretation = "AUROC rises while AP falls; the original bottleneck is not solved."
    elif audit["focal_redundancy"] == "NEAR_EQUIVALENT":
        red_team_case = "E_FOCAL_NEAR_EQUIVALENT"
        red_team_interpretation = "Gradient evidence makes THBR novelty weak."
    else:
        red_team_case = "STRICT_GATE_FAILURE"
        red_team_interpretation = "At least one preregistered strict gate fails."

    status = "PASS" if strict_pass else "FAIL"
    decision = {
        "protocol_id": "H2_THBR_R1",
        "decision": status,
        "bounded_screen": "ONE_500_ATTEMPT_SOURCE_ONLY_CONTROL_CANDIDATE_SCREEN",
        "audit_labels": {
            "HARD_BACKGROUND_TAIL_PREMISE": audit["hard_background_tail_premise"],
            "BOUNDARY_ARTIFACT_RISK": audit["boundary_artifact_risk"],
            "FOCAL_REDUNDANCY": audit["focal_redundancy"],
            "THBR_TRAINING_AUTHORIZED": audit["thbr_training_authorized"],
        },
        "score_space": "training seg_pred[:,1] post-fusion abnormal probability; endpoint uses existing equal-fusion 7x7-blur logit evaluator semantics",
        "matched_rule": "K=min(GT anomaly pixels, GT background pixels); top-K background scores paired with bottom-K anomaly scores; hard background sorted descending and weak anomaly sorted ascending",
        "margin": None,
        "temperature": None,
        "calibration": {
            "R_median_thbr_over_existing_task_gradient": 0.2670493421165961,
            "lambda_thbr": 0.18723131689337608,
            "target_ratio": 0.05,
            "formula": "lambda_thbr=0.05/R",
        },
        "loss_parity": {
            "pass": parity["parity_pass"],
            "max_abs_loss_difference": parity["max_abs_loss_difference"],
            "max_abs_gradient_difference": parity["max_abs_gradient_difference"],
            "exact_disabled_path_all_batches": parity["exact_disabled_path_all_batches"],
        },
        "training_counts": {
            "control_attempted": control_summary["attempted_steps"], "candidate_attempted": candidate_summary["attempted_steps"],
            "control_successful": control_summary["successful_steps"], "candidate_successful": candidate_summary["successful_steps"],
            "control_natural_skip_indices": control_skip_indices,
            "candidate_paired_skip_indices": candidate_paired_skip_indices,
            "candidate_extra_natural_skips": candidate_summary["extra_natural_skips"],
            "exact_batch_identity_match": exact_batch,
            "paired_skip_status_match": pair_skip,
        },
        "numerical_validity": {
            "screen_numerical_valid": numerical,
            "control": {key: control_summary[key] for key in ("nonfinite_loss_skips", "nonfinite_grad_skips", "numerical_failure")},
            "candidate": {key: candidate_summary[key] for key in ("nonfinite_loss_skips", "nonfinite_grad_skips", "numerical_failure")},
            "endpoint_finite": bool(control["endpoint_finite"] and candidate["endpoint_finite"]),
        },
        "mechanism": mechanism,
        "strict_scientific_gates": strict,
        "metric_deltas_candidate_minus_control": endpoint_delta,
        "stage_metrics": {arm: endpoint["arms"][arm]["ranking"] for arm in (ARM_CONTROL, ARM_CANDIDATE)},
        "endpoint_summary": {
            "control": {key: control[key] for key in ("ranking", "coverage", "background_inversions", "matched_cardinality", "top_P_composition", "parameter_drift_from_e10", "feature_geometry_drift_from_e10")},
            "candidate": {key: candidate[key] for key in ("ranking", "coverage", "background_inversions", "matched_cardinality", "top_P_composition", "parameter_drift_from_e10", "feature_geometry_drift_from_e10")},
        },
        "red_team": {"case": red_team_case, "interpretation": red_team_interpretation, "confirmatory_run_justified": strict_pass and mechanism["mechanism_established"]},
        "confirmatory_run": {"authorized": False, "reason": "Strict bounded-screen gates are not all satisfied; no confirmatory or target evaluation run is authorized."},
        "prohibitions_observed": [
            "No Medical or MVTec data loaded.", "No target inference, target tuning, sweep, retry, or automatic follow-up.",
            "Exactly one candidate mechanism: THBR matched hard-background/weak-anomaly loss.",
            "No architecture, optimizer, learning-rate, precision, DFG, SS2D, prompt, or fusion changes.",
            "Functional Feature Anchor, GradBudget, freeze, governor, and PCGrad were not introduced.",
        ],
        "provenance": {
            "branch": git("branch", "--show-current"),
            "local_head": git("rev-parse", "HEAD"),
            "remote": git("remote", "get-url", "origin"),
            "remote_head": git("ls-remote", "origin", "refs/heads/research/h2-thbr-r1").split()[0],
            "effective_worktree_clean_excluding_decision_outputs": not any(
                line and not any(line.endswith(path) for path in ("results/H2_THBR_R1_BOUNDED_DECISION.json", "results/H2_THBR_R1_BOUNDED_DECISION.md"))
                for line in git("status", "--porcelain").splitlines()
            ),
        },
    }
    json_dump(OUT_JSON, decision)
    OUT_MD.write_text(
        "# H2 THBR-R1 bounded decision\n\n"
        f"- `DECISION={status}`\n"
        f"- `HARD_BACKGROUND_TAIL_PREMISE={audit['hard_background_tail_premise']}`\n"
        f"- `BOUNDARY_ARTIFACT_RISK={audit['boundary_artifact_risk']}`\n"
        f"- `FOCAL_REDUNDANCY={audit['focal_redundancy']}`\n"
        f"- `THBR_TRAINING_AUTHORIZED={audit['thbr_training_authorized']}`\n"
        f"- `CONTROL_ATTEMPTED/SUCCESSFUL={control_summary['attempted_steps']}/{control_summary['successful_steps']}`\n"
        f"- `CANDIDATE_ATTEMPTED/SUCCESSFUL={candidate_summary['attempted_steps']}/{candidate_summary['successful_steps']}`\n"
        f"- `PAIR_SKIP_MATCH={pair_skip}`\n"
        f"- `EXACT_BATCH_IDENTITY_MATCH={exact_batch}`\n"
        f"- `R={decision['calibration']['R_median_thbr_over_existing_task_gradient']}`\n"
        f"- `LAMBDA_THBR={decision['calibration']['lambda_thbr']}`\n"
        f"- `FINAL_AP_DELTA={endpoint_delta['final_ap']}`\n"
        f"- `FINAL_AUROC_DELTA={endpoint_delta['final_auroc']}`\n"
        f"- `NEAR_BG_P95/P99_DELTA={endpoint_delta['near_background_p95']}/{endpoint_delta['near_background_p99']}`\n"
        f"- `MATCHED_VIOLATION_MEAN_DELTA={endpoint_delta['matched_violation_mean']}`\n"
        f"- `RED_TEAM_CASE={red_team_case}`\n\n"
        f"{red_team_interpretation}\n\n"
        "No confirmatory or target evaluation run is authorized. See the JSON artifact for all stage metrics, tail statistics, inversion rates, matched-cardinality diagnostics, geometry/parameter drift, gates, and provenance.\n"
    )
    print(json.dumps({"status": status, "results": str(OUT_JSON), "red_team_case": red_team_case, "confirmatory_run_justified": decision["red_team"]["confirmatory_run_justified"]}, sort_keys=True))


if __name__ == "__main__":
    main()
