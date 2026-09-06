#!/usr/bin/env python3
"""Apply the preregistered R2 endpoint gates and write the final decision."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ENDPOINT = REPO / "audit/H2_FUSION_R2_ENDPOINT_EVAL.json"
PARITY = REPO / "audit/H2_FUSION_R2_IMPLEMENTATION_PARITY.json"
PREFLIGHT = REPO / "audit/H2_FUSION_R2_PREFLIGHT.json"
SPLIT = REPO / "audit/H2_FUSION_SPLIT_IDENTITY.json"
START = REPO / "runs/h2_clean_factorial_e20_20260902_ampfix/shared_e1/adapter_1.pth"


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def main() -> None:
    endpoint = json.loads(ENDPOINT.read_text())
    parity = json.loads(PARITY.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    split = json.loads(SPLIT.read_text())
    control = endpoint["arms"]["A_FUSE_SHORT_R2_CONTROL"]
    candidate = endpoint["arms"]["A_FUSE_SHORT_R2_CANDIDATE"]
    delta = endpoint["candidate_minus_control"]
    summaries = endpoint["training_summaries"]

    gates = {
        "numerical_validity": bool(endpoint["numerical_validity"]),
        "exact_batch_identity_match": bool(endpoint["exact_batch_identity_match"]),
        "R_late_lower": candidate["R_late"]["mean"] < control["R_late"]["mean"],
        "final_ap_noninferior": delta["final_ap_delta"] >= -1e-6,
        "final_auroc_noninferior": delta["final_auroc_delta"] >= -1e-6,
        "positive_mean_noninferior": delta["positive_mean_delta"] >= -1e-6,
        "positive_median_noninferior": delta["positive_median_delta"] >= -1e-6,
        "interior_mean_noninferior": delta["interior_mean_delta"] >= -1e-6,
        "interior_median_noninferior": delta["interior_median_delta"] >= -1e-6,
        "stage1_ap_noninferior": delta["stage1_ap_delta"] >= -1e-6,
        "stage1_auroc_noninferior": delta["stage1_auroc_delta"] >= -1e-6,
    }
    scientific_gate = all(
        gates[name] for name in (
            "R_late_lower", "final_ap_noninferior", "final_auroc_noninferior",
            "positive_mean_noninferior", "positive_median_noninferior",
            "interior_mean_noninferior", "interior_median_noninferior",
            "stage1_ap_noninferior", "stage1_auroc_noninferior",
        )
    )
    if not gates["numerical_validity"]:
        bounded_screen = "INVALID_NUMERICAL"
        mechanism_status = "NO_SCIENTIFIC_MECHANISM_DECISION"
    elif scientific_gate:
        bounded_screen = "PASS"
        mechanism_status = "SUPPORTED_BY_R2"
    else:
        bounded_screen = "FAIL"
        mechanism_status = "NOT_SUPPORTED_BY_R2"

    decision = {
        "protocol_id": "H2_FIXED_E1_RELIABILITY_RESIDUAL_STAGE_FUSION_R2",
        "branch": "research/h2-fixed-fusion-bounded-r2",
        "start_checkpoint": str(START.relative_to(REPO)),
        "start_checkpoint_sha256": "7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35",
        "implementation_parity": parity["IMPLEMENTATION_PARITY"],
        "preflight": preflight["PREFLIGHT_BATCH_PARITY"],
        "split": {
            "calibration_count": split["calibration_count"],
            "endpoint_count": split["endpoint_eval_count"],
            "intersection_count": split["intersection_count"],
        },
        "arms": {
            arm: {
                "attempted_steps": summaries[arm]["attempted_steps"],
                "successful_steps": summaries[arm]["successful_steps"],
                "nonfinite_loss_skips": summaries[arm]["nonfinite_loss_skips"],
                "nonfinite_grad_skips": summaries[arm]["nonfinite_grad_skips"],
                "max_consecutive_nonfinite_grad_skips": summaries[arm]["max_consecutive_nonfinite_grad_skips"],
                "optimizer_state_failures": summaries[arm]["optimizer_state_failures"],
                "parameter_corruption": summaries[arm]["parameter_corruption"],
            }
            for arm in ("A_FUSE_SHORT_R2_CONTROL", "A_FUSE_SHORT_R2_CANDIDATE")
        },
        "exact_batch_identity_match": endpoint["exact_batch_identity_match"],
        "R_late": {
            "control_mean": control["R_late"]["mean"],
            "candidate_mean": candidate["R_late"]["mean"],
            "candidate_minus_control": delta["R_late_mean_delta"],
            "candidate_lower": gates["R_late_lower"],
        },
        "endpoint_metric_deltas_candidate_minus_control": {
            key: delta[key] for key in (
                "final_ap_delta", "final_auroc_delta", "positive_mean_delta",
                "positive_median_delta", "interior_mean_delta", "interior_median_delta",
                "stage1_ap_delta", "stage1_auroc_delta", "stage2_ap_delta", "stage3_ap_delta",
            )
        },
        "endpoint_absolute_metrics": {
            "control": {
                "final": control["ranking"]["final"],
                "stage1": control["ranking"]["stage_1"],
                "stage2": control["ranking"]["stage_2"],
                "stage3": control["ranking"]["stage_3"],
            },
            "candidate": {
                "final": candidate["ranking"]["final"],
                "stage1": candidate["ranking"]["stage_1"],
                "stage2": candidate["ranking"]["stage_2"],
                "stage3": candidate["ranking"]["stage_3"],
            },
        },
        "gates": gates,
        "scientific_gate": scientific_gate,
        "BOUNDED_SCREEN": bounded_screen,
        "MECHANISM_STATUS": mechanism_status,
        "FULL_E15_STARTED": False,
        "calibration_subset_used_for_pass_fail": False,
        "medical_evaluated": False,
        "mvtec_evaluated": False,
        "target_data_used": False,
        "next_full_e15": "NOT_STARTED; requires a new explicit user approval after this bounded R2 decision",
        "prohibitions_after_decision": [
            "Do not start full E15 from this result without new user approval.",
            "Do not reinterpret geometry alone as a mechanism pass.",
            "Do not use calibration or target data to revise this decision.",
        ],
        "decision_head": git_head(),
    }
    json_path = REPO / "results/H2_FUSION_R2_BOUNDED_DECISION.json"
    md_path = REPO / "results/H2_FUSION_R2_BOUNDED_DECISION.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    arm_lines = []
    for arm, values in decision["arms"].items():
        arm_lines.append(
            f"- `{arm}`: attempts={values['attempted_steps']}, successful={values['successful_steps']}, "
            f"nonfinite_loss_skips={values['nonfinite_loss_skips']}, "
            f"nonfinite_grad_skips={values['nonfinite_grad_skips']}, "
            f"max_consecutive_grad_skips={values['max_consecutive_nonfinite_grad_skips']}."
        )
    gate_lines = "\n".join(f"- `{name}`: {'PASS' if value else 'FAIL'}" for name, value in gates.items())
    md = f"""# H2 Fixed-Fusion R2 Bounded Decision

`PROTOCOL_ID=H2_FIXED_E1_RELIABILITY_RESIDUAL_STAGE_FUSION_R2`

`BOUNDED_SCREEN={bounded_screen}`

`MECHANISM_STATUS={mechanism_status}`

`NUMERICAL_VALIDITY={'PASS' if endpoint['numerical_validity'] else 'FAIL'}`

`FULL_E15_STARTED=NO`

## Execution identity

- Branch: `research/h2-fixed-fusion-bounded-r2`
- Start checkpoint: `{decision['start_checkpoint']}`
- Start SHA256: `{decision['start_checkpoint_sha256']}`
- Decision head: `{decision['decision_head']}`
- Implementation parity: `{decision['implementation_parity']}`
- 16-batch preflight: `{decision['preflight']}`
- Endpoint split: calibration `96`, endpoint `96`, intersection `0`
- Exact 500-attempt batch identity match: `{'PASS' if endpoint['exact_batch_identity_match'] else 'FAIL'}`

## Arm validity

{chr(10).join(arm_lines)}

## Mechanism and endpoint results

- R_late control mean: `{control['R_late']['mean']:.12f}`
- R_late candidate mean: `{candidate['R_late']['mean']:.12f}`
- Candidate minus control: `{delta['R_late_mean_delta']:.12f}` (lower: `{'PASS' if gates['R_late_lower'] else 'FAIL'}`)
- Final AP delta: `{delta['final_ap_delta']:.12f}`
- Final AUROC delta: `{delta['final_auroc_delta']:.12f}`
- Positive mean/median deltas: `{delta['positive_mean_delta']:.12f}` / `{delta['positive_median_delta']:.12f}`
- Interior mean/median deltas: `{delta['interior_mean_delta']:.12f}` / `{delta['interior_median_delta']:.12f}`
- Stage-1 AP/AUROC deltas: `{delta['stage1_ap_delta']:.12f}` / `{delta['stage1_auroc_delta']:.12f}`
- Stage-2 AP delta: `{delta['stage2_ap_delta']:.12f}`; stage-3 AP delta: `{delta['stage3_ap_delta']:.12f}`

## Preregistered gates

{gate_lines}

The candidate lowered the direct late-stage contribution ratio, but the final
AP and stage-1 noninferiority gates failed. Geometry remains descriptive and
does not override those failures.

Calibration was not used for endpoint pass/fail. Medical, MVTec, target data,
and full E15 were not evaluated.

Next full E15: not started; a new explicit user approval is required.
"""
    md_path.write_text(md)
    print(json.dumps({"BOUNDED_SCREEN": bounded_screen, "MECHANISM_STATUS": mechanism_status, "json": str(json_path), "markdown": str(md_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
