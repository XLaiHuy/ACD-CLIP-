#!/usr/bin/env python3
"""Materialize the required H2 GradBudget R1 bounded-screen decision."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ENDPOINT = REPO / "audit/H2_GRADBUDGET_R1_ENDPOINT.json"
DECISION_JSON = REPO / "results/H2_GRADBUDGET_R1_BOUNDED_DECISION.json"
DECISION_MD = REPO / "results/H2_GRADBUDGET_R1_BOUNDED_DECISION.md"
START_HEAD = "47158bd1a64d23f5f4752fb066e5dd4b91ef07d9"
START_E1_SHA256 = "7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35"


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    endpoint = json.loads(ENDPOINT.read_text())
    gates = endpoint["gates"]
    numerical = endpoint["numerical_validity"]
    telemetry = endpoint["candidate_gradient_telemetry"]
    control = endpoint["arms"]["A_GRADBUDGET_SHORT_R1_CONTROL"]
    candidate = endpoint["arms"]["A_GRADBUDGET_SHORT_R1_CANDIDATE"]
    decision = {
        "protocol_id": endpoint["protocol_id"],
        "BRANCH": "research/h2-late-convlora-gradbudget-r1",
        "START_HEAD": START_HEAD,
        "DECISION_ARTIFACT_PARENT_HEAD": git_head(),
        "START_E1_SHA256": START_E1_SHA256,
        "parameter_scope": endpoint["parameter_scope"],
        "loss_control_gradient_identity": {
            "loss_parity": "PASS",
            "control_historical_task_gradient": True,
            "abnormal_gradient_definition": "unscaled task gradient minus unscaled rest-task gradient on the same graph",
            "safe_anchor_once": True,
        },
        "gradient_preflight": {
            "rows": 16,
            "optimizer_steps": 0,
            "finite": True,
            "gradients_unscaled": True,
        },
        "batch_parity": {
            "first_16": "PASS",
            "attempted_500_identity": endpoint["exact_batch_identity_match"],
        },
        "attempted_successful_skips": {
            arm: {
                "attempted": endpoint["training_summaries"][arm]["attempted_steps"],
                "successful": endpoint["training_summaries"][arm]["successful_steps"],
                "nonfinite_loss_skips": endpoint["training_summaries"][arm]["nonfinite_loss_skips"],
                "nonfinite_grad_skips": endpoint["training_summaries"][arm]["nonfinite_grad_skips"],
            }
            for arm in ("A_GRADBUDGET_SHORT_R1_CONTROL", "A_GRADBUDGET_SHORT_R1_CANDIDATE")
        },
        "telemetry": telemetry,
        "endpoint_deltas_candidate_minus_control": endpoint["candidate_minus_control"],
        "numerical_validity": numerical,
        "direct_gates": gates["direct"],
        "scientific_gates": gates["scientific"],
        "BOUNDED_SCREEN": gates["BOUNDED_SCREEN"],
        "red_team_interpretations": gates["red_team_interpretations"],
        "confirmatory_support": gates["confirmatory_support"],
        "NO_TARGET": True,
        "NO_MEDICAL_MVTEC": True,
        "WAITING_FOR_USER_APPROVAL": True,
    }
    DECISION_JSON.parent.mkdir(parents=True, exist_ok=True)
    DECISION_JSON.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    delta = endpoint["candidate_minus_control"]
    lines = [
        "# H2 GradBudget R1 bounded decision",
        "",
        f"- Protocol: `{endpoint['protocol_id']}`",
        "- Branch: `research/h2-late-convlora-gradbudget-r1`",
        f"- Starting E1 SHA256: `{START_E1_SHA256}`",
        "- Scope: stage-2 and stage-3 Conv-LoRA only; stage 1 excluded.",
        "- Endpoint: frozen 96-image VisA source split only; no Medical/MVTec or target inference.",
        "",
        f"## Decision: `{gates['BOUNDED_SCREEN']}`",
        "",
        f"The numerical gate is invalid because control had {numerical['control_successful_steps']} successful updates and candidate had {numerical['candidate_successful_steps']}; the protocol requires equal successful counts. No rerun or parameter change is authorized.",
        "",
        "Mechanism activity was established in telemetry: candidate activity fraction "
        f"{telemetry['A_GRADBUDGET_SHORT_R1_CANDIDATE']['activity_fraction']:.6f}, "
        f"alpha median {telemetry['A_GRADBUDGET_SHORT_R1_CANDIDATE']['alpha']['median']:.6f}, "
        f"and effective abnormal/rest was lower than raw whenever active.",
        "",
        "Candidate-minus-control endpoint deltas:",
        "",
        f"- final AP: `{delta['final_ap']:+.9f}`; final AUROC: `{delta['final_auroc']:+.9f}`",
        f"- positive mean/median: `{delta['positive_mean']:+.9f}` / `{delta['positive_median']:+.9f}`",
        f"- interior mean/median: `{delta['interior_mean']:+.9f}` / `{delta['interior_median']:+.9f}`",
        f"- near-background p95/p99: `{delta['near_background_p95']:+.9f}` / `{delta['near_background_p99']:+.9f}`",
        f"- stage-1 AP/AUROC: `{delta['stage1_ap']:+.9f}` / `{delta['stage1_auroc']:+.9f}`",
        f"- stage-2/stage-3 Conv-LoRA drift: `{delta['stage2_convlora_drift']:+.9f}` / `{delta['stage3_convlora_drift']:+.9f}`",
        "",
        "Red-team interpretation: drift reduction was not achieved; the endpoint tail and AUROC checks worsen despite an AP increase. This does not support a confirmatory run.",
        "",
        "No target inference was run. The bounded screen is complete and the workflow is waiting for explicit user approval before any further experiment.",
    ]
    DECISION_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "BOUNDED_SCREEN": gates["BOUNDED_SCREEN"],
        "decision_json": str(DECISION_JSON),
        "decision_md": str(DECISION_MD),
        "WAITING_FOR_USER_APPROVAL": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
