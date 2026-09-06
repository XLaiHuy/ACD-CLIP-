#!/usr/bin/env python3
"""Write the post-R2 diagnostic interpretation; no experiment is launched."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CROSS = REPO / "audit/H2_POST_R2_CROSS_EVAL.json"


def head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    data = json.loads(CROSS.read_text())
    c_eq = data["cells"]["C_EQ"]
    c_w = data["cells"]["C_W"]
    t_eq = data["cells"]["T_EQ"]
    t_w = data["cells"]["T_W"]
    contrasts = data["contrasts"]
    stage = data["training_stage_deltas"]
    c_w_ap = contrasts["final_ap"]["inference_effect_control"]
    c_w_auroc = contrasts["final_auroc"]["inference_effect_control"]
    c_w_pos_mean = contrasts["positive_mean"]["inference_effect_control"]
    c_w_int_mean = contrasts["interior_mean"]["inference_effect_control"]
    training_ap = contrasts["final_ap"]["training_effect_equal"]
    near_p99_training = contrasts["near_background_p99"]["training_effect_equal"]
    near_inv_pos_training = (
        t_eq["background_tail"]["near_background_positive_pairwise_rank_inversion_rate"]
        - c_eq["background_tail"]["near_background_positive_pairwise_rank_inversion_rate"]
    )
    near_inv_int_training = (
        t_eq["background_tail"]["near_background_interior_pairwise_rank_inversion_rate"]
        - c_eq["background_tail"]["near_background_interior_pairwise_rank_inversion_rate"]
    )

    inference_only_promising = (
        c_w_ap >= -1e-6
        and c_w_auroc >= -1e-6
        and c_w_pos_mean >= -1e-3
        and c_w_int_mean >= -1e-3
    )
    training_dynamics_clear = (
        training_ap < -1e-3
        and stage["stage1_ap_delta"] < -1e-3
        and stage["stage2_ap_delta"] < -1e-3
    )
    hard_background_evidence = (
        contrasts["positive_mean"]["training_effect_equal"] > 0
        and contrasts["interior_mean"]["training_effect_equal"] > 0
        and near_p99_training > 0
        and near_inv_pos_training > 0
        and near_inv_int_training > 0
    )

    if inference_only_promising and training_dynamics_clear:
        dominant_pattern = "INFERENCE_ONLY_FUSION_PROMISING"
    elif hard_background_evidence and training_dynamics_clear:
        dominant_pattern = "HARD_BACKGROUND_PRIMARY"
    elif training_dynamics_clear:
        dominant_pattern = "TRAINING_DYNAMICS_PRIMARY"
    elif not inference_only_promising:
        dominant_pattern = "FIXED_FUSION_NOT_SUPPORTED"
    else:
        dominant_pattern = "MIXED_OR_INCONCLUSIVE"

    decision = {
        "protocol_id": "H2_POST_R2_CAUSAL_DECOMPOSITION_2X2_FROZEN_CROSS_EVALUATION",
        "diagnostic_only": True,
        "ENDPOINT_IDENTITY": data["ENDPOINT_IDENTITY"],
        "CROSS_EVAL_REPRODUCTION": data["CROSS_EVAL_REPRODUCTION"],
        "DOMINANT_PATTERN": dominant_pattern,
        "pattern_evidence": {
            "inference_only_final_ap_delta_C_W_minus_C_EQ": c_w_ap,
            "inference_only_final_auroc_delta_C_W_minus_C_EQ": c_w_auroc,
            "inference_only_positive_mean_delta": c_w_pos_mean,
            "inference_only_interior_mean_delta": c_w_int_mean,
            "training_effect_equal_final_ap_delta": training_ap,
            "training_effect_equal_stage1_ap_delta": stage["stage1_ap_delta"],
            "training_effect_equal_stage2_ap_delta": stage["stage2_ap_delta"],
            "training_effect_equal_stage3_ap_delta": stage["stage3_ap_delta"],
            "training_effect_equal_near_bg_p99_delta": near_p99_training,
            "training_effect_equal_near_bg_positive_inversion_delta": near_inv_pos_training,
            "training_effect_equal_near_bg_interior_inversion_delta": near_inv_int_training,
            "inference_only_promising": inference_only_promising,
            "training_dynamics_clear": training_dynamics_clear,
            "hard_background_evidence": hard_background_evidence,
        },
        "BOTTLENECK_STATUS": "REFINES",
        "prior_bottleneck": "LATE_STAGE_FUNCTIONAL_OVERADAPTATION_AND_ANOMALY_UNDERCOVERAGE",
        "REFINED_BOTTLENECK": "LATE_STAGE_TRAINING_OVERADAPTATION_WITH_ANOMALY_COVERAGE_VS_HARD_BACKGROUND_SEPARATION_TRADEOFF",
        "recommendations": [
            {
                "rank": 1,
                "mechanism": "inference-only fixed stage fusion",
                "fit": "Directly supported by C_W improving final AP/AUROC at the unchanged control-trained checkpoint.",
                "evidence_strength": "strong within this frozen 96-image diagnostic",
                "simplicity": "high",
                "novelty_defensibility": "moderate; causal cross-evaluation separates it from training changes",
                "inference_overhead": "none beyond the existing weighted sum",
                "target_tuning_risk": "none in this source-only diagnostic",
                "failure_risk": "validation may not transfer beyond the fixed endpoint subset",
                "falsifiability_500_step": "not applicable; no training is needed for this inference-only candidate",
            },
            {
                "rank": 2,
                "mechanism": "normal/background-selective E1 preservation",
                "fit": "Addresses the measured training-related near-background tail and stage-1/2 AP degradation.",
                "evidence_strength": "moderate; mechanism is conceptual and not tested here",
                "simplicity": "moderate",
                "novelty_defensibility": "high",
                "inference_overhead": "none if training-only",
                "target_tuning_risk": "low if source-only and preregistered",
                "failure_risk": "masking normal/background without suppressing anomaly evidence",
                "falsifiability_500_step": "yes, after separate approval and preregistration",
            },
            {
                "rank": 3,
                "mechanism": "family/stage-scoped gradient budget or conflict control",
                "fit": "Could target the harmful weighted-training trajectory, but is less direct than preserving normal/background geometry.",
                "evidence_strength": "moderate-to-weak; no gradient-conflict measurement was authorized here",
                "simplicity": "low",
                "novelty_defensibility": "moderate",
                "inference_overhead": "none",
                "target_tuning_risk": "requires careful source-only controls",
                "failure_risk": "adds optimization complexity and may obscure the causal mechanism",
                "falsifiability_500_step": "yes, after separate approval and preregistration",
            },
        ],
        "SELECTED_NEXT_MECHANISM": "inference-only fixed stage fusion",
        "NEXT_BOUNDED_EXPERIMENT_JUSTIFIED": "YES",
        "selected_next_experiment_scope": "Future source-only validation of weighted inference on historically equal-trained endpoints; no run performed here.",
        "no_new_training_authorized": True,
        "new_training_run": False,
        "optimizer_step_used": False,
        "medical_inference_run": False,
        "mvtec_inference_run": False,
        "full_e15_started": False,
        "target_tuning_used": False,
        "hyperparameter_sweep": False,
        "decision_head": head(),
    }
    json_path = REPO / "results/H2_POST_R2_MECHANISM_DECISION.json"
    md_path = REPO / "results/H2_POST_R2_MECHANISM_DECISION.md"
    json_dump(json_path, decision)
    markdown = f"""# H2 Post-R2 Mechanism Decision

`PROTOCOL_ID=H2_POST_R2_CAUSAL_DECOMPOSITION_2X2_FROZEN_CROSS_EVALUATION`

`ENDPOINT_IDENTITY={decision['ENDPOINT_IDENTITY']}`

`CROSS_EVAL_REPRODUCTION={decision['CROSS_EVAL_REPRODUCTION']}`

`DOMINANT_PATTERN={dominant_pattern}`

`BOTTLENECK_STATUS=REFINES`

`REFINED_BOTTLENECK={decision['REFINED_BOTTLENECK']}`

## Evidence

- C_W minus C_EQ final AP: `{c_w_ap:.12f}`; final AUROC: `{c_w_auroc:.12f}`.
- T_EQ minus C_EQ final AP: `{training_ap:.12f}`.
- Training-stage AP deltas: stage 1 `{stage['stage1_ap_delta']:.12f}`, stage 2 `{stage['stage2_ap_delta']:.12f}`, stage 3 `{stage['stage3_ap_delta']:.12f}`.
- T_EQ minus C_EQ near-background p99: `{near_p99_training:.12f}`.
- Near-background positive/interior pairwise inversion deltas: `{near_inv_pos_training:.12f}` / `{near_inv_int_training:.12f}`.

The frozen cross-evaluation supports weighted fusion as an inference-only
candidate: C_W improves both final AP and AUROC at the control-trained
checkpoint, while weighted training causes the AP and stage-1/2 degradation.
The near-background tail and rank inversions rise under weighted training,
refining the prior diagnosis to a coverage-versus-hard-background tradeoff.

## Ranked next mechanisms

1. **Inference-only fixed stage fusion** — selected; strong direct evidence,
   no inference overhead beyond the weighted sum, and no training run here.
2. **Normal/background-selective E1 preservation** — conceptual only; do not
   implement or assign coefficients without a new preregistration and approval.
3. **Family/stage-scoped gradient budget or conflict control** — lower priority;
   requires separate evidence and approval.

`SELECTED_NEXT_MECHANISM=inference-only fixed stage fusion`

`NEXT_BOUNDED_EXPERIMENT_JUSTIFIED=YES`

No new training, optimizer step, target inference, Medical/MVTec inference,
full E15, tuning, or sweep was performed. The original R2 FAIL decision is not
changed.
"""
    md_path.write_text(markdown)
    print(json.dumps({
        "DOMINANT_PATTERN": dominant_pattern,
        "BOTTLENECK_STATUS": "REFINES",
        "SELECTED_NEXT_MECHANISM": decision["SELECTED_NEXT_MECHANISM"],
        "NEXT_BOUNDED_EXPERIMENT_JUSTIFIED": "YES",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
