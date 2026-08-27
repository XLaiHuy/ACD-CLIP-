from __future__ import annotations

import json
from pathlib import Path

from tools.sabra_v2 import run_p34_scientific_stage2 as p34


ROOT = Path(__file__).resolve().parents[1]
P34_ROOT = ROOT / "research/sabra_v2/region_distill/P34"


def _qualification() -> dict:
    return json.loads((P34_ROOT / "P34_STAGE2_QUALIFICATION.json").read_text(encoding="utf-8"))


def test_p34_reporting_schema_is_validated_against_frozen_source_fields() -> None:
    preflight = json.loads(
        (ROOT / "research/sabra_v2/region_distill/P34_PREFLIGHT_FALSIFICATION.json").read_text(
            encoding="utf-8"
        )
    )
    schema = p34._validate_reporting_source_schema(preflight)
    assert schema["source_key"] == "source_only"
    assert schema["exact_counts_key"] == "exact_counts"
    assert "target_exact_zero_fraction" in schema["required_exact_fields"]
    assert "weight_one_fraction" in schema["required_exact_fields"]


def test_p34_final_report_accepts_actual_qualification_schema_without_legacy_key() -> None:
    qualification = _qualification()
    diagnostics = qualification["diagnostics"]
    assert "source_only_actionability" in diagnostics
    assert "source_only" not in diagnostics

    report = p34._final_report(
        qualification["status"],
        qualification["attempt"],
        qualification["training"],
        qualification["prediction"],
        qualification["metrics"],
        diagnostics,
        qualification["comparison"],
        qualification["gate"],
        qualification["post_run_audit"],
    )

    assert "source-only shaped target" in report
    assert "KeyError" not in report


def test_p34_gate_represents_forbidden_rerun_as_a_pass_condition() -> None:
    attempt_uuid = "frozen-mock-p34-attempt"
    prereg = {
        "future_scientific_gates": {
            "pap_minimum": 0.0,
            "pauroc_minimum": 0.0,
            "global_residual_abs_q99_max": 10.0,
            "normal_score_effect_q99_shift_max": 1.0,
            "nonfinite_loss_count": 0,
            "nonfinite_gradient_count": 0,
            "active_fraction_max_relative_to_p33": 0.9,
            "effective_support_fraction_max_relative_to_p33": 0.9,
            "gini_min_relative_to_p33": 0.1,
        }
    }
    training = {
        "optimizer_steps": p34.P34_EXPECTED_STEPS,
        "objective_count": 1,
        "student_parameter_delta": {"l2": 1.0},
        "teacher_parameter_delta": 0.0,
        "teacher_detached": True,
        "weight_detached": True,
        "target_detached": True,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
        "nonfinite_loss_count": 0,
        "nonfinite_gradient_count": 0,
    }
    prediction = {"attempt_uuid": attempt_uuid, "gt_used": False, "mask_reads": 0}
    freeze = {"attempt_uuid": attempt_uuid, "predictions_frozen": True}
    metrics = {"attempt_uuid": attempt_uuid, "p34_metrics": {"pAP": 1.0, "pAUROC": 1.0}}
    diagnostics = {
        "p34_minus_native_pixel_shift": {"normal": {"q99": 0.0}},
        "residual": {"q99_abs": 1.0},
        "residual_support": {"active_fraction": 0.1, "effective_support_fraction": 0.1, "gini": 0.9},
    }
    input_audit = {
        "held_gt_reads_before_prediction_freeze": 0,
        "held_mask_reads_before_prediction_freeze": 0,
        "held_outcome_metrics_read_before_prediction_freeze": False,
        "cache_rebuilt": False,
    }

    gate = p34._scientific_gate(
        prereg,
        training,
        prediction,
        freeze,
        metrics,
        diagnostics,
        input_audit,
        attempt_uuid,
    )

    assert gate["structural_checks"]["automatic_rerun_forbidden"] is True
    assert "automatic_rerun" not in gate["failures"]
