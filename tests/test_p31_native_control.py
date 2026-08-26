from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from tools.sabra_v2 import p31_native_control as control


def test_p31_is_an_offline_zero_objective_contract() -> None:
    source = inspect.getsource(control)
    assert "import torch" not in source
    assert "optimizer.step" not in source
    assert "RegionResidualAdapter" not in source
    assert control.PROTOCOL_ID == "P31"
    assert control.NON_INFERIORITY_MARGIN == 0.0
    parser = control.make_parser()
    assert parser.parse_args(["preflight", "--cache-root", "/cache"]).command == "preflight"
    assert parser.parse_args(["profile"]).command == "profile"
    with pytest.raises(SystemExit):
        parser.parse_args(["--held-class", "candle"])


def test_production_reference_identity_is_exact_and_independent() -> None:
    native = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    observed = control.native_control(native)
    reference = control.reference_native_control(native)
    assert np.array_equal(observed, native)
    assert np.array_equal(observed, reference)
    assert observed is not native
    assert not np.shares_memory(observed, native)
    assert observed.flags.writeable is False
    assert control.zero_residual_like(native).flags.writeable is False
    diagnostic = control.identity_diagnostic(native, observed)
    assert diagnostic["exact_equal"] is True
    assert diagnostic["independent_storage"] is True
    assert diagnostic["output_delta_l2"] == 0.0
    assert diagnostic["output_max_abs"] == 0.0
    assert diagnostic["objective_count"] == 0
    assert diagnostic["loss"] is None
    assert diagnostic["student_gradient_l2"] == 0.0


def test_identity_rejects_invalid_inputs_without_sanitizing() -> None:
    with pytest.raises(ValueError):
        control.native_control(np.asarray([0.0, np.nan], dtype=np.float32))
    with pytest.raises(ValueError):
        control.native_control(np.asarray(1.0, dtype=np.float32))
    with pytest.raises(TypeError):
        control.native_control(np.asarray(["native"], dtype=object))


def test_metric_comparison_locks_zero_margin_and_does_not_compute_held_metrics() -> None:
    result = control.compare_locked_metrics(
        {"pAP": 0.5, "pAUROC": 0.9},
        {"pAP": 0.49, "pAUROC": 0.9},
    )
    assert result["status"] == "NATIVE_CONTROL_SUPPORTED"
    assert result["differences_native_minus_comparator"]["pAP"] == pytest.approx(0.01)
    assert result["differences_native_minus_comparator"]["pAUROC"] == pytest.approx(0.0)
    assert result["held_metrics_computed"] is False
    with pytest.raises(ValueError):
        control.compare_locked_metrics({"pAP": 0.5, "pAUROC": 0.9}, {"pAP": 0.5, "pAUROC": 0.9}, margin=0.01)
    assert control.compare_locked_metrics(
        {"pAP": 0.49, "pAUROC": 0.9},
        {"pAP": 0.5, "pAUROC": 0.9},
    )["status"] == "NATIVE_CONTROL_FALSIFIED"


def test_synthetic_adversarial_suite_has_all_required_zero_behavior() -> None:
    result = control.synthetic_adversarial_suite()
    names = {row["case"] for row in result["cases"]}
    assert result["case_count"] == 15
    assert names == {
        "exact_zero",
        "near_zero",
        "normal_scale",
        "scale_0.01",
        "scale_0.1",
        "scale_1",
        "scale_10",
        "scale_100",
        "sign_flip",
        "sparse_1pct_corruption",
        "heavy_tail_corruption",
        "mixed_scale_batch",
        "one_extreme_outlier_sample",
        "all_null_no_intervention",
        "high_confidence_intervention",
    }
    assert result["all_finite"] is True
    assert result["all_output_deltas_exact_zero"] is True
    assert result["all_gradients_exact_zero"] is True
    assert result["all_batch_dominance_flags_false"] is True


def test_source_cache_audit_is_source_only(tmp_path: Path) -> None:
    for tier, filename, shape in (
        ("tier_a", "native_logits", (2, 3, 1369, 2)),
        ("tier_b", "teacher_region", (2, 9, 9)),
    ):
        path = tmp_path / tier / "source_class" / f"{filename}.npy"
        path.parent.mkdir(parents=True)
        np.save(path, np.ones(shape, dtype=np.float32))
    result = control.source_cache_audit(tmp_path)
    assert result["source_only"] is True
    assert result["source_labels_read"] == 0
    assert result["source_masks_read"] == 0
    assert result["held_labels_read"] == 0
    assert result["held_masks_read"] == 0
    assert result["new_model_forwards"] == 0
    assert result["all_finite"] is True


def test_authoritative_preregistration_hash_is_embedded() -> None:
    path = Path("research/sabra_v2/region_distill/P31_PREREGISTRATION.md")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == control.PREREGISTRATION_SHA256
    payload = json.loads(Path("research/sabra_v2/region_distill/P31_PREREGISTRATION.json").read_text())
    assert payload["authoritative_markdown_sha256"] == control.PREREGISTRATION_SHA256


def test_profile_is_offline_and_has_zero_execution_counts() -> None:
    result = control.run_speed_profile()
    assert result["status"] == "SPEED_PROFILE_COMPLETE"
    assert result["microprofile_5_step"]["steps"] == 5
    assert result["warmed_profile_40_step"]["steps"] == 40
    assert result["microprofile_5_step"]["new_model_forwards"] == 0
    assert result["warmed_profile_40_step"]["optimizer_steps"] == 0
    assert result["counts"]["new_scientific_held_predictions"] == 0
    assert result["speed_gates"]["inference_overhead_percent"] == 0
