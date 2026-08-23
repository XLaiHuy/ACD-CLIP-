from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.sabra_car.r1_common import EXPECTED_CLASSES, FEATURE_ORDER, THRESHOLDS
from tools.sabra_car.r1_fit import (
    ORIGINAL_MAX_ITER,
    RECOVERY_MAX_ITER,
    RECOVERY_PROTOCOL,
    estimator,
)
from tools.sabra_car.r1_recovery_v1_fit import (
    ORIGINAL_FAILURE,
    RECOVERY_OUTPUT,
    ensure_attempt_available,
    recovery_arguments,
    recovery_identity,
)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_FAILURE_SHA256 = (
    "b9fef1ba2d169c35d2d608820434dbf90a2eca2f3f1503b5a743dc6f72cbf8f5"
)


def test_recovery_changes_only_max_iter_in_estimator_contract():
    original = estimator(ORIGINAL_MAX_ITER).get_params()
    recovery = estimator(RECOVERY_MAX_ITER).get_params()
    assert ORIGINAL_MAX_ITER == 1000
    assert RECOVERY_MAX_ITER == 5000
    assert recovery["max_iter"] == 5000
    assert original | {"max_iter": RECOVERY_MAX_ITER} == recovery
    assert recovery["solver"] == "lbfgs"
    assert recovery["tol"] == 1e-4
    assert recovery["C"] == 1.0
    assert recovery["penalty"] == "l2"
    assert recovery["class_weight"] == "balanced"
    assert recovery["random_state"] == 0
    assert recovery["fit_intercept"] is True
    assert recovery["multi_class"] == "multinomial"
    with pytest.raises(ValueError, match="unauthorized R1 max_iter"):
        estimator(5001)


def test_recovery_frozen_data_order_and_threshold_contract():
    assert EXPECTED_CLASSES == (
        "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
    )
    assert FEATURE_ORDER == (
        "margin_within_image_rank", "robust_margin_normalization", "D_rank",
        "deployment_sensitivity", "E", "peer_coherence", "query_support_mean",
        "peer_eigen_entropy", "stage_query_profile_disagreement",
        "supported_p9_stability", "supported_p16_stability",
    )
    assert THRESHOLDS == (0.50, 0.60, 0.70, 0.80, 0.90)


def test_recovery_identity_schema_and_firewalls_are_frozen():
    args = recovery_arguments()
    identity = recovery_identity()
    assert args.output == RECOVERY_OUTPUT.resolve()
    assert args.skip_hashes is False
    assert args.max_iter == 5000
    assert args.recovery_protocol == RECOVERY_PROTOCOL
    assert identity["original_max_iter"] == 1000
    assert identity["recovery_max_iter"] == 5000
    assert identity["max_recovery_attempts"] == 1
    assert identity["medical_reads"] == 0
    assert identity["mvtec_reads"] == 0
    assert identity["phase2b_training_steps"] == 0


def test_recovery_cannot_overwrite_history_or_start_twice(tmp_path):
    assert ORIGINAL_FAILURE == ROOT / "results/sabra_car/r1/FIT_FAILED.json"
    assert RECOVERY_OUTPUT != ORIGINAL_FAILURE.parent
    assert hashlib.sha256(ORIGINAL_FAILURE.read_bytes()).hexdigest() == ORIGINAL_FAILURE_SHA256
    candidate = tmp_path / "r1_recovery_v1"
    ensure_attempt_available(candidate)
    candidate.mkdir()
    with pytest.raises(RuntimeError, match="attempt already exists or was started"):
        ensure_attempt_available(candidate)


def test_original_r1_gate_literals_remain_frozen():
    source = (ROOT / "tools/sabra_car/r1_evaluate.py").read_text()
    for literal in (
        '"G1_coverage"', '"threshold": ">=0.10"',
        '"G2_opposite_sign_rate"', '"threshold": "<=0.05"',
        '"G3_relative_risk_reduction"', '"threshold": ">=0.25"',
        '"G4_macro_pAP_delta_pp"', '"threshold": ">=0.50"',
        '"G5_macro_pAUROC_delta_pp"', '"threshold": ">=-0.50"',
        '"G6_nonnegative_breadth"', '"threshold": ">=7"',
    ):
        assert literal in source
