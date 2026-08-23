from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.sabra_car.r1_common import EXPECTED_CLASSES, FEATURE_ORDER, THRESHOLDS
from tools.sabra_car.r1_fit import V2_MAX_ITER, V2_SOLVER, estimator
from tools.sabra_car.r1_v2_fit import (
    ORIGINAL_FAILURE,
    V1_STOP,
    V2_OUTPUT,
    V2_PREREG_SHA,
    ensure_attempt_available,
    v2_arguments,
    v2_identity,
)

ORIGINAL_FAILURE_SHA256 = (
    "b9fef1ba2d169c35d2d608820434dbf90a2eca2f3f1503b5a743dc6f72cbf8f5"
)
V1_STOP_SHA256 = (
    "1aed2a970ac788833f2841137906c1ef55b885a879f64627461eaad1fc492559"
)


def test_v2_changes_only_frozen_solver_contract():
    old = estimator(5000, "lbfgs").get_params()
    new = estimator(V2_MAX_ITER, V2_SOLVER).get_params()
    assert V2_SOLVER == "newton-cholesky"
    assert V2_MAX_ITER == 100
    assert old | {"solver": V2_SOLVER, "max_iter": V2_MAX_ITER} == new
    assert new["C"] == 1.0
    assert new["penalty"] == "l2"
    assert new["class_weight"] == "balanced"
    assert new["tol"] == 1e-4
    assert new["fit_intercept"] is True
    assert new["random_state"] == 0
    assert new["multi_class"] == "multinomial"
    with pytest.raises(ValueError, match="unauthorized R1 max_iter/solver"):
        estimator(101, V2_SOLVER)
    with pytest.raises(ValueError, match="unauthorized R1 max_iter/solver"):
        estimator(100, "lbfgs")


def test_v2_frozen_orders_and_thresholds():
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


def test_v2_identity_and_separate_output_are_frozen():
    args = v2_arguments()
    identity = v2_identity()
    assert args.output == V2_OUTPUT.resolve()
    assert args.skip_hashes is False
    assert args.solver == "newton-cholesky"
    assert args.max_iter == 100
    assert args.protocol_prereg_sha == V2_PREREG_SHA
    assert identity["r1_v1_status"] == "COMPUTATIONAL_STOP"
    assert identity["max_v2_attempts"] == 1
    assert identity["medical_reads"] == 0
    assert identity["mvtec_reads"] == 0
    assert identity["phase2b_training_steps"] == 0
    assert V2_OUTPUT != ORIGINAL_FAILURE.parent
    assert V2_OUTPUT != V1_STOP.parent


def test_v2_preserves_history_and_cannot_start_twice(tmp_path):
    assert hashlib.sha256(ORIGINAL_FAILURE.read_bytes()).hexdigest() == ORIGINAL_FAILURE_SHA256
    assert hashlib.sha256(V1_STOP.read_bytes()).hexdigest() == V1_STOP_SHA256
    candidate = tmp_path / "r1_v2"
    ensure_attempt_available(candidate)
    candidate.mkdir()
    with pytest.raises(RuntimeError, match="attempt already exists or was started"):
        ensure_attempt_available(candidate)


def test_original_r1_gate_literals_remain_frozen():
    source = (Path(__file__).resolve().parents[1] / "tools/sabra_car/r1_evaluate.py").read_text()
    for literal in (
        '"G1_coverage"', '"threshold": ">=0.10"',
        '"G2_opposite_sign_rate"', '"threshold": "<=0.05"',
        '"G3_relative_risk_reduction"', '"threshold": ">=0.25"',
        '"G4_macro_pAP_delta_pp"', '"threshold": ">=0.50"',
        '"G5_macro_pAUROC_delta_pp"', '"threshold": ">=-0.50"',
        '"G6_nonnegative_breadth"', '"threshold": ">=7"',
    ):
        assert literal in source
