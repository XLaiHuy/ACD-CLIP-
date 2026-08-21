from __future__ import annotations

from tools.sabra.artifacts import validate_sabra_freeze
from tools.sabra.relational import FEATURE_ORDER, NEED_ORDER


def _predictor(order):
    return {
        "feature_order": list(order),
        "scaler_mean": [0.0] * len(order),
        "scaler_scale": [1.0] * len(order),
        "logistic_coef": [[0.0] * len(order)],
        "logistic_intercept": [0.0],
        "n_features_in": len(order),
        "predictor": "LogisticRegression",
        "settings": {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000, "random_state": 0},
    }


def _freeze():
    return {
        "protocol_version": "SABRA_CANONICAL_V1",
        "status": "FROZEN",
        "source_training_dataset": "VisA",
        "development_dataset": "MVTecAD",
        "final_test_role": "Medical",
        "medical_seen": False,
        "phase2b": {"selected_epoch": 10, "checkpoint_sha256": "abc"},
        "relational": {"implementation": "tools.sabra.relational.build_relational_record", "peer_count": 8},
        "trust": _predictor(FEATURE_ORDER),
        "need": _predictor(NEED_ORDER),
        "correction": {
            "authority": "T*N",
            "formula": "delta=lambda*margin_scale*T*N",
            "direction": "positive_abnormal_only",
            "normal_delta": 0,
            "shared_across_stages": True,
            "lambda_range": [0.0, 1.0],
            "lambda": 0.0,
            "margin_scale": 1.0,
            "margin_scale_definition": "P90(abs(native_margin))",
        },
        "lambda_selection": {"score_formula": ".35*pAUROC+.35*pAP+.15*iAUROC+.15*iAP", "coarse_grid": [0.0], "selected_score": 0.0},
        "provenance": {"git_sha": "abc"},
    }


def test_freeze_validation_and_checkpoint_hash():
    validate_sabra_freeze(_freeze(), checkpoint_sha256="abc")
    try:
        validate_sabra_freeze(_freeze(), checkpoint_sha256="wrong")
    except ValueError:
        pass
    else:
        raise AssertionError("wrong checkpoint hash accepted")


def test_freeze_rejects_wrong_status():
    payload = _freeze()
    payload["status"] = "CANDIDATE"
    try:
        validate_sabra_freeze(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("non-frozen artifact accepted")


def test_freeze_rejects_wrong_feature_order():
    payload = _freeze()
    payload["trust"]["feature_order"] = list(NEED_ORDER)
    try:
        validate_sabra_freeze(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong Trust feature order accepted")
