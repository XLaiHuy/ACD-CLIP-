"""Strict JSON serialization for source calibration and SABRA freeze.

The source-calibration artifact is deliberately lambda-free.  The final
freeze is assembled only after the MVTec development lambda curve has been
selected, and carries enough metadata for a later loader to reject a
different Phase2B checkpoint or a partially specified predictor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .correction import validate_lambda
from .relational import FEATURE_ORDER, NEED_ORDER

PROTOCOL_VERSION = "SABRA_CANONICAL_V1"
METRIC_FORMULA = ".35*pAUROC+.35*pAP+.15*iAUROC+.15*iAP"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _validate_predictor(
    name: str,
    payload: Mapping[str, Any],
    feature_order: Sequence[str],
) -> None:
    if tuple(payload.get("feature_order", ())) != tuple(feature_order):
        raise ValueError(f"{name} feature order mismatch")
    for field in ("scaler_mean", "scaler_scale", "logistic_coef", "logistic_intercept", "settings"):
        if field not in payload:
            raise ValueError(f"{name} artifact missing {field}")
    if payload.get("predictor") != "LogisticRegression":
        raise ValueError(f"{name} predictor must be LogisticRegression")
    if int(payload.get("n_features_in", -1)) != len(feature_order):
        raise ValueError(f"{name} feature width mismatch")
    if len(payload["scaler_mean"]) != len(feature_order) or len(payload["scaler_scale"]) != len(feature_order):
        raise ValueError(f"{name} scaler width mismatch")
    coef = payload["logistic_coef"]
    if not isinstance(coef, list) or len(coef) != 1 or len(coef[0]) != len(feature_order):
        raise ValueError(f"{name} coefficient width mismatch")
    if len(payload["logistic_intercept"]) != 1:
        raise ValueError(f"{name} intercept width mismatch")
    settings = payload["settings"]
    expected = {"C": 1.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000, "random_state": 0}
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"{name} logistic setting {key} mismatch")


def validate_source_calibration(payload: Mapping[str, Any]) -> None:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("source calibration protocol mismatch")
    if payload.get("status") != "SOURCE_FITTED":
        raise ValueError("source calibration status must be SOURCE_FITTED")
    phase2b = payload.get("phase2b", {})
    if not phase2b.get("selected_epoch") or not phase2b.get("checkpoint_sha256"):
        raise ValueError("source calibration must identify the selected Phase2B checkpoint")
    relational = payload.get("relational", {})
    if relational.get("implementation") != "tools.sabra.relational.build_relational_record":
        raise ValueError("relational implementation identity mismatch")
    if int(relational.get("peer_count", -1)) != 8:
        raise ValueError("relational peer count must be eight")
    _validate_predictor("Trust", payload.get("trust", {}), FEATURE_ORDER)
    _validate_predictor("Need", payload.get("need", {}), NEED_ORDER)
    margin = payload.get("margin_scale", {})
    if margin.get("definition") != "P90(abs(native_margin))":
        raise ValueError("margin scale definition mismatch")
    if margin.get("implementation") != "numpy.percentile(method=linear)":
        raise ValueError("margin scale implementation mismatch")
    if float(margin.get("percentile")) != 90.0 or int(margin.get("count", 0)) <= 0:
        raise ValueError("margin scale metadata is incomplete")
    if "lambda" in payload or "lambda" in payload.get("correction", {}):
        raise ValueError("source calibration must not contain lambda")
    if not payload.get("provenance", {}).get("git_sha"):
        raise ValueError("source calibration provenance must include git_sha")


def build_freeze_payload(
    source_calibration: Mapping[str, Any],
    selected_lambda: float,
    selected_score: float,
    git_sha: str,
    coarse_grid: Sequence[float],
    refinement_rule: str = "center +/- 0.05 clamped to [0,1], step 0.005; no duplicate coarse points",
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Create the only accepted final SABRA freeze shape."""
    validate_source_calibration(source_calibration)
    value = validate_lambda(selected_lambda)
    margin = dict(source_calibration["margin_scale"])
    provenance = dict(source_calibration.get("provenance", {}))
    provenance["git_sha"] = str(git_sha)
    if source_hashes is not None:
        provenance["critical_source_hashes"] = dict(source_hashes)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN",
        "source_training_dataset": "VisA",
        "development_dataset": "MVTecAD",
        "final_test_role": "Medical",
        "phase2b": dict(source_calibration["phase2b"]),
        "trust": dict(source_calibration["trust"]),
        "need": dict(source_calibration["need"]),
        "relational": dict(source_calibration["relational"]),
        "correction": {
            "authority": "T*N",
            "formula": "delta=lambda*margin_scale*T*N",
            "direction": "positive_abnormal_only",
            "normal_delta": 0,
            "shared_across_stages": True,
            "lambda_range": [0.0, 1.0],
            "lambda": value,
            "margin_scale": float(margin["value"]),
            "margin_scale_definition": margin["definition"],
        },
        "lambda_selection": {
            "score_formula": METRIC_FORMULA,
            "coarse_grid": [float(item) for item in coarse_grid],
            "refinement_rule": refinement_rule,
            "selected_score": float(selected_score),
        },
        "medical_seen": False,
        "provenance": provenance,
    }
    validate_sabra_freeze(payload)
    return payload


def validate_sabra_freeze(payload: Mapping[str, Any], checkpoint_sha256: str | None = None) -> None:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("SABRA freeze protocol mismatch")
    if payload.get("status") != "FROZEN":
        raise ValueError("SABRA freeze status must be FROZEN")
    if payload.get("source_training_dataset") != "VisA" or payload.get("development_dataset") != "MVTecAD":
        raise ValueError("SABRA data roles are not canonical")
    if payload.get("final_test_role") != "Medical":
        raise ValueError("SABRA final test role must be Medical")
    if payload.get("medical_seen") is not False:
        raise ValueError("SABRA freeze must declare medical_seen=false")
    phase2b = payload.get("phase2b", {})
    expected = phase2b.get("checkpoint_sha256")
    if not phase2b.get("selected_epoch") or not expected:
        raise ValueError("SABRA freeze must identify the selected Phase2B checkpoint")
    if checkpoint_sha256 is not None and expected != checkpoint_sha256:
        raise ValueError("SABRA freeze checkpoint hash mismatch")
    _validate_predictor("Trust", payload.get("trust", {}), FEATURE_ORDER)
    _validate_predictor("Need", payload.get("need", {}), NEED_ORDER)
    relational = payload.get("relational", {})
    if relational.get("implementation") != "tools.sabra.relational.build_relational_record":
        raise ValueError("relational implementation identity mismatch")
    if int(relational.get("peer_count", -1)) != 8:
        raise ValueError("relational peer count must be eight")
    correction = payload.get("correction", {})
    if correction.get("authority") != "T*N":
        raise ValueError("Authority must be T*N")
    if correction.get("formula") != "delta=lambda*margin_scale*T*N":
        raise ValueError("correction formula mismatch")
    if correction.get("direction") != "positive_abnormal_only":
        raise ValueError("correction direction mismatch")
    if correction.get("normal_delta") != 0 or correction.get("shared_across_stages") is not True:
        raise ValueError("correction tensor contract mismatch")
    if list(correction.get("lambda_range", ())) != [0.0, 1.0]:
        raise ValueError("lambda range mismatch")
    validate_lambda(float(correction.get("lambda")))
    if float(correction.get("margin_scale")) < 0.0:
        raise ValueError("margin scale must be non-negative")
    if correction.get("margin_scale_definition") != "P90(abs(native_margin))":
        raise ValueError("frozen margin scale definition mismatch")
    selection = payload.get("lambda_selection", {})
    if selection.get("score_formula") != METRIC_FORMULA or not selection.get("coarse_grid"):
        raise ValueError("lambda selection metadata is incomplete")
    if not payload.get("provenance", {}).get("git_sha"):
        raise ValueError("SABRA freeze provenance must include git_sha")
