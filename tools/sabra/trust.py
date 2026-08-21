"""Frozen Trust-v2 calibration and inference.

The public artifact is exactly a StandardScaler followed by the frozen
binary LogisticRegression specified by the protocol.  Scikit-learn is used
when available; the deterministic Newton fallback exists only so bounded
setup tests can run in the minimal research environment.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .relational import FEATURE_ORDER, trust_features

TRUST_FEATURE_ORDER = FEATURE_ORDER
LOGISTIC_SETTINGS = {
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 0,
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("StandardScaler input must be a non-empty matrix")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale


def _fit_numpy_logistic(values: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float]:
    """Deterministic balanced logistic fit used only without scikit-learn."""
    matrix = np.asarray(values, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("logistic calibration requires both classes")
    n, width = matrix.shape
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(n - y.sum()), 1.0)
    weights = np.where(y > 0.5, n / (2.0 * positives), n / (2.0 * negatives))
    beta = np.zeros(width + 1, dtype=np.float64)
    design = np.column_stack([matrix, np.ones(n, dtype=np.float64)])
    regularizer = np.eye(width + 1, dtype=np.float64)
    regularizer[-1, -1] = 0.0
    for _ in range(LOGISTIC_SETTINGS["max_iter"]):
        probabilities = _sigmoid(design @ beta)
        gradient = design.T @ (weights * (probabilities - y)) + beta / LOGISTIC_SETTINGS["C"]
        gradient[-1] -= beta[-1] / LOGISTIC_SETTINGS["C"]
        curvature = probabilities * (1.0 - probabilities) * weights
        hessian = design.T @ (curvature[:, None] * design) + regularizer / LOGISTIC_SETTINGS["C"] + np.eye(width + 1) * 1e-8
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return beta[:-1], float(beta[-1])


def _fit_estimator(scaled: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float, str]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        coefficients, intercept = _fit_numpy_logistic(scaled, targets)
        return coefficients, intercept, "deterministic_numpy_compatible"
    estimator = LogisticRegression(**LOGISTIC_SETTINGS)
    estimator.fit(scaled, np.asarray(targets, dtype=np.int8).reshape(-1))
    return estimator.coef_.astype(np.float64), float(estimator.intercept_[0]), "sklearn"


def fit_binary_predictor(values: np.ndarray, targets: np.ndarray, feature_order: Iterable[str]) -> dict[str, Any]:
    order = tuple(feature_order)
    matrix = np.asarray(values, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.int8).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != len(order) or matrix.shape[0] != target_array.size:
        raise ValueError("predictor feature matrix does not match feature order")
    mean, scale = fit_standardizer(matrix)
    coefficients, intercept, implementation = _fit_estimator((matrix - mean) / scale, target_array)
    return {
        "feature_order": list(order),
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "logistic_coef": np.asarray(coefficients, dtype=np.float64).reshape(1, -1).tolist(),
        "logistic_intercept": [float(intercept)],
        "n_features_in": len(order),
        "predictor": "LogisticRegression",
        "implementation": implementation,
        "settings": dict(LOGISTIC_SETTINGS),
    }


def frozen_probability(parameters: Mapping[str, Any], values: np.ndarray) -> np.ndarray:
    order = tuple(parameters.get("feature_order", ()))
    matrix = np.asarray(values, dtype=np.float64)
    mean = np.asarray(parameters["scaler_mean"], dtype=np.float64)
    scale = np.asarray(parameters["scaler_scale"], dtype=np.float64)
    coefficients = np.asarray(parameters["logistic_coef"], dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(parameters["logistic_intercept"], dtype=np.float64).reshape(-1)[0])
    if matrix.ndim != 2 or matrix.shape[1] != len(order) or coefficients.size != len(order):
        raise ValueError("frozen predictor feature width mismatch")
    if mean.shape != scale.shape or mean.size != len(order) or np.any(scale <= 0.0):
        raise ValueError("frozen scaler shape/value mismatch")
    return _sigmoid(((matrix - mean) / scale) @ coefficients + intercept).astype(np.float32)


def _records_matrix(records: Iterable[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(records)
    if not rows:
        raise ValueError("no calibration records")
    classes = np.asarray([str(row["class_name"]) for row in rows])
    matrix = np.concatenate([trust_features(row) for row in rows], axis=0)
    targets = np.concatenate([np.asarray(row["trust_target"], dtype=np.int8).reshape(-1) for row in rows])
    if matrix.shape[0] != targets.size:
        raise ValueError("Trust records have inconsistent target widths")
    return matrix, targets, classes


def loco_audit(records: Iterable[Mapping[str, Any]], feature_order: Iterable[str] = TRUST_FEATURE_ORDER) -> dict[str, Any]:
    rows = list(records)
    classes = sorted({str(row["class_name"]) for row in rows})
    held_out: dict[str, list[float]] = {}
    held_out_targets: dict[str, list[int]] = {}
    skipped: dict[str, str] = {}
    for class_name in classes:
        train = [row for row in rows if str(row["class_name"]) != class_name]
        test = [row for row in rows if str(row["class_name"]) == class_name]
        if not train or not test:
            skipped[class_name] = "empty_train_or_test"
            continue
        try:
            x_train, y_train, _ = _records_matrix(train)
            artifact = fit_binary_predictor(x_train, y_train, feature_order)
        except ValueError as exc:
            skipped[class_name] = str(exc)
            continue
        x_test = np.concatenate([trust_features(row) for row in test], axis=0)
        y_test = np.concatenate([np.asarray(row["trust_target"], dtype=np.int8).reshape(-1) for row in test])
        held_out[class_name] = frozen_probability(artifact, x_test).tolist()
        held_out_targets[class_name] = y_test.tolist()
    return {
        "class_names": classes,
        "held_out_predictions": held_out,
        "held_out_targets": held_out_targets,
        "skipped": skipped,
        "feature_order": list(feature_order),
    }


def fit_trust(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    matrix, targets, _ = _records_matrix(rows)
    artifact = fit_binary_predictor(matrix, targets, TRUST_FEATURE_ORDER)
    artifact["loco"] = loco_audit(rows)
    return artifact
