#!/usr/bin/env python3
"""P25R3 exact numerical recovery for the frozen P25R2 ranker.

The scientific model and beta-space objective are inherited unchanged.  Only
the internal coordinates used by the numerical optimizer differ.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any
import uuid

import numpy as np
from scipy.special import expit
import torch

from tools.sabra_cure import patch_actionability_r1 as frozen
from tools.sabra_cure import patch_actionability_r2 as parent
from tools.sabra_cure import r1


PAIR_L2 = frozen.PAIR_L2
FEATURE_ORDER = frozen.FEATURE_ORDER
RELATIVE_GRADIENT_INF_TOLERANCE = 1e-7
ROOT = Path(__file__).resolve().parents[2]
PARENT_OUT = ROOT / "results/sabra_cure/patch_actionability_r2"
OUT = ROOT / "results/sabra_cure/patch_actionability_r3"
DOC_OUT = ROOT / "research/sabra_cure/patch_actionability_r3"
BRANCH = "research/p25r3-sabra-cure-q1-numerical-recovery-v1"
P25R2_TERMINAL_SHA = "99ad3ab6292ca3b95fbda0cb8c6985ed9afe3253"
P25R2_EXECUTION_BASE_SHA = "a50d9a6d0f7c1cea007f6d18ea9cafbb6b8711d0"
P25R2_PREREG_SHA = "233d16b0b29286c8ea73b7886a8d929ca263e5d7"
FORENSIC_SHA = "2d80a02366d19275b47ca5621e34090aeaf18ddd"
PREREG_SHA = "9b641d1b1deff64421736cdaa9dc6a5db8736bc7"
TARGET_HASHES = {
    "candle": "c6322a27c470c374626a51a9beb2f8402fed12e226a475f52d5fbf6131a7e146",
    "capsules": "235003955b3374232859eb7df706e579861e199f57999c89ccf7b76f47f28a39",
    "cashew": "5f67cd2b84ad8b9b5ef0fbcde74de051b6f7107b4f38a6b5100333e316735617",
    "chewinggum": "80bf2ff802297fa5be25dd2175a200c712ffdefa94856c3eb801083bf7c8897d",
    "fryum": "851cbb1a2fa358313e3786842c705eccdef4556a92d59e9544b696ccafb260b4",
    "macaroni1": "f17ae04814d84dca6b0b907e822853529e7940865f5ab558d81dbd7b9b7b8bff",
    "macaroni2": "4bc332ecdd61f46114c039a20f372d8c4f19e361ef6099f0aa9ececef7c6ce37",
    "pcb1": "435caf83b3f09520f0c403966a1a85a6bb9ef2ab7a840b4e99a5510ada83347e",
    "pcb2": "5b0c2162916b4d04d04227202202473cc66dba32fd0de1f626bdaa421bc4747c",
    "pcb3": "359b1c2ab204b0bfe99b08faa8759a0e8887843c39e8bc93b57844feff01fbaf",
    "pcb4": "be64886f65acccfacfc11a35e6177e4cba3916a5c699215550f533a545d0985a",
    "pipe_fryum": "301437a97059d8c823a6fd393e19442ef02b8c0bf6bed341aebcd486beab1ad4",
}


@dataclass(frozen=True)
class PairProblem:
    design: np.ndarray
    weight: np.ndarray
    median: np.ndarray
    iqr: np.ndarray
    scale: np.ndarray
    active: np.ndarray
    provenance: tuple[dict[str, Any], ...]

    @property
    def feature_count(self) -> int:
        return int(self.design.shape[1])


def build_pair_problem(x: np.ndarray, groups: list[tuple[str, np.ndarray]]) -> PairProblem:
    """Reconstruct the exact frozen pair design, then derive train-only scales."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise RuntimeError("P25R3_ENGINEERING_STOP invalid ranker features")
    median, iqr = frozen._feature_scaler(x)
    standardized = (x - median) / iqr
    designs: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    offset = 0
    for name, raw_values in groups:
        values = np.asarray(raw_values, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("P25R3_ENGINEERING_STOP nonfinite rank target")
        left, right, pair_weight = frozen._deterministic_pairs(name, values)
        count = int(len(values))
        designs.append(standardized[offset + left] - standardized[offset + right])
        weights.append(pair_weight)
        provenance.append(
            {"class": name, "pairs": int(len(left)), "decile_rule": "same/adjacent skipped", "cap": frozen.PAIR_CAP}
        )
        offset += count
    if offset != len(x) or not designs:
        raise RuntimeError("P25R3_ENGINEERING_STOP ranker group alignment")
    design = np.concatenate(designs).astype(np.float64, copy=False)
    weight = np.concatenate(weights).astype(np.float64, copy=False)
    weight = weight / weight.mean()
    column_max = np.max(np.abs(design), axis=0)
    active = column_max > 0.0
    scale = np.where(active, np.maximum(column_max, 1.0), 1.0).astype(np.float64)
    if not np.any(active):
        raise RuntimeError("P25R3_ENGINEERING_STOP no active pair-design dimensions")
    if not np.all(np.isfinite(design)) or not np.all(np.isfinite(weight)) or not np.all(np.isfinite(scale)):
        raise RuntimeError("P25R3_ENGINEERING_STOP nonfinite pair problem")
    return PairProblem(design, weight, median, iqr, scale, active, tuple(provenance))


def beta_objective_gradient(problem: PairProblem, beta: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact historical objective and gradient in original beta coordinates."""
    beta = np.asarray(beta, dtype=np.float64).reshape(problem.feature_count)
    margin = problem.design @ beta
    objective = float(np.mean(problem.weight * np.logaddexp(0.0, -margin)) + 0.5 * PAIR_L2 * np.dot(beta, beta))
    factor = -problem.weight * expit(-margin)
    gradient = np.mean(factor[:, None] * problem.design, axis=0) + PAIR_L2 * beta
    if not np.isfinite(objective) or not np.all(np.isfinite(gradient)):
        raise RuntimeError("P25R3_ENGINEERING_STOP nonfinite objective")
    return objective, gradient


def z_to_beta(problem: PairProblem, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(int(np.count_nonzero(problem.active)))
    beta = np.zeros(problem.feature_count, dtype=np.float64)
    beta[problem.active] = z / problem.scale[problem.active]
    return beta


def beta_to_z(problem: PairProblem, beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=np.float64).reshape(problem.feature_count)
    if np.any(beta[~problem.active] != 0.0):
        raise ValueError("inactive beta dimensions must be exactly zero")
    return beta[problem.active] * problem.scale[problem.active]


def transformed_objective_gradient(problem: PairProblem, z: np.ndarray) -> tuple[float, np.ndarray]:
    beta = z_to_beta(problem, z)
    objective, gradient_beta = beta_objective_gradient(problem, beta)
    gradient_z = gradient_beta[problem.active] / problem.scale[problem.active]
    return objective, gradient_z


def transformed_hessian(problem: PairProblem, z: np.ndarray) -> np.ndarray:
    """Exact Hessian of the unchanged objective in active z coordinates."""
    beta = z_to_beta(problem, z)
    margin = problem.design @ beta
    probability = expit(-margin)
    curvature = problem.weight * probability * (1.0 - probability)
    active_scale = problem.scale[problem.active]
    transformed_design = problem.design[:, problem.active] / active_scale
    hessian = (
        transformed_design.T @ (curvature[:, None] * transformed_design) / len(transformed_design)
        + np.diag(PAIR_L2 / np.square(active_scale))
    )
    if not np.all(np.isfinite(hessian)):
        raise RuntimeError("P25R3_ENGINEERING_STOP nonfinite Hessian")
    return hessian


def original_beta_certificate(
    problem: PairProblem,
    beta: np.ndarray,
    *,
    optimizer_success: bool,
    objective_initial: float,
    objective_final: float,
) -> dict[str, Any]:
    beta = np.asarray(beta, dtype=np.float64).reshape(problem.feature_count)
    _, gradient = beta_objective_gradient(problem, beta)
    zero = np.zeros(problem.feature_count, dtype=np.float64)
    _, gradient_initial = beta_objective_gradient(problem, zero)
    gradient_inf = float(np.max(np.abs(gradient)))
    initial_gradient_inf = float(np.max(np.abs(gradient_initial)))
    relative = gradient_inf / max(1.0, initial_gradient_inf)
    finite = bool(
        np.all(np.isfinite(beta))
        and np.all(np.isfinite(gradient))
        and np.isfinite(objective_initial)
        and np.isfinite(objective_final)
    )
    valid = bool(
        optimizer_success
        and finite
        and objective_final <= objective_initial
        and relative <= RELATIVE_GRADIENT_INF_TOLERANCE
    )
    return {
        "valid": valid,
        "finite": finite,
        "gradient_l2": float(np.linalg.norm(gradient)),
        "gradient_inf": gradient_inf,
        "initial_gradient_inf": initial_gradient_inf,
        "relative_gradient_inf": float(relative),
        "relative_gradient_inf_tolerance": RELATIVE_GRADIENT_INF_TOLERANCE,
    }


def fit_ranker(x: np.ndarray, groups: list[tuple[str, np.ndarray]]) -> dict[str, Any]:
    problem = build_pair_problem(x, groups)
    zero_beta = np.zeros(problem.feature_count, dtype=np.float64)
    objective_initial, _ = beta_objective_gradient(problem, zero_beta)
    z = np.zeros(int(np.count_nonzero(problem.active)), dtype=np.float64)
    iterations = 0
    function_evaluations = 0
    line_search_evaluations = 0
    message = "maximum iterations reached"
    for iteration in range(50):
        beta = z_to_beta(problem, z)
        objective_current, gradient_beta = beta_objective_gradient(problem, beta)
        function_evaluations += 1
        certificate_current = original_beta_certificate(
            problem,
            beta,
            optimizer_success=True,
            objective_initial=objective_initial,
            objective_final=objective_current,
        )
        if certificate_current["valid"]:
            message = "original-beta certificate satisfied"
            break
        gradient_z = gradient_beta[problem.active] / problem.scale[problem.active]
        hessian = transformed_hessian(problem, z)
        try:
            direction = np.linalg.solve(hessian, -gradient_z)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("P25R3_ENGINEERING_STOP Newton Hessian solve") from error
        directional_derivative = float(np.dot(gradient_z, direction))
        if not np.isfinite(directional_derivative) or directional_derivative >= 0.0:
            raise RuntimeError("P25R3_ENGINEERING_STOP non-descent Newton direction")
        step = 1.0
        accepted = False
        while step >= 2.0 ** -30:
            trial = z + step * direction
            objective_trial, _ = transformed_objective_gradient(problem, trial)
            function_evaluations += 1
            line_search_evaluations += 1
            if objective_trial <= objective_current + 1e-4 * step * directional_derivative:
                z = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise RuntimeError("P25R3_ENGINEERING_STOP Newton Armijo line search")
        iterations = iteration + 1
    beta = z_to_beta(problem, z)
    objective_final, _ = beta_objective_gradient(problem, beta)
    certificate = original_beta_certificate(
        problem,
        beta,
        optimizer_success=message == "original-beta certificate satisfied",
        objective_initial=objective_initial,
        objective_final=objective_final,
    )
    if not certificate["valid"]:
        raise RuntimeError(f"P25R3_ENGINEERING_STOP invalid original-beta optimum: {certificate}")
    score = rank_predict({"median": problem.median, "iqr": problem.iqr, "beta": beta}, x)
    optimization = {
        "solver": "deterministic-float64-damped-newton",
        "success": bool(certificate["valid"]),
        "status": 0 if certificate["valid"] else 1,
        "message": message,
        "iterations": iterations,
        "function_evaluations": function_evaluations,
        "gradient_evaluations": iterations + 1,
        "line_search_evaluations": line_search_evaluations,
        "objective_initial": objective_initial,
        "objective_final": objective_final,
        "beta_norm": float(np.linalg.norm(beta)),
        "prediction_min": float(np.min(score)),
        "prediction_max": float(np.max(score)),
        "prediction_std": float(np.std(score)),
        "prediction_unique_count": int(np.unique(score).size),
        "certificate": certificate,
    }
    return {
        "median": problem.median,
        "iqr": problem.iqr,
        "beta": beta,
        "preconditioner_scale": problem.scale,
        "active_dimensions": problem.active,
        "pairs": list(problem.provenance),
        "loss": objective_final,
        "pair_count": int(len(problem.design)),
        "feature_order": list(FEATURE_ORDER),
        "optimization": optimization,
    }


def rank_predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return frozen.rank_predict(model, x)


def q1_metrics(v: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    return frozen.q1_metrics(v, score)


def json_safe_model(model: dict[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return convert(model)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def audit_target_artifacts(root: Path = PARENT_OUT) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    total = 0
    for name in r1.CLASSES:
        path = root / "targets" / f"{name}.npz"
        actual_hash = sha256(path)
        if actual_hash != TARGET_HASHES[name]:
            raise RuntimeError(f"P25R3_ENGINEERING_STOP target hash mismatch: {name}")
        with np.load(path, allow_pickle=False) as artifact:
            required = {
                "image_path",
                "image_index",
                "patch_index",
                "rank_stratum",
                "sensitivity_stratum",
                "oracle_direction",
                "native_ap",
                "candidate_ap",
                "V",
            }
            if set(artifact.files) != required:
                raise RuntimeError(f"P25R3_ENGINEERING_STOP target schema mismatch: {name}")
            count = int(len(artifact["V"]))
            if count != 2000 or any(len(artifact[key]) != count for key in required):
                raise RuntimeError(f"P25R3_ENGINEERING_STOP target count/alignment: {name}")
            numeric = [key for key in required if key != "image_path"]
            if any(not np.all(np.isfinite(artifact[key])) for key in numeric):
                raise RuntimeError(f"P25R3_ENGINEERING_STOP nonfinite target: {name}")
            panel_path = root / "panels" / f"{name}.npz"
            with np.load(panel_path, allow_pickle=False) as panel:
                for key in ("image_path", "image_index", "patch_index", "rank_stratum", "sensitivity_stratum"):
                    if not np.array_equal(artifact[key], panel[key]):
                        raise RuntimeError(f"P25R3_ENGINEERING_STOP target/panel alignment: {name}:{key}")
        total += count
        classes[name] = {"path": str(path.relative_to(ROOT)), "sha256": actual_hash, "count": count}
    return {"status": "PASS", "classes": classes, "class_count": len(classes), "total_rows": total}


def static_interface_audit() -> dict[str, Any]:
    specifications = {
        "patch_actionability_r2": (
            parent,
            {
                "r2v2_harm": "module",
                "feature_rows_for_outer": "callable",
                "load_target": "callable",
                "load_panel": "callable",
                "atomic_npz": "callable",
                "atomic_json": "callable",
                "select_source_policy": "callable",
                "class_state": "callable",
                "exact_metrics": "callable",
                "_policy_metrics": "callable",
                "evaluate_q1": "callable",
                "evaluate_q2": "callable",
                "_default": "callable",
                "BENEFIT_EPS": "float",
                "PATCHES": "int",
                "ALPHA": "float",
                "MARGIN_SCALE": "float",
            },
        ),
        "r2v2_harm": (
            parent.r2v2_harm,
            {name: "callable" for name in ("outer", "direction_group", "scaler", "ridge", "pred", "harm_features", "wrong")},
        ),
        "r1": (
            r1,
            {"CLASSES": "tuple", "Shard": "type", "pearson": "callable", "load_shards": "callable"},
        ),
        "patch_actionability_r1": (
            frozen,
            {"PAIR_L2": "float", "FEATURE_ORDER": "tuple", "PAIR_CAP": "int", "_feature_scaler": "callable", "_deterministic_pairs": "callable", "rank_predict": "callable", "q1_metrics": "callable"},
        ),
    }
    references: list[dict[str, Any]] = []
    for object_type, (owner, attributes) in specifications.items():
        for attribute, expected in attributes.items():
            exists = hasattr(owner, attribute)
            value = getattr(owner, attribute, None)
            if expected == "callable":
                exists = exists and callable(value)
            references.append(
                {
                    "object_type": object_type,
                    "referenced_attribute": attribute,
                    "exists": bool(exists),
                    "expected_shape_type": expected,
                    "production_callsite": "tools/sabra_cure/patch_actionability_r3.py",
                }
            )
    return {"status": "PASS" if all(row["exists"] for row in references) else "FAIL", "references": references}


def objective_gradient_parity_audit() -> dict[str, Any]:
    rng = np.random.default_rng(2531)
    x = rng.normal(size=(60, 5)).astype(np.float64)
    x[:, 4] = 2.0
    values = rng.normal(size=60).astype(np.float64)
    problem = build_pair_problem(x, [("a", values[:30]), ("b", values[30:])])
    max_objective_error = 0.0
    max_chain_error = 0.0
    max_finite_difference_error = 0.0
    for beta in (
        np.array([0.1, -0.2, 0.3, -0.1, 0.0]),
        np.array([-0.4, 0.05, -0.07, 0.2, 0.0]),
        np.array([0.01, 0.02, -0.03, 0.04, 0.0]),
    ):
        z = beta_to_z(problem, beta)
        direct_value, direct_gradient = beta_objective_gradient(problem, beta)
        transformed_value, transformed_gradient = transformed_objective_gradient(problem, z)
        expected = direct_gradient[problem.active] / problem.scale[problem.active]
        max_objective_error = max(max_objective_error, abs(direct_value - transformed_value))
        max_chain_error = max(max_chain_error, float(np.max(np.abs(expected - transformed_gradient))))
        epsilon = 1e-6
        finite_difference = np.empty_like(transformed_gradient)
        for index in range(len(z)):
            plus = z.copy(); minus = z.copy()
            plus[index] += epsilon; minus[index] -= epsilon
            finite_difference[index] = (transformed_objective_gradient(problem, plus)[0] - transformed_objective_gradient(problem, minus)[0]) / (2.0 * epsilon)
        max_finite_difference_error = max(max_finite_difference_error, float(np.max(np.abs(finite_difference - transformed_gradient))))
    status = "PASS" if max_objective_error <= 1e-12 and max_chain_error <= 1e-12 and max_finite_difference_error <= 1e-8 else "FAIL"
    return {
        "status": status,
        "vectors": 3,
        "max_objective_error": max_objective_error,
        "max_chain_rule_gradient_error": max_chain_error,
        "max_finite_difference_gradient_error": max_finite_difference_error,
    }


def synthetic_production_rehearsal() -> dict[str, Any]:
    rng = np.random.default_rng(2532)
    x = rng.normal(size=(120, 32)).astype(np.float64)
    x[:, 0] *= 1e8
    x[:, 1] *= 1e-7
    x[:, 31] = 1.0
    latent = 3e-8 * x[:, 0] - 2e6 * x[:, 1] + 0.3 * x[:, 2] - 0.2 * x[:, 3]
    groups = [("a", latent[:40]), ("b", latent[40:80]), ("c", latent[80:])]
    model = fit_ranker(x, groups)
    score = rank_predict(model, x)
    metric = q1_metrics(latent, score)
    serial = json_safe_model(model)
    json.dumps({"model": serial, "metrics": metric}, allow_nan=False)
    passing_folds = {
        name: {
            "metrics": {
                "spearman": 0.3,
                "sign_auc": 0.7,
                "bc20": 0.4,
                "positive_count": 4,
                "negative_count": 4,
            }
        }
        for name in r1.CLASSES
    }
    stopping_folds = json.loads(json.dumps(passing_folds))
    for name in r1.CLASSES[:4]:
        stopping_folds[name]["metrics"]["spearman"] = -0.1
    pass_route = parent.evaluate_q1(passing_folds)
    stop_route = parent.evaluate_q1(stopping_folds)
    if not pass_route["pass"] or stop_route["pass"]:
        raise RuntimeError("P25R3_ENGINEERING_STOP controller routing rehearsal")
    if not model["optimization"]["certificate"]["valid"] or np.unique(score).size <= 1:
        raise RuntimeError("P25R3_ENGINEERING_STOP synthetic production rehearsal")
    return {
        "status": "PASS",
        "feature_count": int(x.shape[1]),
        "training_rows": int(len(x)),
        "pair_count": int(model["pair_count"]),
        "beta_norm": float(np.linalg.norm(model["beta"])),
        "prediction_std": float(np.std(score)),
        "prediction_unique_count": int(np.unique(score).size),
        "certificate": model["optimization"]["certificate"],
        "q1_pass_route": bool(pass_route["pass"]),
        "q1_stop_route": bool(not stop_route["pass"]),
        "strict_json": True,
    }


def source_only_outer_training(held: str, shards: dict[str, r1.Shard]) -> dict[str, Any]:
    """Exact source portion of r2v2_harm.outer, stopping before held access."""
    harm = parent.r2v2_harm
    names = [name for name in r1.CLASSES if name != held]
    groups: list[dict[str, Any]] = []
    for name in names:
        training = [candidate for candidate in names if candidate != name]
        mu, y = harm.direction_group(shards, training, name)
        groups.append({"name": name, "x": shards[name].x, "mu": mu, "y": y, "training": training})
    x_all = np.concatenate([group["x"] for group in groups])
    residual_target = np.concatenate([np.log(np.abs(group["y"] - group["mu"]) + 1e-4) for group in groups])
    uncertainty_median, uncertainty_iqr = harm.scaler(x_all)
    uncertainty_beta, uncertainty_intercept = harm.ridge((x_all - uncertainty_median) / uncertainty_iqr, residual_target)
    for group in groups:
        group["sigma"] = np.exp(
            np.clip(
                harm.pred(group["x"], uncertainty_median, uncertainty_iqr, uncertainty_beta, uncertainty_intercept),
                np.log(1e-4),
                np.log(4),
            )
        )
        group["f"] = harm.harm_features(group["x"], group["mu"], group["sigma"])
        group["w"] = harm.wrong(group["mu"], group["y"])
        group["h"] = group["w"] * np.abs(group["y"])
    for group in groups:
        others = [candidate for candidate in groups if candidate["name"] != group["name"]]
        features = np.concatenate([candidate["f"] for candidate in others])
        median, iqr = harm.scaler(features)
        for key in ("h", "w"):
            beta, intercept = harm.ridge((features - median) / iqr, np.concatenate([candidate[key] for candidate in others]))
            group["r_" + key] = harm.pred(group["f"], median, iqr, beta, intercept)
    level2_harm = np.concatenate([group["r_h"] for group in groups])
    tau_harm = float(np.quantile(level2_harm, 0.2, method="linear"))
    return {"held": held, "level1": groups, "tau_harm": tau_harm}


def known_failure_regression(held: str = "chewinggum") -> dict[str, Any]:
    """Source-only fit rehearsal; no held target, held metric, or held prediction is read."""
    started = time.perf_counter()
    shards, _ = r1.load_shards(True)
    outer = source_only_outer_training(held, shards)
    names = [name for name in r1.CLASSES if name != held]
    source_features: list[np.ndarray] = []
    source_values: list[np.ndarray] = []
    for name in names:
        features, _ = parent.feature_rows_for_outer(held, name, outer, shards, PARENT_OUT)
        source_features.append(features)
        source_values.append(np.asarray(parent.load_target(PARENT_OUT, name)["V"], dtype=np.float64))
    training_x = np.concatenate(source_features)
    groups = list(zip(names, source_values))
    problem = build_pair_problem(training_x, groups)
    zero = np.zeros(problem.feature_count, dtype=np.float64)
    objective_initial, gradient_initial = beta_objective_gradient(problem, zero)
    known_failure = original_beta_certificate(
        problem,
        zero,
        optimizer_success=True,
        objective_initial=objective_initial,
        objective_final=objective_initial,
    )
    model = fit_ranker(training_x, groups)
    if known_failure["valid"] or not model["optimization"]["certificate"]["valid"]:
        raise RuntimeError("P25R3_ENGINEERING_STOP known-failure regression")
    return {
        "status": "PASS",
        "held_excluded": held,
        "held_target_read": False,
        "held_prediction_computed": False,
        "source_classes": names,
        "training_rows": int(len(training_x)),
        "pair_count": int(len(problem.design)),
        "pair_design_max_abs": float(np.max(np.abs(problem.design))),
        "historical_zero_gradient_l2": float(np.linalg.norm(gradient_initial)),
        "historical_zero_gradient_inf": float(np.max(np.abs(gradient_initial))),
        "historical_zero_certificate_valid": bool(known_failure["valid"]),
        "recovered_beta_norm": float(np.linalg.norm(model["beta"])),
        "recovered_optimizer": model["optimization"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def _serial_model(model: dict[str, Any]) -> dict[str, Any]:
    return json_safe_model(model)


def fit_and_score_q1(
    source_features: list[np.ndarray],
    source_values: list[np.ndarray],
    source_names: list[str],
    held_features: np.ndarray,
    held_values: np.ndarray,
) -> dict[str, Any]:
    if not (len(source_features) == len(source_values) == len(source_names)):
        raise RuntimeError("P25R3_ENGINEERING_STOP Q1 source alignment")
    model = fit_ranker(np.concatenate(source_features), list(zip(source_names, source_values)))
    score = rank_predict(model, held_features)
    metrics = q1_metrics(held_values, score)
    metrics["pearson"] = r1.pearson(score, held_values)
    return {"model": model, "score": score, "metrics": metrics}


def q1_fold(held: str, shards: dict[str, r1.Shard], out: Path) -> dict[str, Any]:
    """Frozen Q1 fold with the only change being the certified solver."""
    outer = parent.r2v2_harm.outer(held, shards)
    names = [name for name in r1.CLASSES if name != held]
    source_features: list[np.ndarray] = []
    source_values: list[np.ndarray] = []
    for name in names:
        features, _ = parent.feature_rows_for_outer(held, name, outer, shards, PARENT_OUT)
        source_features.append(features)
        source_values.append(np.asarray(parent.load_target(PARENT_OUT, name)["V"], dtype=np.float64))
    model = fit_ranker(np.concatenate(source_features), list(zip(names, source_values)))
    held_x, action = parent.feature_rows_for_outer(held, held, outer, shards, PARENT_OUT)
    # The held target is opened only after the source-only model is frozen.
    held_target = parent.load_target(PARENT_OUT, held)
    values = np.asarray(held_target["V"], dtype=np.float64)
    score = rank_predict(model, held_x)
    metric = q1_metrics(values, score)
    metric["pearson"] = r1.pearson(score, values)
    metric["positive_count"] = int(np.count_nonzero(values > parent.BENEFIT_EPS))
    metric["negative_count"] = int(np.count_nonzero(values < -parent.BENEFIT_EPS))
    metric["near_zero_count"] = int(np.count_nonzero(np.abs(values) <= parent.BENEFIT_EPS))
    metric["score_variance"] = float(np.var(score))
    optimization = model["optimization"]
    optimization["held_prediction_min"] = float(np.min(score))
    optimization["held_prediction_max"] = float(np.max(score))
    optimization["held_prediction_std"] = float(np.std(score))
    optimization["held_prediction_unique_count"] = int(np.unique(score).size)
    result = {
        "held": held,
        "outer_training": names,
        "feature_order": list(FEATURE_ORDER),
        "model": _serial_model(model),
        "metrics": metric,
        "held_nonkeep_actions": int(np.count_nonzero(action)),
        "held_count": int(len(values)),
        "target_sha256": TARGET_HASHES[held],
    }
    parent.atomic_npz(
        out / "q1" / "folds" / f"{held}.npz",
        image_index=held_target["image_index"],
        patch_index=held_target["patch_index"],
        V=values,
        score=score,
        actions=action,
        features=held_x,
    )
    parent.atomic_json(out / "q1" / "parameters" / f"{held}.json", result)
    return result


def post_execution_q1_audit(out: Path) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    max_prediction_error = 0.0
    for held in r1.CLASSES:
        parameter_path = out / "q1" / "parameters" / f"{held}.json"
        fold_path = out / "q1" / "folds" / f"{held}.npz"
        params = json.loads(parameter_path.read_text())
        raw = params["model"]
        model = {key: np.asarray(value, dtype=np.float64) if key in {"median", "iqr", "beta"} else value for key, value in raw.items()}
        with np.load(fold_path, allow_pickle=False) as artifact:
            score = np.asarray(artifact["score"], dtype=np.float64)
            reconstructed = rank_predict(model, artifact["features"])
            error = float(np.max(np.abs(reconstructed - score)))
            max_prediction_error = max(max_prediction_error, error)
            values = np.asarray(artifact["V"], dtype=np.float64)
            metric = q1_metrics(values, score)
            metric["pearson"] = r1.pearson(score, values)
            for key in ("spearman", "sign_auc", "bc20", "pearson"):
                expected = params["metrics"][key]
                actual = metric[key]
                if expected is None or actual is None:
                    if expected is not actual:
                        raise RuntimeError(f"P25R3_ENGINEERING_STOP metric null mismatch: {held}:{key}")
                elif abs(float(expected) - float(actual)) > 1e-15:
                    raise RuntimeError(f"P25R3_ENGINEERING_STOP metric mismatch: {held}:{key}")
            if params["target_sha256"] != TARGET_HASHES[held]:
                raise RuntimeError(f"P25R3_ENGINEERING_STOP target provenance: {held}")
            if not raw["optimization"]["certificate"]["valid"]:
                raise RuntimeError(f"P25R3_ENGINEERING_STOP invalid certificate: {held}")
        folds[held] = {"prediction_max_abs_error": error, "fold_sha256": sha256(fold_path), "parameter_sha256": sha256(parameter_path)}
    if max_prediction_error != 0.0:
        raise RuntimeError("P25R3_ENGINEERING_STOP serialization parity")
    result = {
        "status": "PASS",
        "folds": folds,
        "fold_count": len(folds),
        "max_prediction_abs_error": max_prediction_error,
        "target_hashes_verified": True,
        "firewall": {"mvtec": 0, "medical": 0, "additional_clip": 0, "phase2b_steps": 0},
    }
    parent.atomic_json(out / "post_execution_audit.json", result)
    return result


def _progress(out: Path, stage: str, q1_done: int, q2_done: int, overall: float, status: str, event: str, started: float) -> None:
    parent.atomic_json(
        out / "PROGRESS.json",
        {
            "current_stage": stage,
            "q1_folds_completed": q1_done,
            "q2_folds_completed": q2_done,
            "overall_progress_percent": overall,
            "last_event": event,
            "elapsed_seconds": time.perf_counter() - started,
            "status": status,
            "firewall": {"mvtec": 0, "medical": 0, "additional_clip": 0, "phase2b": 0},
        },
    )


def pre_execution_audit(out: Path) -> dict[str, Any]:
    started = time.perf_counter()
    target_audit = audit_target_artifacts()
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("START_STATE_FAILURE P25R3 branch")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PREREG_SHA, "HEAD"], cwd=ROOT).returncode != 0:
        raise RuntimeError("START_STATE_FAILURE P25R3 prereg ancestry")
    if (out / "ATTEMPT_STARTED.json").exists():
        raise RuntimeError("P25R3_ENGINEERING_STOP attempt already exists")
    parity = objective_gradient_parity_audit()
    rehearsal = synthetic_production_rehearsal()
    interface = static_interface_audit()
    regression = known_failure_regression()
    expected_gradient = 1750236.1845595562
    if parity["status"] != "PASS" or rehearsal["status"] != "PASS" or interface["status"] != "PASS":
        raise RuntimeError("P25R3_ENGINEERING_STOP numerical rehearsal")
    if abs(regression["historical_zero_gradient_l2"] - expected_gradient) > 1e-8 * expected_gradient:
        raise RuntimeError("P25R3_ENGINEERING_STOP forensic design mismatch")
    projected_q1_seconds = float(regression["elapsed_seconds"] * 12.0 * 1.25)
    result = {
        "status": "PASS",
        "p25r2_terminal_sha": P25R2_TERMINAL_SHA,
        "p25r2_execution_base_sha": P25R2_EXECUTION_BASE_SHA,
        "p25r2_prereg_sha": P25R2_PREREG_SHA,
        "forensic_sha": FORENSIC_SHA,
        "p25r3_prereg_sha": PREREG_SHA,
        "audited_head": git("rev-parse", "HEAD"),
        "runner_sha256": sha256(Path(__file__)),
        "tests_sha256": sha256(ROOT / "tests/test_sabra_cure_patch_actionability_r3.py"),
        "target_artifacts": target_audit,
        "objective_gradient_parity": parity,
        "synthetic_production_rehearsal": rehearsal,
        "static_interface_audit": interface,
        "known_failure_regression": regression,
        "runtime_benchmark": {
            "source_only_recovery_seconds": regression["elapsed_seconds"],
            "projected_q1_seconds_conservative": projected_q1_seconds,
            "projected_q2_seconds": None,
            "peak_memory": "bounded by one outer fold; measured externally in test command",
        },
        "solver": "deterministic-float64-damped-newton",
        "scipy_version_for_expit": __import__("scipy").__version__,
        "objective": "frozen P25R2 beta-space weighted pairwise logistic plus 0.5 L2",
        "relative_gradient_inf_tolerance": RELATIVE_GRADIENT_INF_TOLERANCE,
        "no_attempt_marker": True,
        "target_generation": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "firewall": {"mvtec": 0, "medical": 0, "additional_clip": 0, "phase2b_steps": 0},
    }
    parent.atomic_json(out / "pre_execution_audit.json", result)
    return result


def q2_fold(held: str, shards: dict[str, r1.Shard], out: Path) -> dict[str, Any]:
    """Inherited frozen Q2; the recovered Q1 model is consumed unchanged."""
    params = json.loads((out / "q1" / "parameters" / f"{held}.json").read_text())
    raw = params["model"]
    model = {key: np.asarray(value, dtype=np.float64) if key in {"median", "iqr", "beta"} else value for key, value in raw.items()}
    outer = parent.r2v2_harm.outer(held, shards)
    names = [name for name in r1.CLASSES if name != held]
    groups: list[dict[str, Any]] = []
    for name in names:
        features, action = parent.feature_rows_for_outer(held, name, outer, shards, PARENT_OUT)
        group = next(item for item in outer["level1"] if item["name"] == name)
        groups.append(
            {
                "name": name,
                "score": rank_predict(model, features),
                "risk": np.asarray(group["r_h"], dtype=np.float64),
                "base_action": action,
                "y": np.asarray(group["y"], dtype=np.float64),
                "V": np.asarray(parent.load_target(PARENT_OUT, name)["V"], dtype=np.float64),
            }
        )
    selected, candidates = parent.select_source_policy(groups)
    held_x, held_base_action = parent.feature_rows_for_outer(held, held, outer, shards, PARENT_OUT)
    held_score = rank_predict(model, held_x)
    panel = parent.load_panel(PARENT_OUT, held)
    if selected is None:
        held_action = np.zeros_like(held_base_action, dtype=np.int8)
        policy = {"status": "NO_ELIGIBLE_SOURCE_POLICY", "candidates": candidates, "selected": None}
    else:
        rows = panel["image_index"].astype(np.int64) * parent.PATCHES + panel["patch_index"].astype(np.int64)
        held_risk = np.asarray(outer["risk_h"], dtype=np.float64)[rows]
        chosen = (held_base_action != 0) & (held_risk <= selected["risk_threshold"]) & (held_score > selected["benefit_threshold"])
        held_action = np.where(chosen, held_base_action, 0).astype(np.int8)
        policy = {"status": "SELECTED", "candidates": candidates, "selected": selected}
    action_path = out / "q2" / "actions" / f"{held}.npz"
    parent.atomic_npz(
        action_path,
        image_path=panel["image_path"],
        image_index=panel["image_index"],
        patch_index=panel["patch_index"],
        actions=held_action,
        benefit_score=held_score,
    )
    logits, native, paths, _, masks, _, _, _ = parent.class_state(held)
    full_action = np.zeros((len(paths), parent.PATCHES), dtype=np.int8)
    full_action[panel["image_index"], panel["patch_index"]] = held_action
    from tools.sabra_car.r0_direction import evaluate_correction

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    correction = full_action.astype(np.float32) * float(parent.ALPHA * parent.MARGIN_SCALE)
    score_map, _ = evaluate_correction(logits, masks, correction, device, 4)
    deployed = parent.exact_metrics(score_map.reshape(-1), masks.reshape(-1))
    native_metric = parent.exact_metrics(native.reshape(-1), masks.reshape(-1))
    rows = panel["image_index"].astype(np.int64) * parent.PATCHES + panel["patch_index"].astype(np.int64)
    safety = parent._policy_metrics(held_action, np.asarray(outer["y"])[rows], held_base_action)
    result = {
        "held": held,
        "selection": policy,
        "metrics": {
            "native_pap": float(native_metric["pAP"]),
            "pap": float(deployed["pAP"]),
            "native_pauroc": float(native_metric["pAUROC"]),
            "pauroc": float(deployed["pAUROC"]),
            "delta_pap": float(deployed["pAP"] - native_metric["pAP"]),
        },
        "safety": safety,
        "action_sha256": sha256(action_path),
    }
    parent.atomic_json(out / "q2" / "parameters" / f"{held}.json", result)
    return result


def _target_summary(target_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "REUSED_IMMUTABLE_P25R2_TARGETS",
        "classes": target_audit["classes"],
        "class_count": target_audit["class_count"],
        "targets": target_audit["total_rows"],
        "generation_runs": 0,
        "hashes_verified": True,
    }


def render_final_decision(summary: dict[str, Any]) -> str:
    q1 = summary.get("q1", {})
    q2 = summary.get("q2", {})
    gates = q1.get("gates", {})
    gate_lines = "\n".join(f"- {name}: `{'PASS' if passed else 'FAIL'}`" for name, passed in gates.items()) or "- unavailable"
    return (
        "# P25R3 Final Decision\n\n"
        f"Status: `{summary['status']}`\n\n"
        f"Scientific validity: `{summary.get('scientific_validity', 'NON_INTERPRETABLE')}`\n\n"
        f"Attempt UUID: `{summary.get('attempt', {}).get('attempt_uuid', 'unavailable')}`\n\n"
        f"Q1 pass: `{q1.get('pass')}`\n\n"
        "Q1 frozen gates:\n\n"
        f"{gate_lines}\n\n"
        f"Q2 routing/status: `{q2.get('status', 'ENTERED_AND_REPORTED')}`\n\n"
        "P25R2 targets were reused immutably; P25R2 Q1 coefficients and predictions were not reused. "
        "MVTec and Medical were not accessed, additional CLIP forwards were zero, and Phase2B training steps were zero.\n"
    )


def write_final_decision(summary: dict[str, Any]) -> None:
    path = DOC_OUT / "P25R3_FINAL_DECISION.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(render_final_decision(summary))
    temporary.replace(path)


def post_execution_q2_audit(out: Path) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    for held in r1.CLASSES:
        parameter_path = out / "q2" / "parameters" / f"{held}.json"
        parameter = json.loads(parameter_path.read_text())
        action_path = out / "q2" / "actions" / f"{held}.npz"
        if parameter["action_sha256"] != sha256(action_path):
            raise RuntimeError(f"P25R3_ENGINEERING_STOP Q2 action hash: {held}")
        with np.load(action_path, allow_pickle=False) as artifact:
            count = len(artifact["actions"])
            if count != 2000 or len(artifact["benefit_score"]) != count or not np.all(np.isfinite(artifact["benefit_score"])):
                raise RuntimeError(f"P25R3_ENGINEERING_STOP Q2 action alignment: {held}")
        folds[held] = parameter
    recomputed = parent.evaluate_q2(folds)
    persisted = json.loads((out / "q2_summary.json").read_text())
    if recomputed != persisted:
        raise RuntimeError("P25R3_ENGINEERING_STOP Q2 aggregate recomputation")
    audit = json.loads((out / "post_execution_audit.json").read_text())
    audit["q2"] = {"status": "PASS", "fold_count": len(folds), "aggregate_recomputation": "PASS"}
    parent.atomic_json(out / "post_execution_audit.json", audit)
    return audit


def execute_once(out: Path) -> dict[str, Any]:
    if (out / "ATTEMPT_STARTED.json").exists() or (out / "summary.json").exists():
        raise RuntimeError("P25R3_ENGINEERING_STOP attempt already exists")
    audit = json.loads((out / "pre_execution_audit.json").read_text())
    if audit.get("status") != "PASS":
        raise RuntimeError("P25R3_ENGINEERING_STOP missing pre-execution audit")
    if audit.get("runner_sha256") != sha256(Path(__file__)) or audit.get("tests_sha256") != sha256(ROOT / "tests/test_sabra_cure_patch_actionability_r3.py"):
        raise RuntimeError("P25R3_ENGINEERING_STOP stale execution audit")
    if git("branch", "--show-current") != BRANCH or git("rev-parse", "HEAD") != git("rev-parse", f"origin/{BRANCH}") or git("status", "--porcelain"):
        raise RuntimeError("P25R3_ENGINEERING_STOP unpublished or dirty execution base")
    if subprocess.run(["git", "merge-base", "--is-ancestor", PREREG_SHA, "HEAD"], cwd=ROOT).returncode != 0:
        raise RuntimeError("P25R3_ENGINEERING_STOP prereg ancestry")
    target_audit = audit_target_artifacts()
    started = time.perf_counter()
    marker = {
        "status": "ATTEMPT_STARTED",
        "attempt_uuid": uuid.uuid4().hex,
        "runs": 1,
        "p25r2_terminal_sha": P25R2_TERMINAL_SHA,
        "forensic_sha": FORENSIC_SHA,
        "p25r3_prereg_sha": PREREG_SHA,
        "execution_base_sha": git("rev-parse", "HEAD"),
        "target_generation_runs": 0,
        "solver": "deterministic-float64-damped-newton",
    }
    parent.atomic_json(out / "ATTEMPT_STARTED.json", marker)
    _progress(out, "Q1", 0, 0, 75.0, "RUNNING", "marker_created", started)
    try:
        shards, provenance = r1.load_shards(True)
        q1_folds: dict[str, Any] = {}
        for index, held in enumerate(r1.CLASSES, start=1):
            q1_folds[held] = q1_fold(held, shards, out)
            _progress(out, "Q1", index, 0, 75.0 + 15.0 * index / 12.0, "RUNNING", f"q1_complete:{held}", started)
            print(json.dumps({"event": "P25R3_Q1_COMPLETE", "held": held}), flush=True)
        q1 = parent.evaluate_q1(q1_folds)
        parent.atomic_json(out / "q1_summary.json", q1)
        post_audit = post_execution_q1_audit(out)
        if not q1["pass"]:
            status = "P25_PATCH_BENEFIT_NOT_IDENTIFIABLE"
            summary = {
                "status": status,
                "scientific_validity": "VALID",
                "attempt": marker,
                "target_summary": _target_summary(target_audit),
                "q1": q1,
                "q2": {"status": "NOT_ENTERED_Q1_ROUTING_STOP"},
                "q1_folds_completed": 12,
                "q2_folds_completed": 0,
                "post_execution_audit": post_audit,
                "provenance": provenance,
                "firewall": {"mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0},
            }
            parent.atomic_json(out / "summary.json", summary)
            write_final_decision(summary)
            _progress(out, "TERMINAL", 12, 0, 100.0, "COMPLETED", status, started)
            return summary
        q2_folds: dict[str, Any] = {}
        for index, held in enumerate(r1.CLASSES, start=1):
            q2_folds[held] = q2_fold(held, shards, out)
            _progress(out, "Q2", 12, index, 90.0 + 10.0 * index / 12.0, "RUNNING", f"q2_complete:{held}", started)
            print(json.dumps({"event": "P25R3_Q2_COMPLETE", "held": held}), flush=True)
        q2 = parent.evaluate_q2(q2_folds)
        parent.atomic_json(out / "q2_summary.json", q2)
        post_audit = post_execution_q2_audit(out)
        status = "P25_PATCH_ACTIONABILITY_IDENTIFIED" if q2["pass"] else "P25_PATCH_BENEFIT_NOT_POLICY_TRANSFERABLE"
        summary = {
            "status": status,
            "scientific_validity": "VALID",
            "attempt": marker,
            "target_summary": _target_summary(target_audit),
            "q1": q1,
            "q2": q2,
            "q1_folds_completed": 12,
            "q2_folds_completed": 12,
            "post_execution_audit": post_audit,
            "provenance": provenance,
            "firewall": {"mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0},
        }
        parent.atomic_json(out / "summary.json", summary)
        write_final_decision(summary)
        _progress(out, "TERMINAL", 12, 12, 100.0, "COMPLETED", status, started)
        return summary
    except Exception as error:
        failure = {
            "status": "P25R3_ENGINEERING_STOP",
            "scientific_validity": "NON_INTERPRETABLE",
            "error_type": type(error).__name__,
            "error": str(error),
            "attempt": marker,
            "firewall": {"mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0},
        }
        parent.atomic_json(out / "ENGINEERING_FAILURE.json", failure)
        parent.atomic_json(out / "summary.json", failure)
        write_final_decision(failure)
        _progress(out, "TERMINAL", len(list((out / "q1/folds").glob("*.npz"))), len(list((out / "q2/actions").glob("*.npz"))), 100.0, "FAILED", "P25R3_ENGINEERING_STOP", started)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-execution-audit", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.pre_execution_audit == args.run_once:
        parser.error("choose exactly one action")
    out = args.output.resolve()
    result = pre_execution_audit(out) if args.pre_execution_audit else execute_once(out)
    print(json.dumps(result, indent=2, sort_keys=True, default=parent._default))


if __name__ == "__main__":
    main()
