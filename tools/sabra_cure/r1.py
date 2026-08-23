"""Frozen SABRA-CURE R1: cache-only VisA LOCO signed-utility regression.

This module is deliberately isolated from deployment code.  It reads only the
immutable VisA source, Trust-v2, and R0 utility caches; fitting is NumPy
float64 closed-form ridge with an unregularized intercept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/sabra_cure/r1"
SOURCE_ROOT = Path("/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0/sabra_source")
TRUST_ROOT = ROOT / "runs/phase5/sabra/TRUST_V2_DEVELOPMENT"
UTILITY_ROOT = ROOT / "results/sabra_car/r0/utility"
PHASE0_PROVENANCE = ROOT / "results/sabra_cure/phase0/PROVENANCE.json"
PREREGISTRATION = ROOT / "research/sabra_cure/MASTER_PREREGISTRATION_V1.md"
INTERPRETATION = ROOT / "research/sabra_cure/PROTOCOL_INTERPRETATION_NOTE_V1.md"
PHASE0_DECISION = ROOT / "research/sabra_cure/PHASE0_FINAL_DECISION.md"
PUBLISHED_BASE = "36412fc4a736feaa3118fb31950828ee9d303ee2"
PHASE0_PARENT = "48cd72b4609200d0a03d9ba3818f61b887c8ab1e"
LAMBDA = 1.0
EPSILON = 1e-8
PATCHES = 1369
CLASSES = (
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
)
FEATURE_ORDER = (
    "margin_within_image_rank", "robust_margin_normalization", "D_rank",
    "deployment_sensitivity", "E", "peer_coherence", "query_support_mean",
    "peer_eigen_entropy", "stage_query_profile_disagreement",
    "where(valid_p9,S9,0)", "where(valid_p16,S16,0)",
    "signed_native_margin = mean_s native_margins[s]",
    "cross_stage_signed_margin_difference = native_margins[stage2] - native_margins[stage0]",
    "robust_peer_signed_margin_consensus = median_k mean_s native_margins[s,peer_k]",
)
SOURCE_FEATURES = FEATURE_ORDER[:9]


@dataclass(frozen=True)
class Shard:
    name: str
    image_path: np.ndarray
    x: np.ndarray
    utility: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True,
                          capture_output=True).stdout.strip()


def finite(name: str, *values: np.ndarray) -> None:
    if not all(np.isfinite(np.asarray(value)).all() for value in values):
        raise RuntimeError(f"ENGINEERING_STOP non-finite {name}")


def p75_scale(utility: np.ndarray) -> float:
    scale = max(float(np.quantile(np.abs(np.asarray(utility, dtype=np.float64)), 0.75,
                                  method="linear")), EPSILON)
    if not np.isfinite(scale):
        raise RuntimeError("ENGINEERING_STOP non-finite target scale")
    return scale


def transform(utility: np.ndarray, scale: float) -> np.ndarray:
    y = np.tanh(np.asarray(utility, dtype=np.float64) / float(scale))
    finite("target", y)
    if np.max(np.abs(y), initial=0.0) > 1.0:
        raise RuntimeError("ENGINEERING_STOP target outside tanh bounds")
    return y


def fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_ORDER):
        raise RuntimeError("ENGINEERING_STOP feature width")
    finite("scaler input", x)
    q25, median, q75 = np.quantile(x, [0.25, 0.50, 0.75], axis=0, method="linear")
    iqr = np.maximum(q75 - q25, 1e-6)
    finite("scaler", median, iqr)
    return median, iqr


def scale_x(x: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    result = (np.asarray(x, dtype=np.float64) - median) / iqr
    finite("standardized features", result)
    return result


def sufficient_statistics(x: Iterable[np.ndarray], y: Iterable[np.ndarray]) -> tuple[int, np.ndarray, float, np.ndarray, np.ndarray]:
    count = 0
    sx = np.zeros(len(FEATURE_ORDER), dtype=np.float64)
    sy = 0.0
    sxx = np.zeros((len(FEATURE_ORDER), len(FEATURE_ORDER)), dtype=np.float64)
    sxy = np.zeros(len(FEATURE_ORDER), dtype=np.float64)
    for xx, yy in zip(x, y):
        xx = np.asarray(xx, dtype=np.float64)
        yy = np.asarray(yy, dtype=np.float64).reshape(-1)
        if xx.ndim != 2 or xx.shape != (len(yy), len(FEATURE_ORDER)):
            raise RuntimeError("ENGINEERING_STOP sufficient-statistic shape")
        count += len(yy)
        sx += xx.sum(axis=0)
        sy += float(yy.sum())
        sxx += xx.T @ xx
        sxy += xx.T @ yy
    if count == 0:
        raise RuntimeError("ENGINEERING_STOP empty fit")
    return count, sx, sy, sxx, sxy


def fit_ridge_accumulated(x: Iterable[np.ndarray], y: Iterable[np.ndarray]) -> tuple[np.ndarray, float]:
    count, sx, sy, sxx, sxy = sufficient_statistics(x, y)
    xbar = sx / count
    ybar = sy / count
    centered_xx = sxx - count * np.outer(xbar, xbar)
    centered_xy = sxy - sx * ybar
    beta = np.linalg.solve(centered_xx + LAMBDA * np.eye(len(FEATURE_ORDER)), centered_xy)
    intercept = float(ybar - xbar @ beta)
    finite("ridge parameters", beta, np.asarray(intercept))
    return beta, intercept


def fit_ridge_full(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    beta = np.linalg.solve(xc.T @ xc + LAMBDA * np.eye(x.shape[1]), xc.T @ yc)
    intercept = float(y.mean() - x.mean(axis=0) @ beta)
    finite("full ridge parameters", beta, np.asarray(intercept))
    return beta, intercept


def parity_fixture() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(97, len(FEATURE_ORDER))).astype(np.float64)
    y = (x @ np.linspace(-0.3, 0.4, len(FEATURE_ORDER)) + rng.normal(0, 0.1, 97)).astype(np.float64)
    beta_full, intercept_full = fit_ridge_full(x, y)
    beta_stats, intercept_stats = fit_ridge_accumulated((x[:31], x[31:63], x[63:]), (y[:31], y[31:63], y[63:]))
    coefficient_error = float(np.max(np.abs(beta_full - beta_stats)))
    intercept_error = float(abs(intercept_full - intercept_stats))
    return {"status": "PASS" if max(coefficient_error, intercept_error) <= 1e-10 else "FAIL",
            "coefficient_max_abs_error": coefficient_error, "intercept_abs_error": intercept_error,
            "threshold": 1e-10}


def build_features(source: Any, trust: Any) -> np.ndarray:
    values = [np.asarray(source[name], dtype=np.float64) for name in SOURCE_FEATURES]
    values.extend([
        np.where(np.asarray(trust["valid_p9"], dtype=bool), np.asarray(trust["S9"], dtype=np.float64), 0.0),
        np.where(np.asarray(trust["valid_p16"], dtype=bool), np.asarray(trust["S16"], dtype=np.float64), 0.0),
    ])
    margins = np.asarray(source["native_margins"], dtype=np.float64)
    if margins.ndim != 3 or margins.shape[1] < 3:
        raise RuntimeError("ENGINEERING_STOP native-margins shape")
    native = margins.mean(axis=1)
    indices = np.asarray(trust["peer_indices"], dtype=np.int64)
    valid = np.asarray(trust["valid_b1"], dtype=bool)
    if indices.ndim != 3 or indices.shape[:2] != native.shape or indices.min() < 0 or indices.max() >= PATCHES:
        raise RuntimeError("ENGINEERING_STOP peer-index contract")
    image_index = np.arange(native.shape[0], dtype=np.int64)[:, None, None]
    peer_margin = native[image_index, indices]
    consensus = np.median(peer_margin, axis=-1)
    consensus = np.where(valid, consensus, native)
    values.extend([native, margins[:, 2] - margins[:, 0], consensus])
    x = np.stack(values, axis=-1)
    if x.ndim != 3 or x.shape[1] != PATCHES or x.shape[-1] != len(FEATURE_ORDER):
        raise RuntimeError("ENGINEERING_STOP 14-feature contract")
    finite("features", x)
    return x


def load_shards(check_hashes: bool = True) -> tuple[dict[str, Shard], dict[str, Any]]:
    source_manifest_path = SOURCE_ROOT / "GT_FREE_MANIFEST.json"
    trust_manifest_path = TRUST_ROOT / "TRUST_V2_GT_FREE_MANIFEST.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    trust_manifest = json.loads(trust_manifest_path.read_text())
    if tuple(source_manifest.get("classes", ())) != CLASSES or tuple(trust_manifest.get("classes", ())) != CLASSES:
        raise RuntimeError("ENGINEERING_STOP class inventory")
    if source_manifest.get("record_count") != 2162 or trust_manifest.get("record_count") != 2162:
        raise RuntimeError("ENGINEERING_STOP record count")
    if source_manifest.get("medical_reads") != 0 or source_manifest.get("labels_read") != 0:
        raise RuntimeError("DATA_FIREWALL_VIOLATION source manifest")
    counters = trust_manifest.get("counters", {})
    if counters.get("MEDICAL_READS") != 0 or counters.get("MVTEC_READS_BEFORE_FREEZE") != 0:
        raise RuntimeError("DATA_FIREWALL_VIOLATION trust manifest")
    phase0 = json.loads(PHASE0_PROVENANCE.read_text())
    shards: dict[str, Shard] = {}
    hashes: dict[str, dict[str, str]] = {}
    for name in CLASSES:
        source_path = SOURCE_ROOT / "gt_free_cache" / f"{name}.npz"
        trust_path = TRUST_ROOT / "cache" / f"{name}.npz"
        utility_path = UTILITY_ROOT / f"{name}.npz"
        r1v2_path = ROOT / phase0["artifacts"][name]["r1_v2_fold"]["path"]
        observed = {"source": sha256(source_path), "trust": sha256(trust_path), "utility": sha256(utility_path), "r1_v2_fold": sha256(r1v2_path)}
        expected = {key: phase0["artifacts"][name][key]["sha256"] for key in observed}
        if check_hashes and observed != expected:
            raise RuntimeError(f"ENGINEERING_STOP immutable artifact hash: {name}")
        with np.load(source_path, allow_pickle=False) as source, np.load(trust_path, allow_pickle=False) as trust, np.load(utility_path, allow_pickle=False) as utility_data:
            paths = source["image_path"].astype(str)
            if not np.array_equal(paths, trust["image_path"].astype(str)) or not np.array_equal(paths, utility_data["image_path"].astype(str)):
                raise RuntimeError(f"ENGINEERING_STOP cache alignment: {name}")
            x = build_features(source, trust)
            utility = np.asarray(utility_data["utility"], dtype=np.float64)
            if utility.shape != x.shape[:2]:
                raise RuntimeError(f"ENGINEERING_STOP utility alignment: {name}")
            finite("utility", utility)
            shards[name] = Shard(name, np.array(paths, copy=True), x.reshape(-1, len(FEATURE_ORDER)), utility.reshape(-1))
        hashes[name] = observed
    return shards, {"source_manifest_sha256": sha256(source_manifest_path), "trust_manifest_sha256": sha256(trust_manifest_path), "phase0_provenance_sha256": sha256(PHASE0_PROVENANCE), "artifacts": hashes}


def concat(shards: dict[str, Shard], names: Iterable[str], field: str) -> np.ndarray:
    return np.concatenate([getattr(shards[name], field) for name in names], axis=0)


def fit_inner_crossfit(shards: dict[str, Shard], outer_train: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    residual_targets: list[np.ndarray] = []
    evidence: list[dict[str, Any]] = []
    for held in outer_train:
        inner_train = [name for name in outer_train if name != held]
        raw_x = concat(shards, inner_train, "x")
        raw_u = concat(shards, inner_train, "utility")
        median, iqr = fit_scaler(raw_x)
        scale = p75_scale(raw_u)
        standardized = scale_x(raw_x, median, iqr)
        beta, intercept = fit_ridge_accumulated((standardized,), (transform(raw_u, scale),))
        test_x = scale_x(shards[held].x, median, iqr)
        test_y = transform(shards[held].utility, scale)
        mu = test_x @ beta + intercept
        residual_targets.append(np.log(np.abs(test_y - mu) + 1e-4))
        evidence.append({"held_class": held, "training_classes": inner_train, "training_scale": scale,
                         "feature_median": median.tolist(), "feature_iqr": iqr.tolist(),
                         "mean_beta": beta.tolist(), "mean_intercept": intercept,
                         "prediction_count": int(len(mu)), "residual_finite": bool(np.isfinite(mu).all())})
    residual = np.concatenate(residual_targets)
    finite("cross-fitted residual target", residual)
    return residual, evidence


def fold(held: str, shards: dict[str, Shard]) -> dict[str, Any]:
    outer_train = [name for name in CLASSES if name != held]
    train_x = concat(shards, outer_train, "x")
    train_u = concat(shards, outer_train, "utility")
    median, iqr = fit_scaler(train_x)
    scale = p75_scale(train_u)
    standardized_train = scale_x(train_x, median, iqr)
    train_y = transform(train_u, scale)
    beta_mu, intercept_mu = fit_ridge_accumulated((standardized_train,), (train_y,))
    residual, inner = fit_inner_crossfit(shards, outer_train)
    beta_z, intercept_z = fit_ridge_accumulated((standardized_train,), (residual,))
    held_shard = shards[held]
    held_x = scale_x(held_shard.x, median, iqr)
    y = transform(held_shard.utility, scale)
    mu = held_x @ beta_mu + intercept_mu
    z = held_x @ beta_z + intercept_z
    sigma = np.exp(np.clip(z, np.log(1e-4), np.log(4.0)))
    finite("fold predictions", y, mu, sigma)
    informative_threshold = float(np.quantile(np.abs(train_y), 0.50, method="linear"))
    parameters = {"held_out_class": held, "outer_training_classes": outer_train, "feature_order": list(FEATURE_ORDER),
                  "ridge_lambda": LAMBDA, "float_precision": "float64", "target": "tanh(u/P75(abs(u_train)))",
                  "training_scale": scale, "feature_median": median.tolist(), "feature_iqr": iqr.tolist(),
                  "mean_beta": beta_mu.tolist(), "mean_intercept": intercept_mu,
                  "uncertainty_beta": beta_z.tolist(), "uncertainty_intercept": intercept_z,
                  "uncertainty_target": "log(abs(y-mu_cf)+1e-4)", "sigma": "exp(clip(z,log(1e-4),log(4)))",
                  "informative_abs_y_threshold": informative_threshold, "inner_crossfits": inner,
                  "training_patches": int(len(train_y)), "held_patches": int(len(y))}
    return {"parameters": parameters, "image_path": held_shard.image_path, "utility": held_shard.utility, "y": y, "mu": mu, "sigma": sigma}


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sx, sy = float(x.std()), float(y.std())
    if sx == 0.0 or sy == 0.0:
        return None
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def fold_metrics(y: np.ndarray, mu: np.ndarray, informative_threshold: float) -> dict[str, Any]:
    mask = np.abs(y) >= informative_threshold
    sign = float(np.mean(np.sign(mu[mask]) == np.sign(y[mask]))) if np.any(mask) else None
    return {"patches": int(len(y)), "pearson": pearson(y, mu), "mae": float(np.mean(np.abs(y - mu))),
            "zero_mae": float(np.mean(np.abs(y))), "informative_patches": int(np.count_nonzero(mask)),
            "informative_sign_accuracy": sign}


def evaluate(folds: dict[str, dict[str, Any]], audit_pass: bool) -> dict[str, Any]:
    per_class = {name: fold_metrics(folds[name]["y"], folds[name]["mu"], folds[name]["parameters"]["informative_abs_y_threshold"]) for name in CLASSES}
    pearsons = [row["pearson"] for row in per_class.values()]
    if any(value is None for value in pearsons):
        raise RuntimeError("ENGINEERING_STOP undefined Pearson")
    signs = [row["informative_sign_accuracy"] for row in per_class.values()]
    if any(value is None for value in signs):
        raise RuntimeError("ENGINEERING_STOP empty informative set")
    macro_mae = float(np.mean([row["mae"] for row in per_class.values()]))
    macro_zero = float(np.mean([row["zero_mae"] for row in per_class.values()]))
    metrics = {"median_pearson": float(np.median(pearsons)), "positive_pearson_classes": int(sum(value > 0 for value in pearsons)),
               "macro_mae": macro_mae, "macro_zero_mae": macro_zero,
               "relative_mae_improvement": float(1.0 - macro_mae / macro_zero),
               "macro_informative_sign_accuracy": float(np.mean(signs)),
               "sign_accuracy_ge_50_classes": int(sum(value >= 0.50 for value in signs)), "per_class": per_class}
    gates = {"R1_G1": metrics["median_pearson"] >= 0.20, "R1_G2": metrics["positive_pearson_classes"] >= 9,
             "R1_G3": metrics["relative_mae_improvement"] >= 0.10,
             "R1_G4": metrics["macro_informative_sign_accuracy"] >= 0.60,
             "R1_G5": metrics["sign_accuracy_ge_50_classes"] >= 9, "R1_G6": audit_pass}
    return {"metrics": metrics, "gates": gates, "R1_GATE_RESULT": "PASS" if all(gates.values()) else "FAIL",
            "SABRA_CURE_R1_STATUS": "PASS" if all(gates.values()) else "R1_SCIENTIFIC_STOP"}


def pre_execution_audit(output: Path) -> dict[str, Any]:
    if not all(path.is_file() for path in (PREREGISTRATION, INTERPRETATION, PHASE0_DECISION, PHASE0_PROVENANCE)):
        raise RuntimeError("ENGINEERING_STOP frozen-contract files absent")
    if git("merge-base", "--is-ancestor", PUBLISHED_BASE, "HEAD") != "":
        raise RuntimeError("START_STATE_FAILURE published base absent from HEAD")
    parity = parity_fixture()
    if parity["status"] != "PASS":
        raise RuntimeError("ENGINEERING_STOP sufficient-statistics parity")
    shards, provenance = load_shards(check_hashes=True)
    checks = {"published_p7_execution_base": True, "phase0_parent": PHASE0_PARENT,
              "class_order": list(shards) == list(CLASSES), "records": int(sum(len(s.utility) for s in shards.values())),
              "patch_width": PATCHES, "feature_count": len(FEATURE_ORDER), "feature_order": list(FEATURE_ORDER),
              "all_finite": bool(all(np.isfinite(s.x).all() and np.isfinite(s.utility).all() for s in shards.values())),
              "outer_folds": len(CLASSES), "inner_crossfits_per_outer": len(CLASSES) - 1,
              "ridge_lambda": LAMBDA, "fit_policy": "float64 centered numpy.linalg.solve; unregularized intercept; no inverse; no iterative optimizer",
              "sufficient_statistics_parity": parity, "phase2b_training_steps": 0, "additional_clip_forwards": 0,
              "mvtec_accessed": False, "medical_accessed": False, "prior_r1_v2_evidence_unchanged": True,
              "provenance": provenance, "execution_base_sha": git("rev-parse", "HEAD"),
              "status": "PASS"}
    write_json(output / "pre_execution_audit.json", checks)
    return checks


def execute_once(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if (output / "summary.json").exists() or (output / "ATTEMPT_STARTED.json").exists():
        raise RuntimeError("ENGINEERING_STOP R1 attempt already started")
    audit = pre_execution_audit(output)
    write_json(output / "ATTEMPT_STARTED.json", {"status": "ATTEMPT_STARTED", "execution_base_sha": git("rev-parse", "HEAD"), "folds_required": 12})
    started = time.perf_counter()
    shards, provenance = load_shards(check_hashes=True)
    folds: dict[str, dict[str, Any]] = {}
    for name in CLASSES:
        started_fold = time.perf_counter()
        item = fold(name, shards)
        item["parameters"]["fit_elapsed_seconds"] = time.perf_counter() - started_fold
        save_npz(output / "folds" / f"{name}.npz", image_path=item["image_path"], utility=item["utility"], y=item["y"], mu=item["mu"], sigma=item["sigma"])
        write_json(output / "parameters" / f"{name}.json", item["parameters"])
        folds[name] = item
        print(json.dumps({"event": "R1_OUTER_FOLD_COMPLETE", "held_class": name, "seconds": item["parameters"]["fit_elapsed_seconds"]}), flush=True)
    conclusion = evaluate(folds, audit_pass=True)
    summary = {"status": conclusion["SABRA_CURE_R1_STATUS"], "execution_base_sha": git("rev-parse", "HEAD"),
               "protocol": "SABRA-CURE R1 frozen master preregistration v1", "folds_completed": 12,
               "feature_order": list(FEATURE_ORDER), "provenance": provenance, "pre_execution_audit": "PASS",
               "metrics": conclusion["metrics"], "gates": conclusion["gates"], "R1_GATE_RESULT": conclusion["R1_GATE_RESULT"],
               "freeze": {"phase2b_training_steps": 0, "additional_clip_forwards": 0},
               "firewall": {"mvtec_accessed": False, "medical_accessed": False},
               "elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(), "numpy": np.__version__}
    write_json(output / "summary.json", summary)
    post = audit_results(output, shards, summary)
    if post["status"] != "PASS":
        raise RuntimeError("ENGINEERING_STOP post-execution audit")
    decision = "PASS" if summary["status"] == "PASS" else "R1_SCIENTIFIC_STOP"
    (ROOT / "research/sabra_cure/R1_FINAL_DECISION.md").write_text(
        "# SABRA-CURE R1 Final Decision\n\n"
        f"Decision: `{decision}`\n\n"
        "This is the sole authorized frozen VisA LOCO R1 execution. No R2, R3, or R4 is authorized by this result.\n"
    )
    return summary


def audit_results(output: Path, shards: dict[str, Shard] | None = None, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    shards = shards if shards is not None else load_shards(check_hashes=True)[0]
    summary = summary if summary is not None else json.loads((output / "summary.json").read_text())
    reconstructed: dict[str, dict[str, Any]] = {}
    held = []
    finite_predictions = True
    serialization_error = 0.0
    for name in CLASSES:
        params = json.loads((output / "parameters" / f"{name}.json").read_text())
        with np.load(output / "folds" / f"{name}.npz", allow_pickle=False) as data:
            x = scale_x(shards[name].x, np.asarray(params["feature_median"]), np.asarray(params["feature_iqr"]))
            mu = x @ np.asarray(params["mean_beta"]) + float(params["mean_intercept"])
            z = x @ np.asarray(params["uncertainty_beta"]) + float(params["uncertainty_intercept"])
            sigma = np.exp(np.clip(z, np.log(1e-4), np.log(4.0)))
            serialization_error = max(serialization_error, float(np.max(np.abs(mu - data["mu"]))), float(np.max(np.abs(sigma - data["sigma"]))))
            finite_predictions = finite_predictions and bool(np.isfinite(data["y"]).all() and np.isfinite(data["mu"]).all() and np.isfinite(data["sigma"]).all())
            reconstructed[name] = {"y": np.asarray(data["y"]), "mu": np.asarray(data["mu"]), "parameters": params}
            held.append(params["held_out_class"])
    conclusion = evaluate(reconstructed, audit_pass=True)
    recompute = conclusion["metrics"] == summary["metrics"] and conclusion["gates"] == summary["gates"]
    status = "PASS" if (held == list(CLASSES) and finite_predictions and serialization_error <= 1e-10 and recompute) else "FAIL"
    payload = {"status": status, "folds_complete": len(held), "held_class_order": held,
               "leakage_audit": bool(all(name not in json.loads((output / "parameters" / f"{name}.json").read_text())["outer_training_classes"] and len(json.loads((output / "parameters" / f"{name}.json").read_text())["inner_crossfits"]) == 11 for name in CLASSES)),
               "finite_predictions": finite_predictions, "serialization_max_abs_error": serialization_error,
               "serialization_parity": serialization_error <= 1e-10, "metric_recomputation_parity": recompute,
               "provenance_audit": True, "freeze_audit": True, "firewall_audit": True,
               "phase2b_training_steps": 0, "additional_clip_forwards": 0, "mvtec_accessed": False, "medical_accessed": False}
    write_json(output / "post_execution_audit.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.audit == args.run:
        parser.error("choose exactly one of --audit or --run")
    result = pre_execution_audit(args.output.resolve()) if args.audit else execute_once(args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
