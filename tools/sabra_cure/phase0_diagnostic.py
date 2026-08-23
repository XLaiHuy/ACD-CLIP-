"""Deterministic, cache-only SABRA-CURE Phase-0 diagnostics.

This module reads only frozen VisA source-side caches and existing R0/R1
artifacts. It performs no CLIP forward, training, or target-dataset access.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from tools.sabra_car.r1_common import (
    EXPECTED_CLASSES,
    FEATURE_ORDER,
    THRESHOLDS,
    EPSILON,
    stable_argmax_predictions,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
PARENT_SHA = "48cd72b4609200d0a03d9ba3818f61b887c8ab1e"
SOURCE_ROOT = Path("/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0/sabra_source")
TRUST_ROOT = ROOT / "runs/phase5/sabra/TRUST_V2_DEVELOPMENT"
UTILITY_ROOT = ROOT / "results/sabra_car/r0/utility"
R1_ROOT = ROOT / "results/sabra_car/r1_v2_newton_cholesky"
OUTPUT = ROOT / "results/sabra_cure/phase0"
CANDIDATE_ORDER = (
    "signed_native_margin",
    "cross_stage_signed_margin_difference",
    "query_minus_peer_signed_margin",
    "robust_peer_signed_margin_consensus",
    "signed_relational_residual",
)
ACTION_NAMES = {-1: "SUPPRESS", 0: "KEEP", 1: "BOOST"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    sx = float(x.std())
    sy = float(y.std())
    if sx == 0.0 or sy == 0.0:
        return None
    return scalar(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))


def numeric_stats(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = np.quantile(x, [0.01, 0.25, 0.50, 0.75, 0.99])
    return {
        "count": int(x.size),
        "finite": bool(np.isfinite(x).all()),
        "min": scalar(x.min()),
        "max": scalar(x.max()),
        "mean": scalar(x.mean()),
        "variance": scalar(x.var()),
        "q01": scalar(quantiles[0]),
        "q25": scalar(quantiles[1]),
        "q50": scalar(quantiles[2]),
        "q75": scalar(quantiles[3]),
        "q99": scalar(quantiles[4]),
        "iqr": scalar(quantiles[3] - quantiles[1]),
    }


def distribution(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return {"count": 0, "mean": None, "q10": None, "q25": None, "q50": None, "q75": None, "q90": None}
    q = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "count": int(x.size),
        "mean": scalar(x.mean()),
        "q10": scalar(q[0]),
        "q25": scalar(q[1]),
        "q50": scalar(q[2]),
        "q75": scalar(q[3]),
        "q90": scalar(q[4]),
    }


def confusion(oracle: np.ndarray, prediction: np.ndarray) -> list[list[int]]:
    labels = (-1, 0, 1)
    return [
        [int(np.count_nonzero((oracle == actual) & (prediction == predicted))) for predicted in labels]
        for actual in labels
    ]


def acted_mask(prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.int8)
    result = np.empty(prediction.shape, dtype=bool)
    np.not_equal(prediction, 0, out=result)
    return result


def opposite_mask(prediction: np.ndarray, oracle: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.int8)
    oracle = np.asarray(oracle, dtype=np.int8)
    product = np.empty(prediction.shape, dtype=np.int8)
    np.multiply(prediction, oracle, out=product)
    result = np.empty(prediction.shape, dtype=bool)
    np.equal(product, -1, out=result)
    return result


def action_metrics(oracle: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    oracle = np.asarray(oracle, dtype=np.int8).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.int8).reshape(-1)
    acted = acted_mask(prediction)
    opposite = opposite_mask(prediction, oracle)

    def precision_recall(action: int) -> dict[str, Any]:
        predicted = prediction == action
        actual = oracle == action
        tp = int(np.count_nonzero(predicted & actual))
        return {
            "precision": scalar(tp / np.count_nonzero(predicted)) if np.any(predicted) else None,
            "recall": scalar(tp / np.count_nonzero(actual)) if np.any(actual) else None,
            "true_positive": tp,
            "predicted": int(np.count_nonzero(predicted)),
            "oracle": int(np.count_nonzero(actual)),
        }

    oracle_keep = oracle == 0
    predicted_keep = prediction == 0
    keep_tp = int(np.count_nonzero(oracle_keep & predicted_keep))
    return {
        "patches": int(oracle.size),
        "coverage": scalar(np.mean(acted)),
        "opposite_sign_errors": int(np.count_nonzero(opposite)),
        "opposite_sign_rate": scalar(np.mean(opposite[acted])) if np.any(acted) else None,
        "boost": precision_recall(1),
        "suppress": precision_recall(-1),
        "keep": {
            "oracle_fraction": scalar(np.mean(oracle_keep)),
            "predicted_fraction": scalar(np.mean(predicted_keep)),
            "precision": scalar(keep_tp / np.count_nonzero(predicted_keep)) if np.any(predicted_keep) else None,
            "recall": scalar(keep_tp / np.count_nonzero(oracle_keep)) if np.any(oracle_keep) else None,
        },
        "confusion_matrix": confusion(oracle, prediction),
        "confusion_labels": ["SUPPRESS", "KEEP", "BOOST"],
    }


def threshold_prediction(prediction: np.ndarray, confidence: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(confidence >= threshold, prediction, 0).astype(np.int8)


def risk_row(oracle: np.ndarray, prediction: np.ndarray, confidence: np.ndarray, threshold: float) -> dict[str, Any]:
    action = threshold_prediction(prediction, confidence, threshold)
    metrics = action_metrics(oracle, action)
    return {
        "threshold": scalar(threshold),
        "coverage": metrics["coverage"],
        "opposite_sign_rate": metrics["opposite_sign_rate"],
        "opposite_sign_errors": metrics["opposite_sign_errors"],
        "acted_patches": int(np.count_nonzero(action)),
    }


def binned_risk(values: np.ndarray, opposite: np.ndarray, acted: np.ndarray) -> list[dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    boundaries = np.quantile(values, [0.0, 0.25, 0.50, 0.75, 1.0])
    rows: list[dict[str, Any]] = []
    for index in range(4):
        lo, hi = boundaries[index], boundaries[index + 1]
        selected = (values >= lo) & ((values <= hi) if index == 3 else (values < hi))
        active = np.empty_like(selected)
        np.logical_and(selected, acted, out=active)
        rows.append({
            "quartile": index + 1,
            "lower": scalar(lo),
            "upper": scalar(hi),
            "patch_fraction": scalar(np.mean(selected)),
            "coverage": scalar(np.mean(acted[selected])) if np.any(selected) else None,
            "opposite_sign_rate_when_acted": scalar(np.mean(opposite[active])) if np.any(active) else None,
        })
    return rows


def spatial_concentration(opposite: np.ndarray, acted: np.ndarray) -> dict[str, Any]:
    opposite_grid = np.asarray(opposite, dtype=bool).reshape(-1, 37, 37)
    acted_grid = np.asarray(acted, dtype=bool).reshape(-1, 37, 37)
    horizontal_valid = np.logical_and(acted_grid[:, :, :-1], acted_grid[:, :, 1:])
    vertical_valid = np.logical_and(acted_grid[:, :-1, :], acted_grid[:, 1:, :])
    valid_pairs = int(horizontal_valid.sum() + vertical_valid.sum())
    horizontal_pair = np.empty_like(horizontal_valid)
    vertical_pair = np.empty_like(vertical_valid)
    np.logical_and(opposite_grid[:, :, :-1], opposite_grid[:, :, 1:], out=horizontal_pair)
    np.logical_and(horizontal_pair, horizontal_valid, out=horizontal_pair)
    np.logical_and(opposite_grid[:, :-1, :], opposite_grid[:, 1:, :], out=vertical_pair)
    np.logical_and(vertical_pair, vertical_valid, out=vertical_pair)
    error_pairs = int(horizontal_pair.sum() + vertical_pair.sum())
    active = acted_grid
    base = float(opposite_grid[active].mean()) if np.any(active) else 0.0
    pair_rate = error_pairs / valid_pairs if valid_pairs else 0.0
    expected_pair = base * base
    return {
        "acted_patch_opposite_rate": scalar(base),
        "adjacent_acted_pairs": valid_pairs,
        "adjacent_opposite_pairs": error_pairs,
        "adjacent_opposite_pair_rate": scalar(pair_rate),
        "independent_expected_pair_rate": scalar(expected_pair),
        "adjacency_enrichment": scalar(pair_rate / expected_pair) if expected_pair > 0 else None,
    }


def class_eta_squared(class_arrays: list[np.ndarray]) -> float | None:
    counts = np.asarray([item.size for item in class_arrays], dtype=np.float64)
    means = np.asarray([float(item.mean()) for item in class_arrays], dtype=np.float64)
    total = float(counts.sum())
    grand = float(np.dot(counts, means) / total)
    between = float(np.dot(counts, (means - grand) ** 2))
    within = float(sum(np.square(item.astype(np.float64) - mean).sum() for item, mean in zip(class_arrays, means)))
    return scalar(between / (between + within)) if between + within > 0 else None


def sign_stability(values: list[float | None]) -> float:
    signs = [int(np.sign(item)) for item in values if item is not None and item != 0.0]
    if not signs:
        return 0.0
    return max(signs.count(-1), signs.count(1)) / len(signs)


def partial_corr_from_matrix(matrix: np.ndarray, x_index: int, y_index: int, controls: list[int]) -> float | None:
    indices = [x_index, y_index, *controls]
    sub = matrix[np.ix_(indices, indices)]
    precision = np.linalg.pinv(sub)
    denominator = float(np.sqrt(max(precision[0, 0] * precision[1, 1], 0.0)))
    return scalar(-precision[0, 1] / denominator) if denominator > 0 else None


def confidence_opposite_corr(
    prediction: np.ndarray, oracle: np.ndarray, confidence: np.ndarray
) -> float | None:
    acted = acted_mask(prediction)
    opposite = opposite_mask(prediction, oracle)
    return corr(np.asarray(confidence)[acted], opposite[acted].astype(np.float32))


def transform_target(utility: np.ndarray, scale: float, name: str) -> np.ndarray:
    normalized = np.asarray(utility, dtype=np.float64) / scale
    if name == "T1_raw_clipped_robust_scaled":
        return np.clip(normalized, -5.0, 5.0)
    if name == "T2_tanh_robust_scaled":
        return np.tanh(normalized)
    if name == "T3_signed_log_compressed":
        return np.sign(normalized) * np.log1p(np.abs(normalized))
    raise ValueError(name)


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    mean_x = x64.mean(axis=0)
    mean_y = float(y64.mean())
    centered = x64 - mean_x
    gram = centered.T @ centered
    beta = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), centered.T @ (y64 - mean_y))
    intercept = mean_y - float(mean_x @ beta)
    return beta, intercept


def deterministic_probe(
    class_features: list[np.ndarray],
    class_utility: list[np.ndarray],
    feature_names: list[str],
    selected_target: str,
) -> dict[str, Any]:
    count_per_class = 2048
    subset_x: list[np.ndarray] = []
    subset_u: list[np.ndarray] = []
    subset_class: list[str] = []
    subset_indices: dict[str, list[int]] = {}
    for class_name, x, utility in zip(EXPECTED_CLASSES, class_features, class_utility):
        indices = np.linspace(0, len(utility) - 1, min(count_per_class, len(utility)), dtype=np.int64)
        subset_indices[class_name] = [int(indices[0]), int(indices[-1]), int(len(indices))]
        subset_x.append(x[indices])
        subset_u.append(utility[indices])
        subset_class.extend([class_name] * len(indices))
    x_all = np.concatenate(subset_x, axis=0).astype(np.float64)
    u_all = np.concatenate(subset_u, axis=0).astype(np.float64)
    class_all = np.asarray(subset_class)
    fold_rows: list[dict[str, Any]] = []
    for held_out in EXPECTED_CLASSES:
        train = class_all != held_out
        test = ~train
        scale = max(float(np.quantile(np.abs(u_all[train]), 0.75)), EPSILON)
        y_all = transform_target(u_all, scale, selected_target)
        median = np.median(x_all[train], axis=0)
        iqr = np.quantile(x_all[train], 0.75, axis=0) - np.quantile(x_all[train], 0.25, axis=0)
        iqr = np.maximum(iqr, 1e-6)
        train_x = (x_all[train] - median) / iqr
        test_x = (x_all[test] - median) / iqr
        beta, intercept = ridge_fit(train_x, y_all[train])
        train_prediction = train_x @ beta + intercept
        prediction = test_x @ beta + intercept
        residual = np.abs(y_all[test] - prediction)
        train_residual = np.abs(y_all[train] - train_prediction)
        uncertainty_beta, uncertainty_intercept = ridge_fit(
            train_x, np.log(train_residual + 1e-4)
        )
        predicted_uncertainty = np.exp(test_x @ uncertainty_beta + uncertainty_intercept)
        median_uncertainty = float(np.median(predicted_uncertainty))
        low = predicted_uncertainty <= median_uncertainty
        informative = np.abs(y_all[test]) >= np.quantile(np.abs(y_all[test]), 0.50)
        fold_rows.append({
            "held_out_class": held_out,
            "rows": int(np.count_nonzero(test)),
            "target_scale": scalar(scale),
            "mean_target_pearson": corr(prediction, y_all[test]),
            "informative_sign_accuracy": scalar(np.mean(np.sign(prediction[informative]) == np.sign(y_all[test][informative]))) if np.any(informative) else None,
            "mean_absolute_error": scalar(residual.mean()),
            "uncertainty_residual_pearson": corr(predicted_uncertainty, residual),
            "low_uncertainty_mae": scalar(residual[low].mean()) if np.any(low) else None,
            "high_uncertainty_mae": scalar(residual[~low].mean()) if np.any(~low) else None,
        })
    correlations = [row["mean_target_pearson"] for row in fold_rows if row["mean_target_pearson"] is not None]
    uncertainty = [row["uncertainty_residual_pearson"] for row in fold_rows if row["uncertainty_residual_pearson"] is not None]
    low_better = [
        row["low_uncertainty_mae"] < row["high_uncertainty_mae"]
        for row in fold_rows
        if row["low_uncertainty_mae"] is not None and row["high_uncertainty_mae"] is not None
    ]
    return {
        "label": "PROBE_ONLY_NOT_SCIENTIFIC_RESULT",
        "purpose": "pathology and formulation discrimination only",
        "seed": 0,
        "subset_rule": "2048 evenly spaced flattened patches per VisA class",
        "subset_contract": subset_indices,
        "feature_order": feature_names,
        "target": selected_target,
        "model": "closed-form ridge mean plus separate ridge log-absolute-residual probe",
        "alpha": 1.0,
        "folds": fold_rows,
        "summary": {
            "median_mean_target_pearson": scalar(np.median(correlations)),
            "positive_mean_target_correlation_folds": int(sum(item > 0 for item in correlations)),
            "median_uncertainty_residual_pearson": scalar(np.median(uncertainty)),
            "low_uncertainty_has_lower_mae_folds": int(sum(low_better)),
            "folds": len(fold_rows),
        },
        "cannot_satisfy_scientific_gate": True,
    }


def run() -> dict[str, Any]:
    if subprocess.check_output(["git", "merge-base", "--is-ancestor", PARENT_SHA, "HEAD"], cwd=ROOT).strip():
        raise AssertionError("unexpected merge-base output")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_manifest = SOURCE_ROOT / "GT_FREE_MANIFEST.json"
    trust_manifest = TRUST_ROOT / "TRUST_V2_GT_FREE_MANIFEST.json"
    source_contract = json.loads(source_manifest.read_text())
    trust_contract = json.loads(trust_manifest.read_text())
    if tuple(source_contract["classes"]) != EXPECTED_CLASSES or tuple(trust_contract["classes"]) != EXPECTED_CLASSES:
        raise RuntimeError("class contract mismatch")
    if source_contract["record_count"] != 2162 or trust_contract["record_count"] != 2162:
        raise RuntimeError("record count mismatch")
    if source_contract["medical_reads"] != 0 or source_contract["labels_read"] != 0:
        raise RuntimeError("source firewall contract failed")
    if trust_contract["counters"]["MEDICAL_READS"] != 0 or trust_contract["counters"]["MVTEC_READS_BEFORE_FREEZE"] != 0:
        raise RuntimeError("trust firewall contract failed")

    class_existing: list[np.ndarray] = []
    class_candidates: list[np.ndarray] = []
    class_utility: list[np.ndarray] = []
    class_oracle: list[np.ndarray] = []
    class_prediction: list[np.ndarray] = []
    class_confidence: list[np.ndarray] = []
    class_paths: list[np.ndarray] = []
    provenance: dict[str, Any] = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "parent_sha": PARENT_SHA,
        "class_order": list(EXPECTED_CLASSES),
        "feature_order": list(FEATURE_ORDER),
        "candidate_order": list(CANDIDATE_ORDER),
        "records": 2162,
        "patch_width": 1369,
        "source_manifest": {"path": str(source_manifest), "sha256": sha256_file(source_manifest)},
        "trust_manifest": {"path": str(trust_manifest.relative_to(ROOT)), "sha256": sha256_file(trust_manifest)},
        "artifacts": {},
        "mvtec_access_count": 0,
        "medical_access_count": 0,
        "phase2b_training_steps": 0,
        "new_clip_forwards": 0,
    }
    spatial_by_class: dict[str, Any] = {}
    for class_name in EXPECTED_CLASSES:
        source_path = SOURCE_ROOT / "gt_free_cache" / f"{class_name}.npz"
        trust_path = TRUST_ROOT / "cache" / f"{class_name}.npz"
        utility_path = UTILITY_ROOT / f"{class_name}.npz"
        fold_path = R1_ROOT / "folds" / f"{class_name}.npz"
        provenance["artifacts"][class_name] = {
            "source": {"path": str(source_path), "sha256": sha256_file(source_path)},
            "trust": {"path": str(trust_path.relative_to(ROOT)), "sha256": sha256_file(trust_path)},
            "utility": {"path": str(utility_path.relative_to(ROOT)), "sha256": sha256_file(utility_path)},
            "r1_v2_fold": {"path": str(fold_path.relative_to(ROOT)), "sha256": sha256_file(fold_path)},
        }
        with (
            np.load(source_path, allow_pickle=False) as source,
            np.load(trust_path, allow_pickle=False) as trust,
            np.load(utility_path, allow_pickle=False) as utility_data,
            np.load(fold_path, allow_pickle=False) as fold,
        ):
            paths = source["image_path"].astype(str)
            if not np.array_equal(paths, trust["image_path"].astype(str)):
                raise RuntimeError(f"source/trust path mismatch: {class_name}")
            if not np.array_equal(paths, utility_data["image_path"].astype(str)):
                raise RuntimeError(f"source/utility path mismatch: {class_name}")
            if not np.array_equal(paths, fold["image_path"].astype(str)):
                raise RuntimeError(f"source/R1 path mismatch: {class_name}")
            utility = np.asarray(utility_data["utility"], dtype=np.float32)
            oracle = np.where(utility > EPSILON, 1, np.where(utility < -EPSILON, -1, 0)).astype(np.int8)
            if not np.array_equal(oracle, fold["oracle_action"].astype(np.int8)):
                raise RuntimeError(f"utility/action mismatch: {class_name}")
            existing = np.stack([
                np.asarray(source[name], dtype=np.float32)
                for name in FEATURE_ORDER[:9]
            ] + [
                np.where(trust["valid_p9"], trust["S9"], 0.0).astype(np.float32),
                np.where(trust["valid_p16"], trust["S16"], 0.0).astype(np.float32),
            ], axis=-1)
            margins = np.asarray(source["native_margins"], dtype=np.float32)
            mean_margin = margins.mean(axis=1)
            peer_indices = np.asarray(trust["peer_indices"], dtype=np.int64)
            if peer_indices.min() < 0 or peer_indices.max() >= 1369:
                raise RuntimeError(f"peer index out of range: {class_name}")
            image_index = np.arange(len(paths), dtype=np.int64)[:, None, None]
            peer_margins = mean_margin[image_index, peer_indices]
            peer_mean = peer_margins.mean(axis=-1)
            peer_median = np.median(peer_margins, axis=-1)
            valid = np.asarray(trust["valid_b1"], dtype=bool)
            peer_mean = np.where(valid, peer_mean, mean_margin)
            peer_median = np.where(valid, peer_median, mean_margin)
            candidates = np.stack([
                mean_margin,
                margins[:, 2] - margins[:, 0],
                mean_margin - peer_mean,
                peer_median,
                mean_margin - peer_median,
            ], axis=-1).astype(np.float32)
            probability = np.asarray(fold["probability"], dtype=np.float32)
            classes = np.asarray(fold["classes"], dtype=np.int8)
            if classes.tolist() != [-1, 0, 1]:
                raise RuntimeError(f"R1 class order mismatch: {class_name}")
            prediction, confidence = stable_argmax_predictions(
                probability.reshape(-1, 3), classes
            )
            prediction = prediction.reshape(oracle.shape)
            confidence = confidence.reshape(oracle.shape)
            if not all(np.isfinite(item).all() for item in (existing, candidates, utility, probability, confidence)):
                raise RuntimeError(f"non-finite diagnostic array: {class_name}")
            if existing.shape[:2] != utility.shape or candidates.shape[:2] != utility.shape:
                raise RuntimeError(f"patch alignment mismatch: {class_name}")
            spatial_by_class[class_name] = spatial_concentration(
                opposite_mask(prediction, oracle), acted_mask(prediction)
            )
            class_existing.append(np.array(existing.reshape(-1, len(FEATURE_ORDER)), copy=True))
            class_candidates.append(np.array(candidates.reshape(-1, len(CANDIDATE_ORDER)), copy=True))
            class_utility.append(np.array(utility.reshape(-1), copy=True))
            class_oracle.append(np.array(oracle.reshape(-1), copy=True))
            class_prediction.append(np.array(prediction.reshape(-1), copy=True))
            class_confidence.append(np.array(confidence.reshape(-1), copy=True))
            class_paths.append(paths)

    existing_all = np.concatenate(class_existing)
    candidate_all = np.concatenate(class_candidates)
    utility_all = np.concatenate(class_utility)
    oracle_all = np.concatenate(class_oracle)
    prediction_all = np.concatenate(class_prediction)
    confidence_all = np.concatenate(class_confidence)
    all_features = np.concatenate([existing_all, candidate_all], axis=1)
    all_names = [*FEATURE_ORDER, *CANDIDATE_ORDER]
    opposite_all = opposite_mask(prediction_all, oracle_all)
    acted_all = acted_mask(prediction_all)
    published_selection = json.loads((R1_ROOT / "selection.json").read_text())
    published_counts = {
        str(value): int(np.count_nonzero(prediction_all == value)) for value in (-1, 0, 1)
    }
    if published_counts != published_selection["unfiltered_argmax_action_counts"]:
        raise RuntimeError(
            f"diagnostic stable-argmax count parity failed: observed={published_counts} "
            f"expected={published_selection['unfiltered_argmax_action_counts']}"
        )
    published_rows = {str(row["threshold"]): row for row in published_selection["threshold_rows"]}
    for threshold in (None, *THRESHOLDS):
        key = "unfiltered" if threshold is None else str(threshold)
        row = action_metrics(
            oracle_all,
            prediction_all if threshold is None else threshold_prediction(
                prediction_all, confidence_all, float(threshold)
            ),
        )
        expected = published_rows[key]
        if not np.isclose(row["coverage"], expected["coverage"], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"published R1-v2 coverage parity failed: {key}")
        if not np.isclose(row["opposite_sign_rate"], expected["opposite_sign_rate"], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"published R1-v2 risk parity failed: {key}")

    risk_thresholds: dict[str, Any] = {}
    per_class_thresholds: dict[str, Any] = {}
    for threshold in ("unfiltered", *THRESHOLDS):
        action = prediction_all if threshold == "unfiltered" else threshold_prediction(
            prediction_all, confidence_all, float(threshold)
        )
        risk_thresholds[str(threshold)] = action_metrics(oracle_all, action)
        per_class_thresholds[str(threshold)] = {
            class_name: action_metrics(
                class_oracle[index],
                class_prediction[index] if threshold == "unfiltered" else threshold_prediction(
                    class_prediction[index], class_confidence[index], float(threshold)
                ),
            )
            for index, class_name in enumerate(EXPECTED_CLASSES)
        }
    confidence_groups = {
        "correct_signed_actions": distribution(confidence_all[(prediction_all == oracle_all) & (oracle_all != 0)]),
        "opposite_sign_errors": distribution(confidence_all[opposite_all]),
        "oracle_keep": distribution(confidence_all[oracle_all == 0]),
        "false_boost": distribution(confidence_all[(prediction_all == 1) & (oracle_all != 1)]),
        "false_suppress": distribution(confidence_all[(prediction_all == -1) & (oracle_all != -1)]),
    }
    dense_thresholds = np.linspace(float(confidence_all.min()), 1.0, 201)
    dense_curve = [risk_row(oracle_all, prediction_all, confidence_all, float(item)) for item in dense_thresholds]
    with (OUTPUT / "risk_coverage_posthoc_diagnostic.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(dense_curve[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(dense_curve)
    confidence_risk_global = corr(confidence_all[acted_all], opposite_all[acted_all].astype(np.float32))
    confidence_risk_by_class = {
        class_name: confidence_opposite_corr(
            class_prediction[index], class_oracle[index], class_confidence[index]
        )
        for index, class_name in enumerate(EXPECTED_CLASSES)
    }
    feature_slices = {name: existing_all[:, index] for index, name in enumerate(FEATURE_ORDER)}
    concentration = {
        "class": {
            class_name: action_metrics(class_oracle[index], class_prediction[index])
            for index, class_name in enumerate(EXPECTED_CLASSES)
        },
        "low_high_margin_regions": binned_risk(
            feature_slices["robust_margin_normalization"], opposite_all, acted_all
        ),
        "relational_disagreement_regions": binned_risk(
            feature_slices["stage_query_profile_disagreement"], opposite_all, acted_all
        ),
        "trust_evidence_regions": binned_risk(
            0.5 * (feature_slices["supported_p9_stability"] + feature_slices["supported_p16_stability"]),
            opposite_all,
            acted_all,
        ),
        "stability_evidence_regions": binned_risk(
            np.minimum(feature_slices["supported_p9_stability"], feature_slices["supported_p16_stability"]),
            opposite_all,
            acted_all,
        ),
        "spatial": spatial_by_class,
    }
    risk_audit = {
        "evidence_class": "DERIVED",
        "source": "existing committed R1-v2 OOF probabilities and R0 oracle actions",
        "action_order": ["SUPPRESS", "KEEP", "BOOST"],
        "global_and_threshold_conditioned": risk_thresholds,
        "per_class_and_threshold_conditioned": per_class_thresholds,
        "confidence_distributions": confidence_groups,
        "confidence_opposite_error_correlation": {
            "global_acted_patches": confidence_risk_global,
            "per_class": confidence_risk_by_class,
            "positive_means_confidence_associated_with_more_opposite_sign_error": True,
        },
        "dense_curve": {
            "label": "POST_HOC_DIAGNOSTIC_ONLY",
            "path": "results/sabra_cure/phase0/risk_coverage_posthoc_diagnostic.csv",
            "points": len(dense_curve),
            "not_for_threshold_selection": True,
        },
        "concentration": concentration,
        "oracle_action_counts": {
            ACTION_NAMES[action]: int(np.count_nonzero(oracle_all == action)) for action in (-1, 0, 1)
        },
        "predicted_action_counts": {
            ACTION_NAMES[action]: int(np.count_nonzero(prediction_all == action)) for action in (-1, 0, 1)
        },
    }
    write_json(OUTPUT / "RISK_DIAGNOSTIC.json", risk_audit)

    utility_folds: dict[str, Any] = {}
    transform_names = (
        "T1_raw_clipped_robust_scaled",
        "T2_tanh_robust_scaled",
        "T3_signed_log_compressed",
    )
    transform_folds: dict[str, list[dict[str, Any]]] = {name: [] for name in transform_names}
    for held_index, held_out in enumerate(EXPECTED_CLASSES):
        utility = class_utility[held_index].astype(np.float64)
        absolute = np.abs(utility)
        q = np.quantile(absolute, [0.50, 0.75, 0.90, 0.95, 0.99])
        utility_folds[held_out] = {
            "patches": int(utility.size),
            "finite": bool(np.isfinite(utility).all()),
            "P50_abs": scalar(q[0]),
            "P75_abs": scalar(q[1]),
            "P90_abs": scalar(q[2]),
            "P95_abs": scalar(q[3]),
            "P99_abs": scalar(q[4]),
            "fraction_near_zero_abs_le_1e-8": scalar(np.mean(absolute <= EPSILON)),
            "positive_fraction": scalar(np.mean(utility > EPSILON)),
            "negative_fraction": scalar(np.mean(utility < -EPSILON)),
            "heavy_tail_P99_over_P75": scalar(q[4] / q[1]) if q[1] > 0 else None,
        }
        training = np.concatenate([item for index, item in enumerate(class_utility) if index != held_index])
        scale = max(float(np.quantile(np.abs(training), 0.75)), EPSILON)
        for name in transform_names:
            transformed = transform_target(utility, scale, name)
            abs_t = np.abs(transformed)
            tq = np.quantile(abs_t, [0.50, 0.75, 0.90, 0.95, 0.99])
            transform_folds[name].append({
                "held_out_class": held_out,
                "training_only_scale": scalar(scale),
                "finite": bool(np.isfinite(transformed).all()),
                "sign_preservation": scalar(np.mean(np.sign(transformed) == np.sign(utility))),
                "P50_abs": scalar(tq[0]),
                "P75_abs": scalar(tq[1]),
                "P90_abs": scalar(tq[2]),
                "P95_abs": scalar(tq[3]),
                "P99_abs": scalar(tq[4]),
                "saturation_fraction": scalar(np.mean(abs_t >= (4.999999 if name.startswith("T1") else (0.99 if name.startswith("T2") else np.inf)))),
            })
    transform_summary: dict[str, Any] = {}
    for name, rows in transform_folds.items():
        q90 = np.asarray([row["P90_abs"] for row in rows], dtype=np.float64)
        tail = np.asarray([
            row["P99_abs"] / row["P90_abs"] if row["P90_abs"] and row["P90_abs"] > 0 else np.nan
            for row in rows
        ])
        transform_summary[name] = {
            "all_finite": all(row["finite"] for row in rows),
            "minimum_sign_preservation": min(row["sign_preservation"] for row in rows),
            "P90_cross_fold_cv": scalar(q90.std() / q90.mean()) if q90.mean() > 0 else None,
            "median_P99_over_P90": scalar(np.nanmedian(tail)),
            "max_saturation_fraction": max(row["saturation_fraction"] for row in rows),
            "folds": rows,
        }
    selected_target = "T2_tanh_robust_scaled"
    target_stable = (
        transform_summary[selected_target]["all_finite"]
        and transform_summary[selected_target]["minimum_sign_preservation"] == 1.0
        and transform_summary[selected_target]["P90_cross_fold_cv"] is not None
        and transform_summary[selected_target]["P90_cross_fold_cv"] <= 0.50
        and transform_summary[selected_target]["max_saturation_fraction"] <= 0.25
    )
    utility_audit = {
        "evidence_class": "DERIVED",
        "source_definition": "u_q = -dL/d(delta_q)",
        "finite_difference_parity_source": "results/sabra_car/r0/utility_summary.json",
        "alignment_pass": True,
        "class_order": list(EXPECTED_CLASSES),
        "patch_inventory": int(utility_all.size),
        "folds": utility_folds,
        "target_candidates": transform_summary,
        "selected_target": selected_target if target_stable else None,
        "selected_scale_rule": "P75(abs(training-fold utility)), floored at 1e-8; fitted on 11 training classes only",
        "selection_reason": "bounded monotone odd transform; exact sign preservation; controlled tails; stable cross-fold scale",
        "target_stable": target_stable,
        "mvtec_access_count": 0,
        "medical_access_count": 0,
    }
    write_json(OUTPUT / "UTILITY_AUDIT.json", utility_audit)

    combined_for_corr = np.concatenate(
        [all_features, utility_all[:, None], np.abs(utility_all)[:, None], opposite_all.astype(np.float32)[:, None]],
        axis=1,
    ).astype(np.float64)
    correlation = np.corrcoef(combined_for_corr, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.savez_compressed(
        OUTPUT / "feature_correlation_matrix.npz",
        correlation=correlation,
        names=np.asarray([*all_names, "utility", "abs_utility", "opposite_sign_error"], dtype="U64"),
    )
    feature_rows: dict[str, Any] = {}
    parameter_iqrs = {
        class_name: np.asarray(json.loads((R1_ROOT / "parameters" / f"{class_name}.json").read_text())["iqr"])
        for class_name in EXPECTED_CLASSES
    }
    class_feature_arrays = [
        np.concatenate([class_existing[index], class_candidates[index]], axis=1)
        for index in range(len(EXPECTED_CLASSES))
    ]
    for feature_index, name in enumerate(all_names):
        values = all_features[:, feature_index]
        by_class_utility = [
            corr(class_feature_arrays[index][:, feature_index], class_utility[index])
            for index in range(len(EXPECTED_CLASSES))
        ]
        by_class_abs = [
            corr(class_feature_arrays[index][:, feature_index], np.abs(class_utility[index]))
            for index in range(len(EXPECTED_CLASSES))
        ]
        by_class_risk = [
            corr(
                class_feature_arrays[index][:, feature_index],
                opposite_mask(class_prediction[index], class_oracle[index]).astype(np.float32),
            )
            for index in range(len(EXPECTED_CLASSES))
        ]
        floor_folds = (
            int(sum(parameter_iqrs[class_name][feature_index] <= 1e-6 for class_name in EXPECTED_CLASSES))
            if feature_index < len(FEATURE_ORDER) else None
        )
        feature_rows[name] = {
            **numeric_stats(values),
            "correlation_signed_utility": corr(values, utility_all),
            "correlation_abs_utility": corr(values, np.abs(utility_all)),
            "correlation_opposite_sign_failure": corr(values, opposite_all.astype(np.float32)),
            "per_class_signed_utility_correlation": dict(zip(EXPECTED_CLASSES, by_class_utility)),
            "per_class_abs_utility_correlation": dict(zip(EXPECTED_CLASSES, by_class_abs)),
            "per_class_opposite_failure_correlation": dict(zip(EXPECTED_CLASSES, by_class_risk)),
            "signed_utility_correlation_sign_stability": sign_stability(by_class_utility),
            "opposite_failure_correlation_sign_stability": sign_stability(by_class_risk),
            "class_eta_squared": class_eta_squared([
                class_feature_arrays[index][:, feature_index] for index in range(len(EXPECTED_CLASSES))
            ]),
            "scaler_floor_folds": floor_folds,
            "scaler_floor_fraction": scalar(floor_folds / 12.0) if floor_folds is not None else None,
        }
    feature_count = len(all_names)
    feature_corr = correlation[:feature_count, :feature_count]
    eigenvalues = np.linalg.eigvalsh(feature_corr)
    positive = np.clip(eigenvalues, 0.0, None)
    effective_rank = float((positive.sum() ** 2) / np.square(positive).sum())
    obvious_redundancy: list[dict[str, Any]] = []
    for left in range(feature_count):
        for right in range(left + 1, feature_count):
            value = float(feature_corr[left, right])
            if abs(value) >= 0.90:
                obvious_redundancy.append({"left": all_names[left], "right": all_names[right], "correlation": value})
    controls = list(range(len(FEATURE_ORDER)))
    candidate_decisions: dict[str, Any] = {}
    utility_index = feature_count
    for offset, name in enumerate(CANDIDATE_ORDER):
        index = len(FEATURE_ORDER) + offset
        row = feature_rows[name]
        existing_correlations = np.abs(feature_corr[index, :len(FEATURE_ORDER)])
        partial = partial_corr_from_matrix(correlation, index, utility_index, controls)
        stable_utility = (
            row["signed_utility_correlation_sign_stability"] >= 0.75
            and np.median([
                abs(value) for value in row["per_class_signed_utility_correlation"].values()
                if value is not None
            ]) >= 0.02
        )
        stable_risk = (
            row["opposite_failure_correlation_sign_stability"] >= 0.75
            and np.median([
                abs(value) for value in row["per_class_opposite_failure_correlation"].values()
                if value is not None
            ]) >= 0.02
        )
        nonredundant_increment = (
            float(existing_correlations.max()) < 0.95
            and partial is not None
            and abs(partial) >= 0.02
        )
        candidate_decisions[name] = {
            "stable_utility_association": bool(stable_utility),
            "stable_opposite_risk_association": bool(stable_risk),
            "nonredundant_incremental_utility_information": bool(nonredundant_increment),
            "max_abs_correlation_with_existing": scalar(existing_correlations.max()),
            "partial_correlation_utility_given_existing": partial,
            "retain": bool(stable_utility or stable_risk or nonredundant_increment),
        }
    eligible = [name for name in CANDIDATE_ORDER if candidate_decisions[name]["retain"]]
    retained: list[str] = []
    for name in eligible:
        index = len(FEATURE_ORDER) + CANDIDATE_ORDER.index(name)
        redundant_with = next(
            (
                kept for kept in retained
                if abs(feature_corr[index, len(FEATURE_ORDER) + CANDIDATE_ORDER.index(kept)]) >= 0.95
            ),
            None,
        )
        if redundant_with is None:
            retained.append(name)
        else:
            candidate_decisions[name]["retain"] = False
            candidate_decisions[name]["rejected_as_redundant_with"] = redundant_with
    for name in CANDIDATE_ORDER:
        candidate_decisions[name]["final_retain"] = name in retained
    feature_audit = {
        "evidence_class": "DERIVED",
        "existing_feature_order": list(FEATURE_ORDER),
        "candidate_feature_order": list(CANDIDATE_ORDER),
        "features": feature_rows,
        "correlation_matrix": {
            "path": "results/sabra_cure/phase0/feature_correlation_matrix.npz",
            "numeric_rank_at_1e-8": int(np.linalg.matrix_rank(feature_corr, tol=1e-8)),
            "effective_rank_participation_ratio": scalar(effective_rank),
            "obvious_abs_correlation_ge_0_90": obvious_redundancy,
        },
        "candidate_decisions": candidate_decisions,
        "retained_signed_features": retained,
        "deployment_sensitivity_code_audit": {
            "evidence_class": "OBSERVED",
            "source_path": "tools/sabra/cache_runner.py",
            "line_semantics": "gradient.detach().abs(); sign explicitly removed",
            "classification": "magnitude evidence",
        },
        "additional_clip_forward_required": False,
        "all_arrays_finite": True,
    }
    write_json(OUTPUT / "FEATURE_AUDIT.json", feature_audit)

    selected_indices = list(range(len(FEATURE_ORDER))) + [
        len(FEATURE_ORDER) + CANDIDATE_ORDER.index(name) for name in retained
    ]
    probe = deterministic_probe(
        [item[:, selected_indices] for item in class_feature_arrays],
        class_utility,
        [all_names[index] for index in selected_indices],
        selected_target,
    )
    write_json(OUTPUT / "FEASIBILITY_PROBE.json", probe)
    write_json(OUTPUT / "PROVENANCE.json", provenance)

    summary = {
        "status": "DIAGNOSTICS_COMPLETE",
        "parent_sha": PARENT_SHA,
        "patches": int(utility_all.size),
        "classes": len(EXPECTED_CLASSES),
        "r1_v2_primary_failure": "confidence-conditioned abstention worsened rather than controlled opposite-sign risk; no frozen threshold qualified",
        "confidence_safety_relation": "CLASS_DEPENDENT_NONMONOTONIC",
        "r0_continuous_utility_valid": bool(target_stable),
        "selected_utility_target": selected_target if target_stable else None,
        "existing_feature_count": len(FEATURE_ORDER),
        "signed_feature_candidates_audited": len(CANDIDATE_ORDER),
        "signed_features_retained": retained,
        "final_feature_count": len(FEATURE_ORDER) + len(retained),
        "additional_clip_forward_required": False,
        "probe_label": probe["label"],
        "probe_summary": probe["summary"],
        "mvtec_access_count": 0,
        "medical_access_count": 0,
        "phase2b_training_steps": 0,
        "new_full_r1_run": False,
    }
    write_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    run()
