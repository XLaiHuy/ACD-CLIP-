#!/usr/bin/env python3
"""Image-split, zero-refit RAW-vs-Q cross-state stability autopsy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_accuracy(y_true: np.ndarray, prediction: np.ndarray) -> float:
    recalls = [(prediction[y_true == role] == role).mean() for role in (0, 1) if (y_true == role).any()]
    return float(np.mean(recalls))


def binary_auroc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    positive = score[y_true == 1]
    negative = score[y_true == 0]
    if positive.size == 0 or negative.size == 0:
        return None
    return float((positive[:, None] > negative[None, :]).mean() + .5 * (positive[:, None] == negative[None, :]).mean())


def fit_probe(feature: np.ndarray, labels: np.ndarray) -> dict:
    x = torch.as_tensor(feature, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-6)
    x = (x - mean) / std
    design = torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype)], dim=1)
    class_weight = torch.tensor([0.5 / (1.0 - y.mean()).clamp_min(1e-6), 0.5 / y.mean().clamp_min(1e-6)])
    parameter = torch.zeros(design.shape[1], dtype=x.dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS([parameter], lr=.8, max_iter=100, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        margin = design @ parameter
        loss = torch.nn.functional.binary_cross_entropy_with_logits(margin, y, weight=class_weight[y.long()])
        loss = loss + 1e-4 * parameter[:-1].square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"mean": mean, "std": std, "weight": parameter.detach()}


def predict_probe(probe: dict, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = torch.as_tensor(feature, dtype=torch.float32)
    x = (x - probe["mean"]) / probe["std"]
    design = torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype)], dim=1)
    margin = design @ probe["weight"]
    probability_role1 = torch.sigmoid(margin)
    probability = torch.stack([1.0 - probability_role1, probability_role1], dim=1).numpy()
    return probability, margin.numpy()


def scores(y_true: np.ndarray, probability: np.ndarray, high_confidence: np.ndarray) -> dict:
    prediction = probability.argmax(1)
    result = {"patches": int(y_true.size), "balanced_accuracy": balanced_accuracy(y_true, prediction)}
    for role, name in ((0, "normal_recall"), (1, "anomaly_recall")):
        mask = y_true == role
        result[name] = float((prediction[mask] == role).mean()) if mask.any() else None
    high = high_confidence & (y_true == 1)
    result["high_confidence_anomaly_accuracy"] = float((prediction[high] == 1).mean()) if high.any() else None
    result["auroc"] = binary_auroc(y_true, probability[:, 1])
    return result


def run_direction(
    feature_train: np.ndarray,
    labels_train: np.ndarray,
    feature_source_test: np.ndarray,
    labels_source_test: np.ndarray,
    high_source_test: np.ndarray,
    feature_target_test: np.ndarray,
    labels_target_test: np.ndarray,
    high_target_test: np.ndarray,
) -> dict:
    probe = fit_probe(feature_train, labels_train)
    source_probability, source_margin = predict_probe(probe, feature_source_test)
    target_probability, target_margin = predict_probe(probe, feature_target_test)
    return {
        "source_heldout": scores(labels_source_test, source_probability, high_source_test),
        "cross_state": scores(labels_target_test, target_probability, high_target_test),
        "margin_shift_target_minus_source_mean": float(target_margin.mean() - source_margin.mean()),
        "margin_source_mean": float(source_margin.mean()),
        "margin_target_mean": float(target_margin.mean()),
        "prediction_flip_rate_matched": float((source_probability.argmax(1) != target_probability.argmax(1)).mean()),
    }


def quantiles(values: torch.Tensor) -> dict:
    return {name: float(torch.quantile(values, q)) for name, q in (("p05", .05), ("p50", .50), ("p95", .95))}


def stability(a: torch.Tensor, b: torch.Tensor) -> dict:
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=1)
    centered_a = a - a.mean(0, keepdim=True)
    centered_b = b - b.mean(0, keepdim=True)
    covariance_a = centered_a.T @ centered_a / max(1, a.shape[0] - 1)
    covariance_b = centered_b.T @ centered_b / max(1, b.shape[0] - 1)
    cross = centered_a.T @ centered_b
    procrustes = torch.linalg.svdvals(cross).sum() / (centered_a.norm() * centered_b.norm()).clamp_min(1e-12)
    return {
        "matched_cosine_mean": float(cosine.mean()),
        "matched_cosine_median": float(cosine.median()),
        "matched_cosine_quantiles": quantiles(cosine),
        "mean_shift_l2": float((a.mean(0) - b.mean(0)).norm()),
        "norm_a": {"mean": float(a.norm(dim=1).mean()), "std": float(a.norm(dim=1).std(unbiased=False))},
        "norm_b": {"mean": float(b.norm(dim=1).mean()), "std": float(b.norm(dim=1).std(unbiased=False))},
        "centered_covariance_relative_fro_drift": float((covariance_a - covariance_b).norm() / covariance_a.norm().clamp_min(1e-12)),
        "offline_orthogonal_procrustes_score": float(procrustes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-a", type=Path, required=True)
    parser.add_argument("--state-b", type=Path, required=True)
    parser.add_argument("--state-a-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = torch.load(args.state_a, map_location="cpu", weights_only=False)
    b = torch.load(args.state_b, map_location="cpu", weights_only=False)
    for key in ("image_id", "group_index", "patch_index"):
        if not torch.equal(a[key].long(), b[key].long()):
            raise RuntimeError(f"unmatched support field: {key}")
    common = b["utility_valid"].bool() & b["utility_informative"].bool()
    if int(common.sum()) < 100:
        raise RuntimeError(f"insufficient common B teacher support: {int(common.sum())}")
    image_id = a["image_id"].long()[common]
    y_a = a["teacher_probability"].float()[common].argmax(1).numpy()
    y_b = b["teacher_probability"].float()[common].argmax(1).numpy()
    high_a = (a["teacher_probability"].float()[common, 1] >= .75).numpy()
    high_b = (b["teacher_probability"].float()[common, 1] >= .75).numpy()
    unique_images = torch.unique(image_id).numpy()
    image_anomaly = np.array([bool((y_a[image_id.numpy() == image] == 1).any()) for image in unique_images], dtype=int)
    rng = np.random.default_rng(0)
    test_parts = []
    for role in (0, 1):
        candidates = unique_images[image_anomaly == role].copy()
        rng.shuffle(candidates)
        test_parts.append(candidates[:max(1, int(round(candidates.size * .25)))])
    test_images = np.sort(np.concatenate(test_parts))
    test_set = set(test_images.tolist())
    train_images = np.array([image for image in unique_images if image not in test_set], dtype=unique_images.dtype)
    train_mask = np.isin(image_id.numpy(), train_images)
    test_mask = np.isin(image_id.numpy(), test_images)
    if not (np.unique(y_a[train_mask]).size == np.unique(y_a[test_mask]).size == np.unique(y_b[train_mask]).size == np.unique(y_b[test_mask]).size == 2):
        raise RuntimeError("image split did not retain both roles in each state")
    output = {
        "audit": "RAW_VS_Q_CROSS_STATE_STABILITY",
        "states": {
            "A_failed_direct_head": {
                "capture": str(args.state_a.resolve()), "capture_sha256": sha256(args.state_a),
                "checkpoint": str(args.state_a_checkpoint.resolve()), "checkpoint_sha256": sha256(args.state_a_checkpoint),
            },
            "B_fresh_seed0": {
                "capture": str(args.state_b.resolve()), "capture_sha256": sha256(args.state_b),
                "source_checkpoint": b["source_checkpoint"], "source_checkpoint_sha256": b["source_checkpoint_sha256"],
                "config": b["config"], "config_sha256": b["config_sha256"], "source_commit": b["source_commit"],
                "fresh_parameter_state_hash": b["fresh_parameter_state_hash"],
            },
        },
        "support": {
            "images": int(unique_images.size), "patches_common_informative_valid": int(common.sum()),
            "train_images": [int(value) for value in train_images], "test_images": [int(value) for value in test_images],
            "state_a_teacher_role1_fraction": float(y_a.mean()), "state_b_teacher_role1_fraction": float(y_b.mean()),
            "teacher_hard_role_flip_rate_A_to_B": float((y_a != y_b).mean()),
            "test_or_medical_data": False, "model_forwards_for_offline_analysis": 0,
        },
        "features": {},
    }
    for name, field in (("raw", "raw_router_input"), ("q", "production_q")):
        feature_a = a[field].float()[common]
        feature_b = b[field].float()[common]
        f_a = feature_a.numpy()
        f_b = feature_b.numpy()
        a_to_b = run_direction(f_a[train_mask], y_a[train_mask], f_a[test_mask], y_a[test_mask], high_a[test_mask], f_b[test_mask], y_b[test_mask], high_b[test_mask])
        b_to_a = run_direction(f_b[train_mask], y_b[train_mask], f_b[test_mask], y_b[test_mask], high_b[test_mask], f_a[test_mask], y_a[test_mask], high_a[test_mask])
        output["features"][name] = {
            "within_state_A": a_to_b["source_heldout"], "within_state_B": b_to_a["source_heldout"],
            "A_to_B": {key: value for key, value in a_to_b.items() if key != "source_heldout"},
            "B_to_A": {key: value for key, value in b_to_a.items() if key != "source_heldout"},
            "matched_feature_stability": stability(feature_a, feature_b),
        }
    raw = output["features"]["raw"]
    q = output["features"]["q"]
    raw_within_healthy = min(raw["within_state_A"]["balanced_accuracy"], raw["within_state_B"]["balanced_accuracy"]) >= .85
    q_within_healthy = min(q["within_state_A"]["balanced_accuracy"], q["within_state_B"]["balanced_accuracy"]) >= .85
    raw_cross = min(raw["A_to_B"]["cross_state"]["balanced_accuracy"], raw["B_to_A"]["cross_state"]["balanced_accuracy"])
    q_cross = min(q["A_to_B"]["cross_state"]["balanced_accuracy"], q["B_to_A"]["cross_state"]["balanced_accuracy"])
    raw_cross_healthy = raw_cross >= .80
    q_cross_healthy = q_cross >= .80
    if raw_within_healthy and raw_cross_healthy and not q_cross_healthy:
        decision = "Q_MAPPING_NONSTATIONARY_RAW_STABLE"
    elif raw_within_healthy and not raw_cross_healthy:
        decision = "ROUTER_UPSTREAM_REPRESENTATION_NONSTATIONARY"
    elif not raw_within_healthy:
        decision = "ROUTER_REPRESENTATION_REDESIGN_REQUIRED"
    elif raw_cross_healthy and q_cross_healthy:
        decision = "ROUTER_FEATURE_PATH_MISMATCH"
    else:
        decision = "ROUTER_UPSTREAM_REPRESENTATION_NONSTATIONARY"
    output["decision"] = decision
    output["decision_metrics"] = {
        "raw_within_healthy": raw_within_healthy, "q_within_healthy": q_within_healthy,
        "raw_cross_min_balanced_accuracy": raw_cross, "q_cross_min_balanced_accuracy": q_cross,
        "raw_cross_healthy_threshold_080": raw_cross_healthy, "q_cross_healthy_threshold_080": q_cross_healthy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"decision": decision, "decision_metrics": output["decision_metrics"], "features": output["features"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
