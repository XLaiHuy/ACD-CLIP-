#!/usr/bin/env python3
"""Image-split same/cross-state audit for the frozen native CLIP reference."""
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


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([(pred[y == role] == role).mean() for role in (0, 1)]))


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    positive, negative = score[y == 1], score[y == 0]
    return float((positive[:, None] > negative[None, :]).mean() + .5 * (positive[:, None] == negative[None, :]).mean())


def fit(feature: np.ndarray, y: np.ndarray) -> dict:
    x = torch.as_tensor(feature, dtype=torch.float32)
    target = torch.as_tensor(y, dtype=torch.float32)
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-6)
    design = torch.cat([(x - mean) / std, torch.ones((x.shape[0], 1))], dim=1)
    weight = torch.tensor([.5 / (1.0 - target.mean()).clamp_min(1e-6), .5 / target.mean().clamp_min(1e-6)])
    parameter = torch.zeros(design.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS([parameter], lr=.8, max_iter=100, tolerance_grad=1e-7,
                                  tolerance_change=1e-9, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            design @ parameter, target, weight=weight[target.long()]
        ) + 1e-4 * parameter[:-1].square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"mean": mean, "std": std, "parameter": parameter.detach()}


def predict(probe: dict, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = torch.as_tensor(feature, dtype=torch.float32)
    margin = torch.cat([(x - probe["mean"]) / probe["std"], torch.ones((x.shape[0], 1))], dim=1) @ probe["parameter"]
    return torch.sigmoid(margin).numpy(), margin.numpy()


def metrics(y: np.ndarray, probability: np.ndarray, high: np.ndarray) -> dict:
    pred = (probability >= .5).astype(np.int64)
    result = {
        "patches": int(y.size), "balanced_accuracy": balanced_accuracy(y, pred),
        "normal_recall": float((pred[y == 0] == 0).mean()),
        "anomaly_recall": float((pred[y == 1] == 1).mean()),
        "auroc": auroc(y, probability),
    }
    anomaly_high = high & (y == 1)
    result["high_confidence_anomaly_accuracy"] = float((pred[anomaly_high] == 1).mean()) if anomaly_high.any() else None
    return result


def stability(a: torch.Tensor, b: torch.Tensor) -> dict:
    cosine = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=1)
    return {
        "matched_cosine_mean": float(cosine.mean()), "matched_cosine_median": float(cosine.median()),
        "matched_cosine_p05": float(torch.quantile(cosine, .05)), "matched_cosine_p95": float(torch.quantile(cosine, .95)),
        "max_abs_difference": float((a.float() - b.float()).abs().max()),
        "mean_shift_l2": float((a.float().mean(0) - b.float().mean(0)).norm()),
        "norm_a_mean": float(a.float().norm(dim=1).mean()), "norm_b_mean": float(b.float().norm(dim=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-a", type=Path, required=True)
    parser.add_argument("--state-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a, b = (torch.load(path, map_location="cpu", weights_only=False) for path in (args.state_a, args.state_b))
    for key in ("image_id", "group_index", "patch_index", "region", "target"):
        if not torch.equal(a[key].cpu(), b[key].cpu()):
            raise RuntimeError(f"unmatched support field: {key}")
    y = a["region"].long().numpy()
    if np.unique(y).size != 2:
        raise RuntimeError("fixed TRAIN support lacks both physical regions")
    image_id = a["image_id"].long().numpy()
    images = np.unique(image_id)
    image_has_anomaly = np.array([bool((y[image_id == image] == 1).any()) for image in images], dtype=int)
    rng = np.random.default_rng(0)
    test_images = []
    for role in (0, 1):
        candidates = images[image_has_anomaly == role].copy()
        rng.shuffle(candidates)
        test_images.extend(candidates[:max(1, int(round(candidates.size * .25)))].tolist())
    test_images = np.array(sorted(test_images), dtype=images.dtype)
    train_mask = ~np.isin(image_id, test_images)
    test_mask = np.isin(image_id, test_images)
    if np.unique(y[train_mask]).size != 2 or np.unique(y[test_mask]).size != 2:
        raise RuntimeError("image split did not retain both regions")
    high = (a["teacher_probability"].float()[:, 1] >= .75).numpy()
    features_a, features_b = a["native_reference"].float(), b["native_reference"].float()
    probe_a, probe_b = fit(features_a.numpy()[train_mask], y[train_mask]), fit(features_b.numpy()[train_mask], y[train_mask])
    pa_source, ma_source = predict(probe_a, features_a.numpy()[test_mask])
    pa_cross, ma_cross = predict(probe_a, features_b.numpy()[test_mask])
    pb_source, mb_source = predict(probe_b, features_b.numpy()[test_mask])
    pb_cross, mb_cross = predict(probe_b, features_a.numpy()[test_mask])
    result = {
        "audit": "FROZEN_NATIVE_CLIP_ROUTER_REFERENCE_STABILITY", "test_or_medical_data": False,
        "states": {
            "A": {"label": a["state"], "capture": str(args.state_a.resolve()), "capture_sha256": sha256(args.state_a),
                  "checkpoint": a["source_checkpoint"], "checkpoint_sha256": a["source_checkpoint_sha256"], "config_sha256": a["config_sha256"]},
            "B": {"label": b["state"], "capture": str(args.state_b.resolve()), "capture_sha256": sha256(args.state_b),
                  "checkpoint": b["source_checkpoint"], "checkpoint_sha256": b["source_checkpoint_sha256"], "config_sha256": b["config_sha256"]},
        },
        "support": {"images": int(images.size), "patches": int(y.size), "train_images": [int(i) for i in images if i not in set(test_images.tolist())],
                    "test_images": [int(i) for i in test_images], "anomaly_fraction": float(y.mean()), "labels": "immutable TRAIN physical region N/A"},
        "within_state_A": metrics(y[test_mask], pa_source, high[test_mask]),
        "within_state_B": metrics(y[test_mask], pb_source, high[test_mask]),
        "A_to_B": {"cross_state": metrics(y[test_mask], pa_cross, high[test_mask]),
                   "margin_shift_target_minus_source": float(ma_cross.mean() - ma_source.mean()),
                   "prediction_flip_rate": float(((pa_source >= .5) != (pa_cross >= .5)).mean())},
        "B_to_A": {"cross_state": metrics(y[test_mask], pb_cross, high[test_mask]),
                   "margin_shift_target_minus_source": float(mb_cross.mean() - mb_source.mean()),
                   "prediction_flip_rate": float(((pb_source >= .5) != (pb_cross >= .5)).mean())},
        "matched_feature_stability": stability(features_a, features_b),
    }
    within = min(result["within_state_A"]["balanced_accuracy"], result["within_state_B"]["balanced_accuracy"])
    cross = min(result["A_to_B"]["cross_state"]["balanced_accuracy"], result["B_to_A"]["cross_state"]["balanced_accuracy"])
    if within < .80:
        decision = "FROZEN_ROUTER_REFERENCE_NOT_SEPARABLE"
    elif cross < .80:
        decision = "FROZEN_ROUTER_REFERENCE_NOT_STABLE"
    else:
        decision = "FROZEN_ROUTER_REFERENCE_SEPARABLE_STABLE"
    result["decision"] = decision
    result["decision_metrics"] = {"within_min_balanced_accuracy": within, "cross_min_balanced_accuracy": cross,
                                  "separable_threshold": .80, "stable_threshold": .80}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"decision": decision, "decision_metrics": result["decision_metrics"], "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
