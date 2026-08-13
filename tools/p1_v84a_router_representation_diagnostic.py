#!/usr/bin/env python3
"""Offline, no-optimizer separability and Router query/key diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _stats(values: torch.Tensor) -> dict[str, float | int]:
    values = values.float().flatten()
    if not values.numel():
        return {"count": 0}
    quantiles = torch.quantile(values, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99]))
    return {
        "count": int(values.numel()), "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)), "min": float(values.min()), "max": float(values.max()),
        "p01": float(quantiles[0]), "p05": float(quantiles[1]), "p50": float(quantiles[2]),
        "p95": float(quantiles[3]), "p99": float(quantiles[4]),
    }


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positive = int(labels.sum())
    negative = int(labels.numel() - positive)
    if not positive or not negative:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    sum_positive = ranks[labels.bool()].sum()
    return float((sum_positive - positive * (positive + 1) / 2) / (positive * negative))


def _diagonal_lda(features: torch.Tensor, labels: torch.Tensor, train: torch.Tensor, test: torch.Tensor) -> dict[str, Any]:
    train_labels = labels[train]
    test_labels = labels[test]
    if not ((train_labels == 0).any() and (train_labels == 1).any() and (test_labels == 0).any() and (test_labels == 1).any()):
        return {"status": "insufficient_two_role_holdout"}
    train_features = features[train].float()
    mu0 = train_features[train_labels == 0].mean(0)
    mu1 = train_features[train_labels == 1].mean(0)
    var0 = train_features[train_labels == 0].var(0, unbiased=False)
    var1 = train_features[train_labels == 1].var(0, unbiased=False)
    pooled = (var0 + var1).mul(0.5).clamp_min(1e-8)
    weight = (mu1 - mu0) / pooled
    threshold = 0.5 * torch.dot(weight, mu0 + mu1)
    scores = features[test].float() @ weight - threshold
    prediction = (scores >= 0).long()
    tpr = float((prediction[test_labels == 1] == 1).float().mean())
    tnr = float((prediction[test_labels == 0] == 0).float().mean())
    return {
        "status": "pass",
        "train_rows": int(train.sum()), "test_rows": int(test.sum()),
        "test_accuracy": float((prediction == test_labels).float().mean()),
        "balanced_accuracy": 0.5 * (tpr + tnr), "role0_recall": tnr, "role1_recall": tpr,
        "auc_role1": _auc(scores, test_labels),
        "mean_separation_l2": float((mu1 - mu0).norm()),
    }


def _by_region(features: torch.Tensor, labels: torch.Tensor, regions: torch.Tensor, train: torch.Tensor, test: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, physical_region in (("ALL", None), ("NORMAL", 0), ("ANOMALY", 1)):
        subset = torch.ones_like(labels, dtype=torch.bool) if physical_region is None else regions == physical_region
        out[name] = _diagonal_lda(features, labels, train & subset, test & subset)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = torch.load(args.sample, map_location="cpu", weights_only=False)
    source = torch.load(args.source_cache, map_location="cpu", weights_only=False)
    capture = json.loads(args.capture_report.read_text())
    features = payload["router_input_features"].float()
    queries = payload["queries"].float()
    labels = payload["teacher_hard_role"].long()
    region = payload["physical_region"].long()
    image_index = payload["image_index"].long()
    keys = payload["final_router_keys"].float()
    probs = payload["router_probabilities"].float()
    logits = payload["router_logits"].float()
    train = image_index.remainder(5) != 0
    test = ~train
    source_probs = torch.stack(source["dense"]).squeeze(2).float()[
        image_index, payload["group_index"].long(), payload["patch_index"].long()
    ]
    key_cosine = F.normalize(keys, dim=-1) @ F.normalize(keys, dim=-1).T
    query_key_cosine = F.normalize(queries, dim=-1) @ F.normalize(keys, dim=-1).T
    report = {
        "audit": "P1_V84A_ROUTER_REPRESENTATION_DIAGNOSTIC_OFFLINE",
        "forward_replayed": False, "optimizer_steps": 0, "backward_steps": 0,
        "capture_contract": {
            "capture_status": capture.get("status"),
            "model_state_unchanged": capture.get("model_state_unchanged"),
            "invariants": capture.get("invariants"),
        },
        "sample": {
            "rows": int(labels.numel()), "feature_shape": list(features.shape), "query_shape": list(queries.shape),
            "role_frequency": [float((labels == role).float().mean()) for role in range(2)],
            "region_frequency": [float((region == value).float().mean()) for value in range(2)],
            "feature_l2_norm": _stats(features.norm(dim=-1)),
            "source_probability_max_abs_delta": float((source_probs - probs).abs().max()),
        },
        "input_linear_separability_diagonal_lda": _by_region(features, labels, region, train, test),
        "query_linear_separability_diagonal_lda": _by_region(queries, labels, region, train, test),
        "query_key_geometry": {
            "key_cosine_matrix": key_cosine.tolist(),
            "role0_similarity": _stats(query_key_cosine[:, 0]),
            "role1_similarity": _stats(query_key_cosine[:, 1]),
            "margin_role0_minus_role1": _stats(query_key_cosine[:, 0] - query_key_cosine[:, 1]),
            "by_region": {
                name: {
                    "query_norm": _stats(queries[mask].norm(dim=-1)),
                    "margin": _stats((query_key_cosine[:, 0] - query_key_cosine[:, 1])[mask]),
                    "router_probability_role0": _stats(probs[mask, 0]),
                    "router_logit_margin": _stats((logits[:, 0] - logits[:, 1])[mask]),
                }
                for name, mask in (("NORMAL", region == 0), ("ANOMALY", region == 1))
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    compact = {
        "output": str(args.output),
        "input_balanced_accuracy_all": report["input_linear_separability_diagonal_lda"]["ALL"].get("balanced_accuracy"),
        "input_balanced_accuracy_anomaly": report["input_linear_separability_diagonal_lda"]["ANOMALY"].get("balanced_accuracy"),
        "query_balanced_accuracy_all": report["query_linear_separability_diagonal_lda"]["ALL"].get("balanced_accuracy"),
        "query_balanced_accuracy_anomaly": report["query_linear_separability_diagonal_lda"]["ANOMALY"].get("balanced_accuracy"),
        "source_probability_max_abs_delta": report["sample"]["source_probability_max_abs_delta"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
