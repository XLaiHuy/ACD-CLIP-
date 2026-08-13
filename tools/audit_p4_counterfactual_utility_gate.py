#!/usr/bin/env python3
"""U0: class-held-out counterfactual K1 utility-gating audit.

This is an inference-only audit. It extracts frozen K1 counterfactuals on the
Stage 1.7R VisA-Train manifest, then fits only tiny CPU utility predictors.
No model checkpoint is modified and no patch from a held-out class is used to
fit its fold's standardization or predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audit_p4_k1_oracle_utility import (
    DeterministicVisATrainDataset,
    _build_model as build_k1_model,
    _git_sha,
    _sha256,
)
from audit_p4_mature_phase2b_gate import _phase2b_args
from dataset import DOMAINS
from phase2b_anchor_diagnosis import build_model as build_phase2b_model
from phase2b_anchor_diagnosis import get_class_text_embedding, load_checkpoint
from utils import configure_canonical_fp32, get_phase2b_global_text_features


SCRIPT_VERSION = "u0_counterfactual_utility_gate_v1"
RHO = 0.05
L2 = 1e-4
FOLDS = 4
REGIONS = ("all", "normal", "anomaly")


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """Average ranks for ties; CPU only and adequate for the compact audit."""
    values = values.detach().float().cpu()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty(values.numel(), dtype=torch.float64)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _auroc(scores: torch.Tensor, positive: torch.Tensor) -> float | None:
    positive = positive.detach().bool().cpu()
    scores = scores.detach().float().cpu()
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    ranks = _rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(scores: torch.Tensor, positive: torch.Tensor) -> float | None:
    positive, scores = positive.detach().bool().cpu(), scores.detach().float().cpu()
    n_pos = int(positive.sum())
    if not n_pos:
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    labels = positive[order].float()
    precision = labels.cumsum(0) / torch.arange(1, labels.numel() + 1, dtype=torch.float32)
    return float((precision * labels).sum() / n_pos)


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x, y = x.detach().float().cpu(), y.detach().float().cpu()
    if x.numel() < 2 or float(x.std(unbiased=False)) == 0.0 or float(y.std(unbiased=False)) == 0.0:
        return None
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1])


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float | None:
    if x.numel() < 2:
        return None
    return _pearson(_rankdata(x).float(), _rankdata(y).float())


def _summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().cpu()
    if not values.numel():
        return {"count": 0, "mean": None, "std": None, "p95": None}
    return {"count": int(values.numel()), "mean": float(values.mean()), "std": float(values.std(unbiased=False)), "p95": float(torch.quantile(values, 0.95))}


def _binary_metrics(score: torch.Tensor, utility: torch.Tensor) -> dict[str, Any]:
    positive = utility > 0
    return {
        "count": int(score.numel()), "positive_fraction": float(positive.float().mean()),
        "auroc_a1_better": _auroc(score, positive), "ap_a1_better": _average_precision(score, positive),
        "pearson_q_utility": _pearson(score, utility), "spearman_q_utility": _spearman(score, utility),
    }


def _fit_logistic(train_x: torch.Tensor, train_positive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One L2-regularized, balanced logistic model; no tuning or sweep."""
    mean, std = train_x.mean(dim=0), train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    x = (train_x - mean) / std
    y = train_positive.float()
    n_pos, n_neg = y.sum(), y.numel() - y.sum()
    if not n_pos or not n_neg:
        raise RuntimeError("A class-held-out training fold has only one utility label.")
    weight = torch.where(y > 0, n_neg / n_pos, torch.ones_like(y))
    params = torch.zeros(3, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([params], lr=1.0, max_iter=80, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x @ params[:2] + params[2]
        loss = F.binary_cross_entropy_with_logits(logits, y, weight=weight) + L2 * params[:2].square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return mean, std, params.detach(), torch.tensor(float(n_neg / n_pos))


def _fit_tiny_mlp(train_x: torch.Tensor, train_positive: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.nn.Module, torch.Tensor]:
    """The one authorized non-linear check: a fixed 2->4->1 model."""
    mean, std = train_x.mean(dim=0), train_x.std(dim=0, unbiased=False).clamp_min(1e-6)
    x, y = (train_x - mean) / std, train_positive.float()
    n_pos, n_neg = y.sum(), y.numel() - y.sum()
    if not n_pos or not n_neg:
        raise RuntimeError("A class-held-out training fold has only one utility label.")
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=L2)
    weight = torch.where(y > 0, n_neg / n_pos, torch.ones_like(y))
    for _ in range(300):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x).squeeze(1), y, weight=weight)
        loss.backward()
        optimizer.step()
    return mean, std, model.eval(), torch.tensor(float(n_neg / n_pos))


def _gate_from_logodds(q: torch.Tensor) -> torch.Tensor:
    """q<=0 is an exact no-op; q>0 maps smoothly to a bounded activation."""
    return torch.where(q > 0, torch.sigmoid(q), torch.zeros_like(q))


def _gain(base_logit: torch.Tensor, dynamic_logit: torch.Tensor, target: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    base = F.binary_cross_entropy_with_logits(base_logit, target, reduction="none")
    final = F.binary_cross_entropy_with_logits(base_logit + RHO * gate * (dynamic_logit - base_logit), target, reduction="none")
    return base - final


def _region_counterfactual(data: dict[str, torch.Tensor], gate: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for region, mask in (("all", torch.ones_like(data["target"], dtype=torch.bool)), ("normal", data["target"] == 0), ("anomaly", data["target"] > 0)):
        gain = _gain(data["l0"], data["l1"], data["target"], gate)[mask]
        out[region] = _summary(gain)
    return out


def _utility_retention(data: dict[str, torch.Tensor], gate: torch.Tensor) -> dict[str, float | None]:
    anomaly = data["target"] > 0
    u = data["utility"][anomaly]
    g = gate[anomaly]
    positive, harmful = u.clamp_min(0), (-u).clamp_min(0)
    return {
        "positive_anomaly_utility_retained": None if float(positive.sum()) == 0.0 else float((positive * g).sum() / positive.sum()),
        "harmful_anomaly_utility_rejected": None if float(harmful.sum()) == 0.0 else float((harmful * (1.0 - g)).sum() / harmful.sum()),
    }


def _image_rows(data: dict[str, torch.Tensor], gate: torch.Tensor, variant: str) -> list[dict[str, Any]]:
    rows = []
    for image_id in torch.unique(data["image_id"]).tolist():
        mask_image = data["image_id"] == image_id
        row = {"image_index": int(image_id), "class_name": data["class_name_by_image"][int(image_id)], "variant": variant}
        for region, mask_region in (("all", torch.ones_like(mask_image)), ("normal", data["target"] == 0), ("anomaly", data["target"] > 0)):
            mask = mask_image & mask_region
            if bool(mask.any()):
                row[f"{region}_gain"] = float(_gain(data["l0"], data["l1"], data["target"], gate)[mask].mean())
                row[f"{region}_gate_mean"] = float(gate[mask].mean())
        rows.append(row)
    return rows


def _extract_frozen_data(args: argparse.Namespace) -> dict[str, torch.Tensor | list[str]]:
    k1_dir, manifest_path, p2b_path = args.k1_run_dir.resolve(), args.manifest.resolve(), args.phase2b_checkpoint.resolve()
    config_path, checkpoint_path = k1_dir / "config.json", k1_dir / "adapter_1.pth"
    config, checkpoint = json.loads(config_path.read_text()), torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    p2b_checkpoint = torch.load(p2b_path, map_location="cpu", weights_only=False)
    manifest = json.loads(manifest_path.read_text())
    if config.get("h6_progress_version") != "P4-CSF-K1" or int(p2b_checkpoint.get("epoch", -1)) != 10:
        raise RuntimeError("U0 requires the published K1 and documented mature Phase2B e10 checkpoint.")
    if not torch.cuda.is_available():
        raise RuntimeError("No patch-level cache exists, so U0 requires one frozen CUDA inference extraction.")
    configure_canonical_fp32()
    device = torch.device(f"cuda:{config['cuda_device']}")
    k1 = build_k1_model(config, checkpoint, device)
    p2b_args = _phase2b_args(config["cuda_device"])
    p2b = build_phase2b_model(p2b_args, device)
    if load_checkpoint(p2b, p2b_path, p2b_args, device) != 10:
        raise RuntimeError("Mature Phase2B epoch mismatch.")
    loader = DataLoader(DeterministicVisATrainDataset(manifest, config["img_size"]), batch_size=1, shuffle=False, num_workers=0)
    parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    class_name_by_image: list[str] = []
    with torch.inference_mode():
        for image_index, raw in enumerate(loader):
            image = raw["image"].to(device=device, dtype=torch.float32)
            mask = raw["mask"].to(device=device, dtype=torch.float32)
            class_name = raw["class_name"][0]
            class_name_by_image.append(class_name)
            p2b_tokens, _ = p2b(image)
            p2b_text = get_class_text_embedding(p2b, "VisA", class_name, device, "hybrid", 0.20)
            p2b_map = p2b.vision_text_fusion_gate_seg(torch.stack(p2b_tokens), p2b_text.unsqueeze(1), test_mode=True, domain=DOMAINS["VisA"])
            visual = k1(image, return_phase4_features=True)
            base_text = get_phase2b_global_text_features(k1, "VisA", [class_name], device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
            batch = k1.h6.build_batch(k1, "VisA", [class_name], visual, hybrid_alpha=0.0, base_text_features=base_text)
            for group, patches in enumerate(visual["seg_tokens"]):
                side = int(patches.shape[1] ** 0.5)
                target = F.adaptive_avg_pool2d(mask, (side, side)).flatten(1)
                p2b_patch = F.adaptive_avg_pool2d(p2b_map.unsqueeze(1), (side, side)).flatten(1)
                base_logits = batch["base_group_logits"][group]
                normal, l0, l1 = base_logits[..., 0], base_logits[..., 1], batch["dynamic_abnormal_logits"][group]
                base_logit = l0 - normal
                dynamic_logit = l1 - normal
                utility = F.binary_cross_entropy_with_logits(base_logit, target, reduction="none") - F.binary_cross_entropy_with_logits(dynamic_logit, target, reduction="none")
                n = target.numel()
                parts["l0"].append(base_logit.flatten().cpu())
                parts["l1"].append(dynamic_logit.flatten().cpu())
                parts["target"].append(target.flatten().cpu())
                parts["utility"].append(utility.flatten().cpu())
                parts["p2b"].append(p2b_patch.flatten().cpu())
                parts["class_id"].append(torch.full((n,), len(class_name_by_image) - 1, dtype=torch.long))
                parts["image_id"].append(torch.full((n,), image_index, dtype=torch.long))
    result: dict[str, torch.Tensor | list[str]] = {name: torch.cat(value) for name, value in parts.items()}
    result["class_name_by_image"] = class_name_by_image
    result["k1_checkpoint_sha256"] = _sha256(checkpoint_path)
    result["p2b_checkpoint_sha256"] = _sha256(p2b_path)
    result["manifest_sha256"] = _sha256(manifest_path)
    result["config_sha256"] = _sha256(config_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k1-run-dir", type=Path, default=Path("runs/phase4/k1/short64_seed0_attempt5"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"))
    parser.add_argument("--phase2b-checkpoint", type=Path, default=Path("runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/adapter_10.pth"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase4/k1/stage1_10_u0/K1_COUNTERFACTUAL_UTILITY_GATE_AUDIT.json"))
    parser.add_argument("--seed", type=int, default=1910)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    data = _extract_frozen_data(args)
    class_names = sorted(set(data["class_name_by_image"]))
    if len(class_names) != 12:
        raise RuntimeError(f"expected 12 manifest classes, got {len(class_names)}")
    class_to_fold = {name: index % FOLDS for index, name in enumerate(class_names)}
    patch_class = torch.tensor([class_to_fold[data["class_name_by_image"][int(index)]] for index in data["class_id"].tolist()])
    x = torch.stack((data["l0"], data["l1"] - data["l0"]), dim=1).float()
    utility, target, p2b = data["utility"].float(), data["target"].float(), data["p2b"].float()
    positive = utility > 0
    q = torch.empty_like(utility)
    fold_records, coefficients = [], []
    for fold in range(FOLDS):
        train, valid = patch_class != fold, patch_class == fold
        mean, std, params, imbalance = _fit_logistic(x[train], positive[train])
        q[valid] = ((x[valid] - mean) / std) @ params[:2] + params[2]
        held = [name for name in class_names if class_to_fold[name] == fold]
        fold_records.append({"fold": fold, "held_out_classes": held, "train_patch_count": int(train.sum()), "validation_patch_count": int(valid.sum()), "positive_train_fraction": float(positive[train].float().mean()), "positive_validation_fraction": float(positive[valid].float().mean()), "class_balance_weight": float(imbalance), "feature_mean_train_only": [float(v) for v in mean], "feature_std_train_only": [float(v) for v in std]})
        coefficients.append({"fold": fold, "standardized_l0": float(params[0]), "standardized_delta_logit": float(params[1]), "bias": float(params[2])})
    linear_q = q.clone()
    linear_anomaly = _binary_metrics(linear_q[target > 0], utility[target > 0])
    p2b_anomaly = _binary_metrics(p2b[target > 0], utility[target > 0])
    mlp_q = None
    if linear_anomaly["auroc_a1_better"] <= p2b_anomaly["auroc_a1_better"]:
        mlp_q = torch.empty_like(utility)
        for fold in range(FOLDS):
            train, valid = patch_class != fold, patch_class == fold
            mean, std, model, _ = _fit_tiny_mlp(x[train], positive[train], args.seed + fold + 1)
            with torch.inference_mode():
                mlp_q[valid] = model((x[valid] - mean) / std).squeeze(1)
            fold_records[fold]["tiny_mlp_ran"] = True
        mlp_anomaly = _binary_metrics(mlp_q[target > 0], utility[target > 0])
        if mlp_anomaly["auroc_a1_better"] > linear_anomaly["auroc_a1_better"]:
            q, selected_model = mlp_q, "tiny_mlp_2_4_1"
        else:
            selected_model = "linear_logistic"
    else:
        mlp_anomaly, selected_model = None, "linear_logistic"
    gate = _gate_from_logodds(q)
    data_t: dict[str, Any] = {key: value for key, value in data.items() if key in {"l0", "l1", "target", "utility", "image_id", "class_id", "class_name_by_image"}}
    base_normal_bce = float(F.binary_cross_entropy_with_logits(data["l0"][target == 0], target[target == 0]))
    safety_margin = max(1e-6, 0.001 * base_normal_bce)
    p2b_gate = p2b.clamp(0.0, 1.0)
    class_rows = []
    for class_name in class_names:
        ids = torch.tensor([class_name_by_image == class_name for class_name_by_image in data["class_name_by_image"]])
        mask = ids[data["class_id"]]
        anomaly = mask & (target > 0)
        normal = mask & (target == 0)
        row = {"class_name": class_name, "fold": class_to_fold[class_name], "patch_count": int(mask.sum()), "anomaly_patch_count": int(anomaly.sum()), "normal_patch_count": int(normal.sum()), "utility_predictor_all": _binary_metrics(q[mask], utility[mask]), "utility_predictor_anomaly": _binary_metrics(q[anomaly], utility[anomaly]), "p2b_baseline_all": _binary_metrics(p2b[mask], utility[mask]), "p2b_baseline_anomaly": _binary_metrics(p2b[anomaly], utility[anomaly]), "normal_activation": _summary(gate[normal]), "utility_retention": _utility_retention({key: value[mask] for key, value in data_t.items() if isinstance(value, torch.Tensor)}, gate[mask]), "utility_gate_gain": _region_counterfactual({key: value[mask] for key, value in data_t.items() if isinstance(value, torch.Tensor)}, gate[mask]), "p2b_gate_gain": _region_counterfactual({key: value[mask] for key, value in data_t.items() if isinstance(value, torch.Tensor)}, p2b_gate[mask])}
        class_rows.append(row)
    aggregate = {"utility_predictor_all": _binary_metrics(q, utility), "utility_predictor_anomaly": _binary_metrics(q[target > 0], utility[target > 0]), "p2b_baseline_all": _binary_metrics(p2b, utility), "p2b_baseline_anomaly": _binary_metrics(p2b[target > 0], utility[target > 0]), "normal_activation": _summary(gate[target == 0]), "utility_retention": _utility_retention(data_t, gate), "utility_gate_gain": _region_counterfactual(data_t, gate), "p2b_gate_gain": _region_counterfactual(data_t, p2b_gate), "ungated_k1_gain": _region_counterfactual(data_t, torch.ones_like(gate))}
    image_rows = _image_rows(data_t, gate, "UTILITY_GATE") + _image_rows(data_t, p2b_gate, "P2B_SCORE_GATE")
    class_anomaly_aucs = [row["utility_predictor_anomaly"]["auroc_a1_better"] for row in class_rows]
    class_anomaly_gains = [row["utility_gate_gain"]["anomaly"]["mean"] for row in class_rows]
    class_normal_gains = [row["utility_gate_gain"]["normal"]["mean"] for row in class_rows]
    auroc_values = [value for value in class_anomaly_aucs if value is not None]
    pass_checks = {
        "class_held_out_anomaly_auroc_exceeds_p2b_baseline": aggregate["utility_predictor_anomaly"]["auroc_a1_better"] > aggregate["p2b_baseline_anomaly"]["auroc_a1_better"],
        "class_consistency_all_classes_above_p2b": all(row["utility_predictor_anomaly"]["auroc_a1_better"] is not None and row["utility_predictor_anomaly"]["auroc_a1_better"] > row["p2b_baseline_anomaly"]["auroc_a1_better"] for row in class_rows),
        "normal_noninferiority": aggregate["utility_gate_gain"]["normal"]["mean"] >= -safety_margin,
        "anomaly_gain_positive": aggregate["utility_gate_gain"]["anomaly"]["mean"] > 0.0,
        "no_single_class": min(class_anomaly_gains) > 0.0 and min(auroc_values) > aggregate["p2b_baseline_anomaly"]["auroc_a1_better"],
    }
    decision = "COUNTERFACTUAL_UTILITY_GATE_OFFLINE_PASS" if all(pass_checks.values()) else "UTILITY_GATE_SIGNAL_NOT_GENERALIZABLE"
    output = args.output.resolve()
    report = {
        "decision": decision,
        "pre_registered_protocol": {"dataset": "VisA", "split": "train", "manifest": str(args.manifest), "class_held_out_folds": FOLDS, "class_fold_assignment": class_to_fold, "features": ["base_predictive_logit=L0_normal_minus_abnormal", "delta_logit=L1-L0"], "primary_target": "A1_better=(BCE(y,L0)-BCE(y,L1))>0", "models": "balanced L2 logistic regression; only if its OOF anomaly AUROC does not exceed P2B, one fixed 2->4->1 Tanh MLP (AdamW lr=.02, 300 full-batch CPU steps); no feature or hyperparameter sweep", "gate": "q=selected held-out utility log-odds; q<=0 -> g=0; q>0 -> g=sigmoid(q)", "rho": RHO, "normal_noninferiority_formula": "max(1e-6, 0.001 * BASE_Normal_BCE)", "normal_noninferiority_margin": safety_margin, "base_normal_bce": base_normal_bce, "optimizer_steps_main_model": 0},
        "provenance": {"repo_sha": _git_sha(), "script_version": SCRIPT_VERSION, "script_sha256": _sha256(Path(__file__).resolve()), "k1_checkpoint_sha256": data["k1_checkpoint_sha256"], "p2b_checkpoint_sha256": data["p2b_checkpoint_sha256"], "config_sha256": data["config_sha256"], "manifest_sha256": data["manifest_sha256"], "seed": args.seed, "precision": "frozen FP32 inference; CPU utility fitting"},
        "folds": fold_records, "linear_coefficients": coefficients, "model_comparison": {"linear_logistic_anomaly": linear_anomaly, "p2b_baseline_anomaly": p2b_anomaly, "tiny_mlp_anomaly": mlp_anomaly, "selected_model": selected_model, "tiny_mlp_parameter_count": 17 if mlp_q is not None else 0}, "aggregate": aggregate, "per_class": class_rows, "per_image": image_rows, "pass_checks": pass_checks,
        "interpretation": "All U0 predictor scores are out-of-fold by class. The P2B map is a non-learned baseline evaluated on the same frozen patch set; no Phase2B or K1 parameter receives a utility-model gradient.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "p2b_anomaly_auroc": aggregate["p2b_baseline_anomaly"]["auroc_a1_better"], "utility_anomaly_auroc": aggregate["utility_predictor_anomaly"]["auroc_a1_better"], "normal_gain": aggregate["utility_gate_gain"]["normal"]["mean"], "anomaly_gain": aggregate["utility_gate_gain"]["anomaly"]["mean"], "safety_margin": safety_margin, "pass_checks": pass_checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
