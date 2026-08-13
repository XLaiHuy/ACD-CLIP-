#!/usr/bin/env python3
"""Cache-only TRAIN audit of parameter-free factor-residual self-routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def entropy(probability: torch.Tensor) -> torch.Tensor:
    return -(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(-1) / float(torch.log(torch.tensor(2.0)))


def correction_gain(
    base: torch.Tensor,
    delta: torch.Tensor,
    probability: torch.Tensor,
    target: torch.Tensor,
    act: torch.Tensor,
    scale: float,
    use_act: bool,
) -> torch.Tensor:
    routed = (probability * delta).sum(-1)
    if use_act:
        routed = routed * act
    before = F.binary_cross_entropy_with_logits(base, target, reduction="none")
    after = F.binary_cross_entropy_with_logits(base + scale * routed, target, reduction="none")
    return before - after


def bootstrap_ci(values: torch.Tensor, image_ids: torch.Tensor, selector: torch.Tensor, repeats: int = 1000) -> list[float] | None:
    ids = torch.unique(image_ids[selector])
    if ids.numel() < 2:
        return None
    per_image = torch.stack([values[selector & (image_ids == image_id)].mean() for image_id in ids])
    generator = torch.Generator().manual_seed(0)
    samples = per_image[torch.randint(per_image.numel(), (repeats, per_image.numel()), generator=generator)].mean(1)
    return [float(torch.quantile(samples, 0.025)), float(torch.quantile(samples, 0.975))]


def trimmed_mean(values: torch.Tensor, trim_fraction: float = 0.01) -> float:
    if values.numel() < 100:
        return float(values.mean())
    ordered = values.sort().values
    trim = int(values.numel() * trim_fraction)
    return float(ordered[trim:-trim].mean()) if trim else float(ordered.mean())


def gain_metrics(gain: torch.Tensor, image_ids: torch.Tensor, selector: torch.Tensor) -> dict:
    values = gain[selector]
    return {
        "mean_gain": float(values.mean()),
        "median_gain": float(values.median()),
        "trimmed_mean_gain_1pct": trimmed_mean(values),
        "harm_fraction": float((values < 0).float().mean()),
        "image_bootstrap_ci95": bootstrap_ci(gain, image_ids, selector),
    }


def policy_metrics(
    probability: torch.Tensor,
    teacher: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    target: torch.Tensor,
    act: torch.Tensor,
    image_ids: torch.Tensor,
    selectors: dict[str, torch.Tensor],
) -> dict:
    hard = probability.argmax(-1)
    teacher_hard = teacher.argmax(-1)
    result: dict[str, dict] = {}
    for name, selector in selectors.items():
        if not bool(selector.any()):
            result[name] = {"patches": 0}
            continue
        role0 = (teacher_hard == 0) & selector
        role1 = (teacher_hard == 1) & selector
        recall0 = (hard[role0] == 0).float().mean() if bool(role0.any()) else torch.tensor(float("nan"))
        recall1 = (hard[role1] == 1).float().mean() if bool(role1.any()) else torch.tensor(float("nan"))
        out = {
            "patches": int(selector.sum()),
            "images": int(torch.unique(image_ids[selector]).numel()),
            "p_role0_mean": float(probability[selector, 0].mean()),
            "p_role1_mean": float(probability[selector, 1].mean()),
            "role0_agreement": float(recall0) if bool(role0.any()) else None,
            "role1_agreement": float(recall1) if bool(role1.any()) else None,
            "balanced_role_agreement": float(torch.nanmean(torch.stack([recall0, recall1]))),
            "teacher_kl": float((teacher[selector].clamp_min(1e-12) * (teacher[selector].clamp_min(1e-12).log() - probability[selector].clamp_min(1e-12).log())).sum(-1).mean()),
            "normalized_entropy": float(entropy(probability)[selector].mean()),
            "hard_role1_share": float(hard[selector].float().mean()),
        }
        for key, scale, use_act in (
            ("pre_rho_act1", 1.0, False),
            ("rho005_act1", 0.05, False),
            ("rho005_current_act", 0.05, True),
        ):
            out[key] = gain_metrics(correction_gain(base, delta, probability, target, act, scale, use_act), image_ids, selector)
        result[name] = out
    return result


def residual_stats(delta: torch.Tensor, selector: torch.Tensor) -> dict:
    values = delta[selector]
    return {
        "patches": int(selector.sum()),
        "delta0_mean": float(values[:, 0].mean()),
        "delta0_median": float(values[:, 0].median()),
        "delta1_mean": float(values[:, 1].mean()),
        "delta1_median": float(values[:, 1].median()),
        "p_delta0_negative": float((values[:, 0] < 0).float().mean()),
        "p_delta1_positive": float((values[:, 1] > 0).float().mean()),
    }


def pathology(
    probability: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    target: torch.Tensor,
    act: torch.Tensor,
    image_ids: torch.Tensor,
    selectors: dict[str, torch.Tensor],
) -> dict:
    hard = probability.argmax(-1)
    selected_delta = delta.gather(1, hard.unsqueeze(1)).squeeze(1)
    abs_winner = delta.abs().argmax(-1)
    gain = correction_gain(base, delta, probability, target, act, 0.05, False)
    out = {
        "hard_route_matches_largest_absolute_residual_fraction": float((hard == abs_winner).float().mean()),
        "selected_residual_abs_quantiles": {
            "p50": float(torch.quantile(selected_delta.abs(), 0.50)),
            "p95": float(torch.quantile(selected_delta.abs(), 0.95)),
            "p99": float(torch.quantile(selected_delta.abs(), 0.99)),
            "max": float(selected_delta.abs().max()),
        },
    }
    for name in ("normal", "anomaly", "high_confidence_anomaly"):
        selector = selectors[name]
        if not bool(selector.any()):
            continue
        values = gain[selector]
        top_n = max(1, int(values.numel() * 0.01))
        top_abs = values.abs().topk(top_n).values.sum()
        out[name] = {
            "selected_residual_mean": float(selected_delta[selector].mean()),
            "selected_residual_median": float(selected_delta[selector].median()),
            "gain_top1pct_abs_contribution_fraction": float(top_abs / values.abs().sum().clamp_min(1e-12)),
            "mean_gain": float(values.mean()),
            "median_gain": float(values.median()),
            "trimmed_mean_gain_1pct": trimmed_mean(values),
        }
    normal = selectors["normal"]
    anomaly = selectors["anomaly"]
    out["normal_positive_selected_delta_fraction"] = float((selected_delta[normal] > 0).float().mean())
    out["anomaly_negative_selected_delta_fraction"] = float((selected_delta[anomaly] < 0).float().mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--dfg-capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = torch.load(args.source_capture, map_location="cpu", weights_only=False)
    dfg = torch.load(args.dfg_capture, map_location="cpu", weights_only=False)
    image_ids = source["image_id"].long()
    group_ids = source["group_index"].long()
    patch_ids = source["patch_index"].long()
    delta = source["delta"].float()
    teacher = source["teacher_probability"].float()
    learned = source["probabilities"].float()
    act = source["act_probability"].float()
    target = source["target"].float()
    base = source["base_logit"].float()
    dfg_probability = dfg["dfg_posterior"][group_ids, image_ids, patch_ids].float()
    scores = torch.stack([-delta[:, 0].detach(), delta[:, 1].detach()], dim=-1)
    residual_probability = F.softmax(scores, dim=-1)
    hard_probability = F.one_hot(scores.argmax(-1), num_classes=2).float()
    factor_losses = torch.stack([
        F.binary_cross_entropy_with_logits(base + 0.05 * delta[:, role], target, reduction="none")
        for role in range(2)
    ], dim=1)
    oracle = F.one_hot(factor_losses.argmin(-1), num_classes=2).float()
    high_confidence = (target >= 0.5) & (teacher[:, 1] >= 0.75)
    selectors = {
        "all": torch.ones_like(target, dtype=torch.bool),
        "normal": target < 0.5,
        "anomaly": target >= 0.5,
        "high_confidence_anomaly": high_confidence,
    }
    policies = {
        "failed_learned": learned,
        "uniform": torch.full_like(learned, 0.5),
        "dfg_posterior": dfg_probability,
        "factor_residual_self_router": residual_probability,
        "residual_hard_argmax_diagnostic": hard_probability,
        "r2_teacher": teacher,
        "oracle": oracle,
    }
    metrics = {name: policy_metrics(probability, teacher, delta, base, target, act, image_ids, selectors) for name, probability in policies.items()}
    candidate = metrics["factor_residual_self_router"]
    c_anomaly = candidate["anomaly"]["rho005_act1"]
    c_normal = candidate["normal"]["rho005_act1"]
    nondegenerate = 0.01 < candidate["all"]["p_role1_mean"] < 0.99 and candidate["all"]["normalized_entropy"] > 0.01
    both_roles = 0.01 < candidate["all"]["hard_role1_share"] < 0.99
    pathology_report = pathology(residual_probability, delta, base, target, act, image_ids, selectors)
    decision_checks = {
        "anomaly_rho005_act1_positive": c_anomaly["mean_gain"] > 0.0,
        "normal_rho005_act1_nonharmful": c_normal["mean_gain"] >= 0.0,
        "anomaly_better_than_failed_learned": c_anomaly["mean_gain"] > metrics["failed_learned"]["anomaly"]["rho005_act1"]["mean_gain"],
        "anomaly_better_than_dfg": c_anomaly["mean_gain"] > metrics["dfg_posterior"]["anomaly"]["rho005_act1"]["mean_gain"],
        "nondegenerate": nondegenerate,
        "both_roles_hard_support": both_roles,
        "normal_positive_selected_delta_low": pathology_report["normal_positive_selected_delta_fraction"] <= 0.05,
        "anomaly_negative_selected_delta_low": pathology_report["anomaly_negative_selected_delta_fraction"] <= 0.05,
        "not_extreme_dominated": pathology_report["anomaly"]["gain_top1pct_abs_contribution_fraction"] <= 0.50,
    }
    decision = "FACTOR_RESIDUAL_SELF_ROUTER_OFFLINE_PASS" if all(decision_checks.values()) else "FACTOR_RESIDUAL_SELF_ROUTER_OFFLINE_FAIL"
    report = {
        "decision": decision,
        "status": "PASS" if decision.endswith("PASS") else "FAIL",
        "support": {
            "source_capture": str(args.source_capture.resolve()),
            "dfg_capture": str(args.dfg_capture.resolve()),
            "images": int(torch.unique(image_ids).numel()),
            "patches": int(target.numel()),
            "normal_patches": int(selectors["normal"].sum()),
            "anomaly_patches": int(selectors["anomaly"].sum()),
            "high_confidence_anomaly_patches": int(high_confidence.sum()),
            "model_forwards": 0,
            "optimizer_steps": 0,
            "test_or_medical_data": False,
        },
        "residual_polarity": {name: residual_stats(delta, selector) for name, selector in selectors.items() if name != "all"},
        "factor_residual_self_router": {
            "formula": "softmax(stack([-delta0.detach(), +delta1.detach()]))",
            "score_mean": [float(scores[:, 0].mean()), float(scores[:, 1].mean())],
            "score_std": [float(scores[:, 0].std(unbiased=False)), float(scores[:, 1].std(unbiased=False))],
        },
        "policies": metrics,
        "self_selection_pathology": pathology_report,
        "decision_checks": decision_checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "decision": decision,
        "residual_polarity": report["residual_polarity"],
        "candidate": metrics["factor_residual_self_router"],
        "pathology": pathology_report,
        "decision_checks": decision_checks,
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
