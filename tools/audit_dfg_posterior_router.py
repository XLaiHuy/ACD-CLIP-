#!/usr/bin/env python3
"""Offline train-only counterfactual audit for detached DFG posterior routing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


POLICIES = ("learned_failed", "uniform", "dfg_posterior", "teacher", "oracle")


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    return -(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(-1) / float(torch.log(torch.tensor(2.0)))


def _gain(base: torch.Tensor, delta: torch.Tensor, probability: torch.Tensor, target: torch.Tensor, act: torch.Tensor, scale: float, use_act: bool) -> torch.Tensor:
    correction = (probability * delta).sum(-1)
    if use_act:
        correction = correction * act
    before = F.binary_cross_entropy_with_logits(base, target, reduction="none")
    after = F.binary_cross_entropy_with_logits(base + scale * correction, target, reduction="none")
    return before - after


def _bootstrap_image_ci(values: torch.Tensor, image_ids: torch.Tensor, selector: torch.Tensor, seed: int = 0, repeats: int = 1000) -> list[float] | None:
    selected_images = torch.unique(image_ids[selector])
    if selected_images.numel() < 2:
        return None
    by_image = torch.stack([values[selector & (image_ids == image)].mean() for image in selected_images])
    generator = torch.Generator().manual_seed(seed)
    draws = by_image[torch.randint(by_image.numel(), (repeats, by_image.numel()), generator=generator)].mean(1)
    return [float(torch.quantile(draws, 0.025)), float(torch.quantile(draws, 0.975))]


def _region_metrics(probability: torch.Tensor, teacher: torch.Tensor, delta: torch.Tensor, base: torch.Tensor, target: torch.Tensor, act: torch.Tensor, image_ids: torch.Tensor, selector: torch.Tensor) -> dict:
    if not bool(selector.any()):
        return {"patches": 0}
    teacher_hard = teacher.argmax(-1)
    route_hard = probability.argmax(-1)
    dfg_entropy = _entropy(probability)
    metrics = {
        "patches": int(selector.sum()),
        "images": int(torch.unique(image_ids[selector]).numel()),
        "p_role0_mean": float(probability[selector, 0].mean()),
        "p_role1_mean": float(probability[selector, 1].mean()),
        "agreement_teacher": float((route_hard[selector] == teacher_hard[selector]).float().mean()),
        "kl_to_teacher": float((teacher[selector].clamp_min(1e-12) * (teacher[selector].clamp_min(1e-12).log() - probability[selector].clamp_min(1e-12).log())).sum(-1).mean()),
        "entropy_mean": float(dfg_entropy[selector].mean()),
        "entropy_std": float(dfg_entropy[selector].std(unbiased=False)),
        "route_role1_share": float(route_hard[selector].float().mean()),
    }
    for name, scale, use_act in (("pre_rho_act1", 1.0, False), ("rho005_act1", 0.05, False), ("rho005_current_act", 0.05, True)):
        gain = _gain(base, delta, probability, target, act, scale, use_act)
        metrics[name] = {
            "mean_gain": float(gain[selector].mean()),
            "harm_fraction": float((gain[selector] < 0).float().mean()),
            "image_bootstrap_ci95": _bootstrap_image_ci(gain, image_ids, selector),
        }
    return metrics


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
    posterior = dfg["dfg_posterior"][group_ids, image_ids, patch_ids].float()
    base_logits = dfg["base_group_logits"][group_ids, image_ids, patch_ids].float()
    base = base_logits[:, 1] - base_logits[:, 0]
    if not torch.allclose(base, source["base_logit"].float(), atol=2e-5, rtol=2e-5):
        difference = float((base - source["base_logit"].float()).abs().max())
        raise RuntimeError(f"base-logit capture mismatch max_abs={difference}")
    teacher = source["teacher_probability"].float()
    learned = source["probabilities"].float()
    delta = source["delta"].float()
    act = source["act_probability"].float()
    target = source["target"].float()
    utility = _gain(base, delta, torch.eye(2)[torch.argmin(torch.stack([
        F.binary_cross_entropy_with_logits(base + 0.05 * delta[:, role], target, reduction="none") for role in range(2)
    ]), dim=0)], target, act, 0.05, False)
    # Oracle is the role that minimizes rho=.05 ACT=1 patch BCE.
    oracle_role = torch.argmin(torch.stack([
        F.binary_cross_entropy_with_logits(base + 0.05 * delta[:, role], target, reduction="none") for role in range(2)
    ]), dim=0)
    oracle = F.one_hot(oracle_role, num_classes=2).float()
    policies = {
        "learned_failed": learned,
        "uniform": torch.full_like(learned, 0.5),
        "dfg_posterior": posterior,
        "teacher": teacher,
        "oracle": oracle,
    }
    selectors = {
        "all": torch.ones_like(target, dtype=torch.bool),
        "normal": target < 0.5,
        "anomaly": target >= 0.5,
        "high_confidence_anomaly": (target >= 0.5) & (teacher[:, 1] >= 0.75),
    }
    report = {
        "status": "PASS",
        "support": {
            "source_capture": str(args.source_capture.resolve()),
            "dfg_capture": str(args.dfg_capture.resolve()),
            "patches": int(target.numel()),
            "images": int(torch.unique(image_ids).numel()),
            "base_logit_reconstruction_max_abs_error": float((base - source["base_logit"].float()).abs().max()),
            "dfg_probability_sum_max_abs_error": float((posterior.sum(-1) - 1.0).abs().max()),
            "dfg_logit_std": float(base_logits.std(unbiased=False)),
        },
        "policies": {
            policy: {region: _region_metrics(probability, teacher, delta, base, target, act, image_ids, selector) for region, selector in selectors.items()}
            for policy, probability in policies.items()
        },
        "dfg_vs_learned": {
            region: {
                "agreement_delta": _region_metrics(posterior, teacher, delta, base, target, act, image_ids, selector).get("agreement_teacher", 0.0) - _region_metrics(learned, teacher, delta, base, target, act, image_ids, selector).get("agreement_teacher", 0.0),
                "rho005_act1_gain_delta": _region_metrics(posterior, teacher, delta, base, target, act, image_ids, selector).get("rho005_act1", {}).get("mean_gain", 0.0) - _region_metrics(learned, teacher, delta, base, target, act, image_ids, selector).get("rho005_act1", {}).get("mean_gain", 0.0),
            }
            for region, selector in selectors.items()
        },
        "interpretation_rule": "positive anomaly rho005 ACT=1 gain, nonnegative normal rho005 ACT=1 gain, and nondegenerate entropy/role shares are required for DFG posterior routing",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "dfg_all": report["policies"]["dfg_posterior"]["all"],
        "dfg_anomaly": report["policies"]["dfg_posterior"]["anomaly"],
        "dfg_normal": report["policies"]["dfg_posterior"]["normal"],
    }, indent=2))


if __name__ == "__main__":
    main()
