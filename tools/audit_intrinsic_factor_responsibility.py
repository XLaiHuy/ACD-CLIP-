#!/usr/bin/env python3
"""Cache-only TRAIN audit for state-marginalized factor responsibility."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def normalized_entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1) / torch.log(torch.tensor(2.0))


def gain(base: torch.Tensor, delta: torch.Tensor, p: torch.Tensor, target: torch.Tensor, act: torch.Tensor, rho: float, use_act: bool) -> torch.Tensor:
    routed = (p * delta).sum(-1)
    if use_act:
        routed = routed * act
    return (F.binary_cross_entropy_with_logits(base, target, reduction="none")
            - F.binary_cross_entropy_with_logits(base + rho * routed, target, reduction="none"))


def bootstrap(values: torch.Tensor, image_id: torch.Tensor, mask: torch.Tensor, repeats: int = 1000) -> list[float] | None:
    ids = torch.unique(image_id[mask])
    if ids.numel() < 2:
        return None
    per_image = torch.stack([values[mask & (image_id == item)].mean() for item in ids])
    generator = torch.Generator().manual_seed(0)
    sampled = per_image[torch.randint(per_image.numel(), (repeats, per_image.numel()), generator=generator)].mean(1)
    return [float(torch.quantile(sampled, .025)), float(torch.quantile(sampled, .975))]


def gain_summary(values: torch.Tensor, image_id: torch.Tensor, mask: torch.Tensor) -> dict:
    selected = values[mask]
    ordered = selected.sort().values
    trim = int(.01 * selected.numel())
    return {
        "mean_gain": float(selected.mean()), "median_gain": float(selected.median()),
        "trimmed_mean_gain_1pct": float(ordered[trim:-trim].mean()) if trim else float(selected.mean()),
        "harm_fraction": float((selected < 0).float().mean()), "image_bootstrap_ci95": bootstrap(values, image_id, mask),
    }


def policy_report(name: str, p: torch.Tensor, q: torch.Tensor | None, teacher: torch.Tensor, delta: torch.Tensor,
                  base: torch.Tensor, target: torch.Tensor, act: torch.Tensor, image_id: torch.Tensor, masks: dict[str, torch.Tensor]) -> dict:
    hard = p.argmax(-1)
    output = {"name": name}
    for region, mask in masks.items():
        if not bool(mask.any()):
            continue
        role = 0 if region == "normal" else 1
        record = {
            "patches": int(mask.sum()), "images": int(torch.unique(image_id[mask]).numel()),
            "mean_p0": float(p[mask, 0].mean()), "mean_p1": float(p[mask, 1].mean()),
            "role0_recall": float((hard[mask] == 0).float().mean()) if role == 0 else None,
            "role1_recall": float((hard[mask] == 1).float().mean()) if role == 1 else None,
            "normalized_entropy": float(normalized_entropy(p)[mask].mean()),
            "utility_teacher_agreement": float((hard[mask] == teacher.argmax(-1)[mask]).float().mean()),
            "teacher_kl": float((teacher[mask].clamp_min(1e-12) * (teacher[mask].clamp_min(1e-12).log() - p[mask].clamp_min(1e-12).log())).sum(-1).mean()),
        }
        if q is not None:
            record["q0_minus_q1_margin"] = float((q[mask, 0] - q[mask, 1]).mean())
            record["q1_minus_q0_margin"] = float((q[mask, 1] - q[mask, 0]).mean())
        for key, rho, use_act in (("pre_rho_act1", 1.0, False), ("rho005_act1", .05, False), ("rho005_current_act", .05, True)):
            record[key] = gain_summary(gain(base, delta, p, target, act, rho, use_act), image_id, mask)
        output[region] = record
    all_mask = torch.ones_like(target, dtype=torch.bool)
    output["all"] = {
        "balanced_region_role_accuracy": float(.5 * ((hard[masks["normal"]] == 0).float().mean() + (hard[masks["anomaly"]] == 1).float().mean())),
        "mean_p0": float(p[:, 0].mean()), "mean_p1": float(p[:, 1].mean()),
        "normalized_entropy": float(normalized_entropy(p).mean()), "hard_role1_usage": float(hard.float().mean()),
        "p_std": [float(p[:, 0].std(unbiased=False)), float(p[:, 1].std(unbiased=False))],
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    x = torch.load(args.capture, map_location="cpu", weights_only=False)
    state = x["state_similarity"].float()  # [row, factor, N/A]
    if state.ndim != 3 or tuple(state.shape[1:]) != (2, 2):
        raise ValueError("expected exactly M=2 factor state similarities")
    q = torch.logsumexp(state, dim=-1)
    p_intrinsic = F.softmax(q, dim=-1)
    target = x["target"].float()
    normal, anomaly = target < .5, target >= .5
    teacher = x["teacher_probability"].float()
    high = anomaly & (teacher[:, 1] >= .75)
    masks = {"normal": normal, "anomaly": anomaly, "high_confidence_anomaly": high}
    delta, base, act, image_id = x["delta"].float(), x["base_logit"].float(), x["act_probability"].float(), x["image_id"].long()
    learned = x["learned_probability"].float()
    uniform = torch.full_like(p_intrinsic, .5)
    losses = torch.stack([F.binary_cross_entropy_with_logits(base + .05 * delta[:, role], target, reduction="none") for role in range(2)], dim=1)
    oracle = F.one_hot(losses.argmin(-1), num_classes=2).float()
    policies = {
        "failed_independent_router": (learned, None), "uniform": (uniform, None),
        "intrinsic_factor_responsibility": (p_intrinsic, q), "r2_utility_teacher": (teacher, None), "best_role_oracle": (oracle, None),
    }
    reports = {name: policy_report(name, p, q_value, teacher, delta, base, target, act, image_id, masks) for name, (p, q_value) in policies.items()}
    q_margin = q[:, 0] - q[:, 1]
    intrinsic = reports["intrinsic_factor_responsibility"]
    q_distinct = bool(q_margin.std(unbiased=False) > 1e-6 and q_margin.abs().mean() > 1e-6)
    nondegenerate = bool(.01 < p_intrinsic[:, 1].mean() < .99 and p_intrinsic[:, 1].std(unbiased=False) > 1e-4)
    case_a = (intrinsic["normal"]["role0_recall"] >= .70 and intrinsic["anomaly"]["role1_recall"] >= .70
              and intrinsic["normal"]["rho005_act1"]["mean_gain"] >= 0 and intrinsic["anomaly"]["rho005_act1"]["mean_gain"] > 0)
    decision = "INTRINSIC_FACTOR_RESPONSIBILITY_ALREADY_PRESENT" if case_a else (
        "CASE_B_RESPONSIBILITY_GROUNDING_REQUIRED" if q_distinct and nondegenerate else "INTRINSIC_FACTOR_RESPONSIBILITY_NOT_IDENTIFIABLE"
    )
    result = {
        "audit": "INTRINSIC_FACTOR_RESPONSIBILITY_OFFLINE", "decision": decision,
        "formula": {"state_similarity": "s_m^state = tau_factor * sim(v_i, T_m^state)", "responsibility_score": "q_m = logsumexp(s_m^N, s_m^A)", "probability": "p = softmax_m(q)", "routed_residual": "sum_m p_m * delta_m"},
        "support": {"capture": str(args.capture.resolve()), "images": int(torch.unique(image_id).numel()), "patches": int(target.numel()),
                    "normal_patches": int(normal.sum()), "anomaly_patches": int(anomaly.sum()), "high_confidence_anomaly_patches": int(high.sum()),
                    "model_forwards": 0, "optimizer_steps": 0, "test_or_medical_data": False},
        "compatibility": {"q_mean": [float(q[:, 0].mean()), float(q[:, 1].mean())], "q_std": [float(q[:, 0].std(unbiased=False)), float(q[:, 1].std(unbiased=False))],
                          "q0_minus_q1": {"mean": float(q_margin.mean()), "std": float(q_margin.std(unbiased=False)), "p05": float(torch.quantile(q_margin, .05)), "p95": float(torch.quantile(q_margin, .95))},
                          "q_numerically_distinct": q_distinct, "nondegenerate_probability": nondegenerate},
        "policies": reports,
        "decision_checks": {"case_a_region_routing_and_gain": case_a, "q_numerically_distinct": q_distinct, "p_nondegenerate": nondegenerate},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"decision": decision, "compatibility": result["compatibility"], "intrinsic": intrinsic, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
