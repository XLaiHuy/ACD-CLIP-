#!/usr/bin/env python3
"""Cache-only R2 Router/ACT root-cause decomposition.

Consumes a persisted E3 cache only.  It never constructs the model, invokes
backward, or writes parameters/checkpoints.  The report explicitly records
which requested representation diagnostics are impossible because the legacy
E3 cache did not persist Router input/query/key tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from model.h6.utility_routing import utility_teacher


RHO = 0.05
ROLE_SCALE = 0.0005203147302381694


def _json(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _stats(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().flatten()
    if not values.numel():
        return {"count": 0}
    q = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=values.dtype)
    qq = torch.quantile(values, q)
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p01": float(qq[0]),
        "p05": float(qq[1]),
        "p50": float(qq[2]),
        "p95": float(qq[3]),
        "p99": float(qq[4]),
    }


def _binary_gain(base: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, _binary_gain.targets, reduction="none")
    return (base - loss) / base.clamp_min(0.1)


def _gain_stats(gain: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    values = gain[mask]
    out = _stats(values)
    if values.numel():
        out["positive_fraction"] = float((values > 0).float().mean())
        out["negative_fraction"] = float((values <= 0).float().mean())
    return out


def _hard_frequency(hard: torch.Tensor, mask: torch.Tensor) -> list[float]:
    values = hard[mask]
    if not values.numel():
        return [0.0, 0.0]
    return [float((values == role).float().mean()) for role in range(2)]


def _counterfactual_matrix(
    z0: torch.Tensor,
    residual: torch.Tensor,
    dense: torch.Tensor,
    act: torch.Tensor,
    gain: torch.Tensor,
    q_teacher: torch.Tensor,
    valid: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    winner = gain.argmax(dim=-1, keepdim=True)
    corrections = {
        "best_role_oracle": residual.gather(-1, winner).squeeze(-1),
        "r2_teacher_routing": (q_teacher * residual).sum(dim=-1),
        "current_router": (dense * residual).sum(dim=-1),
    }
    base = F.binary_cross_entropy_with_logits(z0, target, reduction="none")
    _binary_gain.targets = target
    regions = {
        "ALL": valid,
        "NORMAL": valid & (target < 0.5),
        "ANOMALY": valid & (target >= 0.5),
    }
    out: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for region_name, region in regions.items():
        out[region_name] = {}
        for name, correction in corrections.items():
            out[region_name][name] = {}
            for act_name, multiplier in (("ACT_1", 1.0), ("CURRENT_ACT", act)):
                corrected = correction * multiplier
                out[region_name][name][act_name] = {
                    "pre_rho_gain": _gain_stats(_binary_gain(base, z0 + corrected), region),
                    "rho_0_05_gain": _gain_stats(_binary_gain(base, z0 + RHO * corrected), region),
                }
    return out


def _region_masks(valid: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "ALL": valid,
        "NORMAL": valid & (target < 0.5),
        "ANOMALY": valid & (target >= 0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_bytes = args.cache.read_bytes()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    config = json.loads(args.config.read_text())
    required = {"residual", "z0", "dense", "router_logits", "act", "target", "valid"}
    missing = sorted(required.difference(cache))
    if missing:
        raise RuntimeError(f"cache missing required fields: {missing}")

    residual = torch.stack(cache["residual"]).squeeze(2).float()  # [N,G,P,2]
    z0 = torch.stack(cache["z0"]).squeeze(2).float()  # [N,G,P]
    dense = torch.stack(cache["dense"]).squeeze(2).float()
    router_logits = torch.stack(cache["router_logits"]).squeeze(2).float()
    act = torch.stack(cache["act"]).squeeze(2).float()
    target_np = torch.stack([value.squeeze(0) for value in cache["target"]]).float()
    valid_np = torch.stack([value.squeeze(0) for value in cache["valid"]]).bool()
    n, groups, patches, roles = residual.shape
    if roles != 2:
        raise RuntimeError(f"expected R2 cache, got {roles} roles")

    target = target_np[:, None, :].expand(n, groups, patches)
    valid = valid_np[:, None, :].expand(n, groups, patches)
    teacher = utility_teacher(
        z0.permute(1, 0, 2),
        residual.permute(1, 0, 2, 3),
        target_np,
        valid_np,
        rho=RHO,
        router_confidence_mode="margin_rel",
        router_margin_rel_threshold=0.10,
        router_target_mode="patch_zscore_softmax",
        role_topology="r2_normal_anomaly",
        role_teacher_scale=ROLE_SCALE,
    )
    q = teacher["q_router_utility"].permute(1, 0, 2, 3).float()
    informative = teacher["informative"].permute(1, 0, 2).bool()
    role_gap = teacher["role_gap"].permute(1, 0, 2).float()
    role_entropy = teacher["role_entropy"].permute(1, 0, 2).float()
    gain = teacher["gain_rel"].permute(1, 0, 2, 3).float()

    masks = _region_masks(valid, target)
    hard_teacher = q.argmax(dim=-1)
    hard_router = dense.argmax(dim=-1)
    counterfactuals = _counterfactual_matrix(z0, residual, dense, act, gain, q, valid, target)

    teacher_report: dict[str, Any] = {}
    router_report: dict[str, Any] = {}
    supervision_report: dict[str, Any] = {}
    geometry_report: dict[str, Any] = {}
    for region_name, region in masks.items():
        teacher_probability = q[..., 0][region]
        teacher_report[region_name] = {
            "patches": int(region.sum()),
            "hard_preferred_role_frequency": _hard_frequency(hard_teacher, region),
            "role_0_probability": _stats(teacher_probability),
            "confidence": _stats(q.max(dim=-1).values[region]),
            "entropy": _stats(role_entropy[region]),
            "abs_gain_gap": _stats(role_gap.abs()[region]),
            "gain_normal": _stats(gain[..., 0][region]),
            "gain_anomaly": _stats(gain[..., 1][region]),
        }

        router_prob = dense[region]
        q_region = q[region]
        t_hard = hard_teacher[region]
        r_hard = hard_router[region]
        confusion = [
            [int(((t_hard == teacher_role) & (r_hard == router_role)).sum()) for router_role in range(2)]
            for teacher_role in range(2)
        ]
        kl = (q_region * (q_region.clamp_min(1e-12).log() - router_prob.clamp_min(1e-12).log())).sum(-1)
        router_report[region_name] = {
            "patches": int(region.sum()),
            "probability_role_0": _stats(router_prob[..., 0]),
            "probability_role_1": _stats(router_prob[..., 1]),
            "hard_role_frequency": _hard_frequency(r_hard, torch.ones_like(r_hard, dtype=torch.bool)),
            "teacher_router_confusion_rows_teacher_cols_router": confusion,
            "agreement": float((t_hard == r_hard).float().mean()),
            "teacher_to_router_kl": _stats(kl),
            "mean_absolute_probability_gap": _stats((q_region - router_prob).abs().mean(-1)),
        }

        info_region = informative & region
        valid_count = int(valid.sum())
        contribution_mass = q * info_region.unsqueeze(-1).float()
        supervision_report[region_name] = {
            "valid_support": int(region.sum()),
            "informative_support": int(info_region.sum()),
            "informative_fraction_of_region": float(info_region[region].float().mean()),
            "informative_fraction_of_all_valid": float(info_region.sum() / max(valid_count, 1)),
            "target_responsibility_mass": contribution_mass.sum(dim=(0, 1, 2)).tolist(),
            "target_responsibility_mass_fraction_of_all_informative": (
                contribution_mass.sum(dim=(0, 1, 2))
                / (q * informative.unsqueeze(-1).float()).sum().clamp_min(1e-12)
            ).tolist(),
            "hard_teacher_mass": _hard_frequency(hard_teacher, info_region),
        }

        margin = router_logits[..., 0] - router_logits[..., 1]
        router_entropy = -(dense * dense.clamp_min(1e-12).log()).sum(-1)
        geometry_report[region_name] = {
            "router_logit_role_0": _stats(router_logits[..., 0][region]),
            "router_logit_role_1": _stats(router_logits[..., 1][region]),
            "router_logit_margin_role0_minus_role1": _stats(margin[region]),
            "router_abs_logit_margin": _stats(margin.abs()[region]),
            "router_entropy": _stats(router_entropy[region]),
            "router_max_probability": _stats(dense.max(dim=-1).values[region]),
        }

    overall_info = informative & valid
    fields = sorted(cache.keys())
    result = {
        "audit": "P1_V84A_R2_ROUTER_ACT_ROOT_CAUSE_CACHE_ONLY",
        "cache": {
            "path": str(args.cache.resolve()),
            "sha256": hashlib.sha256(cache_bytes).hexdigest(),
            "images": n,
            "groups": groups,
            "patches": patches,
            "fields": fields,
            "forward_replayed": False,
            "optimizer_steps": 0,
            "backward_steps": 0,
        },
        "contract": {
            "role_topology": config.get("h6_role_topology"),
            "num_factors": config.get("h6_num_factors"),
            "rho": RHO,
            "role_teacher_scale": ROLE_SCALE,
            "router_support_normalized": config.get("h6_router_support_normalized"),
            "lambda_h6_router": config.get("lambda_h6_router"),
            "lambda_h6_route": config.get("lambda_h6_route"),
            "lambda_h6_factor_role": config.get("lambda_h6_factor_role"),
            "lambda_h6_actual_local": config.get("lambda_h6_actual_local"),
        },
        "counterfactual_matrix": counterfactuals,
        "teacher": teacher_report,
        "router": router_report,
        "effective_router_supervision": {
            "loss": "support_normalized_utility_router_loss",
            "all_valid_support": int(valid.sum()),
            "all_informative_support": int(overall_info.sum()),
            "all_informative_fraction": float(overall_info.float().mean()),
            "regions": supervision_report,
        },
        "query_key_geometry": geometry_report,
        "representation_diagnostic": {
            "status": "UNAVAILABLE_IN_PERSISTED_CACHE",
            "reason": "e3_forward_cache.pt contains Router logits/probabilities but not router_input_features, queries, concept keys, or key similarities; an input-separability classifier would therefore be unsupported.",
            "new_forward_performed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json(result), indent=2) + "\n")
    compact = {
        "output": str(args.output),
        "teacher_act1_anomaly_rho_gain": result["counterfactual_matrix"]["ANOMALY"]["r2_teacher_routing"]["ACT_1"]["rho_0_05_gain"]["mean"],
        "router_act1_anomaly_rho_gain": result["counterfactual_matrix"]["ANOMALY"]["current_router"]["ACT_1"]["rho_0_05_gain"]["mean"],
        "anomaly_agreement": result["router"]["ANOMALY"]["agreement"],
        "normal_informative_support": result["effective_router_supervision"]["regions"]["NORMAL"]["informative_support"],
        "anomaly_informative_support": result["effective_router_supervision"]["regions"]["ANOMALY"]["informative_support"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
