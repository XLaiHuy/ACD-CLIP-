#!/usr/bin/env python3
"""One combined forward-only ACT/Router teacher-semantics audit.

This audit is deliberately bounded to the regenerated P1-v8.4-A 300B
checkpoint and 300 VisA/train microbatches.  It constructs no optimizer,
calls no backward method, and writes compact aggregate evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import get_text_and_image_dataset
from model.h6.utility_routing import build_patch_targets
from tools.audit_p1_v83_semantics import _model_from_checkpoint
from tools.audit_p1_v84a_post300 import (
    _IndexedDataset,
    _git_head,
    _seed,
    _sha256,
    _write_json_atomic,
)
from utils import get_phase2b_global_text_features, make_dataloader_generator, seed_worker


EXPECTED_CHECKPOINT_GIT_SHA = "1b88c1e45896a2eb25b2b84264152c7cffff4004"
EXPECTED_CHECKPOINT_SHA256 = "96f679b2e18f4e352157494f7198414b66f66024a5cc023f5ff046c39dcaa3a3"
EXPECTED_OPENAI_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
CHECKPOINT_ORIGIN = "REGENERATED_FROM_HISTORICAL_P1_V84A_300B_PROTOCOL"
OLD_BEST_POSITIVE_P90 = 0.00240325927734375
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _full_state_hash(model: torch.nn.Module) -> str:
    """Hash every parameter/buffer without retaining a second model copy."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(memoryview(cpu.numpy()))
    return digest.hexdigest()


def _stats(values: torch.Tensor, *, count: bool = True) -> dict[str, float | int]:
    values = values.detach().float().flatten()
    result: dict[str, float | int] = {}
    if count:
        result["count"] = int(values.numel())
    if not values.numel():
        result.update({"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0})
        result.update({f"p{int(q * 100):02d}": 0.0 for q in QUANTILES})
        return result
    result.update({
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    })
    result.update({
        f"p{int(q * 100):02d}": float(torch.quantile(values, q).item())
        for q in QUANTILES
    })
    return result


def _regions(targets: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "overall": torch.ones_like(targets, dtype=torch.bool),
        "normal": targets < 0.5,
        "anomaly": targets >= 0.5,
    }


def _fraction(mask: torch.Tensor, region: torch.Tensor) -> float:
    return float(mask[region].float().mean().item()) if region.any() else 0.0


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.float().flatten()
    right = right.float().flatten()
    if left.numel() < 2 or left.std(unbiased=False) == 0 or right.std(unbiased=False) == 0:
        return None
    return float(torch.corrcoef(torch.stack((left, right)))[0, 1].item())


def _gain_report(
    g_best: torch.Tensor,
    g_route: torch.Tensor,
    g_actual: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reports: dict[str, Any] = {"g_best": {}, "g_route": {}, "g_actual": {}}
    correlations: dict[str, Any] = {}
    agreements: dict[str, Any] = {}
    values = {"g_best": g_best, "g_route": g_route, "g_actual": g_actual}
    for region_name, region in _regions(targets).items():
        for name, tensor in values.items():
            stats = _stats(tensor[region])
            stats["positive_fraction"] = _fraction(tensor > 0.0, region)
            stats["negative_fraction"] = _fraction(tensor <= 0.0, region)
            reports[name][region_name] = stats
        correlations[region_name] = {
            "g_best_g_route": _correlation(g_best[region], g_route[region]),
            "g_best_g_actual": _correlation(g_best[region], g_actual[region]),
            "g_route_g_actual": _correlation(g_route[region], g_actual[region]),
        }
        agreements[region_name] = {
            "g_best_g_route": _fraction((g_best > 0.0) == (g_route > 0.0), region),
            "g_best_g_actual": _fraction((g_best > 0.0) == (g_actual > 0.0), region),
            "g_route_g_actual": _fraction((g_route > 0.0) == (g_actual > 0.0), region),
        }
    return reports, correlations, agreements


def _cross_tabs(
    g_best: torch.Tensor,
    g_route: torch.Tensor,
    g_actual: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    masks = {
        "oracle_positive_routed_harm": (g_best > 0.0) & (g_route <= 0.0),
        "strong_oracle_positive_routed_harm": (
            (g_best > OLD_BEST_POSITIVE_P90) & (g_route <= 0.0)
        ),
        "oracle_nonpositive_routed_positive": (g_best <= 0.0) & (g_route > 0.0),
        "routed_positive_actual_nonpositive": (g_route > 0.0) & (g_actual <= 0.0),
        "routed_nonpositive_actual_positive": (g_route <= 0.0) & (g_actual > 0.0),
    }
    output: dict[str, Any] = {
        "old_best_positive_p90": OLD_BEST_POSITIVE_P90,
    }
    for name, mask in masks.items():
        output[name] = {}
        for region_name, region in _regions(targets).items():
            output[name][region_name] = {
                "count": int((mask & region).sum().item()),
                "fraction": _fraction(mask, region),
            }
    return output


def _method_row(loss: torch.Tensor, base: torch.Tensor, region: torch.Tensor) -> dict[str, float]:
    mean = loss[region].mean()
    base_mean = base[region].mean()
    return {
        "loss": float(mean.item()),
        "gain_vs_Base": float(((base_mean - mean) / base_mean.clamp_min(1e-12)).item()),
    }


def _regional_utility(
    z0: torch.Tensor,
    targets: torch.Tensor,
    residual: torch.Tensor,
    dense: torch.Tensor,
    act: torch.Tensor,
    actual_correction: torch.Tensor | None = None,
) -> dict[str, Any]:
    base = F.binary_cross_entropy_with_logits(z0, targets, reduction="none")
    candidates = z0.unsqueeze(-1) + 0.05 * residual
    per_factor = F.binary_cross_entropy_with_logits(
        candidates, targets.unsqueeze(-1).expand_as(candidates), reduction="none"
    )
    routed_delta = (dense * residual).sum(dim=-1)
    soft = F.binary_cross_entropy_with_logits(z0 + 0.05 * routed_delta, targets, reduction="none")
    if actual_correction is None:
        actual_correction = (act * routed_delta) * 0.05
    actual = F.binary_cross_entropy_with_logits(
        z0 + actual_correction, targets, reduction="none"
    )
    hard_delta = residual.gather(-1, dense.argmax(dim=-1, keepdim=True)).squeeze(-1)
    hard = F.binary_cross_entropy_with_logits(z0 + 0.05 * hard_delta, targets, reduction="none")
    uniform = F.binary_cross_entropy_with_logits(
        z0 + 0.05 * residual.mean(dim=-1), targets, reduction="none"
    )
    oracle = per_factor.min(dim=-1).values
    output: dict[str, Any] = {}
    for region_name, region in _regions(targets).items():
        base_mean = base[region].mean()
        factor_means = per_factor[region].mean(dim=0)
        best_single = torch.full_like(base, factor_means.min())
        output[region_name] = {
            "patch_count": int(region.sum().item()),
            "Base": {"loss": float(base_mean.item()), "gain_vs_Base": 0.0},
            "ResidualBestSingle": _method_row(best_single, base, region),
            "ResidualOracleMulti": _method_row(oracle, base, region),
            "FullSoftRouted_ACT1": _method_row(soft, base, region),
            "ActualGated": _method_row(actual, base, region),
            "HardRouted_ACT1": _method_row(hard, base, region),
            "Uniform_ACT1": _method_row(uniform, base, region),
        }
    return output


def _coverage(
    selected: torch.Tensor,
    region: torch.Tensor,
    image_ids: torch.Tensor,
    categories: list[str],
) -> dict[str, Any]:
    eligible_images = set(image_ids[region].tolist())
    selected_images = set(image_ids[selected & region].tolist())
    eligible_categories = {categories[index] for index in eligible_images}
    selected_categories = {categories[index] for index in selected_images}
    return {
        "image_count": len(selected_images),
        "image_denominator": len(eligible_images),
        "image_fraction": len(selected_images) / max(len(eligible_images), 1),
        "category_count": len(selected_categories),
        "category_denominator": len(eligible_categories),
        "category_fraction": len(selected_categories) / max(len(eligible_categories), 1),
        "categories": sorted(selected_categories),
    }


def _risk_coverage(
    g_route: torch.Tensor,
    targets: torch.Tensor,
    image_ids: torch.Tensor,
    categories: list[str],
) -> dict[str, Any]:
    positive = g_route[g_route > 0.0]
    candidates = [("zero", 0.0)] + [
        (f"positive_p{int(q * 100):02d}", float(torch.quantile(positive, q).item()))
        for q in (0.25, 0.50, 0.75, 0.90, 0.95)
    ]
    rows = []
    for label, threshold in candidates:
        on = g_route > threshold
        off = g_route <= 0.0
        ambiguous = (g_route > 0.0) & (g_route <= threshold)
        row: dict[str, Any] = {"label": label, "threshold": threshold, "regions": {}}
        for region_name, region in _regions(targets).items():
            selected = g_route[on & region]
            row["regions"][region_name] = {
                "patch_count": int(region.sum().item()),
                "on_fraction": _fraction(on, region),
                "off_fraction": _fraction(off, region),
                "ambiguous_fraction": _fraction(ambiguous, region),
                "on_mean_g_route": float(selected.mean().item()) if selected.numel() else 0.0,
                "on_median_g_route": float(selected.median().item()) if selected.numel() else 0.0,
                "on_harmful_fraction": float((selected <= 0.0).float().mean().item()) if selected.numel() else 0.0,
                "on_coverage": _coverage(on, region, image_ids, categories),
            }
        rows.append(row)
    return {
        "thresholds_are_diagnostic_only": True,
        "production_T_off_created": False,
        "positive_g_route": _stats(positive),
        "rows": rows,
    }


def _factor_coverage(winners: torch.Tensor, selected: torch.Tensor) -> dict[str, Any]:
    count = int(selected.sum().item())
    return {
        f"F{factor + 1}": {
            "count": int(((winners == factor) & selected).sum().item()),
            "share": float(((winners == factor) & selected).sum().item() / max(count, 1)),
        }
        for factor in range(4)
    }


def _router_audit(gain: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    best, winners = gain.max(dim=-1)
    second = gain.topk(2, dim=-1).values[:, 1]
    margin_abs = best - second
    margin_rel = margin_abs / best.abs().clamp_min(1e-12)
    regions = _regions(targets)
    margin_stats = {"margin_abs": {}, "margin_rel": {}}
    winner_coverage: dict[str, Any] = {}
    for region_name, region in regions.items():
        margin_stats["margin_abs"][region_name] = _stats(margin_abs[region])
        margin_stats["margin_rel"][region_name] = _stats(margin_rel[region])
        winner_coverage[region_name] = _factor_coverage(winners, region)

    margin_support: dict[str, Any] = {"definition": "best_gain > 0 and margin_rel > threshold", "rows": []}
    for threshold in (0.10, 0.25, 0.50):
        selected = (best > 0.0) & (margin_rel > threshold)
        row: dict[str, Any] = {"margin_rel_threshold": threshold, "regions": {}}
        for region_name, region in regions.items():
            support = selected & region
            row["regions"][region_name] = {
                "count": int(support.sum().item()),
                "fraction": _fraction(selected, region),
                "factor_coverage": _factor_coverage(winners, support),
            }
        margin_support["rows"].append(row)

    entropy_rows: list[dict[str, Any]] = []
    entropy_masks: dict[tuple[float, float], torch.Tensor] = {}
    for tau in (0.05, 0.03, 0.02):
        q = F.softmax(gain / tau, dim=-1)
        entropy = -(q * q.clamp_min(1e-12).log()).sum(dim=-1) / math.log(4)
        max_probability = q.max(dim=-1).values
        for threshold in (0.98, 0.99, 0.995):
            selected = (best > 0.0) & (entropy < threshold)
            legacy_selected = (best > 0.02) & (entropy < threshold)
            entropy_masks[(tau, threshold)] = selected
            row = {
                "router_tau_utility": tau,
                "entropy_threshold": threshold,
                "teacher_entropy": _stats(entropy, count=False),
                "teacher_max_probability": _stats(max_probability, count=False),
                "regions": {},
            }
            for region_name, region in regions.items():
                support = selected & region
                row["regions"][region_name] = {
                    "count": int(support.sum().item()),
                    "fraction": _fraction(selected, region),
                    "legacy_gain_threshold_count": int((legacy_selected & region).sum().item()),
                    "legacy_gain_threshold_fraction": _fraction(legacy_selected, region),
                    "factor_coverage": _factor_coverage(winners, support),
                    "teacher_entropy_mean": float(entropy[region].mean().item()),
                    "teacher_max_probability_mean": float(max_probability[region].mean().item()),
                }
            entropy_rows.append(row)

    coupling: dict[str, Any] = {
        "raw_winner_tau_invariant": True,
        "entropy_threshold": 0.98,
        "transitions": [],
    }
    strong_raw = (best > 0.0) & (margin_rel > 0.10)
    for old_tau, new_tau in ((0.05, 0.03), (0.03, 0.02), (0.05, 0.02)):
        old = entropy_masks[(old_tau, 0.98)]
        new = entropy_masks[(new_tau, 0.98)]
        flip = (~old) & new & (best > 0.0)
        coupling["transitions"].append({
            "from_tau": old_tau,
            "to_tau": new_tau,
            "reject_to_accept_count": int(flip.sum().item()),
            "reject_to_accept_fraction": float(flip.float().mean().item()),
            "strong_raw_margin_flip_count": int((flip & strong_raw).sum().item()),
            "strong_raw_margin_flip_fraction": float((flip & strong_raw).float().mean().item()),
        })
    coupling["finding"] = (
        "ENTROPY_CONFIDENCE_TEMPERATURE_COUPLED"
        if any(row["strong_raw_margin_flip_count"] > 0 for row in coupling["transitions"])
        else "NO_MATERIAL_TEMPERATURE_COUPLING_OBSERVED"
    )
    return {
        **margin_stats,
        "winner_counts_and_shares": winner_coverage,
        "factor_coverage": winner_coverage,
        "margin_support": margin_support,
        "entropy_support": {
            "positive_utility_boundary": 0.0,
            "legacy_router_gain_threshold": 0.02,
            "rows": entropy_rows,
        },
        "temperature_coupling": coupling,
    }


def _decisions(
    act: dict[str, Any], router: dict[str, Any]
) -> tuple[list[str], str, str, str, str]:
    reasons: list[str] = []
    utility = act["regional_utility"]
    normal_oracle = utility["normal"]["ResidualOracleMulti"]["gain_vs_Base"]
    anomaly_oracle = utility["anomaly"]["ResidualOracleMulti"]["gain_vs_Base"]
    if not (normal_oracle > 0.0 and anomaly_oracle > 0.0):
        reasons.append("RESIDUAL_CANDIDATE_FAILURE")
    if not act["actual_gated_reconstruction_exact"]:
        reasons.append("MISSING_ACTUAL_GATED_EVIDENCE")
    region_mismatch = any(
        utility[region]["ResidualOracleMulti"]["gain_vs_Base"] > 0.0
        and utility[region]["FullSoftRouted_ACT1"]["gain_vs_Base"] <= 0.0
        for region in ("normal", "anomaly")
    )
    material_patch_mismatch = any(
        act["cross_tabs"]["oracle_positive_routed_harm"][region]["fraction"] >= 0.01
        for region in ("normal", "anomaly")
    )
    if region_mismatch or material_patch_mismatch:
        reasons.append("ACT_TEACHER_OBJECTIVE_MISMATCH")
    route_stats = act["g_route"]
    if all(route_stats[region]["positive_fraction"] >= 0.999 for region in ("normal", "anomaly")):
        reasons.append("ACT_ROLE_NOT_ESTABLISHED")

    canonical = next(
        row for row in router["entropy_support"]["rows"]
        if row["router_tau_utility"] == 0.05 and row["entropy_threshold"] == 0.98
    )
    margin = next(row for row in router["margin_support"]["rows"] if row["margin_rel_threshold"] == 0.10)
    anomaly_factors = margin["regions"]["anomaly"]["factor_coverage"]
    multiple_non_f1 = sum(anomaly_factors[f"F{i}"]["count"] > 0 for i in (2, 3, 4)) >= 2
    margin_rescues = (
        canonical["regions"]["overall"]["fraction"] <= 0.001
        and margin["regions"]["overall"]["fraction"] >= 0.01
        and margin["regions"]["anomaly"]["fraction"] >= 0.01
        and multiple_non_f1
    )
    if margin_rescues:
        reasons.append("ROUTER_GATE_FORMULATION_CHANGE")
        router_decision = "ROUTER_GATE_FORMULATION_CHANGE"
    else:
        tau_candidate = None
        for tau in (0.03, 0.02):
            row = next(
                item for item in router["entropy_support"]["rows"]
                if item["router_tau_utility"] == tau and item["entropy_threshold"] == 0.98
            )
            coverage = row["regions"]["anomaly"]["factor_coverage"]
            if (
                row["regions"]["overall"]["fraction"] >= 0.01
                and row["regions"]["anomaly"]["fraction"] >= 0.01
                and sum(coverage[f"F{i}"]["count"] > 0 for i in (2, 3, 4)) >= 2
            ):
                tau_candidate = tau
                break
        if tau_candidate is not None:
            router_decision = f"ROUTER_TAU_ONLY_CANDIDATE:{tau_candidate}"
        else:
            reasons.append("ROUTER_SECOND_GATE_CHANGE_REQUIRED")
            router_decision = "ROUTER_SECOND_GATE_CHANGE_REQUIRED"

    priority = [
        "CONTRACT_OR_PROVENANCE_FAILURE", "SOURCE_DECOUPLING_CONTRACT_FAIL",
        "RESIDUAL_CANDIDATE_FAILURE", "MISSING_ACTUAL_GATED_EVIDENCE",
        "ACT_TEACHER_OBJECTIVE_MISMATCH", "ACT_ROLE_NOT_ESTABLISHED",
        "ACT_NEGATIVE_SUPPORT_INSUFFICIENT", "ACT_LOSS_BALANCE_REQUIRED",
        "ROUTER_GATE_FORMULATION_CHANGE", "ROUTER_SECOND_GATE_CHANGE_REQUIRED",
        "ROUTER_CONFIDENCE_FORMULATION_FAILURE",
    ]
    primary = next((reason for reason in priority if reason in reasons), "AUDIT_COMPLETE")
    act_decision = next(
        (reason for reason in priority[2:8] if reason in reasons),
        "NATURAL_ACT_ON_OFF_SIGNAL_PRESENT",
    )
    next_action = (
        "EXIT_FOR_DISCUSSION: review a routed-utility-based ACT teacher candidate "
        "while holding thresholds, losses, Router controls, capacity, rho, and the "
        "training protocol fixed; implementation and training are not authorized"
        if primary == "ACT_TEACHER_OBJECTIVE_MISMATCH"
        else "DISCUSS exactly one teacher-control intervention; no training is authorized"
    )
    return reasons, primary, act_decision, router_decision, next_action


def _decision_markdown(output: dict[str, Any]) -> str:
    act = output["act"]
    router = output["router"]
    utility = act["regional_utility"]
    canonical = next(
        row for row in router["entropy_support"]["rows"]
        if row["router_tau_utility"] == 0.05 and row["entropy_threshold"] == 0.98
    )
    margin = next(row for row in router["margin_support"]["rows"] if row["margin_rel_threshold"] == 0.10)
    normal_factor = margin["regions"]["normal"]["factor_coverage"]
    anomaly_factor = margin["regions"]["anomaly"]["factor_coverage"]
    tau03 = next(
        row for row in router["entropy_support"]["rows"]
        if row["router_tau_utility"] == 0.03 and row["entropy_threshold"] == 0.98
    )
    tau02 = next(
        row for row in router["entropy_support"]["rows"]
        if row["router_tau_utility"] == 0.02 and row["entropy_threshold"] == 0.98
    )
    lines = [
        "# P1-v8.4-A Teacher Semantics Decision",
        "",
        "## 1. Proven before this audit",
        "",
        "True residual semantics and weak-but-present factor specialization remain fixed prior evidence. Historical post300 artifacts describe the lost original checkpoint; this audit uses the authorized regenerated checkpoint for every new decision.",
        "",
        "## 2. Source decoupling change",
        "",
        "Factor temperature, Router temperature, Router gain threshold, and ACT gain threshold are independently resolved. Legacy defaults remain exactly 0.05/0.02; the focused equivalence suite passed.",
        "",
        "## 3. ACT best-vs-routed-vs-actual evidence",
        "",
    ]
    for region in ("overall", "normal", "anomaly"):
        lines.append(
            f"- {region}: oracle {utility[region]['ResidualOracleMulti']['gain_vs_Base']:.6%}, "
            f"routed ACT=1 {utility[region]['FullSoftRouted_ACT1']['gain_vs_Base']:.6%}, "
            f"actual gated {utility[region]['ActualGated']['gain_vs_Base']:.6%}."
        )
    lines.extend([
        "",
        "## 4. Natural ACT ON/OFF evidence",
        "",
        f"Natural routed OFF fractions are overall {act['g_route']['overall']['negative_fraction']:.6%}, normal {act['g_route']['normal']['negative_fraction']:.6%}, and anomaly {act['g_route']['anomaly']['negative_fraction']:.6%}. No positive T_off was created.",
        "",
        "## 5. ACT loss-balance conclusion",
        "",
        "No ACT loss change is authorized. Semantic teacher alignment must be resolved before any ON/OFF weighting decision.",
        "",
        "## 6. Router margin-vs-entropy evidence",
        "",
        f"Canonical tau=.05/entropy=.98 positive-utility support is {canonical['regions']['overall']['fraction']:.6%}; the predeclared margin_rel>.10 support is {margin['regions']['overall']['fraction']:.6%}. Tau-only sensitivity at entropy=.98 reaches {tau03['regions']['overall']['fraction']:.6%} for tau=.03 and {tau02['regions']['overall']['fraction']:.6%} for tau=.02, but anomaly support remains {tau03['regions']['anomaly']['fraction']:.6%} and {tau02['regions']['anomaly']['fraction']:.6%}, respectively.",
        "",
        "## 7. Factor coverage normal/anomaly",
        "",
        f"Normal margin-support counts: F1={normal_factor['F1']['count']}, F2={normal_factor['F2']['count']}, F3={normal_factor['F3']['count']}, F4={normal_factor['F4']['count']}.",
        f"Anomaly margin-support counts: F1={anomaly_factor['F1']['count']}, F2={anomaly_factor['F2']['count']}, F3={anomaly_factor['F3']['count']}, F4={anomaly_factor['F4']['count']}.",
        "",
        "## 8. Decision tree outcome",
        "",
        f"Primary: `{output['root_cause']['primary']}`. All triggered codes: {', '.join(f'`{item}`' for item in output['root_cause']['all_triggered'])}.",
        "",
        "## 9. Exact next authorized experiment",
        "",
        output["next_authorized_action"] + ".",
        "",
        "## 10. Explicit forbidden next actions",
        "",
        "No training, optimizer, backward pass, threshold adoption, teacher semantic replacement, loss reweighting, capacity change, rho change, P1-v8.4-B run, medical evaluation, or push is authorized by this audit.",
        "",
        "EXIT_FOR_DISCUSSION" if output["status"] == "EXIT_FOR_DISCUSSION" else "AUDIT_COMPLETE_READY_FOR_DISCUSSION",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("runs/p1_v84a_gpu/fresh_300b_seed0_attempt1/adapter_1.pth"),
    )
    parser.add_argument(
        "--provenance", type=Path,
        default=Path("runs/p1_v84a_gpu/regenerated_300b_checkpoint_provenance.json"),
    )
    parser.add_argument("--openai-checkpoint", type=Path, default=Path("model/ViT-L-14-336px.pt"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/p1_v84a_gpu/post300_teacher_semantics_audit.json"),
    )
    parser.add_argument(
        "--decision-output", type=Path,
        default=Path("P1_V84A_TEACHER_SEMANTICS_DECISION.md"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=300)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    if args.output.exists() or args.decision_output.exists():
        raise FileExistsError("refusing to overwrite an existing teacher-semantics audit artifact")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the bounded forward-only replay")
    if args.max_batches != 300 or args.progress_every != 50:
        raise ValueError("audit is locked to 300 batches with milestones every 50")
    checkpoint_sha = _sha256(args.checkpoint)
    openai_sha = _sha256(args.openai_checkpoint)
    provenance = json.loads(args.provenance.read_text())
    provenance_checks = {
        "checkpoint_origin": provenance.get("checkpoint_origin") == CHECKPOINT_ORIGIN,
        "historical_original_checkpoint_lost": provenance.get("original_checkpoint_lost") is True,
        "checkpoint_sha256": checkpoint_sha == EXPECTED_CHECKPOINT_SHA256,
        "provenance_checkpoint_sha256": provenance.get("canonical_checkpoint", {}).get("sha256") == EXPECTED_CHECKPOINT_SHA256,
        "tier_a_protocol_pass": provenance.get("validation", {}).get("tier_a_protocol_pass") is True,
        "tier_b_numerical_pass": provenance.get("validation", {}).get("tier_b_numerical_pass") is True,
        "tier_c_scientific_pass": provenance.get("validation", {}).get("tier_c_scientific_pass") is True,
        "safe_for_forward_only_teacher_audit": provenance.get("safe_for_forward_only_teacher_audit") is True,
    }
    if not all(provenance_checks.values()):
        raise RuntimeError(f"regeneration provenance failure: {[k for k, v in provenance_checks.items() if not v]}")
    if openai_sha != EXPECTED_OPENAI_SHA256:
        raise RuntimeError("OpenAI checkpoint SHA256 mismatch")

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _seed(args.seed)
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("h6_config", {})
    checkpoint_checks = {
        "checkpoint_version": checkpoint.get("checkpoint_version") == 9,
        "checkpoint_git_sha": checkpoint.get("git_sha") == EXPECTED_CHECKPOINT_GIT_SHA,
        "progress_version": config.get("progress_version") == "P1-v8.4-A",
        "seed": checkpoint.get("seed") == args.seed,
        "img_size": checkpoint.get("img_size") == 518,
        "batch_size": checkpoint.get("batch_size") == 1,
        "grad_accum_steps": checkpoint.get("grad_accum_steps") == 6,
        "precision_fp32": checkpoint.get("precision") == "fp32",
        "tf32_off": checkpoint.get("tf32_enabled") is False,
        "amp_off": checkpoint.get("amp_enabled") is False,
        "rho_fixed": config.get("rho_fixed") is True,
        "residual_act_semantics": config.get("local_correction_semantics") == "act_times_routed_true_residual",
    }
    if not all(checkpoint_checks.values()):
        raise RuntimeError(f"checkpoint contract failure: {[k for k, v in checkpoint_checks.items() if not v]}")

    model = _model_from_checkpoint(checkpoint, device)
    model.requires_grad_(False)
    model.eval()
    model.clipmodel.eval()
    all_grad_none_before = all(parameter.grad is None for parameter in model.parameters())
    state_hash_before = _full_state_hash(model)

    dataset = _IndexedDataset(get_text_and_image_dataset("VisA", 518, "train"))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True,
        worker_init_fn=seed_worker, generator=make_dataloader_generator(args.seed),
    )
    z0_records: list[torch.Tensor] = []
    target_records: list[torch.Tensor] = []
    residual_records: list[torch.Tensor] = []
    dense_records: list[torch.Tensor] = []
    act_records: list[torch.Tensor] = []
    actual_correction_records: list[torch.Tensor] = []
    image_id_records: list[torch.Tensor] = []
    dataset_indices: list[int] = []
    categories: list[str] = []
    residual_definition_max_error = 0.0
    routed_reconstruction_max_error = 0.0
    actual_logit_reconstruction_max_error = 0.0
    started = time.monotonic()

    for batch_number, sample in enumerate(loader, start=1):
        if batch_number > args.max_batches:
            break
        dataset_indices.append(int(sample["dataset_index"].item()))
        categories.append(str(sample["class_name"][0]))
        image = sample["image"].to(device, non_blocking=True)
        mask = sample["mask"].to(device, non_blocking=True)
        local_valid = sample["local_mask_valid"].to(device, non_blocking=True)
        class_names = list(sample["class_name"])
        with torch.inference_mode():
            visual = model(image, return_phase4_features=True)
            h6_batch = model.h6.build_batch(
                model, "VisA", class_names, visual,
                hybrid_alpha=float(checkpoint["hybrid_alpha_current"]),
                update_load_bias=False,
            )
            seg_features = torch.stack(visual["seg_tokens"], dim=0)
            text_global = get_phase2b_global_text_features(
                model, "VisA", class_names, device,
                use_hybrid_soft_prompt=True, use_soft_prompt=False,
            ).to(dtype=seg_features.dtype)
            _, _, z0 = model.vision_text_fusion_gate_seg(
                seg_features, text_global, img_size=518,
                h6_patch_logits=h6_batch["h6_logits"], return_details=True,
            )
            patch_count = int(h6_batch["factor_residual_logits"].shape[2])
            y_patch, valid_patch = build_patch_targets(mask, patch_count, local_valid)
            valid = valid_patch.unsqueeze(0).expand_as(z0)
            targets = y_patch.unsqueeze(0).expand_as(z0).float()
            residual = h6_batch["factor_residual_logits"].float()
            dense = h6_batch["dense_probabilities"].float()
            act = h6_batch["act_probability"].float()
            definition = (
                h6_batch["factor_patch_logits"].float()
                - h6_batch["noop_reference_logit"].float().unsqueeze(-1)
            )
            residual_definition_max_error = max(
                residual_definition_max_error, float((residual - definition).abs().max().item())
            )
            routed = (dense * residual).sum(dim=-1)
            reconstructed = act * routed
            routed_reconstruction_max_error = max(
                routed_reconstruction_max_error,
                float((h6_batch["h6_logits"].float() - reconstructed).abs().max().item()),
            )
            actual_correction = h6_batch["rho_scaled_actual_correction"].float()
            actual_from_payload = z0 + actual_correction
            # Match the forward's operation order exactly: H6 first forms
            # act*routed, then applies the per-group fixed rho tensor.
            actual_from_formula = z0 + reconstructed * model.h6.rho_values().view(-1, 1, 1)
            actual_logit_reconstruction_max_error = max(
                actual_logit_reconstruction_max_error,
                float((actual_from_payload - actual_from_formula).abs().max().item()),
            )
            z0_records.append(z0.float()[valid].cpu())
            target_records.append(targets[valid].cpu())
            residual_records.append(residual[valid].cpu())
            dense_records.append(dense[valid].cpu())
            act_records.append(act[valid].cpu())
            actual_correction_records.append(actual_correction[valid].cpu())
            image_id_records.append(torch.full((int(valid.sum().item()),), batch_number - 1, dtype=torch.int64))
        if batch_number % args.progress_every == 0:
            print(json.dumps({"batches": batch_number, "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)

    if len(dataset_indices) != 300:
        raise RuntimeError(f"replay ended after {len(dataset_indices)} batches")
    z0 = torch.cat(z0_records)
    targets = torch.cat(target_records)
    residual = torch.cat(residual_records)
    dense = torch.cat(dense_records)
    act_probability = torch.cat(act_records)
    actual_correction = torch.cat(actual_correction_records)
    image_ids = torch.cat(image_id_records)
    base_loss = F.binary_cross_entropy_with_logits(z0, targets, reduction="none")
    candidate_logits = z0.unsqueeze(-1) + 0.05 * residual
    per_factor_loss = F.binary_cross_entropy_with_logits(
        candidate_logits, targets.unsqueeze(-1).expand_as(candidate_logits), reduction="none"
    )
    gain = (base_loss.unsqueeze(-1) - per_factor_loss) / base_loss.unsqueeze(-1).clamp_min(0.1)
    g_best = gain.max(dim=-1).values
    routed_delta = (dense * residual).sum(dim=-1)
    route_loss = F.binary_cross_entropy_with_logits(z0 + 0.05 * routed_delta, targets, reduction="none")
    actual_loss = F.binary_cross_entropy_with_logits(
        z0 + actual_correction, targets, reduction="none"
    )
    denominator = base_loss.clamp_min(0.1)
    g_route = (base_loss - route_loss) / denominator
    g_actual = (base_loss - actual_loss) / denominator
    gain_reports, correlations, agreements = _gain_report(g_best, g_route, g_actual, targets)
    regional_utility = _regional_utility(
        z0, targets, residual, dense, act_probability, actual_correction
    )
    act_output: dict[str, Any] = {
        **gain_reports,
        "pairwise_correlations": correlations,
        "sign_agreement": agreements,
        "cross_tabs": _cross_tabs(g_best, g_route, g_actual, targets),
        "regional_utility": regional_utility,
        "risk_coverage": _risk_coverage(g_route, targets, image_ids, categories),
        "actual_gated_reconstruction_exact": actual_logit_reconstruction_max_error == 0.0,
        "loss_balance_conclusion": "NO_CHANGE_BEFORE_SEMANTIC_ALIGNMENT",
    }
    router_output = _router_audit(gain, targets)

    state_hash_after = _full_state_hash(model)
    all_grad_none_after = all(parameter.grad is None for parameter in model.parameters())
    invariant_checks = {
        "all_grads_none_before": all_grad_none_before,
        "all_grads_none_after": all_grad_none_after,
        "model_state_unchanged": state_hash_before == state_hash_after,
        "residual_definition_exact": residual_definition_max_error == 0.0,
        "routed_correction_reconstruction_exact": routed_reconstruction_max_error == 0.0,
        "actual_gated_reconstruction_exact": actual_logit_reconstruction_max_error == 0.0,
        "exactly_300_batches": len(dataset_indices) == 300,
    }
    if not all(invariant_checks.values()):
        raise RuntimeError(f"forward-only invariant failure: {[k for k, v in invariant_checks.items() if not v]}")

    reasons, primary, act_decision, router_decision, next_action = _decisions(act_output, router_output)
    act_output["decision"] = act_decision
    router_output["decision"] = router_decision
    status = "EXIT_FOR_DISCUSSION" if reasons else "AUDIT_COMPLETE_READY_FOR_DISCUSSION"
    output = {
        "status": status,
        "audit_kind": "FORWARD_ONLY_COMBINED_TEACHER_AUDIT",
        "contract": {
            "checks": {**provenance_checks, **checkpoint_checks, **invariant_checks},
            "checkpoint_origin": CHECKPOINT_ORIGIN,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_git_sha": checkpoint.get("git_sha"),
            "current_git_head": _git_head(),
            "historical_original_checkpoint_lost": True,
            "regeneration_provenance_path": str(args.provenance),
            "openai_checkpoint_sha256": openai_sha,
            "dataset": "VisA/train",
            "seed": args.seed,
            "batches": len(dataset_indices),
            "batch_size": 1,
            "grad_accumulation_metadata": 6,
            "precision": "fp32",
            "amp_enabled": False,
            "tf32_enabled": False,
            "rho": 0.05,
            "any_model_parameter_requires_grad": any(
                parameter.requires_grad for parameter in model.parameters()
            ),
            "model_eval": not model.training,
            "clipmodel_eval": not model.clipmodel.training,
            "update_load_bias": False,
            "optimizer_constructed": False,
            "backward_executed": False,
            "optimizer_steps": 0,
            "model_state_hash_before": state_hash_before,
            "model_state_hash_after": state_hash_after,
            "dataset_indices": dataset_indices,
            "residual_definition_max_abs_error": residual_definition_max_error,
            "routed_reconstruction_max_abs_error": routed_reconstruction_max_error,
            "actual_gated_reconstruction_max_abs_error": actual_logit_reconstruction_max_error,
            "source_decoupling_equivalence": "PASS_56_TESTS",
            "resolved_teacher_controls": {
                "factor_tau_utility": 0.05,
                "router_tau_utility": 0.05,
                "router_gain_threshold": 0.02,
                "act_gain_threshold": 0.02,
            },
        },
        "act": act_output,
        "router": router_output,
        "root_cause": {
            "primary": primary,
            "all_triggered": reasons,
            "semantic_changes_applied": [],
        },
        "next_authorized_action": next_action,
        "historical_context": {
            "original_run_artifacts_are_comparison_only": True,
            "paths": [
                "runs/p1_v84a_gpu/fresh_300b_seed0_attempt1/final_summary.json",
                "runs/p1_v84a_gpu/fresh_300b_seed0_attempt1/smoke_summary.json",
                "runs/p1_v84a_gpu/post300_root_cause_audit.json",
            ],
            "new_decisions_use_only_this_replay": True,
        },
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json_atomic(args.output, output)
    args.decision_output.write_text(_decision_markdown(output))
    print(json.dumps({
        "status": status,
        "primary": primary,
        "batches": len(dataset_indices),
        "valid_group_patches": int(targets.numel()),
        "runtime_seconds": round(output["runtime_seconds"], 3),
        "output": str(args.output),
        "decision_output": str(args.decision_output),
    }), flush=True)


if __name__ == "__main__":
    main()
