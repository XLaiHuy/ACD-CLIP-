"""Diagnostics-only aggregation for the bounded P1-v8.3 specialization probe."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from model.h6.utility_routing import utility_diagnostics


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().cpu()


def capture_utility_record(
    payload: dict[str, torch.Tensor],
    dense_probabilities: torch.Tensor,
    y_patch: torch.Tensor,
    utility_router_loss: torch.Tensor,
    *,
    act_probability: torch.Tensor | None = None,
    act_logits: torch.Tensor | None = None,
    act_payload: dict[str, torch.Tensor] | None = None,
    utility_act_loss: torch.Tensor | None = None,
    actual_gated_loss: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Detach the sufficient utility evidence for exact later aggregation."""
    keys = (
        "z0", "candidate_logits", "loss_base", "loss_per_factor", "gain_rel",
        "q_utility", "responsibility", "normalized_entropy", "best_gain_rel",
        "winner", "informative", "valid",
    )
    record = {key: payload[key].detach().cpu() for key in keys}
    # P1-v8.4-A carries the routed ACT teacher object in the utility payload.
    # Keep it in the diagnostics-only trajectory so each milestone can report
    # the actual routed gain without changing any training computation.
    for key in ("routed_delta", "routed_logits", "loss_routed", "routed_gain_rel"):
        if key in payload:
            record[key] = payload[key].detach().cpu()
    record["dense_probabilities"] = dense_probabilities.detach().cpu()
    record["y_patch"] = y_patch.detach().cpu()
    record["utility_router_loss"] = _cpu(utility_router_loss)
    if act_probability is not None or act_payload is not None:
        if act_probability is None or act_payload is None:
            raise ValueError("ACT probability and payload must be supplied together")
        record["act_probability"] = _cpu(act_probability)
        record["act_logits"] = _cpu(
            act_logits
            if act_logits is not None
            else torch.logit(act_probability.float().clamp(1e-6, 1.0 - 1e-6))
        )
        for key in ("target", "positive", "negative", "ambiguous", "support"):
            record[f"act_{key}"] = act_payload[key].detach().cpu()
        record["utility_act_loss"] = _cpu(
            utility_act_loss if utility_act_loss is not None else utility_router_loss * 0.0
        )
        if actual_gated_loss is not None:
            record["actual_gated_loss"] = _cpu(actual_gated_loss)
    return record


def _cat(records: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
    # All utility tensors use [G,B,P,...], while y_patch uses [B,P].
    dimension = 0 if key == "y_patch" else 1
    return torch.cat([record[key] for record in records], dim=dimension)


def _stats(values: torch.Tensor, quantiles: Iterable[float]) -> dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        result = {"mean": 0.0, "std": 0.0}
        result.update({f"p{int(q * 100):02d}": 0.0 for q in quantiles})
        return result
    result = {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }
    for quantile in quantiles:
        result[f"p{int(quantile * 100):02d}"] = float(torch.quantile(values, quantile).item())
    return result


def _python(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, dict):
        return {key: _python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python(item) for item in value]
    return value


def aggregate_utility_records(
    records: list[dict[str, torch.Tensor]],
    *,
    gain_threshold: float,
    entropy_threshold: float,
    rho: float = 0.05,
) -> dict[str, Any]:
    """Compute exact loss-space and gate diagnostics over a batch interval."""
    if not records:
        raise ValueError("at least one trajectory record is required")
    payload = {
        key: _cat(records, key)
        for key in (
            "z0", "candidate_logits", "loss_base", "loss_per_factor", "gain_rel",
            "q_utility", "responsibility", "normalized_entropy", "best_gain_rel",
            "winner", "informative", "valid",
        )
    }
    routed_loss_full = (
        _cat(records, "loss_routed").float()
        if all("loss_routed" in record for record in records) else None
    )
    actual_gated_loss_full = (
        _cat(records, "actual_gated_loss").float()
        if all("actual_gated_loss" in record for record in records) else None
    )
    dense = _cat(records, "dense_probabilities")
    y_patch = _cat(records, "y_patch")
    exact = utility_diagnostics(payload, dense, y_patch, rho=rho)
    valid = payload["valid"].bool()
    best_gain = payload["best_gain_rel"][valid]
    entropy = payload["normalized_entropy"][valid]
    max_probability = payload["q_utility"].max(dim=-1).values[valid]
    margins = payload["gain_rel"].topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)[valid]
    gain_pass = (payload["best_gain_rel"] > float(gain_threshold)) & valid
    entropy_pass = (payload["normalized_entropy"] < float(entropy_threshold)) & valid
    informative = payload["informative"].bool()
    valid_count = int(valid.sum().item())

    targets = y_patch.unsqueeze(0).expand_as(valid)
    breakdown: dict[str, dict[str, float | int]] = {}
    for name, region in (
        ("normal", valid & (targets < 0.5)),
        ("anomaly", valid & (targets >= 0.5)),
    ):
        count = int(region.sum().item())
        gain_values = payload["gain_rel"][region]
        best_values = payload["best_gain_rel"][region]
        entropy_values = payload["normalized_entropy"][region]
        all_harm = payload["gain_rel"].max(dim=-1).values <= 0
        breakdown[name] = {
            "valid_patch_count": count,
            "gain_rel_mean": float(gain_values.mean().item()) if count else 0.0,
            "best_gain_rel_mean": float(best_values.mean().item()) if count else 0.0,
            "teacher_entropy_mean": float(entropy_values.mean().item()) if count else 0.0,
            "informative_fraction": float(informative[region].float().mean().item()) if count else 0.0,
            "all_harm_fraction": float(all_harm[region].float().mean().item()) if count else 0.0,
        }

    supervised_count = int(informative.sum().item())
    result = _python(exact)
    result.update({
        "gain_threshold": float(gain_threshold),
        "entropy_threshold": float(entropy_threshold),
        "gain_threshold_pass_fraction": float(gain_pass.sum().item() / max(valid_count, 1)),
        "entropy_threshold_pass_fraction": float(entropy_pass.sum().item() / max(valid_count, 1)),
        "informative_fraction": float(supervised_count / max(valid_count, 1)),
        "best_gain_rel": _stats(best_gain, (0.50, 0.75, 0.90, 0.95, 0.99)),
        "best_second_utility_margin_distribution": _stats(margins, (0.50, 0.90, 0.95, 0.99)),
        "normalized_teacher_entropy": _stats(entropy, (0.01, 0.05, 0.10, 0.50, 0.90)),
        "teacher_max_probability_distribution": _stats(max_probability, (0.50, 0.90, 0.95, 0.99)),
        "normal_anomaly_breakdown": breakdown,
        "router_supervised_patch_count": supervised_count,
        "router_supervised_patch_fraction": float(supervised_count / max(valid_count, 1)),
        "valid_patch_count": valid_count,
        "utility_router_loss": float(torch.stack([record["utility_router_loss"] for record in records]).mean().item()),
        "router_utility_activity": (
            "active" if supervised_count else "inactive_due_to_teacher_gate"
        ),
    })
    routed_gain_full = None
    if all("routed_gain_rel" in record for record in records):
        routed_gain_full = _cat(records, "routed_gain_rel").float()
        routed_gain = routed_gain_full[valid]
        result["routed_gain_rel"] = _stats(
            routed_gain, (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
        )
        result["routed_gain_nonpositive_fraction"] = float(
            (routed_gain <= 0.0).float().mean().item()
        ) if routed_gain.numel() else 0.0
        result["routed_gain_positive_fraction"] = float(
            (routed_gain > 0.0).float().mean().item()
        ) if routed_gain.numel() else 0.0

    # Region-split loss-space comparisons are diagnostics only.  They use
    # the same tensors as the training operation order, so FullSoft is the
    # current routed correction and ActualGated is the actual ACT gate.
    targets = y_patch.unsqueeze(0).expand_as(valid)
    local = (payload["candidate_logits"] - payload["z0"].unsqueeze(-1)) / max(float(rho), 1e-12)
    uniform_logits = payload["z0"] + float(rho) * local.mean(dim=-1)
    hard_index = dense.float().argmax(dim=-1, keepdim=True)
    hard_logits = payload["z0"] + float(rho) * local.gather(-1, hard_index).squeeze(-1)
    uniform_loss_full = F.binary_cross_entropy_with_logits(
        uniform_logits, targets.float(), reduction="none"
    )
    hard_loss_full = F.binary_cross_entropy_with_logits(
        hard_logits, targets.float(), reduction="none"
    )

    def _region_mean(values: torch.Tensor | None, mask: torch.Tensor) -> float | None:
        if values is None or not bool(mask.any().item()):
            return None
        return float(values[mask].mean().item())

    def _region_utility(mask: torch.Tensor) -> dict[str, float | None]:
        factor_means = [
            _region_mean(payload["loss_per_factor"][..., factor], mask)
            for factor in range(payload["loss_per_factor"].shape[-1])
        ]
        factor_means_valid = [value for value in factor_means if value is not None]
        oracle_value = _region_mean(payload["loss_per_factor"].min(dim=-1).values, mask)
        return {
            "Base": _region_mean(payload["loss_base"], mask),
            "ResidualBestSingle": (
                min(factor_means_valid) if factor_means_valid else None
            ),
            "ResidualOracleMulti": oracle_value,
            "FullSoftRouted_ACT1": _region_mean(routed_loss_full, mask),
            "ActualGated": _region_mean(actual_gated_loss_full, mask),
            "HardRouted_ACT1": _region_mean(hard_loss_full, mask),
            "Uniform_ACT1": _region_mean(uniform_loss_full, mask),
        }

    result["utility_regions"] = {
        "overall": _region_utility(valid),
        "normal": _region_utility(valid & (targets < 0.5)),
        "anomaly": _region_utility(valid & (targets >= 0.5)),
    }
    if all("act_probability" in record for record in records):
        act_probability = _cat(records, "act_probability").float()
        act_logits = _cat(records, "act_logits").float() if all(
            "act_logits" in record for record in records
        ) else torch.logit(act_probability.clamp(1e-6, 1.0 - 1e-6))
        act_positive = _cat(records, "act_positive").bool()
        act_negative = _cat(records, "act_negative").bool()
        act_ambiguous = _cat(records, "act_ambiguous").bool()
        act_support = _cat(records, "act_support").bool()
        region_targets = y_patch.unsqueeze(0).expand_as(act_probability)

        def safe_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
            return float(values[mask].mean().item()) if mask.any() else 0.0

        def safe_fraction(mask: torch.Tensor, region: torch.Tensor) -> float:
            return float(mask[region].float().mean().item()) if region.any() else 0.0

        def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
            scores = scores.flatten().float()
            labels = labels.flatten().bool()
            positive_count = int(labels.sum().item())
            negative_count = int((~labels).sum().item())
            if not positive_count or not negative_count:
                return None
            order = scores.argsort()
            sorted_scores = scores[order]
            sorted_ranks = torch.arange(
                1, scores.numel() + 1, dtype=torch.float32
            )
            # Average tied ranks without a Python loop.  The previous
            # implementation scanned the full trajectory once per unique
            # probability, which became quadratic for 300B telemetry.
            _, inverse, counts = torch.unique_consecutive(
                sorted_scores, return_inverse=True, return_counts=True
            )
            ends = counts.cumsum(dim=0)
            starts = torch.cat((torch.zeros(1, dtype=torch.long), ends[:-1]))
            rank_prefix = torch.cat((torch.zeros(1), sorted_ranks.cumsum(dim=0)))
            average_ranks = (
                rank_prefix[ends] - rank_prefix[starts]
            ) / counts.float()
            ranks_sorted = average_ranks[inverse]
            ranks = torch.empty_like(ranks_sorted)
            ranks[order] = ranks_sorted
            rank_sum = ranks[labels].sum()
            value = (
                rank_sum - positive_count * (positive_count + 1) / 2.0
            ) / float(positive_count * negative_count)
            return float(value.item())

        def safe_correlation(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float | None:
            x_values = x[mask].flatten().float()
            y_values = y[mask].flatten().float()
            if x_values.numel() < 2:
                return None
            x_centered = x_values - x_values.mean()
            y_centered = y_values - y_values.mean()
            denominator = x_centered.norm() * y_centered.norm()
            if float(denominator.item()) <= 0.0:
                return None
            return float((x_centered * y_centered).sum().div(denominator).item())

        def cross_tab(region: torch.Tensor) -> dict[str, int | float]:
            region_valid = valid & region
            count = int(region_valid.sum().item())
            high = act_probability >= 0.5
            cells = {
                "teacher_on_act_high": int((act_positive & high & region_valid).sum().item()),
                "teacher_on_act_low": int((act_positive & ~high & region_valid).sum().item()),
                "teacher_off_act_high": int((act_negative & high & region_valid).sum().item()),
                "teacher_off_act_low": int((act_negative & ~high & region_valid).sum().item()),
            }
            cells["valid_count"] = count
            cells["act_high_fraction"] = float(
                (high & region_valid).float().sum().item() / max(count, 1)
            )
            return cells

        normal = valid & (region_targets < 0.5)
        anomaly = valid & (region_targets >= 0.5)
        on = valid & act_positive
        off = valid & act_negative
        supported_scores = act_probability[act_support]
        supported_labels = act_positive[act_support]
        result["act"] = {
            "probability_mean": safe_mean(act_probability, valid),
            "probability_normal_mean": safe_mean(act_probability, normal),
            "probability_anomaly_mean": safe_mean(act_probability, anomaly),
            "probability_stats": _stats(act_probability[valid], (0.01, 0.05, 0.50, 0.95, 0.99)),
            "probability_normal_stats": _stats(act_probability[normal], (0.05, 0.50, 0.95)),
            "probability_anomaly_stats": _stats(act_probability[anomaly], (0.05, 0.50, 0.95)),
            "logit_stats": _stats(act_logits[valid], (0.01, 0.05, 0.50, 0.95, 0.99)),
            "logit_normal_stats": _stats(act_logits[normal], (0.05, 0.50, 0.95)),
            "logit_anomaly_stats": _stats(act_logits[anomaly], (0.05, 0.50, 0.95)),
            "probability_on_mean": safe_mean(act_probability, on),
            "probability_off_mean": safe_mean(act_probability, off),
            "separation": safe_mean(act_probability, on) - safe_mean(act_probability, off),
            "probability_on_normal_mean": safe_mean(act_probability, on & normal),
            "probability_off_normal_mean": safe_mean(act_probability, off & normal),
            "separation_normal": (
                safe_mean(act_probability, on & normal)
                - safe_mean(act_probability, off & normal)
            ),
            "probability_on_anomaly_mean": safe_mean(act_probability, on & anomaly),
            "probability_off_anomaly_mean": safe_mean(act_probability, off & anomaly),
            "separation_anomaly": (
                safe_mean(act_probability, on & anomaly)
                - safe_mean(act_probability, off & anomaly)
            ),
            "target_positive_fraction": safe_fraction(act_positive, valid),
            "target_negative_fraction": safe_fraction(act_negative, valid),
            "target_ambiguous_fraction": safe_fraction(act_ambiguous, valid),
            "target_positive_normal_fraction": safe_fraction(act_positive, normal),
            "target_negative_normal_fraction": safe_fraction(act_negative, normal),
            "target_ambiguous_normal_fraction": safe_fraction(act_ambiguous, normal),
            "target_positive_anomaly_fraction": safe_fraction(act_positive, anomaly),
            "target_negative_anomaly_fraction": safe_fraction(act_negative, anomaly),
            "target_ambiguous_anomaly_fraction": safe_fraction(act_ambiguous, anomaly),
            "teacher_auroc": binary_auroc(supported_scores, supported_labels),
            "teacher_act_cross_tab": {
                "overall": cross_tab(torch.ones_like(valid, dtype=torch.bool)),
                "normal": cross_tab(region_targets < 0.5),
                "anomaly": cross_tab(region_targets >= 0.5),
            },
            "g_route_act_probability_correlation": {
                "overall": safe_correlation(
                    routed_gain_full, act_probability, valid
                ) if routed_gain_full is not None else None,
                "normal": safe_correlation(
                    routed_gain_full, act_probability, normal
                ) if routed_gain_full is not None else None,
                "anomaly": safe_correlation(
                    routed_gain_full, act_probability, anomaly
                ) if routed_gain_full is not None else None,
            },
            "utility_act_loss": float(torch.stack([
                record["utility_act_loss"] for record in records
            ]).mean().item()),
        }
    return result


def teacher_sensitivity_grid(
    records: list[dict[str, torch.Tensor]],
    *,
    gain_threshold: float,
    taus: Iterable[float] = (0.05, 0.03, 0.02),
    entropy_thresholds: Iterable[float] = (0.98, 0.99, 0.995),
) -> list[dict[str, Any]]:
    """Evaluate alternate teacher calibration without gradients or optimizer steps."""
    gain_rel = _cat(records, "gain_rel").float()
    valid = _cat(records, "valid").bool()
    best_gain, winners = gain_rel.max(dim=-1)
    rows = []
    for tau in taus:
        q = F.softmax(gain_rel / float(tau), dim=-1)
        entropy = -(q * q.clamp_min(1e-12).log()).sum(dim=-1) / torch.log(
            torch.tensor(float(q.shape[-1]))
        )
        max_probability = q.max(dim=-1).values
        for threshold in entropy_thresholds:
            informative = valid & (best_gain > float(gain_threshold)) & (entropy < float(threshold))
            winner_shares = [
                float(((winners == factor) & valid).sum().item() / max(int(valid.sum().item()), 1))
                for factor in range(q.shape[-1])
            ]
            rows.append({
                "tau_utility": float(tau),
                "entropy_threshold": float(threshold),
                "teacher_entropy": float(entropy[valid].mean().item()),
                "teacher_max_probability": float(max_probability[valid].mean().item()),
                "informative_fraction": float(informative.sum().item() / max(int(valid.sum().item()), 1)),
                "winner_shares": winner_shares,
            })
    return rows


def write_trajectory_artifacts(
    output_dir: str | Path,
    milestones: list[dict[str, Any]],
    final_summary: dict[str, Any] | None = None,
) -> None:
    """Atomically persist JSON plus a compact, analysis-friendly CSV."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectory.json"
    temporary = trajectory_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"milestones": milestones}, indent=2, sort_keys=True) + "\n")
    temporary.replace(trajectory_path)

    scalar_keys = (
        "G_local", "G_multi", "Base", "BestSingle", "OracleMulti", "Uniform",
        "SoftRouted", "HardRouted", "capture", "gain_threshold_pass_fraction",
        "entropy_threshold_pass_fraction", "informative_fraction", "teacher_entropy",
        "teacher_max_probability", "best_second_utility_margin", "all_harm_fraction",
    )
    rows = []
    for milestone in milestones:
        row = {"batch": milestone["batch"], "optimizer_steps": milestone["optimizer_steps"]}
        for interval in ("cumulative", "recent_window"):
            evidence = milestone[interval]
            for key in scalar_keys:
                row[f"{interval}_{key}"] = evidence.get(key)
        structure = milestone["structure"]
        row.update({
            "factor_effective_rank": structure["factor_embedding_effective_rank"],
            "factor_embedding_cos_mean": structure["factor_embedding_pairwise_cosine_mean"],
            "factor_patch_corr_mean": structure["factor_patch_pairwise_correlation_mean"],
        })
        rows.append(row)
    csv_path = output / "trajectory.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if final_summary is not None:
        summary_path = output / "final_summary.json"
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(final_summary, indent=2, sort_keys=True) + "\n")
        temporary.replace(summary_path)
