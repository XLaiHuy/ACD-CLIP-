"""P1-v8.3 utility teacher, dense router losses, and exact diagnostics."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def router_target_distribution(
    gain_rel: torch.Tensor,
    *,
    tau_utility: float,
    mode: str = "legacy_raw_softmax",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached Router target q and its per-patch zero-spread mask.

    ``patch_zscore_softmax`` standardizes only within a patch.  It therefore
    changes target scale, never gain ordering or Router eligibility.
    """
    if mode == "legacy_raw_softmax":
        return F.softmax(gain_rel.detach() / float(tau_utility), dim=-1), torch.zeros(
            gain_rel.shape[:-1], dtype=torch.bool, device=gain_rel.device
        )
    if mode != "patch_zscore_softmax":
        raise ValueError("router_target_mode must be 'legacy_raw_softmax' or 'patch_zscore_softmax'")
    detached = gain_rel.detach()
    centered = detached - detached.mean(dim=-1, keepdim=True)
    sigma = centered.square().mean(dim=-1, keepdim=True).sqrt()
    zero_spread = sigma.squeeze(-1) <= 1e-12
    return F.softmax(centered / sigma.clamp_min(1e-12), dim=-1), zero_spread


def build_patch_targets(
    masks: torch.Tensor,
    patch_count: int,
    local_mask_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Area-pool masks and validity to the square CLIP patch grid."""
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    if masks.ndim != 4:
        raise ValueError("masks must be [B,H,W] or [B,1,H,W]")
    grid = int(math.isqrt(int(patch_count)))
    if grid * grid != int(patch_count):
        raise ValueError("patch_count must be a perfect square")
    targets = F.adaptive_avg_pool2d(masks.float(), (grid, grid)).flatten(1).clamp(0.0, 1.0)
    if local_mask_valid is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        if local_mask_valid.ndim == 3:
            local_mask_valid = local_mask_valid.unsqueeze(1)
        if local_mask_valid.shape != masks.shape:
            raise ValueError("local_mask_valid must have the same shape as masks")
        valid_fraction = F.adaptive_avg_pool2d(
            local_mask_valid.float(), (grid, grid)
        ).flatten(1)
        valid = valid_fraction >= 1.0 - 1e-6
    return targets, valid


def exploration_epsilon(
    epoch_one_based: int,
    total_epochs: int,
    start: float = 0.15,
    end: float = 0.05,
) -> float:
    if total_epochs <= 1:
        return float(end)
    progress = min(1.0, max(0.0, (int(epoch_one_based) - 1) / (int(total_epochs) - 1)))
    return float(start + progress * (end - start))


def utility_teacher(
    base_logits: torch.Tensor,
    factor_local_evidence: torch.Tensor,
    y_patch: torch.Tensor,
    local_mask_valid: torch.Tensor,
    *,
    rho: float = 0.05,
    denominator_floor: float = 0.1,
    tau_utility: float = 0.05,
    factor_tau_utility: float | None = None,
    router_tau_utility: float | None = None,
    epsilon: float = 0.15,
    gain_threshold: float = 0.02,
    router_gain_threshold: float | None = None,
    entropy_threshold: float = 0.98,
    router_confidence_mode: str = "entropy",
    router_margin_rel_threshold: float = 0.10,
    router_target_mode: str = "legacy_raw_softmax",
    routed_probabilities: torch.Tensor | None = None,
    role_topology: str = "flat",
    role_teacher_scale: float | None = None,
) -> Dict[str, torch.Tensor]:
    """Build detached factor and router utility teachers.

    The legacy ``tau_utility`` and ``gain_threshold`` remain the defaults for
    both consumers.  Supplying a decoupled control changes only its named
    consumer, preserving existing P1-v8.3/v8.4-A behavior when unset.
    """
    if factor_local_evidence.ndim != 4:
        raise ValueError("factor_local_evidence must be [G,B,P,M]")
    groups, batch, patches, factors = factor_local_evidence.shape
    if role_topology not in {"flat", "r2_normal_anomaly"}:
        raise ValueError("role_topology must be 'flat' or 'r2_normal_anomaly'")
    if role_topology == "r2_normal_anomaly":
        if factors != 2:
            raise ValueError("r2_normal_anomaly requires exactly two role outputs")
        if role_teacher_scale is None or float(role_teacher_scale) <= 0.0:
            raise ValueError("r2_normal_anomaly requires a positive global role_teacher_scale")
    if base_logits.shape != (groups, batch, patches):
        raise ValueError("base_logits must be [G,B,P]")
    if y_patch.shape != (batch, patches) or local_mask_valid.shape != (batch, patches):
        raise ValueError("patch targets/validity must be [B,P]")
    factor_tau = float(tau_utility if factor_tau_utility is None else factor_tau_utility)
    router_tau = float(tau_utility if router_tau_utility is None else router_tau_utility)
    router_gain = float(
        gain_threshold if router_gain_threshold is None else router_gain_threshold
    )
    if factor_tau <= 0 or router_tau <= 0 or denominator_floor <= 0:
        raise ValueError("factor/router utility temperatures and denominator_floor must be positive")
    if router_confidence_mode not in {"entropy", "margin_rel"}:
        raise ValueError("router_confidence_mode must be 'entropy' or 'margin_rel'")
    if float(router_margin_rel_threshold) < 0.0:
        raise ValueError("router_margin_rel_threshold must be non-negative")
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("epsilon must be in [0,1]")
    if float(rho) not in (0.0, 0.05):
        raise ValueError("rho may only be canonical 0.05 or diagnostic 0")
    if (
        routed_probabilities is not None
        and routed_probabilities.shape != factor_local_evidence.shape
    ):
        raise ValueError("routed_probabilities must match factor_local_evidence [G,B,P,M]")

    targets = y_patch.unsqueeze(0).expand(groups, -1, -1).float()
    z0 = base_logits.detach().float()
    local = factor_local_evidence.float()
    loss_base = F.binary_cross_entropy_with_logits(z0, targets, reduction="none")
    candidate_logits = z0.unsqueeze(-1) + float(rho) * local
    loss_per_factor = F.binary_cross_entropy_with_logits(
        candidate_logits, targets.unsqueeze(-1).expand_as(candidate_logits), reduction="none"
    )
    gain_rel = (loss_base.unsqueeze(-1) - loss_per_factor) / loss_base.unsqueeze(-1).clamp_min(
        float(denominator_floor)
    )
    role_gap = None
    role_probability = None
    if role_topology == "r2_normal_anomaly":
        # Role 0 is normal/background and role 1 is anomaly. The scale is a
        # frozen, global TRAIN statistic; no per-patch normalization is used.
        role_gap = gain_rel[..., 0] - gain_rel[..., 1]
        role_probability = torch.sigmoid(role_gap / float(role_teacher_scale)).detach()
        q_factor = torch.stack((role_probability, 1.0 - role_probability), dim=-1)
        q_router = q_factor
        router_target_zero_spread = torch.zeros(
            gain_rel.shape[:-1], dtype=torch.bool, device=gain_rel.device
        )
        responsibility = q_factor.detach()
    else:
        q_factor = F.softmax(gain_rel.detach() / factor_tau, dim=-1)
        q_router, router_target_zero_spread = router_target_distribution(
            gain_rel, tau_utility=router_tau, mode=router_target_mode
        )
        responsibility = (
            (1.0 - float(epsilon)) * q_factor + float(epsilon) / factors
        ).detach()
    normalized_entropy = -(
        q_router * q_router.clamp_min(1e-12).log()
    ).sum(dim=-1) / math.log(factors)
    detached_gains = gain_rel.detach()
    best_gain_rel, winners = detached_gains.max(dim=-1)
    # Excluding the selected winner, rather than relying on topk's tie order,
    # preserves the legacy argmax winner while making a tied best gain yield a
    # zero margin as required by the margin-eligibility candidate.
    winner_one_hot = F.one_hot(winners, num_classes=factors).bool()
    second_gain_rel = detached_gains.masked_fill(winner_one_hot, float("-inf")).max(dim=-1).values
    margin_abs = best_gain_rel - second_gain_rel
    margin_rel = margin_abs / best_gain_rel.abs().clamp_min(1e-12)
    valid = local_mask_valid.unsqueeze(0).expand(groups, -1, -1)
    if router_confidence_mode == "entropy":
        # Exact legacy formulation: gain threshold and q_router entropy gate.
        informative = (
            (best_gain_rel > router_gain)
            & (normalized_entropy < float(entropy_threshold))
            & valid
        )
    else:
        # Margin mode only changes Router eligibility.  It keeps factor q,
        # responsibility, Router q softness, ACT, and residual semantics intact.
        informative = (
            (best_gain_rel > 0.0)
            & (margin_rel > float(router_margin_rel_threshold))
            & valid
        )
    payload = {
        "z0": z0,
        "candidate_logits": candidate_logits,
        "loss_base": loss_base,
        "loss_per_factor": loss_per_factor,
        "gain_rel": gain_rel,
        "q_factor_utility": q_factor.detach(),
        "q_router_utility": q_router.detach(),
        "router_target_zero_spread": router_target_zero_spread.detach(),
        # Backward-compatible public alias used by router losses/diagnostics.
        "q_utility": q_router.detach(),
        "responsibility": responsibility,
        "normalized_entropy": normalized_entropy.detach(),
        "best_gain_rel": best_gain_rel,
        "second_gain_rel": second_gain_rel.detach(),
        "margin_abs": margin_abs.detach(),
        "margin_rel": margin_rel.detach(),
        "winner": winners.detach(),
        "informative": informative.detach(),
        "valid": valid,
        "role_topology": role_topology,
    }
    if role_topology == "r2_normal_anomaly":
        role_entropy = -(
            q_router * q_router.clamp_min(1e-12).log()
        ).sum(dim=-1)
        payload.update({
            "role_gap": role_gap.detach(),
            "role_scale": torch.full_like(role_gap, float(role_teacher_scale)).detach(),
            "role_probability": role_probability.detach(),
            "role_entropy": role_entropy.detach(),
        })
    if routed_probabilities is not None:
        # The ACT target is a teacher: detach both the current forward mixture
        # and factor evidence so labels cannot create a gradient path through
        # either Router or factors.  Factor responsibility and Router q above
        # remain functions of factor-level gains only.
        routed_delta = (
            routed_probabilities.detach().float() * local.detach()
        ).sum(dim=-1)
        routed_logits = z0 + float(rho) * routed_delta
        loss_routed = F.binary_cross_entropy_with_logits(
            routed_logits, targets, reduction="none"
        )
        routed_gain_rel = (
            (loss_base.detach() - loss_routed)
            / loss_base.detach().clamp_min(float(denominator_floor))
        )
        payload.update({
            "routed_delta": routed_delta.detach(),
            "routed_logits": routed_logits.detach(),
            "loss_routed": loss_routed.detach(),
            "routed_gain_rel": routed_gain_rel.detach(),
        })
    return payload


def routed_residual_correction(
    act_probability: torch.Tensor,
    factor_probabilities: torch.Tensor,
    factor_residual_logits: torch.Tensor,
) -> torch.Tensor:
    """Apply continuous ACT to a dense mixture of true residual factors.

    The multiplication is deliberately explicit: supplying an ACT probability
    of exactly zero yields an exactly-zero correction tensor, so the caller's
    base logits are preserved bit-for-bit by the local branch.
    """
    if factor_probabilities.shape != factor_residual_logits.shape:
        raise ValueError(
            "factor probabilities/residual logits must have identical shape; "
            f"got {tuple(factor_probabilities.shape)} vs {tuple(factor_residual_logits.shape)}"
        )
    if act_probability.shape != factor_residual_logits.shape[:-1]:
        raise ValueError("act_probability must match [G,B,P]")
    return act_probability * (
        factor_probabilities.float() * factor_residual_logits.float()
    ).sum(dim=-1)


def act_teacher(
    utility_payload: Dict[str, torch.Tensor],
    *,
    gain_threshold: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """Detached three-zone ACT teacher from routed residual utility.

    The utility object is the ACT=1 correction made by the current routed
    mixture, exactly matching the action ACT gates. Positive support is
    strictly above ``gain_threshold``; non-positive gain is negative support;
    the open interval in between is intentionally left ambiguous and receives
    no ACT loss.
    """
    if float(gain_threshold) < 0.0:
        raise ValueError("gain_threshold must be non-negative")
    if "routed_gain_rel" not in utility_payload:
        raise ValueError(
            "ACT teacher requires routed_gain_rel from the current routed mixture"
        )
    routed_gain = utility_payload["routed_gain_rel"].detach().float()
    valid = utility_payload["valid"].detach().bool()
    positive = valid & (routed_gain > float(gain_threshold))
    negative = valid & (routed_gain <= 0.0)
    ambiguous = valid & (routed_gain > 0.0) & (routed_gain <= float(gain_threshold))
    support = positive | negative
    target = positive.to(routed_gain.dtype)
    return {
        "target": target.detach(),
        "positive": positive.detach(),
        "negative": negative.detach(),
        "ambiguous": ambiguous.detach(),
        "support": support.detach(),
        "valid": valid.detach(),
        "routed_residual_gain": routed_gain.detach(),
    }


def effective_number_act_loss(
    act_logits: torch.Tensor,
    teacher: Dict[str, torch.Tensor],
    y_patch: torch.Tensor,
    *,
    beta: float = 0.999,
) -> torch.Tensor:
    """Region-balanced ACT BCE over positive and negative teacher support.

    Normal/anomaly regions receive inverse effective-number patch weights and
    are combined in one normalized weighted mean. Ambiguous patches have zero
    weight, avoiding the old hard 50/50 mean-of-region-means construction.
    """
    if not 0.0 < float(beta) < 1.0:
        raise ValueError("beta must be in (0, 1)")
    if act_logits.shape != teacher["target"].shape:
        raise ValueError("act_logits and ACT target must have identical [G,B,P] shape")
    if y_patch.shape != act_logits.shape[1:]:
        raise ValueError("y_patch must be [B,P]")
    per_patch = F.binary_cross_entropy_with_logits(
        act_logits.float(), teacher["target"].float(), reduction="none"
    )
    targets = y_patch.unsqueeze(0).expand_as(act_logits)
    support = teacher["support"].bool()
    weights = torch.zeros_like(per_patch)
    for region in (support & (targets < 0.5), support & (targets >= 0.5)):
        count = int(region.sum().item())
        if count:
            effective_n = -math.expm1(count * math.log(float(beta))) / (1.0 - float(beta))
            weights[region] = 1.0 / effective_n
    denominator = weights.sum()
    return (
        (per_patch * weights).sum() / denominator
        if float(denominator.detach().item()) > 0.0
        else per_patch.sum() * 0.0
    )


def act_diagnostics(
    act_probability: torch.Tensor,
    teacher: Dict[str, torch.Tensor],
    y_patch: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Support and probability diagnostics split by physical patch region."""
    if act_probability.shape != teacher["target"].shape:
        raise ValueError("ACT probability and teacher tensors must have identical shape")
    targets = y_patch.unsqueeze(0).expand_as(act_probability)
    valid = teacher["valid"].bool()
    zero = act_probability.float().sum() * 0.0

    def fraction(mask: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
        return mask[region].float().mean() if region.any() else zero

    def mean_probability(region: torch.Tensor) -> torch.Tensor:
        return act_probability.float()[region].mean() if region.any() else zero

    normal = valid & (targets < 0.5)
    anomaly = valid & (targets >= 0.5)
    return {
        "act_probability_mean": mean_probability(valid),
        "act_probability_normal_mean": mean_probability(normal),
        "act_probability_anomaly_mean": mean_probability(anomaly),
        "act_target_positive_fraction": fraction(teacher["positive"], valid),
        "act_target_negative_fraction": fraction(teacher["negative"], valid),
        "act_target_ambiguous_fraction": fraction(teacher["ambiguous"], valid),
        "act_target_positive_normal_fraction": fraction(teacher["positive"], normal),
        "act_target_negative_normal_fraction": fraction(teacher["negative"], normal),
        "act_target_ambiguous_normal_fraction": fraction(teacher["ambiguous"], normal),
        "act_target_positive_anomaly_fraction": fraction(teacher["positive"], anomaly),
        "act_target_negative_anomaly_fraction": fraction(teacher["negative"], anomaly),
        "act_target_ambiguous_anomaly_fraction": fraction(teacher["ambiguous"], anomaly),
        "act_support_fraction": fraction(teacher["support"], valid),
    }


def _balanced_binary_mean(values: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    zero = values.sum() * 0.0
    normal = valid & (targets < 0.5)
    anomaly = valid & (targets >= 0.5)
    pieces = []
    if normal.any():
        pieces.append(values[normal].mean())
    if anomaly.any():
        pieces.append(values[anomaly].mean())
    return torch.stack(pieces).mean() if pieces else zero


def utility_factor_loss(payload: Dict[str, torch.Tensor], y_patch: torch.Tensor) -> torch.Tensor:
    """Responsibility-weight factor candidates; balance normal/anomaly evidence only."""
    per_patch = (payload["responsibility"].detach() * payload["loss_per_factor"]).sum(dim=-1)
    targets = y_patch.unsqueeze(0).expand_as(per_patch)
    return _balanced_binary_mean(per_patch, targets, payload["valid"])


def effective_number_utility_factor_loss(
    payload: Dict[str, torch.Tensor],
    y_patch: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Patch-weighted factor BCE using inverse effective region counts.

    This is the class-balanced effective-number construction applied to patch
    classes, followed by one normalized weighted mean over all valid patches.
    It deliberately does not reweight already-reduced region means.
    """
    if not 0.0 < float(beta) < 1.0:
        raise ValueError("beta must be in (0, 1)")
    per_patch = (
        payload["responsibility"].detach() * payload["loss_per_factor"]
    ).sum(dim=-1)
    targets = y_patch.unsqueeze(0).expand_as(per_patch)
    valid = payload["valid"]
    weights = torch.zeros_like(per_patch)
    for region in (valid & (targets < 0.5), valid & (targets >= 0.5)):
        count = int(region.sum().item())
        if count:
            effective_n = -math.expm1(count * math.log(float(beta))) / (1.0 - float(beta))
            weights[region] = 1.0 / effective_n
    denominator = weights.sum()
    return (
        (per_patch * weights).sum() / denominator
        if float(denominator.detach().item()) > 0.0
        else per_patch.sum() * 0.0
    )


def utility_router_loss(
    dense_probabilities: torch.Tensor,
    payload: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Dense router cross-entropy against detached utility teacher on informative patches."""
    if dense_probabilities.shape != payload["q_utility"].shape:
        raise ValueError("dense_probabilities and q_utility must have identical [G,B,P,M] shape")
    teacher = payload["q_utility"].detach()
    ce = -(teacher * dense_probabilities.float().clamp_min(1e-12).log()).sum(dim=-1)
    informative = payload["informative"]
    return ce[informative].mean() if informative.any() else ce.sum() * 0.0


def support_normalized_utility_router_loss(
    dense_probabilities: torch.Tensor,
    payload: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Masked utility CE divided by all valid support, including masked zeros."""
    if dense_probabilities.shape != payload["q_utility"].shape:
        raise ValueError("dense_probabilities and q_utility must have identical [G,B,P,M] shape")
    teacher = payload["q_utility"].detach()
    ce = -(teacher * dense_probabilities.float().clamp_min(1e-12).log()).sum(dim=-1)
    valid_count = int(payload["valid"].sum().item())
    if not valid_count:
        return ce.sum() * 0.0
    return (ce * payload["informative"].float()).sum() / float(valid_count)


def r2_responsibility_balanced_utility_router_loss(
    dense_probabilities: torch.Tensor,
    payload: Dict[str, torch.Tensor],
    *,
    role_weights: tuple[float, float],
) -> torch.Tensor:
    """Fixed R2 teacher CE balanced by TRAIN responsibility mass.

    The weights are frozen before training from TRAIN-only informative teacher
    mass. Unlike the legacy support-normalized form, this denominator is the
    weighted informative support itself, so inactive patches do not attenuate
    the Router gradient.
    """
    if payload.get("role_topology") != "r2_normal_anomaly":
        raise ValueError("R2 responsibility balancing requires r2_normal_anomaly")
    if dense_probabilities.shape != payload["q_utility"].shape or dense_probabilities.shape[-1] != 2:
        raise ValueError("R2 Router probabilities/teacher must match [G,B,P,2]")
    weights = torch.as_tensor(
        role_weights, dtype=dense_probabilities.dtype, device=dense_probabilities.device,
    )
    if weights.shape != (2,) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("R2 Router role_weights must be two finite positive values")
    teacher = payload["q_utility"].detach()
    per_role_ce = -teacher * dense_probabilities.float().clamp_min(1e-12).log()
    informative = payload["informative"].float().unsqueeze(-1)
    weighted_target = informative * teacher * weights.view(1, 1, 1, 2)
    denominator = weighted_target.sum()
    if float(denominator.detach().item()) == 0.0:
        return dense_probabilities.sum() * 0.0
    return (informative * per_role_ce * weights.view(1, 1, 1, 2)).sum() / denominator


def r2_region_normalized_utility_router_loss(
    dense_probabilities: torch.Tensor,
    payload: Dict[str, torch.Tensor],
    patch_targets: torch.Tensor,
    *,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """R2 Router CE normalized independently on normal and anomaly support."""
    if payload.get("role_topology") != "r2_normal_anomaly":
        raise ValueError("R2 region normalization requires r2_normal_anomaly")
    if dense_probabilities.shape != payload["q_utility"].shape or dense_probabilities.shape[-1] != 2:
        raise ValueError("R2 Router probabilities/teacher must match [G,B,P,2]")
    if patch_targets.shape != dense_probabilities.shape[1:3]:
        raise ValueError("patch_targets must match Router [B,P] support")
    if "role_gap" not in payload or "role_scale" not in payload:
        raise ValueError("R2 region normalization requires frozen R2 role gap and scale")

    teacher = payload["q_utility"].detach()
    ce = -(teacher * dense_probabilities.float().clamp_min(1e-12).log()).sum(dim=-1)
    targets = patch_targets.detach().to(device=ce.device).unsqueeze(0).expand_as(ce)
    support = payload["informative"].bool() & payload["valid"].bool()
    normal_support = support & (targets < 0.5)
    anomaly_support = support & (targets >= 0.5)
    gap = payload["role_gap"].detach().abs().to(dtype=ce.dtype)
    scale = payload["role_scale"].detach().to(dtype=ce.dtype)
    if not torch.isfinite(gap).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("R2 region normalization requires finite positive frozen role scale")
    utility_weight = 1.0 + gap / (gap + scale)
    if not torch.isfinite(utility_weight).all() or (utility_weight < 1.0).any() or (utility_weight > 2.0).any():
        raise ValueError("R2 region utility weights must be finite and bounded in [1, 2]")

    def _region_mean(region: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weighted_support = utility_weight * region.to(dtype=ce.dtype)
        denominator = weighted_support.sum()
        numerator = (weighted_support * ce).sum()
        loss = numerator / denominator if float(denominator.detach().item()) > 0.0 else ce.sum() * 0.0
        return loss, numerator, denominator

    normal_loss, normal_numerator, normal_denominator = _region_mean(normal_support)
    anomaly_loss, anomaly_numerator, anomaly_denominator = _region_mean(anomaly_support)
    normal_present = normal_support.any()
    anomaly_present = anomaly_support.any()
    if normal_present and anomaly_present:
        loss = 0.5 * (normal_loss + anomaly_loss)
        normal_contribution = 0.5 * normal_loss
        anomaly_contribution = 0.5 * anomaly_loss
    elif normal_present:
        loss, normal_contribution, anomaly_contribution = normal_loss, normal_loss, anomaly_loss
    elif anomaly_present:
        loss, normal_contribution, anomaly_contribution = anomaly_loss, normal_loss, anomaly_loss
    else:
        loss = ce.sum() * 0.0
        normal_contribution = loss
        anomaly_contribution = loss
    if not return_components:
        return loss
    return loss, {
        "normal_support_count": normal_support.sum().detach(),
        "anomaly_support_count": anomaly_support.sum().detach(),
        "normal_present": normal_present.detach(),
        "anomaly_present": anomaly_present.detach(),
        "utility_weight_min": utility_weight.min().detach(),
        "utility_weight_max": utility_weight.max().detach(),
        "normal_numerator": normal_numerator.detach(),
        "anomaly_numerator": anomaly_numerator.detach(),
        "normal_denominator": normal_denominator.detach(),
        "anomaly_denominator": anomaly_denominator.detach(),
        "normal_loss": normal_loss,
        "anomaly_loss": anomaly_loss,
        "normal_contribution": normal_contribution,
        "anomaly_contribution": anomaly_contribution,
    }


def r2_region_grounded_router_loss(dense_probabilities: torch.Tensor, patch_targets: torch.Tensor, valid_patch: torch.Tensor, *, return_components: bool = False):
    """Immutable TRAIN-region CE for R2 roles, independent of model utility."""
    if dense_probabilities.ndim != 4 or dense_probabilities.shape[-1] != 2:
        raise ValueError("R2 grounded Router probabilities must be [G,B,P,2]")
    if patch_targets.shape != dense_probabilities.shape[1:3] or valid_patch.shape != patch_targets.shape:
        raise ValueError("grounded Router target/valid support must be [B,P]")
    targets = patch_targets.detach().to(device=dense_probabilities.device)
    valid = valid_patch.detach().bool().to(device=dense_probabilities.device)
    target = targets.unsqueeze(0).expand(dense_probabilities.shape[:-1])
    ce = -dense_probabilities.float().clamp_min(1e-12).log()
    normal = valid.unsqueeze(0).expand_as(target) & (target < 0.5)
    anomaly = valid.unsqueeze(0).expand_as(target) & (target >= 0.5)
    normal_loss = ce[..., 0][normal].mean() if bool(normal.any()) else ce.sum() * 0.0
    anomaly_loss = ce[..., 1][anomaly].mean() if bool(anomaly.any()) else ce.sum() * 0.0
    normal_present, anomaly_present = normal.any(), anomaly.any()
    if normal_present and anomaly_present:
        loss = 0.5 * (normal_loss + anomaly_loss)
        normal_contribution, anomaly_contribution = 0.5 * normal_loss, 0.5 * anomaly_loss
    elif normal_present:
        loss, normal_contribution, anomaly_contribution = normal_loss, normal_loss, anomaly_loss
    elif anomaly_present:
        loss, normal_contribution, anomaly_contribution = anomaly_loss, normal_loss, anomaly_loss
    else:
        loss = ce.sum() * 0.0
        normal_contribution = anomaly_contribution = loss
    if not return_components:
        return loss
    return loss, {
        "normal_support_count": normal.sum().detach(), "anomaly_support_count": anomaly.sum().detach(),
        "normal_present": normal_present.detach(), "anomaly_present": anomaly_present.detach(),
        "normal_loss": normal_loss, "anomaly_loss": anomaly_loss,
        "normal_contribution": normal_contribution, "anomaly_contribution": anomaly_contribution,
    }



def r2_intrinsic_responsibility_loss(logits: torch.Tensor, patch_targets: torch.Tensor, valid_patch: torch.Tensor, *, return_components: bool = False):
    """Balanced immutable-region CE on intrinsic factor compatibility logits."""
    if logits.ndim != 4 or logits.shape[-1] != 2:
        raise ValueError("intrinsic responsibility logits must be [G,B,P,2]")
    target = patch_targets.detach().to(logits.device).unsqueeze(0).expand(logits.shape[:-1])
    valid = valid_patch.detach().bool().to(logits.device).unsqueeze(0).expand_as(target)
    normal, anomaly = valid & (target < .5), valid & (target >= .5)
    ce = F.cross_entropy(logits.float().movedim(-1, 1), target.long(), reduction="none")
    normal_loss = ce[normal].mean() if normal.any() else ce.sum() * 0.0
    anomaly_loss = ce[anomaly].mean() if anomaly.any() else ce.sum() * 0.0
    if normal.any() and anomaly.any():
        loss, nc, ac = .5 * (normal_loss + anomaly_loss), .5 * normal_loss, .5 * anomaly_loss
    elif normal.any():
        loss, nc, ac = normal_loss, normal_loss, anomaly_loss
    elif anomaly.any():
        loss, nc, ac = anomaly_loss, normal_loss, anomaly_loss
    else:
        loss = ce.sum() * 0.0; nc = ac = loss
    if not return_components:
        return loss
    return loss, {"normal_support_count": normal.sum().detach(), "anomaly_support_count": anomaly.sum().detach(),
                  "normal_present": normal.any().detach(), "anomaly_present": anomaly.any().detach(),
                  "normal_loss": normal_loss, "anomaly_loss": anomaly_loss,
                  "normal_contribution": nc, "anomaly_contribution": ac}

def utility_diagnostics(
    payload: Dict[str, torch.Tensor],
    dense_probabilities: torch.Tensor,
    y_patch: torch.Tensor,
    *,
    rho: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Exact loss-space Base/Best/Oracle/routing comparisons and router diagnostics."""
    valid = payload["valid"]
    base = payload["loss_base"]
    per_factor = payload["loss_per_factor"]
    local = (payload["candidate_logits"] - payload["z0"].unsqueeze(-1)) / max(float(rho), 1e-12)
    targets = y_patch.unsqueeze(0).expand_as(base)

    def valid_mean(value: torch.Tensor) -> torch.Tensor:
        return value[valid].mean() if valid.any() else value.sum() * 0.0

    factor_valid = valid.unsqueeze(-1).expand_as(per_factor)
    factor_means = torch.stack([
        per_factor[..., factor][valid].mean() if valid.any() else per_factor[..., factor].sum() * 0.0
        for factor in range(per_factor.shape[-1])
    ])
    best_single = factor_means.min()
    oracle = valid_mean(per_factor.min(dim=-1).values)
    uniform_logits = payload["z0"] + float(rho) * local.mean(dim=-1)
    soft_logits = payload["z0"] + float(rho) * (dense_probabilities.float() * local).sum(dim=-1)
    hard_index = dense_probabilities.argmax(dim=-1, keepdim=True)
    hard_local = local.gather(-1, hard_index).squeeze(-1)
    hard_logits = payload["z0"] + float(rho) * hard_local
    uniform = valid_mean(F.binary_cross_entropy_with_logits(uniform_logits, targets, reduction="none"))
    soft = valid_mean(F.binary_cross_entropy_with_logits(soft_logits, targets, reduction="none"))
    hard = valid_mean(F.binary_cross_entropy_with_logits(hard_logits, targets, reduction="none"))
    base_mean = valid_mean(base)
    base_denominator_valid = base_mean > 1e-12
    safe_base_denominator = torch.where(
        base_denominator_valid, base_mean, torch.ones_like(base_mean)
    )
    g_local = torch.where(
        base_denominator_valid,
        (base_mean - oracle) / safe_base_denominator,
        torch.zeros_like(base_mean),
    )
    g_multi = torch.where(
        base_denominator_valid,
        (best_single - oracle) / safe_base_denominator,
        torch.zeros_like(base_mean),
    )
    capture_denominator = uniform - oracle
    capture_valid = capture_denominator > 1e-12
    safe_capture_denominator = torch.where(
        capture_valid, capture_denominator, torch.ones_like(capture_denominator)
    )
    capture = torch.where(
        capture_valid,
        (uniform - soft) / safe_capture_denominator,
        torch.zeros_like(capture_denominator),
    )
    teacher = payload["q_utility"]
    router = dense_probabilities.float().clamp_min(1e-12)
    informative = payload["informative"]
    teacher_router_kl_patch = (
        teacher * (teacher.clamp_min(1e-12).log() - router.log())
    ).sum(dim=-1)
    teacher_router_kl = (
        teacher_router_kl_patch[informative].mean()
        if informative.any() else teacher_router_kl_patch.sum() * 0.0
    )
    winners = payload["winner"]
    winner_shares = torch.stack([
        ((winners == factor) & valid).sum().float() / valid.sum().clamp_min(1)
        for factor in range(per_factor.shape[-1])
    ])
    all_harm = (payload["gain_rel"].max(dim=-1).values <= 0) & valid
    noop_candidates = torch.cat([base.unsqueeze(-1), per_factor], dim=-1)
    noop_winner = noop_candidates.argmin(dim=-1)
    noop_winner_shares = torch.stack([
        ((noop_winner == index) & valid).sum().float() / valid.sum().clamp_min(1)
        for index in range(per_factor.shape[-1] + 1)
    ])
    oracle_with_noop = valid_mean(noop_candidates.min(dim=-1).values)
    return {
        "Base": base_mean.detach(),
        "BestSingle": best_single.detach(),
        "OracleMulti": oracle.detach(),
        "OracleWithNoOp": oracle_with_noop.detach(),
        "Uniform": uniform.detach(),
        "SoftRouted": soft.detach(),
        "HardRouted": hard.detach(),
        "G_local": g_local.detach(),
        "G_multi": g_multi.detach(),
        "base_denominator_valid": base_denominator_valid.detach(),
        "capture": capture.detach(),
        "capture_denominator": capture_denominator.detach(),
        "capture_valid": capture_valid.detach(),
        "L_base": base_mean.detach(),
        "L_per_factor": factor_means.detach(),
        "gain_rel_mean": payload["gain_rel"][factor_valid].mean().detach(),
        "best_second_utility_margin": (
            payload["gain_rel"].topk(2, dim=-1).values.diff(dim=-1).abs().mean().detach()
        ),
        "teacher_entropy": payload["normalized_entropy"][valid].mean().detach(),
        "teacher_max_probability": teacher.max(dim=-1).values[valid].mean().detach(),
        "informative_fraction": informative.float()[valid].mean().detach(),
        "all_harm_fraction": all_harm.float()[valid].mean().detach(),
        "winner_shares": winner_shares.detach(),
        "noop_winner_shares": noop_winner_shares.detach(),
        "noop_selected_fraction": noop_winner_shares[0].detach(),
        "router_top1_agreement": (
            (dense_probabilities.argmax(dim=-1) == winners)[informative].float().mean().detach()
            if informative.any() else base_mean.detach() * 0.0
        ),
        "teacher_router_KL": teacher_router_kl.detach(),
        "router_entropy": (-(router * router.log()).sum(dim=-1) / math.log(router.shape[-1]))[valid].mean().detach(),
        "router_usage": router[valid].mean(dim=0).detach(),
    }
