"""P1-v8.3 utility teacher, dense router losses, and exact diagnostics."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


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
    routed_probabilities: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Build detached factor and router utility teachers.

    The legacy ``tau_utility`` and ``gain_threshold`` remain the defaults for
    both consumers.  Supplying a decoupled control changes only its named
    consumer, preserving existing P1-v8.3/v8.4-A behavior when unset.
    """
    if factor_local_evidence.ndim != 4:
        raise ValueError("factor_local_evidence must be [G,B,P,M]")
    groups, batch, patches, factors = factor_local_evidence.shape
    if factors != 4:
        raise ValueError("P1-v8.3 is locked to M=4")
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
    q_factor = F.softmax(gain_rel.detach() / factor_tau, dim=-1)
    q_router = F.softmax(gain_rel.detach() / router_tau, dim=-1)
    responsibility = (
        (1.0 - float(epsilon)) * q_factor + float(epsilon) / factors
    ).detach()
    normalized_entropy = -(
        q_router * q_router.clamp_min(1e-12).log()
    ).sum(dim=-1) / math.log(factors)
    best_gain_rel, winners = gain_rel.detach().max(dim=-1)
    informative = (
        (best_gain_rel > router_gain)
        & (normalized_entropy < float(entropy_threshold))
        & local_mask_valid.unsqueeze(0)
    )
    payload = {
        "z0": z0,
        "candidate_logits": candidate_logits,
        "loss_base": loss_base,
        "loss_per_factor": loss_per_factor,
        "gain_rel": gain_rel,
        "q_factor_utility": q_factor.detach(),
        "q_router_utility": q_router.detach(),
        # Backward-compatible public alias used by router losses/diagnostics.
        "q_utility": q_router.detach(),
        "responsibility": responsibility,
        "normalized_entropy": normalized_entropy.detach(),
        "best_gain_rel": best_gain_rel,
        "winner": winners.detach(),
        "informative": informative.detach(),
        "valid": local_mask_valid.unsqueeze(0).expand(groups, -1, -1),
    }
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
