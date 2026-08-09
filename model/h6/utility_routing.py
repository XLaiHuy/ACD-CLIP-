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
    epsilon: float = 0.15,
    gain_threshold: float = 0.02,
    entropy_threshold: float = 0.98,
) -> Dict[str, torch.Tensor]:
    """Build a detached utility responsibility from relative BCE improvement."""
    if factor_local_evidence.ndim != 4:
        raise ValueError("factor_local_evidence must be [G,B,P,M]")
    groups, batch, patches, factors = factor_local_evidence.shape
    if factors != 4:
        raise ValueError("P1-v8.3 is locked to M=4")
    if base_logits.shape != (groups, batch, patches):
        raise ValueError("base_logits must be [G,B,P]")
    if y_patch.shape != (batch, patches) or local_mask_valid.shape != (batch, patches):
        raise ValueError("patch targets/validity must be [B,P]")
    if tau_utility <= 0 or denominator_floor <= 0:
        raise ValueError("tau_utility and denominator_floor must be positive")
    if not 0.0 <= float(epsilon) <= 1.0:
        raise ValueError("epsilon must be in [0,1]")
    if float(rho) not in (0.0, 0.05):
        raise ValueError("rho may only be canonical 0.05 or diagnostic 0")

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
    q = F.softmax(gain_rel.detach() / float(tau_utility), dim=-1)
    responsibility = ((1.0 - float(epsilon)) * q + float(epsilon) / factors).detach()
    normalized_entropy = -(
        q * q.clamp_min(1e-12).log()
    ).sum(dim=-1) / math.log(factors)
    best_gain_rel, winners = gain_rel.detach().max(dim=-1)
    informative = (
        (best_gain_rel > float(gain_threshold))
        & (normalized_entropy < float(entropy_threshold))
        & local_mask_valid.unsqueeze(0)
    )
    return {
        "z0": z0,
        "candidate_logits": candidate_logits,
        "loss_base": loss_base,
        "loss_per_factor": loss_per_factor,
        "gain_rel": gain_rel,
        "q_utility": q.detach(),
        "responsibility": responsibility,
        "normalized_entropy": normalized_entropy.detach(),
        "best_gain_rel": best_gain_rel,
        "winner": winners.detach(),
        "informative": informative.detach(),
        "valid": local_mask_valid.unsqueeze(0).expand(groups, -1, -1),
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
    g_local = base_mean - best_single
    g_multi = best_single - oracle
    denominator = (base_mean - oracle).clamp_min(1e-12)
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
    return {
        "Base": base_mean.detach(),
        "BestSingle": best_single.detach(),
        "OracleMulti": oracle.detach(),
        "Uniform": uniform.detach(),
        "SoftRouted": soft.detach(),
        "HardRouted": hard.detach(),
        "G_local": g_local.detach(),
        "G_multi": g_multi.detach(),
        "capture": ((base_mean - soft) / denominator).detach(),
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
        "router_top1_agreement": (
            (dense_probabilities.argmax(dim=-1) == winners)[informative].float().mean().detach()
            if informative.any() else base_mean.detach() * 0.0
        ),
        "teacher_router_KL": teacher_router_kl.detach(),
        "router_entropy": (-(router * router.log()).sum(dim=-1) / math.log(router.shape[-1]))[valid].mean().detach(),
        "router_usage": router[valid].mean(dim=0).detach(),
    }
