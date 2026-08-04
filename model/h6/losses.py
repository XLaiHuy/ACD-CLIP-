"""Numerically explicit auxiliary losses for H6 Progress 1."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def _patch_labels(mask: torch.Tensor, patch_count: int) -> torch.Tensor:
    grid = int(math.isqrt(int(patch_count)))
    if grid * grid != patch_count:
        raise ValueError(f"H6 center loss requires a square patch grid, got P={patch_count}")
    reduced = F.adaptive_max_pool2d(mask.float(), output_size=(grid, grid))
    return reduced.flatten(start_dim=1) > 0


def _nearest_prototype_distance(tokens: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    # tokens [B,P,D], prototypes [B,M,D] -> nearest squared L2 [B,P]
    distance = (tokens.float().unsqueeze(2) - prototypes.float().unsqueeze(1)).pow(2).mean(dim=-1)
    return distance.min(dim=2).values


def center_loss(
    projected_levels: torch.Tensor,
    prototype_normal: torch.Tensor,
    prototype_abnormal: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Anomaly-preserving CoPS center loss with robust empty-set handling."""
    if projected_levels.ndim != 4:
        raise ValueError("projected_levels must be [G,B,P,D]")
    groups, batch, patches, _ = projected_levels.shape
    if masks.shape[0] != batch or labels.shape[0] != batch:
        raise ValueError("mask/label batch does not match projected visual features")
    anomaly = _patch_labels(masks, patches)
    labels = labels.reshape(batch).bool()
    terms = []
    for group in range(groups):
        distances_normal = _nearest_prototype_distance(projected_levels[group], prototype_normal)
        distances_abnormal = _nearest_prototype_distance(projected_levels[group], prototype_abnormal)
        normal_image = ~labels
        if normal_image.any():
            terms.append(distances_normal[normal_image].mean())
        anomalous_image = labels
        if anomalous_image.any():
            normal_patch = anomalous_image[:, None] & ~anomaly
            abnormal_patch = anomalous_image[:, None] & anomaly
            if normal_patch.any():
                terms.append(distances_normal[normal_patch].mean())
            if abnormal_patch.any():
                terms.append(distances_abnormal[abnormal_patch].mean())
    if not terms:
        return projected_levels.float().sum() * 0.0
    return torch.stack(terms).mean()


def _factor_weighted_distance(distance: torch.Tensor, assignment: torch.Tensor, selector: torch.Tensor) -> torch.Tensor | None:
    if not selector.any():
        return None
    return (distance[selector] * assignment[selector]).sum(dim=-1).mean()


def factor_aware_center_loss(
    projected_levels: torch.Tensor,
    prototype_normal: torch.Tensor,
    prototype_abnormal: torch.Tensor,
    dense_probabilities: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    detach_assignment: bool = True,
    margin: float = 0.0,
) -> torch.Tensor:
    """CoPS center loss that preserves the router's factor assignment.

    The legacy center loss minimized distance to the nearest prototype across
    all factors.  That can erase factor specialization because every patch can
    always choose whichever factor is easiest.  This version weights each
    factor-specific prototype distance by the dense router probabilities.  The
    dense assignment is detached by default so the center term shapes prototypes
    without directly pulling the router toward a degenerate assignment.
    """
    if projected_levels.ndim != 4:
        raise ValueError("projected_levels must be [G,B,P,D]")
    if dense_probabilities.ndim != 4:
        raise ValueError("dense_probabilities must be [G,B,P,M]")
    groups, batch, patches, dim = projected_levels.shape
    if prototype_normal.shape != (batch, dense_probabilities.shape[-1], dim):
        raise ValueError("prototype_normal must be [B,M,D] and match projected feature dim")
    if prototype_abnormal.shape != prototype_normal.shape:
        raise ValueError("prototype_abnormal must match prototype_normal")
    if dense_probabilities.shape[:3] != (groups, batch, patches):
        raise ValueError("dense_probabilities must match projected_levels [G,B,P,M]")
    if masks.shape[0] != batch or labels.shape[0] != batch:
        raise ValueError("mask/label batch does not match projected visual features")

    assignment = dense_probabilities.float()
    if detach_assignment:
        assignment = assignment.detach()
    anomaly = _patch_labels(masks, patches)
    labels = labels.reshape(batch).bool()
    terms = []
    margin_terms = []
    for group in range(groups):
        tokens = projected_levels[group].float()
        distance_normal = (tokens.unsqueeze(2) - prototype_normal.float().unsqueeze(1)).pow(2).mean(dim=-1)
        distance_abnormal = (tokens.unsqueeze(2) - prototype_abnormal.float().unsqueeze(1)).pow(2).mean(dim=-1)
        assignment_g = assignment[group]

        normal_image = (~labels)[:, None].expand(batch, patches)
        term = _factor_weighted_distance(distance_normal, assignment_g, normal_image)
        if term is not None:
            terms.append(term)

        anomalous_image = labels[:, None].expand(batch, patches)
        normal_patch = anomalous_image & ~anomaly
        abnormal_patch = anomalous_image & anomaly
        term = _factor_weighted_distance(distance_normal, assignment_g, normal_patch)
        if term is not None:
            terms.append(term)
        term = _factor_weighted_distance(distance_abnormal, assignment_g, abnormal_patch)
        if term is not None:
            terms.append(term)

        if margin > 0.0:
            margin_normal = F.relu(distance_normal - distance_abnormal + float(margin))
            margin_abnormal = F.relu(distance_abnormal - distance_normal + float(margin))
            term = _factor_weighted_distance(margin_normal, assignment_g, normal_image | normal_patch)
            if term is not None:
                margin_terms.append(term)
            term = _factor_weighted_distance(margin_abnormal, assignment_g, abnormal_patch)
            if term is not None:
                margin_terms.append(term)

    if not terms and not margin_terms:
        return projected_levels.float().sum() * 0.0
    all_terms = terms + margin_terms
    return torch.stack(all_terms).mean()


def router_teacher_loss(
    projected_levels: torch.Tensor,
    prototype_normal: torch.Tensor,
    prototype_abnormal: torch.Tensor,
    dense_probabilities: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.15,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """State-aware detached prototype teacher for dense router probabilities.

    Teacher direction is prototype -> router only: prototype similarities are
    detached before softmax, while the cross-entropy updates the dense router
    query/key path.  KL units are unrelated to this loss.
    """
    if temperature <= 0:
        raise ValueError("router teacher temperature must be positive")
    if projected_levels.ndim != 4 or dense_probabilities.ndim != 4:
        raise ValueError("projected_levels and dense_probabilities must be [G,B,P,*]")
    groups, batch, patches, dim = projected_levels.shape
    if dense_probabilities.shape[:3] != (groups, batch, patches):
        raise ValueError("dense_probabilities must match projected_levels [G,B,P,M]")
    if prototype_normal.shape != (batch, dense_probabilities.shape[-1], dim):
        raise ValueError("prototype_normal must be [B,M,D]")
    if prototype_abnormal.shape != prototype_normal.shape:
        raise ValueError("prototype_abnormal must match prototype_normal")

    anomaly = _patch_labels(masks, patches)
    image_is_abnormal = labels.reshape(batch).bool()[:, None]
    abnormal_patch = image_is_abnormal & anomaly
    terms = []
    entropies = []
    normalized_entropies = []
    max_probabilities = []
    patch_stds = []
    usages = []
    unique_pairs = []
    router_kls = []
    raw_similarity_means = []
    raw_similarity_factor_stds = []
    raw_similarity_patch_stds = []
    raw_similarity_mins = []
    raw_similarity_maxs = []
    raw_logit_ranges = []
    normal_patch_counts = []
    abnormal_patch_counts = []
    for group in range(groups):
        tokens = F.normalize(projected_levels[group].float(), dim=-1)
        normal_proto = F.normalize(prototype_normal.float(), dim=-1)
        abnormal_proto = F.normalize(prototype_abnormal.float(), dim=-1)
        similarity_normal = torch.einsum("bpd,bmd->bpm", tokens, normal_proto)
        similarity_abnormal = torch.einsum("bpd,bmd->bpm", tokens, abnormal_proto)
        similarity = torch.where(abnormal_patch[..., None], similarity_abnormal, similarity_normal)
        raw_similarity_means.append(similarity.detach().mean(dim=(0, 1)))
        raw_similarity_factor_stds.append(similarity.detach().std(dim=-1, unbiased=False).mean())
        raw_similarity_patch_stds.append(similarity.detach().std(dim=(0, 1), unbiased=False).mean())
        raw_similarity_mins.append(similarity.detach().min())
        raw_similarity_maxs.append(similarity.detach().max())
        raw_logit_ranges.append((similarity.detach().max() - similarity.detach().min()) / float(temperature))
        normal_patch_counts.append((~abnormal_patch).sum())
        abnormal_patch_counts.append(abnormal_patch.sum())
        teacher = F.softmax(similarity.detach() / float(temperature), dim=-1)
        student_log = dense_probabilities[group].float().clamp_min(1e-8).log()
        terms.append(-(teacher * student_log).sum(dim=-1).mean())
        entropy = (-(teacher * teacher.clamp_min(1e-8).log()).sum(dim=-1)).mean()
        entropies.append(entropy)
        normalized_entropies.append(entropy / math.log(float(dense_probabilities.shape[-1])))
        max_probabilities.append(teacher.max(dim=-1).values.mean())
        patch_stds.append(teacher.float().std(dim=(0, 1), unbiased=False).mean())
        usage = teacher.float().mean(dim=(0, 1))
        usages.append(usage)
        topk = torch.topk(teacher, k=min(2, teacher.shape[-1]), dim=-1).indices
        pairs = torch.sort(topk.detach().long(), dim=-1).values.reshape(-1, topk.shape[-1])
        unique_pairs.append(torch.tensor(torch.unique(pairs, dim=0).shape[0], device=teacher.device))
        router_kls.append((teacher * (teacher.clamp_min(1e-8).log() - student_log)).sum(dim=-1).mean())
    if not terms:
        zero = projected_levels.float().sum() * 0.0
        return zero, {"router_teacher_entropy": zero.detach()}
    loss = torch.stack(terms).mean()
    diagnostics = {
        "router_teacher_entropy": torch.stack(entropies).mean().detach(),
        "teacher_entropy": torch.stack(normalized_entropies).detach(),
        "teacher_max_probability": torch.stack(max_probabilities).detach(),
        "teacher_probability_std_across_patches": torch.stack(patch_stds).detach(),
        "teacher_usage": torch.stack(usages).detach(),
        "teacher_unique_topk_pairs": torch.stack(unique_pairs).long().detach(),
        "teacher_router_kl": torch.stack(router_kls).detach(),
        "teacher_raw_similarity_mean_per_factor": torch.stack(raw_similarity_means).detach(),
        "teacher_raw_similarity_std_across_factors": torch.stack(raw_similarity_factor_stds).detach(),
        "teacher_raw_similarity_std_across_patches": torch.stack(raw_similarity_patch_stds).detach(),
        "teacher_raw_similarity_min": torch.stack(raw_similarity_mins).detach(),
        "teacher_raw_similarity_max": torch.stack(raw_similarity_maxs).detach(),
        "teacher_raw_logit_range": torch.stack(raw_logit_ranges).detach(),
        "normal_patch_count": torch.stack(normal_patch_counts).detach(),
        "abnormal_patch_count": torch.stack(abnormal_patch_counts).detach(),
        "router_teacher_finite": torch.isfinite(loss).detach(),
    }
    return loss, diagnostics


def _teacher_distribution_diagnostics(probabilities: torch.Tensor) -> Dict[str, torch.Tensor]:
    factors = probabilities.shape[-1]
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    normalized_entropy = entropy.mean(dim=(1, 2)) / math.log(float(factors))
    max_prob = probabilities.max(dim=-1).values.mean(dim=(1, 2))
    patch_std = probabilities.float().std(dim=(1, 2), unbiased=False).mean(dim=-1)
    usage = probabilities.float().mean(dim=(1, 2))
    topk = torch.topk(probabilities, k=min(2, factors), dim=-1).indices
    unique = []
    for group in range(probabilities.shape[0]):
        pairs = torch.sort(topk[group].detach().long(), dim=-1).values.reshape(-1, topk.shape[-1])
        unique.append(torch.tensor(torch.unique(pairs, dim=0).shape[0], device=probabilities.device))
    return {
        "entropy": normalized_entropy.detach(),
        "max_probability": max_prob.detach(),
        "probability_std_across_patches": patch_std.detach(),
        "usage": usage.detach(),
        "unique_topk_pairs": torch.stack(unique).long().detach(),
    }


def teacher_candidate_diagnostics(
    projected_levels: torch.Tensor,
    prototype_normal: torch.Tensor,
    prototype_abnormal: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Compare diagnostic teacher similarities without changing the train loss."""
    if projected_levels.ndim != 4:
        raise ValueError("projected_levels must be [G,B,P,D]")
    groups, batch, patches, _ = projected_levels.shape
    anomaly = _patch_labels(masks, patches)
    image_is_abnormal = labels.reshape(batch).bool()[:, None]
    abnormal_patch = image_is_abnormal & anomaly
    raw_probs = []
    centered_probs = []
    distance_probs = []
    for group in range(groups):
        tokens = projected_levels[group].float()
        normal_proto = prototype_normal.float()
        abnormal_proto = prototype_abnormal.float()
        chosen_proto = torch.where(abnormal_patch[..., None, None], abnormal_proto[:, None], normal_proto[:, None])

        raw_sim = torch.einsum(
            "bpd,bpmd->bpm",
            F.normalize(tokens, dim=-1),
            F.normalize(chosen_proto, dim=-1),
        )
        center = tokens.mean(dim=1, keepdim=True).detach()
        token_residual = F.normalize(tokens - center, dim=-1)
        proto_residual = F.normalize(chosen_proto - center.unsqueeze(2), dim=-1)
        centered_sim = torch.einsum("bpd,bpmd->bpm", token_residual, proto_residual)
        distance_sim = -(tokens.unsqueeze(2) - chosen_proto).pow(2).mean(dim=-1)

        raw_probs.append(F.softmax(raw_sim.detach() / float(temperature), dim=-1))
        centered_probs.append(F.softmax(centered_sim.detach() / float(temperature), dim=-1))
        distance_probs.append(F.softmax(distance_sim.detach() / float(temperature), dim=-1))

    output = {}
    for name, probs in (
        ("teacher_raw_candidate", torch.stack(raw_probs, dim=0)),
        ("teacher_centered_candidate", torch.stack(centered_probs, dim=0)),
        ("teacher_distance_candidate", torch.stack(distance_probs, dim=0)),
    ):
        diag = _teacher_distribution_diagnostics(probs)
        for key, value in diag.items():
            output[f"{name}_{key}"] = value
    return output


def concept_key_diversity_loss(concept_keys: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Cosine-margin anti-collapse loss for actual router concept keys."""
    if concept_keys.ndim != 2:
        raise ValueError("concept_keys must be [M,D]")
    factors = concept_keys.shape[0]
    if factors <= 1:
        return concept_keys.float().sum() * 0.0
    keys = F.normalize(concept_keys.float(), dim=-1)
    cosine = keys @ keys.T
    offdiag_mask = ~torch.eye(factors, device=concept_keys.device, dtype=torch.bool)
    excess = F.relu(cosine[offdiag_mask] - float(margin))
    return excess.pow(2).mean()


def pairwise_vector_diagnostics(vectors: torch.Tensor, prefix: str) -> Dict[str, torch.Tensor]:
    if vectors.ndim != 2:
        raise ValueError("vectors must be [M,D]")
    if vectors.shape[0] <= 1:
        zero = vectors.float().sum() * 0.0
        return {
            f"{prefix}_cos_mean": zero.detach(),
            f"{prefix}_cos_max": zero.detach(),
            f"{prefix}_l2_min": zero.detach(),
        }
    values = vectors.float()
    cosine = F.normalize(values, dim=-1) @ F.normalize(values, dim=-1).T
    mask = ~torch.eye(values.shape[0], device=values.device, dtype=torch.bool)
    l2 = torch.cdist(values, values)[mask]
    return {
        f"{prefix}_cos_mean": cosine[mask].mean().detach(),
        f"{prefix}_cos_max": cosine[mask].abs().max().detach(),
        f"{prefix}_l2_min": l2.min().detach(),
    }


def factor_stage_diagnostics(values: torch.Tensor, prefix: str, factor_dim: int) -> Dict[str, torch.Tensor]:
    """Pairwise factor identity diagnostics for arbitrary stage tensors."""
    factor_dim = factor_dim % values.ndim
    moved = values.float().movedim(factor_dim, 0)
    flattened = moved.reshape(moved.shape[0], -1)
    return pairwise_vector_diagnostics(flattened, prefix)


def factor_orthogonal_loss(factor_bank: torch.Tensor) -> torch.Tensor:
    """Weakly diversify normal-to-abnormal directions in FP32."""
    if factor_bank.ndim != 5:
        raise ValueError("factor_bank must be [G,B,M,768,2]")
    directions = factor_bank[..., 1] - factor_bank[..., 0]
    directions = F.normalize(directions.float(), dim=-1)
    gram = torch.einsum("gbmd,gbnd->gbmn", directions, directions)
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    return (gram - identity).pow(2).mean()


def dynamic_residual_diversity_loss(dynamic_text: torch.Tensor, hard_frozen: torch.Tensor) -> torch.Tensor:
    """Diversify dynamic text residual directions around the frozen hard anchor."""
    if dynamic_text.ndim != 5:
        raise ValueError("dynamic_text must be [G,B,M,768,2]")
    if hard_frozen.ndim == 4:
        hard_frozen = hard_frozen.unsqueeze(2)
    if hard_frozen.ndim != 5:
        raise ValueError("hard_frozen must be [G,B,768,2] or [G,B,M,768,2]")
    dynamic_text = F.normalize(dynamic_text.float(), dim=3)
    hard_frozen = F.normalize(hard_frozen.float(), dim=3).expand_as(dynamic_text)
    residual = dynamic_text - hard_frozen
    directions = residual[..., 1] - residual[..., 0]
    directions = F.normalize(directions.float(), dim=-1)
    gram = torch.einsum("gbmd,gbnd->gbmn", directions, directions)
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    return (gram - identity).pow(2).mean()


def dynamic_residual_diagnostics(dynamic_text: torch.Tensor, hard_frozen: torch.Tensor) -> Dict[str, torch.Tensor]:
    if hard_frozen.ndim == 4:
        hard_frozen = hard_frozen.unsqueeze(2)
    dynamic_text = F.normalize(dynamic_text.float(), dim=3)
    hard_frozen = F.normalize(hard_frozen.float(), dim=3).expand_as(dynamic_text)
    residual = dynamic_text - hard_frozen
    directions = F.normalize((residual[..., 1] - residual[..., 0]).float(), dim=-1)
    # Average over levels/batch; this preserves the factor dimension being audited.
    vectors = directions.mean(dim=(0, 1))
    return pairwise_vector_diagnostics(vectors, "dynamic_residual")


def prototype_diagnostics(prototype_normal: torch.Tensor, prototype_abnormal: torch.Tensor) -> Dict[str, torch.Tensor]:
    if prototype_normal.shape != prototype_abnormal.shape or prototype_normal.ndim != 3:
        raise ValueError("prototypes must both be [B,M,D]")
    paired = F.normalize((prototype_abnormal.float() - prototype_normal.float()).mean(dim=0), dim=-1)
    out = pairwise_vector_diagnostics(paired, "prototype")
    out["prototype_variance"] = torch.stack([prototype_normal.float(), prototype_abnormal.float()]).var(
        dim=(0, 1, 3), unbiased=False
    ).mean().detach()
    return out


def routing_balance_loss(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 4:
        raise ValueError("routing probabilities must be [G,B,P,M]")
    usage = probabilities.float().mean(dim=(0, 1, 2))
    target = torch.full_like(usage, 1.0 / usage.numel())
    return (usage - target).pow(2).sum()


def h6_loss_diagnostics(
    factor_bank: torch.Tensor,
    probabilities: torch.Tensor,
    prototype_normal: torch.Tensor,
    prototype_abnormal: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    directions = F.normalize((factor_bank[..., 1] - factor_bank[..., 0]).float(), dim=-1)
    direction_cosine = torch.einsum("gbmd,gbnd->gbmn", directions, directions)
    return {
        "direction_offdiag_absmax": (direction_cosine - torch.eye(direction_cosine.shape[-1], device=direction_cosine.device)).abs().amax().detach(),
        "prototype_normal_norm": prototype_normal.float().norm(dim=-1).mean().detach(),
        "prototype_abnormal_norm": prototype_abnormal.float().norm(dim=-1).mean().detach(),
        "center_distance": (prototype_normal.float() - prototype_abnormal.float()).norm(dim=-1).mean().detach(),
        "factor_bank_finite": torch.isfinite(factor_bank).all().detach(),
        "routing_finite": torch.isfinite(probabilities).all().detach(),
    }
