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
    mode: str = "raw_cosine",
    confidence_gate_enabled: bool = False,
    entropy_threshold: float = 0.98,
    probability_std_threshold: float = 1e-3,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """State-aware detached prototype teacher for dense router probabilities.

    Teacher direction is prototype -> router only: prototype similarities are
    detached before softmax, while the cross-entropy updates the dense router
    query/key path.  KL units are unrelated to this loss.
    """
    if temperature <= 0:
        raise ValueError("router teacher temperature must be positive")
    if mode not in {"raw_cosine", "state_centered_cosine", "negative_squared_distance"}:
        raise ValueError(f"unsupported router teacher mode: {mode}")
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
    informative_counts = []
    valid_counts = []
    active_levels = []
    for group in range(groups):
        tokens = projected_levels[group].float()
        normal_proto = prototype_normal.float()
        abnormal_proto = prototype_abnormal.float()
        chosen_proto = torch.where(abnormal_patch[..., None, None], abnormal_proto[:, None], normal_proto[:, None])
        selected_center = torch.where(
            abnormal_patch[..., None],
            abnormal_proto.mean(dim=1).detach()[:, None, :],
            normal_proto.mean(dim=1).detach()[:, None, :],
        )
        raw_similarity = torch.einsum(
            "bpd,bpmd->bpm",
            F.normalize(tokens, dim=-1),
            F.normalize(chosen_proto, dim=-1),
        )
        centered_similarity = torch.einsum(
            "bpd,bpmd->bpm",
            F.normalize(tokens - selected_center, dim=-1),
            F.normalize(chosen_proto - selected_center.unsqueeze(2), dim=-1),
        )
        distance_similarity = -(tokens.unsqueeze(2) - chosen_proto).pow(2).mean(dim=-1)
        if mode == "raw_cosine":
            similarity = raw_similarity
        elif mode == "state_centered_cosine":
            similarity = centered_similarity
        else:
            similarity = distance_similarity
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
        per_patch_loss = -(teacher * student_log).sum(dim=-1)
        per_patch_entropy = -(teacher * teacher.clamp_min(1e-8).log()).sum(dim=-1)
        normalized_patch_entropy = per_patch_entropy / math.log(float(dense_probabilities.shape[-1]))
        per_patch_std = teacher.float().std(dim=-1, unbiased=False)
        valid_patch = torch.ones_like(per_patch_loss, dtype=torch.bool)
        informative_patch = (
            valid_patch
            & (normalized_patch_entropy < float(entropy_threshold))
            & (per_patch_std > float(probability_std_threshold))
        )
        selected_patch = informative_patch if confidence_gate_enabled else valid_patch
        informative_counts.append(informative_patch.sum())
        valid_counts.append(valid_patch.sum())
        active_levels.append(torch.as_tensor(bool(selected_patch.any().item()), device=teacher.device))
        if selected_patch.any():
            terms.append(per_patch_loss[selected_patch].mean())
        else:
            terms.append(dense_probabilities[group].float().sum() * 0.0)
        entropy = per_patch_entropy.mean()
        entropies.append(entropy)
        normalized_entropies.append(normalized_patch_entropy.mean())
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
        "teacher_informative_patch_count": torch.stack(informative_counts).detach(),
        "teacher_valid_patch_count": torch.stack(valid_counts).detach(),
        "teacher_informative_patch_fraction": (
            torch.stack(informative_counts).float() / torch.stack(valid_counts).float().clamp_min(1.0)
        ).detach(),
        "teacher_active_levels": torch.stack(active_levels).bool().detach(),
        "teacher_gate_reason": torch.as_tensor(
            0 if (not confidence_gate_enabled or torch.stack(informative_counts).sum().item() > 0) else 1,
            device=projected_levels.device,
        ),
        "router_teacher_finite": torch.isfinite(loss).detach(),
    }
    return loss, diagnostics


def _teacher_distribution_diagnostics(probabilities: torch.Tensor, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        "logit_range": (logits.float().amax(dim=(1, 2, 3)) - logits.float().amin(dim=(1, 2, 3))).detach(),
        "logit_std_across_factors": logits.float().std(dim=-1, unbiased=False).mean(dim=(1, 2)).detach(),
        "logit_std_across_patches": logits.float().std(dim=(1, 2), unbiased=False).mean(dim=-1).detach(),
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
    raw_logits = []
    centered_logits = []
    distance_logits = []
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
        center = torch.where(
            abnormal_patch[..., None],
            abnormal_proto.mean(dim=1).detach()[:, None, :],
            normal_proto.mean(dim=1).detach()[:, None, :],
        )
        token_residual = F.normalize(tokens - center, dim=-1)
        proto_residual = F.normalize(chosen_proto - center.unsqueeze(2), dim=-1)
        centered_sim = torch.einsum("bpd,bpmd->bpm", token_residual, proto_residual)
        distance_sim = -(tokens.unsqueeze(2) - chosen_proto).pow(2).mean(dim=-1)

        raw_probs.append(F.softmax(raw_sim.detach() / float(temperature), dim=-1))
        centered_probs.append(F.softmax(centered_sim.detach() / float(temperature), dim=-1))
        distance_probs.append(F.softmax(distance_sim.detach() / float(temperature), dim=-1))
        raw_logits.append(raw_sim.detach())
        centered_logits.append(centered_sim.detach())
        distance_logits.append(distance_sim.detach())

    output = {}
    for name, probs, logits in (
        ("teacher_raw_candidate", torch.stack(raw_probs, dim=0), torch.stack(raw_logits, dim=0)),
        ("teacher_centered_candidate", torch.stack(centered_probs, dim=0), torch.stack(centered_logits, dim=0)),
        ("teacher_distance_candidate", torch.stack(distance_probs, dim=0), torch.stack(distance_logits, dim=0)),
    ):
        diag = _teacher_distribution_diagnostics(probs, logits)
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


def delta_t_diversity_loss(dynamic_text: torch.Tensor, hard_frozen: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal squared cosine of dynamic residual directions only."""
    if dynamic_text.ndim != 5:
        raise ValueError("dynamic_text must be [G,B,M,768,2]")
    if hard_frozen.ndim == 4:
        hard_frozen = hard_frozen.unsqueeze(2)
    if hard_frozen.ndim != 5:
        raise ValueError("hard_frozen must be [G,B,768,2] or [G,B,M,768,2]")
    residual = F.normalize(dynamic_text.float(), dim=3) - F.normalize(
        hard_frozen.float(), dim=3
    ).expand_as(dynamic_text)
    delta_t = residual[..., 1] - residual[..., 0]
    directions = F.normalize(delta_t.float(), dim=-1, eps=1e-6)
    gram = torch.einsum("gbmd,gbnd->gbmn", directions, directions)
    mask = ~torch.eye(gram.shape[-1], device=gram.device, dtype=torch.bool)
    return gram[..., mask].square().mean()


def functional_factor_diversity_loss(
    factor_patch_logits: torch.Tensor,
    patch_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decorrelate centered factor patch functions, returning loss and correlation matrix.

    ``factor_patch_logits`` is [G,B,P,M].  We detach the supplied informative
    patch weights so GT/high-confidence anchor selection cannot be optimized as
    a proxy for diversity.
    """
    if factor_patch_logits.ndim != 4:
        raise ValueError("factor_patch_logits must be [G,B,P,M]")
    logits = factor_patch_logits.float()
    if patch_weights is None:
        weights = torch.ones_like(logits[..., 0])
    else:
        if patch_weights.shape != logits.shape[:-1]:
            raise ValueError("patch_weights must be [G,B,P]")
        weights = patch_weights.detach().float().clamp_min(0.0)
    weights = weights * torch.isfinite(logits).all(dim=-1).float()
    denom = weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
    centered = logits - (weights.unsqueeze(-1) * logits).sum(dim=2, keepdim=True) / denom.unsqueeze(-1)
    weighted = centered * weights.sqrt().unsqueeze(-1)
    normalized = F.normalize(weighted.movedim(-1, -2), dim=-1, eps=1e-6)
    correlation = torch.matmul(normalized, normalized.transpose(-1, -2))
    mask = ~torch.eye(correlation.shape[-1], device=correlation.device, dtype=torch.bool)
    return correlation[..., mask].square().mean(), correlation.mean(dim=(0, 1))


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


def assigned_expert_loss(expert_patch_logits: torch.Tensor, prediction_probabilities: torch.Tensor,
                         masks: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Every expert learns normal; detached router responsibility learns anomaly."""
    if expert_patch_logits.ndim != 4 or prediction_probabilities.shape != expert_patch_logits.shape:
        raise ValueError("expert logits and prediction probabilities must both be [G,B,P,M]")
    _, _, patches, _ = expert_patch_logits.shape
    patch_is_abnormal = _patch_labels(masks, patches).to(expert_patch_logits.device)
    logits, weights = expert_patch_logits.float(), prediction_probabilities.detach().float()
    normal_mask = (~patch_is_abnormal).unsqueeze(0).unsqueeze(-1).expand_as(logits)
    zero = logits.sum() * 0.0
    normal_loss = F.binary_cross_entropy_with_logits(logits[normal_mask], torch.zeros_like(logits[normal_mask])) if normal_mask.any() else zero
    abnormal_mask = patch_is_abnormal.unsqueeze(0).unsqueeze(-1).expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
    denominator = (weights * abnormal_mask).sum()
    assigned = (weights * bce * abnormal_mask).sum() / denominator.clamp_min(1e-8) if denominator.detach().item() > 0 else zero
    return normal_loss + assigned, {"expert_normal_all": normal_loss.detach(), "expert_abnormal_assigned": assigned.detach()}


def expert_advantage_loss(expert_patch_logits: torch.Tensor, topk_indices: torch.Tensor,
                          masks: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    if expert_patch_logits.ndim != 4 or topk_indices.ndim != 4:
        raise ValueError("expert logits [G,B,P,M] and indices [G,B,P,K] are required")
    _, _, patches, factors = expert_patch_logits.shape
    abnormal = _patch_labels(masks, patches).to(expert_patch_logits.device).unsqueeze(0).unsqueeze(-1)
    selected = F.one_hot(topk_indices.detach().long(), factors).any(dim=-2) & abnormal
    nonselected = (~selected) & abnormal
    if not selected.any() or not nonselected.any():
        return expert_patch_logits.float().sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(expert_patch_logits.float(), torch.ones_like(expert_patch_logits.float()), reduction="none")
    return F.relu(bce[selected].mean() - bce[nonselected].mean() + float(margin))


def expert_etf_loss(delta_tangent: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    if delta_tangent.ndim != 4:
        raise ValueError("delta_tangent must be [G,B,M,D]")
    factors = delta_tangent.shape[2]
    if factors < 2:
        return delta_tangent.sum() * 0.0, delta_tangent.norm(dim=-1).mean().detach()
    norm = delta_tangent.float().norm(dim=-1, keepdim=True)
    directions = delta_tangent.float() / norm.clamp_min(eps)
    gram = torch.einsum("gbmd,gbnd->gbmn", directions, directions)
    target = torch.full_like(gram, -1.0 / float(factors - 1))
    target.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    valid = (norm.squeeze(-1) > eps).all(dim=-1)
    per = (gram - target).pow(2).mean(dim=(-1, -2))
    return (per[valid].mean() if valid.any() else delta_tangent.float().sum() * 0.0), norm.squeeze(-1).mean().detach()


def dual_routing_balance_loss(dense: torch.Tensor, prediction: torch.Tensor, sparse_weight: float = 1.0) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    dense_loss, prediction_loss = routing_balance_loss(dense), routing_balance_loss(prediction)
    return dense_loss + float(sparse_weight) * prediction_loss, {"dense_cv2": dense_loss.detach(), "prediction_cv2": prediction_loss.detach()}


def compute_final_expert_hard_cosine(
    expert_factor_bank: torch.Tensor,
    hard_frozen: torch.Tensor,
) -> torch.Tensor:
    """State-preserving FP32 cosine between final expert and frozen hard banks.

    The factor bank is [G, B, M, D, 2] while the frozen hard bank is
    [G, B, D, 2].  Moving the final state axis before cosine makes the
    embedding axis unambiguous and prevents accidentally comparing the two
    normal/abnormal states (an axis of length two) instead of D.
    """
    if expert_factor_bank.ndim != 5:
        raise ValueError("expert_factor_bank must be [G,B,M,D,2]")
    if hard_frozen.ndim != 4:
        raise ValueError("hard_frozen must be [G,B,D,2]")
    groups, batch, _, dim, states = expert_factor_bank.shape
    if hard_frozen.shape != (groups, batch, dim, states) or states != 2:
        raise ValueError("hard_frozen must match [G,B,D,2] from expert_factor_bank")
    final_mean = expert_factor_bank.float().mean(dim=2)  # [G,B,D,2]
    final_state = final_mean.movedim(-1, -2)             # [G,B,2,D]
    hard_state = hard_frozen.detach().float().movedim(-1, -2)
    return F.cosine_similarity(final_state, hard_state, dim=-1)


def factor_bank_comparison_diagnostics(
    left: torch.Tensor,
    right: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """FP32 state-preserving comparison for two [G,B,M,D,2] factor banks."""
    if left.ndim != 5 or right.ndim != 5 or left.shape != right.shape or left.shape[-1] != 2:
        raise ValueError("both factor banks must have matching [G,B,M,D,2] shape")
    left_state = left.float().movedim(-1, -2)    # [G,B,M,2,D]
    right_state = right.detach().float().movedim(-1, -2)
    cosine = F.cosine_similarity(left_state, right_state, dim=-1)
    difference = (left.float() - right.detach().float()).abs()
    return {
        "cos_mean": cosine.mean().detach(),
        "cos_min": cosine.min().detach(),
        "cos_p05": torch.quantile(cosine.reshape(-1), 0.05).detach(),
        "max_abs_diff": difference.max().detach(),
    }


def factor_bank_against_reference_diagnostics(
    factor_bank: torch.Tensor,
    reference_bank: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compare [G,B,M,D,2] to a reference [G,B,D,2] on the embedding axis."""
    if factor_bank.ndim != 5 or reference_bank.ndim != 4:
        raise ValueError("factor_bank/reference_bank must be [G,B,M,D,2]/[G,B,D,2]")
    if factor_bank.shape[:2] + factor_bank.shape[3:] != reference_bank.shape:
        raise ValueError("reference_bank must match factor bank [G,B,D,2]")
    broadcast = reference_bank.detach().unsqueeze(2).expand_as(factor_bank)
    return factor_bank_comparison_diagnostics(factor_bank, broadcast)


def expert_clip_anchor_loss(expert_factor_bank: torch.Tensor, hard_frozen: torch.Tensor,
                            min_cosine: float = 0.70) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = compute_final_expert_hard_cosine(expert_factor_bank, hard_frozen)
    return F.relu(float(min_cosine) - cosine).pow(2).mean(), cosine.detach()


def expert_radius_loss(relative_ratio: torch.Tensor, maximum: float) -> torch.Tensor:
    return F.relu(relative_ratio.float() - float(maximum)).pow(2).mean()


def expert_dead_counts(usage: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return boolean [G,M] dead mask and [G] dead-factor counts."""
    if usage.ndim != 2:
        raise ValueError("usage must be [G,M]")
    mask = usage.detach().float() <= float(threshold)
    return mask, mask.sum(dim=-1)


def sum_loss_components(components: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Single source of truth for the loss assembled by the training loop."""
    if not components:
        raise ValueError("loss components must not be empty")
    return sum(components.values())


def expert_patch_function_diagnostics(
    expert_patch_logits: torch.Tensor, topk_indices: torch.Tensor, masks: torch.Tensor, margin: float,
) -> Dict[str, torch.Tensor]:
    """Detached assigned-patch observability using the exact loss geometry."""
    groups, batch, patches, factors = expert_patch_logits.shape
    abnormal_patch = _patch_labels(masks, patches).to(expert_patch_logits.device)
    normal_patch = ~abnormal_patch
    selected = F.one_hot(topk_indices.detach().long(), factors).any(dim=-2)
    abnormal = abnormal_patch.unsqueeze(0).unsqueeze(-1).expand_as(selected)
    selected_abnormal, nonselected_abnormal = selected & abnormal, (~selected) & abnormal
    bce = F.binary_cross_entropy_with_logits(
        expert_patch_logits.float(), torch.ones_like(expert_patch_logits.float()), reduction="none"
    )
    zero = bce.sum() * 0.0
    selected_loss = bce[selected_abnormal].mean() if selected_abnormal.any() else zero
    nonselected_loss = bce[nonselected_abnormal].mean() if nonselected_abnormal.any() else zero
    valid_comparison = selected_abnormal.any() and nonselected_abnormal.any()
    return {
        "expert_normal_patch_count": normal_patch.float().sum().detach(),
        "expert_abnormal_patch_count": abnormal_patch.float().sum().detach(),
        "expert_valid_patch_count": torch.tensor(float(batch * patches), device=bce.device),
        "selected_expert_loss": selected_loss.detach(),
        "nonselected_expert_loss": nonselected_loss.detach(),
        "selected_minus_nonselected_loss": (selected_loss - nonselected_loss).detach(),
        "expert_advantage_margin_satisfied_fraction": (
            (bce[selected_abnormal] + float(margin) < nonselected_loss).float().mean().detach()
            if valid_comparison else zero.detach()
        ),
        "expert_advantage_valid_count": torch.tensor(
            float(selected_abnormal.sum().item() if valid_comparison else 0), device=bce.device
        ),
    }


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

def build_semantic_roles(
    masks: torch.Tensor,
    labels: torch.Tensor,
    patch_count: int,
    local_mask_valid: torch.Tensor,
    core_threshold: float = 0.99,
    boundary_threshold: float = 0.01,
    num_roles: int = 4,
    role_topology: str = "legacy_m4",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      q_role: [B, P, 4] soft indexed targets (one-hot of hard_role for now, or bounded distribution)
      hard_role: [B, P] integer roles 0-3
      mask_coverage: [B, P] float average of mask inside each patch
      local_valid_patch: [B, P] boolean mask of patches that should contribute to local losses
      local_valid_image: [B] boolean mask of images that have valid positive masks
    """
    if role_topology not in {"legacy_m4", "r2_normal_anomaly"}:
        raise ValueError("role_topology must be 'legacy_m4' or 'r2_normal_anomaly'")
    if role_topology == "r2_normal_anomaly" and int(num_roles) != 2:
        raise ValueError("r2_normal_anomaly requires num_roles=2")
    if role_topology == "legacy_m4" and int(num_roles) != 4:
        raise ValueError("legacy_m4 semantic roles require num_roles=4")
    B = masks.shape[0]
    grid = int(math.isqrt(int(patch_count)))
    mask_coverage = F.adaptive_avg_pool2d(masks.float(), output_size=(grid, grid)).view(B, patch_count)
    valid_coverage = F.adaptive_avg_pool2d(local_mask_valid.float(), output_size=(grid, grid)).view(B, patch_count)
    is_anomaly = labels.bool()
    
    local_valid_image = is_anomaly & (mask_coverage.sum(dim=-1) > 0)
    local_valid_patch = (~is_anomaly).unsqueeze(1).expand(B, patch_count) | local_valid_image.unsqueeze(1).expand(B, patch_count)
    local_valid_patch = local_valid_patch & (valid_coverage >= 0.5)
    
    hard_role = torch.zeros(B, patch_count, dtype=torch.long, device=masks.device)
    if role_topology == "r2_normal_anomaly":
        # Physical region supplies only the normal/anomaly prior. Utility
        # teacher probabilities remain the actual training supervision. A
        # normal/background role is assigned per patch, including background
        # patches inside an anomalous image; the anomaly role is reserved for
        # patches with positive mask coverage.
        anomaly_patch = local_valid_image.unsqueeze(1) & (mask_coverage > boundary_threshold)
        hard_role[anomaly_patch] = 1
    else:
        # Legacy M4 normal/outside/boundary/core semantics.
        hard_role[local_valid_image] = 1
        boundary_mask = local_valid_image.unsqueeze(1) & (mask_coverage > boundary_threshold) & (mask_coverage < core_threshold)
        hard_role[boundary_mask] = 2
        core_mask = local_valid_image.unsqueeze(1) & (mask_coverage >= core_threshold)
        hard_role[core_mask] = 3
    q_role = F.one_hot(hard_role, num_classes=int(num_roles)).float()
    
    return q_role, hard_role, mask_coverage, local_valid_patch, local_valid_image

def active_role_balanced_router_loss(
    dense_probabilities: torch.Tensor, 
    q_role: torch.Tensor, 
    hard_role: torch.Tensor, 
    local_valid_patch: torch.Tensor
) -> torch.Tensor:
    """
    dense_probabilities: [G, B, P, 4]
    q_role: [B, P, 4] soft targets
    hard_role: [B, P] integer 0-3
    local_valid_patch: [B, P] boolean
    """
    G, B, P, M = dense_probabilities.shape
    
    # [G, B, P]
    ce_patch = -(q_role.unsqueeze(0) * dense_probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    
    hard_role_exp = hard_role.unsqueeze(0).expand(G, B, P)
    valid_exp = local_valid_patch.unsqueeze(0).expand(G, B, P)
    
    loss_sum = 0.0
    valid_roles = 0
    
    for m in range(M):
        # Find patches that are valid and belong to hard role m
        role_mask = valid_exp & (hard_role_exp == m)
        count = role_mask.sum()
        if count > 0:
            role_ce = ce_patch[role_mask].sum() / count
            loss_sum += role_ce
            valid_roles += 1
            
    if valid_roles == 0:
        return dense_probabilities.sum() * 0.0
        
    return loss_sum / valid_roles

def get_desired_correction(
    mask_coverage: torch.Tensor, 
    base_abnormal_minus_normal: torch.Tensor, 
    correction_max: float = 10.0, 
    epsilon: float = 1e-4
) -> torch.Tensor:
    target_prob = mask_coverage.clamp(epsilon, 1.0 - epsilon)
    target_logit = torch.log(target_prob / (1.0 - target_prob))
    required = target_logit.unsqueeze(0) - base_abnormal_minus_normal.detach()
    required = required.clamp(-correction_max, correction_max)
    return required

def _role_balanced_smooth_l1(
    actual: torch.Tensor,
    desired: torch.Tensor,
    q_role: torch.Tensor,
    hard_role: torch.Tensor,
    local_valid_patch: torch.Tensor,
    role_idx: int,
    beta: float = 1.0
) -> torch.Tensor:
    G, B, P = actual.shape
    hard_role_exp = hard_role.unsqueeze(0).expand(G, B, P)
    valid_exp = local_valid_patch.unsqueeze(0).expand(G, B, P)
    role_mask = valid_exp & (hard_role_exp == role_idx)
    
    count = role_mask.sum()
    if count == 0:
        return actual.sum() * 0.0
        
    diff = actual - desired
    loss = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=beta, reduction='none')
    
    q_role_exp = q_role[..., role_idx].unsqueeze(0).expand(G, B, P)
    return (loss * q_role_exp)[role_mask].sum() / count

def factor_specific_residual_role_loss(
    rho_factor_patch_logits: torch.Tensor, 
    q_role: torch.Tensor,
    hard_role: torch.Tensor,
    mask_coverage: torch.Tensor,
    local_valid_patch: torch.Tensor,
    base_abnormal_minus_normal: torch.Tensor,
    correction_max: float = 10.0,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """
    rho_factor_patch_logits: [G, B, P, M]
    """
    G, B, P, M = rho_factor_patch_logits.shape
    desired = get_desired_correction(mask_coverage, base_abnormal_minus_normal, correction_max, epsilon) # [G, B, P]
    
    losses = []
    
    for m in range(M):
        actual = rho_factor_patch_logits[..., m]
        if M == 2:
            target_desired = torch.zeros_like(desired) if m == 0 else desired
        elif m == 0:
            target_desired = torch.zeros_like(desired)
        elif m == 1:
            target_desired = torch.minimum(desired, torch.zeros_like(desired))
        elif m == 2:
            target_desired = desired
        else:
            target_desired = torch.maximum(desired, torch.zeros_like(desired))

        role_loss = _role_balanced_smooth_l1(actual, target_desired, q_role, hard_role, local_valid_patch, m)
        if role_loss.requires_grad or role_loss.item() != 0.0:
            losses.append(role_loss)
            
    if not losses:
        return rho_factor_patch_logits.sum() * 0.0
    return sum(losses) / len(losses)

def actual_local_residual_loss(
    rho_h6_logits: torch.Tensor, 
    q_role: torch.Tensor,
    hard_role: torch.Tensor,
    mask_coverage: torch.Tensor,
    local_valid_patch: torch.Tensor,
    base_abnormal_minus_normal: torch.Tensor,
    correction_max: float = 10.0,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """
    rho_h6_logits: [G, B, P]
    """
    G, B, P = rho_h6_logits.shape
    desired = get_desired_correction(mask_coverage, base_abnormal_minus_normal, correction_max, epsilon)
    
    losses = []
    num_roles = int(q_role.shape[-1])
    for m in range(num_roles):
        actual = rho_h6_logits
        if num_roles == 2:
            target_desired = torch.zeros_like(desired) if m == 0 else desired
        elif m == 0:
            target_desired = torch.zeros_like(desired)
        elif m == 1:
            target_desired = torch.minimum(desired, torch.zeros_like(desired))
        elif m == 2:
            target_desired = desired
        else:
            target_desired = torch.maximum(desired, torch.zeros_like(desired))

        role_loss = _role_balanced_smooth_l1(actual, target_desired, q_role, hard_role, local_valid_patch, m)
        if role_loss.requires_grad or role_loss.item() != 0.0:
            losses.append(role_loss)
            
    if not losses:
        return rho_h6_logits.sum() * 0.0
    return sum(losses) / len(losses)
