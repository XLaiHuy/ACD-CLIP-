"""Frozen CIR-V2 primitives ported from commit 9cc0ad4cc6b34e34a8c15e74df881866516b3181.

This module intentionally contains only the frozen, GT-free relational math.
It has no dataset, checkpoint, SABRA, or target-label dependencies.  The
training integration must pass ``V2_TRANSPORT_DIRECTION`` explicitly; the
legacy V1 default is retained only because it is part of the frozen primitive
API and is never used by the H2 clean path.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


FROZEN_CIR_COMMIT = "9cc0ad4cc6b34e34a8c15e74df881866516b3181"
PEER_COUNT = 8
GROUP_COUNT = 3
MAD_CONSTANT = 1.4826
CIR_EPS = 1e-6
V1_TRANSPORT_DIRECTION = "abnormal_plus_normal_minus"
V2_TRANSPORT_DIRECTION = "abnormal_minus_normal_plus"
PATCH_GRID = (37, 37)
PATCH_COUNT = PATCH_GRID[0] * PATCH_GRID[1]


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def midpoint_median(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Return the ordinary median, using a midpoint for even sample counts."""
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.ndim == 0:
        raise ValueError("midpoint median requires a non-scalar tensor")
    dim = dim if dim >= 0 else values.ndim + dim
    if dim < 0 or dim >= values.ndim:
        raise ValueError(f"invalid median dimension {dim} for {tuple(values.shape)}")
    ordered = torch.sort(values, dim=dim).values
    count = int(ordered.shape[dim])
    if count < 1:
        raise ValueError("midpoint median cannot operate on an empty dimension")
    if count % 2:
        return ordered.select(dim, count // 2)
    return (ordered.select(dim, count // 2 - 1) + ordered.select(dim, count // 2)) * 0.5


def robust_peer_delta(
    observed_margin: torch.Tensor,
    peer_margins: torch.Tensor,
    *,
    eps: float = CIR_EPS,
    mad_constant: float = MAD_CONSTANT,
    peer_count: int = PEER_COUNT,
    peer_dim: int = -1,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute detached robust signed evidence from exactly eight peers."""
    if peer_margins.ndim < 1:
        raise ValueError("peer margins must have a peer dimension")
    peer_dim = int(peer_dim)
    if peer_dim < 0:
        peer_dim += peer_margins.ndim
    if peer_dim < 0 or peer_dim >= peer_margins.ndim:
        raise ValueError(f"invalid peer dimension {peer_dim} for {tuple(peer_margins.shape)}")
    observed_shape = tuple(peer_margins.shape[:peer_dim] + peer_margins.shape[peer_dim + 1:])
    if observed_shape != tuple(observed_margin.shape):
        raise ValueError(
            f"observed margin {tuple(observed_margin.shape)} and peer margins "
            f"{tuple(peer_margins.shape)} are incompatible"
        )
    if int(peer_margins.shape[peer_dim]) != int(peer_count):
        raise ValueError(f"CIR requires K={peer_count} peers, got {peer_margins.shape[peer_dim]}")
    if float(eps) <= 0 or float(mad_constant) <= 0:
        raise ValueError("eps and mad_constant must be positive")
    observed = observed_margin.detach().float()
    peers = peer_margins.detach().float()
    _require_finite(observed, "observed_margin")
    _require_finite(peers, "peer_margins")
    center = midpoint_median(peers, dim=peer_dim)
    mad = midpoint_median((peers - center.unsqueeze(peer_dim)).abs(), dim=peer_dim)
    scale = float(mad_constant) * mad
    z = (observed - center) / (scale + float(eps))
    delta = torch.tanh(z).detach()
    _require_finite(delta, "delta")
    if return_stats:
        return delta, {
            "center": center.detach(),
            "mad": mad.detach(),
            "scale": scale.detach(),
            "z": z.detach(),
        }
    return delta


def transport_weights(
    native_weights: torch.Tensor,
    delta: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Apply one KL-antisymmetric transport to weights across groups."""
    if native_weights.ndim < 1 or delta.shape not in (native_weights.shape[:-1], native_weights.shape):
        raise ValueError(
            f"native weights {tuple(native_weights.shape)} and delta "
            f"{tuple(delta.shape)} are incompatible"
        )
    weights = native_weights.float()
    evidence = delta.detach().float()
    if evidence.shape == weights.shape[:-1]:
        evidence = evidence.unsqueeze(-1)
    _require_finite(weights, "native_weights")
    _require_finite(evidence, "delta")
    if (weights <= 0).any():
        raise ValueError("native DFG weights must be strictly positive")
    if float(alpha) == 0.0:
        # Preserve exact alpha=0 parity, rather than introducing log/exp roundoff.
        return weights.clone()
    transported = F.softmax(
        torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
        + float(alpha) * evidence,
        dim=-1,
    )
    _require_finite(transported, "transported_weights")
    return transported


def transport_pair(
    native_normal: torch.Tensor,
    native_abnormal: torch.Tensor,
    delta: torch.Tensor,
    alpha: float,
    transport_direction: str = V1_TRANSPORT_DIRECTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport both class weights with an explicit architecture direction."""
    if native_normal.shape != native_abnormal.shape:
        raise ValueError("normal and abnormal native weights must have the same shape")
    direction = str(transport_direction)
    if direction == V1_TRANSPORT_DIRECTION:
        normal_delta, abnormal_delta = -delta, delta
    elif direction == V2_TRANSPORT_DIRECTION:
        normal_delta, abnormal_delta = delta, -delta
    else:
        raise ValueError(f"unsupported CIR transport direction: {direction!r}")
    return (
        transport_weights(native_normal, normal_delta, alpha),
        transport_weights(native_abnormal, abnormal_delta, alpha),
    )


def select_gt_free_peers(
    stage_features: torch.Tensor,
    stage_margins: torch.Tensor,
    *,
    peer_count: int = PEER_COUNT,
    spatial_radius: int = 3,
) -> dict[str, torch.Tensor]:
    """Select deterministic normal-like peers from detached image features."""
    if stage_features.ndim != 4 or stage_margins.ndim != 3:
        raise ValueError("expected stage_features [S,B,P,D] and margins [S,B,P]")
    stages, batch, patches, _ = stage_features.shape
    if stage_margins.shape != (stages, batch, patches):
        raise ValueError("stage feature/margin shapes do not match")
    if int(peer_count) < 1:
        raise ValueError("peer_count must be positive")
    side = int(round(patches ** 0.5))
    if side * side != patches:
        raise ValueError(f"peer selection requires a square patch grid, got P={patches}")
    if int(spatial_radius) < 0:
        raise ValueError("spatial_radius must be non-negative")
    with torch.no_grad():
        features = stage_features.detach().float()
        margins = stage_margins.detach().float()
        _require_finite(features, "stage_features")
        _require_finite(margins, "stage_margins")
        shared = F.normalize(features.mean(dim=0), dim=-1)
        pooled_margins = margins.mean(dim=0)
        stage_centers = midpoint_median(pooled_margins, dim=-1)
        normal_like = pooled_margins <= stage_centers.unsqueeze(-1)
        similarity = torch.einsum("bpd,bqd->bpq", shared, shared)
        coords = torch.arange(patches, device=features.device)
        yy = torch.div(coords, side, rounding_mode="floor")
        xx = torch.remainder(coords, side)
        spatial_ok = torch.maximum(
            (yy[:, None] - yy[None, :]).abs(),
            (xx[:, None] - xx[None, :]).abs(),
        ) > int(spatial_radius)
        allowed = normal_like[:, None, :] & spatial_ok.unsqueeze(0)
        scores = similarity.masked_fill(~allowed, float("-inf"))
        order = torch.argsort(scores, dim=-1, descending=True, stable=True)
        candidate_count = allowed.sum(dim=-1)
        valid = candidate_count >= int(peer_count)
        peer_indices = order[..., : int(peer_count)]
        peer_indices = torch.where(valid.unsqueeze(-1), peer_indices, torch.zeros_like(peer_indices))
    return {
        "peer_indices": peer_indices.detach().long(),
        "valid": valid.detach(),
        "candidate_count": candidate_count.detach().long(),
        "normal_like": normal_like.detach(),
        "shared_features": shared.detach(),
    }


def gather_peer_values(values: torch.Tensor, peer_indices: torch.Tensor) -> torch.Tensor:
    """Gather ``[S,B,P]`` or ``[S,B,P,G]`` values at ``[B,P,K]`` peers."""
    if peer_indices.ndim != 3:
        raise ValueError("peer indices must have shape [B,P,K]")
    if values.ndim == 3:
        stages, batch, patches = values.shape
        if peer_indices.shape[:2] != (batch, patches):
            raise ValueError("peer indices do not match values geometry")
        index = peer_indices.clamp(0, patches - 1).unsqueeze(0).expand(stages, -1, -1, -1)
        source = values.unsqueeze(-1).expand(-1, -1, -1, peer_indices.shape[-1])
        return source.gather(2, index)
    if values.ndim == 4:
        stages, batch, patches, groups = values.shape
        if peer_indices.shape[:2] != (batch, patches):
            raise ValueError("peer indices do not match values geometry")
        index = peer_indices.clamp(0, patches - 1).unsqueeze(0).unsqueeze(-1).expand(
            stages, -1, -1, -1, groups
        )
        source = values.unsqueeze(3).expand(-1, -1, -1, peer_indices.shape[-1], -1)
        return source.gather(2, index)
    raise ValueError("values must be [S,B,P] or [S,B,P,G]")


def _canonicalize_score_inputs(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Normalize supported compact score shapes to stage-first canonical shapes."""
    if image_features.ndim == 3:
        image = image_features.float().unsqueeze(0)
        squeeze_stage = True
    elif image_features.ndim == 4:
        image = image_features.float()
        squeeze_stage = False
    else:
        raise ValueError("image_features must be [B,P,D] or [S,B,P,D]")
    stages, batch, patches, dim = image.shape
    if text_features.ndim == 3:
        text = text_features.float().unsqueeze(0).unsqueeze(0)
    elif text_features.ndim == 4:
        text = text_features.float().unsqueeze(0)
    elif text_features.ndim == 5:
        text = text_features.float()
    else:
        raise ValueError("text_features must be [G,D,C], [B,G,D,C], or [S,B,G,D,C]")
    if text.shape[-2] != dim:
        raise ValueError(f"text dimension {text.shape[-2]} does not match image dimension {dim}")
    groups, classes = int(text.shape[-3]), int(text.shape[-1])
    if groups < 1 or classes < 1:
        raise ValueError("text features must contain at least one group and class")
    if weights.ndim == 2:
        weight = weights.float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif weights.ndim == 3:
        if weights.shape[1] == groups and weights.shape[0] == batch:
            weight = weights.float().unsqueeze(0).unsqueeze(2)
        elif weights.shape[1] == patches and weights.shape[0] == batch:
            weight = weights.float().unsqueeze(0).unsqueeze(-1)
        else:
            raise ValueError("ambiguous 3D weights; expected [B,G,C] or [B,P,G]")
    elif weights.ndim == 4:
        if weights.shape[0] == batch and weights.shape[1] == patches:
            weight = weights.float().unsqueeze(0)
        elif weights.shape[0] == stages and weights.shape[1] == batch:
            weight = weights.float().unsqueeze(2)
        else:
            raise ValueError("4D weights must be [B,P,G,C] or [S,B,G,C]")
    elif weights.ndim == 5:
        weight = weights.float()
    else:
        raise ValueError("weights must be [B,G,C], [B,P,G,C], [S,B,G,C], or [S,B,P,G,C]")
    if weight.shape[-2] != groups:
        raise ValueError(f"weight group dimension {weight.shape[-2]} does not match text groups {groups}")
    if weight.shape[-1] == 1 and classes != 1:
        weight = weight.expand(*weight.shape[:-1], classes)
    if weight.shape[-1] != classes:
        raise ValueError(f"weight class dimension {weight.shape[-1]} does not match text classes {classes}")
    text = text.expand(stages, batch, groups, dim, classes)
    weight = weight.expand(stages, batch, patches, groups, classes)
    _require_finite(image, "image_features")
    _require_finite(text, "text_features")
    _require_finite(weight, "weights")
    return image, text, weight, squeeze_stage


def score_reference(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps: float = CIR_EPS,
) -> torch.Tensor:
    """CIR_REFERENCE: normalize the weighted text prototype before dotting."""
    image, text, weight, squeeze_stage = _canonicalize_score_inputs(image_features, text_features, weights)
    if float(eps) <= 0:
        raise ValueError("eps must be positive")
    weighted_text = torch.einsum("sbpgc,sbgdc->sbpdc", weight, text)
    prototype = weighted_text / torch.sqrt(weighted_text.square().sum(dim=-2, keepdim=True) + float(eps))
    scores = 10.0 * torch.einsum("sbpd,sbpdc->sbpc", image, prototype)
    return scores[0] if squeeze_stage else scores


def score_optimized(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    weights: torch.Tensor,
    *,
    eps: float = CIR_EPS,
) -> torch.Tensor:
    """CIR_OPTIMIZED exact score-space path using text Gram matrices."""
    image, text, weight, squeeze_stage = _canonicalize_score_inputs(image_features, text_features, weights)
    if float(eps) <= 0:
        raise ValueError("eps must be positive")
    patch_text_dot = torch.einsum("sbpd,sbgdc->sbpgc", image, text)
    numerator = torch.sum(weight * patch_text_dot, dim=-2)
    gram = torch.einsum("sbgdc,sbhdc->sbghc", text, text)
    denominator_sq = torch.einsum("sbpgc,sbghc,sbphc->sbpc", weight, gram, weight)
    denominator = torch.sqrt(denominator_sq.clamp_min(0.0) + float(eps))
    scores = 10.0 * numerator / denominator
    return scores[0] if squeeze_stage else scores


def cir_logits_from_native_weights(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    native_weights: torch.Tensor,
    delta: torch.Tensor,
    alpha: float,
    *,
    score_mode: str = "optimized",
    eps: float = CIR_EPS,
    transport_direction: str = V1_TRANSPORT_DIRECTION,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transport native DFG weights and score both classes."""
    if image_features.ndim == 3:
        image = image_features.unsqueeze(0)
        squeeze_stage = True
    else:
        image = image_features
        squeeze_stage = False
    stages, batch, patches, _ = image.shape
    if native_weights.ndim == 3:
        if native_weights.shape[0] == batch:
            native = native_weights.unsqueeze(0).unsqueeze(2)
        else:
            native = native_weights.unsqueeze(2)
    elif native_weights.ndim == 4:
        native = native_weights
        if native.shape[2] != patches:
            native = native.unsqueeze(2)
    elif native_weights.ndim == 5:
        native = native_weights
    else:
        raise ValueError("native_weights must be [S,B,G,2] or [S,B,P,G,2]")
    if native.shape[-1] != 2 or native.shape[0] != stages or native.shape[1] != batch:
        raise ValueError("native weight shape must be [S,B,G,2]")
    native = native.expand(stages, batch, patches, native.shape[-2], 2).float()
    groups = int(native.shape[-2])
    if delta.ndim == 2:
        if tuple(delta.shape) != (batch, patches):
            raise ValueError(f"patch delta must be [B,P], got {tuple(delta.shape)}")
        evidence = delta.unsqueeze(0).unsqueeze(-1).expand(stages, batch, patches, groups)
    elif delta.ndim == 4:
        evidence = delta
    else:
        raise ValueError("delta must be legacy [B,P] or contract [S,B,P,G]")
    if tuple(evidence.shape) != (stages, batch, patches, groups):
        raise ValueError(f"delta geometry mismatch: {tuple(evidence.shape)}")
    normal, abnormal = transport_pair(
        native[..., 0], native[..., 1], evidence, alpha,
        transport_direction=transport_direction,
    )
    transported = torch.stack([normal, abnormal], dim=-1)
    scorer = score_optimized if str(score_mode).lower() in {"optimized", "opt"} else score_reference
    scores = scorer(image, text_features, transported, eps=eps)
    native_scores = scorer(image, text_features, native, eps=eps)
    if squeeze_stage:
        return scores, native_scores
    return scores, native_scores


def peer_delta_from_native_margins(
    stage_features: torch.Tensor,
    native_margins: torch.Tensor,
    *,
    peer_count: int = PEER_COUNT,
    spatial_radius: int = 3,
    eps: float = CIR_EPS,
    mad_constant: float = MAD_CONSTANT,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build shared peers and robust per-stage/per-group evidence."""
    if stage_features.ndim != 4 or native_margins.ndim != 4:
        raise ValueError("expected stage_features [S,B,P,D] and native_margins [S,B,P,G]")
    if tuple(stage_features.shape[:3]) != tuple(native_margins.shape[:3]):
        raise ValueError("stage feature/margin geometries do not match")
    observed = native_margins.detach().float()
    selection_margins = observed.mean(dim=-1)
    peers = select_gt_free_peers(
        stage_features.detach(),
        selection_margins,
        peer_count=peer_count,
        spatial_radius=spatial_radius,
    )
    peer_margins = gather_peer_values(observed, peers["peer_indices"])
    delta, stats = robust_peer_delta(
        observed,
        peer_margins,
        eps=eps,
        mad_constant=mad_constant,
        peer_count=peer_count,
        peer_dim=-2,
        return_stats=True,
    )
    valid = peers["valid"]
    delta = torch.where(valid.unsqueeze(0).unsqueeze(-1), delta, torch.zeros_like(delta))
    stats.update({
        "peer_indices": peers["peer_indices"],
        "valid": valid,
        "candidate_count": peers["candidate_count"],
        "peer_margins": peer_margins,
        "observed_margin": observed,
        "selection_margins": selection_margins,
    })
    return delta.detach(), stats
