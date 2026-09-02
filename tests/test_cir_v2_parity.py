import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from h2_clean.cir_v2 import (
    V2_TRANSPORT_DIRECTION,
    cir_logits_from_native_weights,
    midpoint_median,
    peer_delta_from_native_margins,
    robust_peer_delta,
    select_gt_free_peers,
    transport_pair,
    transport_weights,
)
from model.adapter import ACDCLIP


class TinySS2DBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(6, 6)

    def forward(self, tokens):
        return self.proj(tokens.mean(dim=1))

class TinyH2(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_groups = 3
        self.dfg_mode = "attn"
        self.dfg_attn_dim = 4
        self.dfg_attn_tau = 8.0
        self.use_ss2d_dfg = True
        self.dfg_ss2d_fusion = "weight_residual"
        self.dfg_weight_residual_fp32 = True
        self.dfg_beta = 0.10
        self.image_adapter = nn.ModuleDict({
            "vision_text_q": nn.ModuleList([nn.Linear(6, 4) for _ in range(3)]),
            "vision_text_k": nn.ModuleList([nn.Linear(6, 4) for _ in range(3)]),
            "dfg_ss2d_branches": nn.ModuleList([TinySS2DBranch() for _ in range(3)]),
        })


    def _h2_cir_dfg_weights(self, img_feat, group_text, group_index):
        return ACDCLIP._h2_cir_dfg_weights(self, img_feat, group_text, group_index)

    def _h2_cir_native_weights_logits(self, seg_features, text_features):
        return ACDCLIP._h2_cir_native_weights_logits(
            self,
            seg_features,
            text_features,
        )
def frozen_midpoint_median(values, dim=-1):
    ordered = torch.sort(values, dim=dim).values
    count = int(ordered.shape[dim])
    if count % 2:
        return ordered.select(dim, count // 2)
    return (ordered.select(dim, count // 2 - 1) + ordered.select(dim, count // 2)) * 0.5


def frozen_select_peers(stage_features, stage_margins, peer_count=8, spatial_radius=3):
    features = stage_features.detach().float()
    margins = stage_margins.detach().float()
    shared = F.normalize(features.mean(dim=0), dim=-1)
    pooled_margins = margins.mean(dim=0)
    centers = frozen_midpoint_median(pooled_margins, dim=-1)
    normal_like = pooled_margins <= centers.unsqueeze(-1)
    similarity = torch.einsum("bpd,bqd->bpq", shared, shared)
    patches = int(stage_features.shape[2])
    side = int(round(patches ** 0.5))
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
    indices = order[..., : int(peer_count)]
    indices = torch.where(valid.unsqueeze(-1), indices, torch.zeros_like(indices))
    return {
        "peer_indices": indices.long(),
        "valid": valid,
        "candidate_count": candidate_count.long(),
    }


def frozen_peer_delta(stage_features, native_margins, peer_count=8, spatial_radius=3):
    observed = native_margins.detach().float()
    peers = frozen_select_peers(
        stage_features,
        observed.mean(dim=-1),
        peer_count=peer_count,
        spatial_radius=spatial_radius,
    )
    index = peers["peer_indices"].unsqueeze(0).unsqueeze(-1).expand(
        observed.shape[0], -1, -1, -1, observed.shape[-1]
    )
    source = observed.unsqueeze(3).expand(
        -1, -1, -1, peers["peer_indices"].shape[-1], -1
    )
    peer_values = source.gather(2, index)
    center = frozen_midpoint_median(peer_values, dim=-2)
    mad = frozen_midpoint_median(
        (peer_values - center.unsqueeze(-2)).abs(),
        dim=-2,
    )
    z = (observed - center) / (1.4826 * mad + 1e-6)
    delta = torch.tanh(z).detach()
    delta = torch.where(
        peers["valid"].unsqueeze(0).unsqueeze(-1),
        delta,
        torch.zeros_like(delta),
    )
    return delta, peers


def frozen_transport(native_weights, delta, alpha):
    weights = native_weights.float()
    evidence = delta.detach().float()
    if evidence.shape == weights.shape[:-1]:
        evidence = evidence.unsqueeze(-1)
    return F.softmax(
        torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
        + float(alpha) * evidence,
        dim=-1,
    )


def frozen_h2_dfg_weights(model, img_feat, group_text, group_index):
    v_gap = img_feat.mean(dim=1)
    v_ss2d = None
    if model.use_ss2d_dfg:
        v_ss2d = model.image_adapter["dfg_ss2d_branches"][group_index](img_feat)
    text_normal = group_text[..., 0]
    text_abnormal = group_text[..., 1]
    k_normal = model.image_adapter["vision_text_k"][group_index](text_normal)
    k_abnormal = model.image_adapter["vision_text_k"][group_index](text_abnormal)
    scale = (model.dfg_attn_dim ** 0.5) * model.dfg_attn_tau
    if model.use_ss2d_dfg and model.dfg_ss2d_fusion == "weight_residual":
        q_gap = model.image_adapter["vision_text_q"][group_index](v_gap)
        q_ss2d = model.image_adapter["vision_text_q"][group_index](v_ss2d)
        if model.dfg_weight_residual_fp32:
            q_gap_for_attn = q_gap.float()
            q_ss2d_for_attn = q_ss2d.float()
            k_normal_for_attn = k_normal.float()
            k_abnormal_for_attn = k_abnormal.float()
        else:
            q_gap_for_attn = q_gap
            q_ss2d_for_attn = q_ss2d
            k_normal_for_attn = k_normal
            k_abnormal_for_attn = k_abnormal
        scores_gap_normal = torch.einsum("bd,bnd->bn", q_gap_for_attn, k_normal_for_attn) / scale
        scores_gap_abnormal = torch.einsum("bd,bnd->bn", q_gap_for_attn, k_abnormal_for_attn) / scale
        scores_ss2d_normal = torch.einsum("bd,bnd->bn", q_ss2d_for_attn, k_normal_for_attn) / scale
        scores_ss2d_abnormal = torch.einsum("bd,bnd->bn", q_ss2d_for_attn, k_abnormal_for_attn) / scale
        weights_normal = (1.0 - model.dfg_beta) * F.softmax(scores_gap_normal, dim=1)
        weights_normal = weights_normal + model.dfg_beta * F.softmax(scores_ss2d_normal, dim=1)
        weights_abnormal = (1.0 - model.dfg_beta) * F.softmax(scores_gap_abnormal, dim=1)
        weights_abnormal = weights_abnormal + model.dfg_beta * F.softmax(scores_ss2d_abnormal, dim=1)
        return weights_normal.to(dtype=text_normal.dtype), weights_abnormal.to(dtype=text_abnormal.dtype)
    q = model.image_adapter["vision_text_q"][group_index](v_gap)
    weights_normal = F.softmax(torch.einsum("bd,bnd->bn", q, k_normal) / scale, dim=1)
    weights_abnormal = F.softmax(torch.einsum("bd,bnd->bn", q, k_abnormal) / scale, dim=1)
    return weights_normal, weights_abnormal


def frozen_native(model, seg_features, text_features):
    group_text = text_features.permute(1, 0, 2, 3)
    weights = []
    logits = []
    for stage in range(int(seg_features.shape[0])):
        normal, abnormal = frozen_h2_dfg_weights(model, seg_features[stage], group_text, stage)
        weights.append(torch.stack([normal, abnormal], dim=-1))
        fused_normal = torch.einsum("bn,bnd->bd", normal, group_text[..., 0])
        fused_abnormal = torch.einsum("bn,bnd->bd", abnormal, group_text[..., 1])
        fused = torch.stack(
            [F.normalize(fused_normal, dim=-1), F.normalize(fused_abnormal, dim=-1)],
            dim=-1,
        )
        logits.append(torch.matmul(10.0 * seg_features[stage], fused))
    return torch.stack(weights, dim=0), torch.stack(logits, dim=0)


def frozen_score(image_features, text_features, weights, eps=1e-6):
    image = image_features.float()
    text = text_features.float().unsqueeze(0)
    weight = weights.float()
    if weight.ndim == 4:
        weight = weight.unsqueeze(2)
    if weight.ndim != 5:
        raise ValueError("frozen score weights must be [S,B,G,C] or [S,B,P,G,C]")
    weight = weight.expand(
        image.shape[0], image.shape[1], image.shape[2], weight.shape[-2], weight.shape[-1]
    )
    patch_text_dot = torch.einsum("sbpd,sbgdc->sbpgc", image, text)
    numerator = torch.sum(weight * patch_text_dot, dim=-2)
    gram = torch.einsum("sbgdc,sbhdc->sbghc", text, text)
    denominator_sq = torch.einsum("sbpgc,sbghc,sbphc->sbpc", weight, gram, weight)
    denominator = torch.sqrt(denominator_sq.clamp_min(0.0) + float(eps))
    return 10.0 * numerator / denominator


def make_case(seed=311):
    torch.manual_seed(seed)
    model = TinyH2()
    seg_features = torch.randn(3, 2, 49, 6)
    text_features = torch.randn(3, 2, 6, 2)
    native_weights, native_logits = ACDCLIP._h2_cir_native_weights_logits(model, seg_features, text_features)
    group_text = text_features.permute(1, 0, 2, 3)
    visual = F.normalize(seg_features.float(), dim=-1)
    prompts = F.normalize(group_text.float(), dim=-2)
    margins = torch.einsum("sbpd,bgdc->sbpgc", visual, prompts)[..., 1]
    margins = margins - torch.einsum("sbpd,bgdc->sbpgc", visual, prompts)[..., 0]
    return model, seg_features, text_features, native_weights, native_logits, margins


def test_cir_alpha05_reference_output_parity_synthetic():
    _, seg, text, native_weights, _, margins = make_case()
    clean_delta, clean_stats = peer_delta_from_native_margins(seg.detach(), margins.detach())
    frozen_delta, frozen_peers = frozen_peer_delta(seg, margins)
    torch.testing.assert_close(clean_delta, frozen_delta, rtol=0.0, atol=0.0)
    assert torch.equal(clean_stats["peer_indices"], frozen_peers["peer_indices"])
    clean_logits, clean_native_scores = cir_logits_from_native_weights(
        seg, text.permute(1, 0, 2, 3), native_weights, clean_delta, 0.5,
        score_mode="optimized", eps=1e-6,
        transport_direction=V2_TRANSPORT_DIRECTION,
    )
    normal, abnormal = frozen_transport(native_weights[..., 0].unsqueeze(2).expand(-1, -1, 49, -1), clean_delta, 0.5), frozen_transport(
        native_weights[..., 1].unsqueeze(2).expand(-1, -1, 49, -1), -clean_delta, 0.5
    )
    frozen_weights = torch.stack([normal, abnormal], dim=-1)
    frozen_scores = frozen_score(seg, text.permute(1, 0, 2, 3), frozen_weights)
    frozen_native_scores = frozen_score(
        seg, text.permute(1, 0, 2, 3), native_weights
    )
    torch.testing.assert_close(clean_logits, frozen_scores, rtol=0.0, atol=3e-6)
    torch.testing.assert_close(clean_native_scores, frozen_native_scores, rtol=0.0, atol=3e-6)


def test_cir_alpha05_reference_grad_parity_synthetic():
    clean_model, seg, text, native_weights, _, margins = make_case(seed=312)
    ref_model = copy.deepcopy(clean_model)
    clean_seg = seg.clone().requires_grad_(True)
    ref_seg = seg.clone().requires_grad_(True)
    clean_native, _ = ACDCLIP._h2_cir_native_weights_logits(clean_model, clean_seg, text)
    ref_native, _ = frozen_native(ref_model, ref_seg, text)
    clean_delta, _ = peer_delta_from_native_margins(clean_seg.detach(), margins.detach())
    ref_delta, _ = frozen_peer_delta(ref_seg, margins)
    clean_scores, _ = cir_logits_from_native_weights(
        clean_seg, text.permute(1, 0, 2, 3), clean_native, clean_delta, 0.5,
        score_mode="optimized", eps=1e-6,
        transport_direction=V2_TRANSPORT_DIRECTION,
    )
    ref_weights_normal, ref_weights_abnormal = frozen_transport(
        ref_native[..., 0].unsqueeze(2).expand(-1, -1, 49, -1),
        ref_delta,
        0.5,
    ), frozen_transport(
        ref_native[..., 1].unsqueeze(2).expand(-1, -1, 49, -1),
        -ref_delta,
        0.5,
    )
    ref_scores = frozen_score(
        ref_seg,
        text.permute(1, 0, 2, 3),
        torch.stack([ref_weights_normal, ref_weights_abnormal], dim=-1),
    )
    clean_grads = torch.autograd.grad(
        clean_scores.sum(),
        tuple(clean_model.parameters()) + (clean_seg,),
        allow_unused=True,
    )
    ref_grads = torch.autograd.grad(
        ref_scores.sum(),
        tuple(ref_model.parameters()) + (ref_seg,),
        allow_unused=True,
    )
    for left, right in zip(clean_grads, ref_grads):
        if left is None or right is None:
            assert left is None and right is None
        else:
            torch.testing.assert_close(left, right, rtol=0.0, atol=3e-6)


def test_native_h2_dfg_weights_and_logits_parity():
    model, seg, text, clean_weights, clean_logits, _ = make_case(seed=313)
    frozen_weights, frozen_logits = frozen_native(model, seg, text)
    torch.testing.assert_close(clean_weights, frozen_weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(clean_logits, frozen_logits, rtol=0.0, atol=0.0)
    assert torch.all(clean_weights > 0)
    assert torch.allclose(clean_weights.sum(dim=-2), torch.ones_like(clean_weights.sum(dim=-2)))


def test_transport_normal_abnormal_weights_parity_v2():
    _, _, _, native_weights, _, _ = make_case(seed=314)
    delta = torch.linspace(-0.9, 0.9, 3 * 2 * 49 * 3).reshape(3, 2, 49, 3)
    clean_normal, clean_abnormal = transport_pair(
        native_weights[..., 0].unsqueeze(2).expand(-1, -1, 49, -1),
        native_weights[..., 1].unsqueeze(2).expand(-1, -1, 49, -1),
        delta,
        0.5,
        transport_direction=V2_TRANSPORT_DIRECTION,
    )
    frozen_normal = frozen_transport(
        native_weights[..., 0].unsqueeze(2).expand(-1, -1, 49, -1), delta, 0.5
    )
    frozen_abnormal = frozen_transport(
        native_weights[..., 1].unsqueeze(2).expand(-1, -1, 49, -1), -delta, 0.5
    )
    torch.testing.assert_close(clean_normal, frozen_normal, rtol=0.0, atol=0.0)
    torch.testing.assert_close(clean_abnormal, frozen_abnormal, rtol=0.0, atol=0.0)


def test_delta_parity_v2():
    _, seg, _, _, _, margins = make_case(seed=315)
    clean_delta, _ = peer_delta_from_native_margins(seg, margins)
    frozen_delta, _ = frozen_peer_delta(seg, margins)
    torch.testing.assert_close(clean_delta, frozen_delta, rtol=0.0, atol=0.0)


def test_final_cir_logits_strict_reference_parity():
    _, seg, text, native_weights, _, margins = make_case(seed=316)
    delta, _ = peer_delta_from_native_margins(seg, margins)
    clean_scores, _ = cir_logits_from_native_weights(
        seg, text.permute(1, 0, 2, 3), native_weights, delta, 0.5,
        score_mode="optimized", eps=1e-6,
        transport_direction=V2_TRANSPORT_DIRECTION,
    )
    normal = frozen_transport(
        native_weights[..., 0].unsqueeze(2).expand(-1, -1, 49, -1), delta, 0.5
    )
    abnormal = frozen_transport(
        native_weights[..., 1].unsqueeze(2).expand(-1, -1, 49, -1), -delta, 0.5
    )
    reference = frozen_score(
        seg, text.permute(1, 0, 2, 3), torch.stack([normal, abnormal], dim=-1)
    )
    torch.testing.assert_close(clean_scores, reference, rtol=0.0, atol=3e-6)


def test_peer_indices_parity_including_constructed_ties():
    features = torch.zeros(3, 1, 49, 6)
    features[..., 0] = 1.0
    margins = torch.zeros(3, 1, 49)
    clean = select_gt_free_peers(features, margins, peer_count=8, spatial_radius=0)
    frozen = frozen_select_peers(features, margins, peer_count=8, spatial_radius=0)
    assert torch.equal(clean["peer_indices"], frozen["peer_indices"])
    assert torch.equal(clean["valid"], frozen["valid"])
    assert torch.equal(clean["peer_indices"][0, 0], torch.arange(1, 9))
