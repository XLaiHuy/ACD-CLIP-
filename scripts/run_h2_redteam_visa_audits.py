#!/usr/bin/env python3
"""Bounded source-only H2 red-team audits.

This script never touches Medical, MVTec, or target labels, and it never
trains. It probes the historical H2 E1/E5/E10/E15 adapter checkpoints on one
deterministic augmented VisA batch and independently re-evaluates the frozen
CIR-V2 formulas.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset import get_text_and_image_dataset
from h2_clean.contract import SafeImageAdapterAnchor, make_dataloader_generator, seed_everything
from model.adapter import ACDCLIP
from model.clip import create_model
from train import compute_hybrid_k_regularization
from utils import calculate_seg_loss, get_hybrid_soft_prompt_single_class_text_embedding


HISTORICAL_H2_ROOT = Path(
    "/home/ai4/caohuy/ACD-CLIP-base-new-phase1/runs/phase2b/"
    "phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_"
    "test6medical7to15_fromscratch"
)
EPOCHS = (1, 5, 10, 15)
H2_LAMBDA_KG = 0.01
H2_LAMBDA_K = 0.002
ANCHOR_LAMBDA = 0.001
IMG_SIZE = 518
BATCH_SIZE = 6
SEED = 0
PEER_COUNT = 8
SPATIAL_RADIUS = 3
ALPHA = 0.5
STRICT_ATOL = 3e-6
MAD_CONSTANT = 1.4826
CIR_EPS = 1e-6


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def midpoint_median(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    ordered = torch.sort(values, dim=dim).values
    count = int(ordered.shape[dim])
    if count % 2:
        return ordered.select(dim, count // 2)
    return (ordered.select(dim, count // 2 - 1) + ordered.select(dim, count // 2)) * 0.5


def frozen_select_peers(
    stage_features: torch.Tensor,
    stage_margins: torch.Tensor,
    peer_count: int = PEER_COUNT,
    spatial_radius: int = SPATIAL_RADIUS,
) -> dict[str, torch.Tensor]:
    features = stage_features.detach().float()
    margins = stage_margins.detach().float()
    _, _, patches, _ = features.shape
    side = int(round(patches ** 0.5))
    shared = F.normalize(features.mean(dim=0), dim=-1)
    pooled = margins.mean(dim=0)
    center = midpoint_median(pooled, dim=-1)
    normal_like = pooled <= center.unsqueeze(-1)
    similarity = torch.einsum("bpd,bqd->bpq", shared, shared)
    coords = torch.arange(patches, device=features.device)
    yy = torch.div(coords, side, rounding_mode="floor")
    xx = torch.remainder(coords, side)
    spatial_ok = torch.maximum(
        (yy[:, None] - yy[None, :]).abs(),
        (xx[:, None] - xx[None, :]).abs(),
    ) > int(spatial_radius)
    allowed = normal_like[:, None, :] & spatial_ok.unsqueeze(0)
    order = torch.argsort(
        similarity.masked_fill(~allowed, float("-inf")),
        dim=-1,
        descending=True,
        stable=True,
    )
    candidate_count = allowed.sum(dim=-1)
    valid = candidate_count >= int(peer_count)
    indices = order[..., : int(peer_count)]
    indices = torch.where(valid.unsqueeze(-1), indices, torch.zeros_like(indices))
    return {
        "peer_indices": indices.long(),
        "candidate_count": candidate_count.long(),
        "valid": valid,
        "normal_like": normal_like,
    }


def frozen_delta(
    stage_features: torch.Tensor,
    native_margins: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    selection = native_margins.detach().float().mean(dim=-1)
    peers = frozen_select_peers(stage_features, selection)
    indices = peers["peer_indices"]
    source = native_margins.detach().float()
    gather_index = indices.unsqueeze(0).unsqueeze(-1).expand(
        source.shape[0], -1, -1, -1, source.shape[-1]
    )
    peer_values = source.unsqueeze(3).expand(
        -1, -1, -1, indices.shape[-1], -1
    ).gather(2, gather_index)
    center = midpoint_median(peer_values, dim=-2)
    mad = midpoint_median(
        (peer_values - center.unsqueeze(-2)).abs(),
        dim=-2,
    )
    z = (source - center) / (MAD_CONSTANT * mad + CIR_EPS)
    delta = torch.tanh(z).detach()
    delta = torch.where(
        peers["valid"].unsqueeze(0).unsqueeze(-1),
        delta,
        torch.zeros_like(delta),
    )
    return delta, {
        "peer_indices": indices,
        "candidate_count": peers["candidate_count"],
        "valid": peers["valid"],
        "mad": mad,
        "z": z,
    }


def frozen_h2_dfg_weights(
    model: ACDCLIP,
    img_feat: torch.Tensor,
    group_text: torch.Tensor,
    group_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    v_gap = img_feat.mean(dim=1)
    v_ss2d = model.image_adapter["dfg_ss2d_branches"][group_index](img_feat)
    text_normal = group_text[..., 0]
    text_abnormal = group_text[..., 1]
    key = model.image_adapter["vision_text_k"][group_index]
    query = model.image_adapter["vision_text_q"][group_index]
    k_normal = key(text_normal)
    k_abnormal = key(text_abnormal)
    scale = (model.dfg_attn_dim ** 0.5) * model.dfg_attn_tau
    q_gap = query(v_gap)
    q_ss2d = query(v_ss2d)
    if model.dfg_weight_residual_fp32:
        q_gap = q_gap.float()
        q_ss2d = q_ss2d.float()
        k_normal = k_normal.float()
        k_abnormal = k_abnormal.float()
    scores_gap_normal = torch.einsum("bd,bnd->bn", q_gap, k_normal) / scale
    scores_gap_abnormal = torch.einsum("bd,bnd->bn", q_gap, k_abnormal) / scale
    scores_ss2d_normal = torch.einsum("bd,bnd->bn", q_ss2d, k_normal) / scale
    scores_ss2d_abnormal = torch.einsum("bd,bnd->bn", q_ss2d, k_abnormal) / scale
    normal = (1.0 - model.dfg_beta) * F.softmax(scores_gap_normal, dim=1)
    normal = normal + model.dfg_beta * F.softmax(scores_ss2d_normal, dim=1)
    abnormal = (1.0 - model.dfg_beta) * F.softmax(scores_gap_abnormal, dim=1)
    abnormal = abnormal + model.dfg_beta * F.softmax(scores_ss2d_abnormal, dim=1)
    return normal.to(dtype=text_normal.dtype), abnormal.to(dtype=text_abnormal.dtype)


def frozen_native(
    model: ACDCLIP,
    seg_features: torch.Tensor,
    text_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_text = text_features.permute(1, 0, 2, 3)
    weights = []
    logits = []
    for stage in range(int(seg_features.shape[0])):
        normal, abnormal = frozen_h2_dfg_weights(
            model, seg_features[stage], group_text, stage
        )
        weights.append(torch.stack([normal, abnormal], dim=-1))
        normal_text = torch.einsum("bn,bnd->bd", normal, group_text[..., 0])
        abnormal_text = torch.einsum("bn,bnd->bd", abnormal, group_text[..., 1])
        fused = torch.stack(
            [F.normalize(normal_text, dim=-1), F.normalize(abnormal_text, dim=-1)],
            dim=-1,
        )
        logits.append(torch.matmul(10.0 * seg_features[stage], fused))
    return torch.stack(weights), torch.stack(logits)


def frozen_score(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    stages, batch, patches, _ = image_features.shape
    groups = int(text_features.shape[1])
    text = text_features.unsqueeze(0).expand(stages, batch, groups, -1, -1).float()
    weight = weights.float()
    if weight.ndim == 4:
        weight = weight.unsqueeze(2).expand(stages, batch, patches, groups, 2)
    patch_text_dot = torch.einsum("sbpd,sbgdc->sbpgc", image_features.float(), text)
    numerator = torch.sum(weight * patch_text_dot, dim=-2)
    gram = torch.einsum("sbgdc,sbhdc->sbghc", text, text)
    denominator_sq = torch.einsum("sbpgc,sbghc,sbphc->sbpc", weight, gram, weight)
    denominator = torch.sqrt(denominator_sq.clamp_min(0.0) + CIR_EPS)
    return 10.0 * numerator / denominator


def frozen_transport(
    native_weights: torch.Tensor,
    delta: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    native = native_weights.float().unsqueeze(2)
    normal = F.softmax(
        torch.log(native[..., 0].clamp_min(torch.finfo(torch.float32).tiny))
        + float(alpha) * delta.float(),
        dim=-1,
    )
    abnormal = F.softmax(
        torch.log(native[..., 1].clamp_min(torch.finfo(torch.float32).tiny))
        - float(alpha) * delta.float(),
        dim=-1,
    )
    return normal, abnormal


def make_model(device: torch.device) -> ACDCLIP:
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=IMG_SIZE,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    clip_model.set_grad_checkpointing(True)
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=3,
        image_adapt_weight=0.2,
        conv_lora_rank=8,
        conv_lora_alpha=2.0,
        conv_kernel_size_list=[3, 5],
        text_adapt_weight=0.2,
        lora_rank=16,
        lora_alpha=2.0,
        dfg_mode="attn",
        dfg_attn_dim=256,
        dfg_attn_tau=8.0,
        use_ss2d_dfg=True,
        dfg_gamma_max=0.2,
        dfg_ss2d_fusion="weight_residual",
        dfg_beta=0.1,
        dfg_beta_schedule="warmup010",
        dfg_beta_target=0.1,
        dfg_beta_current=0.1,
        dfg_weight_residual_fp32=True,
        use_soft_prompt=False,
        soft_prompt_ctx_len=4,
        soft_prompt_init="phrase",
        soft_prompt_init_phrase="a photo of a",
    ).to(device)
    model.eval()
    model.use_hybrid_soft_prompt = True
    model.prompt_mode = "hybrid"
    model.requires_grad_(False)
    return model


def load_h2_checkpoint(model: ACDCLIP, epoch: int) -> dict[str, Any]:
    path = HISTORICAL_H2_ROOT / f"adapter_{epoch}.pth"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.image_adapter.load_state_dict(payload["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(payload["text_adapter"], strict=True)
    if "soft_prompt" in payload:
        model.soft_prompt.load_state_dict(payload["soft_prompt"], strict=True)
    model.dfg_beta = float(payload["dfg_beta_current"])
    model.hybrid_alpha_current = float(payload["hybrid_alpha_current"])
    model.hybrid_alpha_max = float(payload["hybrid_alpha_max"])
    model.soft_prompt_freeze_epochs = int(payload["soft_prompt_freeze_epochs"])
    return payload


def make_fixed_batch() -> tuple[dict[str, Any], dict[str, Any]]:
    seed_everything(SEED, deterministic_algorithms=True)
    dataset = get_text_and_image_dataset("VisA", IMG_SIZE, "train")
    generator = make_dataloader_generator(SEED + 104729)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )
    batch = next(iter(loader))
    metadata = {
        "dataset": "VisA",
        "stage": "train",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "num_workers": 0,
        "file_names": list(batch["file_name"]),
        "image_sha256": tensor_sha256(batch["image"]),
        "mask_sha256": tensor_sha256(batch["mask"]),
        "class_names": list(batch["class_name"]),
        "labels": batch["label"].tolist(),
    }
    return batch, metadata


def build_text_features(
    model: ACDCLIP,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_by_class = {}
    kg_losses = []
    k_losses = []
    with torch.no_grad():
        for class_name in sorted(set(batch["class_name"])):
            text, kg_loss, _, components = get_hybrid_soft_prompt_single_class_text_embedding(
                model,
                "VisA",
                class_name,
                device,
                return_kg=True,
                return_components=True,
            )
            k_loss, _ = compute_hybrid_k_regularization(
                model,
                components["hard_text"],
                components["soft_text"],
                float(model.hybrid_alpha_current),
            )
            text_by_class[class_name] = text
            kg_losses.append(kg_loss)
            k_losses.append(k_loss)
    text_features = torch.stack(
        [text_by_class[name] for name in batch["class_name"]],
        dim=0,
    ).permute(1, 0, 2, 3)
    return text_features, torch.stack(kg_losses).mean(), torch.stack(k_losses).mean()


def family_for(name: str) -> str:
    for prefix in ("dfg_ss2d_branches", "dfg_raw_gamma", "vision_text_q", "vision_text_k"):
        if name.startswith(prefix):
            return prefix
    return "other_image_adapter"


def norm_from_grads(
    names: list[str],
    grads: tuple[torch.Tensor | None, ...],
    selected: set[str] | None = None,
) -> float:
    total = 0.0
    for name, grad in zip(names, grads):
        if selected is not None and name not in selected:
            continue
        if grad is not None:
            total += float(grad.detach().float().square().sum().item())
    return total ** 0.5


def anchor_measurement(
    model: ACDCLIP,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    anchor_path = HISTORICAL_H2_ROOT / "adapter_1.pth"
    anchor = SafeImageAdapterAnchor.from_checkpoint(anchor_path, device)
    image_params = {
        name: parameter
        for name, parameter in sorted(model.image_adapter.named_parameters())
    }
    names = list(image_params)
    params = [image_params[name] for name in names]
    rows: dict[str, Any] = {}
    for epoch in (5, 10, 15):
        load_h2_checkpoint(model, epoch)
        model.image_adapter.requires_grad_(True)
        model.text_adapter.requires_grad_(False)
        model.soft_prompt.requires_grad_(False)
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)
        text_features, kg_loss, k_loss = build_text_features(model, batch, device)
        seg_tokens, det_tokens = model(image)
        seg_features = torch.stack(seg_tokens, dim=0)
        det_features = torch.stack(det_tokens, dim=0)
        cls_pred = torch.stack(
            [
                torch.matmul(
                    det_features[group].unsqueeze(1),
                    text_features[group],
                ).squeeze(1)
                for group in range(det_features.shape[0])
            ],
            dim=0,
        ).mean(dim=0)
        cls_loss = F.cross_entropy(cls_pred, labels)
        seg_pred = model.vision_text_fusion_gate_seg(
            seg_features, text_features, cir_training=False
        )
        seg_loss = calculate_seg_loss(seg_pred, mask)
        task_loss = cls_loss + seg_loss + H2_LAMBDA_KG * kg_loss + H2_LAMBDA_K * k_loss
        raw_anchor_loss = anchor.loss(model.image_adapter)
        task_grads = torch.autograd.grad(
            task_loss, params, retain_graph=True, allow_unused=True
        )
        anchor_grads = torch.autograd.grad(
            raw_anchor_loss, params, retain_graph=False, allow_unused=True
        )
        task_norm = norm_from_grads(names, task_grads)
        anchor_norm = norm_from_grads(names, anchor_grads)
        global_raw_ratio = anchor_norm / max(task_norm, 1e-12)
        families = {}
        family_names = sorted({family_for(name) for name in names})
        for family in family_names:
            selected = {name for name in names if family_for(name) == family}
            task_family_norm = norm_from_grads(names, task_grads, selected)
            anchor_family_norm = norm_from_grads(names, anchor_grads, selected)
            raw_ratio = anchor_family_norm / max(task_family_norm, 1e-12)
            families[family] = {
                "task_grad_norm": task_family_norm,
                "anchor_grad_raw_norm": anchor_family_norm,
                "raw_gradient_ratio": raw_ratio,
                "lambda_times_ratio": ANCHOR_LAMBDA * raw_ratio,
            }
        rows[str(epoch)] = {
            "checkpoint": str(HISTORICAL_H2_ROOT / f"adapter_{epoch}.pth"),
            "checkpoint_sha256": hashlib.sha256(
                (HISTORICAL_H2_ROOT / f"adapter_{epoch}.pth").read_bytes()
            ).hexdigest(),
            "raw_anchor_loss": float(raw_anchor_loss.detach().item()),
            "task_loss": float(task_loss.detach().item()),
            "cls_loss": float(cls_loss.detach().item()),
            "seg_loss": float(seg_loss.detach().item()),
            "kg_loss": float(kg_loss.detach().item()),
            "k_loss": float(k_loss.detach().item()),
            "global": {
                "task_grad_norm": task_norm,
                "anchor_grad_raw_norm": anchor_norm,
                "raw_gradient_ratio": global_raw_ratio,
                "lambda": ANCHOR_LAMBDA,
                "lambda_times_ratio": ANCHOR_LAMBDA * global_raw_ratio,
            },
            "families": families,
        }
        model.zero_grad(set_to_none=True)
        del task_grads, anchor_grads, task_loss, raw_anchor_loss
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ratios = [rows[str(epoch)]["global"]["lambda_times_ratio"] for epoch in (5, 10, 15)]
    raw_ratios = [rows[str(epoch)]["global"]["raw_gradient_ratio"] for epoch in (5, 10, 15)]
    if not all(np.isfinite(ratios + raw_ratios)):
        classification = "UNKNOWN"
    elif max(ratios) < 1e-3:
        classification = "EFFECTIVELY_INACTIVE"
    elif max(ratios) > 1.0:
        classification = "EXCESSIVE"
    elif min(ratios) >= 1e-3:
        classification = "ACTIVE"
    else:
        classification = "UNKNOWN"
    rows["classification"] = {
        "anchor_strength": classification,
        "lambda_changed": "NO",
        "global_lambda_times_ratio_min": min(ratios),
        "global_lambda_times_ratio_max": max(ratios),
        "source_only_preregistered_scaling_rule": (
            "If classified EFFECTIVELY_INACTIVE, set the future source-only "
            "anchor lambda to 0.1 divided by the median E5/E10/E15 raw global "
            "gradient ratio; do not apply this rule in this red-team run."
        ),
    }
    return rows


def cir_measurement(
    model: ACDCLIP,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    load_h2_checkpoint(model, 10)
    model.requires_grad_(False)
    image = batch["image"].to(device)
    with torch.no_grad():
        text_features, _, _ = build_text_features(model, batch, device)
        seg_tokens, _ = model(image)
        seg_features = torch.stack(seg_tokens, dim=0)
        native_weights, native_logits = model._h2_cir_native_weights_logits(
            seg_features, text_features
        )
        group_text = text_features.permute(1, 0, 2, 3)
        visual = F.normalize(seg_features.float(), dim=-1)
        prompts = F.normalize(group_text.float(), dim=-2)
        similarities = torch.einsum("sbpd,bgdc->sbpgc", visual, prompts)
        margins = similarities[..., 1] - similarities[..., 0]
        cir_v2 = __import__("h2_clean.cir_v2", fromlist=[
            "cir_logits_from_native_weights",
            "peer_delta_from_native_margins",
            "transport_pair",
        ])
        candidate_delta, candidate_stats = cir_v2.peer_delta_from_native_margins(
            seg_features.detach(),
            margins.detach(),
            peer_count=PEER_COUNT,
            spatial_radius=SPATIAL_RADIUS,
        )
        candidate_logits, candidate_native_scores = cir_v2.cir_logits_from_native_weights(
            seg_features,
            group_text,
            native_weights,
            candidate_delta,
            ALPHA,
            score_mode="optimized",
            eps=CIR_EPS,
            transport_direction="abnormal_minus_normal_plus",
        )
        candidate_probability = model.vision_text_fusion_gate_seg(
            seg_features,
            text_features,
            img_size=IMG_SIZE,
            cir_training=True,
            cir_alpha=ALPHA,
            cir_peer_count=PEER_COUNT,
            cir_spatial_radius=SPATIAL_RADIUS,
        )
        reference_weights, reference_native_logits = frozen_native(
            model, seg_features, text_features
        )
        reference_delta, reference_stats = frozen_delta(seg_features, margins)
        reference_normal, reference_abnormal = frozen_transport(
            reference_weights, reference_delta, ALPHA
        )
        reference_transport = torch.stack(
            [reference_normal, reference_abnormal], dim=-1
        )
        reference_logits = frozen_score(
            seg_features, group_text, reference_transport
        )
        reference_native_scores = frozen_score(
            seg_features, group_text, reference_weights
        )
        reference_maps = reference_logits.permute(0, 1, 3, 2).reshape(
            3, image.shape[0], 2, 37, 37
        )
        reference_maps = F.interpolate(
            reference_maps.reshape(-1, 2, 37, 37),
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=True,
        ).reshape(3, image.shape[0], 2, IMG_SIZE, IMG_SIZE)
        reference_probability = F.softmax(reference_maps.mean(dim=0), dim=1)
        native_patch = native_weights.unsqueeze(2).expand(
            -1, -1, seg_features.shape[2], -1, -1
        )
        candidate_normal, candidate_abnormal = cir_v2.transport_pair(
            native_patch[..., 0],
            native_patch[..., 1],
            candidate_delta,
            ALPHA,
            transport_direction="abnormal_minus_normal_plus",
        )
        candidate_transport = torch.stack(
            [candidate_normal, candidate_abnormal], dim=-1
        )
        peer_equal = torch.equal(
            candidate_stats["peer_indices"], reference_stats["peer_indices"]
        )
        max_diffs = {
            "native_weight_max_abs_diff": float(
                (native_weights - reference_weights).abs().max().item()
            ),
            "native_logit_max_abs_diff": float(
                (native_logits - reference_native_logits).abs().max().item()
            ),
            "transport_weight_max_abs_diff": float(
                (candidate_transport - reference_transport).abs().max().item()
            ),
            "delta_max_abs_diff": float(
                (candidate_delta - reference_delta).abs().max().item()
            ),
            "native_score_max_abs_diff": float(
                (candidate_native_scores - reference_native_scores).abs().max().item()
            ),
            "cir_logit_max_abs_diff": float(
                (candidate_logits - reference_logits).abs().max().item()
            ),
            "final_probability_max_abs_diff": float(
                (candidate_probability - reference_probability).abs().max().item()
            ),
        }
        return {
            "checkpoint": str(HISTORICAL_H2_ROOT / "adapter_10.pth"),
            "alpha": ALPHA,
            "strict_atol": STRICT_ATOL,
            **max_diffs,
            "peer_index_equal_including_ties": peer_equal,
            "candidate_valid_fraction": float(
                candidate_stats["valid"].float().mean().item()
            ),
            "reference_valid_fraction": float(
                reference_stats["valid"].float().mean().item()
            ),
            "candidate_candidate_count_mean": float(
                candidate_stats["candidate_count"].float().mean().item()
            ),
            "reference_candidate_count_mean": float(
                reference_stats["candidate_count"].float().mean().item()
            ),
            "parity_pass": bool(
                peer_equal
                and all(value <= STRICT_ATOL for value in max_diffs.values())
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cir", "anchor", "all"], default="all")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not HISTORICAL_H2_ROOT.is_dir():
        raise FileNotFoundError(HISTORICAL_H2_ROOT)
    for epoch in EPOCHS:
        if not (HISTORICAL_H2_ROOT / f"adapter_{epoch}.pth").is_file():
            raise FileNotFoundError(HISTORICAL_H2_ROOT / f"adapter_{epoch}.pth")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch, batch_metadata = make_fixed_batch()
    model = make_model(device)
    report: dict[str, Any] = {
        "scope": "VisA train only; no Medical/MVTec/target labels; no training",
        "environment": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "device": str(device),
            "torch": torch.__version__,
        },
        "fixed_batch": batch_metadata,
    }
    if args.mode in ("cir", "all"):
        report["cir"] = cir_measurement(model, batch, device)
    if args.mode in ("anchor", "all"):
        report["anchor"] = anchor_measurement(model, batch, device)
    output = REPO / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
