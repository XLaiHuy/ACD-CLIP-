"""CIR_DFG_RMT_V1 runtime: native DFG plus detached GT-free robust transport."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from model.phase2b_runtime import (
    IMAGE_SIZE,
    PATCH_GRID,
    STAGES,
    deploy_native_logits,
)
from .core import (
    cir_logits_from_native_weights,
    peer_delta_from_native_margins,
)


@dataclass
class CIRForward:
    seg_features: torch.Tensor
    det_features: torch.Tensor
    text_features: torch.Tensor
    native_weights: torch.Tensor
    native_logits: torch.Tensor
    native_margin: torch.Tensor
    delta: torch.Tensor
    peer_indices: torch.Tensor
    peer_valid: torch.Tensor
    peer_candidate_count: torch.Tensor
    peer_margins: torch.Tensor
    delta_stats: dict[str, torch.Tensor]
    cir_logits: torch.Tensor
    cir_margin: torch.Tensor
    cir_patch_probability: torch.Tensor
    cir_segmentation_probability: torch.Tensor
    cir_deployed_logits: torch.Tensor
    cir_training_segmentation_probability: torch.Tensor
    classification_logits: torch.Tensor
    classification_probability: torch.Tensor

    def as_dict(self) -> dict[str, Any]:
        return {name: value for name, value in self.__dict__.items()}


def _text_features(model: Any, dataset_name: str, class_names: Sequence[str], device: torch.device, config: Mapping[str, Any]) -> torch.Tensor:
    from model.phase2b_runtime import _text_features as parent_text
    text_config = dict(config)
    text_config["use_hybrid_soft_prompt"] = bool(getattr(model, "use_hybrid_soft_prompt", config.get("use_hybrid_soft_prompt", False)))
    text_config["use_soft_prompt"] = bool(getattr(model, "use_soft_prompt", config.get("use_soft_prompt", False)) )
    # Parent returns [G,B,D,2]; CIR core consumes [B,G,D,2].
    return parent_text(model, dataset_name, class_names, device, text_config).permute(1, 0, 2, 3).contiguous().float()


def _training_probability(native_logits: torch.Tensor, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    if native_logits.ndim != 4 or native_logits.shape[0] != STAGES or native_logits.shape[-1] != 2:
        raise ValueError(f"expected [S,B,P,2], got {tuple(native_logits.shape)}")
    stages, batch, patches, classes = native_logits.shape
    grid = int(round(patches ** 0.5))
    if grid * grid != patches:
        raise ValueError("training probability requires square patch grid")
    maps = native_logits.permute(0, 1, 3, 2).reshape(stages * batch, classes, grid, grid)
    maps = F.interpolate(maps, size=int(image_size), mode="bilinear", align_corners=True)
    maps = maps.reshape(stages, batch, classes, int(image_size), int(image_size)).mean(dim=0)
    return F.softmax(maps, dim=1)


def _native_weights(model: Any, seg_features: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    if seg_features.ndim != 4 or text.ndim != 4:
        raise ValueError("expected seg [S,B,P,D] and text [B,G,D,2]")
    rows = []
    for stage in range(int(seg_features.shape[0])):
        values = model.compute_dfg_weights(seg_features[stage], text, stage)
        rows.append(torch.stack([values["normal"], values["abnormal"]], dim=-1))
    native = torch.stack(rows, dim=0).float()
    if native.shape[:2] != seg_features.shape[:2] or native.shape[-1] != 2:
        raise ValueError(f"native DFG weight shape mismatch: {tuple(native.shape)}")
    return native


def _transport_summaries(native_weights: torch.Tensor, transported_logits: torch.Tensor, delta: torch.Tensor) -> dict[str, float]:
    # This is diagnostic only; it never enters the objective.
    normal = native_weights[..., 0]
    abnormal = native_weights[..., 1]
    return {
        "delta_mean": float(delta.detach().mean()),
        "delta_abs_mean": float(delta.detach().abs().mean()),
        "delta_saturation_fraction": float((delta.detach().abs() > 0.95).float().mean()),
        "native_normal_entropy": float((-(normal * normal.clamp_min(1e-8).log()).sum(-1)).mean().detach()),
        "native_abnormal_entropy": float((-(abnormal * abnormal.clamp_min(1e-8).log()).sum(-1)).mean().detach()),
        "cir_margin_mean": float((transported_logits[..., 1] - transported_logits[..., 0]).detach().mean()),
    }


def forward_cir(
    model: Any,
    image: torch.Tensor,
    class_names: Sequence[str] | str,
    device: torch.device,
    config: Mapping[str, Any],
    *,
    domain: str = "Industrial",
    require_grad: bool = False,
    dataset_name: str | None = None,
    precomputed_text_features: torch.Tensor | None = None,
) -> CIRForward:
    """Run one CIR forward while keeping the peer/transport path GT-free."""
    names = [class_names] if isinstance(class_names, str) else list(class_names)
    if image.ndim != 4 or tuple(image.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"unexpected CIR input image shape: {tuple(image.shape)}")
    context = torch.enable_grad() if require_grad else torch.inference_mode()
    with context:
        text = precomputed_text_features
        if text is None:
            text = _text_features(model, str(dataset_name or config.get("source_dataset", "VisA")), names, device, config)
        else:
            # Accept either parent [G,B,D,2] or CIR [B,G,D,2] convention.
            text = text.to(device=device, dtype=torch.float32)
            if text.ndim == 4 and text.shape[0] == int(config["n_groups"]) and text.shape[1] == image.shape[0]:
                text = text.permute(1, 0, 2, 3).contiguous()
        text = text.to(device=device, dtype=torch.float32)
        visual = model(image.to(device=device).float(), return_phase4_features=True)
        seg_features = torch.stack([value.float() for value in visual["seg_tokens"]], dim=0)
        det_features = torch.stack([value.float() for value in visual["det_tokens"]], dim=0)
        if tuple(seg_features.shape[:3]) != (STAGES, image.shape[0], int(PATCH_GRID[0] * PATCH_GRID[1])):
            raise ValueError(f"CIR segmentation geometry mismatch: {tuple(seg_features.shape)}")
        if seg_features.shape[-1] != 768:
            raise ValueError(f"CIR projected dimension mismatch: {tuple(seg_features.shape)}")
        parent_text = text.permute(1, 0, 2, 3).contiguous()
        _, native_logits, native_margin = model.vision_text_fusion_gate_seg(
            seg_features, parent_text, img_size=int(config.get("img_size", IMAGE_SIZE)),
            test_mode=False, domain=domain, return_details=True,
        )
        native_logits = native_logits.float()
        native_margin = native_margin.float()
        native_weights = _native_weights(model, seg_features, text)
        delta, delta_stats = peer_delta_from_native_margins(
            seg_features.detach(), native_margin.detach(),
            peer_count=int(config["rmt_peer_count"]),
            spatial_radius=int(config.get("rmt_spatial_radius", 3)),
            eps=float(config["rmt_eps"]),
            mad_constant=float(config["rmt_mad_constant"]),
        )
        score_mode = str(config.get("rmt_score_mode", "exact_score_space")).lower()
        scorer_mode = "optimized" if score_mode == "exact_score_space" else score_mode
        cir_logits, _native_score_path = cir_logits_from_native_weights(
            seg_features, text, native_weights, delta,
            float(config["rmt_transport_alpha"]),
            score_mode=scorer_mode,
            eps=float(config["rmt_eps"]),
        )
        cir_logits = cir_logits.float()
        cir_margin = cir_logits[..., 1] - cir_logits[..., 0]
        cir_patch_probability = F.softmax(cir_logits, dim=-1)[..., 1]
        cir_segmentation_probability, deployed_logits = deploy_native_logits(
            cir_logits, patch_grid=PATCH_GRID, image_size=int(config.get("img_size", IMAGE_SIZE)), domain=domain,
        )
        training_probability = _training_probability(cir_logits, int(config.get("img_size", IMAGE_SIZE)))
        classification_logits = torch.einsum("sbd,sbdc->sbc", det_features, parent_text).mean(dim=0)
        classification_probability = F.softmax(classification_logits.float(), dim=1)[:, 1]
    delta_stats = dict(delta_stats)
    delta_stats["transport"] = _transport_summaries(native_weights, cir_logits, delta)
    return CIRForward(
        seg_features=seg_features,
        det_features=det_features,
        text_features=text,
        native_weights=native_weights,
        native_logits=native_logits,
        native_margin=native_margin,
        delta=delta,
        peer_indices=delta_stats.pop("peer_indices"),
        peer_valid=delta_stats.pop("valid"),
        peer_candidate_count=delta_stats.pop("candidate_count"),
        peer_margins=delta_stats.pop("peer_margins"),
        delta_stats=delta_stats,
        cir_logits=cir_logits,
        cir_margin=cir_margin,
        cir_patch_probability=cir_patch_probability,
        cir_segmentation_probability=cir_segmentation_probability[:, 1],
        cir_deployed_logits=deployed_logits,
        cir_training_segmentation_probability=training_probability,
        classification_logits=classification_logits,
        classification_probability=classification_probability,
    )
