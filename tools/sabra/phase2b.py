"""Frozen Phase2B loading and deployment helpers for setup smoke tests.

This module deliberately exposes no H6 route, residual, optimizer, or
backward path.  The only deployment operator here is the source Phase2B
native-logit deployment: Gaussian blur, aligned bilinear resize, stage mean,
and two-class softmax.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from model.adapter import ACDCLIP  # noqa: E402
from model.clip import create_model  # noqa: E402
from model.checkpoint_utils import load_adapter_checkpoint  # noqa: E402
from utils import get_phase2b_global_text_features  # noqa: E402


PATCH_GRID = (37, 37)
PATCH_COUNT = PATCH_GRID[0] * PATCH_GRID[1]
IMAGE_SIZE = 518
STAGES = 3
PROJECTED_PATCH_DIM = 768


def build_frozen_phase2b(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    clip_asset: Path,
    device: torch.device,
) -> ACDCLIP:
    """Build the adapter state used by Phase2B without constructing H6."""
    os.environ["ACDCLIP_CLIP_VITL14_336"] = str(clip_asset.resolve())
    direct = inspect.signature(ACDCLIP.__init__).parameters
    kwargs = {
        name: config[name]
        for name, parameter in direct.items()
        if name not in {"self", "clip_model", "kwargs"}
        and parameter.kind is not inspect.Parameter.VAR_KEYWORD
        and name in config
    }
    # The authoritative adapter_5 checkpoint contains Phase2B image/text
    # adapters and no H6 state.  Keeping H6 disabled makes that boundary
    # explicit and prevents H6 from becoming a predictor/router by accident.
    kwargs["h6_progress"] = 0
    kwargs["dfg_beta_current"] = float(
        checkpoint.get("dfg_beta_current", checkpoint.get("dfg_beta", config.get("dfg_beta", 0.0)))
    )
    clip = create_model(
        config["model_name"],
        img_size=int(config["img_size"]),
        device=device,
        pretrained="openai",
        require_pretrained=True,
        precision="fp32",
    )
    if config.get("grad_checkpointing"):
        clip.set_grad_checkpointing(True)
    model = ACDCLIP(clip_model=clip, **kwargs).to(device)
    load_adapter_checkpoint(model, checkpoint)
    model.set_dfg_beta(float(checkpoint.get("dfg_beta_current", model.dfg_beta)))
    model.use_hybrid_soft_prompt = bool(checkpoint.get("use_hybrid_soft_prompt", True))
    model.use_soft_prompt = bool(checkpoint.get("use_soft_prompt", False))
    model.hybrid_alpha_current = float(checkpoint.get("hybrid_alpha_current", 0.0))
    model.prompt_mode = str(checkpoint.get("prompt_mode", "hybrid"))
    model.eval()
    model.clipmodel.eval()
    model.requires_grad_(False)
    return model


def deploy_native_logits(
    native: torch.Tensor,
    patch_grid: tuple[int, int] = PATCH_GRID,
    image_size: int = IMAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce Phase2B's source deployment from native stage logits."""
    if native.ndim != 4 or native.shape[0] != STAGES or native.shape[-1] != 2:
        raise RuntimeError(f"invalid native logits shape: {tuple(native.shape)}")
    if native.shape[2] != patch_grid[0] * patch_grid[1]:
        raise RuntimeError(f"invalid patch count: {tuple(native.shape)}")
    outputs = []
    for stage in range(native.shape[0]):
        logits = native[stage].permute(0, 2, 1).reshape(native.shape[1], 2, *patch_grid)
        logits = gaussian_blur2d(logits, (7, 7), (1, 1))
        outputs.append(
            F.interpolate(
                logits,
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=True,
            )
        )
    final_logits = torch.stack(outputs).mean(dim=0)
    return F.softmax(final_logits, dim=1), final_logits


def deploy_with_delta(
    native: torch.Tensor,
    delta: torch.Tensor,
    patch_grid: tuple[int, int] = PATCH_GRID,
    image_size: int = IMAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sidecar deployment interface; Prompt 1 only exercises delta=zeros."""
    if delta.shape != native.shape:
        raise ValueError(f"delta shape {tuple(delta.shape)} != native {tuple(native.shape)}")
    return deploy_native_logits(native + delta, patch_grid=patch_grid, image_size=image_size)


@torch.inference_mode()
def forward_phase2b(
    model: ACDCLIP,
    image: torch.Tensor,
    class_name: str,
    device: torch.device,
    image_size: int = IMAGE_SIZE,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Run one frozen image-only Phase2B forward and retain native outputs."""
    if image.ndim != 4 or tuple(image.shape[1:]) != (3, image_size, image_size):
        raise ValueError(f"unexpected input image shape: {tuple(image.shape)}")
    visual = model(image.to(device).float(), return_phase4_features=True)
    features = torch.stack([value.float() for value in visual["seg_tokens"]])
    text = get_phase2b_global_text_features(
        model,
        "VisA",
        [class_name],
        device,
        use_hybrid_soft_prompt=True,
        use_soft_prompt=False,
    ).float()
    source_probability, native, native_margin = model.vision_text_fusion_gate_seg(
        features,
        text,
        img_size=image_size,
        test_mode=True,
        domain="Industrial",
        return_details=True,
    )
    reconstructed_probability, final_logits = deploy_native_logits(native)
    zero_probability, zero_logits = deploy_with_delta(native, torch.zeros_like(native))
    return {
        "visual_features": [value for value in visual["seg_tokens"]],
        "features": features,
        "text": text,
        "source_probability": source_probability,
        "native": native,
        "native_margin": native_margin,
        "reconstructed_probability": reconstructed_probability,
        "final_logits": final_logits,
        "zero_probability": zero_probability,
        "zero_logits": zero_logits,
    }
