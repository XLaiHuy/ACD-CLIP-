"""Authoritative Phase2B runtime and native deployment contract."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d

from .phase2b_legacy_bridge import (
    assert_canonical_config,
    assert_legacy_branch_disabled,
    build_adapter,
    load_adapter_state,
    runtime_audit,
    trainable_parameter_summary,
)

PROTOCOL_VERSION = "PHASE2B_CANONICAL_V1"
PATCH_GRID = (37, 37)
PATCH_COUNT = PATCH_GRID[0] * PATCH_GRID[1]
IMAGE_SIZE = 518
STAGES = 3
PROJECTED_PATCH_DIM = 768


@dataclass
class Phase2BForward:
    """Typed output shared by training, selection, calibration, and testing."""

    seg_features: torch.Tensor
    det_features: torch.Tensor
    text_features: torch.Tensor
    native_logits: torch.Tensor
    native_margin: torch.Tensor
    native_segmentation_probability: torch.Tensor
    deployed_logits: torch.Tensor
    classification_probability: torch.Tensor
    training_segmentation_probability: torch.Tensor | None = None

    def as_dict(self) -> dict[str, torch.Tensor | None]:
        return {
            "seg_features": self.seg_features,
            "det_features": self.det_features,
            "text_features": self.text_features,
            "native_logits": self.native_logits,
            "native_margin": self.native_margin,
            "native_segmentation_probability": self.native_segmentation_probability,
            "deployed_logits": self.deployed_logits,
            "classification_probability": self.classification_probability,
            "training_segmentation_probability": self.training_segmentation_probability,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_phase2b_config(payload)
    return payload


def validate_phase2b_config(config: Mapping[str, Any]) -> None:
    required = {
        "protocol_version",
        "model_name",
        "img_size",
        "n_groups",
        "dfg_mode",
        "soft_prompt_ctx_len",
        "precision",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"canonical Phase2B config missing fields: {missing}")
    if config["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError(f"unsupported Phase2B protocol: {config['protocol_version']!r}")
    if int(config["img_size"]) != IMAGE_SIZE:
        raise ValueError(f"canonical Phase2B requires img_size={IMAGE_SIZE}")
    if int(config["n_groups"]) != STAGES:
        raise ValueError(f"canonical Phase2B requires n_groups={STAGES}")
    if str(config["precision"]) != "fp32":
        raise ValueError("canonical Phase2B requires fp32")
    assert_canonical_config(config)


def _checkpoint_payload(checkpoint: Mapping[str, Any] | str | Path, device: torch.device) -> dict[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        return torch.load(Path(checkpoint), map_location=device, weights_only=False)
    return dict(checkpoint)


def build_phase2b_trainable(
    config: Mapping[str, Any],
    clip_asset: str | Path,
    device: torch.device,
) -> Any:
    validate_phase2b_config(config)
    model = build_adapter(config, clip_asset, device, trainable=True)
    assert_legacy_branch_disabled(model)
    return model


def build_phase2b_frozen(
    config: Mapping[str, Any],
    checkpoint: Mapping[str, Any] | str | Path,
    clip_asset: str | Path,
    device: torch.device,
) -> Any:
    validate_phase2b_config(config)
    payload = _checkpoint_payload(checkpoint, device)
    model = build_adapter(config, clip_asset, device, checkpoint=payload, trainable=False)
    load_adapter_state(model, payload)
    model.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("frozen Phase2B runtime has trainable parameters")
    assert_legacy_branch_disabled(model)
    return model


def load_phase2b_checkpoint(
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    clip_asset: str | Path,
    device: torch.device,
) -> Any:
    return build_phase2b_frozen(config, checkpoint_path, clip_asset, device)




def _deployment_parameters(domain: str) -> tuple[int, float]:
    if str(domain) == "Medical":
        return 9, 1.5
    return 7, 1.0


def deploy_native_logits(
    native_logits: torch.Tensor,
    patch_grid: tuple[int, int] = PATCH_GRID,
    image_size: int = IMAGE_SIZE,
    domain: str = "Industrial",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian blur -> aligned bilinear resize -> stage mean -> softmax."""
    if native_logits.ndim != 4:
        raise ValueError(f"native logits must be [S,B,P,2], got {tuple(native_logits.shape)}")
    if native_logits.shape[0] != STAGES or native_logits.shape[-1] != 2:
        raise ValueError(f"native logits contract violation: {tuple(native_logits.shape)}")
    if native_logits.shape[2] != int(patch_grid[0]) * int(patch_grid[1]):
        raise ValueError(f"native patch count contract violation: {tuple(native_logits.shape)}")
    kernel, sigma = _deployment_parameters(domain)
    outputs = []
    for stage in range(STAGES):
        logits = native_logits[stage].permute(0, 2, 1).reshape(native_logits.shape[1], 2, *patch_grid)
        logits = gaussian_blur2d(logits, (kernel, kernel), (sigma, sigma))
        outputs.append(
            F.interpolate(logits, size=(image_size, image_size), mode="bilinear", align_corners=True)
        )
    deployed_logits = torch.stack(outputs, dim=0).mean(dim=0)
    return F.softmax(deployed_logits, dim=1), deployed_logits


def deploy_with_delta(
    native_logits: torch.Tensor,
    delta: torch.Tensor,
    patch_grid: tuple[int, int] = PATCH_GRID,
    image_size: int = IMAGE_SIZE,
    domain: str = "Industrial",
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(delta.shape) != tuple(native_logits.shape):
        raise ValueError(f"delta shape {tuple(delta.shape)} != native {tuple(native_logits.shape)}")
    return deploy_native_logits(
        native_logits + delta,
        patch_grid=patch_grid,
        image_size=image_size,
        domain=domain,
    )


def _text_features(model: Any, dataset_name: str, class_names: Sequence[str], device: torch.device, config: Mapping[str, Any]) -> torch.Tensor:
    from utils import get_phase2b_global_text_features

    return get_phase2b_global_text_features(
        model,
        dataset_name,
        list(class_names),
        device,
        use_hybrid_soft_prompt=bool(config.get("use_hybrid_soft_prompt", False)),
        use_soft_prompt=bool(config.get("use_soft_prompt", False)),
    ).float()


def _native_from_visual(model: Any, image: torch.Tensor, text: torch.Tensor, domain: str, image_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    visual = model(image.float(), return_phase4_features=True)
    seg_features = torch.stack([value.float() for value in visual["seg_tokens"]], dim=0)
    det_features = torch.stack([value.float() for value in visual["det_tokens"]], dim=0)
    if tuple(seg_features.shape[::2]) != (STAGES, PATCH_COUNT):
        raise ValueError(f"segmentation feature geometry mismatch: {tuple(seg_features.shape)}")
    if tuple(seg_features.shape[-1:]) != (PROJECTED_PATCH_DIM,):
        raise ValueError(f"projected patch dimension mismatch: {tuple(seg_features.shape)}")
    training_probability, native_logits, native_margin = model.vision_text_fusion_gate_seg(
        seg_features,
        text,
        img_size=image_size,
        test_mode=False,
        domain=domain,
        return_details=True,
    )
    return seg_features, det_features, training_probability, native_logits, native_margin


def forward_phase2b(
    model: Any,
    image: torch.Tensor,
    class_names: Sequence[str] | str,
    device: torch.device,
    config: Mapping[str, Any],
    domain: str = "Industrial",
    require_grad: bool = False,
    dataset_name: str | None = None,
) -> Phase2BForward:
    """Run one canonical Phase2B forward and retain all audit tensors."""
    names = [class_names] if isinstance(class_names, str) else list(class_names)
    if image.ndim != 4 or image.shape[1] != 3 or image.shape[2] != IMAGE_SIZE or image.shape[3] != IMAGE_SIZE:
        raise ValueError(f"unexpected input image shape: {tuple(image.shape)}")
    assert_legacy_branch_disabled(model)
    context = torch.enable_grad() if require_grad else torch.inference_mode()
    with context:
        text = _text_features(model, str(dataset_name or config.get("dataset", "VisA")), names, device, config)
        seg_features, det_features, training_probability, native, margin = _native_from_visual(
            model, image.to(device), text, domain, int(config["img_size"])
        )
        deployed_probability, deployed_logits = deploy_native_logits(
            native,
            image_size=int(config["img_size"]),
            domain=domain,
        )
        cls_logits = torch.stack(
            [torch.matmul(det_features[index].unsqueeze(1), text[index]).squeeze(1) for index in range(STAGES)],
            dim=0,
        ).mean(dim=0)
        classification_probability = F.softmax(cls_logits, dim=1)[:, 1]
    return Phase2BForward(
        seg_features=seg_features,
        det_features=det_features,
        text_features=text,
        native_logits=native,
        native_margin=margin,
        native_segmentation_probability=deployed_probability[:, 1],
        deployed_logits=deployed_logits,
        classification_probability=classification_probability,
        training_segmentation_probability=training_probability,
    )


def trainable_parameter_counts(model: Any) -> dict[str, int]:
    return trainable_parameter_summary(model)
