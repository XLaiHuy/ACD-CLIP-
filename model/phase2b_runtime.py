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
    native_patch_probability: torch.Tensor
    native_segmentation_probability: torch.Tensor
    deployed_logits: torch.Tensor
    deployed_segmentation_probability: torch.Tensor
    classification_logits: torch.Tensor
    classification_probability: torch.Tensor
    training_segmentation_probability: torch.Tensor | None = None
    prompt_diagnostics: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "seg_features": self.seg_features,
            "det_features": self.det_features,
            "text_features": self.text_features,
            "native_logits": self.native_logits,
            "native_margin": self.native_margin,
            "native_patch_probability": self.native_patch_probability,
            "native_segmentation_probability": self.native_segmentation_probability,
            "deployed_logits": self.deployed_logits,
            "deployed_segmentation_probability": self.deployed_segmentation_probability,
            "classification_logits": self.classification_logits,
            "classification_probability": self.classification_probability,
            "training_segmentation_probability": self.training_segmentation_probability,
            "prompt_diagnostics": self.prompt_diagnostics,
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
        "protocol_version", "model_name", "img_size", "n_groups", "dfg_mode",
        "soft_prompt_ctx_len", "precision",
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
    if any(key in config for key in ("effective_batch_size", "micro_batch_size", "batch_size", "grad_accum_steps")):
        effective = int(config.get("micro_batch_size", config.get("batch_size", 1))) * int(config.get("grad_accum_steps", 1))
        declared = int(config.get("effective_batch_size", effective))
        if effective != declared:
            raise ValueError("micro_batch_size * grad_accum_steps must equal effective_batch_size")
        if declared != 6:
            raise ValueError("canonical Phase2B effective batch must be six")
    assert_canonical_config(config)


def configure_canonical_fp32() -> None:
    """Set the numerical policy explicitly for every canonical process."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _checkpoint_payload(checkpoint: Mapping[str, Any] | str | Path, device: torch.device) -> dict[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        return torch.load(Path(checkpoint), map_location=device, weights_only=False)
    return dict(checkpoint)


def build_phase2b_trainable(config: Mapping[str, Any], clip_asset: str | Path, device: torch.device) -> Any:
    validate_phase2b_config(config)
    model = build_adapter(config, clip_asset, device, trainable=True)
    model.use_hybrid_soft_prompt = bool(config.get("use_hybrid_soft_prompt", False))
    model.use_soft_prompt = bool(config.get("use_soft_prompt", False))
    model.hybrid_alpha_max = float(config.get("hybrid_alpha_max", 0.2))
    model.soft_prompt_freeze_epochs = int(config.get("soft_prompt_freeze_epochs", 3))
    model.prompt_mode = "hybrid" if model.use_hybrid_soft_prompt else "soft" if model.use_soft_prompt else "hard"
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
    return (9, 1.5) if str(domain) == "Medical" else (7, 1.0)


def deploy_native_logits(
    native_logits: torch.Tensor,
    patch_grid: tuple[int, int] = PATCH_GRID,
    image_size: int = IMAGE_SIZE,
    domain: str = "Industrial",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original deployment: Gaussian blur -> aligned bilinear resize -> stage mean -> softmax."""
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
        outputs.append(F.interpolate(logits, size=(image_size, image_size), mode="bilinear", align_corners=True))
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
    return deploy_native_logits(native_logits + delta, patch_grid=patch_grid, image_size=image_size, domain=domain)


def compute_deployment_sensitivity(native_logits: torch.Tensor, domain: str = "Industrial") -> torch.Tensor:
    """Compute the audited GT-free positive-logit deployment sensitivity.

    A shared intervention changes the abnormal native logit at a patch in all
    stages.  The derivative is taken through the original deployment operator,
    weighted by the base abnormal-probability derivative, and returned as an
    absolute [B,P] feature.  It never reads labels or masks.
    """
    native = native_logits.detach().clone().float()
    if native.ndim != 4 or native.shape[0] != STAGES or native.shape[-1] != 2:
        raise ValueError("native_logits must be [3,B,P,2]")
    with torch.no_grad():
        base_probability, base_logits = deploy_native_logits(native, domain=domain)
    shared = torch.zeros(native.shape[1:3], dtype=native.dtype, device=native.device, requires_grad=True)
    one_stage = torch.stack([torch.zeros_like(shared), shared], dim=-1)
    delta = one_stage.unsqueeze(0).expand(STAGES, -1, -1, -1)
    _, changed_logits = deploy_with_delta(native, delta, domain=domain)
    response = changed_logits[:, 1] - base_logits[:, 1]
    probability_weight = base_probability[:, 1] * (1.0 - base_probability[:, 1])
    objective = (response * probability_weight).mean()
    gradient = torch.autograd.grad(objective, shared, only_inputs=True)[0]
    sensitivity = gradient.abs()
    if not torch.isfinite(sensitivity).all():
        raise FloatingPointError("deployment sensitivity is non-finite")
    return sensitivity.detach()


def _text_features(model: Any, dataset_name: str, class_names: Sequence[str], device: torch.device, config: Mapping[str, Any]) -> torch.Tensor:
    from utils import get_phase2b_global_text_features
    return get_phase2b_global_text_features(
        model, dataset_name, list(class_names), device,
        use_hybrid_soft_prompt=bool(config.get("use_hybrid_soft_prompt", False)),
        use_soft_prompt=bool(config.get("use_soft_prompt", False)),
    ).float()


def _native_from_visual(
    model: Any,
    image: torch.Tensor,
    text: torch.Tensor,
    domain: str,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    visual = model(image.float(), return_phase4_features=True)
    seg_features = torch.stack([value.float() for value in visual["seg_tokens"]], dim=0)
    det_features = torch.stack([value.float() for value in visual["det_tokens"]], dim=0)
    if tuple(seg_features.shape[::2]) != (STAGES, PATCH_COUNT):
        raise ValueError(f"segmentation feature geometry mismatch: {tuple(seg_features.shape)}")
    if tuple(seg_features.shape[-1:]) != (PROJECTED_PATCH_DIM,):
        raise ValueError(f"projected patch dimension mismatch: {tuple(seg_features.shape)}")
    training_probability, native_logits, native_margin = model.vision_text_fusion_gate_seg(
        seg_features, text, img_size=image_size, test_mode=False, domain=domain, return_details=True,
    )
    return seg_features, det_features, training_probability.float(), native_logits.float(), native_margin.float()


def forward_phase2b(
    model: Any,
    image: torch.Tensor,
    class_names: Sequence[str] | str,
    device: torch.device,
    config: Mapping[str, Any],
    domain: str = "Industrial",
    require_grad: bool = False,
    dataset_name: str | None = None,
    precomputed_text_features: torch.Tensor | None = None,
) -> Phase2BForward:
    """Run one canonical Phase2B forward and retain all audit tensors."""
    names = [class_names] if isinstance(class_names, str) else list(class_names)
    if image.ndim != 4 or image.shape[1] != 3 or image.shape[2] != IMAGE_SIZE or image.shape[3] != IMAGE_SIZE:
        raise ValueError(f"unexpected input image shape: {tuple(image.shape)}")
    assert_legacy_branch_disabled(model)
    context = torch.enable_grad() if require_grad else torch.inference_mode()
    with context:
        text = precomputed_text_features if precomputed_text_features is not None else _text_features(
            model, str(dataset_name or config.get("dataset", "VisA")), names, device, config,
        )
        text = text.to(device=device, dtype=torch.float32)
        seg_features, det_features, training_probability, native, margin = _native_from_visual(
            model, image.to(device), text, domain, int(config["img_size"]),
        )
        deployed_probability, deployed_logits = deploy_native_logits(native, image_size=int(config["img_size"]), domain=domain)
        native_patch_probability = F.softmax(native, dim=-1)[..., 1]
        classification_logits = torch.einsum("sbd,sbdc->sbc", det_features, text).mean(dim=0)
        classification_probability = F.softmax(classification_logits.float(), dim=1)[:, 1]
    return Phase2BForward(
        seg_features=seg_features,
        det_features=det_features,
        text_features=text,
        native_logits=native,
        native_margin=margin,
        native_patch_probability=native_patch_probability,
        native_segmentation_probability=deployed_probability[:, 1],
        deployed_logits=deployed_logits,
        deployed_segmentation_probability=deployed_probability[:, 1],
        classification_logits=classification_logits,
        classification_probability=classification_probability,
        training_segmentation_probability=training_probability,
    )


def trainable_parameter_counts(model: Any) -> dict[str, int]:
    return trainable_parameter_summary(model)
