"""Optional, train-only image-parameter anchor used by the bounded solution test."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ImageParameterAnchor:
    """Frozen reference tensors and a normalized distance to a live adapter."""

    def __init__(self, reference: Mapping[str, torch.Tensor], *, checkpoint_sha256: str, epoch: int, config_sha256: str | None, device: torch.device) -> None:
        self.reference_checkpoint_sha256 = str(checkpoint_sha256)
        self.reference_epoch = int(epoch)
        self.reference_config_sha256 = None if config_sha256 is None else str(config_sha256)
        self.reference = {name: value.detach().to(device=device).clone() for name, value in reference.items()}
        for value in self.reference.values():
            value.requires_grad_(False)

    def loss(self, module: torch.nn.Module) -> torch.Tensor:
        live = dict(module.named_parameters())
        if set(live) != set(self.reference):
            missing = sorted(set(live) - set(self.reference))
            extra = sorted(set(self.reference) - set(live))
            raise ValueError(f"image anchor parameter identity mismatch; missing={missing[:3]} extra={extra[:3]}")
        terms: list[torch.Tensor] = []
        for name, parameter in live.items():
            reference = self.reference[name].to(device=parameter.device, dtype=parameter.dtype)
            if tuple(parameter.shape) != tuple(reference.shape):
                raise ValueError(f"image anchor shape mismatch for {name}: {tuple(parameter.shape)} != {tuple(reference.shape)}")
            denominator = reference.detach().float().pow(2).sum().clamp_min(1.0e-12)
            terms.append((parameter.float() - reference.detach().float()).pow(2).sum() / denominator)
        return torch.stack(terms).mean() if terms else next(module.parameters()).sum() * 0.0

    def metadata(self, coefficient: float) -> dict[str, Any]:
        return {
            "enabled": bool(float(coefficient) > 0.0),
            "lambda_image_anchor": float(coefficient),
            "reference_checkpoint_sha256": self.reference_checkpoint_sha256,
            "reference_epoch": self.reference_epoch,
            "reference_config_sha256": self.reference_config_sha256,
            "scope": "image_adapter_parameters_only",
            "train_only": True,
        }


def load_image_parameter_anchor(path: Path, model: Any, device: torch.device) -> ImageParameterAnchor:
    """Load and validate a frozen parent checkpoint's image adapter state."""
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("image_adapter"), Mapping):
        raise ValueError(f"anchor checkpoint lacks image_adapter state: {path}")
    if payload.get("precision") != "fp32" or payload.get("amp_enabled") is True or payload.get("tf32_enabled") is True:
        raise ValueError("image anchor reference violates the FP32 contract")
    reference = {str(name): value.detach().float().cpu().clone() for name, value in payload["image_adapter"].items() if isinstance(value, torch.Tensor)}
    expected = {str(name): value for name, value in model.image_adapter.named_parameters()}
    if set(reference) != set(expected):
        raise ValueError("image anchor reference does not match the current image_adapter parameter set")
    for name, value in expected.items():
        if tuple(value.shape) != tuple(reference[name].shape):
            raise ValueError(f"image anchor reference shape mismatch for {name}")
    return ImageParameterAnchor(
        reference,
        checkpoint_sha256=sha256_file(path),
        epoch=int(payload.get("epoch", -1)),
        config_sha256=payload.get("config_sha256"),
        device=device,
    )
