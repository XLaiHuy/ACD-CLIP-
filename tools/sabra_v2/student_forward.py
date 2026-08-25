"""GT-free P27 student inference on frozen P26 Phase2B outputs."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from model.phase2b_runtime import deploy_native_logits
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_pool import symmetric_margin_delta, upsample_region_map


@dataclass(frozen=True)
class RegionStudentForward:
    """Student residuals and canonical deployment outputs; never contains GT."""

    region_residual: torch.Tensor
    patch_residual: torch.Tensor
    corrected_logits: torch.Tensor
    native_probability: torch.Tensor
    deployed_probability: torch.Tensor
    deployed_logits: torch.Tensor


def assert_frozen_phase2b(phase2b: nn.Module, adapter: RegionResidualAdapter) -> None:
    """Enforce that optimization ownership belongs exclusively to P27 adapter."""
    trainable_phase2b = [name for name, parameter in phase2b.named_parameters() if parameter.requires_grad]
    if trainable_phase2b:
        raise RuntimeError(f"Phase2B must be frozen; trainable parameters: {trainable_phase2b}")
    frozen_adapter = [name for name, parameter in adapter.named_parameters() if not parameter.requires_grad]
    if frozen_adapter:
        raise RuntimeError(f"P27 adapter must remain trainable; frozen parameters: {frozen_adapter}")


def forward_region_student(
    adapter: RegionResidualAdapter, seg_features: torch.Tensor, native_logits: torch.Tensor
) -> RegionStudentForward:
    """Apply the approved symmetric residual before unchanged P26 deployment."""
    region_residual = adapter(seg_features)
    patch_residual = upsample_region_map(region_residual)
    corrected_logits = symmetric_margin_delta(native_logits, patch_residual)
    native_probability, _ = deploy_native_logits(native_logits, domain="Industrial")
    deployed_probability, deployed_logits = deploy_native_logits(corrected_logits, domain="Industrial")
    return RegionStudentForward(
        region_residual=region_residual,
        patch_residual=patch_residual,
        corrected_logits=corrected_logits,
        native_probability=native_probability,
        deployed_probability=deployed_probability,
        deployed_logits=deployed_logits,
    )
