"""Explicit, checkpointable training precision policies for H2 runs."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch


_VALID_PRECISIONS = ("fp32", "fp16", "bf16")


@dataclass(frozen=True)
class PrecisionPolicy:
    """Keep autocast and loss-scaling decisions in one audited location.

    Parameters and optimizer state are deliberately not cast here.  AMP only
    affects compute performed inside :meth:`autocast`.
    """

    name: str

    def __post_init__(self) -> None:
        if self.name not in _VALID_PRECISIONS:
            raise ValueError(f"unsupported precision policy {self.name!r}")

    @property
    def autocast_enabled(self) -> bool:
        return self.name != "fp32"

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        if self.name == "fp16":
            return torch.float16
        if self.name == "bf16":
            return torch.bfloat16
        return None

    @property
    def gradscaler_enabled(self) -> bool:
        # BF16 has FP32-like exponent range and must not use FP16 loss scaling.
        return self.name == "fp16"

    def autocast(self, device: str | torch.device) -> ContextManager[None]:
        device_type = torch.device(device).type
        if not self.autocast_enabled:
            return nullcontext()
        if device_type != "cuda" and self.name == "fp16":
            raise ValueError("FP16 autocast training requires CUDA")
        return torch.autocast(
            device_type=device_type,
            dtype=self.autocast_dtype,
            enabled=True,
        )


def resolve_precision_policy(
        precision: str | None,
        *,
        legacy_amp: bool = False,
) -> PrecisionPolicy:
    """Resolve the explicit policy while retaining legacy ``--amp`` callers."""
    if precision is None:
        return PrecisionPolicy("fp16" if legacy_amp else "fp32")
    if legacy_amp and precision == "fp32":
        raise ValueError("--amp conflicts with --precision fp32")
    return PrecisionPolicy(precision)
