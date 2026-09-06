"""Explicit, checkpointable training precision policies for H2 runs."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch


_VALID_PRECISIONS = ("fp32", "fp16", "bf16")
HISTORICAL_MIXED_FP16_FP32_V1 = "HISTORICAL_MIXED_FP16_FP32_V1"
BF16_V1 = "BF16_V1"
REPAIRED_MIXED_FP16_FP32_EXPERIMENTAL = "REPAIRED_MIXED_FP16_FP32_EXPERIMENTAL"
NAMED_PRECISION_PROTOCOLS = (
    HISTORICAL_MIXED_FP16_FP32_V1,
    BF16_V1,
    REPAIRED_MIXED_FP16_FP32_EXPERIMENTAL,
)


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


@dataclass(frozen=True)
class PrecisionRuntimeMode:
    """Named scientific precision runtime, separate from autocast dtype."""

    protocol_name: str
    policy: PrecisionPolicy
    later_transformer_fp32_islands: bool


def resolve_precision_runtime_mode(
        protocol_name: str | None,
        precision: str | None,
        *,
        legacy_amp: bool = False,
        legacy_local_fp32_islands: bool | None = None,
) -> PrecisionRuntimeMode:
    definitions = {
        HISTORICAL_MIXED_FP16_FP32_V1: ("fp16", False),
        BF16_V1: ("bf16", False),
        REPAIRED_MIXED_FP16_FP32_EXPERIMENTAL: ("fp16", True),
    }
    if protocol_name is not None:
        if protocol_name not in definitions:
            raise ValueError(
                f"unsupported precision protocol {protocol_name!r}; "
                f"expected one of {NAMED_PRECISION_PROTOCOLS}"
            )
        expected_precision, expected_islands = definitions[protocol_name]
        if precision is not None and precision != expected_precision:
            raise ValueError(f"{protocol_name} requires --precision {expected_precision}")
        if legacy_amp and expected_precision != "fp16":
            raise ValueError(f"--amp conflicts with {protocol_name}")
        if legacy_local_fp32_islands is not None and bool(legacy_local_fp32_islands) != expected_islands:
            raise ValueError(f"{protocol_name} requires later transformer FP32 islands={expected_islands}")
        return PrecisionRuntimeMode(protocol_name, PrecisionPolicy(expected_precision), expected_islands)

    policy = resolve_precision_policy(precision, legacy_amp=legacy_amp)
    islands = True if legacy_local_fp32_islands is None else bool(legacy_local_fp32_islands)
    inferred = {
        ("fp16", False): HISTORICAL_MIXED_FP16_FP32_V1,
        ("fp16", True): REPAIRED_MIXED_FP16_FP32_EXPERIMENTAL,
        ("bf16", False): BF16_V1,
    }.get((policy.name, islands), f"LEGACY_{policy.name.upper()}_UNNAMED")
    return PrecisionRuntimeMode(inferred, policy, islands)


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
