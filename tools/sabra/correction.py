"""Authority and native-logit correction operator."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from model.phase2b_runtime import deploy_native_logits, deploy_with_delta


def validate_lambda(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"lambda must be in [0,1], got {value}")
    return value


def margin_scale_p90(margins: np.ndarray) -> dict[str, Any]:
    values = np.asarray(margins, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("margin scale requires finite non-empty VisA margins")
    scale = float(np.percentile(np.abs(values), 90.0, method="linear"))
    return {"definition": "P90(abs(native_margin))", "percentile": 90.0, "implementation": "numpy.percentile(method=linear)", "value": scale, "count": int(values.size)}


def authority(trust_probability: np.ndarray, need_probability: np.ndarray) -> np.ndarray:
    trust = np.asarray(trust_probability, dtype=np.float32)
    need = np.asarray(need_probability, dtype=np.float32)
    if trust.shape != need.shape:
        raise ValueError("Trust and Need shapes differ")
    return np.clip(trust, 0.0, 1.0) * np.clip(need, 0.0, 1.0)


def correction_values(lambda_value: float, margin_scale: float, trust_probability: np.ndarray, need_probability: np.ndarray) -> np.ndarray:
    value = validate_lambda(lambda_value)
    scale = float(margin_scale)
    if scale < 0.0 or not np.isfinite(scale):
        raise ValueError("margin scale must be finite and non-negative")
    return (value * scale * authority(trust_probability, need_probability)).astype(np.float32)


def build_delta(native_logits: torch.Tensor, correction: torch.Tensor | np.ndarray) -> torch.Tensor:
    if native_logits.ndim != 4 or native_logits.shape[-1] != 2:
        raise ValueError("native logits must be [S,B,P,2]")
    values = torch.as_tensor(correction, dtype=native_logits.dtype, device=native_logits.device)
    if tuple(values.shape) != tuple(native_logits.shape[1:3]):
        raise ValueError(f"correction shape {tuple(values.shape)} != [B,P] {tuple(native_logits.shape[1:3])}")
    if bool((values < 0).any()):
        raise ValueError("correction must be non-negative")
    zeros = torch.zeros_like(values)
    one_stage = torch.stack([zeros, values], dim=-1)
    delta = one_stage.unsqueeze(0).expand(native_logits.shape[0], -1, -1, -1).clone()
    if not torch.equal(delta[..., 0], torch.zeros_like(delta[..., 0])):
        raise AssertionError("normal-channel delta is not zero")
    if not torch.equal(delta[0, ..., 1], delta[-1, ..., 1]):
        raise AssertionError("correction is not shared across stages")
    return delta


def corrected_deployment(native_logits: torch.Tensor, correction: torch.Tensor | np.ndarray, domain: str = "Industrial") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta = build_delta(native_logits, correction)
    native_probability, native_deployed_logits = deploy_native_logits(native_logits, domain=domain)
    corrected_probability, corrected_deployed_logits = deploy_with_delta(native_logits, delta, domain=domain)
    return corrected_probability, native_deployed_logits, corrected_deployed_logits
