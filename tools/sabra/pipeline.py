"""Inference-only Phase2B versus frozen SABRA composition."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from model.phase2b_runtime import Phase2BForward, compute_deployment_sensitivity, deploy_with_delta
from .artifacts import validate_sabra_freeze
from .correction import build_delta, correction_values
from .relational import build_relational_record, need_features, trust_features
from .trust import frozen_probability


def corrected_from_forward(
    forward: Phase2BForward,
    freeze: Mapping[str, Any],
    domain: str = "Industrial",
    backend: str | None = None,
) -> dict[str, Any]:
    """Apply only the frozen relational/Trust/Need correction to one forward."""
    validate_sabra_freeze(freeze)
    native = forward.native_logits.detach()
    features = forward.seg_features.detach()
    stored_backend = str(freeze["relational"]["backend"]).lower()
    if backend is not None and str(backend).lower() != stored_backend:
        raise ValueError(
            f"requested SABRA backend {backend!r} does not match frozen backend {stored_backend!r}"
        )
    selected_backend = stored_backend
    corrections: list[np.ndarray] = []
    trust_values: list[np.ndarray] = []
    need_values: list[np.ndarray] = []
    evidence_values: list[np.ndarray] = []
    sensitivity_values: list[np.ndarray] = []
    for batch_index in range(native.shape[1]):
        sensitivity = compute_deployment_sensitivity(native[:, batch_index:batch_index + 1], domain=domain)[0].cpu().numpy()
        if not np.isfinite(sensitivity).all() or not np.any(np.abs(sensitivity) > 0):
            raise ValueError("computed deployment sensitivity is missing or all zero")
        record = build_relational_record(
            features[:, batch_index].cpu().numpy(),
            forward.native_margin[:, batch_index].detach().cpu().numpy(),
            deployment_sensitivity=sensitivity,
            backend=selected_backend,
        )
        trust = frozen_probability(freeze["trust"], trust_features(record))
        need = frozen_probability(freeze["need"], need_features(record))
        correction = correction_values(
            float(freeze["correction"]["lambda"]),
            float(freeze["correction"]["margin_scale"]),
            trust,
            need,
        )
        corrections.append(correction)
        trust_values.append(trust)
        need_values.append(need)
        evidence_values.append(np.asarray(record["E"], dtype=np.float32))
        sensitivity_values.append(sensitivity)
    correction_tensor = torch.as_tensor(np.stack(corrections, axis=0), dtype=native.dtype, device=native.device)
    delta = build_delta(native, correction_tensor)
    corrected_probability, corrected_logits = deploy_with_delta(native, delta, domain=domain)
    return {
        "corrected_probability": corrected_probability,
        "corrected_logits": corrected_logits,
        "delta": delta,
        "trust": np.stack(trust_values),
        "need": np.stack(need_values),
        "authority": np.stack(trust_values) * np.stack(need_values),
        "evidence": np.stack(evidence_values),
        "deployment_sensitivity": np.stack(sensitivity_values),
        "backend": selected_backend,
    }


def compare_forward(forward: Phase2BForward, freeze: Mapping[str, Any], domain: str = "Industrial") -> dict[str, Any]:
    corrected = corrected_from_forward(forward, freeze, domain=domain)
    # The public comparison map is the abnormal channel; retain the full
    # two-class deployment separately so callers cannot accidentally score the
    # normal channel as an additional pixel population.
    return {
        "native_probability": forward.native_segmentation_probability,
        "corrected_probability": corrected["corrected_probability"][:, 1],
        "corrected_probability_full": corrected["corrected_probability"],
        "classification_probability": forward.classification_probability,
        **{key: value for key, value in corrected.items() if key != "corrected_probability"},
    }
