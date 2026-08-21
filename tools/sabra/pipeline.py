"""Inference-only Phase2B versus frozen SABRA composition."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from model.phase2b_runtime import Phase2BForward, deploy_with_delta
from .artifacts import validate_sabra_freeze
from .correction import build_delta, correction_values
from .relational import build_relational_record, trust_features, need_features
from .trust import frozen_probability


def corrected_from_forward(forward: Phase2BForward, freeze: Mapping[str, Any], domain: str = "Industrial") -> dict[str, Any]:
    """Apply only the frozen relational/Trust/Need correction to one forward."""
    validate_sabra_freeze(freeze)
    native = forward.native_logits.detach()
    features = forward.seg_features.detach()
    corrections = []
    trust_values = []
    need_values = []
    evidence_values = []
    for batch_index in range(native.shape[1]):
        record = build_relational_record(
            features[:, batch_index].cpu().numpy(),
            forward.native_margin[:, batch_index].detach().cpu().numpy(),
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
        evidence_values.append(record["E"])
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
    }


def compare_forward(forward: Phase2BForward, freeze: Mapping[str, Any], domain: str = "Industrial") -> dict[str, Any]:
    corrected = corrected_from_forward(forward, freeze, domain=domain)
    return {
        "native_probability": forward.native_segmentation_probability,
        "corrected_probability": corrected["corrected_probability"][:, 1],
        "classification_probability": forward.classification_probability,
        **corrected,
    }
