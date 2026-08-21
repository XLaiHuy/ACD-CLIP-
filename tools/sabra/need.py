"""Need-C1 intervention oracle and frozen predictor."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from model.phase2b_runtime import deploy_with_delta
from .relational import NEED_ORDER, need_features
from .trust import fit_binary_predictor, frozen_probability


def intervention_delta(intervention: torch.Tensor, native_logits: torch.Tensor) -> torch.Tensor:
    """Broadcast one positive abnormal intervention over all three stages."""
    if native_logits.ndim != 4 or native_logits.shape[0] != 3 or native_logits.shape[-1] != 2:
        raise ValueError("native logits must be [3,B,P,2]")
    if tuple(intervention.shape) != tuple(native_logits.shape[1:3]):
        raise ValueError("intervention must be [B,P]")
    if bool((intervention < 0).any()):
        raise ValueError("Need intervention must be non-negative")
    zeros = torch.zeros_like(intervention)
    one_stage = torch.stack([zeros, intervention], dim=-1)
    delta = one_stage.unsqueeze(0).expand(native_logits.shape[0], -1, -1, -1)
    if not torch.equal(delta[..., 0], torch.zeros_like(delta[..., 0])):
        raise AssertionError("normal-channel intervention is non-zero")
    if not torch.equal(delta[0, ..., 1], delta[-1, ..., 1]):
        raise AssertionError("Need intervention is not shared across stages")
    return delta


def _segmentation_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The bounded oracle loss used for the intervention derivative."""
    target = target.float()
    if target.ndim == 3:
        target = target.unsqueeze(1)
    abnormal = probability[:, 1:2]
    normal = probability[:, 0:1]
    bce = F.binary_cross_entropy(abnormal.clamp(1e-6, 1 - 1e-6), target)
    dice_abnormal = 1.0 - (2.0 * (abnormal * target).flatten(1).sum(1) + 1.0) / (
        abnormal.flatten(1).sum(1) + target.flatten(1).sum(1) + 1.0
    )
    dice_normal = 1.0 - (2.0 * (normal * (1.0 - target)).flatten(1).sum(1) + 1.0) / (
        normal.flatten(1).sum(1) + (1.0 - target).flatten(1).sum(1) + 1.0
    )
    return bce + dice_abnormal.mean() + dice_normal.mean()


def need_oracle(
    native_logits: torch.Tensor,
    target_mask: torch.Tensor,
    domain: str = "Industrial",
) -> dict[str, torch.Tensor]:
    """Compute signed utility with frozen Phase2B and gradient only on d."""
    native = native_logits.detach()
    intervention = torch.zeros(native.shape[1:3], dtype=native.dtype, device=native.device, requires_grad=True)
    delta = intervention_delta(intervention, native)
    probability, _ = deploy_with_delta(native, delta, domain=domain)
    loss = _segmentation_loss(probability, target_mask.to(native.device))
    gradient = torch.autograd.grad(loss, intervention, only_inputs=True)[0]
    utility = -gradient
    return {
        "signed_utility": utility.detach(),
        "target": (utility > 1e-8).to(torch.int8),
        "intervention_gradient": gradient.detach(),
    }


def _loco_need_audit(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    classes = sorted({str(row["class_name"]) for row in rows})
    held_out: dict[str, list[float]] = {}
    held_out_targets: dict[str, list[int]] = {}
    skipped: dict[str, str] = {}
    for class_name in classes:
        train = [row for row in rows if str(row["class_name"]) != class_name]
        test = [row for row in rows if str(row["class_name"]) == class_name]
        if not train or not test:
            skipped[class_name] = "empty_train_or_test"
            continue
        try:
            x_train = np.concatenate([need_features(row) for row in train], axis=0)
            y_train = np.concatenate([np.asarray(row["need_target"], dtype=np.int8).reshape(-1) for row in train])
            artifact = fit_binary_predictor(x_train, y_train, NEED_ORDER)
        except ValueError as exc:
            skipped[class_name] = str(exc)
            continue
        x_test = np.concatenate([need_features(row) for row in test], axis=0)
        y_test = np.concatenate([np.asarray(row["need_target"], dtype=np.int8).reshape(-1) for row in test])
        held_out[class_name] = frozen_probability(artifact, x_test).tolist()
        held_out_targets[class_name] = y_test.tolist()
    return {
        "class_names": classes,
        "held_out_predictions": held_out,
        "held_out_targets": held_out_targets,
        "skipped": skipped,
        "feature_order": list(NEED_ORDER),
    }


def fit_need(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("no Need calibration records")
    matrix = np.concatenate([need_features(row) for row in rows], axis=0)
    targets = np.concatenate([np.asarray(row["need_target"], dtype=np.int8).reshape(-1) for row in rows])
    if matrix.shape[0] != targets.size:
        raise ValueError("Need records have inconsistent target widths")
    artifact = fit_binary_predictor(matrix, targets, NEED_ORDER)
    artifact["oracle_definition"] = "signed utility = -dL/dd; target = signed utility > 1e-8"
    artifact["loco"] = _loco_need_audit(rows)
    return artifact


def need_probability(parameters: Mapping[str, Any], records: Iterable[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.concatenate([need_features(row) for row in records], axis=0)
    return frozen_probability(parameters, matrix)
