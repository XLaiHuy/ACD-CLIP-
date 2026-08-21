"""Need-C1 intervention oracle and frozen predictor."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
import torch

from model.phase2b_runtime import deploy_with_delta
from utils import calculate_seg_loss
from .relational import NEED_ORDER, need_features
from .trust import fit_binary_predictor, frozen_probability


def intervention_delta(intervention: torch.Tensor, native_logits: torch.Tensor) -> torch.Tensor:
    """Broadcast one abnormal-channel intervention over all three stages."""
    if native_logits.ndim != 4 or native_logits.shape[0] != 3 or native_logits.shape[-1] != 2:
        raise ValueError("native logits must be [3,B,P,2]")
    if tuple(intervention.shape) != tuple(native_logits.shape[1:3]):
        raise ValueError("intervention must be [B,P]")
    zeros = torch.zeros_like(intervention)
    one_stage = torch.stack([zeros, intervention], dim=-1)
    delta = one_stage.unsqueeze(0).expand(native_logits.shape[0], -1, -1, -1)
    if not torch.equal(delta[..., 0], torch.zeros_like(delta[..., 0])):
        raise AssertionError("normal-channel intervention is non-zero")
    if not torch.equal(delta[0, ..., 1], delta[-1, ..., 1]):
        raise AssertionError("Need intervention is not shared across stages")
    return delta


def _oracle_loss(native_logits: torch.Tensor, intervention: torch.Tensor, target_mask: torch.Tensor, domain: str) -> torch.Tensor:
    probability, _ = deploy_with_delta(native_logits, intervention_delta(intervention, native_logits), domain=domain)
    return calculate_seg_loss(probability.float(), target_mask.to(device=native_logits.device, dtype=torch.float32))


def need_oracle(
    native_logits: torch.Tensor,
    target_mask: torch.Tensor,
    domain: str = "Industrial",
    phase2b_model: Any | None = None,
) -> dict[str, torch.Tensor | dict[str, Any]]:
    """Compute Need utility using exactly the canonical Phase2B segmentation loss."""
    if phase2b_model is not None:
        if any(parameter.requires_grad for parameter in phase2b_model.parameters()):
            raise AssertionError("Phase2B must be frozen during the Need oracle")
        if any(parameter.grad is not None for parameter in phase2b_model.parameters()):
            raise AssertionError("Phase2B gradients must be None before the Need oracle")
    native = native_logits.detach().clone().float()
    intervention = torch.zeros(native.shape[1:3], dtype=native.dtype, device=native.device, requires_grad=True)
    loss = _oracle_loss(native, intervention, target_mask, domain)
    gradient = torch.autograd.grad(loss, intervention, only_inputs=True, create_graph=False)[0]
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Need intervention gradient is non-finite")
    if phase2b_model is not None and any(parameter.grad is not None for parameter in phase2b_model.parameters()):
        raise AssertionError("Phase2B gradients changed during the Need oracle")
    utility = -gradient
    return {
        "signed_utility": utility.detach(),
        "target": (utility > 1e-8).to(torch.int8),
        "intervention_gradient": gradient.detach(),
        "loss": loss.detach(),
    }


def finite_difference_need_parity(
    native_logits: torch.Tensor,
    target_mask: torch.Tensor,
    domain: str = "Industrial",
    patch_indices: tuple[int, ...] = (0, 684, 1368),
    epsilon: float = 1e-4,
) -> dict[str, Any]:
    """Compare analytic Need utility to the audited central finite difference."""
    oracle = need_oracle(native_logits, target_mask, domain=domain)
    analytic = oracle["signed_utility"].detach().cpu().numpy()[0]
    rows = []
    for patch in patch_indices:
        plus = torch.zeros(native_logits.shape[1:3], dtype=native_logits.dtype, device=native_logits.device)
        minus = torch.zeros_like(plus)
        plus[:, patch] = float(epsilon)
        minus[:, patch] = -float(epsilon)
        with torch.no_grad():
            loss_plus = float(_oracle_loss(native_logits.detach(), plus, target_mask, domain).item())
            loss_minus = float(_oracle_loss(native_logits.detach(), minus, target_mask, domain).item())
        finite = -(loss_plus - loss_minus) / (2.0 * float(epsilon))
        value = float(analytic[patch])
        rows.append({"patch": int(patch), "analytic": value, "finite_difference": finite, "abs_error": abs(value - finite), "sign_match": bool(np.sign(value) == np.sign(finite) or abs(value) <= 1e-8 or abs(finite) <= 1e-8)})
    tolerance = max((2e-3 + 2e-2 * max(abs(row["analytic"]), abs(row["finite_difference"])) for row in rows), default=0.0)
    passed = all(row["abs_error"] <= 2e-3 + 2e-2 * max(abs(row["analytic"]), abs(row["finite_difference"])) for row in rows)
    return {"status": "PASS" if passed else "FAIL", "epsilon": float(epsilon), "rows": rows, "absolute_tolerance": 2e-3, "relative_tolerance": 2e-2, "max_tolerance": tolerance}


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
    return {"class_names": classes, "held_out_predictions": held_out, "held_out_targets": held_out_targets, "skipped": skipped, "feature_order": list(NEED_ORDER)}


def fit_need(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("no Need calibration records")
    matrix = np.concatenate([need_features(row) for row in rows], axis=0)
    targets = np.concatenate([np.asarray(row["need_target"], dtype=np.int8).reshape(-1) for row in rows])
    if matrix.shape[0] != targets.size:
        raise ValueError("Need records have inconsistent target widths")
    artifact = fit_binary_predictor(matrix, targets, NEED_ORDER)
    artifact["oracle_definition"] = "signed utility = -dL/dd using utils.calculate_seg_loss; target = signed utility > 1e-8"
    artifact["loco"] = _loco_need_audit(rows)
    return artifact


def need_probability(parameters: Mapping[str, Any], records: Iterable[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.concatenate([need_features(row) for row in records], axis=0)
    return frozen_probability(parameters, matrix)
