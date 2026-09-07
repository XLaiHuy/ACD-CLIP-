"""Parameter-free abnormal-Dice gradient budgeting for H2 bounded R1."""
from __future__ import annotations

from collections.abc import Sequence

import torch


GRAD_BUDGET_EPS = 1.0e-12


def _as_float32(value: torch.Tensor | None, parameter: torch.nn.Parameter) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(parameter.detach(), dtype=torch.float32)
    return value.detach().float()


def _joint_norm(values: Sequence[torch.Tensor]) -> float:
    if not values:
        return 0.0
    total = torch.zeros((), device=values[0].device, dtype=torch.float32)
    for value in values:
        total = total + value.square().sum()
    return float(total.sqrt().item())


def _cosine(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float | None:
    if not left or not right:
        return None
    dot = torch.zeros((), device=left[0].device, dtype=torch.float32)
    left_sq = torch.zeros_like(dot)
    right_sq = torch.zeros_like(dot)
    for left_value, right_value in zip(left, right):
        dot = dot + (left_value * right_value).sum()
        left_sq = left_sq + left_value.square().sum()
        right_sq = right_sq + right_value.square().sum()
    denominator = left_sq.sqrt() * right_sq.sqrt()
    if float(denominator.item()) == 0.0:
        return None
    return float((dot / denominator).item())


def budgeted_gradient(
        parameters: Sequence[torch.nn.Parameter],
        rest_gradients: Sequence[torch.Tensor | None],
        abnormal_gradients: Sequence[torch.Tensor | None],
        *,
        enabled: bool,
        eps: float = GRAD_BUDGET_EPS,
) -> tuple[list[torch.Tensor | None], dict[str, float | bool | None]]:
    """Return ``g_rest + alpha*g_abnormal`` over one joint parameter set.

    Inputs are expected to be unscaled gradients.  The helper never normalizes,
    projects, clips, or otherwise rotates the abnormal gradient direction.
    ``enabled=False`` is the control path and forces alpha to exactly one.
    """
    if len(parameters) != len(rest_gradients) or len(parameters) != len(abnormal_gradients):
        raise ValueError("gradient component lists must match parameter scope")
    rest = [_as_float32(value, parameter) for parameter, value in zip(parameters, rest_gradients)]
    abnormal = [_as_float32(value, parameter) for parameter, value in zip(parameters, abnormal_gradients)]
    abnormal_norm = _joint_norm(abnormal)
    rest_norm = _joint_norm(rest)
    raw_ratio = abnormal_norm / (rest_norm + float(eps))
    computed_alpha = min(1.0, rest_norm / (abnormal_norm + float(eps)))
    alpha = computed_alpha if enabled else 1.0
    effective = [
        None if rest_source is None and abnormal_source is None else rest_value + float(alpha) * abnormal_value
        for rest_source, abnormal_source, rest_value, abnormal_value in zip(
            rest_gradients, abnormal_gradients, rest, abnormal
        )
    ]
    effective_abnormal_norm = float(alpha) * abnormal_norm
    effective_ratio = effective_abnormal_norm / (rest_norm + float(eps))
    metrics: dict[str, float | bool | None] = {
        "g_abn_norm": abnormal_norm,
        "g_rest_norm": rest_norm,
        "raw_ratio": raw_ratio,
        "alpha": float(alpha),
        "computed_alpha": float(computed_alpha),
        "budget_active": bool(enabled and computed_alpha < 1.0),
        "effective_abn_norm": effective_abnormal_norm,
        "effective_ratio": effective_ratio,
        "cosine_abn_rest": _cosine(abnormal, rest),
        "eps": float(eps),
        "gradients_unscaled": True,
    }
    return effective, metrics
