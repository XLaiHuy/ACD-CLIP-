"""One bounded H2 functional feature-anchor mechanism.

The module intentionally contains only token-wise cosine geometry against a
frozen E1 teacher.  It has no projection, parameters, or target-data path.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


# Zero-based indexing of the three adapter-stage tensors: stages 2 and 3 only.
FUNCTIONAL_ANCHOR_STAGES = (1, 2)
FUNCTIONAL_CONFIG_KEYS = frozenset({
    "use_functional_feature_anchor",
    "functional_anchor_lambda",
    "functional_anchor_reference_sha256",
    "functional_anchor_stages",
})


def freeze_e1_teacher(module: torch.nn.Module) -> torch.nn.Module:
    """Put an E1 reference model in inference-only state."""
    module.eval()
    module.requires_grad_(False)
    return module


def functional_feature_anchor_loss(
        student_stages: Sequence[torch.Tensor],
        teacher_stages: Sequence[torch.Tensor],
        *,
        stages: tuple[int, int] = FUNCTIONAL_ANCHOR_STAGES,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean normalized token cosine loss on exactly stages 2 and 3.

    Inputs are one [batch, tokens, channels] tensor per native stage.  Teacher
    tensors are detached even if a caller accidentally supplies tensors made
    outside a no-grad region.
    """
    if tuple(stages) != FUNCTIONAL_ANCHOR_STAGES:
        raise ValueError("functional anchor must use exactly native stages 2 and 3")
    if len(student_stages) != len(teacher_stages):
        raise ValueError("student/teacher stage count differs")
    losses = []
    metrics: dict[str, float] = {}
    for stage in stages:
        student, teacher = student_stages[stage], teacher_stages[stage].detach()
        if student.ndim != 3 or teacher.ndim != 3:
            raise ValueError("functional anchor requires [batch, tokens, channels] maps")
        if tuple(student.shape) != tuple(teacher.shape):
            raise ValueError(
                f"stage {stage + 1} token shape mismatch: {tuple(student.shape)} != {tuple(teacher.shape)}"
            )
        normalized_student = F.normalize(student.float(), dim=-1)
        normalized_teacher = F.normalize(teacher.float(), dim=-1)
        stage_loss = (1.0 - (normalized_student * normalized_teacher).sum(dim=-1)).mean()
        if not torch.isfinite(stage_loss):
            raise FloatingPointError(f"non-finite functional anchor at stage {stage + 1}")
        losses.append(stage_loss)
        metrics[f"functional_anchor_stage{stage + 1}"] = float(stage_loss.detach().cpu())
    loss = torch.stack(losses).mean()
    metrics["functional_anchor_total"] = float(loss.detach().cpu())
    return loss, metrics


def lambda_from_gradient_norms(
        task_norms: Sequence[float],
        functional_norms: Sequence[float],
        *,
        target_ratio: float = 0.10,
) -> tuple[float, float]:
    """Return the single fixed lambda from the median unweighted ratio."""
    if len(task_norms) != len(functional_norms) or not task_norms:
        raise ValueError("calibration requires matched non-empty norm lists")
    task = torch.as_tensor(task_norms, dtype=torch.float64)
    functional = torch.as_tensor(functional_norms, dtype=torch.float64)
    if not torch.isfinite(task).all() or not torch.isfinite(functional).all():
        raise FloatingPointError("non-finite calibration norm")
    if (task <= 0).any() or (functional <= 0).any():
        raise ValueError("calibration needs positive task and functional norms")
    raw_ratio = torch.median(functional / task).item()
    if raw_ratio <= 0.0:
        raise ValueError("functional gradient is zero; lambda is undefined")
    return float(target_ratio / raw_ratio), float(raw_ratio)


def bounded_config_mismatches(
        control: Mapping[str, object], candidate: Mapping[str, object],
) -> list[str]:
    """Compare matched arms while allowing only the feature-anchor identity."""
    keys = set(control) | set(candidate)
    return sorted(
        key for key in keys
        if key not in FUNCTIONAL_CONFIG_KEYS and control.get(key) != candidate.get(key)
    )


def require_source_only(dataset: str) -> None:
    if dataset != "VisA":
        raise ValueError("bounded functional-anchor test is source-only and requires dataset=VisA")
