import argparse
import collections
import contextlib
import csv
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_text_and_image_dataset
from utils import (
    calculate_seg_loss,
    configure_canonical_fp32,
    get_hybrid_soft_prompt_single_class_text_embedding,
    get_phase2b_global_text_features,
    get_multiple_adapted_single_class_text_embedding,
    get_soft_prompt_single_class_text_embedding,
    log_preflight,
    make_dataloader_generator,
    seed_worker,
)
from dataset.info import log_data_root
from model.adapter import (
    ACDCLIP
)
from model.checkpoint_utils import build_phase4_checkpoint
from model.clip import create_model
from model.h6.gated_early_stop import GateDecision, H6StructuralGateState, StructuralGateConfig
from model.h6.losses import (
    center_loss,
    concept_key_diversity_loss,
    factor_aware_center_loss,
    prototype_diagnostics,
    teacher_candidate_diagnostics,
    router_teacher_loss,
    routing_balance_loss,
    assigned_expert_loss, expert_advantage_loss, expert_etf_loss,
    dual_routing_balance_loss, expert_clip_anchor_loss, expert_radius_loss,
    expert_dead_counts, expert_patch_function_diagnostics, sum_loss_components,
    delta_t_diversity_loss, functional_factor_diversity_loss,
)
from model.h6.cluster_responsibility import cluster_responsibility_loss
from model.h6.utility_routing import (
    act_diagnostics, act_teacher,
    build_patch_targets, exploration_epsilon, utility_diagnostics,
    effective_number_act_loss,
    effective_number_utility_factor_loss,
    support_normalized_utility_router_loss,
    utility_factor_loss, utility_router_loss, utility_teacher,
)
from model.h6.specialization_trajectory import (
    aggregate_utility_records,
    capture_utility_record,
    teacher_sensitivity_grid,
    write_trajectory_artifacts,
)
from model.h6.losses import (
    build_semantic_roles,
    active_role_balanced_router_loss,
    factor_specific_residual_role_loss,
    actual_local_residual_loss,
)

def tensor_debug_stats(tensor):
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        return tensor
    with torch.no_grad():
        data = tensor.detach().float()
        finite = torch.isfinite(data)
        stats = {
            "shape": list(data.shape),
            "dtype": str(tensor.dtype),
            "finite": bool(finite.all().item()),
            "nan_count": int(torch.isnan(data).sum().item()),
            "posinf_count": int(torch.isposinf(data).sum().item()),
            "neginf_count": int(torch.isneginf(data).sum().item()),
        }
        if finite.any():
            finite_data = data[finite]
            stats.update({
                "min": float(finite_data.min().item()),
                "max": float(finite_data.max().item()),
                "mean": float(finite_data.mean().item()),
                "std": float(finite_data.std(unbiased=False).item()),
                "absmax": float(finite_data.abs().max().item()),
            })
        else:
            stats.update({
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "absmax": None,
            })
        return stats


def diagnostics_to_python(diagnostics):
    if torch.is_tensor(diagnostics):
        value = diagnostics.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(diagnostics, dict):
        return {key: diagnostics_to_python(value) for key, value in diagnostics.items()}
    if isinstance(diagnostics, (list, tuple)):
        return [diagnostics_to_python(value) for value in diagnostics]
    return diagnostics


def scalar_metric_value(value):
    """Convert tensor or Python-number metrics to a scalar float."""
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError("metric values must be scalar")
        return float(value.detach().float().item())
    return float(value)


def grad_accum_window_size(batch_index: int, total_batches: int, grad_accum_steps: int) -> int:
    """Return the actual divisor for the accumulation window containing a batch."""
    if batch_index < 1 or total_batches < batch_index or grad_accum_steps < 1:
        raise ValueError("invalid gradient-accumulation geometry")
    window_start = ((int(batch_index) - 1) // int(grad_accum_steps)) * int(grad_accum_steps) + 1
    return min(int(grad_accum_steps), int(total_batches) - window_start + 1)


def p1_v83_structure_diagnostics(h6_batch):
    """Measure factor/state separation without changing the training graph."""
    dynamic = h6_batch["dynamic_text"].detach().float()
    state = h6_batch["state_tokens"].detach().float()
    logits = h6_batch["factor_patch_logits"].detach().float()
    if dynamic.ndim != 5 or state.ndim != 4 or logits.ndim != 4:
        raise ValueError("unexpected P1-v8.3 factor diagnostic tensor rank")

    factor_count = logits.shape[-1]
    if dynamic.shape[2] != factor_count or state.shape[1] != factor_count:
        raise ValueError("inconsistent P1-v8.3 factor dimensions")

    embedding_rows = dynamic.movedim(2, 0).reshape(factor_count, -1)
    embedding_unit = F.normalize(embedding_rows, dim=-1)
    embedding_cosine = embedding_unit @ embedding_unit.transpose(0, 1)
    pair_mask = torch.triu(
        torch.ones(factor_count, factor_count, dtype=torch.bool, device=logits.device),
        diagonal=1,
    )
    embedding_pair_cosine = embedding_cosine[pair_mask]
    embedding_pair_l2 = torch.pdist(embedding_rows)
    singular_values = torch.linalg.svdvals(embedding_rows)
    singular_energy = singular_values.square()
    singular_prob = singular_energy / singular_energy.sum().clamp_min(1e-12)
    embedding_effective_rank = torch.exp(
        -(singular_prob * singular_prob.clamp_min(1e-12).log()).sum()
    )

    state_rows = state.movedim(1, 0).reshape(factor_count, -1)
    state_pair_l2 = torch.pdist(state_rows)

    logit_rows = logits.movedim(-1, 0).reshape(factor_count, -1)
    centered_logits = logit_rows - logit_rows.mean(dim=-1, keepdim=True)
    logit_unit = F.normalize(centered_logits, dim=-1)
    logit_pair_correlation = (logit_unit @ logit_unit.transpose(0, 1))[pair_mask]
    logit_pair_max_difference = (
        logit_rows[:, None, :] - logit_rows[None, :, :]
    ).abs().amax(dim=-1)[pair_mask]

    def functional_metrics(functions):
        rows = functions.movedim(-1, 0).reshape(factor_count, -1)
        centered = rows - rows.mean(dim=-1, keepdim=True)
        correlations = (
            F.normalize(centered, dim=-1) @ F.normalize(centered, dim=-1).transpose(0, 1)
        )[pair_mask]
        singular_values = torch.linalg.svdvals(rows)
        energy = singular_values.square()
        probabilities = energy / energy.sum().clamp_min(1e-12)
        effective_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        )
        return correlations, effective_rank

    residual_logits = h6_batch.get("factor_residual_logits")
    if residual_logits is None:
        residual_logits = logits
    residual_logits = residual_logits.detach().float()
    residual_correlation, residual_effective_rank = functional_metrics(residual_logits)
    absolute_correlation, absolute_effective_rank = functional_metrics(logits)

    return {
        "factor_embedding_pairwise_cosine_mean": embedding_pair_cosine.mean(),
        "factor_embedding_pairwise_cosine_max": embedding_pair_cosine.max(),
        "factor_embedding_pairwise_cosine_min": embedding_pair_cosine.min(),
        "factor_embedding_pairwise_l2_mean": embedding_pair_l2.mean(),
        "factor_embedding_pairwise_l2_min": embedding_pair_l2.min(),
        "factor_embedding_pairwise_l2_max": embedding_pair_l2.max(),
        "factor_embedding_effective_rank": embedding_effective_rank,
        "state_pairwise_l2_min": state_pair_l2.min(),
        "state_pairwise_l2_mean": state_pair_l2.mean(),
        "state_pairwise_l2_max": state_pair_l2.max(),
        "factor_patch_pairwise_correlation_mean": logit_pair_correlation.mean(),
        "factor_patch_pairwise_correlation_max": logit_pair_correlation.max(),
        "factor_patch_pairwise_correlation_min": logit_pair_correlation.min(),
        "factor_patch_pairwise_max_difference": logit_pair_max_difference.max(),
        "factor_patch_std_across_factors": logits.std(dim=-1, unbiased=False).mean(),
        "factor_patch_outputs_exactly_collapsed": (
            logit_pair_max_difference.max() == 0
        ).float(),
        "absolute_factor_correlation_mean": absolute_correlation.mean(),
        "absolute_factor_effective_rank": absolute_effective_rank,
        "residual_factor_correlation_mean": residual_correlation.mean(),
        "residual_factor_correlation_min": residual_correlation.min(),
        "residual_factor_correlation_max": residual_correlation.max(),
        "residual_factor_effective_rank": residual_effective_rank,
        "absolute_vs_residual_variance_ratio": (
            residual_logits.var(unbiased=False) / logits.var(unbiased=False).clamp_min(1e-12)
        ),
        "residual_positive_fraction": (residual_logits > 0).float().mean(),
        "residual_negative_fraction": (residual_logits < 0).float().mean(),
    }


def h6_drift_gradient_attribution(losses, parameter_groups):
    """Measure component gradients without writing ``.grad`` or mutating parameters."""
    grouped = {
        name: [parameter for parameter in parameters if parameter.requires_grad]
        for name, parameters in parameter_groups.items()
    }
    unique_parameters, seen = [], set()
    for parameters in grouped.values():
        for parameter in parameters:
            if id(parameter) not in seen:
                unique_parameters.append(parameter); seen.add(id(parameter))
    before_grads = {
        id(parameter): None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in unique_parameters
    }
    result = {}
    for name, payload in losses.items():
        loss, weight = payload
        differentiable = bool(torch.is_tensor(loss) and loss.requires_grad)
        active = bool(float(weight) != 0.0 and differentiable)
        entry = {
            "raw": float(loss.detach().float().item()),
            "weight": float(weight),
            "differentiable": differentiable,
            "active": active,
            "raw_gradient_norms": {},
            "weighted_gradient_norms": {},
        }
        if differentiable:
            raw_gradients = torch.autograd.grad(
                loss, unique_parameters, retain_graph=True, allow_unused=True,
            )
            raw_by_id = {
                id(parameter): gradient
                for parameter, gradient in zip(unique_parameters, raw_gradients)
            }
            for group_name, parameters in grouped.items():
                raw_squared = [
                    raw_by_id[id(parameter)].detach().float().pow(2).sum()
                    for parameter in parameters if raw_by_id[id(parameter)] is not None
                ]
                raw_norm = float(torch.stack(raw_squared).sum().sqrt().item()) if raw_squared else 0.0
                weighted_norm = abs(float(weight)) * raw_norm
                entry["raw_gradient_norms"][group_name] = raw_norm
                entry["weighted_gradient_norms"][group_name] = weighted_norm
                # Backward-compatible group fields have always been lambda-weighted.
                entry[group_name] = weighted_norm
        result[name] = entry
    for parameter in unique_parameters:
        before = before_grads[id(parameter)]
        after = parameter.grad
        if before is None:
            assert after is None, "drift attribution must not populate parameter.grad"
        else:
            assert after is not None and torch.equal(before, after), "drift attribution must not mutate parameter.grad"
    def _ratios(basis: str):
        key = f"{basis}_gradient_norms"
        task_shared = result.get("main_task", {}).get(key, {}).get("shared_semantic", 0.0)
        denominator = task_shared if task_shared > 1e-12 else None

        def _ratio(component: str):
            numerator = result.get(component, {}).get(key, {}).get("shared_semantic", 0.0)
            return None if denominator is None else numerator / denominator

        return {
            "assigned_to_task_shared_grad_ratio": _ratio("assigned_expert"),
            "anchor_to_task_shared_grad_ratio": _ratio("expert_anchor"),
            "center_to_task_shared_grad_ratio": _ratio("center"),
            "utility_factor_to_task_shared_grad_ratio": _ratio("utility_factor"),
            "utility_router_to_task_shared_grad_ratio": _ratio("utility_router"),
            "utility_act_to_task_shared_grad_ratio": _ratio("utility_act"),
            "total_aux_to_task_shared_grad_ratio": None if denominator is None else sum(
                entry.get(key, {}).get("shared_semantic", 0.0)
                for component, entry in result.items()
                if component != "main_task" and entry["active"]
            ) / denominator,
        }

    raw_ratios = _ratios("raw")
    weighted_ratios = _ratios("weighted")
    return {
        "components": result,
        "ratio_basis": "lambda_weighted",
        "ratios": weighted_ratios,
        "raw_ratios": raw_ratios,
        "weighted_ratios": weighted_ratios,
    }


def h6_drift_parameter_groups(model):
    """Relevant parameter groups for a no-step P1 utility attribution probe."""
    semantic_state_prefixes = (
        "normal_query", "abnormal_query", "prototype_attention", "normal_state_update",
        "abnormal_state_update", "state_to_context", "factor_id", "concept_slots",
    )
    vae_class_prefixes = ("class_vae", "class_to_context", "gamma_class")
    shared_semantic, prototype_state, vae_class_path = [], [], []
    for name, parameter in model.h6.semantic_core.named_parameters():
        if name.startswith(semantic_state_prefixes):
            prototype_state.append(parameter)
        elif name.startswith(vae_class_prefixes):
            vae_class_path.append(parameter)
        else:
            shared_semantic.append(parameter)
    paired = model.h6.paired_experts
    return {
        "shared_semantic": shared_semantic,
        "dynamic_context": [model.soft_prompt.ctx_normal, model.soft_prompt.ctx_abnormal],
        "prototype_state": prototype_state,
        "state_path": prototype_state,
        "vae_class_path": vae_class_path,
        "text_adapter": list(model.text_adapter.parameters()),
        "text_lora": list(model.text_adapter.parameters()),
        "expert_B": [] if paired is None else [paired.expert_B],
        "expert_state_projection": [] if paired is None else list(paired.state_projection.parameters()),
        "router_query_key": list(model.h6.router.parameters()),
        "router": list(model.h6.router.parameters()),
        "act_head": [] if model.h6.act_head is None else list(model.h6.act_head.parameters()),
    }


def pcgrad_project_two_task(main_gradients, factor_gradients):
    """Paper-faithful two-objective PCGrad projection over one parameter group."""
    if len(main_gradients) != len(factor_gradients):
        raise ValueError("PCGrad gradient lists must have identical lengths")
    dot = sum(
        (main.float() * factor.float()).sum()
        for main, factor in zip(main_gradients, factor_gradients)
    )
    main_norm_sq = sum(main.float().square().sum() for main in main_gradients)
    factor_norm_sq = sum(factor.float().square().sum() for factor in factor_gradients)
    conflict = bool(dot.detach().item() < 0.0 and main_norm_sq.detach().item() > 0.0 and factor_norm_sq.detach().item() > 0.0)
    if conflict:
        projected_main = [
            main - (dot / factor_norm_sq).to(main) * factor
            for main, factor in zip(main_gradients, factor_gradients)
        ]
        projected_factor = [
            factor - (dot / main_norm_sq).to(factor) * main
            for main, factor in zip(main_gradients, factor_gradients)
        ]
    else:
        projected_main = [gradient.clone() for gradient in main_gradients]
        projected_factor = [gradient.clone() for gradient in factor_gradients]
    return projected_main, projected_factor, {
        "conflict": conflict,
        "dot_main_factor": float(dot.detach().item()),
        "main_norm": float(main_norm_sq.detach().sqrt().item()),
        "factor_norm": float(factor_norm_sq.detach().sqrt().item()),
        "cos_main_factor": (
            float((dot / (main_norm_sq.sqrt() * factor_norm_sq.sqrt())).detach().item())
            if main_norm_sq.detach().item() > 0.0 and factor_norm_sq.detach().item() > 0.0
            else None
        ),
    }


def primary_anchored_factor_surgery(main_gradients, factor_gradients):
    """Remove only the factor component that conflicts with the primary gradient.

    This is a main-preserving auxiliary projection, not symmetric PCGrad.  Dots
    and norms are computed over the whole shared parameter group so the decision
    is optimizer-window and vector-group consistent.
    """
    if len(main_gradients) != len(factor_gradients):
        raise ValueError("primary/factor gradient lists must have identical lengths")
    dot = sum(
        (main.float() * factor.float()).sum()
        for main, factor in zip(main_gradients, factor_gradients)
    )
    main_norm_sq = sum(main.float().square().sum() for main in main_gradients)
    factor_norm_sq = sum(factor.float().square().sum() for factor in factor_gradients)
    conflict = bool(
        dot.detach().item() < 0.0
        and main_norm_sq.detach().item() > 0.0
        and factor_norm_sq.detach().item() > 0.0
    )
    unchanged_main = [gradient.clone() for gradient in main_gradients]
    if conflict:
        safe_factor = [
            factor - (dot / main_norm_sq).to(factor) * main
            for main, factor in zip(main_gradients, factor_gradients)
        ]
    else:
        safe_factor = [gradient.clone() for gradient in factor_gradients]

    safe_dot = sum(
        (main.float() * factor.float()).sum()
        for main, factor in zip(unchanged_main, safe_factor)
    )
    safe_factor_norm_sq = sum(
        factor.float().square().sum() for factor in safe_factor
    )
    removed_norm_sq = sum(
        (raw.float() - safe.float()).square().sum()
        for raw, safe in zip(factor_gradients, safe_factor)
    )
    main_change_norm_sq = sum(
        (raw.float() - unchanged.float()).square().sum()
        for raw, unchanged in zip(main_gradients, unchanged_main)
    )
    denominator = main_norm_sq.sqrt() * factor_norm_sq.sqrt()
    return unchanged_main, safe_factor, {
        "conflict": conflict,
        "dot_main_factor": float(dot.detach().item()),
        "dot_main_safe_factor": float(safe_dot.detach().item()),
        "main_norm": float(main_norm_sq.detach().sqrt().item()),
        "factor_norm": float(factor_norm_sq.detach().sqrt().item()),
        "safe_factor_norm": float(safe_factor_norm_sq.detach().sqrt().item()),
        "removed_factor_component_norm": float(removed_norm_sq.detach().sqrt().item()),
        "main_gradient_exact_change_norm": float(main_change_norm_sq.detach().sqrt().item()),
        "cos_main_factor": (
            float((dot / denominator).detach().item())
            if denominator.detach().item() > 1e-12
            else None
        ),
    }


def apply_primary_anchored_factor_correction(
    parameters, raw_factor_gradients, safe_factor_gradients
):
    """Replace raw factor contribution in ``.grad`` without touching main/router."""
    if not (
        len(parameters) == len(raw_factor_gradients) == len(safe_factor_gradients)
    ):
        raise ValueError("parameter and factor gradient lists must have identical lengths")
    correction_norm_sq = None
    for parameter, raw_factor, safe_factor in zip(
        parameters, raw_factor_gradients, safe_factor_gradients
    ):
        correction = safe_factor - raw_factor
        correction_sq = correction.detach().float().square().sum()
        correction_norm_sq = (
            correction_sq
            if correction_norm_sq is None
            else correction_norm_sq + correction_sq
        )
        if parameter.grad is None:
            parameter.grad = correction.clone()
        else:
            parameter.grad.add_(correction.to(parameter.grad))
    return 0.0 if correction_norm_sq is None else float(correction_norm_sq.sqrt().item())


def _gradient_vector_geometry(main_gradients, factor_gradients, router_gradients):
    def dot(left, right):
        return sum((a.float() * b.float()).sum() for a, b in zip(left, right))

    main_sq = dot(main_gradients, main_gradients)
    factor_sq = dot(factor_gradients, factor_gradients)
    router_sq = dot(router_gradients, router_gradients)
    mf = dot(main_gradients, factor_gradients)
    mr = dot(main_gradients, router_gradients)
    fr = dot(factor_gradients, router_gradients)
    combined_sq = factor_sq + router_sq + 2.0 * fr
    main_norm = main_sq.clamp_min(0.0).sqrt()
    factor_norm = factor_sq.clamp_min(0.0).sqrt()
    router_norm = router_sq.clamp_min(0.0).sqrt()
    combined_norm = combined_sq.clamp_min(0.0).sqrt()

    def cosine(numerator, left_norm, right_norm):
        denominator = left_norm * right_norm
        return None if denominator.detach().item() <= 1e-12 else float((numerator / denominator).detach().item())

    return {
        "main_norm": float(main_norm.detach().item()),
        "factor_norm": float(factor_norm.detach().item()),
        "router_norm": float(router_norm.detach().item()),
        "factor_to_main": None if main_norm.detach().item() <= 1e-12 else float((factor_norm / main_norm).detach().item()),
        "router_to_main": None if main_norm.detach().item() <= 1e-12 else float((router_norm / main_norm).detach().item()),
        "combined_to_main": None if main_norm.detach().item() <= 1e-12 else float((combined_norm / main_norm).detach().item()),
        "cos_main_factor": cosine(mf, main_norm, factor_norm),
        "cos_main_router": cosine(mr, main_norm, router_norm),
        "cos_factor_router": cosine(fr, factor_norm, router_norm),
        "cos_main_combined": cosine(mf + mr, main_norm, combined_norm),
    }


def write_json_atomic(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def current_git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def gated_abort_artifacts(
    *,
    save_path: str,
    epoch: int,
    decision,
    gate_state: H6StructuralGateState,
    gate_config: StructuralGateConfig,
    metrics: dict,
    diagnostics: dict,
    teacher_state: dict,
    sparse_ratio: float,
    routing_mode: str,
    alpha: float,
    trust_region_weight: float,
    latest_checkpoint_path: str | None,
    args,
    payload: dict | None = None,
) -> tuple[str, str, str]:
    report = {
        "epoch": int(epoch),
        "abort_reason": decision.abort_reason,
        "gate_counters": dict(gate_state.counters),
        "thresholds": gate_config.to_dict(),
        "metrics": diagnostics_to_python(metrics),
        "diagnostics": diagnostics_to_python(diagnostics),
        "per_level_decisions": decision.per_level,
        "routing_mode": routing_mode,
        "sparse_ratio": float(sparse_ratio),
        "teacher_state": diagnostics_to_python(teacher_state),
        "alpha": float(alpha),
        "trust_region_weight": float(trust_region_weight),
        "latest_checkpoint_path": latest_checkpoint_path,
        "command_configuration": vars(args),
        "git_head": current_git_head(),
        **H6StructuralGateState.decision_to_dict(decision),
    }
    pth_path = os.path.join(save_path, f"gated_abort_epoch_{epoch}.pth")
    json_path = os.path.join(save_path, f"gated_abort_epoch_{epoch}.json")
    marker_path = os.path.join(save_path, "GATED_TRAIN_ABORTED")
    torch.save(
        {
            "epoch": int(epoch),
            "reason": decision.abort_reason,
            "gate_report": report,
            "checkpoint": payload,
        },
        pth_path,
    )
    write_json_atomic(json_path, report)
    Path(marker_path).write_text(f"{decision.abort_reason}\n")
    return pth_path, json_path, marker_path


def factor_gradient_diagnostics(gradient: torch.Tensor | None) -> dict[str, torch.Tensor | None]:
    if gradient is None:
        return {
            "factor_grad_norms": None,
            "factor_grad_cos_mean": None,
            "factor_grad_cos_max": None,
            "factor_grad_cos_min": None,
            "factor_grad_l2_min": None,
        }
    grad = gradient.detach().float()
    if grad.ndim == 3:
        # Prototype-like gradients are [B,M,D]; average batch, keep factor M.
        grad = grad.mean(dim=0)
    if grad.ndim > 2:
        grad = grad.reshape(grad.shape[0], -1)
    norms = grad.norm(dim=-1)
    if grad.shape[0] <= 1:
        zero = grad.sum() * 0.0
        return {
            "factor_grad_norms": norms.detach(),
            "factor_grad_cos_mean": zero.detach(),
            "factor_grad_cos_max": zero.detach(),
            "factor_grad_cos_min": zero.detach(),
            "factor_grad_l2_min": zero.detach(),
        }
    normalized = F.normalize(grad, dim=-1)
    cosine = normalized @ normalized.T
    mask = ~torch.eye(grad.shape[0], device=grad.device, dtype=torch.bool)
    l2 = torch.cdist(grad, grad)[mask]
    return {
        "factor_grad_norms": norms.detach(),
        "factor_grad_cos_mean": cosine[mask].mean().detach(),
        "factor_grad_cos_max": cosine[mask].abs().max().detach(),
        "factor_grad_cos_min": cosine[mask].min().detach(),
        "factor_grad_l2_min": l2.min().detach(),
    }


def p1_v83_model_gradient_diagnostics(model, h6_batch, device):
    """Read current accumulated gradients for a diagnostics-only milestone."""
    core = model.h6.semantic_core
    result = factor_gradient_diagnostics(core.concept_slots.grad)
    dynamic_grad = h6_batch["dynamic_text"].grad
    result["dynamic_residual_grad_norms"] = (
        None if dynamic_grad is None
        else dynamic_grad.detach().float().norm(dim=3).mean(dim=(0, 1, 3)).detach()
    )
    factor_id_grad = core.factor_id_projection.weight.grad
    result["factor_id_projection_grad_norm"] = (
        None if factor_id_grad is None else factor_id_grad.detach().float().norm()
    )
    result["factor_generator_identity_grad_norm"] = None
    result["factor_generator_context_grad_norm"] = None
    result["factor_generator_head_grad_norms"] = None
    if getattr(core, "factor_generator_specialization_enabled", False):
        identity_grad = core.factor_id_embedding.grad
        if identity_grad is not None:
            result["factor_generator_identity_grad_norm"] = identity_grad.detach().float().norm()
        context_grads = [
            parameter.grad.detach().float().norm()
            for parameter in core.factor_id_to_context.parameters() if parameter.grad is not None
        ]
        if context_grads:
            result["factor_generator_context_grad_norm"] = torch.stack(context_grads).norm()
        result["factor_generator_head_grad_norms"] = torch.stack([
            head.weight.grad.detach().float().norm()
            if head.weight.grad is not None else torch.zeros((), device=device)
            for head in core.factor_output_heads
        ])
    shared_trunk_norms = [
        module_grad_norm(getattr(core, name, None)) for name in (
            "normal_state_update", "abnormal_state_update",
            "state_to_context_normal", "state_to_context_abnormal",
        )
    ]
    shared_trunk_norms = [value for value in shared_trunk_norms if value is not None]
    result["dynamic_prompt_shared_trunk_grad_norm"] = (
        torch.stack(shared_trunk_norms).norm() if shared_trunk_norms else None
    )
    vae = getattr(core, "class_vae", None)
    result["vae_mu_grad_norm"] = module_grad_norm(getattr(vae, "mu", None))
    result["vae_logvar_grad_norm"] = module_grad_norm(getattr(vae, "logvar", None))
    result["vae_decoder_grad_norm"] = module_grad_norm(getattr(vae, "decoder", None))
    result["class_to_context_grad_norm"] = module_grad_norm(getattr(core, "class_to_context", None))
    prototype_norms = [module_grad_norm(getattr(core, "prototype_attention", None))]
    prototype_norms = [value for value in prototype_norms if value is not None]
    result["prototype_modules_grad_norm"] = (
        torch.stack(prototype_norms).norm() if prototype_norms else None
    )
    result["router_grad_norm"] = module_grad_norm(model.h6.router)
    result["act_head_grad_norm"] = module_grad_norm(getattr(model.h6, "act_head", None))
    result["rho_gate_grad_norm"] = module_grad_norm(getattr(model.h6, "rho", None))
    result["phase2b_image_adapter_grad_norm"] = module_grad_norm(model.image_adapter)
    result["phase2b_text_adapter_grad_norm"] = module_grad_norm(model.text_adapter)
    dfg_module = (
        model.image_adapter["vision_text_gate"]
        if "vision_text_gate" in model.image_adapter
        else model.image_adapter["vision_text_q"]
        if "vision_text_q" in model.image_adapter else None
    )
    result["dfg_grad_norm"] = module_grad_norm(dfg_module)
    return result


def save_nonfinite_diagnostics(
        save_path: str,
        epoch_one_based: int,
        batch_idx: int,
        non_finite_loss_skips: int,
        model: ACDCLIP,
        tensors: dict,
        metadata: dict,
):
    diag_dir = os.path.join(save_path, "nonfinite_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    diag_path = os.path.join(
        diag_dir,
        f"epoch_{epoch_one_based:03d}_batch_{batch_idx:05d}_skip_{non_finite_loss_skips:04d}.pth",
    )
    payload = {
        "epoch": epoch_one_based,
        "batch_idx": batch_idx,
        "non_finite_loss_skips": non_finite_loss_skips,
        "metadata": metadata,
        "tensor_stats": {name: tensor_debug_stats(value) for name, value in tensors.items()},
        "dfg_diagnostics": diagnostics_to_python(model.get_dfg_diagnostics()),
    }
    torch.save(payload, diag_path)
    return diag_path


def has_non_finite_grad(optimizer: torch.optim.Optimizer) -> bool:
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return True
    return False


def first_nonfinite_trainable_parameter(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad and not torch.isfinite(param).all():
            return name, tensor_debug_stats(param)
    return None, None


def get_dfg_beta_for_epoch(
        epoch_one_based: int,
        dfg_beta_schedule: str,
        dfg_beta_target: float,
        dfg_beta: float,
) -> float:
    if dfg_beta_schedule == "fixed":
        return float(dfg_beta)
    if dfg_beta_schedule == "warmup010":
        if epoch_one_based <= 3:
            return 0.0
        if epoch_one_based <= 6:
            return min(0.05, float(dfg_beta_target))
        return float(dfg_beta_target)
    raise ValueError(f"Unknown dfg_beta_schedule: {dfg_beta_schedule}")


def mean_stats(stats_list):
    if not stats_list:
        return {}
    return {
        key: float(np.mean([stats[key] for stats in stats_list]))
        for key in stats_list[0].keys()
    }


def grad_norm_or_none(param: torch.nn.Parameter):
    if param.grad is None:
        return None
    return float(param.grad.detach().float().norm().item())


def module_grad_norm(module: torch.nn.Module | None) -> torch.Tensor | None:
    if module is None:
        return None
    squared_norms = [p.grad.detach().float().norm().square() for p in module.parameters() if p.grad is not None]
    if not squared_norms:
        return None
    return torch.stack(squared_norms).sum().sqrt()


def get_hybrid_alpha_for_epoch(epoch_one_based: int, hybrid_alpha_max: float, soft_prompt_freeze_epochs: int):
    if epoch_one_based <= soft_prompt_freeze_epochs:
        return 0.0
    warm_epoch = epoch_one_based - soft_prompt_freeze_epochs
    if warm_epoch == 1:
        return 0.25 * hybrid_alpha_max
    if warm_epoch == 2:
        return 0.50 * hybrid_alpha_max
    return hybrid_alpha_max


def get_optimizer_lr(optimizer: torch.optim.Optimizer, group_name: str):
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return group["lr"]
    return None


def apply_soft_prompt_lr_policy(optimizer: torch.optim.Optimizer, frozen: bool):
    for group in optimizer.param_groups:
        if group.get("name") == "soft_prompt" and "constant_lr" in group:
            group["lr"] = 0.0 if frozen else group["constant_lr"]


def detached_linear(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    bias = None if linear.bias is None else linear.bias.detach()
    return F.linear(x, linear.weight.detach(), bias)


def compute_hybrid_k_regularization(
        model: ACDCLIP,
        hard_text: torch.Tensor,
        soft_text: torch.Tensor,
        alpha: float,
):
    """Regularize hybrid text in DFG K-space while keeping W_K fixed for this loss."""
    if not hasattr(model, "image_adapter") or "vision_text_k" not in model.image_adapter:
        return torch.zeros((), device=soft_text.device), {}

    hard_anchor = hard_text.detach()
    main_for_k = F.normalize((1.0 - alpha) * hard_anchor + alpha * soft_text, dim=1)
    hard_states = hard_anchor.permute(0, 2, 1)  # [n_groups, 2, 768]
    main_states = main_for_k.permute(0, 2, 1)  # [n_groups, 2, 768]

    stage_losses = []
    stage_cosines = []
    stats = {}
    for stage_idx, key_proj in enumerate(model.image_adapter["vision_text_k"]):
        k_main = detached_linear(key_proj, main_states)
        k_hard = detached_linear(key_proj, hard_states).detach()
        k_main = F.normalize(k_main, dim=-1)
        k_hard = F.normalize(k_hard, dim=-1)
        cosine = F.cosine_similarity(k_main, k_hard, dim=-1)  # [n_groups, 2]
        stage_losses.append((1.0 - cosine).mean())
        stage_cosines.append(cosine.detach())

        prefix = f"stage{stage_idx + 1}"
        stats[f"{prefix}_k_cos_mean"] = float(cosine.detach().mean().item())
        stats[f"{prefix}_k_cos_normal"] = float(cosine.detach()[:, 0].mean().item())
        stats[f"{prefix}_k_cos_abnormal"] = float(cosine.detach()[:, 1].mean().item())
        stats[f"{prefix}_k_loss"] = float((1.0 - cosine.detach()).mean().item())

    k_loss = torch.stack(stage_losses).mean()
    all_cosines = torch.stack(stage_cosines, dim=0)  # [stages, n_groups, 2]
    stats.update({
        "k_cos_mean": float(all_cosines.mean().item()),
        "k_cos_normal": float(all_cosines[..., 0].mean().item()),
        "k_cos_abnormal": float(all_cosines[..., 1].mean().item()),
    })
    return k_loss, stats


def clip_module_grad(module: torch.nn.Module, grad_clip_norm: float):
    if grad_clip_norm is None or grad_clip_norm <= 0:
        return None
    return nn.utils.clip_grad_norm_(module.parameters(), grad_clip_norm)


def train(
        model: ACDCLIP,
        dataset_name: str,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: str | torch.device,
        total_epoch: int,
        save_path: str,
        logger: logging.Logger,
        use_amp: bool = False,
        dfg_beta_schedule: str = "fixed",
        dfg_beta_target: float = 0.10,
        dfg_beta: float = 0.10,
        non_finite_loss_abort_threshold: int = 20,
        lambda_kg: float = 1e-3,
        lambda_k: float = 0.0,
        hybrid_alpha_max: float = 0.2,
        soft_prompt_freeze_epochs: int = 3,
        grad_clip_norm: float = 1.0,
):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(0, total_epoch):
        epoch_one_based = epoch + 1
        use_hybrid_soft_prompt = bool(getattr(model, "use_hybrid_soft_prompt", False))
        use_soft_prompt = bool(getattr(model, "use_soft_prompt", False))
        soft_prompt_frozen = False
        hybrid_alpha_current = 0.0
        if use_hybrid_soft_prompt:
            hybrid_alpha_current = get_hybrid_alpha_for_epoch(
                epoch_one_based,
                hybrid_alpha_max=hybrid_alpha_max,
                soft_prompt_freeze_epochs=soft_prompt_freeze_epochs,
            )
            soft_prompt_frozen = epoch_one_based <= soft_prompt_freeze_epochs
            model.hybrid_alpha_current = hybrid_alpha_current
            model.soft_prompt.requires_grad_(not soft_prompt_frozen)
            model.text_adapter.requires_grad_(True)
            apply_soft_prompt_lr_policy(optimizer, soft_prompt_frozen)
        beta_current = get_dfg_beta_for_epoch(
            epoch_one_based,
            dfg_beta_schedule,
            dfg_beta_target,
            dfg_beta,
        )
        model.set_dfg_beta(beta_current)
        logger.info(f"training epoch {epoch_one_based} / {total_epoch}")
        logger.info(
            "dfg_beta_state epoch=%d dfg_ss2d_fusion=%s dfg_beta_schedule=%s "
            "dfg_beta_target=%s dfg_beta_current=%s",
            epoch_one_based,
            model.dfg_ss2d_fusion,
            dfg_beta_schedule,
            dfg_beta_target,
            model.dfg_beta,
        )
        if use_hybrid_soft_prompt:
            logger.info(
                "hybrid_state epoch=%d effective_prompt_mode=%s effective_alpha=%s "
                "hard_branch_lora_used=True soft_branch_lora_used=False soft_prompt_frozen=%s "
                "soft_prompt_freeze_epochs=%d lambda_kg=%s lambda_k=%s grad_clip_norm=%s "
                "image_lr=%s text_lr=%s soft_lr=%s",
                epoch_one_based,
                getattr(model, "prompt_mode", "hybrid"),
                hybrid_alpha_current,
                soft_prompt_frozen,
                soft_prompt_freeze_epochs,
                lambda_kg,
                lambda_k,
                grad_clip_norm,
                get_optimizer_lr(optimizer, "image_adapter"),
                get_optimizer_lr(optimizer, "text_adapter"),
                get_optimizer_lr(optimizer, "soft_prompt"),
            )
        loss_list = []
        loss_main_list = []
        seg_loss_list = []
        cls_loss_list = []
        kg_loss_list = []
        k_loss_list = []
        soft_prompt_stats_list = []
        k_reg_stats_list = []
        soft_prompt_grad_stats_list = []
        non_finite_loss_skips = 0
        non_finite_grad_skips = 0
        tqdm_train_loader = tqdm(train_loader)
        for batch_idx, input_data in enumerate(tqdm_train_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            class_names = input_data["class_name"]
            # get adapted text embedding
            epoch_text_feature_dict = {}
            kg_losses = []
            k_losses = []
            batch_soft_stats = []
            batch_k_stats = []
            for class_name in list(set(class_names)):
                if use_hybrid_soft_prompt:
                    if lambda_k > 0:
                        (
                            text_embedding_levels,
                            kg_loss_class,
                            soft_stats,
                            components,
                        ) = get_hybrid_soft_prompt_single_class_text_embedding(
                            model, dataset_name, class_name, device, return_kg=True, return_components=True
                        )
                        k_loss_class, k_stats = compute_hybrid_k_regularization(
                            model,
                            components["hard_text"],
                            components["soft_text"],
                            hybrid_alpha_current,
                        )
                        k_losses.append(k_loss_class)
                        batch_k_stats.append(k_stats)
                    else:
                        text_embedding_levels, kg_loss_class, soft_stats = get_hybrid_soft_prompt_single_class_text_embedding(
                            model, dataset_name, class_name, device, return_kg=True
                        )
                    kg_losses.append(kg_loss_class)
                    batch_soft_stats.append(soft_stats)
                elif use_soft_prompt:
                    text_embedding_levels, kg_loss_class, soft_stats = get_soft_prompt_single_class_text_embedding(
                        model, dataset_name, class_name, device, return_kg=True
                    )
                    kg_losses.append(kg_loss_class)
                    batch_soft_stats.append(soft_stats)
                else:
                    text_embedding_levels = get_multiple_adapted_single_class_text_embedding(
                        model, dataset_name, class_name, device
                    )
                epoch_text_feature_dict[class_name] = text_embedding_levels  # [n_groups, 768, 2]
            epoch_text_features = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )  # [bs, n_groups, 768, 2]
            epoch_text_features = epoch_text_features.permute(1, 0, 2, 3)  # [n_groups, bs, 768, 2]
            if kg_losses:
                kg_loss = torch.stack(kg_losses).mean()
                soft_prompt_stats_list.extend(batch_soft_stats)
            else:
                kg_loss = torch.zeros((), device=device)
            if k_losses:
                k_loss = torch.stack(k_losses).mean()
                k_reg_stats_list.extend(batch_k_stats)
            else:
                k_loss = torch.zeros((), device=device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                seg_tokens, det_tokens = model(image)  # [bs, patch_size, 768] * n_groups, [bs, 768] * n_groups
                seg_features = torch.stack(seg_tokens, dim=0)  # [n_groups, bs, patch_num, 768]
                det_features = torch.stack(det_tokens, dim=0)  # [n_groups, bs, 768]
                cls_pred = [
                    torch.matmul(
                        det_features[i].unsqueeze(dim=1),  # [bs, 1, 768]
                        epoch_text_features[i],  # [bs, 768, 2]
                    ).squeeze(1)
                    for i in range(det_features.shape[0])
                ]  # [bs, 2] * n_groups
                cls_pred = torch.stack(cls_pred, dim=0).mean(dim=0)  # [bs, 2]
                cls_loss = F.cross_entropy(cls_pred, label)
                # [bs, 2, img_size, img_size]
                seg_pred = model.vision_text_fusion_gate_seg(seg_features, epoch_text_features)
                seg_loss = calculate_seg_loss(seg_pred, mask)
                loss_main = cls_loss + seg_loss
                loss = loss_main + lambda_kg * kg_loss + lambda_k * k_loss
            if not torch.isfinite(loss).all():
                non_finite_loss_skips += 1
                diag_path = save_nonfinite_diagnostics(
                    save_path=save_path,
                    epoch_one_based=epoch_one_based,
                    batch_idx=batch_idx,
                    non_finite_loss_skips=non_finite_loss_skips,
                    model=model,
                    tensors={
                        "image": image,
                        "mask": mask,
                        "epoch_text_features": epoch_text_features,
                        "seg_features": seg_features,
                        "det_features": det_features,
                        "cls_pred": cls_pred,
                        "cls_loss": cls_loss,
                        "seg_pred": seg_pred,
                        "seg_loss": seg_loss,
                        "kg_loss": kg_loss,
                        "k_loss": k_loss,
                        "loss": loss,
                    },
                    metadata={
                        "use_amp": use_amp,
                        "dfg_ss2d_fusion": model.dfg_ss2d_fusion,
                        "dfg_beta_schedule": dfg_beta_schedule,
                        "dfg_beta_target": dfg_beta_target,
                        "dfg_beta_current": model.dfg_beta,
                        "dfg_weight_residual_fp32": model.dfg_weight_residual_fp32,
                        "prompt_mode": getattr(model, "prompt_mode", "hard"),
                        "use_soft_prompt": getattr(model, "use_soft_prompt", False),
                        "use_hybrid_soft_prompt": getattr(model, "use_hybrid_soft_prompt", False),
                        "hybrid_alpha_current": getattr(model, "hybrid_alpha_current", 0.0),
                        "soft_prompt_frozen": soft_prompt_frozen,
                        "lambda_kg": lambda_kg,
                        "lambda_k": lambda_k,
                        "class_names": list(class_names),
                        "labels": label.detach().cpu().tolist(),
                    },
                )
                logger.warning(
                    "non-finite loss at epoch %d batch=%d skip=%d "
                    "loss_finite=%s cls_loss_finite=%s seg_loss_finite=%s "
                    "cls_pred_finite=%s seg_pred_finite=%s diag=%s",
                    epoch_one_based,
                    batch_idx,
                    non_finite_loss_skips,
                    bool(torch.isfinite(loss).all().item()),
                    bool(torch.isfinite(cls_loss).all().item()),
                    bool(torch.isfinite(seg_loss).all().item()),
                    bool(torch.isfinite(cls_pred).all().item()),
                    bool(torch.isfinite(seg_pred).all().item()),
                    diag_path,
                )
                optimizer.zero_grad(set_to_none=True)
                if (
                        non_finite_loss_abort_threshold >= 0
                        and non_finite_loss_skips > non_finite_loss_abort_threshold
                ):
                    raise RuntimeError(
                        "Aborting training because non_finite_loss="
                        f"{non_finite_loss_skips} exceeded threshold "
                        f"{non_finite_loss_abort_threshold} at epoch {epoch_one_based}. "
                        f"Latest diagnostics: {diag_path}"
                    )
                continue
            seg_loss_list.append(seg_loss.item())
            kg_loss_list.append(kg_loss.item())
            k_loss_list.append(k_loss.item())
            # backward
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if has_non_finite_grad(optimizer):
                logger.warning("non-finite gradient at epoch %d; skipping optimizer step", epoch + 1)
                non_finite_grad_skips += 1
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue
            if use_soft_prompt or use_hybrid_soft_prompt:
                soft_prompt_grad_stats_list.append({
                    "ctx_grad_norm_normal": grad_norm_or_none(model.soft_prompt.ctx_normal),
                    "ctx_grad_norm_abnormal": grad_norm_or_none(model.soft_prompt.ctx_abnormal),
                })
            # clip gradient
            clip_module_grad(model.image_adapter, grad_clip_norm)
            if use_hybrid_soft_prompt:
                clip_module_grad(model.text_adapter, grad_clip_norm)
                if not soft_prompt_frozen:
                    clip_module_grad(model.soft_prompt, grad_clip_norm)
            elif use_soft_prompt:
                clip_module_grad(model.soft_prompt, grad_clip_norm)
            else:
                clip_module_grad(model.text_adapter, grad_clip_norm)
            # update parameters
            scaler.step(optimizer)
            scaler.update()
            bad_param_name, bad_param_stats = first_nonfinite_trainable_parameter(model)
            if bad_param_name is not None:
                diag_path = save_nonfinite_diagnostics(
                    save_path=save_path,
                    epoch_one_based=epoch_one_based,
                    batch_idx=batch_idx,
                    non_finite_loss_skips=non_finite_loss_skips,
                    model=model,
                    tensors={
                        bad_param_name: dict(model.named_parameters())[bad_param_name],
                        "loss": loss,
                        "cls_loss": cls_loss,
                        "seg_loss": seg_loss,
                        "kg_loss": kg_loss,
                        "k_loss": k_loss,
                        "cls_pred": cls_pred,
                        "seg_pred": seg_pred,
                    },
                    metadata={
                        "reason": "non_finite_trainable_parameter_after_optimizer_step",
                        "bad_param_name": bad_param_name,
                        "bad_param_stats": bad_param_stats,
                        "use_amp": use_amp,
                        "dfg_ss2d_fusion": model.dfg_ss2d_fusion,
                        "dfg_beta_schedule": dfg_beta_schedule,
                        "dfg_beta_target": dfg_beta_target,
                        "dfg_beta_current": model.dfg_beta,
                        "dfg_weight_residual_fp32": model.dfg_weight_residual_fp32,
                        "prompt_mode": getattr(model, "prompt_mode", "hard"),
                        "use_soft_prompt": getattr(model, "use_soft_prompt", False),
                        "use_hybrid_soft_prompt": getattr(model, "use_hybrid_soft_prompt", False),
                        "hybrid_alpha_current": getattr(model, "hybrid_alpha_current", 0.0),
                        "soft_prompt_frozen": soft_prompt_frozen,
                        "lambda_kg": lambda_kg,
                        "lambda_k": lambda_k,
                    },
                )
                logger.error(
                    "non-finite trainable parameter after optimizer step at epoch=%d batch=%d "
                    "param=%s stats=%s diag=%s",
                    epoch_one_based,
                    batch_idx,
                    bad_param_name,
                    bad_param_stats,
                    diag_path,
                )
                raise RuntimeError(
                    "Aborting training because trainable parameter became non-finite after "
                    f"optimizer step: {bad_param_name}. Diagnostics: {diag_path}"
                )
            loss_main_list.append(loss_main.item())
            loss_list.append(loss.item())
            cls_loss_list.append(cls_loss.item())
            postfix = {
                "epoch": f"{epoch + 1} / {total_epoch}",
                "loss": f"{loss.item():.4f}",
                "main_loss": f"{loss_main.item():.4f}",
                "det_loss": f"{cls_loss.item():.4f}",
                "seg_loss": f"{seg_loss.item():.4f}",
                "mean_seg_loss": f"{np.mean(seg_loss_list):.4f}",
                "mean_loss": f"{np.mean(loss_list):.4f}",
            }
            if use_hybrid_soft_prompt:
                postfix["kg_loss"] = f"{kg_loss.item():.5f}"
                if lambda_k > 0:
                    postfix["k_loss"] = f"{k_loss.item():.5f}"
                    postfix["wk_loss"] = f"{(lambda_k * k_loss).item():.5f}"
                postfix["alpha"] = f"{hybrid_alpha_current:.3f}"
                postfix["frozen"] = soft_prompt_frozen
                postfix["text_lr"] = get_optimizer_lr(optimizer, "text_adapter")
                postfix["image_lr"] = get_optimizer_lr(optimizer, "image_adapter")
                postfix["soft_lr"] = get_optimizer_lr(optimizer, "soft_prompt")
            elif use_soft_prompt:
                postfix["kg_loss"] = f"{kg_loss.item():.5f}"
                postfix["image_lr"] = get_optimizer_lr(optimizer, "image_adapter")
                postfix["soft_lr"] = get_optimizer_lr(optimizer, "soft_prompt")
            else:
                postfix["text_lr"] = get_optimizer_lr(optimizer, "text_adapter")
                postfix["image_lr"] = get_optimizer_lr(optimizer, "image_adapter")
            tqdm_train_loader.set_postfix(postfix)
        logger.info(
            "mean_loss=%s, mean_loss_main=%s, mean_cls_loss=%s, mean_seg_loss=%s",
            np.mean(loss_list),
            np.mean(loss_main_list),
            np.mean(cls_loss_list),
            np.mean(seg_loss_list),
        )
        if use_soft_prompt or use_hybrid_soft_prompt:
            logger.info(
                "soft_prompt_epoch epoch=%d prompt_mode=%s mean_kg_loss=%s lambda_kg=%s "
                "mean_k_loss=%s weighted_k_loss=%s lambda_k=%s k_stats=%s "
                "hybrid_alpha=%s effective_prompt_mode=%s effective_alpha=%s "
                "hard_branch_lora_used=%s soft_branch_lora_used=False "
                "soft_prompt_frozen=%s grad_clip_norm=%s stats=%s ctx_stats=%s grad_stats=%s "
                "text_encoder_frozen=True text_lora_used=%s",
                epoch + 1,
                getattr(model, "prompt_mode", "soft"),
                float(np.mean(kg_loss_list)) if kg_loss_list else None,
                lambda_kg,
                float(np.mean(k_loss_list)) if k_loss_list else None,
                float(lambda_k * np.mean(k_loss_list)) if k_loss_list else None,
                lambda_k,
                mean_stats(k_reg_stats_list),
                getattr(model, "hybrid_alpha_current", 0.0),
                getattr(model, "prompt_mode", "soft"),
                getattr(model, "hybrid_alpha_current", 0.0),
                use_hybrid_soft_prompt,
                soft_prompt_frozen,
                grad_clip_norm,
                mean_stats(soft_prompt_stats_list),
                model.soft_prompt.stats(),
                mean_stats([s for s in soft_prompt_grad_stats_list if None not in s.values()]),
                use_hybrid_soft_prompt,
            )
        logger.info(
            "skip_counts epoch=%d non_finite_loss=%d non_finite_grad=%d",
            epoch + 1,
            non_finite_loss_skips,
            non_finite_grad_skips,
        )
        if model.dfg_mode == "attn":
            diagnostics = model.get_dfg_diagnostics()
            for key, value in diagnostics.items():
                if value is None:
                    continue
                if torch.is_tensor(value) and value.ndim > 0:
                    value = value.tolist()
                elif torch.is_tensor(value):
                    value = value.item()
                logger.info("dfg_diag epoch=%d %s=%s", epoch + 1, key, value)
        scheduler.step()
        if use_hybrid_soft_prompt:
            apply_soft_prompt_lr_policy(optimizer, soft_prompt_frozen)
        ckp_path = os.path.join(save_path, f"adapter_{epoch + 1}.pth")
        model_dict = {
            "epoch": epoch + 1,
            "n_groups": model.n_groups,
            "dfg_mode": model.dfg_mode,
            "dfg_attn_dim": model.dfg_attn_dim,
            "dfg_attn_tau": model.dfg_attn_tau,
            "use_ss2d_dfg": model.use_ss2d_dfg,
            "dfg_gamma_max": model.dfg_gamma_max,
            "dfg_ss2d_fusion": model.dfg_ss2d_fusion,
            "dfg_beta": model.dfg_beta,
            "dfg_beta_schedule": dfg_beta_schedule,
            "dfg_beta_target": dfg_beta_target,
            "dfg_beta_current": model.dfg_beta,
            "dfg_weight_residual_fp32": model.dfg_weight_residual_fp32,
            "prompt_mode": getattr(model, "prompt_mode", "hard"),
            "use_soft_prompt": bool(getattr(model, "use_soft_prompt", False)),
            "use_hybrid_soft_prompt": bool(getattr(model, "use_hybrid_soft_prompt", False)),
            "soft_prompt_ctx_len": getattr(model, "soft_prompt_ctx_len", 4),
            "soft_prompt_init": getattr(model, "soft_prompt_init", "phrase"),
            "soft_prompt_init_phrase": getattr(model, "soft_prompt_init_phrase", "a photo of a"),
            "hybrid_alpha_current": getattr(model, "hybrid_alpha_current", 0.0),
            "hybrid_alpha_max": hybrid_alpha_max,
            "soft_prompt_freeze_epochs": soft_prompt_freeze_epochs,
            "grad_clip_norm": grad_clip_norm,
            "lambda_kg": lambda_kg,
            "lambda_k": lambda_k,
            "k_reg_detached_wk": bool(lambda_k > 0),
            "k_reg_per_stage": bool(lambda_k > 0),
            "text_adapter": model.text_adapter.state_dict(),
            "image_adapter": model.image_adapter.state_dict()
        }
        if use_soft_prompt or use_hybrid_soft_prompt:
            model_dict["soft_prompt"] = model.soft_prompt.state_dict()
        torch.save(model_dict, ckp_path)
    return model


def set_phase4_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_h6_vae_beta(
        epoch_one_based: int,
        beta_max: float,
        zero_epochs: int = 0,
        warmup_epochs: int = 4,
) -> float:
    zero_epochs = max(0, int(zero_epochs))
    warmup_epochs = max(1, int(warmup_epochs))
    if zero_epochs == 0:
        if epoch_one_based <= 1:
            return 0.0
        return min(float(beta_max), float(epoch_one_based - 1) * float(beta_max) / float(warmup_epochs))
    if epoch_one_based <= zero_epochs:
        return 0.0
    warmup_epoch = int(epoch_one_based) - zero_epochs
    return min(float(beta_max), float(warmup_epoch) * float(beta_max) / float(warmup_epochs))


def linear_ramp_weight(epoch_one_based: int, start_epoch: int, warmup_epochs: int, maximum: float) -> float:
    if maximum <= 0.0 or epoch_one_based < start_epoch:
        return 0.0
    warmup_epochs = max(1, int(warmup_epochs))
    step = int(epoch_one_based) - int(start_epoch) + 1
    return min(float(maximum), float(maximum) * float(step) / float(warmup_epochs))


def router_specialization_failed(
        sparse_ratio: float,
        sparse_dead_factors: torch.Tensor,
        unique_topk_pairs: torch.Tensor,
        max_sparse_dead_factors: int,
        min_unique_topk_pairs: int,
) -> bool:
    if float(sparse_ratio) < 0.50:
        return False
    return bool(
        (sparse_dead_factors >= int(max_sparse_dead_factors) + 1).any()
        or (unique_topk_pairs <= int(min_unique_topk_pairs) - 1).any()
    )


def _phase4_autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _set_soft_prompt_lr(optimizer: torch.optim.Optimizer, frozen: bool) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == "soft_prompt":
            group["lr"] = 0.0 if frozen else group["constant_lr"]


def _h6_optimizer_groups(model: ACDCLIP, args) -> list[dict]:
    groups = [
        {"name": "text_adapter", "params": list(model.text_adapter.parameters()), "lr": args.text_lr},
        {"name": "image_adapter", "params": list(model.image_adapter.parameters()), "lr": args.image_lr},
        {
            "name": "soft_prompt",
            "params": list(model.soft_prompt.parameters()),
            "lr": 0.0,
            "constant_lr": args.soft_prompt_lr,
            "weight_decay": 0.0,
        },
    ]
    partition_by_id = {}
    for partition_name, params in model.h6.parameter_partitions().items():
        for parameter in params:
            partition_by_id[id(parameter)] = partition_name
    named_h6 = dict(model.h6.named_parameters())
    grouped = {}
    for name, parameter in named_h6.items():
        if not parameter.requires_grad:
            continue
        partition = partition_by_id.get(id(parameter))
        if partition is None:
            raise RuntimeError(f"unpartitioned H6 parameter: {name}")
        lr = 5e-5 if partition == "h6_dynamic_prompt" else 1e-4
        zero_decay = (
            parameter.ndim < 2
            or "norm" in name.lower()
            or "embedding" in name.lower()
            or "concept_slots" in name
            or "raw" in name
        )
        key = (partition, lr, 0.0 if zero_decay else 0.01)
        grouped.setdefault(key, []).append(parameter)
    for (partition, lr, weight_decay), params in grouped.items():
        groups.append({"name": partition, "params": params, "lr": lr, "weight_decay": weight_decay})
    return groups


def _phase2b_config_from_args(args) -> dict:
    return {
        "git_sha": current_git_head(),
        "precision": args.precision,
        "tf32_enabled": bool(
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
        ),
        "amp_enabled": bool(args.amp),
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "n_groups": args.n_groups,
        "image_levels": [8, 16, 24] if args.n_groups == 3 else None,
        "text_levels": [4, 8, 12] if args.n_groups == 3 else None,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "conv_lora_rank": args.conv_lora_rank,
        "conv_lora_alpha": args.conv_lora_alpha,
        "conv_kernel_size_list": list(args.conv_kernel_size_list),
        "image_adapt_weight": args.image_adapt_weight,
        "text_adapt_weight": args.text_adapt_weight,
        "dfg_mode": args.dfg_mode,
        "dfg_attn_dim": args.dfg_attn_dim,
        "dfg_attn_tau": args.dfg_attn_tau,
        "use_ss2d_dfg": args.use_ss2d_dfg,
        "dfg_ss2d_fusion": args.dfg_ss2d_fusion,
        "dfg_beta": args.dfg_beta,
        "dfg_beta_schedule": args.dfg_beta_schedule,
        "dfg_beta_target": args.dfg_beta_target,
        "image_lr": args.image_lr,
        "text_lr": args.text_lr,
        "soft_prompt_lr": args.soft_prompt_lr,
        "lr_gamma": args.lr_gamma,
        "grad_clip_norm": args.grad_clip_norm,
        "grad_checkpointing": args.grad_checkpointing,
        "hybrid_alpha_max": args.hybrid_alpha_max,
        "soft_prompt_freeze_epochs": args.soft_prompt_freeze_epochs,
        "h6_router_soft_epochs": args.h6_router_soft_epochs,
        "h6_dense_routing_epochs": args.h6_router_soft_epochs,
        "h6_sparse_start_epoch": args.h6_router_soft_epochs + 1,
        "h6_sparse_transition_epochs": args.h6_sparse_transition_epochs,
        "h6_sparse_full_epoch": args.h6_router_soft_epochs + args.h6_sparse_transition_epochs,
        "h6_center_factor_aware": args.h6_center_factor_aware,
        "h6_center_detach_assignment": args.h6_center_detach_assignment,
        "h6_center_margin": args.h6_center_margin,
        "h6_kl_zero_epochs": args.h6_kl_zero_epochs,
        "h6_kl_warmup_epochs": args.h6_kl_warmup_epochs,
        "h6_kl_free_bits": args.h6_kl_free_bits,
        "beta_h6_vae_kl": args.beta_h6_vae_kl,
        "lambda_h6_delta_div": args.lambda_h6_delta_div,
        "lambda_h6_func_div": args.lambda_h6_func_div,
        "h6_cluster_responsibility": args.h6_cluster_responsibility,
        "h6_cluster_centroid_path": args.h6_cluster_centroid_path,
        "h6_cluster_temperature": args.h6_cluster_temperature,
        "h6_lambda_cluster_resp": args.h6_lambda_cluster_resp,
        "h6_cluster_centroid_sha256": getattr(args, "h6_cluster_centroid_sha256", None),
        "h6_vae_class_ratio": args.h6_vae_class_ratio,
        "lambda_h6_concept_key_diversity": args.lambda_h6_concept_key_diversity,
        "h6_concept_key_cosine_margin": args.h6_concept_key_cosine_margin,
        "h6_concept_key_diversity_start_epoch": args.h6_concept_key_diversity_start_epoch,
        "h6_concept_key_diversity_warmup_epochs": args.h6_concept_key_diversity_warmup_epochs,
        "h6_slot_init_enabled": args.h6_slot_init_enabled,
        "h6_slot_init_scale": args.h6_slot_init_scale,
        "h6_slot_init_seed_offset": args.h6_slot_init_seed_offset,
        "h6_factor_grad_diagnostics": args.h6_factor_grad_diagnostics,
        "h6_late_factor_identity_enabled": args.h6_late_factor_identity_enabled,
        "h6_factor_id_scale": args.h6_factor_id_scale,
        "h6_factor_id_max_ratio": args.h6_factor_id_max_ratio,
        "h6_factor_generator_specialization_enabled": args.h6_factor_generator_specialization_enabled,
        "h6_factor_head_init_scale": args.h6_factor_head_init_scale,
        "h6_factor_local_dynamic_mix": args.h6_factor_local_dynamic_mix,
        "h6_router_query_mode": args.h6_router_query_mode,
        "h6_router_query_global_weight": args.h6_router_query_global_weight,
        "h6_router_local_bypass_scale": args.h6_router_local_bypass_scale,
        "h6_router_local_bypass_max_ratio": args.h6_router_local_bypass_max_ratio,
        "h6_router_local_projection_seed_offset": args.h6_router_local_projection_seed_offset,
        "h6_router_key_anchor_enabled": args.h6_router_key_anchor_enabled,
        "h6_router_key_anchor_seed_offset": args.h6_router_key_anchor_seed_offset,
        "h6_router_key_adaptation_initial_ratio": args.h6_router_key_adaptation_initial_ratio,
        "h6_router_key_adaptation_max_ratio": args.h6_router_key_adaptation_max_ratio,
        "h6_factor_context_anchor_enabled": args.h6_factor_context_anchor_enabled,
        "h6_factor_context_anchor_seed_offset": args.h6_factor_context_anchor_seed_offset,
        "h6_factor_context_adaptation_initial_ratio": args.h6_factor_context_adaptation_initial_ratio,
        "h6_factor_context_adaptation_max_ratio": args.h6_factor_context_adaptation_max_ratio,
        "h6_factor_identity_tangent_projection_enabled": args.h6_factor_identity_tangent_projection_enabled,
        "lambda_h6_dynamic_mean_anchor": args.lambda_h6_dynamic_mean_anchor,
        "h6_dynamic_mean_anchor_min_cosine": args.h6_dynamic_mean_anchor_min_cosine,
        "h6_dynamic_mean_anchor_start_epoch": args.h6_dynamic_mean_anchor_start_epoch,
        "h6_dynamic_mean_anchor_warmup_epochs": args.h6_dynamic_mean_anchor_warmup_epochs,
        "h6_load_bias_enabled": args.h6_load_bias_enabled,
        "h6_load_bias_momentum": args.h6_load_bias_momentum,
        "h6_load_bias_step": args.h6_load_bias_step,
        "h6_load_bias_max": args.h6_load_bias_max,
        "lambda_h6_router_teacher": args.lambda_h6_router_teacher,
        "h6_router_teacher_temperature": args.h6_router_teacher_temperature,
        "h6_router_teacher_start_epoch": args.h6_router_teacher_start_epoch,
        "h6_router_teacher_warmup_epochs": args.h6_router_teacher_warmup_epochs,
        "h6_router_teacher_mode": args.h6_router_teacher_mode,
        "h6_teacher_confidence_gate": args.h6_teacher_confidence_gate,
        "h6_teacher_entropy_threshold": args.h6_teacher_entropy_threshold,
        "h6_teacher_prob_std_threshold": args.h6_teacher_prob_std_threshold,
        "h6_router_failure_patience": args.h6_router_failure_patience,
        "h6_router_max_sparse_dead_factors": args.h6_router_max_sparse_dead_factors,
        "h6_router_min_unique_topk_pairs": args.h6_router_min_unique_topk_pairs,
        "h6_structural_gate_enabled": args.h6_structural_gate_enabled,
        "h6_structural_gate_mode": args.h6_structural_gate_mode,
        "h6_structural_gate_patience": args.h6_structural_gate_patience,
        "h6_structural_gate_dense_start_epoch": args.h6_structural_gate_dense_start_epoch,
        "h6_structural_gate_require_all_levels": args.h6_structural_gate_require_all_levels,
        "h6_structural_gate_reset_state": args.h6_structural_gate_reset_state,
        "h6_gate_query_rank_max": args.h6_gate_query_rank_max,
        "h6_gate_query_top1_energy_min": args.h6_gate_query_top1_energy_min,
        "h6_gate_query_cosine_min": args.h6_gate_query_cosine_min,
        "h6_gate_logit_std_max": args.h6_gate_logit_std_max,
        "h6_gate_key_cosine_max": args.h6_gate_key_cosine_max,
        "h6_gate_key_l2_min": args.h6_gate_key_l2_min,
        "h6_gate_dynamic_cosine_min": args.h6_gate_dynamic_cosine_min,
        "h6_gate_dynamic_orth_center": args.h6_gate_dynamic_orth_center,
        "h6_gate_dynamic_orth_tolerance": args.h6_gate_dynamic_orth_tolerance,
        "h6_gate_hard_anchor_cosine_min": args.h6_gate_hard_anchor_cosine_min,
        "h6_gate_sparse_min_ratio": args.h6_gate_sparse_min_ratio,
        "h6_gate_max_sparse_dead_factors": args.h6_gate_max_sparse_dead_factors,
        "h6_gate_min_unique_topk_pairs": args.h6_gate_min_unique_topk_pairs,
        "h6_progress_version": args.h6_progress_version,
        "h6_global_text_mode": args.h6_global_text_mode,
        "h6_prediction_routing": args.h6_prediction_routing,
        "h6_local_factor_mode": args.h6_local_factor_mode,
        "h6_local_center_mix": args.h6_local_center_mix,
        "h6_local_factor_spread": args.h6_local_factor_spread,
        "lambda_h6_factor": args.lambda_h6_factor,
        "lambda_h6_router": args.lambda_h6_router,
        "lambda_h6_act": args.lambda_h6_act,
        "h6_act_effective_beta": args.h6_act_effective_beta,
        "h6_utility_factor_effective_beta": args.h6_utility_factor_effective_beta,
        "h6_router_support_normalized": bool(args.h6_router_support_normalized),
        "h6_pcgrad_main_factor": bool(args.h6_pcgrad_main_factor),
        "h6_primary_anchored_factor_surgery": bool(
            args.h6_primary_anchored_factor_surgery
        ),
        "h6_collect_router_gradient_geometry": bool(
            args.h6_collect_router_gradient_geometry
        ),
        "h6_utility_denominator_floor": args.h6_utility_denominator_floor,
        "h6_tau_utility": args.h6_tau_utility,
        "h6_utility_gain_threshold": args.h6_utility_gain_threshold,
        "factor_tau_utility": args.h6_factor_tau_utility,
        "router_tau_utility": args.h6_router_tau_utility,
        "router_gain_threshold": args.h6_router_gain_threshold,
        "act_gain_threshold": args.h6_act_gain_threshold,
        "h6_utility_entropy_threshold": args.h6_utility_entropy_threshold,
        "h6_exploration_start": args.h6_exploration_start,
        "h6_exploration_end": args.h6_exploration_end,
        "h6_exploration_total_epochs": args.h6_exploration_total_epochs,
        "h6_expert_enabled": args.h6_expert_enabled,
        "h6_expert_bottleneck": args.h6_expert_bottleneck,
        "h6_expert_fofs_seed_offset": args.h6_expert_fofs_seed_offset,
        "h6_expert_state_condition_scale": args.h6_expert_state_condition_scale,
        "h6_expert_scale_target": args.h6_expert_scale_target,
        "h6_expert_scale_start_epoch": args.h6_expert_scale_start_epoch,
        "h6_expert_scale_warmup_epochs": args.h6_expert_scale_warmup_epochs,
        "h6_expert_max_relative_ratio": args.h6_expert_max_relative_ratio,
        "lambda_h6_expert": args.lambda_h6_expert,
        "lambda_h6_advantage": args.lambda_h6_advantage,
        "lambda_h6_etf": args.lambda_h6_etf,
        "lambda_h6_expert_anchor": args.lambda_h6_expert_anchor,
        "lambda_h6_expert_radius": args.lambda_h6_expert_radius,
        "h6_expert_start_epoch": args.h6_expert_start_epoch,
        "h6_expert_warmup_epochs": args.h6_expert_warmup_epochs,
        "h6_advantage_start_epoch": args.h6_advantage_start_epoch,
        "h6_advantage_warmup_epochs": args.h6_advantage_warmup_epochs,
        "h6_advantage_margin": args.h6_advantage_margin,
        "h6_etf_start_epoch": args.h6_etf_start_epoch,
        "h6_etf_warmup_epochs": args.h6_etf_warmup_epochs,
        "lambda_h6_balance_final": args.lambda_h6_balance_final,
        "h6_balance_decay_epochs": args.h6_balance_decay_epochs,
    }


def resolve_act_gain_threshold(
    progress_version: str,
    explicit_threshold: float | None,
    legacy_threshold: float,
) -> float:
    """Resolve the ACT-only threshold without changing legacy/v8.3 routing."""
    if explicit_threshold is not None:
        return float(explicit_threshold)
    if progress_version == "P1-v8.4-A":
        return 0.0
    return float(legacy_threshold)


def train_h6_progress1(
        model: ACDCLIP,
        dataset_name: str,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: torch.device,
        args,
        logger: logging.Logger,
):
    """Train only the requested Progress 1 path; no checkpoint is loaded here."""
    if not model.h6_enabled or model.h6_progress != 1:
        raise ValueError("train_h6_progress1 requires an H6 Progress 1 model")
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.precision == "fp16"))
    loss_weights = {
        "center": args.lambda_h6_center,
        "center_factor_aware": bool(args.h6_center_factor_aware),
        "center_detach_assignment": bool(args.h6_center_detach_assignment),
        "center_margin": args.h6_center_margin,
        "router_teacher": args.lambda_h6_router_teacher,
        "router_teacher_temperature": args.h6_router_teacher_temperature,
        "router_teacher_start_epoch": args.h6_router_teacher_start_epoch,
        "router_teacher_warmup_epochs": args.h6_router_teacher_warmup_epochs,
        "teacher_confidence_gate": bool(args.h6_teacher_confidence_gate),
        "teacher_entropy_threshold": args.h6_teacher_entropy_threshold,
        "teacher_prob_std_threshold": args.h6_teacher_prob_std_threshold,
        "router_teacher_mode": args.h6_router_teacher_mode,
        "vae_rec": args.lambda_h6_vae_rec,
        "vae_kl_zero_epochs": args.h6_kl_zero_epochs,
        "vae_kl_warmup_epochs": args.h6_kl_warmup_epochs,
        "vae_kl_free_bits": args.h6_kl_free_bits,
        "vae_class_ratio": args.h6_vae_class_ratio,
        "orth": args.lambda_h6_orth,
        "delta_t_diversity": args.lambda_h6_delta_div,
        "functional_factor_diversity": args.lambda_h6_func_div,
        "balance": args.lambda_h6_balance,
        "concept_key_diversity": args.lambda_h6_concept_key_diversity,
        "concept_key_cosine_margin": args.h6_concept_key_cosine_margin,
        "concept_key_diversity_start_epoch": args.h6_concept_key_diversity_start_epoch,
        "concept_key_diversity_warmup_epochs": args.h6_concept_key_diversity_warmup_epochs,
        "kg": args.lambda_kg,
        "k": args.lambda_k,
        "dynamic_mean_anchor": args.lambda_h6_dynamic_mean_anchor,
        "dynamic_mean_anchor_min_cosine": args.h6_dynamic_mean_anchor_min_cosine,
        "dynamic_mean_anchor_start_epoch": args.h6_dynamic_mean_anchor_start_epoch,
        "dynamic_mean_anchor_warmup_epochs": args.h6_dynamic_mean_anchor_warmup_epochs,
        "cluster_responsibility": args.h6_cluster_responsibility,
        "cluster_temperature": args.h6_cluster_temperature,
        "cluster_loss_weight": args.h6_lambda_cluster_resp,
        "utility_factor": args.lambda_h6_factor,
        "utility_router": args.lambda_h6_router,
        "utility_act": args.lambda_h6_act,
        "act_effective_beta": args.h6_act_effective_beta,
    }
    structural_gate_config = StructuralGateConfig.from_args(args)
    structural_gate = H6StructuralGateState(structural_gate_config)
    if args.h6_structural_gate_reset_state:
        structural_gate.reset()
    router_failure_streak = 0
    task_loss_history: list[float] = []
    saved_checkpoints: list[str] = []
    wiring_factor_collapse_probes = 0
    for epoch_zero_based in range(args.epoch):
        epoch = epoch_zero_based + 1
        started = time.monotonic()
        model.train()
        # Conv-LoRA applies 2-D kernels to every visual patch.  The OpenCLIP
        # tower is frozen, so keep it in eval mode: its default PatchDropout
        # otherwise removes a random subset of patches during model.train(),
        # destroying the square patch grid required by the convolutional LoRA.
        model.clipmodel.eval()
        model.h6.set_epoch(epoch)
        hybrid_alpha = get_hybrid_alpha_for_epoch(epoch, args.hybrid_alpha_max, args.soft_prompt_freeze_epochs)
        model.hybrid_alpha_current = hybrid_alpha
        soft_prompt_frozen = epoch <= args.soft_prompt_freeze_epochs
        model.soft_prompt.requires_grad_(not soft_prompt_frozen)
        _set_soft_prompt_lr(optimizer, soft_prompt_frozen)
        model.set_dfg_beta(get_dfg_beta_for_epoch(epoch, args.dfg_beta_schedule, args.dfg_beta_target, args.dfg_beta))
        beta_vae_kl = get_h6_vae_beta(epoch, args.beta_h6_vae_kl, args.h6_kl_zero_epochs, args.h6_kl_warmup_epochs)
        router_teacher_weight = linear_ramp_weight(
            epoch,
            args.h6_router_teacher_start_epoch,
            args.h6_router_teacher_warmup_epochs,
            args.lambda_h6_router_teacher,
        )
        metrics = collections.defaultdict(list)
        factor_grad_diag = {
            "factor_grad_norms": None,
            "factor_grad_cos_mean": None,
            "factor_grad_cos_max": None,
            "factor_grad_cos_min": None,
            "factor_grad_l2_min": None,
            "dynamic_residual_grad_norms": None,
            "factor_id_projection_grad_norm": None,
            "factor_generator_identity_grad_norm": None,
            "factor_generator_context_grad_norm": None,
            "factor_generator_head_grad_norms": None,
            "dynamic_prompt_shared_trunk_grad_norm": None,
            "vae_mu_grad_norm": None,
            "vae_logvar_grad_norm": None,
            "prototype_modules_grad_norm": None,
            "router_grad_norm": None,
            "rho_gate_grad_norm": None,
            "phase2b_image_adapter_grad_norm": None,
            "phase2b_text_adapter_grad_norm": None,
            "dfg_grad_norm": None,
        }
        act_diag: dict[str, torch.Tensor] = {}
        act_runtime_diag: dict[str, object] = {}
        if model.h6.progress_version == "P1-v8.4-A":
            act_output = model.h6.act_head[-1]
            act_runtime_diag = {
                "act_output_weight_norm_initial": float(
                    act_output.weight.detach().float().norm().item()
                ),
                "act_output_bias_norm_initial": float(
                    act_output.bias.detach().float().norm().item()
                ),
                "act_probability_mean_before_first_update": None,
                "act_head_gradient_norm_before_step1": None,
                "act_output_weight_norm_before_step1": None,
                "act_output_weight_norm_after_step1": None,
                "act_probability_mean_after_step1": None,
                "post_step_upstream_act_gradient_norm": None,
                "post_step_upstream_act_weighted_gradient_norm": None,
                "post_step_supervised_batch": None,
                "post_step_support_count": 0,
                "post_step_zero_support_batches": [],
                "residual_definition_max_error": 0.0,
                "local_correction_reconstruction_max_error": 0.0,
            }
        optimizer.zero_grad(set_to_none=True)
        # Structural gates must describe the epoch, never just its final batch.
        epoch_diag_sum: dict[str, torch.Tensor] = {}
        epoch_diag_count: dict[str, int] = {}
        epoch_probe_values: dict[str, list[torch.Tensor]] = {}
        drift_snapshots: dict[str, dict] = {}
        drift_gradient_report = None
        optimizer_step_count = 0
        trajectory_milestones = sorted(set(args.h6_trajectory_milestones))
        trajectory_enabled = bool(trajectory_milestones)
        trajectory_gradient_batches = {32, 128, 300} & set(trajectory_milestones)
        trajectory_records: list[dict[str, torch.Tensor]] = []
        trajectory_outputs: list[dict] = []
        trajectory_structures: dict[int, dict] = {}
        trajectory_attribution: dict[int, dict] = {}
        trajectory_gradients: dict[int, dict] = {}
        trajectory_previous_batch = 0
        epoch_probe_keys = {
            "expert_delta_norm_mean", "expert_delta_valid_fraction", "final_expert_direction_cos_max",
            "expert_patch_logit_std_across_experts", "final_expert_mean_hard_cos_mean", "unique_topk_pairs",
        }
        gradient_samples: list[dict[str, torch.Tensor]] = []
        gradient_surgery_enabled = bool(
            args.h6_pcgrad_main_factor or args.h6_primary_anchored_factor_surgery
        )
        collect_router_gradient_geometry = bool(
            args.h6_pcgrad_main_factor or args.h6_collect_router_gradient_geometry
        )
        pcgrad_shared_parameters = (
            [
                parameter
                for parameter in h6_drift_parameter_groups(model)["shared_semantic"]
                if parameter.requires_grad
            ]
            if gradient_surgery_enabled else []
        )
        pcgrad_main_buffer = [torch.zeros_like(parameter) for parameter in pcgrad_shared_parameters]
        pcgrad_factor_buffer = [torch.zeros_like(parameter) for parameter in pcgrad_shared_parameters]
        # Zero placeholders keep geometry schemas stable when the optional
        # diagnostics-only router autograd traversal is disabled.
        pcgrad_router_buffer = [
            torch.zeros_like(parameter) for parameter in pcgrad_shared_parameters
        ]
        pcgrad_window_records: list[dict] = []
        # Compact, diagnostics-only telemetry for the bounded P1-v8.4-A
        # smoke.  These records deliberately contain scalars only so the
        # required per-batch/per-step invariants remain auditable without
        # retaining model tensors or changing the training graph.
        batch_runtime_records: list[dict] = []
        optimizer_step_runtime_records: list[dict] = []
        current_window_batch_records: list[dict] = []
        pcgrad_window_counts = {
            "normal_patch_count": 0,
            "anomaly_patch_count": 0,
            "valid_patch_count": 0,
            "informative_group_patch_count": 0,
            "valid_group_patch_count": 0,
            "factor_loss_sum": 0.0,
            "router_loss_sum": 0.0,
            "microbatch_count": 0,
        }
        epoch_batch_limit = min(len(train_loader), int(args.h6_smoke_max_batches)) if args.h6_smoke_max_batches > 0 else len(train_loader)
        progress = tqdm(train_loader, desc=f"[PHASE4-P1][TRAIN][epoch {epoch:02d}/{args.epoch:02d}]")

        def _runtime_stats(value):
            tensor = value.detach().float()
            return {
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std(unbiased=False).item()),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
            }

        def _runtime_finite_gradients():
            return bool(all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad.detach()).all().item())
                for parameter in model.parameters()
            ))

        def _runtime_finite_parameters():
            return bool(all(
                bool(torch.isfinite(parameter.detach()).all().item())
                for parameter in model.parameters()
                if parameter.requires_grad
            ))

        def _runtime_region_means(value, valid, y_patch):
            targets = y_patch.unsqueeze(0).expand_as(valid)
            regions = {
                "overall": valid,
                "normal": valid & (targets < 0.5),
                "anomaly": valid & (targets >= 0.5),
            }
            return {
                name: float(value.detach().float()[region].mean().item())
                if bool(region.any().item()) else 0.0
                for name, region in regions.items()
            }

        def _record_drift_snapshot(name: str, batch_payload: dict):
            router_diag = batch_payload.get("router_diagnostics", {})
            comparison_keys = (
                "hard_frozen_vs_pre_expert", "hard_adapted_vs_pre_expert",
                "pre_expert_vs_expected_noop", "final_expert_vs_pre_expert",
            )
            metric_keys = ("cos_mean", "cos_min", "cos_p05", "max_abs_diff")
            snapshot = {
                f"{comparison}_{metric}": diagnostics_to_python(router_diag.get(f"{comparison}_{metric}"))
                for comparison in comparison_keys for metric in metric_keys
            }
            snapshot.update({
                "alpha_current": float(hybrid_alpha),
                "expert_scale_current": diagnostics_to_python(batch_payload["expert_scale"]),
            })
            drift_snapshots[name] = snapshot

        for batch_idx, input_data in enumerate(progress, start=1):
            if batch_idx > epoch_batch_limit:
                break
            batch_runtime_record = None
            residual_definition_error = None
            routed_correction_error = None
            actual_gated_reconstruction_error = None
            upstream_act_gradient_norm = None
            image = input_data["image"].to(device, non_blocking=args.pin_memory)
            mask = input_data["mask"].to(device, non_blocking=args.pin_memory)

            # Default to a fully valid mask if local_mask_valid is not present in the batch
            if "local_mask_valid" in input_data:
                local_mask_valid = input_data["local_mask_valid"].to(device, non_blocking=args.pin_memory)
            else:
                local_mask_valid = torch.ones_like(mask)
            label = input_data["label"].to(device, non_blocking=args.pin_memory)
            class_names = list(input_data["class_name"])
            with _phase4_autocast(device, args.precision):
                visual_output = model(image, return_phase4_features=True)
                h6_batch = model.h6.build_batch(
                    model, dataset_name, class_names, visual_output, hybrid_alpha=hybrid_alpha
                )
                if model.h6.progress_version == "P1-v8.4-A":
                    if batch_idx == 1:
                        act_runtime_diag["act_probability_mean_before_first_update"] = float(
                            h6_batch["act_probability"].detach().float().mean().item()
                        )
                    residual_error = (
                        h6_batch["factor_residual_logits"]
                        - (
                            h6_batch["factor_patch_logits"]
                            - h6_batch["noop_reference_logit"].unsqueeze(-1)
                        )
                    ).detach().float().abs().max().item()
                    reconstructed_local = h6_batch["act_probability"] * (
                        h6_batch["dense_probabilities"]
                        * h6_batch["factor_residual_logits"]
                    ).sum(dim=-1)
                    correction_error = (
                        h6_batch["h6_logits"] - reconstructed_local
                    ).detach().float().abs().max().item()
                    residual_definition_error = float(residual_error)
                    routed_correction_error = float(correction_error)
                    actual_expected_correction = (
                        h6_batch["rho"].view(-1, 1, 1).to(h6_batch["h6_logits"].dtype)
                        * h6_batch["h6_logits"]
                    )
                    actual_gated_reconstruction_error = float(
                        (
                            h6_batch["rho_scaled_actual_correction"]
                            - actual_expected_correction
                        ).detach().float().abs().max().item()
                    )
                    act_runtime_diag["residual_definition_max_error"] = max(
                        float(act_runtime_diag["residual_definition_max_error"]),
                        float(residual_error),
                    )
                    act_runtime_diag["local_correction_reconstruction_max_error"] = max(
                        float(act_runtime_diag["local_correction_reconstruction_max_error"]),
                        float(correction_error),
                    )
                if args.h6_cluster_responsibility:
                    h6_cluster_resp, q_cluster, cluster_diag = cluster_responsibility_loss(
                        h6_batch["router_patch_features"],
                        model.h6.cluster_centroids,
                        h6_batch["dense_probabilities"],
                        args.h6_cluster_temperature,
                    )
                    for diag_name, diag_value in cluster_diag.items():
                        value = diag_value.detach().float()
                        epoch_diag_sum[diag_name] = epoch_diag_sum.get(diag_name, torch.zeros_like(value)) + value
                        epoch_diag_count[diag_name] = epoch_diag_count.get(diag_name, 0) + 1
                else:
                    h6_cluster_resp = h6_batch["dense_probabilities"].sum() * 0.0
                    q_cluster = None
                    cluster_diag = {}
                if args.h6_drift_diagnostics and batch_idx == 1:
                    _record_drift_snapshot("batch_000_before_first_backward", h6_batch)
                if args.h6_drift_diagnostics and batch_idx == epoch_batch_limit:
                    _record_drift_snapshot("final_smoke_batch", h6_batch)
                for diag_name, diag_value in h6_batch.get("router_diagnostics", {}).items():
                    if torch.is_tensor(diag_value):
                        # Keep integer Top-K counts and boolean finite flags as
                        # epoch means too; downstream logger/gates consume both.
                        value = diag_value.detach().float()
                        epoch_diag_sum[diag_name] = epoch_diag_sum.get(diag_name, torch.zeros_like(value)) + value
                        epoch_diag_count[diag_name] = epoch_diag_count.get(diag_name, 0) + 1
                        if diag_name in epoch_probe_keys:
                            epoch_probe_values.setdefault(diag_name, []).append(value.mean().cpu())
                if args.h6_factor_grad_diagnostics and (
                    batch_idx == 1 or batch_idx in trajectory_gradient_batches
                ):
                    h6_batch["dynamic_text"].retain_grad()
                seg_features = torch.stack(visual_output["seg_tokens"], dim=0)
                det_features = torch.stack(visual_output["det_tokens"], dim=0)
                if args.h6_global_text_mode in ("phase2b_hybrid", "hard_anchor"):
                    is_hybrid = args.h6_global_text_mode == "phase2b_hybrid"
                    text_global = get_phase2b_global_text_features(
                        model, dataset_name, class_names, device,
                        use_hybrid_soft_prompt=is_hybrid,
                        use_soft_prompt=args.use_soft_prompt
                    ).to(dtype=det_features.dtype)
                else:
                    text_global = h6_batch["text_global"].to(dtype=det_features.dtype)
                cls_pred = torch.stack([
                    torch.matmul(det_features[level].unsqueeze(1), text_global[level]).squeeze(1)
                    for level in range(model.n_groups)
                ], dim=0).mean(dim=0)
                cls_loss = F.cross_entropy(cls_pred.float(), label)
                seg_pred, base_group_logits, base_abnormal_minus_normal = model.vision_text_fusion_gate_seg(
                    seg_features,
                    text_global,
                    img_size=args.img_size,
                    h6_patch_logits=h6_batch["h6_logits"],
                    return_details=True,
                )
                base_abnormal_minus_normal = base_abnormal_minus_normal.detach()

                seg_loss = calculate_seg_loss(seg_pred.float(), mask.float())
                task_loss = cls_loss + seg_loss
                if args.h6_center_factor_aware:
                    h6_center = factor_aware_center_loss(
                        h6_batch["projected_levels"],
                        h6_batch["prototype_normal"],
                        h6_batch["prototype_abnormal"],
                        h6_batch["dense_probabilities"],
                        mask,
                        label,
                        detach_assignment=args.h6_center_detach_assignment,
                        margin=args.h6_center_margin,
                    )
                else:
                    h6_center = center_loss(
                        h6_batch["projected_levels"],
                        h6_batch["prototype_normal"],
                        h6_batch["prototype_abnormal"],
                        mask,
                        label,
                    )
                h6_orth = h6_batch["residual_diversity"]
                h6_delta_div = (
                    delta_t_diversity_loss(h6_batch["dynamic_text"], h6_batch["hard_frozen"])
                    if args.lambda_h6_delta_div > 0.0
                    else h6_orth * 0.0
                )
                if args.lambda_h6_func_div > 0.0:
                    factor_patch_logits = h6_batch["factor_patch_logits"]
                    hard_direction = h6_batch["hard_frozen"].float()[..., 1] - h6_batch["hard_frozen"].float()[..., 0]
                    hard_logits = torch.einsum("gbpd,gbd->gbp", F.normalize(seg_features.float(), dim=-1), hard_direction)
                    hard_centered = hard_logits - hard_logits.mean(dim=2, keepdim=True)
                    confidence = hard_centered.abs() / hard_centered.std(dim=2, keepdim=True).clamp_min(1e-6)
                    patch_weights = 1.0 + confidence.detach()
                    patch_labels = F.adaptive_max_pool2d(mask.float(), (int(factor_patch_logits.shape[2] ** 0.5),) * 2).flatten(1)
                    patch_weights = patch_weights + patch_labels.detach().unsqueeze(0)
                    h6_func_div, h6_func_corr = functional_factor_diversity_loss(factor_patch_logits, patch_weights)
                    epoch_diag_sum["functional_factor_correlation_matrix"] = epoch_diag_sum.get("functional_factor_correlation_matrix", torch.zeros_like(h6_func_corr)) + h6_func_corr.detach()
                    epoch_diag_count["functional_factor_correlation_matrix"] = epoch_diag_count.get("functional_factor_correlation_matrix", 0) + 1
                else:
                    h6_func_div = h6_orth * 0.0
                h6_balance = routing_balance_loss(h6_batch["dense_probabilities"])
                if router_teacher_weight > 0.0:
                    h6_router_teacher, teacher_diag = router_teacher_loss(
                        h6_batch["projected_levels"],
                        h6_batch["prototype_normal"],
                        h6_batch["prototype_abnormal"],
                        h6_batch["dense_probabilities"],
                        mask,
                        label,
                        temperature=args.h6_router_teacher_temperature,
                        mode=args.h6_router_teacher_mode,
                        confidence_gate_enabled=args.h6_teacher_confidence_gate,
                        entropy_threshold=args.h6_teacher_entropy_threshold,
                        probability_std_threshold=args.h6_teacher_prob_std_threshold,
                    )
                    teacher_diag.update(
                        teacher_candidate_diagnostics(
                            h6_batch["projected_levels"],
                            h6_batch["prototype_normal"],
                            h6_batch["prototype_abnormal"],
                            mask,
                            label,
                            temperature=args.h6_router_teacher_temperature,
                        )
                    )
                else:
                    h6_router_teacher = h6_batch["dense_probabilities"].float().sum() * 0.0
                    teacher_diag = {"router_teacher_entropy": h6_router_teacher.detach()}
                teacher_entropy_mean = (
                    teacher_diag["teacher_entropy"].float().mean()
                    if torch.is_tensor(teacher_diag.get("teacher_entropy", None))
                    else torch.tensor(1.0, device=device)
                )
                teacher_prob_std_mean = (
                    teacher_diag["teacher_probability_std_across_patches"].float().mean()
                    if torch.is_tensor(teacher_diag.get("teacher_probability_std_across_patches", None))
                    else torch.tensor(0.0, device=device)
                )
                teacher_informative = bool(
                    teacher_diag.get("teacher_informative_patch_count", torch.zeros((), device=device))
                    .detach()
                    .float()
                    .sum()
                    .cpu()
                    .item()
                    > 0
                )
                effective_router_teacher_weight = (
                    router_teacher_weight
                    if (not args.h6_teacher_confidence_gate or teacher_informative)
                    else 0.0
                )
                h6_kl_raw = h6_batch["kl"]
                h6_kl_effective = torch.clamp(h6_kl_raw, min=float(args.h6_kl_free_bits))
                concept_key_diversity_weight = linear_ramp_weight(
                    epoch,
                    args.h6_concept_key_diversity_start_epoch,
                    args.h6_concept_key_diversity_warmup_epochs,
                    args.lambda_h6_concept_key_diversity,
                )
                h6_concept_key_diversity = concept_key_diversity_loss(
                    h6_batch["raw_semantic_keys"],
                    margin=args.h6_concept_key_cosine_margin,
                )
                dynamic_mean_anchor_weight = linear_ramp_weight(
                    epoch,
                    args.h6_dynamic_mean_anchor_start_epoch,
                    args.h6_dynamic_mean_anchor_warmup_epochs,
                    args.lambda_h6_dynamic_mean_anchor,
                )
                expert_weight = linear_ramp_weight(epoch, args.h6_expert_start_epoch, args.h6_expert_warmup_epochs, args.lambda_h6_expert)
                advantage_weight = linear_ramp_weight(epoch, args.h6_advantage_start_epoch, args.h6_advantage_warmup_epochs, args.lambda_h6_advantage)
                etf_weight = linear_ramp_weight(epoch, args.h6_etf_start_epoch, args.h6_etf_warmup_epochs, args.lambda_h6_etf)
                balance_weight = args.lambda_h6_balance
                # Option A-prime Losses
                h6_route = 0.0
                h6_factor_role = 0.0
                h6_actual_local = 0.0
                if any(getattr(args, k, 0.0) > 0.0 for k in ["lambda_h6_route", "lambda_h6_factor_role", "lambda_h6_actual_local"]):
                    B, P = base_abnormal_minus_normal.shape[1:3]
                    q_role, hard_role, mask_coverage, local_valid_patch, local_valid_image = build_semantic_roles(mask, label, P, local_mask_valid)

                    if getattr(args, "lambda_h6_route", 0.0) > 0.0:
                        h6_route = active_role_balanced_router_loss(h6_batch["dense_probabilities"], q_role, hard_role, local_valid_patch)

                    if getattr(args, "lambda_h6_factor_role", 0.0) > 0.0:
                        h6_factor_role = factor_specific_residual_role_loss(h6_batch["rho_scaled_factor_correction"], q_role, hard_role, mask_coverage, local_valid_patch, base_abnormal_minus_normal)

                    if getattr(args, "lambda_h6_actual_local", 0.0) > 0.0:
                        h6_actual_local = actual_local_residual_loss(h6_batch["rho_scaled_actual_correction"], q_role, hard_role, mask_coverage, local_valid_patch, base_abnormal_minus_normal)

                h6_utility_factor = h6_orth * 0.0
                h6_utility_router = h6_orth * 0.0
                h6_utility_act = h6_orth * 0.0
                utility_payload = None
                act_payload = None
                if model.h6.progress_version in {"P1-v8.3", "P1-v8.4-A"}:
                    patch_count = h6_batch["factor_patch_logits"].shape[2]
                    y_patch, utility_valid = build_patch_targets(mask, patch_count, local_mask_valid)
                    epsilon_total_epochs = (
                        args.h6_exploration_total_epochs
                        if args.h6_exploration_total_epochs is not None
                        else args.epoch
                    )
                    epsilon = exploration_epsilon(
                        epoch, epsilon_total_epochs,
                        args.h6_exploration_start, args.h6_exploration_end,
                    )
                    utility_payload = utility_teacher(
                        base_abnormal_minus_normal,
                        (
                            h6_batch["factor_residual_logits"]
                            if model.h6.progress_version == "P1-v8.4-A"
                            else h6_batch["factor_patch_logits"]
                        ),
                        y_patch, utility_valid, rho=0.05,
                        denominator_floor=args.h6_utility_denominator_floor,
                        tau_utility=args.h6_tau_utility,
                        factor_tau_utility=args.h6_factor_tau_utility,
                        router_tau_utility=args.h6_router_tau_utility,
                        epsilon=epsilon,
                        gain_threshold=args.h6_utility_gain_threshold,
                        router_gain_threshold=args.h6_router_gain_threshold,
                        entropy_threshold=args.h6_utility_entropy_threshold,
                        routed_probabilities=(
                            h6_batch["prediction_probabilities"]
                            if model.h6.progress_version == "P1-v8.4-A"
                            else None
                        ),
                    )
                    h6_utility_factor = (
                        effective_number_utility_factor_loss(
                            utility_payload,
                            y_patch,
                            beta=args.h6_utility_factor_effective_beta,
                        )
                        if args.h6_utility_factor_effective_beta is not None
                        else utility_factor_loss(utility_payload, y_patch)
                    )
                    h6_utility_router = (
                        support_normalized_utility_router_loss(
                            h6_batch["dense_probabilities"], utility_payload
                        )
                        if args.h6_router_support_normalized
                        else utility_router_loss(
                            h6_batch["dense_probabilities"], utility_payload
                        )
                    )
                    if model.h6.progress_version == "P1-v8.4-A":
                        act_payload = act_teacher(
                            utility_payload,
                            gain_threshold=args.h6_act_gain_threshold,
                        )
                        h6_utility_act = effective_number_act_loss(
                            h6_batch["act_logits"],
                            act_payload,
                            y_patch,
                            beta=args.h6_act_effective_beta,
                        )
                        act_diag = act_diagnostics(
                            h6_batch["act_probability"], act_payload, y_patch
                        )
                        for diag_name, diag_value in act_diag.items():
                            value = diag_value.detach().float()
                            epoch_diag_sum[diag_name] = epoch_diag_sum.get(
                                diag_name, torch.zeros_like(value)
                            ) + value
                            epoch_diag_count[diag_name] = epoch_diag_count.get(diag_name, 0) + 1
                        if (
                            optimizer_step_count >= 1
                            and act_runtime_diag["post_step_upstream_act_gradient_norm"] is None
                        ):
                            support_count = int(act_payload["support"].sum().item())
                            if support_count == 0:
                                act_runtime_diag["post_step_zero_support_batches"].append(batch_idx)
                            else:
                                upstream_gradient = torch.autograd.grad(
                                    h6_utility_act,
                                    h6_batch["router_patch_features"],
                                    retain_graph=True,
                                    allow_unused=True,
                                )[0]
                                upstream_norm = (
                                    0.0 if upstream_gradient is None
                                    else float(upstream_gradient.detach().float().norm().item())
                                )
                                upstream_act_gradient_norm = upstream_norm
                                act_runtime_diag.update({
                                    "post_step_upstream_act_gradient_norm": upstream_norm,
                                    "post_step_upstream_act_weighted_gradient_norm": (
                                        upstream_norm * float(args.lambda_h6_act)
                                    ),
                                    "post_step_supervised_batch": batch_idx,
                                    "post_step_support_count": support_count,
                                })
                    if model.h6.progress_version == "P1-v8.4-A":
                        runtime_valid = act_payload["valid"].bool()
                        runtime_targets = y_patch.unsqueeze(0).expand_as(runtime_valid)
                        routed_gain = utility_payload["routed_gain_rel"].detach().float()
                        expected_positive = runtime_valid & (
                            routed_gain > float(args.h6_act_gain_threshold)
                        )
                        expected_negative = runtime_valid & (routed_gain <= 0.0)
                        expected_ambiguous = runtime_valid & (
                            (routed_gain > 0.0)
                            & (routed_gain <= float(args.h6_act_gain_threshold))
                        )
                        actual_logits = utility_payload["z0"].detach().float() + (
                            h6_batch["rho"].detach().float().view(-1, 1, 1)
                            * h6_batch["h6_logits"].detach().float()
                        )
                        actual_loss = F.binary_cross_entropy_with_logits(
                            actual_logits,
                            runtime_targets.float(),
                            reduction="none",
                        )

                        def _support_counts(mask):
                            count = int(mask.sum().item())
                            denominator = int(runtime_valid.sum().item())
                            return {
                                "on": int(act_payload["positive"][mask].sum().item()),
                                "off": int(act_payload["negative"][mask].sum().item()),
                                "ambiguous": int(act_payload["ambiguous"][mask].sum().item()),
                                "valid": count,
                                "on_fraction": float(
                                    act_payload["positive"][mask].float().mean().item()
                                ) if count else 0.0,
                                "off_fraction": float(
                                    act_payload["negative"][mask].float().mean().item()
                                ) if count else 0.0,
                                "ambiguous_fraction": float(
                                    act_payload["ambiguous"][mask].float().mean().item()
                                ) if count else 0.0,
                                "support_fraction": float(
                                    act_payload["support"][mask].float().mean().item()
                                ) if count else 0.0,
                                "global_valid": denominator,
                            }

                        normal_runtime = runtime_valid & (runtime_targets < 0.5)
                        anomaly_runtime = runtime_valid & (runtime_targets >= 0.5)
                        batch_runtime_record = {
                            "batch": int(batch_idx),
                            "optimizer_step_before": int(optimizer_step_count),
                            "optimizer_step": None,
                            "rho": [
                                float(value)
                                for value in h6_batch["rho"].detach().float().tolist()
                            ],
                            "rho_trainable": bool(model.h6.rho.raw.requires_grad),
                            "finite_parameters_before": _runtime_finite_parameters(),
                            "finite_gradients_after_backward": None,
                            "finite_parameters_after_step": None,
                            "reconstruction": {
                                "residual_definition_max_abs_error": residual_definition_error,
                                "routed_correction_max_abs_error": routed_correction_error,
                                "actual_gated_max_abs_error": actual_gated_reconstruction_error,
                                "surgery_max_abs_error": None,
                                "main_exact_change_max_abs_error": None,
                            },
                            "act": {
                                "probability": _runtime_stats(h6_batch["act_probability"]),
                                "logits": _runtime_stats(h6_batch["act_logits"]),
                                "head_raw_gradient_norm": None,
                                "head_weighted_gradient_norm": None,
                                "upstream_gradient_norm": upstream_act_gradient_norm,
                                "output_weight_norm": float(
                                    model.h6.act_head[-1].weight.detach().float().norm().item()
                                ),
                                "output_bias_norm": float(
                                    model.h6.act_head[-1].bias.detach().float().norm().item()
                                ),
                            },
                            "support": {
                                "overall": _support_counts(runtime_valid),
                                "normal": _support_counts(normal_runtime),
                                "anomaly": _support_counts(anomaly_runtime),
                            },
                            "label_semantics": {
                                "threshold": float(args.h6_act_gain_threshold),
                                "routed_gain": _runtime_stats(routed_gain[runtime_valid]),
                                "positive_mismatch_count": int(
                                    (act_payload["positive"] != expected_positive).sum().item()
                                ),
                                "negative_mismatch_count": int(
                                    (act_payload["negative"] != expected_negative).sum().item()
                                ),
                                "ambiguous_mismatch_count": int(
                                    (act_payload["ambiguous"] != expected_ambiguous).sum().item()
                                ),
                            },
                            "utility": {
                                "Base": _runtime_region_means(
                                    utility_payload["loss_base"], runtime_valid, y_patch
                                ),
                                "FullSoftRouted_ACT1": _runtime_region_means(
                                    utility_payload["loss_routed"], runtime_valid, y_patch
                                ),
                                "ActualGated": _runtime_region_means(
                                    actual_loss, runtime_valid, y_patch
                                ),
                                "g_route": _runtime_region_means(
                                    routed_gain, runtime_valid, y_patch
                                ),
                            },
                        }
                        batch_runtime_records.append(batch_runtime_record)
                        current_window_batch_records.append(batch_runtime_record)
                    if gradient_surgery_enabled:
                        physical_valid = utility_valid
                        physical_anomaly = physical_valid & (y_patch >= 0.5)
                        pcgrad_window_counts["normal_patch_count"] += int(
                            (physical_valid & ~physical_anomaly).sum().item()
                        )
                        pcgrad_window_counts["anomaly_patch_count"] += int(
                            physical_anomaly.sum().item()
                        )
                        pcgrad_window_counts["valid_patch_count"] += int(physical_valid.sum().item())
                        pcgrad_window_counts["informative_group_patch_count"] += int(
                            utility_payload["informative"].sum().item()
                        )
                        pcgrad_window_counts["valid_group_patch_count"] += int(
                            utility_payload["valid"].sum().item()
                        )
                        pcgrad_window_counts["factor_loss_sum"] += float(
                            h6_utility_factor.detach().item()
                        )
                        pcgrad_window_counts["router_loss_sum"] += float(
                            h6_utility_router.detach().item()
                        )
                        pcgrad_window_counts["microbatch_count"] += 1
                    utility_diag = utility_diagnostics(
                        utility_payload, h6_batch["dense_probabilities"], y_patch, rho=0.05
                    )
                    utility_diag["exploration_epsilon"] = torch.tensor(epsilon, device=device)
                    for diag_name, diag_value in utility_diag.items():
                        value = diag_value.detach().float()
                        epoch_diag_sum[diag_name] = epoch_diag_sum.get(
                            diag_name, torch.zeros_like(value)
                        ) + value
                        epoch_diag_count[diag_name] = epoch_diag_count.get(diag_name, 0) + 1
                    if trajectory_enabled:
                        trajectory_records.append(capture_utility_record(
                            utility_payload, h6_batch["dense_probabilities"], y_patch,
                            h6_utility_router,
                            act_probability=h6_batch.get("act_probability"),
                            act_logits=h6_batch.get("act_logits"),
                            act_payload=act_payload,
                            utility_act_loss=h6_utility_act,
                            actual_gated_loss=actual_loss,
                        ))
                    if not trajectory_enabled or batch_idx in trajectory_milestones:
                        structure_diag = p1_v83_structure_diagnostics(h6_batch)
                        if trajectory_enabled:
                            trajectory_structures[batch_idx] = diagnostics_to_python(structure_diag)
                        for diag_name, diag_value in structure_diag.items():
                            value = diag_value.detach().float()
                            epoch_diag_sum[diag_name] = epoch_diag_sum.get(
                                diag_name, torch.zeros_like(value)
                            ) + value
                            epoch_diag_count[diag_name] = epoch_diag_count.get(diag_name, 0) + 1

                if model.h6.expert_enabled:
                    h6_expert, expert_terms = assigned_expert_loss(h6_batch["expert_patch_logits"], h6_batch["prediction_probabilities"], mask)
                    h6_advantage = expert_advantage_loss(h6_batch["expert_patch_logits"], h6_batch["topk_indices"], mask, args.h6_advantage_margin)
                    h6_etf, expert_delta_norm = expert_etf_loss(h6_batch["expert_delta_tangent"])
                    final_balance = args.lambda_h6_balance if args.lambda_h6_balance_final is None else args.lambda_h6_balance_final
                    decay = min(1.0, max(0.0, (epoch - args.h6_router_soft_epochs) / max(1, args.h6_balance_decay_epochs)))
                    balance_weight = args.lambda_h6_balance + decay * (final_balance - args.lambda_h6_balance)
                    h6_balance, balance_terms = dual_routing_balance_loss(h6_batch["dense_probabilities"], h6_batch["prediction_probabilities"])
                    h6_expert_anchor, expert_anchor_cos = expert_clip_anchor_loss(h6_batch["active_factor_bank"], h6_batch["hard_frozen"], args.h6_expert_anchor_min_cosine)
                    if args.lambda_h6_expert_radius > 0.0:
                        h6_expert_radius = expert_radius_loss(h6_batch["expert_scale"], args.h6_expert_max_relative_ratio)
                    else:
                        h6_expert_radius = h6_orth * 0.0

                    # The returned h6_expert_scale from the gate is used downstream
                    # only for diagnostics.  Actual forward uses min(target, ...).
                    expert_function = expert_patch_function_diagnostics(
                        h6_batch["expert_patch_logits"], h6_batch["topk_indices"], mask, args.h6_advantage_margin
                    )
                else:
                    zero = h6_balance * 0.0
                    h6_expert = h6_advantage = h6_etf = h6_expert_anchor = h6_expert_radius = zero
                    expert_delta_norm = zero.detach(); expert_anchor_cos = zero.detach()
                    expert_terms = {"expert_normal_all": zero.detach(), "expert_abnormal_assigned": zero.detach()}
                    balance_terms = {"dense_cv2": h6_balance.detach(), "prediction_cv2": zero.detach()}
                    expert_function = {
                        "expert_normal_patch_count": zero.detach(), "expert_abnormal_patch_count": zero.detach(),
                        "expert_valid_patch_count": zero.detach(), "selected_expert_loss": zero.detach(),
                        "nonselected_expert_loss": zero.detach(), "selected_minus_nonselected_loss": zero.detach(),
                        "expert_advantage_margin_satisfied_fraction": zero.detach(), "expert_advantage_valid_count": zero.detach(),
                    }
                for diag_name, diag_value in expert_function.items():
                    value = diag_value.detach().float()
                    epoch_diag_sum[diag_name] = epoch_diag_sum.get(diag_name, torch.zeros_like(value)) + value
                    epoch_diag_count[diag_name] = epoch_diag_count.get(diag_name, 0) + 1
                loss_components = {
                    "task": task_loss, "center": args.lambda_h6_center * h6_center,
                    "router_teacher": effective_router_teacher_weight * h6_router_teacher,
                    "vae_rec": args.lambda_h6_vae_rec * h6_batch["reconstruction"],
                    "vae_kl": beta_vae_kl * h6_kl_effective, "kg": args.lambda_kg * h6_batch["kg_loss"],
                    "legacy_pre_expert_orth": args.lambda_h6_orth * h6_orth,
                    "delta_t_diversity": args.lambda_h6_delta_div * h6_delta_div,
                    "functional_factor_diversity": args.lambda_h6_func_div * h6_func_div,
                    "cluster_responsibility": args.h6_lambda_cluster_resp * h6_cluster_resp,
                    "route": getattr(args, "lambda_h6_route", 0.0) * h6_route,
                    "factor_role": getattr(args, "lambda_h6_factor_role", 0.0) * h6_factor_role,
                    "actual_local": getattr(args, "lambda_h6_actual_local", 0.0) * h6_actual_local,
                    "utility_factor": args.lambda_h6_factor * h6_utility_factor,
                    "utility_router": args.lambda_h6_router * h6_utility_router,
                    "utility_act": args.lambda_h6_act * h6_utility_act,
                    "balance": balance_weight * h6_balance,
                    "raw_key_diversity": concept_key_diversity_weight * h6_concept_key_diversity,
                    "dynamic_mean_anchor": dynamic_mean_anchor_weight * h6_batch["dynamic_mean_anchor_loss_raw"],
                    "expert_assigned": expert_weight * h6_expert, "expert_advantage": advantage_weight * h6_advantage,
                    "expert_etf": etf_weight * h6_etf, "expert_anchor": args.lambda_h6_expert_anchor * h6_expert_anchor,
                    "expert_radius": args.lambda_h6_expert_radius * h6_expert_radius,
                }
                loss_component_sum = sum_loss_components(loss_components)
                total_loss = loss_component_sum
                loss_component_residual = total_loss - loss_component_sum
                if batch_idx in set(args.h6_wiring_probe_batches):
                    factor_patch_logits = h6_batch["factor_patch_logits"]
                    delta_t = (
                        h6_batch["dynamic_text"].float()
                        - h6_batch["hard_frozen"].unsqueeze(2).expand_as(h6_batch["dynamic_text"]).float()
                    ).permute(0, 1, 2, 4, 3).reshape(-1, model.h6.num_factors, 2 * model.h6.text_dim)
                    delta_unit = F.normalize(delta_t, dim=-1, eps=1e-6)
                    delta_cos = torch.bmm(delta_unit, delta_unit.transpose(1, 2))
                    offdiag = ~torch.eye(model.h6.num_factors, device=device, dtype=torch.bool)
                    factor_std = factor_patch_logits.std(dim=-1, unbiased=False).mean()
                    factor_max_diff = (
                        factor_patch_logits.max(dim=-1).values - factor_patch_logits.min(dim=-1).values
                    ).mean()
                    collapsed = bool(factor_max_diff.detach().abs().item() <= 1e-9)
                    wiring_factor_collapse_probes = wiring_factor_collapse_probes + 1 if collapsed else 0
                    write_json_atomic(
                        Path(args.save_path) / "wiring_probes" / f"batch_{batch_idx:03d}.json",
                        diagnostics_to_python({
                            "epoch": epoch,
                            "batch": batch_idx,
                            "loss_total": total_loss,
                            "loss_delta_t_diversity_raw": h6_delta_div,
                            "loss_delta_t_diversity_weighted": args.lambda_h6_delta_div * h6_delta_div,
                            "loss_cluster_responsibility_raw": h6_cluster_resp,
                            "loss_cluster_responsibility_weighted": args.h6_lambda_cluster_resp * h6_cluster_resp,
                            "cluster_target": None if q_cluster is None else q_cluster,
                            "cluster_diagnostics": cluster_diag,
                            "delta_t_norm_mean": delta_t.norm(dim=-1).mean(),
                            "delta_t_norm_min": delta_t.norm(dim=-1).min(),
                            "delta_t_cosine_offdiag_mean": delta_cos[..., offdiag].mean(),
                            "delta_t_cosine_offdiag_max": delta_cos[..., offdiag].max(),
                            "factor_patch_logit_std_across_factors": factor_std,
                            "factor_patch_logit_max_diff_across_factors": factor_max_diff,
                            "factor_outputs_collapsed": collapsed,
                            "consecutive_factor_output_collapse_probes": wiring_factor_collapse_probes,
                            "router": h6_batch["router_diagnostics"],
                        }),
                    )
                    if wiring_factor_collapse_probes >= 2:
                        raise RuntimeError(
                            "factor outputs became exactly identical for two consecutive wiring probes"
                        )
                if (args.h6_drift_diagnostics and batch_idx == 1) or (
                    trajectory_enabled and batch_idx in trajectory_gradient_batches
                ):
                    attribution = h6_drift_gradient_attribution(
                        {
                            "main_task": (task_loss, 1.0),
                            "assigned_expert": (h6_expert, expert_weight),
                            "advantage": (h6_advantage, advantage_weight),
                            "expert_anchor": (h6_expert_anchor, args.lambda_h6_expert_anchor),
                            "center": (h6_center, args.lambda_h6_center),
                            "dynamic_mean_anchor": (h6_batch["dynamic_mean_anchor_loss_raw"], dynamic_mean_anchor_weight),
                            "utility_factor": (h6_utility_factor, args.lambda_h6_factor),
                            "utility_router": (h6_utility_router, args.lambda_h6_router),
                            "utility_act": (h6_utility_act, args.lambda_h6_act),
                        },
                        h6_drift_parameter_groups(model),
                    )
                    if args.h6_drift_diagnostics and batch_idx == 1:
                        drift_gradient_report = attribution
                    if trajectory_enabled and batch_idx in trajectory_gradient_batches:
                        trajectory_attribution[batch_idx] = attribution
            if args.h6_smoke_forward_only:
                with torch.no_grad(), _phase4_autocast(device, args.precision):
                    repeated_h6_batch = model.h6.build_batch(
                        model, dataset_name, class_names, visual_output,
                        hybrid_alpha=hybrid_alpha, update_load_bias=False,
                    )
                dense_probabilities = h6_batch["dense_probabilities"].detach().float()
                factor_logits = h6_batch["factor_patch_logits"].detach().float()
                state_tokens = h6_batch["state_tokens"].detach().float()
                tensors = {
                    "base_group_logits": base_group_logits,
                    "base_abnormal_minus_normal": base_abnormal_minus_normal,
                    "factor_patch_logits": factor_logits,
                    "dense_probabilities": dense_probabilities,
                    "state_tokens": state_tokens,
                    "class_token": h6_batch["class_token"],
                    "hard_frozen": h6_batch["hard_frozen"],
                    "dynamic_text": h6_batch["dynamic_text"],
                    "total_loss": total_loss,
                }
                if model.h6.residual_act_enabled:
                    tensors.update({
                        "noop_reference_logit": h6_batch["noop_reference_logit"],
                        "factor_residual_logits": h6_batch["factor_residual_logits"],
                        "act_logits": h6_batch["act_logits"],
                        "act_probability": h6_batch["act_probability"],
                    })
                finite = {
                    name: bool(torch.isfinite(value.detach().float()).all().item())
                    for name, value in tensors.items()
                }
                router_sum_error = (dense_probabilities.sum(dim=-1) - 1.0).abs().max()
                factor_max_difference = (
                    factor_logits.max(dim=-1).values - factor_logits.min(dim=-1).values
                ).abs().max()
                state_max_difference = (
                    state_tokens.max(dim=1).values - state_tokens.min(dim=1).values
                ).abs().max()
                class_repeat_error = (
                    h6_batch["class_token"].detach().float()
                    - repeated_h6_batch["class_token"].detach().float()
                ).abs().max()
                hard_frozen_repeat_error = (
                    h6_batch["hard_frozen"].detach().float()
                    - repeated_h6_batch["hard_frozen"].detach().float()
                ).abs().max()
                dynamic_hard_difference = (
                    h6_batch["dynamic_text"].detach().float()
                    - h6_batch["hard_frozen"].detach().float().unsqueeze(2)
                ).abs().max()
                rho = h6_batch["rho"].detach().float()
                checks = {
                    "all_finite": all(finite.values()),
                    "router_probabilities_sum_to_one": bool(router_sum_error.item() <= 1e-6),
                    "state_factor_specific": bool(state_max_difference.item() > 1e-9),
                    "class_deterministic": bool(class_repeat_error.item() <= 1e-7),
                    "hard_frozen_stable": bool(hard_frozen_repeat_error.item() <= 1e-7),
                    "dynamic_adapted_differs_from_hard_frozen": bool(dynamic_hard_difference.item() > 1e-9),
                    "rho_fixed_005": bool(torch.equal(rho, torch.full_like(rho, 0.05))),
                    "factor_outputs_not_identical": bool(factor_max_difference.item() > 1e-9),
                    "optimizer_untouched": len(optimizer.state) == 0,
                }
                payload = diagnostics_to_python({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "checks": checks,
                    "finite": finite,
                    "shapes": {name: list(value.shape) for name, value in tensors.items()},
                    "router_sum_max_error": router_sum_error,
                    "factor_max_difference": factor_max_difference,
                    "state_max_difference": state_max_difference,
                    "class_repeat_max_error": class_repeat_error,
                    "hard_frozen_repeat_max_error": hard_frozen_repeat_error,
                    "dynamic_hard_max_difference": dynamic_hard_difference,
                    "rho": rho,
                    "utility": utility_diag,
                    "gpu_allocated_bytes": (
                        torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                    ),
                    "gpu_reserved_bytes": (
                        torch.cuda.memory_reserved(device) if device.type == "cuda" else 0
                    ),
                    "gpu_peak_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                    ),
                    "gpu_peak_reserved_bytes": (
                        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
                    ),
                })
                write_json_atomic(Path(args.save_path) / "forward_probe.json", payload)
                failed = [name for name, passed in checks.items() if not passed]
                if failed:
                    raise RuntimeError(f"P1-v8.3 forward-only probe failed: {failed}")
                logger.info("P1-v8.3 forward-only probe PASS: %s", payload)
                return model
            if not torch.isfinite(total_loss).all():
                if structural_gate_config.enabled:
                    decision = GateDecision(
                        hard_failure=True,
                        abort_reason="h6_nonfinite_total_loss",
                        fatal_metrics=["total_loss"],
                    )
                    gated_abort_artifacts(
                        save_path=args.save_path,
                        epoch=epoch,
                        decision=decision,
                        gate_state=structural_gate,
                        gate_config=structural_gate_config,
                        metrics={"total": total_loss.detach(), "task": task_loss.detach()},
                        diagnostics=h6_batch.get("router_diagnostics", {}),
                        teacher_state=teacher_diag,
                        sparse_ratio=float(h6_batch["sparse_ratio"].detach().item()),
                        routing_mode="batch_fatal",
                        alpha=hybrid_alpha,
                        trust_region_weight=dynamic_mean_anchor_weight,
                        latest_checkpoint_path=None,
                        args=args,
                        payload=None,
                    )
                    sys.exit(42)
                raise RuntimeError(f"non-finite H6 loss at epoch={epoch}, batch={batch_idx}")
            accumulation_divisor = grad_accum_window_size(
                batch_idx, epoch_batch_limit, args.grad_accum_steps
            )
            if gradient_surgery_enabled:
                component_losses = [
                    task_loss,
                    args.lambda_h6_factor * h6_utility_factor,
                ]
                component_buffers = [pcgrad_main_buffer, pcgrad_factor_buffer]
                if collect_router_gradient_geometry:
                    component_losses.append(args.lambda_h6_router * h6_utility_router)
                    component_buffers.append(pcgrad_router_buffer)
                for component_loss, component_buffer in zip(component_losses, component_buffers):
                    component_gradients = torch.autograd.grad(
                        component_loss / accumulation_divisor,
                        pcgrad_shared_parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    for index, (parameter, gradient) in enumerate(
                        zip(pcgrad_shared_parameters, component_gradients)
                    ):
                        if gradient is not None:
                            component_buffer[index].add_(gradient.detach().to(parameter))
            scaler.scale(total_loss / accumulation_divisor).backward()
            if batch_runtime_record is not None:
                raw_act_grad = module_grad_norm(model.h6.act_head)
                raw_act_grad_value = (
                    None if raw_act_grad is None
                    else float(raw_act_grad.detach().float().item())
                )
                batch_runtime_record["finite_gradients_after_backward"] = (
                    _runtime_finite_gradients()
                )
                batch_runtime_record["act"]["head_raw_gradient_norm"] = raw_act_grad_value
                batch_runtime_record["act"]["head_weighted_gradient_norm"] = (
                    None if raw_act_grad_value is None
                    else raw_act_grad_value * float(args.lambda_h6_act)
                )
            if args.h6_factor_grad_diagnostics and batch_idx == 1:
                factor_grad_diag.update(
                    factor_gradient_diagnostics(model.h6.semantic_core.concept_slots.grad)
                )
                dynamic_grad = h6_batch["dynamic_text"].grad
                if dynamic_grad is not None:
                    factor_grad_diag["dynamic_residual_grad_norms"] = (
                        dynamic_grad.detach().float().norm(dim=3).mean(dim=(0, 1, 3)).detach()
                    )
                factor_id_grad = model.h6.semantic_core.factor_id_projection.weight.grad
                if factor_id_grad is not None:
                    factor_grad_diag["factor_id_projection_grad_norm"] = factor_id_grad.detach().float().norm()
                core = model.h6.semantic_core
                if getattr(core, "factor_generator_specialization_enabled", False):
                    identity_grad = core.factor_id_embedding.grad
                    if identity_grad is not None:
                        factor_grad_diag["factor_generator_identity_grad_norm"] = identity_grad.detach().float().norm()
                    context_grads = [
                        p.grad.detach().float().norm()
                        for p in core.factor_id_to_context.parameters() if p.grad is not None
                    ]
                    if context_grads:
                        factor_grad_diag["factor_generator_context_grad_norm"] = torch.stack(context_grads).norm()
                    factor_grad_diag["factor_generator_head_grad_norms"] = torch.stack([
                        head.weight.grad.detach().float().norm()
                        if head.weight.grad is not None else torch.zeros((), device=device)
                        for head in core.factor_output_heads
                    ])
                shared_trunk_norms = [
                    module_grad_norm(getattr(core, name, None)) for name in (
                        "normal_state_update", "abnormal_state_update",
                        "state_to_context_normal", "state_to_context_abnormal",
                    )
                ]
                shared_trunk_norms = [value for value in shared_trunk_norms if value is not None]
                if shared_trunk_norms:
                    factor_grad_diag["dynamic_prompt_shared_trunk_grad_norm"] = torch.stack(shared_trunk_norms).norm()
                vae = getattr(core, "class_vae", None)
                factor_grad_diag["vae_mu_grad_norm"] = module_grad_norm(getattr(vae, "mu", None))
                factor_grad_diag["vae_logvar_grad_norm"] = module_grad_norm(getattr(vae, "logvar", None))
                factor_grad_diag["vae_decoder_grad_norm"] = module_grad_norm(getattr(vae, "decoder", None))
                factor_grad_diag["class_to_context_grad_norm"] = module_grad_norm(
                    getattr(core, "class_to_context", None)
                )
                prototype_norms = [module_grad_norm(getattr(core, "prototype_attention", None))]
                prototype_norms = [value for value in prototype_norms if value is not None]
                if prototype_norms:
                    factor_grad_diag["prototype_modules_grad_norm"] = torch.stack(prototype_norms).norm()
                factor_grad_diag["router_grad_norm"] = module_grad_norm(model.h6.router)
                factor_grad_diag["act_head_grad_norm"] = module_grad_norm(
                    getattr(model.h6, "act_head", None)
                )
                rho_modules = [getattr(model.h6, "rho", None)]
                rho_norms = [module_grad_norm(module) for module in rho_modules]
                rho_norms = [value for value in rho_norms if value is not None]
                if rho_norms:
                    factor_grad_diag["rho_gate_grad_norm"] = torch.stack(rho_norms).norm()
                factor_grad_diag["phase2b_image_adapter_grad_norm"] = module_grad_norm(model.image_adapter)
                factor_grad_diag["phase2b_text_adapter_grad_norm"] = module_grad_norm(model.text_adapter)
                dfg_module = (
                    model.image_adapter["vision_text_gate"]
                    if "vision_text_gate" in model.image_adapter
                    else model.image_adapter["vision_text_q"]
                    if "vision_text_q" in model.image_adapter else None
                )
                factor_grad_diag["dfg_grad_norm"] = module_grad_norm(dfg_module)
            if trajectory_enabled and batch_idx in trajectory_gradient_batches:
                trajectory_gradients[batch_idx] = diagnostics_to_python(
                    p1_v83_model_gradient_diagnostics(model, h6_batch, device)
                )
            if args.h6_smoke_backward_only:
                all_gradients_finite = all(
                    parameter.grad is None or torch.isfinite(parameter.grad.detach()).all().item()
                    for parameter in model.parameters()
                )

                def _positive_gradient(value) -> bool:
                    return bool(
                        torch.is_tensor(value)
                        and value.numel() > 0
                        and torch.isfinite(value.detach().float()).all().item()
                        and value.detach().float().abs().max().item() > 0.0
                    )

                vae_class_alive = any(_positive_gradient(factor_grad_diag.get(name)) for name in (
                    "vae_mu_grad_norm", "vae_decoder_grad_norm", "class_to_context_grad_norm",
                ))
                informative = bool(
                    utility_payload is not None and utility_payload["informative"].any().item()
                )
                router_gradient_alive = _positive_gradient(factor_grad_diag.get("router_grad_norm"))
                checks = {
                    "finite_loss": bool(torch.isfinite(total_loss.detach()).all().item()),
                    "finite_gradients": bool(all_gradients_finite),
                    "state_path_gradient": _positive_gradient(
                        factor_grad_diag.get("dynamic_prompt_shared_trunk_grad_norm")
                    ),
                    "vae_class_path_gradient": vae_class_alive,
                    "text_lora_gradient": _positive_gradient(
                        factor_grad_diag.get("phase2b_text_adapter_grad_norm")
                    ),
                    "factor_specific_gradient": _positive_gradient(
                        factor_grad_diag.get("factor_grad_norms")
                    ),
                    "base_adapter_gradient": _positive_gradient(
                        factor_grad_diag.get("phase2b_image_adapter_grad_norm")
                    ),
                    "dfg_gradient": _positive_gradient(factor_grad_diag.get("dfg_grad_norm")),
                    "router_gradient_when_applicable": (not informative) or router_gradient_alive,
                    "act_gradient_when_enabled": (
                        not model.h6.residual_act_enabled
                        or _positive_gradient(factor_grad_diag.get("act_head_grad_norm"))
                    ),
                    "rho_gradient_none": model.h6.rho.raw.grad is None,
                    "rho_fixed_005": bool(torch.equal(
                        model.h6.rho_values().detach().cpu(),
                        torch.full((model.n_groups,), 0.05),
                    )),
                    "optimizer_step_not_executed": len(optimizer.state) == 0,
                }
                payload = diagnostics_to_python({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "checks": checks,
                    "loss_total": total_loss,
                    "loss_task": task_loss,
                    "loss_utility_factor": h6_utility_factor,
                    "loss_utility_router": h6_utility_router,
                    "utility_informative": informative,
                    "gradient_norms": factor_grad_diag,
                    "gradient_attribution": drift_gradient_report,
                    "gpu_allocated_bytes": (
                        torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                    ),
                    "gpu_reserved_bytes": (
                        torch.cuda.memory_reserved(device) if device.type == "cuda" else 0
                    ),
                    "gpu_peak_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                    ),
                    "gpu_peak_reserved_bytes": (
                        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
                    ),
                })
                write_json_atomic(Path(args.save_path) / "backward_probe.json", payload)
                failed = [name for name, passed in checks.items() if not passed]
                if failed:
                    raise RuntimeError(f"P1-v8.3 backward-only probe failed: {failed}")
                logger.info("P1-v8.3 backward-only probe PASS: %s", payload)
                return model
            do_step = batch_idx % args.grad_accum_steps == 0 or batch_idx == epoch_batch_limit
            if do_step:
                scaler.unscale_(optimizer)
                step_act_head_grad_norm = module_grad_norm(model.h6.act_head)
                step_act_head_grad_value = (
                    None if step_act_head_grad_norm is None
                    else float(step_act_head_grad_norm.detach().float().item())
                )
                step_output_weight_before = float(
                    model.h6.act_head[-1].weight.detach().float().norm().item()
                ) if model.h6.progress_version == "P1-v8.4-A" else None
                step_output_bias_before = float(
                    model.h6.act_head[-1].bias.detach().float().norm().item()
                ) if model.h6.progress_version == "P1-v8.4-A" else None
                if (
                    model.h6.progress_version == "P1-v8.4-A"
                    and optimizer_step_count == 0
                ):
                    act_runtime_diag["act_head_gradient_norm_before_step1"] = float(
                        module_grad_norm(model.h6.act_head).detach().float().item()
                    )
                    act_runtime_diag["act_output_weight_norm_before_step1"] = float(
                        model.h6.act_head[-1].weight.detach().float().norm().item()
                    )
                if gradient_surgery_enabled:
                    raw_geometry = _gradient_vector_geometry(
                        pcgrad_main_buffer, pcgrad_factor_buffer, pcgrad_router_buffer
                    )
                    if args.h6_primary_anchored_factor_surgery:
                        projected_main, projected_factor, pcgrad_decision = (
                            primary_anchored_factor_surgery(
                                pcgrad_main_buffer, pcgrad_factor_buffer
                            )
                        )
                        correction_norm = apply_primary_anchored_factor_correction(
                            pcgrad_shared_parameters,
                            pcgrad_factor_buffer,
                            projected_factor,
                        )
                        pcgrad_decision.update({
                            "factor_correction_norm": correction_norm,
                            "correction_reconstruction_error_norm": abs(
                                correction_norm
                                - pcgrad_decision["removed_factor_component_norm"]
                            ),
                            "router_gradient_projected": False,
                        })
                        artifact_name = "primary_anchored_factor_surgery_windows.json"
                        formula = (
                            "main-preserving auxiliary projection on accumulated "
                            "main/factor shared-semantic gradients"
                        )
                        decision_key = "primary_anchored_factor_surgery"
                    else:
                        projected_main, projected_factor, pcgrad_decision = (
                            pcgrad_project_two_task(
                                pcgrad_main_buffer, pcgrad_factor_buffer
                            )
                        )
                        for parameter, raw_main, raw_factor, new_main, new_factor in zip(
                            pcgrad_shared_parameters,
                            pcgrad_main_buffer,
                            pcgrad_factor_buffer,
                            projected_main,
                            projected_factor,
                        ):
                            correction = (new_main + new_factor) - (raw_main + raw_factor)
                            if parameter.grad is None:
                                parameter.grad = correction.clone()
                            else:
                                parameter.grad.add_(correction.to(parameter.grad))
                        artifact_name = "pcgrad_windows.json"
                        formula = (
                            "two-objective symmetric PCGrad on accumulated "
                            "main/factor shared-semantic gradients"
                        )
                        decision_key = "pcgrad"
                    projected_geometry = _gradient_vector_geometry(
                        projected_main, projected_factor, pcgrad_router_buffer
                    )
                    valid_count = pcgrad_window_counts["valid_patch_count"]
                    valid_group_count = pcgrad_window_counts["valid_group_patch_count"]
                    microbatch_count = max(1, pcgrad_window_counts["microbatch_count"])
                    pcgrad_window_records.append({
                        "optimizer_step": optimizer_step_count + 1,
                        "ending_batch": batch_idx,
                        "microbatch_count": pcgrad_window_counts["microbatch_count"],
                        "counts": {
                            **{
                                key: value for key, value in pcgrad_window_counts.items()
                                if not key.endswith("_sum") and key != "microbatch_count"
                            },
                            "anomaly_fraction": (
                                pcgrad_window_counts["anomaly_patch_count"] / valid_count
                                if valid_count else 0.0
                            ),
                            "informative_fraction": (
                                pcgrad_window_counts["informative_group_patch_count"] / valid_group_count
                                if valid_group_count else 0.0
                            ),
                        },
                        "losses": {
                            "factor": pcgrad_window_counts["factor_loss_sum"] / microbatch_count,
                            "router": pcgrad_window_counts["router_loss_sum"] / microbatch_count,
                        },
                        "raw_weighted_geometry": raw_geometry,
                        decision_key: pcgrad_decision,
                        "safe_weighted_geometry": projected_geometry,
                    })
                    write_json_atomic(
                        Path(args.save_path) / "optimizer_windows" / artifact_name,
                        {
                            "formula": formula,
                            "main_gradient_policy": (
                                "exactly preserved"
                                if args.h6_primary_anchored_factor_surgery
                                else "symmetric PCGrad projection"
                            ),
                            "router": (
                                "support-normalized fixed-lambda gradient tracked but not projected"
                                if collect_router_gradient_geometry
                                else "support-normalized fixed-lambda gradient not isolated and not projected"
                            ),
                            "lambda_factor": args.lambda_h6_factor,
                            "lambda_router": args.lambda_h6_router,
                            "factor_effective_beta": args.h6_utility_factor_effective_beta,
                            "windows": pcgrad_window_records,
                        },
                    )
                    for buffer in (
                        pcgrad_main_buffer, pcgrad_factor_buffer, pcgrad_router_buffer
                    ):
                        for tensor in buffer:
                            tensor.zero_()
                    for key in pcgrad_window_counts:
                        pcgrad_window_counts[key] = 0.0 if key.endswith("_sum") else 0
                if has_non_finite_grad(optimizer):
                    if batch_runtime_record is not None:
                        batch_runtime_record["finite_gradients_after_backward"] = False
                    optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError(f"non-finite H6 gradient at epoch={epoch}, batch={batch_idx}")
                def _grad_norm(parameter):
                    return parameter.grad.detach().float().norm() if parameter is not None and parameter.grad is not None else torch.zeros((), device=device)
                if model.h6.paired_experts is not None:
                    expert_b_grad = model.h6.paired_experts.expert_B.grad
                    expert_b_per_factor = (
                        expert_b_grad.detach().float().norm(dim=(1, 2)) if expert_b_grad is not None
                        else torch.zeros(model.h6.num_factors, device=device)
                    )
                    gradient_samples.append({
                        "expert_B_grad_norm": _grad_norm(model.h6.paired_experts.expert_B),
                        "expert_B_grad_norm_per_factor": expert_b_per_factor,
                        "expert_state_projection_grad_norm": _grad_norm(model.h6.paired_experts.state_projection.weight),
                        "router_query_grad_norm": _grad_norm(model.h6.router.local_query_projector[0].weight) + _grad_norm(model.h6.router.query_projector[0].weight),
                        "router_key_adaptation_grad_norm": _grad_norm(model.h6.semantic_core.router_key.weight),
                    })
                nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer_step_count += 1
                needs_v84_post_step = bool(
                    model.h6.progress_version == "P1-v8.4-A"
                    and optimizer_step_count == 1
                    and act_runtime_diag["act_output_weight_norm_after_step1"] is None
                )
                if needs_v84_post_step:
                    act_runtime_diag["act_output_weight_norm_after_step1"] = float(
                        model.h6.act_head[-1].weight.detach().float().norm().item()
                    )
                if (
                    needs_v84_post_step
                    or (
                        args.h6_drift_diagnostics
                        and "batch_000_after_first_optimizer_step" not in drift_snapshots
                    )
                ):
                    with torch.no_grad(), _phase4_autocast(device, args.precision):
                        post_step_visual = model(image, return_phase4_features=True)
                        post_step_batch = model.h6.build_batch(
                            model, dataset_name, class_names, post_step_visual,
                            hybrid_alpha=hybrid_alpha, update_load_bias=False,
                        )
                    if needs_v84_post_step:
                        act_runtime_diag["act_probability_mean_after_step1"] = float(
                            post_step_batch["act_probability"].detach().float().mean().item()
                        )
                    if args.h6_drift_diagnostics:
                        _record_drift_snapshot(
                            "batch_000_after_first_optimizer_step", post_step_batch
                        )
                if model.h6.progress_version == "P1-v8.4-A":
                    surgery_record = (
                        pcgrad_window_records[-1]
                        if pcgrad_window_records else {}
                    )
                    surgery_decision = surgery_record.get(
                        "primary_anchored_factor_surgery", {}
                    )
                    step_surgery_error = float(
                        surgery_decision.get(
                            "correction_reconstruction_error_norm", float("inf")
                        )
                    )
                    step_main_exact_change = float(
                        surgery_decision.get(
                            "main_gradient_exact_change_norm", float("inf")
                        )
                    )
                    step_finite_parameters = _runtime_finite_parameters()
                    step_finite_gradients = _runtime_finite_gradients()
                    step_weight_after = float(
                        model.h6.act_head[-1].weight.detach().float().norm().item()
                    )
                    step_bias_after = float(
                        model.h6.act_head[-1].bias.detach().float().norm().item()
                    )
                    step_upstream_values = [
                        record["act"]["upstream_gradient_norm"]
                        for record in current_window_batch_records
                        if record["act"]["upstream_gradient_norm"] is not None
                    ]
                    step_batch_start = (
                        current_window_batch_records[0]["batch"]
                        if current_window_batch_records else batch_idx
                    )
                    step_reconstruction = {
                        "residual_definition_max_abs_error": max(
                            (
                                record["reconstruction"][
                                    "residual_definition_max_abs_error"
                                ]
                                for record in current_window_batch_records
                                if record["reconstruction"][
                                    "residual_definition_max_abs_error"
                                ] is not None
                            ),
                            default=float("inf"),
                        ),
                        "routed_correction_max_abs_error": max(
                            (
                                record["reconstruction"][
                                    "routed_correction_max_abs_error"
                                ]
                                for record in current_window_batch_records
                                if record["reconstruction"][
                                    "routed_correction_max_abs_error"
                                ] is not None
                            ),
                            default=float("inf"),
                        ),
                        "actual_gated_max_abs_error": max(
                            (
                                record["reconstruction"][
                                    "actual_gated_max_abs_error"
                                ]
                                for record in current_window_batch_records
                                if record["reconstruction"][
                                    "actual_gated_max_abs_error"
                                ] is not None
                            ),
                            default=float("inf"),
                        ),
                        "surgery_max_abs_error": step_surgery_error,
                        "main_exact_change_max_abs_error": step_main_exact_change,
                    }
                    for record in current_window_batch_records:
                        record["optimizer_step"] = int(optimizer_step_count)
                        record["finite_parameters_after_step"] = step_finite_parameters
                        record["reconstruction"]["surgery_max_abs_error"] = step_surgery_error
                        record["reconstruction"][
                            "main_exact_change_max_abs_error"
                        ] = step_main_exact_change
                    optimizer_step_runtime_records.append({
                        "optimizer_step": int(optimizer_step_count),
                        "batch_range": [int(step_batch_start), int(batch_idx)],
                        "microbatch_count": len(current_window_batch_records),
                        "rho": [
                            float(value)
                            for value in model.h6.rho_values().detach().float().tolist()
                        ],
                        "rho_trainable": bool(model.h6.rho.raw.requires_grad),
                        "finite_gradients_before_step": step_finite_gradients,
                        "finite_parameters_after_step": step_finite_parameters,
                        "reconstruction": step_reconstruction,
                        "gradient_scale": {
                            "definition": (
                                "ACT head total raw gradient norm before step divided by "
                                "primary-anchored shared MAIN gradient norm before step; "
                                "weighted ratio multiplies raw ratio by lambda_act"
                            ),
                            "main_gradient_norm_before_step": surgery_decision.get(
                                "main_norm"
                            ),
                            "raw_ratio": (
                                None
                                if step_act_head_grad_value is None
                                or surgery_decision.get("main_norm") in (None, 0.0)
                                else float(step_act_head_grad_value)
                                / float(surgery_decision["main_norm"])
                            ),
                            "weighted_ratio": (
                                None
                                if step_act_head_grad_value is None
                                or surgery_decision.get("main_norm") in (None, 0.0)
                                else float(step_act_head_grad_value)
                                / float(surgery_decision["main_norm"])
                                * float(args.lambda_h6_act)
                            ),
                        },
                        "act": {
                            "head_raw_gradient_norm_before_step": step_act_head_grad_value,
                            "head_weighted_gradient_norm_before_step": (
                                None if step_act_head_grad_value is None
                                else step_act_head_grad_value * float(args.lambda_h6_act)
                            ),
                            "upstream_gradient_norms_after_previous_step": step_upstream_values,
                            "output_weight_norm_before_step": step_output_weight_before,
                            "output_weight_norm_after_step": step_weight_after,
                            "output_bias_norm_before_step": step_output_bias_before,
                            "output_bias_norm_after_step": step_bias_after,
                            "probability_mean_before_step": (
                                current_window_batch_records[-1]["act"]["probability"]["mean"]
                                if current_window_batch_records else None
                            ),
                            "probability_mean_after_step": (
                                float(post_step_batch["act_probability"].detach().float().mean().item())
                                if needs_v84_post_step else None
                            ),
                        },
                    })
                    current_window_batch_records.clear()
                optimizer.zero_grad(set_to_none=True)
            if trajectory_enabled and batch_idx in trajectory_milestones:
                rho_values_now = model.h6.rho_values().detach().float()
                if not torch.equal(rho_values_now, torch.full_like(rho_values_now, 0.05)):
                    raise RuntimeError(f"P1-v8.3 rho changed at trajectory batch {batch_idx}")
                if model.h6.rho.raw.grad is not None:
                    raise RuntimeError(f"P1-v8.3 rho unexpectedly received gradient at batch {batch_idx}")
                cumulative = aggregate_utility_records(
                    trajectory_records,
                    gain_threshold=args.h6_router_gain_threshold,
                    entropy_threshold=args.h6_utility_entropy_threshold,
                )
                recent = aggregate_utility_records(
                    trajectory_records[trajectory_previous_batch:batch_idx],
                    gain_threshold=args.h6_router_gain_threshold,
                    entropy_threshold=args.h6_utility_entropy_threshold,
                )
                milestone_payload = {
                    "batch": batch_idx,
                    "optimizer_steps": optimizer_step_count,
                    "cumulative_range": [1, batch_idx],
                    "recent_window_range": [trajectory_previous_batch + 1, batch_idx],
                    "cumulative": cumulative,
                    "recent_window": recent,
                    "structure": trajectory_structures[batch_idx],
                    "gradients": trajectory_gradients.get(batch_idx),
                    "gradient_attribution": diagnostics_to_python(
                        trajectory_attribution.get(batch_idx)
                    ),
                    "rho": diagnostics_to_python(rho_values_now),
                    "gpu": {
                        "allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
                        "reserved_bytes": torch.cuda.memory_reserved(device) if device.type == "cuda" else 0,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
                    },
                    "elapsed_seconds": time.monotonic() - started,
                }
                trajectory_outputs.append(milestone_payload)
                trajectory_previous_batch = batch_idx
                write_json_atomic(
                    Path(args.save_path) / "milestones" / f"batch_{batch_idx:03d}.json",
                    milestone_payload,
                )
                write_trajectory_artifacts(args.save_path, trajectory_outputs)
            for key, value in {
                "total": total_loss, "task": task_loss, "cls": cls_loss, "seg": seg_loss,
                "center": h6_center,
                "router_teacher": h6_router_teacher,
                "router_teacher_weighted": effective_router_teacher_weight * h6_router_teacher,
                "router_teacher_scheduled_weight": torch.as_tensor(router_teacher_weight, device=device),
                "router_teacher_effective_weight": torch.as_tensor(effective_router_teacher_weight, device=device),
                "vae_rec": h6_batch["reconstruction"],
                "vae_kl_raw": h6_kl_raw,
                "vae_kl_effective": h6_kl_effective,
                "utility_factor": h6_utility_factor,
                "utility_router": h6_utility_router,
                "utility_act": h6_utility_act,
                "utility_act_weighted": args.lambda_h6_act * h6_utility_act,
                "kg": h6_batch["kg_loss"], "orth": h6_orth, "balance": h6_balance,
                "concept_key_diversity_raw": h6_concept_key_diversity,
                "concept_key_diversity_weighted": concept_key_diversity_weight * h6_concept_key_diversity,
                "dynamic_mean_anchor_raw": h6_batch["dynamic_mean_anchor_loss_raw"],
                "dynamic_mean_anchor_weighted": dynamic_mean_anchor_weight * h6_batch["dynamic_mean_anchor_loss_raw"],
                "route_raw": h6_route,
                "route_weight": torch.as_tensor(getattr(args, "lambda_h6_route", 0.0), device=device),
                "route_weighted": getattr(args, "lambda_h6_route", 0.0) * h6_route,
                "factor_role_raw": h6_factor_role,
                "factor_role_weight": torch.as_tensor(getattr(args, "lambda_h6_factor_role", 0.0), device=device),
                "factor_role_weighted": getattr(args, "lambda_h6_factor_role", 0.0) * h6_factor_role,
                "actual_local_raw": h6_actual_local,
                "actual_local_weight": torch.as_tensor(getattr(args, "lambda_h6_actual_local", 0.0), device=device),
                "actual_local_weighted": getattr(args, "lambda_h6_actual_local", 0.0) * h6_actual_local,
                "dynamic_mean_anchor_weight": torch.as_tensor(dynamic_mean_anchor_weight, device=device),
                "expert": h6_expert, "advantage": h6_advantage, "etf": h6_etf,
                "expert_anchor": h6_expert_anchor, "expert_radius": h6_expert_radius,
                "expert_weight": torch.as_tensor(expert_weight, device=device),
                "advantage_weight": torch.as_tensor(advantage_weight, device=device),
                "etf_weight": torch.as_tensor(etf_weight, device=device),
                "balance_weight": torch.as_tensor(balance_weight, device=device),
                "expert_delta_norm": expert_delta_norm,
                "loss_component_sum": loss_component_sum, "loss_component_residual": loss_component_residual,
                "expert_assigned_normal": expert_terms["expert_normal_all"],
                "expert_assigned_abnormal": expert_terms["expert_abnormal_assigned"],
                "selected_expert_loss": expert_function["selected_expert_loss"],
                "nonselected_expert_loss": expert_function["nonselected_expert_loss"],
                "expert_advantage_margin_satisfied_fraction": expert_function["expert_advantage_margin_satisfied_fraction"],
                "expert_assigned_raw": h6_expert,
                "expert_assigned_weight": torch.as_tensor(expert_weight, device=device),
                "expert_assigned_weighted": expert_weight * h6_expert,
                "expert_advantage_raw": h6_advantage,
                "expert_advantage_weight": torch.as_tensor(advantage_weight, device=device),
                "expert_advantage_weighted": advantage_weight * h6_advantage,
                "expert_etf_raw": h6_etf,
                "expert_etf_weight": torch.as_tensor(etf_weight, device=device),
                "expert_etf_weighted": etf_weight * h6_etf,
                "expert_balance_dense_raw": balance_terms["dense_cv2"],
                "expert_balance_prediction_raw": balance_terms["prediction_cv2"],
                "expert_balance_weight": torch.as_tensor(balance_weight, device=device),
                "expert_balance_weighted": balance_weight * h6_balance,
                "expert_anchor_raw": h6_expert_anchor,
                "expert_anchor_weight": torch.as_tensor(args.lambda_h6_expert_anchor, device=device),
                "expert_anchor_weighted": args.lambda_h6_expert_anchor * h6_expert_anchor,
                "expert_radius_raw": h6_expert_radius,
                "expert_radius_weight": torch.as_tensor(args.lambda_h6_expert_radius, device=device),
                "expert_radius_weighted": args.lambda_h6_expert_radius * h6_expert_radius,
                "legacy_pre_expert_orth_raw": h6_orth,
                "legacy_pre_expert_orth_weight": torch.as_tensor(args.lambda_h6_orth, device=device),
                "legacy_pre_expert_orth_weighted": args.lambda_h6_orth * h6_orth,
                "delta_t_diversity_raw": h6_delta_div,
                "delta_t_diversity_weight": torch.as_tensor(args.lambda_h6_delta_div, device=device),
                "delta_t_diversity_weighted": args.lambda_h6_delta_div * h6_delta_div,
                "functional_factor_diversity_raw": h6_func_div,
                "functional_factor_diversity_weight": torch.as_tensor(args.lambda_h6_func_div, device=device),
                "functional_factor_diversity_weighted": args.lambda_h6_func_div * h6_func_div,
                "cluster_resp_raw": h6_cluster_resp,
                "cluster_resp_weight": torch.as_tensor(args.h6_lambda_cluster_resp, device=device),
                "cluster_resp_weighted": args.h6_lambda_cluster_resp * h6_cluster_resp,
                "cluster_target_entropy": cluster_diag.get("cluster_target_entropy", h6_cluster_resp.detach() * 0.0),
                "cluster_router_entropy": cluster_diag.get("cluster_router_entropy", h6_cluster_resp.detach() * 0.0),
            }.items():
                metrics[key].append(scalar_metric_value(value))
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / 2**30
                reserved = torch.cuda.memory_reserved(device) / 2**30
                peak = torch.cuda.max_memory_allocated(device) / 2**30
            else:
                allocated = reserved = peak = 0.0
            elapsed = time.monotonic() - started
            remaining = max(epoch_batch_limit - batch_idx, 0)
            eta = elapsed / batch_idx * remaining if batch_idx else 0.0
            progress_postfix = {
                "loss": f"{metrics['total'][-1]:.4f}", "task": f"{metrics['task'][-1]:.4f}",
                "center": f"{metrics['center'][-1]:.4f}", "vae": f"{metrics['vae_rec'][-1]:.4f}",
                "kl": f"{metrics['vae_kl_raw'][-1]:.5f}", "rt": f"{metrics['router_teacher'][-1]:.4f}",
                "sr": f"{float(h6_batch['sparse_ratio'].detach().item()):.2f}", "kg": f"{metrics['kg'][-1]:.5f}",
                "orth": f"{metrics['orth'][-1]:.4f}", "balance": f"{metrics['balance'][-1]:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}", "alpha": f"{hybrid_alpha:.2f}",
                "gamma_s": f"{h6_batch['gamma_state'].detach().item():.3f}",
                "gamma_c": f"{h6_batch['gamma_class'].detach().item():.3f}",
                "rho": "/".join(f"{x:.2f}" for x in h6_batch["rho"].detach().cpu()),
                "vram": f"{allocated:.2f}/{reserved:.2f}/{peak:.2f}G", "eta": f"{eta / 60:.1f}m",
            }
            if getattr(model.h6, "progress_version", "P1-v6") == "P1-v7-full":
                progress_postfix.pop("orth", None)
            progress.set_postfix(progress_postfix)
        scheduler.step()
        _set_soft_prompt_lr(optimizer, soft_prompt_frozen)
        diagnostics = {
            key: value / float(epoch_diag_count[key])
            for key, value in epoch_diag_sum.items()
        }
        for key, values in epoch_probe_values.items():
            stacked = torch.stack(values)
            diagnostics[f"{key}_min"] = stacked.min().to(device)
            diagnostics[f"{key}_p05"] = torch.quantile(stacked, 0.05).to(device)
        diagnostics["gate_sample_count"] = torch.tensor(float(epoch_batch_limit), device=device)
        if model.h6.expert_enabled:
            diagnostics["expert_anchor_floor"] = torch.tensor(
                float(args.h6_expert_anchor_min_cosine), device=device
            )
        proto_diag = prototype_diagnostics(h6_batch["prototype_normal"], h6_batch["prototype_abnormal"])
        dynamic_hard_cosine = F.cosine_similarity(
            h6_batch["dynamic_text"].float(),
            h6_batch["hard_frozen"].unsqueeze(2).expand_as(h6_batch["dynamic_text"]).float(),
            dim=3,
        ).mean()
        center_distance = (h6_batch["prototype_normal"].float() - h6_batch["prototype_abnormal"].float()).norm(dim=-1).mean()
        sparse_ratio = float(diagnostics["sparse_ratio"].detach().item())
        routing_mode = "dense" if sparse_ratio == 0.0 else ("straight_through_sparse" if sparse_ratio == 1.0 else "mixed")
        mu_std = float(h6_batch["mu"].detach().float().std(unbiased=False).item())
        if beta_vae_kl > 0.0 and mu_std < 0.003:
            logger.warning("h6_vae_mu_collapse_warning epoch=%d mu_std=%s beta_vae_kl=%s", epoch, mu_std, beta_vae_kl)
        def _teacher_value(key: str):
            value = teacher_diag.get(key, None)
            if torch.is_tensor(value):
                return value.detach().cpu().tolist()
            return None

        def _diag_float(key: str) -> float:
            return float(diagnostics[key].detach().float().cpu().item())

        epoch_metric_means = {key: float(np.mean(values)) for key, values in metrics.items() if values}
        gradient_epoch = {
            key: torch.stack([sample[key] for sample in gradient_samples]).mean(dim=0)
            for key in gradient_samples[0]
        } if gradient_samples else {
            "expert_B_grad_norm": torch.zeros((), device=device),
            "expert_B_grad_norm_per_factor": torch.zeros(model.h6.num_factors, device=device),
            "expert_state_projection_grad_norm": torch.zeros((), device=device),
            "router_query_grad_norm": torch.zeros((), device=device),
            "router_key_adaptation_grad_norm": torch.zeros((), device=device),
        }
        epoch_metric_means["task_rolling_median"] = (
            float(np.median(task_loss_history[-5:])) if task_loss_history else epoch_metric_means.get("task", 0.0)
        )
        gate_decision = structural_gate.evaluate(
            epoch=epoch,
            diagnostics=diagnostics,
            epoch_metrics=epoch_metric_means,
            teacher_diag=teacher_diag,
            sparse_ratio=sparse_ratio,
            hybrid_alpha=hybrid_alpha,
            hybrid_alpha_max=args.hybrid_alpha_max,
            router_teacher_weight=router_teacher_weight,
            router_teacher_target=args.lambda_h6_router_teacher,
            dynamic_mean_anchor_weight=dynamic_mean_anchor_weight,
            dynamic_mean_anchor_target=args.lambda_h6_dynamic_mean_anchor,
            query_mode=args.h6_router_query_mode,
            tangent_enabled=bool(args.h6_factor_identity_tangent_projection_enabled),
            expert_enabled=bool(model.h6.expert_enabled),
            expert_scale=float(model.h6.expert_scale()),
            expert_scale_target=float(model.h6.expert_scale_target),
            etf_weight=float(etf_weight),
            etf_target=float(args.lambda_h6_etf),
            gate_sample_count=len(train_loader) if args.h6_smoke_max_batches == 0 else epoch_batch_limit,
        )
        task_loss_history.append(epoch_metric_means.get("task", 0.0))

        is_v7 = getattr(model.h6, "progress_version", "P1-v6") == "P1-v7-full"
        if not is_v7:
            write_json_atomic(
                Path(args.save_path) / "diagnostics" / f"epoch_{epoch:03d}.json",
                diagnostics_to_python({
                    "progress_version": model.h6.progress_version,
                    "loss_components": epoch_metric_means,
                    "gradients": factor_grad_diag,
                    "router": {
                        "dense_usage": diagnostics.get("dense_factor_usage"),
                        "sparse_usage": diagnostics.get("sparse_factor_usage"),
                        "unique_topk_pairs": diagnostics.get("unique_topk_pairs"),
                    },
                    "gate": {
                        "hard_failure": gate_decision.hard_failure,
                        "soft_warnings": gate_decision.soft_warnings,
                        "abort_reason": gate_decision.abort_reason,
                        "sample_count": epoch_batch_limit,
                    },
                }),
            )
            loss_row = {
                "epoch": epoch,
                **epoch_metric_means,
                **{f"weight_{key}": value for key, value in loss_weights.items()},
            }
            loss_csv_path = Path(args.save_path) / "losses.csv"
            with open(loss_csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(loss_row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(loss_row)
            gradient_row = {"epoch": epoch}
            for key, value in diagnostics_to_python(factor_grad_diag).items():
                gradient_row[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
            gradient_csv_path = Path(args.save_path) / "gradient_norms.csv"
            with open(gradient_csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(gradient_row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(gradient_row)
        if is_v7:
            def _d(name, default=0.0):
                value = diagnostics.get(name)
                if value is None:
                    return default
                if torch.is_tensor(value):
                    return value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
                return value
            dead_threshold = float(args.h6_expert_dead_usage_threshold)
            dense_dead_mask, dense_dead_count = expert_dead_counts(diagnostics["dense_factor_usage"], dead_threshold)
            sparse_dead_mask, sparse_dead_count = expert_dead_counts(diagnostics["sparse_factor_usage"], dead_threshold)
            gate_mode = structural_gate_config.mode
            gate_raw_state = (
                "hard_failure" if gate_decision.hard_failure
                else "warning" if gate_decision.soft_warnings or any(gate_decision.failed.values())
                else "ok"
            )
            expert_ready = float(model.h6.expert_scale()) >= 0.95 * float(model.h6.expert_scale_target)
            gate_effective_state = (
                ("warmup" if not expert_ready else "monitoring")
                if gate_mode == "monitor" else gate_raw_state
            )
            logger.info(
                "phase4_p1_v7_full epoch=%d total=%.6f task=%.6f cls=%.6f seg=%.6f lr=%.3e "
                "expert_scale=%.5f expert_delta_norm=%s expert_delta_cos=%s final_expert_direction_cos=%s expert_logit_std=%s expert_B_grad_norm=%s "
                "assigned_w=%.6f advantage_w=%.6f ETF_w=%.6f balance_w=%.6f anchor_w=%.6f radius_w=%.6f "
                "anchor_cos=%s radius_ratio=%s clamp_frac=%s dense_usage=%s sparse_usage=%s dead_experts=%s unique_pairs=%s "
                "gate_mode=%s gate_raw_state=%s gate_effective_state=%s hard_failure=%s abort_reason=%s",
                epoch, epoch_metric_means["total"], epoch_metric_means["task"], epoch_metric_means["cls"], epoch_metric_means["seg"], optimizer.param_groups[0]["lr"],
                float(model.h6.expert_scale()), _d("expert_delta_norm_mean"), _d("expert_delta_tangent_cos_mean"), _d("final_expert_direction_cos_mean"),
                _d("expert_patch_logit_std_across_experts"), diagnostics_to_python(gradient_epoch["expert_B_grad_norm"]),
                epoch_metric_means["expert_assigned_weighted"], epoch_metric_means["expert_advantage_weighted"], epoch_metric_means["expert_etf_weighted"],
                epoch_metric_means["expert_balance_weighted"], epoch_metric_means["expert_anchor_weighted"], epoch_metric_means["expert_radius_weighted"],
                _d("final_expert_mean_hard_cos_mean"), _d("expert_residual_relative_ratio_mean"), _d("expert_residual_clamp_fraction"),
                _d("dense_factor_usage"), _d("sparse_factor_usage"), diagnostics_to_python(sparse_dead_count), _d("unique_topk_pairs"),
                gate_mode, gate_raw_state, gate_effective_state, gate_decision.hard_failure, gate_decision.abort_reason,
            )
            detailed_loss_components = {
                key: value for key, value in epoch_metric_means.items() if key != "orth"
            }
            write_json_atomic(
                Path(args.save_path) / "diagnostics" / f"epoch_{epoch:03d}.json",
                diagnostics_to_python({
                    "progress_version": model.h6.progress_version, "checkpoint_version": 7,
                    "loss_components": {**detailed_loss_components, "component_residual": epoch_metric_means["loss_component_residual"]},
                    "expert_schedule": {"current": model.h6.expert_scale(), "target": model.h6.expert_scale_target},
                    "expert_geometry": {
                        key: _d(key) for key in diagnostics
                        if key.startswith(("expert_", "pre_expert_", "final_expert_"))
                    },
                    "router": {
                        "dense_usage": _d("dense_factor_usage"), "sparse_usage": _d("sparse_factor_usage"),
                        "dense_dead_mask": diagnostics_to_python(dense_dead_mask), "dense_dead_count": diagnostics_to_python(dense_dead_count),
                        "sparse_dead_mask": diagnostics_to_python(sparse_dead_mask), "sparse_dead_count": diagnostics_to_python(sparse_dead_count),
                        "dead_usage_threshold": dead_threshold, "unique_topk_pairs": _d("unique_topk_pairs"),
                    },
                    "gate": {
                        "mode": gate_mode, "raw_state": gate_raw_state, "effective_state": gate_effective_state,
                        "soft_warnings": gate_decision.soft_warnings, "hard_failure": gate_decision.hard_failure,
                        "abort_reason": gate_decision.abort_reason, "decision": structural_gate.decision_to_dict(gate_decision),
                        "sample_count": epoch_batch_limit,
                    },
                    "gradients": diagnostics_to_python(gradient_epoch), "patch_counts": {key: _d(key) for key in diagnostics if key.startswith("expert_") and key.endswith("patch_count")},
                }),
            )
            if args.h6_drift_diagnostics:
                write_json_atomic(
                    Path(args.save_path) / "diagnostics" / f"drift_epoch_{epoch:03d}.json",
                    diagnostics_to_python({
                        "progress_version": model.h6.progress_version,
                        "snapshots": drift_snapshots,
                        "gradient_attribution": drift_gradient_report,
                        "loss_components": detailed_loss_components,
                        "expert_B_grad_norm": diagnostics_to_python(gradient_epoch["expert_B_grad_norm"]),
                        "shared_semantic_gradient": None if drift_gradient_report is None else drift_gradient_report["components"]["main_task"].get("shared_semantic"),
                        "alpha_current": hybrid_alpha,
                        "expert_scale_current": model.h6.expert_scale(),
                    }),
                )

        if model.h6.progress_version in {"P1-v8.3", "P1-v8.4-A"} and args.h6_smoke_max_batches > 0:
            def _positive_diagnostic(name):
                value = factor_grad_diag.get(name)
                return bool(
                    torch.is_tensor(value)
                    and torch.isfinite(value.detach().float()).all().item()
                    and value.detach().float().abs().max().item() > 0.0
                )

            rho_values = model.h6.rho_values().detach().float()
            utility_epoch = {
                key: diagnostics.get(key)
                for key in set(utility_diag) | set(act_diag)
                if key in diagnostics
            }
            structure_epoch = {
                key: diagnostics.get(key) for key in structure_diag if key in diagnostics
            }
            checks = {
                "all_epoch_metrics_finite": all(
                    np.isfinite(value) for value in epoch_metric_means.values()
                ),
                "utility_losses_finite": all(np.isfinite(epoch_metric_means[key]) for key in (
                    "utility_factor", "utility_router", "utility_act",
                )),
                "optimizer_step_executed": optimizer_step_count >= 1,
                "rho_fixed_005": bool(torch.equal(
                    rho_values, torch.full_like(rho_values, 0.05)
                )),
                "rho_gradient_none": model.h6.rho.raw.grad is None,
                "state_path_alive": _positive_diagnostic(
                    "dynamic_prompt_shared_trunk_grad_norm"
                ),
                "class_vae_path_alive": any(_positive_diagnostic(name) for name in (
                    "vae_mu_grad_norm", "vae_decoder_grad_norm", "class_to_context_grad_norm",
                )),
                "text_lora_alive": _positive_diagnostic(
                    "phase2b_text_adapter_grad_norm"
                ),
                "factor_gradients_alive": _positive_diagnostic("factor_grad_norms"),
                "router_gradient_alive": _positive_diagnostic("router_grad_norm"),
                "factor_outputs_not_collapsed": bool(
                    structure_epoch["factor_patch_outputs_exactly_collapsed"].item() == 0.0
                    and wiring_factor_collapse_probes < 2
                ),
                "router_dense": routing_mode == "dense" and args.h6_prediction_routing == "dense",
                "legacy_auxiliaries_off": all(value == 0.0 for value in (
                    args.lambda_h6_balance, args.lambda_h6_center, args.lambda_h6_orth,
                    args.lambda_h6_route, args.lambda_h6_factor_role,
                    args.lambda_h6_actual_local, args.lambda_h6_func_div,
                    args.lambda_h6_router_teacher, args.h6_lambda_cluster_resp,
                )),
                "experts_off": not model.h6.expert_enabled,
            }
            if model.h6.progress_version == "P1-v8.4-A":
                expected_steps = (
                    epoch_batch_limit // args.grad_accum_steps
                    + int(epoch_batch_limit % args.grad_accum_steps != 0)
                )
                surgery_decisions = [
                    window.get("primary_anchored_factor_surgery", {})
                    for window in pcgrad_window_records
                ]
                main_exact_change_max = max(
                    (float(item.get("main_gradient_exact_change_norm", float("inf")))
                     for item in surgery_decisions),
                    default=float("inf"),
                )
                surgery_reconstruction_error_max = max(
                    (float(item.get("correction_reconstruction_error_norm", float("inf")))
                     for item in surgery_decisions),
                    default=float("inf"),
                )
                act_mean = float(diagnostics["act_probability_mean"].item())
                act_runtime_diag.update({
                    "main_gradient_exact_change_max": main_exact_change_max,
                    "surgery_correction_reconstruction_error_max": (
                        surgery_reconstruction_error_max
                    ),
                })
                checks.update({
                    "residual_semantics_enabled": model.h6.residual_act_enabled,
                    "act_head_present": model.h6.act_head is not None,
                    "act_forward_finite": bool(torch.isfinite(h6_batch["act_probability"]).all().item()),
                    "act_loss_finite": bool(np.isfinite(epoch_metric_means["utility_act"])),
                    "act_gradient_alive": bool(
                        act_runtime_diag["act_head_gradient_norm_before_step1"] is not None
                        and act_runtime_diag["act_head_gradient_norm_before_step1"] > 0.0
                    ),
                    "act_initial_mean_half": bool(
                        act_runtime_diag["act_probability_mean_before_first_update"] is not None
                        and abs(
                            act_runtime_diag["act_probability_mean_before_first_update"] - 0.5
                        ) <= 1e-7
                    ),
                    "act_output_weight_initially_zero": bool(
                        act_runtime_diag["act_output_weight_norm_initial"] == 0.0
                        and act_runtime_diag["act_output_weight_norm_before_step1"] == 0.0
                    ),
                    "act_output_weight_left_zero_after_step1": bool(
                        act_runtime_diag["act_output_weight_norm_after_step1"] is not None
                        and act_runtime_diag["act_output_weight_norm_after_step1"] > 0.0
                    ),
                    "post_step_upstream_act_path_reachable": bool(
                        act_runtime_diag["post_step_upstream_act_gradient_norm"] is not None
                        and act_runtime_diag["post_step_upstream_act_gradient_norm"] > 0.0
                    ),
                    "act_not_saturated": 1e-4 < act_mean < 1.0 - 1e-4,
                    "exact_noop_residual_invariant": (
                        act_runtime_diag["residual_definition_max_error"] == 0.0
                    ),
                    "local_correction_reconstruction": (
                        act_runtime_diag["local_correction_reconstruction_max_error"]
                        <= 1e-7
                    ),
                    "main_gradient_exact_change_zero": main_exact_change_max == 0.0,
                    "surgery_correction_reconstruction": (
                        surgery_reconstruction_error_max <= 1e-12
                    ),
                    "optimizer_window_count_exact": optimizer_step_count == expected_steps,
                })
            smoke_payload = diagnostics_to_python({
                "status": "PASS" if all(checks.values()) else "FAIL",
                "epoch": epoch,
                "sample_count": epoch_batch_limit,
                "grad_accum_steps": args.grad_accum_steps,
                "optimizer_step_count": optimizer_step_count,
                "checks": checks,
                "failed": [name for name, passed in checks.items() if not passed],
                "rho": rho_values,
                "losses": {
                    key: epoch_metric_means[key] for key in (
                        "task", "utility_factor", "utility_router", "utility_act", "total"
                    )
                },
                "utility": utility_epoch,
                "structure": structure_epoch,
                "gradients": factor_grad_diag,
                "act_runtime": act_runtime_diag,
                "runtime_telemetry": {
                    "batch_records": batch_runtime_records,
                    "optimizer_step_records": optimizer_step_runtime_records,
                },
                "gpu_allocated_bytes": (
                    torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                ),
                "gpu_reserved_bytes": (
                    torch.cuda.memory_reserved(device) if device.type == "cuda" else 0
                ),
                "gpu_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                ),
                "gpu_peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
                ),
            })
            write_json_atomic(Path(args.save_path) / "smoke_summary.json", smoke_payload)
            if trajectory_enabled:
                expected_steps = epoch_batch_limit // args.grad_accum_steps
                if epoch_batch_limit % args.grad_accum_steps:
                    expected_steps += 1
                trajectory_ok = bool(
                    not smoke_payload["failed"]
                    and len(trajectory_outputs) == len(trajectory_milestones)
                    and trajectory_outputs[-1]["batch"] == epoch_batch_limit
                    and optimizer_step_count == expected_steps
                )
                final_cumulative = trajectory_outputs[-1]["cumulative"]
                sensitivity = None
                if final_cumulative["informative_fraction"] <= 0.001:
                    sensitivity = teacher_sensitivity_grid(
                        trajectory_records,
                        gain_threshold=args.h6_router_gain_threshold,
                    )
                    write_json_atomic(
                        Path(args.save_path) / "teacher_sensitivity.json",
                        {
                            "mode": "offline_no_optimizer_step",
                            "canonical_trajectory_unchanged": True,
                            "grid": sensitivity,
                        },
                    )
                final_summary = {
                    "status": "PASS" if trajectory_ok else "FAIL",
                    "git_sha": current_git_head(),
                    "seed": args.seed,
                    "batches_executed": epoch_batch_limit,
                    "grad_accum_steps": args.grad_accum_steps,
                    "expected_optimizer_steps": expected_steps,
                    "optimizer_step_count": optimizer_step_count,
                    "runtime_seconds": time.monotonic() - started,
                    "peak_gpu_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                    ),
                    "peak_gpu_reserved_bytes": (
                        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
                    ),
                    "milestones_requested": trajectory_milestones,
                    "milestones_completed": [item["batch"] for item in trajectory_outputs],
                    "final_cumulative": final_cumulative,
                    "final_recent_window": trajectory_outputs[-1]["recent_window"],
                    "final_structure": trajectory_outputs[-1]["structure"],
                    "final_gradients": trajectory_outputs[-1]["gradients"],
                    "final_gradient_attribution": trajectory_outputs[-1]["gradient_attribution"],
                    "rho": diagnostics_to_python(rho_values),
                    "teacher_sensitivity_audit": sensitivity,
                    "canonical_training_configuration_unchanged_by_sensitivity": True,
                    "scientific_interpretation": "PENDING_EVIDENCE_REVIEW",
                }
                write_trajectory_artifacts(args.save_path, trajectory_outputs, final_summary)
            if smoke_payload["failed"]:
                raise RuntimeError(
                    f"{model.h6.progress_version} bounded smoke failed: {smoke_payload['failed']}"
                )

        (logger.info if not is_v7 else (lambda *args, **kwargs: None))(
            "%s progress_version=%s epoch=%d total=%s task=%s cls=%s seg=%s center=%s router_teacher=%s "
            "router_teacher_weighted=%s vae_rec=%s vae_kl_raw=%s vae_kl_effective=%s beta_vae_kl=%s "
            "kg=%s orth=%s balance=%s alpha=%s sparse_ratio=%s routing_mode=%s gamma_state=%s gamma_class=%s rho=%s lr=%s "
            "gate_enabled=%s gate_state=%s gate_soft_warnings=%s gate_query_collapse_count=%s "
            "gate_key_failure_count=%s gate_factor_collapse_count=%s gate_semantic_drift_count=%s "
            "gate_sparse_collapse_count=%s gate_hard_failure=%s gate_abort_reason=%s "
            "gate_query_failed_levels=%s gate_sparse_failed_levels=%s "
            "router_query_mode=%s query_global_weight=%s local_bypass_ratio_mean=%s local_bypass_ratio_max=%s "
            "raw_query_cos_mean=%s local_query_cos_mean=%s final_query_cos_mean=%s "
            "final_query_effective_rank=%s final_query_top1_energy_ratio=%s "
            "raw_concept_key_cos_mean=%s raw_concept_key_cos_max=%s final_router_key_cos_mean=%s "
            "final_router_key_cos_max=%s final_router_key_l2_min=%s router_key_adaptation_ratio_mean=%s "
            "router_key_adaptation_ratio_max=%s factor_context_anchor_cos_mean=%s factor_context_anchor_cos_max=%s "
            "factor_identity_tangent_base_abs_cos_mean=%s factor_identity_tangent_base_abs_cos_max=%s "
            "factor_identity_tangent_pair_cos_mean=%s factor_identity_tangent_pair_cos_max=%s "
            "factor_identity_tangent_l2_min=%s context_angle_change_degrees_mean=%s context_angle_change_degrees_max=%s "
            "dynamic_mean_hard_cos=%s dynamic_mean_anchor_loss_raw=%s dynamic_mean_anchor_loss_weighted=%s "
            "dynamic_mean_anchor_weight=%s "
            "dense_usage=%s sparse_usage=%s topk_freq=%s dense_entropy=%s sparse_entropy=%s dense_dead=%s sparse_dead=%s "
            "unique_topk_pairs=%s max_dense_usage=%s max_sparse_usage=%s router_logit_std=%s router_prob_std=%s "
            "query_variance=%s concept_key_cos_mean=%s concept_key_cos_max=%s mu_mean=%s mu_std=%s decoded_mu_std=%s "
            "class_semantic_std=%s logvar_min=%s logvar_max=%s center_distance=%s dynamic_hard_cos=%s load_bias=%s ema_topk_usage=%s "
            "slot_init_scale=%s slot_initial_cos_mean=%s slot_initial_cos_max=%s slot_initial_l2_min=%s "
            "concept_key_diversity_raw=%s concept_key_diversity_weighted=%s concept_key_l2_min=%s "
            "dynamic_residual_cos_mean=%s dynamic_residual_cos_max=%s dynamic_residual_l2_min=%s dynamic_residual_grad_norms=%s "
            "teacher_entropy=%s teacher_max_prob=%s teacher_prob_std=%s teacher_usage=%s teacher_unique_topk_pairs=%s teacher_router_kl=%s "
            "prototype_cos_mean=%s prototype_cos_max=%s prototype_l2_min=%s prototype_variance=%s "
            "factor_grad_norms=%s factor_grad_cos_mean=%s factor_grad_cos_max=%s factor_grad_l2_min=%s "
            "level_input_checks=%s level_query_difference=%s level_logit_difference=%s "
            "teacher_scheduled_weight=%s teacher_effective_weight=%s teacher_confidence_gate=%s teacher_informative=%s "
            "teacher_entropy_threshold=%s teacher_prob_std_threshold=%s "
            "teacher_raw_similarity_mean_per_factor=%s teacher_raw_similarity_std_across_factors=%s "
            "teacher_raw_similarity_std_across_patches=%s teacher_raw_similarity_min=%s teacher_raw_similarity_max=%s "
            "teacher_raw_logit_range=%s normal_patch_count=%s abnormal_patch_count=%s "
            "teacher_raw_candidate_entropy=%s teacher_raw_candidate_max_prob=%s teacher_raw_candidate_prob_std=%s "
            "teacher_raw_candidate_usage=%s teacher_raw_candidate_unique_topk_pairs=%s "
            "teacher_centered_candidate_entropy=%s teacher_centered_candidate_max_prob=%s teacher_centered_candidate_prob_std=%s "
            "teacher_centered_candidate_usage=%s teacher_centered_candidate_unique_topk_pairs=%s "
            "teacher_distance_candidate_entropy=%s teacher_distance_candidate_max_prob=%s teacher_distance_candidate_prob_std=%s "
            "teacher_distance_candidate_usage=%s teacher_distance_candidate_unique_topk_pairs=%s "
            "router_patch_count=%s router_softmax_dim=%s router_topk_dim=%s "
            "query_pairwise_cos_mean_across_patches=%s query_pairwise_cos_max_across_patches=%s "
            "query_variance_across_patches=%s query_effective_rank=%s query_singular_value_ratio=%s "
            "per_factor_logit_std_across_patches=%s "
            "stage_concept_slots_cos_mean=%s stage_concept_slots_cos_max=%s stage_concept_slots_l2_min=%s "
            "stage_concept_keys_cos_mean=%s stage_concept_keys_cos_max=%s stage_concept_keys_l2_min=%s "
            "stage_prototype_normal_cos_mean=%s stage_prototype_normal_cos_max=%s stage_prototype_normal_l2_min=%s "
            "stage_state_to_context_raw_cos_mean=%s stage_state_to_context_raw_cos_max=%s stage_state_to_context_raw_l2_min=%s "
            "stage_state_to_context_with_identity_cos_mean=%s stage_state_to_context_with_identity_cos_max=%s stage_state_to_context_with_identity_l2_min=%s "
            "stage_state_to_context_norm_cos_mean=%s stage_state_to_context_norm_cos_max=%s stage_state_to_context_norm_l2_min=%s "
            "stage_context_before_encoder_cos_mean=%s stage_context_before_encoder_cos_max=%s stage_context_before_encoder_l2_min=%s "
            "stage_dynamic_text_raw_cos_mean=%s stage_dynamic_text_raw_cos_max=%s stage_dynamic_text_raw_l2_min=%s "
            "stage_dynamic_text_norm_cos_mean=%s stage_dynamic_text_norm_cos_max=%s stage_dynamic_text_norm_l2_min=%s "
            "late_factor_identity_enabled=%s factor_id_scale=%s factor_id_max_ratio=%s "
            "factor_id_residual_norm_mean=%s factor_id_residual_norm_max=%s "
            "factor_id_residual_to_context_ratio_mean=%s factor_id_residual_to_context_ratio_max=%s "
            "factor_id_projection_grad_norm=%s teacher_mode=%s teacher_confidence_gate_enabled=%s "
            "teacher_informative_patch_fraction=%s teacher_informative_patch_count=%s teacher_valid_patch_count=%s "
            "teacher_active_levels=%s teacher_gate_reason=%s",
            "phase4_p1_debug" if is_v7 else "phase4_p1_v6", model.h6.progress_version, epoch, *(float(np.mean(metrics[key])) for key in (
                "total", "task", "cls", "seg", "center", "router_teacher", "router_teacher_weighted",
                "vae_rec", "vae_kl_raw", "vae_kl_effective"
            )),
            beta_vae_kl,
            *(float(np.mean(metrics[key])) for key in ("kg", "orth", "balance")),
            hybrid_alpha, sparse_ratio, routing_mode,
            float(h6_batch["gamma_state"].detach().item()), float(h6_batch["gamma_class"].detach().item()),
            h6_batch["rho"].detach().float().cpu().tolist(), optimizer.param_groups[0]["lr"],
            structural_gate_config.enabled,
            gate_decision.state_label,
            gate_decision.soft_warnings,
            structural_gate.counters["query_collapse"],
            structural_gate.counters["key_anchor_failure"],
            structural_gate.counters["factor_collapse"],
            structural_gate.counters["semantic_drift"],
            structural_gate.counters["sparse_collapse"],
            gate_decision.hard_failure,
            gate_decision.abort_reason,
            gate_decision.per_level.get("query_failed_levels", []),
            gate_decision.per_level.get("sparse_failed_levels", []),
            args.h6_router_query_mode,
            args.h6_router_query_global_weight,
            _diag_float("local_bypass_to_learned_ratio_mean"),
            _diag_float("local_bypass_to_learned_ratio_max"),
            diagnostics["raw_query_pairwise_cos_mean"].cpu().tolist(),
            diagnostics["local_query_pairwise_cos_mean"].cpu().tolist(),
            diagnostics["final_query_pairwise_cos_mean"].cpu().tolist(),
            diagnostics["final_query_effective_rank"].cpu().tolist(),
            diagnostics["final_query_top1_energy_ratio"].cpu().tolist(),
            _diag_float("raw_concept_key_cos_mean"),
            _diag_float("raw_concept_key_cos_max"),
            _diag_float("final_router_key_cos_mean"),
            _diag_float("final_router_key_cos_max"),
            _diag_float("final_router_key_l2_min"),
            _diag_float("router_key_adaptation_ratio_mean"),
            _diag_float("router_key_adaptation_ratio_max"),
            _diag_float("factor_context_anchor_cos_mean"),
            _diag_float("factor_context_anchor_cos_max"),
            _diag_float("factor_identity_tangent_base_abs_cos_mean"),
            _diag_float("factor_identity_tangent_base_abs_cos_max"),
            _diag_float("factor_identity_tangent_pair_cos_mean"),
            _diag_float("factor_identity_tangent_pair_cos_max"),
            _diag_float("factor_identity_tangent_l2_min"),
            _diag_float("context_angle_change_degrees_mean"),
            _diag_float("context_angle_change_degrees_max"),
            diagnostics["dynamic_mean_hard_cos"].cpu().tolist(),
            _diag_float("dynamic_mean_anchor_loss_raw"),
            float(np.mean(metrics["dynamic_mean_anchor_weighted"])),
            float(np.mean(metrics["dynamic_mean_anchor_weight"])),
            diagnostics["dense_factor_usage"].cpu().tolist(), diagnostics["sparse_factor_usage"].cpu().tolist(),
            diagnostics["selected_topk_frequency"].cpu().tolist(),
            diagnostics["dense_normalized_entropy"].cpu().tolist(), diagnostics["sparse_normalized_entropy"].cpu().tolist(),
            (diagnostics["dense_factor_usage"] < 0.01).sum(dim=-1).cpu().tolist(), diagnostics["sparse_factor_usage"].lt(0.01).sum(dim=-1).cpu().tolist(),
            diagnostics["unique_topk_pairs"].cpu().tolist(),
            diagnostics["dense_factor_usage"].max(dim=-1).values.cpu().tolist(), diagnostics["sparse_factor_usage"].max(dim=-1).values.cpu().tolist(),
            diagnostics["router_logit_std"].cpu().tolist(), diagnostics["router_prob_std"].cpu().tolist(),
            diagnostics["query_variance"].cpu().tolist(), float(diagnostics["concept_key_cos_mean"].cpu().item()),
            float(diagnostics["concept_key_cos_max"].cpu().item()),
            float(h6_batch["mu"].detach().float().mean().item()), mu_std,
            float(h6_batch["decoded_mu"].detach().float().std(unbiased=False).item()),
            float(h6_batch["class_semantic"].detach().float().std(unbiased=False).item()),
            float(h6_batch["logvar"].detach().float().min().item()), float(h6_batch["logvar"].detach().float().max().item()),
            float(center_distance.detach().item()), float(dynamic_hard_cosine.detach().item()),
            diagnostics["load_bias"].cpu().tolist(), diagnostics["ema_topk_usage"].cpu().tolist(),
            args.h6_slot_init_scale,
            float(diagnostics["slot_initial_cos_mean"].cpu().item()),
            float(diagnostics["slot_initial_cos_max"].cpu().item()),
            float(diagnostics["slot_initial_l2_min"].cpu().item()),
            float(np.mean(metrics["concept_key_diversity_raw"])),
            float(np.mean(metrics["concept_key_diversity_weighted"])),
            float(diagnostics["concept_key_l2_min"].cpu().item()),
            float(diagnostics["dynamic_residual_cos_mean"].cpu().item()),
            float(diagnostics["dynamic_residual_cos_max"].cpu().item()),
            float(diagnostics["dynamic_residual_l2_min"].cpu().item()),
            None if factor_grad_diag["dynamic_residual_grad_norms"] is None else factor_grad_diag["dynamic_residual_grad_norms"].cpu().tolist(),
            teacher_diag.get("teacher_entropy", torch.empty(0, device=device)).detach().cpu().tolist()
            if torch.is_tensor(teacher_diag.get("teacher_entropy", None)) else None,
            teacher_diag.get("teacher_max_probability", torch.empty(0, device=device)).detach().cpu().tolist()
            if torch.is_tensor(teacher_diag.get("teacher_max_probability", None)) else None,
            teacher_diag.get("teacher_probability_std_across_patches", torch.empty(0, device=device)).detach().cpu().tolist()
            if torch.is_tensor(teacher_diag.get("teacher_probability_std_across_patches", None)) else None,
            teacher_diag.get("teacher_usage", torch.empty(0, device=device)).detach().cpu().tolist()
            if torch.is_tensor(teacher_diag.get("teacher_usage", None)) else None,
            teacher_diag.get("teacher_unique_topk_pairs", torch.empty(0, device=device)).detach().cpu().tolist()
            if torch.is_tensor(teacher_diag.get("teacher_unique_topk_pairs", None)) else None,
            teacher_diag.get("teacher_router_kl", torch.empty(0, device=device)).detach().cpu().tolist()
            if torch.is_tensor(teacher_diag.get("teacher_router_kl", None)) else None,
            float(proto_diag["prototype_cos_mean"].cpu().item()),
            float(proto_diag["prototype_cos_max"].cpu().item()),
            float(proto_diag["prototype_l2_min"].cpu().item()),
            float(proto_diag["prototype_variance"].cpu().item()),
            None if factor_grad_diag["factor_grad_norms"] is None else factor_grad_diag["factor_grad_norms"].cpu().tolist(),
            None if factor_grad_diag["factor_grad_cos_mean"] is None else float(factor_grad_diag["factor_grad_cos_mean"].cpu().item()),
            None if factor_grad_diag["factor_grad_cos_max"] is None else float(factor_grad_diag["factor_grad_cos_max"].cpu().item()),
            None if factor_grad_diag["factor_grad_l2_min"] is None else float(factor_grad_diag["factor_grad_l2_min"].cpu().item()),
            {"alias": bool(diagnostics["level_input_alias"].cpu().item()), "input_diff": float(diagnostics["level_input_difference"].cpu().item())},
            float(diagnostics["level_query_difference"].cpu().item()),
            float(diagnostics["level_logit_difference"].cpu().item()),
            float(np.mean(metrics["router_teacher_scheduled_weight"])),
            float(np.mean(metrics["router_teacher_effective_weight"])),
            bool(args.h6_teacher_confidence_gate),
            bool(teacher_informative),
            args.h6_teacher_entropy_threshold,
            args.h6_teacher_prob_std_threshold,
            _teacher_value("teacher_raw_similarity_mean_per_factor"),
            _teacher_value("teacher_raw_similarity_std_across_factors"),
            _teacher_value("teacher_raw_similarity_std_across_patches"),
            _teacher_value("teacher_raw_similarity_min"),
            _teacher_value("teacher_raw_similarity_max"),
            _teacher_value("teacher_raw_logit_range"),
            _teacher_value("normal_patch_count"),
            _teacher_value("abnormal_patch_count"),
            _teacher_value("teacher_raw_candidate_entropy"),
            _teacher_value("teacher_raw_candidate_max_probability"),
            _teacher_value("teacher_raw_candidate_probability_std_across_patches"),
            _teacher_value("teacher_raw_candidate_usage"),
            _teacher_value("teacher_raw_candidate_unique_topk_pairs"),
            _teacher_value("teacher_centered_candidate_entropy"),
            _teacher_value("teacher_centered_candidate_max_probability"),
            _teacher_value("teacher_centered_candidate_probability_std_across_patches"),
            _teacher_value("teacher_centered_candidate_usage"),
            _teacher_value("teacher_centered_candidate_unique_topk_pairs"),
            _teacher_value("teacher_distance_candidate_entropy"),
            _teacher_value("teacher_distance_candidate_max_probability"),
            _teacher_value("teacher_distance_candidate_probability_std_across_patches"),
            _teacher_value("teacher_distance_candidate_usage"),
            _teacher_value("teacher_distance_candidate_unique_topk_pairs"),
            int(diagnostics["router_patch_count"].detach().cpu().item()),
            int(diagnostics["router_softmax_dim"].detach().cpu().item()),
            int(diagnostics["router_topk_dim"].detach().cpu().item()),
            diagnostics["query_pairwise_cos_mean_across_patches"].cpu().tolist(),
            diagnostics["query_pairwise_cos_max_across_patches"].cpu().tolist(),
            diagnostics["query_variance_across_patches"].cpu().tolist(),
            diagnostics["query_effective_rank"].cpu().tolist(),
            diagnostics["query_singular_value_ratio"].cpu().tolist(),
            diagnostics["per_factor_logit_std_across_patches"].cpu().tolist(),
            _diag_float("stage_concept_slots_cos_mean"), _diag_float("stage_concept_slots_cos_max"), _diag_float("stage_concept_slots_l2_min"),
            _diag_float("stage_concept_keys_cos_mean"), _diag_float("stage_concept_keys_cos_max"), _diag_float("stage_concept_keys_l2_min"),
            _diag_float("stage_prototype_normal_cos_mean"), _diag_float("stage_prototype_normal_cos_max"), _diag_float("stage_prototype_normal_l2_min"),
            _diag_float("stage_state_to_context_raw_cos_mean"), _diag_float("stage_state_to_context_raw_cos_max"), _diag_float("stage_state_to_context_raw_l2_min"),
            _diag_float("stage_state_to_context_with_identity_cos_mean"), _diag_float("stage_state_to_context_with_identity_cos_max"), _diag_float("stage_state_to_context_with_identity_l2_min"),
            _diag_float("stage_state_to_context_norm_cos_mean"), _diag_float("stage_state_to_context_norm_cos_max"), _diag_float("stage_state_to_context_norm_l2_min"),
            _diag_float("stage_context_before_encoder_cos_mean"), _diag_float("stage_context_before_encoder_cos_max"), _diag_float("stage_context_before_encoder_l2_min"),
            _diag_float("stage_dynamic_text_raw_cos_mean"), _diag_float("stage_dynamic_text_raw_cos_max"), _diag_float("stage_dynamic_text_raw_l2_min"),
            _diag_float("stage_dynamic_text_norm_cos_mean"), _diag_float("stage_dynamic_text_norm_cos_max"), _diag_float("stage_dynamic_text_norm_l2_min"),
            bool(diagnostics["late_factor_identity_enabled"].detach().cpu().item()),
            float(diagnostics["factor_id_scale"].detach().float().cpu().item()),
            float(diagnostics["factor_id_max_ratio"].detach().float().cpu().item()),
            _diag_float("factor_id_residual_norm_mean"),
            _diag_float("factor_id_residual_norm_max"),
            _diag_float("factor_id_residual_to_context_ratio_mean"),
            _diag_float("factor_id_residual_to_context_ratio_max"),
            None if factor_grad_diag["factor_id_projection_grad_norm"] is None else float(factor_grad_diag["factor_id_projection_grad_norm"].cpu().item()),
            args.h6_router_teacher_mode,
            bool(args.h6_teacher_confidence_gate),
            _teacher_value("teacher_informative_patch_fraction"),
            _teacher_value("teacher_informative_patch_count"),
            _teacher_value("teacher_valid_patch_count"),
            _teacher_value("teacher_active_levels"),
            _teacher_value("teacher_gate_reason"),
        )
        payload = build_phase4_checkpoint(
            model,
            epoch=epoch,
            seed=args.seed,
            precision=args.precision,
            phase2b_config=_phase2b_config_from_args(args),
            loss_weights={**loss_weights, "vae_kl_current": beta_vae_kl},
            structural_gate_config=structural_gate_config.to_dict(),
            structural_gate_state=structural_gate.state_dict(),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        latest_checkpoint_path = os.path.join(args.save_path, f"adapter_{epoch}.pth")
        torch.save(payload, latest_checkpoint_path)
        saved_checkpoints.append(latest_checkpoint_path)
        if gate_decision.hard_failure:
            gated_abort_artifacts(
                save_path=args.save_path,
                epoch=epoch,
                decision=gate_decision,
                gate_state=structural_gate,
                gate_config=structural_gate_config,
                metrics=epoch_metric_means,
                diagnostics=diagnostics,
                teacher_state=teacher_diag,
                sparse_ratio=sparse_ratio,
                routing_mode=routing_mode,
                alpha=hybrid_alpha,
                trust_region_weight=dynamic_mean_anchor_weight,
                latest_checkpoint_path=latest_checkpoint_path,
                args=args,
                payload=payload,
            )
            logger.error(
                "h6_structural_gate_abort epoch=%d reason=%s diagnostic=%s",
                epoch,
                gate_decision.abort_reason,
                os.path.join(args.save_path, f"gated_abort_epoch_{epoch}.json"),
            )
            sys.exit(42)
        sparse_dead = diagnostics["sparse_factor_usage"].lt(0.01).sum(dim=-1)
        unique_pairs = diagnostics["unique_topk_pairs"]
        sparse_failure = router_specialization_failed(
            sparse_ratio,
            sparse_dead,
            unique_pairs,
            args.h6_router_max_sparse_dead_factors,
            args.h6_router_min_unique_topk_pairs,
        )
        if structural_gate_config.enabled:
            sparse_failure = False
        router_failure_streak = router_failure_streak + 1 if sparse_failure else 0
        if router_failure_streak >= int(args.h6_router_failure_patience):
            diagnostic_path = os.path.join(args.save_path, f"h6_router_specialization_failed_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "reason": "h6_router_specialization_failed",
                "sparse_dead_factors": sparse_dead.detach().cpu(),
                "unique_topk_pairs": unique_pairs.detach().cpu(),
                "router_diagnostics": diagnostics_to_python(diagnostics),
                "checkpoint": payload,
            }, diagnostic_path)
            logger.error(
                "h6_router_specialization_failed epoch=%d sparse_dead=%s unique_topk_pairs=%s diagnostic=%s",
                epoch, sparse_dead.cpu().tolist(), unique_pairs.cpu().tolist(), diagnostic_path,
            )
            raise RuntimeError(f"h6_router_specialization_failed diagnostic={diagnostic_path}")
    if structural_gate_config.enabled:
        final_checkpoint = saved_checkpoints[-1] if saved_checkpoints else None
        write_json_atomic(
            os.path.join(args.save_path, "GATED_TRAIN_COMPLETED.json"),
            {
                "final_epoch": int(args.epoch),
                "final_checkpoint": final_checkpoint,
                "gate_counters": dict(structural_gate.counters),
                "hard_gate_fired": False,
                "checkpoint_list": list(saved_checkpoints),
                "configuration_fingerprint": {
                    "git_head": current_git_head(),
                    "gate": structural_gate_config.to_dict(),
                    "args": vars(args),
                },
            },
        )
    return model


def main():
    parser = argparse.ArgumentParser(description="End To End Training.")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="clip model to use (default: ViT-L-14-336)",
    )
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--dataset", type=str, default="VisA")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epoch", type=int, default=20, help="epochs for training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp32")
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    parser.add_argument("--cuda_device", type=int, default=0, help="cuda device id")

    parser.add_argument("--save_path", type=str, default="ckpt/test")

    # settings
    parser.add_argument("--n_groups", type=int, default=4, help="number of groups for adapter")
    parser.add_argument("--image_adapt_weight", type=float, default=0.2)
    parser.add_argument("--conv_lora_rank", type=int, default=8, help="rank for LoRA adapters")
    parser.add_argument("--conv_lora_alpha", type=float, default=2.0, help="alpha for LoRA adapters")
    parser.add_argument(
        "--conv_kernel_size_list", type=int, nargs="+", default=[3, 5],
        help="kernel size for convolutional LoRA adapters"
    )

    parser.add_argument("--text_adapt_weight", type=float, default=0.2)
    parser.add_argument("--lora_rank", type=int, default=16, help="rank for LoRA adapters")
    parser.add_argument("--lora_alpha", type=float, default=2.0, help="alpha for LoRA adapters")

    parser.add_argument("--image_lr", type=float, default=0.001, help="learning rate for image adapter")
    parser.add_argument("--text_lr", type=float, default=0.0005, help="learning rate for text adapter")
    parser.add_argument("--use_soft_prompt", action="store_true", help="enable Phase2B KgCoOp-style soft prompt")
    parser.add_argument("--use_hybrid_soft_prompt", action="store_true", help="enable Phase2B hard-soft hybrid prompt")
    parser.add_argument("--hybrid_alpha_max", type=float, default=0.2)
    parser.add_argument("--soft_prompt_freeze_epochs", type=int, default=3)
    parser.add_argument("--soft_prompt_ctx_len", type=int, default=4)
    parser.add_argument("--soft_prompt_lr", type=float, default=1e-4)
    parser.add_argument("--soft_prompt_init", type=str, choices=["phrase", "random"], default="phrase")
    parser.add_argument("--soft_prompt_init_phrase", type=str, default="a photo of a")
    parser.add_argument("--lambda_kg", type=float, default=1e-3)
    parser.add_argument("--lambda_k", type=float, default=0.0, help="hybrid K-space regularization weight")
    parser.add_argument("--h6_progress", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--h6_smoke_max_batches", type=int, default=0,
        help="Development-only cap per epoch for Phase4 runtime smoke checks; 0 means the full loader.",
    )
    parser.add_argument(
        "--h6_smoke_forward_only", action="store_true",
        help="Development-only one-batch forward probe; records diagnostics and performs no backward/step.",
    )
    parser.add_argument(
        "--h6_smoke_backward_only", action="store_true",
        help="Development-only one-batch backward probe; records gradients and performs no optimizer step.",
    )
    parser.add_argument(
        "--h6_wiring_probe_batches", type=int, nargs="*", default=[],
        help="Optional one-based batch indices for compact factor-specialization wiring probes.",
    )
    parser.add_argument(
        "--h6_trajectory_milestones", type=int, nargs="*", default=[],
        help="Opt-in diagnostics-only cumulative/window milestone reporting.",
    )
    parser.add_argument(
        "--h6_drift_diagnostics", action=argparse.BooleanOptionalAction, default=False,
        help="One-batch bank snapshots and autograd attribution for a diagnostic smoke only.",
    )
    parser.add_argument("--h6_num_factors", type=int, default=4)
    parser.add_argument("--h6_top_k", type=int, default=2)
    parser.add_argument("--h6_bank_dim", type=int, default=256)
    parser.add_argument("--h6_router_dim", type=int, default=128)
    parser.add_argument("--h6_router_temperature", type=float, default=1.0)
    parser.add_argument("--h6_router_soft_epochs", type=int, default=2)
    parser.add_argument("--h6_dense_routing_epochs", type=int, default=None)
    parser.add_argument("--h6_sparse_start_epoch", type=int, default=None)
    parser.add_argument("--h6_sparse_transition_epochs", type=int, default=1)
    parser.add_argument("--lambda_h6_router_teacher", type=float, default=0.0)
    parser.add_argument("--h6_router_teacher_temperature", type=float, default=0.15)
    parser.add_argument("--h6_router_teacher_start_epoch", type=int, default=3)
    parser.add_argument("--h6_router_teacher_warmup_epochs", type=int, default=3)
    parser.add_argument(
        "--h6_router_teacher_mode",
        choices=["raw_cosine", "state_centered_cosine", "negative_squared_distance"],
        default="raw_cosine",
    )
    parser.add_argument("--h6_teacher_confidence_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--h6_teacher_entropy_threshold", type=float, default=0.98)
    parser.add_argument("--h6_global_text_mode", type=str, choices=["hard_anchor", "phase2b_hybrid", "dynamic_legacy"], default="hard_anchor")
    parser.add_argument("--h6_prediction_routing", type=str, choices=["dense", "scheduled_topk", "readiness_topk"], default="dense")
    parser.add_argument("--h6_diagnostics_mode", type=str, choices=["none", "light", "full"], default="light")
    parser.add_argument("--h6_diagnostics_interval", type=int, default=1)
    parser.add_argument("--h6_load_bias_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_load_bias_momentum", type=float, default=0.9)
    parser.add_argument("--h6_load_bias_step", type=float, default=0.001)
    parser.add_argument("--h6_load_bias_max", type=float, default=0.03)
    parser.add_argument("--h6_router_failure_patience", type=int, default=2)
    parser.add_argument("--h6_router_max_sparse_dead_factors", type=int, default=1)
    parser.add_argument("--h6_router_min_unique_topk_pairs", type=int, default=2)
    parser.add_argument("--h6_expert_dead_usage_threshold", type=float, default=0.01)
    parser.add_argument("--h6_structural_gate_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_structural_gate_mode", choices=["abort", "monitor", "off"], default="abort")
    parser.add_argument("--h6_structural_gate_patience", type=int, default=2)
    parser.add_argument("--h6_structural_gate_dense_start_epoch", type=int, default=8)
    parser.add_argument("--h6_structural_gate_require_all_levels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--h6_structural_gate_reset_state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_gate_query_rank_max", type=float, default=1.10)
    parser.add_argument("--h6_gate_query_top1_energy_min", type=float, default=0.995)
    parser.add_argument("--h6_gate_query_cosine_min", type=float, default=0.9999)
    parser.add_argument("--h6_gate_logit_std_max", type=float, default=1e-6)
    parser.add_argument("--h6_gate_key_cosine_max", type=float, default=0.95)
    parser.add_argument("--h6_gate_key_l2_min", type=float, default=0.05)
    parser.add_argument("--h6_gate_dynamic_cosine_min", type=float, default=0.999)
    parser.add_argument("--h6_gate_dynamic_orth_center", type=float, default=0.75)
    parser.add_argument("--h6_gate_dynamic_orth_tolerance", type=float, default=0.005)
    parser.add_argument("--h6_gate_hard_anchor_cosine_min", type=float, default=0.30)
    parser.add_argument("--h6_gate_sparse_min_ratio", type=float, default=0.50)
    parser.add_argument("--h6_gate_max_sparse_dead_factors", type=int, default=1)
    parser.add_argument("--h6_gate_min_unique_topk_pairs", type=int, default=2)
    parser.add_argument("--h6_vae_hidden_dim", type=int, default=512)
    parser.add_argument("--h6_vae_latent_dim", type=int, default=256)
    parser.add_argument("--h6_vae_class_ratio", type=float, default=0.25)
    parser.add_argument("--h6_slot_init_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_slot_init_scale", type=float, default=0.02)
    parser.add_argument("--h6_slot_init_seed_offset", type=int, default=6100)
    parser.add_argument("--h6_factor_grad_diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_late_factor_identity_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_factor_id_scale", type=float, default=0.02)
    parser.add_argument("--h6_factor_id_max_ratio", type=float, default=0.05)
    parser.add_argument(
        "--h6_factor_generator_specialization_enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Tier-1 factor IDs and lightweight factor-specific residual heads.",
    )
    parser.add_argument("--h6_factor_head_init_scale", type=float, default=1e-3)
    parser.add_argument(
        "--h6_factor_local_dynamic_mix", type=float, default=0.0,
        help="Opt-in local CoPS blend of dynamic factors; global hard-anchor text remains unchanged.",
    )
    parser.add_argument(
        "--h6_router_query_mode",
        choices=["raw", "local_residual", "local_global_bypass"],
        default="local_global_bypass",
    )
    parser.add_argument("--h6_router_query_global_weight", type=float, default=0.10)
    parser.add_argument("--h6_router_local_bypass_scale", type=float, default=0.10)
    parser.add_argument("--h6_router_local_bypass_max_ratio", type=float, default=0.20)
    parser.add_argument("--h6_router_local_projection_seed_offset", type=int, default=7200)
    parser.add_argument("--h6_router_key_anchor_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--h6_router_key_anchor_seed_offset", type=int, default=7300)
    parser.add_argument("--h6_router_key_adaptation_initial_ratio", type=float, default=0.10)
    parser.add_argument("--h6_router_key_adaptation_max_ratio", type=float, default=0.25)
    parser.add_argument("--h6_factor_context_anchor_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--h6_factor_context_anchor_seed_offset", type=int, default=7400)
    parser.add_argument("--h6_factor_context_adaptation_initial_ratio", type=float, default=0.10)
    parser.add_argument("--h6_factor_context_adaptation_max_ratio", type=float, default=0.25)
    parser.add_argument("--h6_factor_identity_tangent_projection_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lambda_h6_dynamic_mean_anchor", type=float, default=0.001)
    parser.add_argument("--h6_dynamic_mean_anchor_min_cosine", type=float, default=0.70)
    parser.add_argument("--h6_dynamic_mean_anchor_start_epoch", type=int, default=4)
    parser.add_argument("--h6_dynamic_mean_anchor_warmup_epochs", type=int, default=3)
    parser.add_argument("--h6_expert_bottleneck", type=int, default=64)
    parser.add_argument(
        "--h6_progress_version",
        choices=["P1-v6", "P1-v7-full", "P1-v8-minimal", "P1-v8.3", "P1-v8.4-A"],
        default="P1-v8.3",
    )
    parser.add_argument("--h6_local_factor_mode", type=str, choices=["legacy_mix", "center_spread"], default="center_spread")
    parser.add_argument("--h6_local_center_mix", type=float, default=0.05)
    parser.add_argument("--h6_local_factor_spread", type=float, default=0.10)
    parser.add_argument("--h6_expert_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_expert_fofs_seed_offset", type=int, default=7500)
    parser.add_argument("--h6_expert_state_condition_scale", type=float, default=0.25)
    parser.add_argument("--h6_expert_scale_target", type=float, default=0.10)
    parser.add_argument("--h6_expert_scale_start_epoch", type=int, default=1)
    parser.add_argument("--h6_expert_scale_warmup_epochs", type=int, default=6)
    parser.add_argument("--h6_expert_max_relative_ratio", type=float, default=0.10)
    parser.add_argument("--lambda_h6_expert", type=float, default=0.0)
    parser.add_argument("--h6_expert_start_epoch", type=int, default=1)
    parser.add_argument("--h6_expert_warmup_epochs", type=int, default=3)
    parser.add_argument("--lambda_h6_advantage", type=float, default=0.0)
    parser.add_argument("--h6_advantage_start_epoch", type=int, default=4)
    parser.add_argument("--h6_advantage_warmup_epochs", type=int, default=3)
    parser.add_argument("--h6_advantage_margin", type=float, default=0.05)
    parser.add_argument("--lambda_h6_etf", type=float, default=0.0)
    parser.add_argument("--h6_etf_start_epoch", type=int, default=3)
    parser.add_argument("--h6_etf_warmup_epochs", type=int, default=4)
    parser.add_argument("--lambda_h6_balance_final", type=float, default=None)
    parser.add_argument("--h6_balance_decay_epochs", type=int, default=4)
    parser.add_argument("--lambda_h6_expert_anchor", type=float, default=0.0)
    parser.add_argument("--h6_expert_anchor_min_cosine", type=float, default=0.70)
    parser.add_argument("--lambda_h6_expert_radius", type=float, default=0.0)
    parser.add_argument("--lambda_h6_center", type=float, default=0.0)
    parser.add_argument("--h6_center_factor_aware", action="store_true")
    parser.add_argument("--h6_center_detach_assignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--h6_center_margin", type=float, default=0.0)
    parser.add_argument("--lambda_h6_vae_rec", type=float, default=0.05)
    parser.add_argument("--beta_h6_vae_kl", type=float, default=1e-4)
    parser.add_argument("--h6_kl_zero_epochs", type=int, default=0)
    parser.add_argument("--h6_kl_warmup_epochs", type=int, default=4)
    parser.add_argument("--h6_kl_free_bits", type=float, default=0.0)
    parser.add_argument("--lambda_h6_orth", type=float, default=0.0)
    parser.add_argument("--lambda_h6_delta_div", type=float, default=0.0)
    parser.add_argument("--lambda_h6_func_div", type=float, default=0.0)
    parser.add_argument("--lambda_h6_route", type=float, default=0.0)
    parser.add_argument("--lambda_h6_factor_role", type=float, default=0.0)
    parser.add_argument("--lambda_h6_actual_local", type=float, default=0.0)
    parser.add_argument("--lambda_h6_factor", type=float, default=0.10)
    parser.add_argument("--lambda_h6_router", type=float, default=0.10)
    parser.add_argument(
        "--lambda_h6_act", type=float, default=0.0,
        help="P1-v8.4-A ACT utility weight; choose by no-step gradient calibration.",
    )
    parser.add_argument(
        "--h6_act_effective_beta", type=float, default=0.999,
        help="Inverse effective-number normal/anomaly weighting for ACT support.",
    )
    parser.add_argument(
        "--h6_utility_factor_effective_beta", type=float, default=None,
        help="Use inverse-effective-number patch weighting for P1-v8.3 factor utility.",
    )
    parser.add_argument(
        "--h6_router_support_normalized", action=argparse.BooleanOptionalAction, default=False,
        help="Divide masked router CE by all valid patch support.",
    )
    parser.add_argument(
        "--h6_pcgrad_main_factor", action=argparse.BooleanOptionalAction, default=False,
        help="Apply two-objective PCGrad to accumulated main/factor shared-semantic gradients.",
    )
    parser.add_argument(
        "--h6_primary_anchored_factor_surgery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Project only the conflicting auxiliary factor component while preserving "
            "the accumulated primary shared-semantic gradient exactly."
        ),
    )
    parser.add_argument(
        "--h6_collect_router_gradient_geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Diagnostics only: collect isolated router shared-gradient geometry.",
    )
    parser.add_argument("--h6_utility_denominator_floor", type=float, default=0.10)
    parser.add_argument("--h6_tau_utility", type=float, default=0.05)
    parser.add_argument("--h6_utility_gain_threshold", type=float, default=0.02)
    parser.add_argument("--h6_factor_tau_utility", type=float, default=None)
    parser.add_argument("--h6_router_tau_utility", type=float, default=None)
    parser.add_argument("--h6_router_gain_threshold", type=float, default=None)
    parser.add_argument("--h6_act_gain_threshold", type=float, default=None)
    parser.add_argument("--h6_utility_entropy_threshold", type=float, default=0.98)
    parser.add_argument("--h6_exploration_start", type=float, default=0.15)
    parser.add_argument("--h6_exploration_end", type=float, default=0.05)
    parser.add_argument(
        "--h6_exploration_total_epochs",
        type=int,
        default=None,
        help=(
            "Canonical horizon for the utility exploration schedule. "
            "Diagnostic one-epoch runners must set this to the intended training horizon."
        ),
    )
    parser.add_argument("--h6_cluster_responsibility", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_cluster_centroid_path", type=str, default=None)
    parser.add_argument("--h6_cluster_temperature", type=float, default=0.10)
    parser.add_argument("--h6_lambda_cluster_resp", type=float, default=0.0)
    parser.add_argument("--h6_cluster_tie_factor_ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_cluster_tie_router_keys", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda_h6_balance", type=float, default=0.0)
    parser.add_argument("--lambda_h6_concept_key_diversity", type=float, default=0.0)
    parser.add_argument("--h6_concept_key_cosine_margin", type=float, default=0.5)
    parser.add_argument("--h6_concept_key_diversity_start_epoch", type=int, default=1)
    parser.add_argument("--h6_concept_key_diversity_warmup_epochs", type=int, default=3)
    parser.add_argument("--lambda_h6_visual_residual", type=float, default=0.01, help="reserved for Progress 2")
    parser.add_argument("--lambda_h6_consistency", type=float, default=0.01, help="reserved for Progress 3")
    parser.add_argument("--h6_two_view", action="store_true", help="reserved for Progress 3")
    parser.add_argument("--lr_gamma", type=float, default=0.9, help="learning rate decay factor")
    parser.add_argument(
        "--dfg_mode",
        type=str,
        choices=["mlp", "attn"],
        default="mlp",
        help="DFG fusion mode: original MLP gate or Phase 1A dual-softmax attention",
    )
    parser.add_argument("--dfg_attn_dim", type=int, default=256, help="attention dimension for Phase 1A DFG")
    parser.add_argument("--dfg_attn_tau", type=float, default=4.0, help="fixed attention temperature for Phase 1A DFG")
    parser.add_argument("--use_ss2d_dfg", action="store_true", help="enable Phase 1B SS2D residual query branch")
    parser.add_argument("--dfg_gamma_max", type=float, default=0.2, help="max abs SS2D residual scale for Phase 1B")
    parser.add_argument(
        "--dfg_ss2d_fusion",
        type=str,
        choices=["feature_residual", "weight_residual"],
        default="feature_residual",
        help="SS2D DFG fusion mode: feature residual query shift or post-softmax weight residual",
    )
    parser.add_argument("--dfg_beta", type=float, default=0.10, help="fixed beta for weight_residual SS2D DFG")
    parser.add_argument(
        "--dfg_beta_schedule",
        type=str,
        choices=["fixed", "warmup010"],
        default="fixed",
        help="beta schedule for weight_residual SS2D DFG",
    )
    parser.add_argument("--dfg_beta_target", type=float, default=0.10, help="target beta for beta schedules")
    parser.add_argument(
        "--non_finite_loss_abort_threshold",
        type=int,
        default=20,
        help="Abort an epoch when non-finite loss skips exceed this value. Use -1 to disable.",
    )
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="clip trainable adapter gradients; <=0 disables")
    parser.add_argument("--amp", action="store_true", help="enable Automatic Mixed Precision training")
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="enable activation checkpointing to reduce ViT memory usage",
    )
    parser.add_argument("--num_workers", type=int, default=4 if os.name != "nt" else 0)
    parser.add_argument("--pin_memory", action="store_true", default=False)

    parser.add_argument("--h6_teacher_prob_std_threshold", type=float, default=0.0)

    args = parser.parse_args()
    args.h6_factor_tau_utility = (
        args.h6_tau_utility
        if args.h6_factor_tau_utility is None
        else args.h6_factor_tau_utility
    )
    args.h6_router_tau_utility = (
        args.h6_tau_utility
        if args.h6_router_tau_utility is None
        else args.h6_router_tau_utility
    )
    args.h6_router_gain_threshold = (
        args.h6_utility_gain_threshold
        if args.h6_router_gain_threshold is None
        else args.h6_router_gain_threshold
    )
    args.h6_act_gain_threshold = resolve_act_gain_threshold(
        args.h6_progress_version,
        args.h6_act_gain_threshold,
        args.h6_utility_gain_threshold,
    )
    configure_canonical_fp32()
    if args.h6_trajectory_milestones:
        if args.h6_trajectory_milestones != sorted(set(args.h6_trajectory_milestones)):
            raise ValueError("--h6_trajectory_milestones must be unique and increasing")
        if args.h6_smoke_max_batches <= 0:
            raise ValueError("trajectory milestones require --h6_smoke_max_batches")
        if args.h6_trajectory_milestones[-1] != args.h6_smoke_max_batches:
            raise ValueError("final trajectory milestone must equal --h6_smoke_max_batches")
        if args.epoch != 1 or args.h6_progress_version not in {"P1-v8.3", "P1-v8.4-A"}:
            raise ValueError("trajectory diagnostics are restricted to one-epoch P1-v8.3/v8.4-A probes")
        if args.h6_smoke_forward_only or args.h6_smoke_backward_only:
            raise ValueError("trajectory diagnostics require normal optimizer execution")
    if args.h6_dense_routing_epochs is not None:
        args.h6_router_soft_epochs = int(args.h6_dense_routing_epochs)
    if args.h6_sparse_start_epoch is not None:
        expected_sparse_start = int(args.h6_router_soft_epochs) + 1
        if int(args.h6_sparse_start_epoch) != expected_sparse_start:
            raise ValueError(
                "--h6_sparse_start_epoch must equal --h6_dense_routing_epochs + 1 "
                f"(or --h6_router_soft_epochs + 1); got {args.h6_sparse_start_epoch} vs {expected_sparse_start}"
            )
    if args.h6_router_soft_epochs < 0:
        raise ValueError("--h6_router_soft_epochs/--h6_dense_routing_epochs must be >= 0")
    if args.h6_sparse_transition_epochs < 1:
        raise ValueError("--h6_sparse_transition_epochs must be >= 1")
    if args.h6_router_teacher_temperature <= 0:
        raise ValueError("--h6_router_teacher_temperature must be > 0")
    if not 0 <= args.h6_teacher_entropy_threshold <= 1:
        raise ValueError("--h6_teacher_entropy_threshold must be in [0, 1]")
    if args.h6_teacher_prob_std_threshold < 0:
        raise ValueError("--h6_teacher_prob_std_threshold must be >= 0")
    if not 0 <= args.h6_load_bias_momentum < 1:
        raise ValueError("--h6_load_bias_momentum must be in [0, 1)")
    if args.h6_load_bias_step < 0 or args.h6_load_bias_max < 0:
        raise ValueError("--h6_load_bias_step/max must be >= 0")
    if args.h6_structural_gate_patience < 1:
        raise ValueError("--h6_structural_gate_patience must be >= 1")
    if not 0 <= args.h6_expert_dead_usage_threshold <= 1:
        raise ValueError("--h6_expert_dead_usage_threshold must be in [0, 1]")
    if args.h6_structural_gate_dense_start_epoch < 1:
        raise ValueError("--h6_structural_gate_dense_start_epoch must be >= 1")
    if args.h6_gate_query_rank_max < 1.0:
        raise ValueError("--h6_gate_query_rank_max must be >= 1")
    if not 0 <= args.h6_gate_query_top1_energy_min <= 1:
        raise ValueError("--h6_gate_query_top1_energy_min must be in [0, 1]")
    if not -1 <= args.h6_gate_query_cosine_min <= 1:
        raise ValueError("--h6_gate_query_cosine_min must be in [-1, 1]")
    if args.h6_gate_logit_std_max < 0:
        raise ValueError("--h6_gate_logit_std_max must be >= 0")
    if not -1 <= args.h6_gate_key_cosine_max <= 1:
        raise ValueError("--h6_gate_key_cosine_max must be in [-1, 1]")
    if args.h6_gate_key_l2_min < 0:
        raise ValueError("--h6_gate_key_l2_min must be >= 0")
    if not -1 <= args.h6_gate_dynamic_cosine_min <= 1:
        raise ValueError("--h6_gate_dynamic_cosine_min must be in [-1, 1]")
    if args.h6_gate_dynamic_orth_tolerance < 0:
        raise ValueError("--h6_gate_dynamic_orth_tolerance must be >= 0")
    if not -1 <= args.h6_gate_hard_anchor_cosine_min <= 1:
        raise ValueError("--h6_gate_hard_anchor_cosine_min must be in [-1, 1]")
    if not 0 <= args.h6_gate_sparse_min_ratio <= 1:
        raise ValueError("--h6_gate_sparse_min_ratio must be in [0, 1]")
    if not 0 <= args.h6_vae_class_ratio <= 1:
        raise ValueError("--h6_vae_class_ratio must be in [0, 1]")
    if args.h6_slot_init_scale < 0:
        raise ValueError("--h6_slot_init_scale must be >= 0")
    if args.h6_factor_id_scale < 0 or args.h6_factor_id_max_ratio < 0:
        raise ValueError("--h6_factor_id_scale/max_ratio must be >= 0")
    if args.h6_router_query_global_weight < 0:
        raise ValueError("--h6_router_query_global_weight must be >= 0")
    if args.h6_router_local_bypass_scale < 0 or args.h6_router_local_bypass_max_ratio < 0:
        raise ValueError("--h6_router_local_bypass_scale/max_ratio must be >= 0")
    if args.h6_router_key_adaptation_initial_ratio < 0 or args.h6_router_key_adaptation_max_ratio < 0:
        raise ValueError("--h6_router_key_adaptation_initial_ratio/max_ratio must be >= 0")
    if args.h6_factor_context_adaptation_initial_ratio < 0 or args.h6_factor_context_adaptation_max_ratio < 0:
        raise ValueError("--h6_factor_context_adaptation_initial_ratio/max_ratio must be >= 0")
    if args.lambda_h6_dynamic_mean_anchor < 0:
        raise ValueError("--lambda_h6_dynamic_mean_anchor must be >= 0")
    if not -1 <= args.h6_dynamic_mean_anchor_min_cosine <= 1:
        raise ValueError("--h6_dynamic_mean_anchor_min_cosine must be in [-1, 1]")
    if args.h6_dynamic_mean_anchor_start_epoch < 1:
        raise ValueError("--h6_dynamic_mean_anchor_start_epoch must be >= 1")
    if args.h6_dynamic_mean_anchor_warmup_epochs < 1:
        raise ValueError("--h6_dynamic_mean_anchor_warmup_epochs must be >= 1")
    if not -1 <= args.h6_concept_key_cosine_margin <= 1:
        raise ValueError("--h6_concept_key_cosine_margin must be in [-1, 1]")
    if args.h6_concept_key_diversity_warmup_epochs < 1:
        raise ValueError("--h6_concept_key_diversity_warmup_epochs must be >= 1")
    if args.h6_kl_zero_epochs < 0:
        raise ValueError("--h6_kl_zero_epochs must be >= 0")
    if args.h6_kl_warmup_epochs < 1:
        raise ValueError("--h6_kl_warmup_epochs must be >= 1")
    if args.h6_kl_free_bits < 0:
        raise ValueError("--h6_kl_free_bits must be >= 0")
    if args.h6_center_margin < 0:
        raise ValueError("--h6_center_margin must be >= 0")
    if args.use_soft_prompt and args.use_hybrid_soft_prompt:
        raise ValueError("--use_soft_prompt and --use_hybrid_soft_prompt are mutually exclusive")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if args.h6_progress == 1:
        if args.h6_num_factors != 4 or args.h6_top_k != 2:
            raise ValueError("Phase4 Progress 1 is locked to --h6_num_factors 4 and --h6_top_k 2")
        if args.h6_progress_version == "P1-v7-full" and not args.h6_expert_enabled:
            raise ValueError("P1-v7-full requires --h6_expert_enabled")
        if args.h6_progress_version != "P1-v7-full" and args.h6_expert_enabled:
            raise ValueError("paired experts are explicit P1-v7-full only")
        if args.h6_progress_version in {"P1-v8.3", "P1-v8.4-A"}:
            contract_version = args.h6_progress_version
            if args.precision != "fp32" or args.amp:
                raise ValueError(f"{contract_version} requires --precision fp32 with AMP disabled")
            if args.h6_local_factor_mode != "center_spread":
                raise ValueError(f"{contract_version} requires --h6_local_factor_mode center_spread")
            if abs(args.h6_local_center_mix - 0.05) > 1e-12 or abs(args.h6_local_factor_spread - 0.10) > 1e-12:
                raise ValueError(f"{contract_version} requires center/spread geometry 0.05/0.10")
            if args.img_size != 518 or not args.grad_checkpointing:
                raise ValueError(f"{contract_version} requires --img_size 518 --grad_checkpointing")
            if args.h6_smoke_forward_only and args.h6_smoke_max_batches != 1:
                raise ValueError("--h6_smoke_forward_only requires --h6_smoke_max_batches 1")
            if args.h6_smoke_backward_only and args.h6_smoke_max_batches != 1:
                raise ValueError("--h6_smoke_backward_only requires --h6_smoke_max_batches 1")
            if args.h6_smoke_forward_only and args.h6_smoke_backward_only:
                raise ValueError("forward-only and backward-only probes are mutually exclusive")
            if args.h6_global_text_mode != "phase2b_hybrid":
                raise ValueError(f"{contract_version} requires --h6_global_text_mode phase2b_hybrid")
            if not (
                args.dfg_mode == "attn"
                and args.dfg_attn_dim == 256
                and abs(args.dfg_attn_tau - 8.0) <= 1e-12
                and args.use_ss2d_dfg
                and args.dfg_ss2d_fusion == "weight_residual"
                and abs(args.dfg_beta - 0.10) <= 1e-12
                and args.dfg_beta_schedule == "warmup010"
                and abs(args.dfg_beta_target - 0.10) <= 1e-12
            ):
                raise ValueError(f"{contract_version} requires the canonical Phase2B-style DFG+SS2D base path")
            if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
                raise ValueError(f"{contract_version} requires TF32 disabled")
            if args.batch_size != 1 or args.grad_accum_steps != 6:
                raise ValueError(f"{contract_version} requires --batch_size 1 --grad_accum_steps 6")
            if args.h6_prediction_routing != "dense" or args.h6_load_bias_enabled:
                raise ValueError(f"{contract_version} requires dense routing with load bias disabled")
            if args.h6_cluster_responsibility or args.h6_expert_enabled:
                raise ValueError(f"{contract_version} disables cluster responsibility and paired experts")
            legacy_weights = {
                "balance": args.lambda_h6_balance, "center": args.lambda_h6_center,
                "orth": args.lambda_h6_orth, "route": args.lambda_h6_route,
                "factor_role": args.lambda_h6_factor_role, "actual_local": args.lambda_h6_actual_local,
                "functional_diversity": args.lambda_h6_func_div,
                "router_teacher": args.lambda_h6_router_teacher,
            }
            enabled_legacy = {name: value for name, value in legacy_weights.items() if float(value) != 0.0}
            if enabled_legacy:
                raise ValueError(f"{contract_version} legacy auxiliary losses must be OFF: {enabled_legacy}")
            if (
                args.h6_factor_tau_utility <= 0
                or args.h6_router_tau_utility <= 0
                or args.h6_utility_denominator_floor <= 0
            ):
                raise ValueError(f"{contract_version} utility temperature and floor must be positive")
            if not 0.0 < args.h6_act_effective_beta < 1.0:
                raise ValueError("--h6_act_effective_beta must be in (0, 1)")
            if args.lambda_h6_act < 0.0:
                raise ValueError("--lambda_h6_act must be non-negative")
            if contract_version == "P1-v8.3" and args.lambda_h6_act != 0.0:
                raise ValueError("P1-v8.3 path requires --lambda_h6_act 0")
            if contract_version == "P1-v8.4-A" and args.lambda_h6_act <= 0.0:
                raise ValueError("P1-v8.4-A requires calibrated --lambda_h6_act > 0")
            if args.h6_utility_factor_effective_beta is not None and not (
                0.0 < args.h6_utility_factor_effective_beta < 1.0
            ):
                raise ValueError("--h6_utility_factor_effective_beta must be in (0, 1)")
            if args.h6_pcgrad_main_factor and args.h6_primary_anchored_factor_surgery:
                raise ValueError(
                    "symmetric PCGrad and primary-anchored factor surgery are mutually exclusive"
                )
            if (
                args.h6_pcgrad_main_factor or args.h6_primary_anchored_factor_surgery
            ) and args.h6_utility_factor_effective_beta is None:
                raise ValueError(
                    "P1-v8.3 gradient surgery requires the audited effective-number factor loss"
                )
            if (
                args.h6_collect_router_gradient_geometry
                and not (
                    args.h6_pcgrad_main_factor
                    or args.h6_primary_anchored_factor_surgery
                )
            ):
                raise ValueError(
                    "router gradient geometry requires an enabled main/factor surgery mode"
                )
            if (
                args.h6_exploration_total_epochs is not None
                and args.h6_exploration_total_epochs < args.epoch
            ):
                raise ValueError(
                    "--h6_exploration_total_epochs must be at least --epoch"
                )
        if args.h6_two_view:
            raise ValueError("--h6_two_view belongs to Progress 3 and is not implemented in Progress 1")
        if torch.cuda.is_available() and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support BF16. Use --precision fp16 or --precision fp32.")

        if (args.h6_progress == 1) and args.lambda_h6_actual_local > 0.0:
            assert not args.h6_load_bias_enabled, "Candidate 1 must disable load bias"
            assert args.lambda_h6_balance == 0.0, "Candidate 1 must disable balance"
            assert not getattr(args, "h6_cluster_responsibility_enabled", False), "Candidate 1 must disable cluster responsibility"
            assert getattr(args, "lambda_h6_functional_decorrelation", 0.0) == 0.0, "Candidate 1 must disable functional decorrelation"
            assert getattr(args, "lambda_h6_router_teacher", 0.0) == 0.0, "Candidate 1 must disable router teacher"
            assert getattr(args, "lambda_h6_center", 0.0) == 0.0, "Candidate 1 must disable center losses"
            assert getattr(args, "lambda_h6_dynamic_mean_anchor", 0.0) == 0.0, "Candidate 1 must disable center losses"
            assert not getattr(args, "h6_expert_enabled", False), "Candidate 1 must disable experts"
            assert getattr(args, "h6_prediction_routing", "dense") == "dense", "Candidate 1 must disable Top-K prediction"
            assert not getattr(args, "h6_rho_trainable", False), "Candidate 1 must disable rho training"

        set_phase4_seed(args.seed)
    # ========================================================
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)

    logger = logging.getLogger(__name__)
    log_name = "run.log" if args.h6_trajectory_milestones else "train.log"
    logging.basicConfig(
        filename=os.path.join(args.save_path, log_name),
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(filename)s %(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("args: %s", vars(args))
    log_data_root(logger)
    log_preflight(logger)
    run_config_path = Path(args.save_path) / "config.json"
    if not run_config_path.exists():
        write_json_atomic(run_config_path, vars(args))
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    if args.grad_checkpointing:
        clip_model.set_grad_checkpointing(True)
    clip_model.eval()
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=args.n_groups,
        image_adapt_weight=args.image_adapt_weight,
        conv_lora_rank=args.conv_lora_rank,
        conv_lora_alpha=args.conv_lora_alpha,
        conv_kernel_size_list=args.conv_kernel_size_list,
        text_adapt_weight=args.text_adapt_weight,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        dfg_mode=args.dfg_mode,
        dfg_attn_dim=args.dfg_attn_dim,
        dfg_attn_tau=args.dfg_attn_tau,
        use_ss2d_dfg=args.use_ss2d_dfg,
        dfg_gamma_max=args.dfg_gamma_max,
        dfg_ss2d_fusion=args.dfg_ss2d_fusion,
        dfg_beta=args.dfg_beta,
        dfg_beta_schedule=args.dfg_beta_schedule,
        dfg_beta_target=args.dfg_beta_target,
        dfg_beta_current=args.dfg_beta,
        use_soft_prompt=args.use_soft_prompt,
        soft_prompt_ctx_len=args.soft_prompt_ctx_len,
        soft_prompt_init=args.soft_prompt_init,
        soft_prompt_init_phrase=args.soft_prompt_init_phrase,
        h6_progress=args.h6_progress,
        h6_num_factors=args.h6_num_factors,
        h6_top_k=args.h6_top_k,
        h6_bank_dim=args.h6_bank_dim,
        h6_router_dim=args.h6_router_dim,
        h6_router_temperature=args.h6_router_temperature,
        h6_router_soft_epochs=args.h6_router_soft_epochs,
        h6_sparse_transition_epochs=args.h6_sparse_transition_epochs,
        h6_load_bias_enabled=args.h6_load_bias_enabled,
        h6_load_bias_momentum=args.h6_load_bias_momentum,
        h6_load_bias_step=args.h6_load_bias_step,
        h6_load_bias_max=args.h6_load_bias_max,
        h6_vae_hidden_dim=args.h6_vae_hidden_dim,
        h6_vae_latent_dim=args.h6_vae_latent_dim,
        h6_vae_class_ratio=args.h6_vae_class_ratio,
        h6_slot_init_enabled=args.h6_slot_init_enabled,
        h6_slot_init_scale=args.h6_slot_init_scale,
        h6_slot_init_seed_offset=args.h6_slot_init_seed_offset,
        h6_factor_grad_diagnostics=args.h6_factor_grad_diagnostics,
        h6_late_factor_identity_enabled=args.h6_late_factor_identity_enabled,
        h6_factor_id_scale=args.h6_factor_id_scale,
        h6_factor_id_max_ratio=args.h6_factor_id_max_ratio,
        h6_factor_generator_specialization_enabled=args.h6_factor_generator_specialization_enabled,
        h6_factor_head_init_scale=args.h6_factor_head_init_scale,
        h6_factor_local_dynamic_mix=args.h6_factor_local_dynamic_mix,
        h6_cluster_responsibility=args.h6_cluster_responsibility,
        h6_cluster_temperature=args.h6_cluster_temperature,
        h6_router_query_mode=args.h6_router_query_mode,
        h6_router_query_global_weight=args.h6_router_query_global_weight,
        h6_router_local_bypass_scale=args.h6_router_local_bypass_scale,
        h6_router_local_bypass_max_ratio=args.h6_router_local_bypass_max_ratio,
        h6_router_local_projection_seed_offset=args.h6_router_local_projection_seed_offset,
        h6_router_key_anchor_enabled=args.h6_router_key_anchor_enabled,
        h6_router_key_anchor_seed_offset=args.h6_router_key_anchor_seed_offset,
        h6_router_key_adaptation_initial_ratio=args.h6_router_key_adaptation_initial_ratio,
        h6_router_key_adaptation_max_ratio=args.h6_router_key_adaptation_max_ratio,
        h6_factor_context_anchor_enabled=args.h6_factor_context_anchor_enabled,
        h6_factor_context_anchor_seed_offset=args.h6_factor_context_anchor_seed_offset,
        h6_factor_context_adaptation_initial_ratio=args.h6_factor_context_adaptation_initial_ratio,
        h6_factor_context_adaptation_max_ratio=args.h6_factor_context_adaptation_max_ratio,
        h6_factor_identity_tangent_projection_enabled=args.h6_factor_identity_tangent_projection_enabled,
        lambda_h6_dynamic_mean_anchor=args.lambda_h6_dynamic_mean_anchor,
        h6_dynamic_mean_anchor_min_cosine=args.h6_dynamic_mean_anchor_min_cosine,
        h6_dynamic_mean_anchor_start_epoch=args.h6_dynamic_mean_anchor_start_epoch,
        h6_dynamic_mean_anchor_warmup_epochs=args.h6_dynamic_mean_anchor_warmup_epochs,
        h6_router_teacher_mode=args.h6_router_teacher_mode,
        h6_progress_version=args.h6_progress_version,
        h6_local_factor_mode=args.h6_local_factor_mode,
        h6_local_center_mix=args.h6_local_center_mix,
        h6_local_factor_spread=args.h6_local_factor_spread,
        h6_expert_enabled=args.h6_expert_enabled,
        h6_expert_bottleneck=args.h6_expert_bottleneck,
        h6_expert_fofs_seed_offset=args.h6_expert_fofs_seed_offset,
        h6_expert_state_condition_scale=args.h6_expert_state_condition_scale,
        h6_expert_scale_target=args.h6_expert_scale_target,
        h6_expert_scale_start_epoch=args.h6_expert_scale_start_epoch,
        h6_expert_scale_warmup_epochs=args.h6_expert_scale_warmup_epochs,
        h6_expert_max_relative_ratio=args.h6_expert_max_relative_ratio,
        h6_prediction_routing=args.h6_prediction_routing,
        diagnostics_mode=args.h6_diagnostics_mode,
        diagnostics_interval=args.h6_diagnostics_interval,
    ).to(device)
    model.eval()
    model.use_hybrid_soft_prompt = bool(args.use_hybrid_soft_prompt)
    model.prompt_mode = "hybrid" if args.use_hybrid_soft_prompt else ("soft" if args.use_soft_prompt else "hard")
    model.hybrid_alpha_current = 0.0
    model.hybrid_alpha_max = args.hybrid_alpha_max
    model.soft_prompt_freeze_epochs = args.soft_prompt_freeze_epochs

    if args.h6_progress == 1:
        model.prompt_mode = "h6_dynamic"
        model.use_soft_prompt = False
        model.use_hybrid_soft_prompt = True
        model.requires_grad_(False)
        model.image_adapter.requires_grad_(True)
        model.text_adapter.requires_grad_(True)
        model.soft_prompt.requires_grad_(False)
        model.h6.requires_grad_(True)
        model.h6.rho.raw.requires_grad_(False)
        if args.h6_cluster_responsibility:
            if args.h6_lambda_cluster_resp <= 0.0:
                raise ValueError("--h6_cluster_responsibility requires --h6_lambda_cluster_resp > 0")
            if not args.h6_cluster_centroid_path:
                raise ValueError("--h6_cluster_responsibility requires --h6_cluster_centroid_path")
            centroid_path = Path(args.h6_cluster_centroid_path)
            if not centroid_path.is_file():
                raise FileNotFoundError(f"Tier-3 centroid file does not exist: {centroid_path}")
            args.h6_cluster_centroid_sha256 = hashlib.sha256(centroid_path.read_bytes()).hexdigest()
            cluster_metadata = model.h6.load_cluster_centroids(str(centroid_path))
            logger.info(
                "tier3_cluster_bound path=%s sha256=%s metadata=%s",
                centroid_path, args.h6_cluster_centroid_sha256, cluster_metadata,
            )
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        logger.info("phase4_progress=1 trainable parameters=%s frozen parameters=%s", f"{trainable_params:,}", f"{frozen_params:,}")
        optimizer = torch.optim.Adam(_h6_optimizer_groups(model, args))
        lr_scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_gamma)
        model.h6_global_text_mode = args.h6_global_text_mode
        if args.h6_progress_version in {"P1-v8.3", "P1-v8.4-A"}:
            optimizer_parameter_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            h6_config = model.h6.config_dict()
            contract_checks = {
                "openai_only_initialization": True,
                "no_phase2b_checkpoint_load": True,
                "fp32": args.precision == "fp32",
                "tf32_off": not (
                    torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
                ),
                "amp_off": not args.amp,
                "gradient_checkpointing_on": args.grad_checkpointing,
                "batch1_accum6": args.batch_size == 1 and args.grad_accum_steps == 6,
                "img_size_518": args.img_size == 518,
                "phase2b_hybrid_global": args.h6_global_text_mode == "phase2b_hybrid",
                "global_hybrid_soft_prompt_on": bool(model.use_hybrid_soft_prompt),
                "dfg_ss2d_base": (
                    args.dfg_mode == "attn" and args.use_ss2d_dfg
                    and args.dfg_ss2d_fusion == "weight_residual"
                ),
                "center_spread_005_010": (
                    model.h6.local_factor_mode == "center_spread"
                    and abs(model.h6.local_center_mix - 0.05) <= 1e-12
                    and abs(model.h6.local_factor_spread - 0.10) <= 1e-12
                ),
                "four_factors": model.h6.num_factors == 4,
                "structured_state_class": (
                    h6_config["structured_text_enabled"]
                    and h6_config["state_token_factor_specific"]
                    and h6_config["class_token_deterministic_decoder_mu"]
                ),
                "dynamic_text_lora": h6_config["dynamic_text_adapt_text"],
                "dense_routing": model.h6.router.prediction_routing == "dense",
                "rho_fixed_005": bool(torch.equal(
                    model.h6.rho_values().detach().cpu(),
                    torch.full((model.n_groups,), 0.05),
                )),
                "rho_no_grad": not model.h6.rho.raw.requires_grad,
                "rho_absent_optimizer": id(model.h6.rho.raw) not in optimizer_parameter_ids,
                "experts_off": model.h6.paired_experts is None,
                "load_bias_off": not model.h6.load_bias_enabled,
                "balance_off": args.lambda_h6_balance == 0.0,
                "topk_prediction_off": model.h6.router.prediction_routing == "dense",
                "cluster_responsibility_off": not model.h6.cluster_responsibility_enabled,
                "functional_diversity_off": args.lambda_h6_func_div == 0.0,
                "act_contract": (
                    model.h6.act_head is not None
                    and h6_config["act_enabled"] is True
                    and h6_config["local_correction_semantics"] == "act_times_routed_true_residual"
                    and abs(args.h6_act_effective_beta - 0.999) <= 1e-12
                    and args.lambda_h6_act > 0.0
                    if args.h6_progress_version == "P1-v8.4-A"
                    else model.h6.act_head is None and h6_config["act_enabled"] is False
                ),
                "text_lora_trainable": any(p.requires_grad for p in model.text_adapter.parameters()),
                "effective_number_factor_beta_0999": (
                    args.h6_utility_factor_effective_beta is not None
                    and abs(args.h6_utility_factor_effective_beta - 0.999) <= 1e-12
                ),
                "router_support_normalized": bool(args.h6_router_support_normalized),
                "primary_anchored_factor_surgery": bool(
                    args.h6_primary_anchored_factor_surgery
                ),
                "pcgrad_main_factor_off": not bool(args.h6_pcgrad_main_factor),
            }
            failed_contracts = [name for name, passed in contract_checks.items() if not passed]
            model_preflight = {
                "status": "PASS" if not failed_contracts else "FAIL",
                "git_sha": current_git_head(),
                "device": str(device),
                "checks": contract_checks,
                "failed": failed_contracts,
                "h6_config": h6_config,
                "dataloader": {
                    "seed": args.seed,
                    "num_workers": args.num_workers,
                    "pin_memory": args.pin_memory,
                    "persistent_workers": False,
                    "prefetch_factor": 2 if args.num_workers > 0 else None,
                },
            }
            write_json_atomic(Path(args.save_path) / "model_preflight.json", model_preflight)
            if failed_contracts:
                raise RuntimeError(f"{args.h6_progress_version} model preflight failed: {failed_contracts}")
        dataset = get_text_and_image_dataset(args.dataset, args.img_size, "train")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            worker_init_fn=seed_worker,
            generator=make_dataloader_generator(args.seed),
        )
        logger.info("phase4_progress=1 training from OpenAI CLIP only; no Phase2B checkpoint will be loaded")
        train_h6_progress1(model, args.dataset, dataloader, optimizer, lr_scheduler, device, args, logger)
        return

    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    if args.use_hybrid_soft_prompt:
        model.text_adapter.requires_grad_(True)
        model.soft_prompt.requires_grad_(False)
    elif args.use_soft_prompt:
        model.soft_prompt.requires_grad_(True)
        model.text_adapter.requires_grad_(False)
    else:
        model.text_adapter.requires_grad_(True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info("trainable parameters: %s", f"{trainable_params:,}")
    logger.info("frozen parameters: %s", f"{frozen_params:,}")
    logger.info("dfg_weight_residual_fp32=%s", model.dfg_weight_residual_fp32)

    # set optimizer
    if args.use_hybrid_soft_prompt:
        optimizer = torch.optim.Adam([
            {
                "name": "text_adapter",
                "params": model.text_adapter.parameters(),
                "lr": args.text_lr,
            },
            {
                "name": "image_adapter",
                "params": model.image_adapter.parameters(),
                "lr": args.image_lr,
            },
            {
                "name": "soft_prompt",
                "params": model.soft_prompt.parameters(),
                "lr": 0.0,
                "constant_lr": args.soft_prompt_lr,
            },
        ])
    elif args.use_soft_prompt:
        optimizer = torch.optim.Adam([
            {
                "name": "image_adapter",
                "params": model.image_adapter.parameters(),
                "lr": args.image_lr,
            },
            {
                "name": "soft_prompt",
                "params": model.soft_prompt.parameters(),
                "lr": args.soft_prompt_lr,
            },
        ])
    else:
        optimizer = torch.optim.Adam([
            {
                "name": "text_adapter",
                "params": model.text_adapter.parameters(),
                "lr": args.text_lr,
            },
            {
                "name": "image_adapter",
                "params": model.image_adapter.parameters(),
                "lr": args.image_lr,
            },
        ])
    lr_scheduler = StepLR(
        optimizer,
        step_size=1,
        gamma=args.lr_gamma,
    )
    # load dataset
    logger.info("loading dataset ...")
    dataset = get_text_and_image_dataset(
        args.dataset,
        args.img_size,
        "train"
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=make_dataloader_generator(args.seed),
    )
    logger.info("training ...")
    model = train(
        model=model,
        dataset_name=args.dataset,
        train_loader=dataloader,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        device=device,
        total_epoch=args.epoch,
        save_path=args.save_path,
        logger=logger,
        use_amp=args.amp,
        dfg_beta_schedule=args.dfg_beta_schedule,
        dfg_beta_target=args.dfg_beta_target,
        dfg_beta=args.dfg_beta,
        non_finite_loss_abort_threshold=args.non_finite_loss_abort_threshold,
        lambda_kg=args.lambda_kg,
        lambda_k=args.lambda_k,
        hybrid_alpha_max=args.hybrid_alpha_max,
        soft_prompt_freeze_epochs=args.soft_prompt_freeze_epochs,
        grad_clip_norm=args.grad_clip_norm,
    )


if __name__ == "__main__":
    main()
