import argparse
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
    get_hybrid_soft_prompt_single_class_text_embedding,
    get_phase2b_global_text_features,
    get_multiple_adapted_single_class_text_embedding,
    get_soft_prompt_single_class_text_embedding,
)
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
        active = bool(float(weight) != 0.0 and torch.is_tensor(loss) and loss.requires_grad)
        entry = {"raw": float(loss.detach().float().item()), "weight": float(weight), "active": active}
        if active:
            gradients = torch.autograd.grad(
                loss * float(weight), unique_parameters, retain_graph=True, allow_unused=True,
            )
            by_id = {id(parameter): gradient for parameter, gradient in zip(unique_parameters, gradients)}
            for group_name, parameters in grouped.items():
                squared = [by_id[id(parameter)].detach().float().pow(2).sum()
                           for parameter in parameters if by_id[id(parameter)] is not None]
                entry[group_name] = float(torch.stack(squared).sum().sqrt().item()) if squared else 0.0
        result[name] = entry
    for parameter in unique_parameters:
        before = before_grads[id(parameter)]
        after = parameter.grad
        if before is None:
            assert after is None, "drift attribution must not populate parameter.grad"
        else:
            assert after is not None and torch.equal(before, after), "drift attribution must not mutate parameter.grad"
    task_shared = result.get("main_task", {}).get("shared_semantic", 0.0)
    denominator = task_shared if task_shared > 1e-12 else None
    ratios = {
        "assigned_to_task_shared_grad_ratio": None if denominator is None else result.get("assigned_expert", {}).get("shared_semantic", 0.0) / denominator,
        "anchor_to_task_shared_grad_ratio": None if denominator is None else result.get("expert_anchor", {}).get("shared_semantic", 0.0) / denominator,
        "center_to_task_shared_grad_ratio": None if denominator is None else result.get("center", {}).get("shared_semantic", 0.0) / denominator,
        "total_aux_to_task_shared_grad_ratio": None if denominator is None else sum(
            entry.get("shared_semantic", 0.0) for key, entry in result.items() if key != "main_task" and entry["active"]
        ) / denominator,
    }
    return {"components": result, "ratios": ratios}


def h6_drift_parameter_groups(model):
    """Non-overlapping parameter groups for one-batch P1-v7 attribution."""
    semantic_state_prefixes = (
        "normal_query", "abnormal_query", "prototype_attention", "normal_state_update",
        "abnormal_state_update", "state_to_context", "class_to_context",
    )
    shared_semantic, prototype_state = [], []
    for name, parameter in model.h6.semantic_core.named_parameters():
        (prototype_state if name.startswith(semantic_state_prefixes) else shared_semantic).append(parameter)
    paired = model.h6.paired_experts
    return {
        "shared_semantic": shared_semantic,
        "dynamic_context": [model.soft_prompt.ctx_normal, model.soft_prompt.ctx_abnormal],
        "prototype_state": prototype_state,
        "text_adapter": list(model.text_adapter.parameters()),
        "expert_B": [] if paired is None else [paired.expert_B],
        "expert_state_projection": [] if paired is None else list(paired.state_projection.parameters()),
        "router_query_key": list(model.h6.router.parameters()),
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
        "factor_grad_l2_min": l2.min().detach(),
    }


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
        metrics = {key: [] for key in (
            "total", "task", "cls", "seg", "center", "router_teacher", "router_teacher_weighted",
            "router_teacher_scheduled_weight", "router_teacher_effective_weight",
            "vae_rec", "vae_kl_raw", "vae_kl_effective", "kg", "orth", "balance",
            "concept_key_diversity_raw", "concept_key_diversity_weighted",
            "dynamic_mean_anchor_raw", "dynamic_mean_anchor_weighted", "dynamic_mean_anchor_weight",
            "expert", "advantage", "etf", "expert_anchor", "expert_radius", "expert_weight",
            "advantage_weight", "etf_weight", "balance_weight", "expert_delta_norm",
            "expert_assigned_raw", "expert_assigned_weight", "expert_assigned_weighted",
            "expert_advantage_raw", "expert_advantage_weight", "expert_advantage_weighted",
            "expert_etf_raw", "expert_etf_weight", "expert_etf_weighted",
            "expert_balance_dense_raw", "expert_balance_prediction_raw", "expert_balance_weight", "expert_balance_weighted",
            "expert_anchor_raw", "expert_anchor_weight", "expert_anchor_weighted",
            "expert_radius_raw", "expert_radius_weight", "expert_radius_weighted",
            "loss_component_sum", "loss_component_residual", "expert_assigned_normal", "expert_assigned_abnormal",
            "selected_expert_loss", "nonselected_expert_loss", "expert_advantage_margin_satisfied_fraction",
            "legacy_pre_expert_orth_raw", "legacy_pre_expert_orth_weight", "legacy_pre_expert_orth_weighted",
            "delta_t_diversity_raw", "delta_t_diversity_weight", "delta_t_diversity_weighted",
            "functional_factor_diversity_raw", "functional_factor_diversity_weight", "functional_factor_diversity_weighted",
            "cluster_resp_raw", "cluster_resp_weight", "cluster_resp_weighted",
            "cluster_target_entropy", "cluster_router_entropy",
        )}
        factor_grad_diag = {
            "factor_grad_norms": None,
            "factor_grad_cos_mean": None,
            "factor_grad_cos_max": None,
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
        optimizer.zero_grad(set_to_none=True)
        # Structural gates must describe the epoch, never just its final batch.
        epoch_diag_sum: dict[str, torch.Tensor] = {}
        epoch_diag_count: dict[str, int] = {}
        epoch_probe_values: dict[str, list[torch.Tensor]] = {}
        drift_snapshots: dict[str, dict] = {}
        drift_gradient_report = None
        epoch_probe_keys = {
            "expert_delta_norm_mean", "expert_delta_valid_fraction", "final_expert_direction_cos_max",
            "expert_patch_logit_std_across_experts", "final_expert_mean_hard_cos_mean", "unique_topk_pairs",
        }
        gradient_samples: list[dict[str, torch.Tensor]] = []
        epoch_batch_limit = min(len(train_loader), int(args.h6_smoke_max_batches)) if args.h6_smoke_max_batches > 0 else len(train_loader)
        progress = tqdm(train_loader, desc=f"[PHASE4-P1][TRAIN][epoch {epoch:02d}/{args.epoch:02d}]")

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
            image = input_data["image"].to(device, non_blocking=args.pin_memory)
            mask = input_data["mask"].to(device, non_blocking=args.pin_memory)
            label = input_data["label"].to(device, non_blocking=args.pin_memory)
            class_names = list(input_data["class_name"])
            with _phase4_autocast(device, args.precision):
                visual_output = model(image, return_phase4_features=True)
                h6_batch = model.h6.build_batch(
                    model, dataset_name, class_names, visual_output, hybrid_alpha=hybrid_alpha
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
                if args.h6_factor_grad_diagnostics and batch_idx == 1:
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
                seg_pred = model.vision_text_fusion_gate_seg(
                    seg_features,
                    text_global,
                    img_size=args.img_size,
                    h6_patch_logits=h6_batch["h6_logits"],
                )
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
                    factor_bank = h6_batch["factor_bank"].float()
                    factor_patch_logits = torch.einsum(
                        "gbpd,gbmd->gbpm", F.normalize(seg_features.float(), dim=-1),
                        factor_bank[..., 1] - factor_bank[..., 0],
                    ) * model.h6.h6_logit_temperature
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
                if model.h6.expert_enabled:
                    h6_expert, expert_terms = assigned_expert_loss(h6_batch["expert_patch_logits"], h6_batch["prediction_probabilities"], mask)
                    h6_advantage = expert_advantage_loss(h6_batch["expert_patch_logits"], h6_batch["topk_indices"], mask, args.h6_advantage_margin)
                    h6_etf, expert_delta_norm = expert_etf_loss(h6_batch["expert_delta_tangent"])
                    final_balance = args.lambda_h6_balance if args.lambda_h6_balance_final is None else args.lambda_h6_balance_final
                    decay = min(1.0, max(0.0, (epoch - args.h6_router_soft_epochs) / max(1, args.h6_balance_decay_epochs)))
                    balance_weight = args.lambda_h6_balance + decay * (final_balance - args.lambda_h6_balance)
                    h6_balance, balance_terms = dual_routing_balance_loss(h6_batch["dense_probabilities"], h6_batch["prediction_probabilities"])
                    h6_expert_anchor, expert_anchor_cos = expert_clip_anchor_loss(h6_batch["active_factor_bank"], h6_batch["hard_frozen"], args.h6_expert_anchor_min_cosine)
                    h6_expert_radius = expert_radius_loss(h6_batch["expert_relative_ratio"], args.h6_expert_max_relative_ratio)
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
                    factor_bank = h6_batch["factor_bank"].float()
                    factor_patch_logits = torch.einsum(
                        "gbpd,gbmd->gbpm",
                        F.normalize(seg_features.float(), dim=-1),
                        factor_bank[..., 1] - factor_bank[..., 0],
                    ) * model.h6.h6_logit_temperature
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
                if args.h6_drift_diagnostics and batch_idx == 1:
                    drift_gradient_report = h6_drift_gradient_attribution(
                        {
                            "main_task": (task_loss, 1.0),
                            "assigned_expert": (h6_expert, expert_weight),
                            "advantage": (h6_advantage, advantage_weight),
                            "expert_anchor": (h6_expert_anchor, args.lambda_h6_expert_anchor),
                            "center": (h6_center, args.lambda_h6_center),
                            "dynamic_mean_anchor": (h6_batch["dynamic_mean_anchor_loss_raw"], dynamic_mean_anchor_weight),
                        },
                        h6_drift_parameter_groups(model),
                    )
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
            scaler.scale(total_loss / args.grad_accum_steps).backward()
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
                prototype_norms = [module_grad_norm(getattr(core, "prototype_attention", None))]
                prototype_norms = [value for value in prototype_norms if value is not None]
                if prototype_norms:
                    factor_grad_diag["prototype_modules_grad_norm"] = torch.stack(prototype_norms).norm()
                factor_grad_diag["router_grad_norm"] = module_grad_norm(model.h6.router)
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
            do_step = batch_idx % args.grad_accum_steps == 0 or batch_idx == epoch_batch_limit
            if do_step:
                scaler.unscale_(optimizer)
                if has_non_finite_grad(optimizer):
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
                optimizer.zero_grad(set_to_none=True)
                if args.h6_drift_diagnostics and "batch_000_after_first_optimizer_step" not in drift_snapshots:
                    with torch.no_grad(), _phase4_autocast(device, args.precision):
                        post_step_visual = model(image, return_phase4_features=True)
                        post_step_batch = model.h6.build_batch(
                            model, dataset_name, class_names, post_step_visual,
                            hybrid_alpha=hybrid_alpha, update_load_bias=False,
                        )
                    _record_drift_snapshot("batch_000_after_first_optimizer_step", post_step_batch)
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
                "kg": h6_batch["kg_loss"], "orth": h6_orth, "balance": h6_balance,
                "concept_key_diversity_raw": h6_concept_key_diversity,
                "concept_key_diversity_weighted": concept_key_diversity_weight * h6_concept_key_diversity,
                "dynamic_mean_anchor_raw": h6_batch["dynamic_mean_anchor_loss_raw"],
                "dynamic_mean_anchor_weighted": dynamic_mean_anchor_weight * h6_batch["dynamic_mean_anchor_loss_raw"],
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
                metrics[key].append(float(value.detach().float().item()))
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
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
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
        "--h6_wiring_probe_batches", type=int, nargs="*", default=[],
        help="Optional one-based batch indices for compact factor-specialization wiring probes.",
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
    parser.add_argument("--h6_progress_version", choices=["P1-v6", "P1-v7-full", "P1-v8-minimal"], default="P1-v6")
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
    parser.add_argument("--lambda_h6_center", type=float, default=0.10)
    parser.add_argument("--h6_center_factor_aware", action="store_true")
    parser.add_argument("--h6_center_detach_assignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--h6_center_margin", type=float, default=0.0)
    parser.add_argument("--lambda_h6_vae_rec", type=float, default=0.05)
    parser.add_argument("--beta_h6_vae_kl", type=float, default=1e-4)
    parser.add_argument("--h6_kl_zero_epochs", type=int, default=0)
    parser.add_argument("--h6_kl_warmup_epochs", type=int, default=4)
    parser.add_argument("--h6_kl_free_bits", type=float, default=0.0)
    parser.add_argument("--lambda_h6_orth", type=float, default=1e-3)
    parser.add_argument("--lambda_h6_delta_div", type=float, default=0.0)
    parser.add_argument("--lambda_h6_func_div", type=float, default=0.0)
    parser.add_argument("--h6_cluster_responsibility", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_cluster_centroid_path", type=str, default=None)
    parser.add_argument("--h6_cluster_temperature", type=float, default=0.10)
    parser.add_argument("--h6_lambda_cluster_resp", type=float, default=0.0)
    parser.add_argument("--h6_cluster_tie_factor_ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_cluster_tie_router_keys", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda_h6_balance", type=float, default=1e-2)
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
        if args.h6_two_view:
            raise ValueError("--h6_two_view belongs to Progress 3 and is not implemented in Progress 1")
        if torch.cuda.is_available() and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("GPU does not support BF16. Use --precision fp16 or --precision fp32.")
        set_phase4_seed(args.seed)
    # ========================================================
    # check save_path and setting logger
    os.makedirs(args.save_path, exist_ok=True)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "train.log"),
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(filename)s %(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("args: %s", vars(args))
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
        dataset = get_text_and_image_dataset(args.dataset, args.img_size, "train")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
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
