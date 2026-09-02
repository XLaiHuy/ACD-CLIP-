import argparse
import hashlib
import json
import logging
import os
from typing import Any

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
    get_multiple_adapted_single_class_text_embedding,
    get_soft_prompt_single_class_text_embedding,
)
from model.adapter import (
    ACDCLIP
)
from model.clip import create_model
from h2_clean.contract import (
    ANCHOR_FAMILY_BUDGET_DEFAULT,
    SafeImageAdapterAnchor,
    aggregate_anchor_family_metrics,
    apply_family_safe_anchor_budget,
    build_full_checkpoint,
    collect_family_gradient_metrics,
    environment_manifest,
    current_git_sha,
    make_dataloader_generator,
    EpochWorkerInit,
    operational_config_from_mapping,
    parent_scientific_config,
    restore_full_checkpoint,
    scientific_config_from_mapping,
    validate_resume_identity,
    seed_everything,
    sha256_file,
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


def tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()

def diagnostics_to_python(diagnostics):
    converted = {}
    for key in sorted(diagnostics):
        value = diagnostics[key]
        if value is None:
            converted[key] = None
        elif torch.is_tensor(value):
            if value.ndim == 0:
                item = value.item()
                converted[key] = item
            else:
                converted[key] = value.tolist()
        else:
            converted[key] = value
    return converted


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
        "tensor_stats": {name: tensor_debug_stats(tensors[name]) for name in sorted(tensors)},
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
        for key in sorted(stats_list[0])
    }


def grad_norm_or_none(param: torch.nn.Parameter):
    if param.grad is None:
        return None
    return float(param.grad.detach().float().norm().item())


def anchor_gradient_audit_ratio(
        loss: torch.Tensor,
        anchor_loss: torch.Tensor,
        anchor_parameters: list[torch.nn.Parameter],
        anchor_lambda: float,
        device: str | torch.device,
) -> float:
    """Measure anchor/task gradient ratio without mutating gradients."""
    anchor_grads = torch.autograd.grad(
        anchor_lambda * anchor_loss,
        anchor_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    task_grads = torch.autograd.grad(
        loss - anchor_lambda * anchor_loss,
        anchor_parameters,
        retain_graph=True,
        allow_unused=True,
    )
    anchor_norm_sq = sum(
        (gradient.detach().float().square().sum() for gradient in anchor_grads if gradient is not None),
        torch.zeros((), device=device),
    )
    task_norm_sq = sum(
        (gradient.detach().float().square().sum() for gradient in task_grads if gradient is not None),
        torch.zeros((), device=device),
    )
    return float((anchor_norm_sq.sqrt() / task_norm_sq.sqrt().clamp_min(1e-12)).item())

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
        anchor: SafeImageAdapterAnchor | None = None,
        anchor_lambda: float = 0.0,
        use_cir_training: bool = False,
        cir_alpha: float = 0.0,
        cir_peer_count: int = 8,
        cir_spatial_radius: int = 3,
        data_generator: torch.Generator | None = None,
        worker_init: EpochWorkerInit | None = None,
        start_epoch: int = 0,
        global_step: int = 0,
        checkpoint_config: dict[str, Any] | None = None,
        parent_checkpoint_config: dict[str, Any] | None = None,
        operational_checkpoint_config: dict[str, Any] | None = None,
        repo: str = ".",
        clip_sha256: str | None = None,
        dataset_manifest_sha256: str | None = None,
        seed: int = 0,
        precision: str = "amp",
        tf32_enabled: bool = False,
        resume_payload: dict[str, Any] | None = None,
        max_batches: int | None = None,
        anchor_grad_audit_interval: int = 0,
        trace_batch_identity: bool = False,
        anchor_gradient_budget: bool = False,
        anchor_family_budget: float = ANCHOR_FAMILY_BUDGET_DEFAULT,
        anchor_family_audit: bool = False,
):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if resume_payload is not None:
        restored_epoch, restored_step = restore_full_checkpoint(
            resume_payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            dataloader_generator=data_generator,
            expected_scientific_config=checkpoint_config,
            expected_parent_config=parent_checkpoint_config,
            expected_total_epoch=total_epoch,
            expected_seed=seed,
            expected_clip_sha256=clip_sha256,
            expected_manifest_sha256=dataset_manifest_sha256,
            expected_git_sha=current_git_sha(repo),
        )
    if not 0.0 <= float(anchor_family_budget) <= 1.0:
        raise ValueError("anchor_family_budget must be in [0, 1]")
    if anchor_gradient_budget and (anchor is None or anchor_lambda <= 0.0):
        raise ValueError(
            "anchor_gradient_budget requires a positive anchor_lambda and Anchor"
        )
    if anchor_lambda > 0.0 and anchor is None:
        raise ValueError("anchor_lambda > 0 requires a SafeImageAdapterAnchor")
    for epoch in range(int(start_epoch), total_epoch):
        epoch_one_based = epoch + 1
        # Historical H2 keeps all modules in eval mode; trainable adapter
        # parameters still receive gradients and optimizer updates.
        model.eval()
        model.image_encoder.eval()
        model.clipmodel.eval()
        if worker_init is not None:
            worker_init.set_epoch(epoch_one_based)
        if data_generator is not None:
            data_generator.manual_seed(int(seed) + 104729 * epoch_one_based)
        use_hybrid_soft_prompt = bool(getattr(model, "use_hybrid_soft_prompt", False))
        use_soft_prompt = bool(getattr(model, "use_soft_prompt", False))
        soft_prompt_frozen = False
        hybrid_alpha_current = 0.0
        anchor_family_metrics_list = []
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
        anchor_loss_list = []
        anchor_gradient_ratio_list = []
        cir_stats_list = []
        non_finite_loss_skips = 0
        non_finite_grad_skips = 0
        tqdm_train_loader = tqdm(train_loader)
        for batch_idx, input_data in enumerate(tqdm_train_loader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device)
            label = input_data["label"].to(device)
            class_names = input_data["class_name"]
            if trace_batch_identity and batch_idx < 5:
                logger.info(
                    "batch_identity epoch=%d batch=%d file_names=%s image_sha256=%s mask_sha256=%s",
                    epoch_one_based, batch_idx, list(input_data["file_name"]),
                    tensor_bytes_sha256(image), tensor_bytes_sha256(mask),
                )
            # get adapted text embedding
            epoch_text_feature_dict = {}
            kg_losses = []
            k_losses = []
            batch_soft_stats = []
            batch_k_stats = []
            for class_name in sorted(set(class_names)):
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
                seg_pred = model.vision_text_fusion_gate_seg(
                    seg_features,
                    epoch_text_features,
                    cir_training=use_cir_training,
                    cir_alpha=cir_alpha,
                    cir_peer_count=cir_peer_count,
                    cir_spatial_radius=cir_spatial_radius,
                )
                seg_loss = calculate_seg_loss(seg_pred, mask)
                loss_main = cls_loss + seg_loss
                anchor_loss = anchor.loss(model.image_adapter) if anchor is not None else torch.zeros((), device=device)
                if anchor_gradient_budget:
                    task_loss = loss_main + lambda_kg * kg_loss + lambda_k * k_loss
                    loss = task_loss + anchor_lambda * anchor_loss
                else:
                    task_loss = None
                    loss = loss_main + lambda_kg * kg_loss + lambda_k * k_loss + anchor_lambda * anchor_loss
            if use_cir_training:
                cir_stats_list.append(dict(getattr(model, "_last_cir_stats", {})))
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
            anchor_loss_list.append(anchor_loss.item())
            if not anchor_gradient_budget and anchor is not None and anchor_lambda > 0.0 and anchor_grad_audit_interval > 0 and batch_idx % int(anchor_grad_audit_interval) == 0:
                anchor_parameters = [
                    parameter
                    for _, parameter in sorted(model.image_adapter.named_parameters())
                    if parameter.requires_grad
                ]
                anchor_gradient_ratio_list.append(
                    anchor_gradient_audit_ratio(
                        loss,
                        anchor_loss,
                        anchor_parameters,
                        anchor_lambda,
                        device,
                    )
                )
            # backward
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(retain_graph=anchor_gradient_budget)
            scaler.unscale_(optimizer)
            if has_non_finite_grad(optimizer):
                logger.warning("non-finite gradient at epoch %d; skipping optimizer step", epoch + 1)
                non_finite_grad_skips += 1
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue
            if anchor_gradient_budget:
                image_named_parameters = [
                    (name, parameter)
                    for name, parameter in sorted(
                        model.image_adapter.named_parameters(),
                        key=lambda item: item[0],
                    )
                    if parameter.requires_grad
                ]
                image_names = [name for name, _ in image_named_parameters]
                image_parameters = [parameter for _, parameter in image_named_parameters]
                task_gradients = torch.autograd.grad(
                    task_loss,
                    image_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                raw_anchor_gradients = torch.autograd.grad(
                    anchor_loss,
                    image_parameters,
                    retain_graph=False,
                    allow_unused=True,
                )
                metrics = apply_family_safe_anchor_budget(
                    model.image_adapter,
                    sorted(model.named_parameters(), key=lambda item: item[0]),
                    task_gradients=dict(zip(image_names, task_gradients)),
                    raw_anchor_gradients=dict(zip(image_names, raw_anchor_gradients)),
                    anchor_lambda=anchor_lambda,
                    rho=anchor_family_budget,
                    total_trainable_parameters=None,
                )
                if anchor_family_audit:
                    anchor_family_metrics_list.append(metrics)
                    logger.info(
                        "anchor_family_step epoch=%d batch=%d metrics=%s",
                        epoch_one_based,
                        batch_idx,
                        json.dumps(metrics, sort_keys=True),
                    )
            elif anchor_family_audit:
                metrics = collect_family_gradient_metrics(
                    model.image_adapter,
                    sorted(model.named_parameters(), key=lambda item: item[0]),
                    anchor_lambda=0.0,
                    rho=anchor_family_budget,
                    total_trainable_parameters=None,
                )
                anchor_family_metrics_list.append(metrics)
                logger.info(
                    "anchor_family_step epoch=%d batch=%d metrics=%s",
                    epoch_one_based,
                    batch_idx,
                    json.dumps(metrics, sort_keys=True),
                )
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
            global_step += 1
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
        if anchor_family_metrics_list:
            logger.info(
                "anchor_family_epoch epoch=%d metrics=%s",
                epoch_one_based,
                json.dumps(aggregate_anchor_family_metrics(anchor_family_metrics_list), sort_keys=True),
            )
        logger.info(
            "anchor_state epoch=%d enabled=%s lambda=%s mean_anchor=%s gradient_ratio=%s audit_interval=%s formula=%s",
            epoch_one_based,
            anchor is not None and anchor_lambda > 0.0,
            anchor_lambda,
            float(np.mean(anchor_loss_list)) if anchor_loss_list else 0.0,
            float(np.mean(anchor_gradient_ratio_list)) if anchor_gradient_ratio_list else None,
            anchor_grad_audit_interval,
            "global_sum_squared_delta_over_global_sum_squared_reference_plus_eps",
        )
        logger.info(
            "cir_state epoch=%d enabled=%s alpha=%s stats=%s",
            epoch_one_based,
            bool(use_cir_training and cir_alpha != 0.0),
            cir_alpha,
            cir_stats_list[-1] if cir_stats_list else {"enabled": False},
        )
        logger.info(
            "skip_counts epoch=%d non_finite_loss=%d non_finite_grad=%d",
            epoch + 1,
            non_finite_loss_skips,
            non_finite_grad_skips,
        )
        if model.dfg_mode == "attn":
            diagnostics = model.get_dfg_diagnostics()
            for key in sorted(diagnostics):
                value = diagnostics[key]
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
            "anchor_gradient_budget": bool(anchor_gradient_budget),
            "anchor_family_budget": float(anchor_family_budget),
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
        # New clean runs carry a resumable optimizer/RNG/scheduler payload;
        # the historical top-level adapter aliases are retained above.
        model_dict.update(build_full_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch_one_based,
            global_step=global_step,
            config=checkpoint_config or {},
            parent_config=parent_checkpoint_config,
            operational_config=operational_checkpoint_config,
            repo=repo,
            clip_sha256=clip_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            dataloader_generator=data_generator,
            anchor=anchor,
            anchor_lambda=anchor_lambda,
            seed=seed,
            precision=precision,
            tf32_enabled=tf32_enabled,
        ))
        torch.save(model_dict, ckp_path)
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
    parser.add_argument("--protocol_horizon", type=int, default=None, help="scientific total epoch horizon")

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
    parser.add_argument("--dfg_weight_residual_fp32", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic_algorithms", action="store_true")
    parser.add_argument("--use_safe_anchor", action="store_true")
    parser.add_argument("--anchor_lambda", type=float, default=0.0)
    parser.add_argument("--anchor_reference_path", type=str, default=None)
    parser.add_argument("--anchor_gradient_budget", action="store_true", help="cap Anchor gradients independently within each image-adapter family")
    parser.add_argument("--anchor_family_budget", type=float, default=ANCHOR_FAMILY_BUDGET_DEFAULT, help="maximum effective Anchor/task gradient ratio per active family")
    parser.add_argument("--anchor_family_audit", action="store_true", help="emit per-step and per-epoch Anchor family telemetry")
    parser.add_argument("--use_cir_training", action="store_true")
    parser.add_argument("--cir_alpha", type=float, default=0.0)
    parser.add_argument("--cir_peer_count", type=int, default=8)
    parser.add_argument("--cir_spatial_radius", type=int, default=3)
    parser.add_argument("--cir_transport_direction", choices=["abnormal_minus_normal_plus"], default="abnormal_minus_normal_plus")
    parser.add_argument("--cir_score_mode", choices=["exact_score_space"], default="exact_score_space")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--max_batches", type=int, default=None, help="bounded smoke/debug batches per epoch")
    parser.add_argument("--anchor_grad_audit_interval", type=int, default=0, help="0 disables per-batch anchor gradient telemetry")
    parser.add_argument("--trace_batch_identity", action="store_true", help="log first five post-augmentation batch identities for smoke checks")

    args = parser.parse_args()
    if args.protocol_horizon is None:
        args.protocol_horizon = args.epoch
    if args.anchor_grad_audit_interval < 0:
        raise ValueError("--anchor_grad_audit_interval must be non-negative")
    if args.use_soft_prompt and args.use_hybrid_soft_prompt:
        raise ValueError("--use_soft_prompt and --use_hybrid_soft_prompt are mutually exclusive")
    if args.anchor_lambda < 0:
        raise ValueError("--anchor_lambda must be non-negative")
    if not 0.0 <= args.anchor_family_budget <= 1.0:
        raise ValueError("--anchor_family_budget must be in [0, 1]")
    if args.anchor_gradient_budget and not args.use_safe_anchor:
        raise ValueError("--anchor_gradient_budget requires --use_safe_anchor")
    if args.use_safe_anchor and args.anchor_lambda == 0.0:
        raise ValueError("--use_safe_anchor requires --anchor_lambda > 0")
    if args.use_cir_training and args.cir_alpha == 0.0:
        raise ValueError("--use_cir_training requires a non-zero --cir_alpha")
    if args.cir_peer_count < 1 or args.cir_spatial_radius < 0:
        raise ValueError("invalid CIR peer geometry")
    if args.use_cir_training and args.dfg_mode != "attn":
        raise ValueError("exact CIR-V2 requires --dfg_mode attn")
    seed_everything(args.seed, deterministic_algorithms=args.deterministic_algorithms)
    repo = os.path.dirname(os.path.abspath(__file__))
    clip_path = os.path.join(repo, "model", "ViT-L-14-336px.pt")
    manifest_path = os.path.join(repo, "dataset", "hub", args.dataset + ".jsonl")
    clip_sha256 = sha256_file(clip_path) if os.path.isfile(clip_path) else None
    dataset_manifest_sha256 = sha256_file(manifest_path) if os.path.isfile(manifest_path) else None
    implementation_git_sha = current_git_sha(repo)
    tf32_enabled = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False
    anchor_reference_sha256 = (
        sha256_file(args.anchor_reference_path)
        if args.anchor_reference_path is not None and os.path.isfile(args.anchor_reference_path)
        else None
    )
    checkpoint_config = scientific_config_from_mapping(
        vars(args),
        clip_sha256=clip_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        implementation_git_sha=implementation_git_sha,
        anchor_reference_sha256=anchor_reference_sha256,
        tf32_enabled=tf32_enabled,
    )
    parent_checkpoint_config = scientific_config_from_mapping(
        vars(args),
        clip_sha256=clip_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        implementation_git_sha=implementation_git_sha,
        anchor_reference_sha256=None,
        tf32_enabled=tf32_enabled,
        parent=True,
    )
    operational_checkpoint_config = operational_config_from_mapping(vars(args))
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
    logger.info("environment: %s", environment_manifest())
    logger.info("args: %s", vars(args))
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
        dfg_weight_residual_fp32=args.dfg_weight_residual_fp32,
        use_soft_prompt=args.use_soft_prompt,
        soft_prompt_ctx_len=args.soft_prompt_ctx_len,
        soft_prompt_init=args.soft_prompt_init,
        soft_prompt_init_phrase=args.soft_prompt_init_phrase,
    ).to(device)
    model.eval()
    model.use_hybrid_soft_prompt = bool(args.use_hybrid_soft_prompt)
    model.prompt_mode = "hybrid" if args.use_hybrid_soft_prompt else ("soft" if args.use_soft_prompt else "hard")
    model.hybrid_alpha_current = 0.0
    model.hybrid_alpha_max = args.hybrid_alpha_max
    model.soft_prompt_freeze_epochs = args.soft_prompt_freeze_epochs

    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    anchor = None
    if args.use_safe_anchor:
        if args.anchor_reference_path is not None:
            anchor = SafeImageAdapterAnchor.from_checkpoint(args.anchor_reference_path, device)
        elif args.resume is None:
            anchor = SafeImageAdapterAnchor.from_module(model.image_adapter)
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
    data_generator = make_dataloader_generator(args.seed)
    worker_init = EpochWorkerInit(args.seed)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init,
        generator=data_generator,
    )
    resume_payload = None
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_identity(
            resume_payload,
            expected_scientific_config=checkpoint_config,
            expected_parent_config=parent_checkpoint_config,
            expected_total_epoch=args.epoch,
            expected_seed=args.seed,
            expected_clip_sha256=clip_sha256,
            expected_manifest_sha256=dataset_manifest_sha256,
            expected_git_sha=implementation_git_sha,
        )
        start_epoch = int(resume_payload["epoch"])
        global_step = int(resume_payload["global_step"])
        if args.use_safe_anchor and anchor is None:
            anchor = SafeImageAdapterAnchor.from_checkpoint(args.resume, device)
        logger.info("resume=%s epoch=%s global_step=%s", args.resume, start_epoch, global_step)
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
        anchor=anchor,
        anchor_lambda=args.anchor_lambda,
        use_cir_training=args.use_cir_training,
        cir_alpha=args.cir_alpha,
        cir_peer_count=args.cir_peer_count,
        cir_spatial_radius=args.cir_spatial_radius,
        data_generator=data_generator,
        worker_init=worker_init,
        start_epoch=start_epoch,
        global_step=global_step,
        checkpoint_config=checkpoint_config,
        parent_checkpoint_config=parent_checkpoint_config,
        operational_checkpoint_config=operational_checkpoint_config,
        repo=repo,
        clip_sha256=clip_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        seed=args.seed,
        precision="amp" if args.amp else "fp32",
        tf32_enabled=torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
        max_batches=args.max_batches,
        anchor_grad_audit_interval=args.anchor_grad_audit_interval,
        trace_batch_identity=args.trace_batch_identity,
        anchor_gradient_budget=args.anchor_gradient_budget,
        anchor_family_budget=args.anchor_family_budget,
        anchor_family_audit=args.anchor_family_audit,
    )


if __name__ == "__main__":
    main()
