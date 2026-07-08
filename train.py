import argparse
import logging
import os

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
    converted = {}
    for key, value in diagnostics.items():
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


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8):
    p = p.float().clamp_min(eps)
    q = q.float().clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    return (0.5 * kl_pm + 0.5 * kl_qm).clamp_min(0.0)


def compute_stage_routing_consistency(
        model: ACDCLIP,
        seg_features: torch.Tensor,
        hard_text_features: torch.Tensor,
        soft_text_features: torch.Tensor,
        alpha: float,
        loss_type: str,
        margin: float,
        detach_visual: bool,
        detach_qk: bool,
):
    if loss_type == "none":
        return torch.zeros((), device=soft_text_features.device), {}
    if model.dfg_mode != "attn":
        raise ValueError("stage consistency requires dfg_mode='attn'")

    hard_anchor = hard_text_features.detach()
    stage_text_features = F.normalize((1.0 - alpha) * hard_anchor + alpha * soft_text_features, dim=2)
    hard_group_text = hard_anchor.permute(1, 0, 2, 3)  # [bs, n_groups, 768, 2]
    stage_group_text = stage_text_features.permute(1, 0, 2, 3)

    losses = []
    js_normal_values = []
    js_abnormal_values = []
    active_normal_values = []
    active_abnormal_values = []
    stats = {}
    for stage_idx in range(seg_features.shape[0]):
        img_feat = seg_features[stage_idx]
        with torch.no_grad():
            w_hard_normal, w_hard_abnormal = model._vision_text_attention_routing_weights(
                img_feat,
                hard_group_text,
                stage_idx,
                detach_qk=False,
                detach_visual=False,
            )
        w_stage_normal, w_stage_abnormal = model._vision_text_attention_routing_weights(
            img_feat,
            stage_group_text,
            stage_idx,
            detach_qk=detach_qk,
            detach_visual=detach_visual,
        )
        js_normal = js_divergence(w_stage_normal, w_hard_normal.detach())
        js_abnormal = js_divergence(w_stage_abnormal, w_hard_abnormal.detach())
        if loss_type == "js":
            loss_normal = js_normal
            loss_abnormal = js_abnormal
            active_normal = torch.ones_like(js_normal)
            active_abnormal = torch.ones_like(js_abnormal)
        elif loss_type == "js_margin":
            loss_normal = torch.relu(js_normal - margin)
            loss_abnormal = torch.relu(js_abnormal - margin)
            active_normal = (js_normal > margin).float()
            active_abnormal = (js_abnormal > margin).float()
        else:
            raise ValueError(f"Unknown stage_consistency_loss: {loss_type}")

        losses.extend([loss_normal.mean(), loss_abnormal.mean()])
        js_normal_values.append(js_normal.detach())
        js_abnormal_values.append(js_abnormal.detach())
        active_normal_values.append(active_normal.detach())
        active_abnormal_values.append(active_abnormal.detach())

        prefix = f"stage{stage_idx + 1}"
        stats[f"{prefix}_stage_js_normal"] = float(js_normal.detach().mean().item())
        stats[f"{prefix}_stage_js_abnormal"] = float(js_abnormal.detach().mean().item())
        stats[f"{prefix}_stage_active_normal"] = float(active_normal.detach().mean().item())
        stats[f"{prefix}_stage_active_abnormal"] = float(active_abnormal.detach().mean().item())
        stats[f"{prefix}_w_hard_normal_sum_error"] = float((w_hard_normal.sum(dim=1) - 1).abs().max().item())
        stats[f"{prefix}_w_hard_abnormal_sum_error"] = float((w_hard_abnormal.sum(dim=1) - 1).abs().max().item())
        stats[f"{prefix}_w_stage_normal_sum_error"] = float((w_stage_normal.sum(dim=1) - 1).abs().max().detach().item())
        stats[f"{prefix}_w_stage_abnormal_sum_error"] = float((w_stage_abnormal.sum(dim=1) - 1).abs().max().detach().item())
        for route_idx in range(w_stage_normal.shape[1]):
            stats[f"{prefix}_w_hard_normal_g{route_idx + 1}"] = float(
                w_hard_normal[:, route_idx].detach().mean().item()
            )
            stats[f"{prefix}_w_stage_normal_g{route_idx + 1}"] = float(
                w_stage_normal[:, route_idx].detach().mean().item()
            )
            stats[f"{prefix}_w_hard_abnormal_g{route_idx + 1}"] = float(
                w_hard_abnormal[:, route_idx].detach().mean().item()
            )
            stats[f"{prefix}_w_stage_abnormal_g{route_idx + 1}"] = float(
                w_stage_abnormal[:, route_idx].detach().mean().item()
            )

    if not losses:
        return torch.zeros((), device=soft_text_features.device), {}

    stage_loss = torch.stack(losses).mean()
    normal_js = torch.cat(js_normal_values)
    abnormal_js = torch.cat(js_abnormal_values)
    normal_active = torch.cat(active_normal_values)
    abnormal_active = torch.cat(active_abnormal_values)
    stats.update({
        "mean_stage_js": float(torch.cat([normal_js, abnormal_js]).mean().item()),
        "normal_stage_js": float(normal_js.mean().item()),
        "abnormal_stage_js": float(abnormal_js.mean().item()),
        "stage_active_fraction": float(torch.cat([normal_active, abnormal_active]).mean().item()),
        "normal_stage_active_fraction": float(normal_active.mean().item()),
        "abnormal_stage_active_fraction": float(abnormal_active.mean().item()),
        "w_hard_normal_mean": float(1.0 / model.n_groups),
        "w_stage_normal_mean": float(1.0 / model.n_groups),
        "w_hard_abnormal_mean": float(1.0 / model.n_groups),
        "w_stage_abnormal_mean": float(1.0 / model.n_groups),
        "detach_visual": float(bool(detach_visual)),
        "detach_qk": float(bool(detach_qk)),
    })
    return stage_loss, stats


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
        lambda_stage: float = 0.0,
        stage_consistency_loss: str = "none",
        stage_consistency_margin: float = 0.02,
        stage_consistency_update_soft_only: bool = False,
        stage_consistency_detach_visual: bool = False,
        stage_consistency_detach_qk: bool = False,
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
                "soft_prompt_freeze_epochs=%d lambda_kg=%s lambda_k=%s lambda_stage=%s "
                "stage_consistency_loss=%s stage_consistency_margin=%s "
                "stage_update_soft_only=%s stage_detach_visual=%s stage_detach_qk=%s grad_clip_norm=%s "
                "image_lr=%s text_lr=%s soft_lr=%s",
                epoch_one_based,
                getattr(model, "prompt_mode", "hybrid"),
                hybrid_alpha_current,
                soft_prompt_frozen,
                soft_prompt_freeze_epochs,
                lambda_kg,
                lambda_k,
                lambda_stage,
                stage_consistency_loss,
                stage_consistency_margin,
                stage_consistency_update_soft_only,
                stage_consistency_detach_visual,
                stage_consistency_detach_qk,
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
        stage_loss_list = []
        soft_prompt_stats_list = []
        k_reg_stats_list = []
        stage_stats_list = []
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
            hard_text_feature_dict = {}
            soft_text_feature_dict = {}
            kg_losses = []
            k_losses = []
            batch_soft_stats = []
            batch_k_stats = []
            for class_name in list(set(class_names)):
                if use_hybrid_soft_prompt:
                    stage_enabled = lambda_stage > 0 and stage_consistency_loss != "none"
                    if lambda_k > 0 or stage_enabled:
                        (
                            text_embedding_levels,
                            kg_loss_class,
                            soft_stats,
                            components,
                        ) = get_hybrid_soft_prompt_single_class_text_embedding(
                            model, dataset_name, class_name, device, return_kg=True, return_components=True
                        )
                        if lambda_k > 0:
                            k_loss_class, k_stats = compute_hybrid_k_regularization(
                                model,
                                components["hard_text"],
                                components["soft_text"],
                                hybrid_alpha_current,
                            )
                            k_losses.append(k_loss_class)
                            batch_k_stats.append(k_stats)
                        if stage_enabled:
                            hard_text_feature_dict[class_name] = components["hard_text"]
                            soft_text_feature_dict[class_name] = components["soft_text"]
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
            if hard_text_feature_dict:
                hard_text_features = torch.stack(
                    [hard_text_feature_dict[class_name] for class_name in class_names],
                    dim=0,
                ).permute(1, 0, 2, 3)
                soft_text_features = torch.stack(
                    [soft_text_feature_dict[class_name] for class_name in class_names],
                    dim=0,
                ).permute(1, 0, 2, 3)
            else:
                hard_text_features = None
                soft_text_features = None
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
                if lambda_stage > 0 and stage_consistency_loss != "none":
                    if hard_text_features is None or soft_text_features is None:
                        raise RuntimeError("stage consistency requires hybrid hard/soft text components")
                    stage_loss, stage_stats = compute_stage_routing_consistency(
                        model=model,
                        seg_features=seg_features,
                        hard_text_features=hard_text_features,
                        soft_text_features=soft_text_features,
                        alpha=hybrid_alpha_current,
                        loss_type=stage_consistency_loss,
                        margin=stage_consistency_margin,
                        detach_visual=stage_consistency_detach_visual or stage_consistency_update_soft_only,
                        detach_qk=stage_consistency_detach_qk or stage_consistency_update_soft_only,
                    )
                    stage_stats_list.append(stage_stats)
                else:
                    stage_loss = torch.zeros((), device=device)
                loss = loss_main + lambda_kg * kg_loss + lambda_k * k_loss + lambda_stage * stage_loss
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
                        "stage_loss": stage_loss,
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
                        "lambda_stage": lambda_stage,
                        "stage_consistency_loss": stage_consistency_loss,
                        "stage_consistency_margin": stage_consistency_margin,
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
            stage_loss_list.append(stage_loss.item())
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
                        "stage_loss": stage_loss,
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
                        "lambda_stage": lambda_stage,
                        "stage_consistency_loss": stage_consistency_loss,
                        "stage_consistency_margin": stage_consistency_margin,
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
                if lambda_stage > 0 and stage_consistency_loss != "none":
                    postfix["stage_loss"] = f"{stage_loss.item():.5f}"
                    postfix["wstage"] = f"{(lambda_stage * stage_loss).item():.5f}"
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
                "mean_stage_loss=%s weighted_stage_loss=%s lambda_stage=%s "
                "stage_consistency_loss=%s stage_consistency_margin=%s stage_stats=%s "
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
                float(np.mean(stage_loss_list)) if stage_loss_list else None,
                float(lambda_stage * np.mean(stage_loss_list)) if stage_loss_list else None,
                lambda_stage,
                stage_consistency_loss,
                stage_consistency_margin,
                mean_stats(stage_stats_list),
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
            "lambda_stage": lambda_stage,
            "stage_consistency_loss": stage_consistency_loss,
            "stage_consistency_margin": stage_consistency_margin,
            "stage_consistency_update_soft_only": stage_consistency_update_soft_only,
            "stage_consistency_detach_visual": stage_consistency_detach_visual,
            "stage_consistency_detach_qk": stage_consistency_detach_qk,
            "k_reg_detached_wk": bool(lambda_k > 0),
            "k_reg_per_stage": bool(lambda_k > 0),
            "text_adapter": model.text_adapter.state_dict(),
            "image_adapter": model.image_adapter.state_dict()
        }
        if use_soft_prompt or use_hybrid_soft_prompt:
            model_dict["soft_prompt"] = model.soft_prompt.state_dict()
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
    parser.add_argument("--lambda_stage", type=float, default=0.0, help="Phase3B DFG routing consistency weight")
    parser.add_argument(
        "--stage_consistency_loss",
        type=str,
        choices=["none", "js", "js_margin"],
        default="none",
        help="Phase3B stage routing consistency loss",
    )
    parser.add_argument(
        "--stage_consistency_margin",
        type=float,
        default=0.02,
        help="JS margin for Phase3B stage routing trust region",
    )
    parser.add_argument(
        "--stage_consistency_update_soft_only",
        action="store_true",
        help="detach hard/visual/qk paths so L_stage mainly regularizes soft prompt",
    )
    parser.add_argument(
        "--stage_consistency_detach_visual",
        action="store_true",
        help="detach visual features and SS2D branch from Phase3B stage loss",
    )
    parser.add_argument(
        "--stage_consistency_detach_qk",
        action="store_true",
        help="use detached W_Q/W_K parameters for Phase3B stage loss",
    )
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

    args = parser.parse_args()
    if args.use_soft_prompt and args.use_hybrid_soft_prompt:
        raise ValueError("--use_soft_prompt and --use_hybrid_soft_prompt are mutually exclusive")
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
    ).to(device)
    model.eval()
    model.use_hybrid_soft_prompt = bool(args.use_hybrid_soft_prompt)
    model.prompt_mode = "hybrid" if args.use_hybrid_soft_prompt else ("soft" if args.use_soft_prompt else "hard")
    model.hybrid_alpha_current = 0.0
    model.hybrid_alpha_max = args.hybrid_alpha_max
    model.soft_prompt_freeze_epochs = args.soft_prompt_freeze_epochs

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
        lambda_stage=args.lambda_stage,
        stage_consistency_loss=args.stage_consistency_loss,
        stage_consistency_margin=args.stage_consistency_margin,
        stage_consistency_update_soft_only=args.stage_consistency_update_soft_only,
        stage_consistency_detach_visual=args.stage_consistency_detach_visual,
        stage_consistency_detach_qk=args.stage_consistency_detach_qk,
        hybrid_alpha_max=args.hybrid_alpha_max,
        soft_prompt_freeze_epochs=args.soft_prompt_freeze_epochs,
        grad_clip_norm=args.grad_clip_norm,
    )


if __name__ == "__main__":
    main()
