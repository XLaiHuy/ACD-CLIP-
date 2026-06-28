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
    calculate_seg_loss, get_multiple_adapted_single_class_text_embedding
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
):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(0, total_epoch):
        epoch_one_based = epoch + 1
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
        loss_list = []
        seg_loss_list = []
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
            for class_name in list(set(class_names)):
                text_embedding_levels = get_multiple_adapted_single_class_text_embedding(
                    model, dataset_name, class_name, device
                )
                epoch_text_feature_dict[class_name] = text_embedding_levels  # [n_groups, 768, 2]
            epoch_text_features = torch.stack(
                [epoch_text_feature_dict[class_name] for class_name in class_names],
                dim=0,
            )  # [bs, n_groups, 768, 2]
            epoch_text_features = epoch_text_features.permute(1, 0, 2, 3)  # [n_groups, bs, 768, 2]
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
                loss = cls_loss + seg_loss
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
                        "loss": loss,
                    },
                    metadata={
                        "use_amp": use_amp,
                        "dfg_ss2d_fusion": model.dfg_ss2d_fusion,
                        "dfg_beta_schedule": dfg_beta_schedule,
                        "dfg_beta_target": dfg_beta_target,
                        "dfg_beta_current": model.dfg_beta,
                        "dfg_weight_residual_fp32": model.dfg_weight_residual_fp32,
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
            # clip gradient
            nn.utils.clip_grad_norm_(model.image_adapter.parameters(), 1.0)
            nn.utils.clip_grad_norm_(model.text_adapter.parameters(), 1.0)
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
            loss_list.append(loss.item())
            tqdm_train_loader.set_postfix({
                "epoch": f"{epoch + 1} / {total_epoch}",
                "loss": f"{loss.item():.4f}",
                "det_loss": f"{loss.item() - seg_loss.item():.4f}",
                "seg_loss": f"{seg_loss.item():.4f}",
                "mean_seg_loss": f"{np.mean(seg_loss_list):.4f}",
                "mean_loss": f"{np.mean(loss_list):.4f}",
                "text_lr": optimizer.param_groups[0]["lr"],
                "image_lr": optimizer.param_groups[1]["lr"],
            })
        logger.info(f"mean_loss={np.mean(loss_list)}, mean_seg_loss={np.mean(seg_loss_list)}")
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
            "text_adapter": model.text_adapter.state_dict(),
            "image_adapter": model.image_adapter.state_dict()
        }
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
    parser.add_argument("--amp", action="store_true", help="enable Automatic Mixed Precision training")
    parser.add_argument(
        "--grad_checkpointing",
        action="store_true",
        help="enable activation checkpointing to reduce ViT memory usage",
    )
    parser.add_argument("--num_workers", type=int, default=4 if os.name != "nt" else 0)

    args = parser.parse_args()
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
    ).to(device)
    model.eval()

    model.requires_grad_(False)
    model.text_adapter.requires_grad_(True)
    model.image_adapter.requires_grad_(True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info("trainable parameters: %s", f"{trainable_params:,}")
    logger.info("frozen parameters: %s", f"{frozen_params:,}")
    logger.info("dfg_weight_residual_fp32=%s", model.dfg_weight_residual_fp32)

    # set optimizer
    optimizer = torch.optim.Adam([
        {
            "params": model.text_adapter.parameters(),
            "lr": args.text_lr,
        },
        {
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
    )


if __name__ == "__main__":
    main()
