#!/usr/bin/env python3
"""Dedicated Phase2C A-prime/B training entrypoint. Importing this module never launches training."""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import csv
import json
import logging
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from dataset import get_text_and_image_dataset
from dataset.info import DOMAINS
from model.adapter import ACDCLIP
from model.clip import create_model
from phase2c_pcgrad import apply_pcgrad
from phase2c_pcgrad_diagnostics import PCGRAD_DIAGNOSTIC_FIELDS, run_pcgrad_diagnostics
from phase2c_utils import (
    EpochDeterministicSampler,
    alpha_for_epoch,
    append_csv,
    persist_diagnostic_batch_ids,
    phase2c_config,
    run_gradient_diagnostics,
    seed_everything,
    seed_worker,
    write_selection,
)
from train import (
    apply_soft_prompt_lr_policy,
    clip_module_grad,
    compute_hybrid_k_regularization,
    first_nonfinite_trainable_parameter,
    get_dfg_beta_for_epoch,
    has_non_finite_grad,
    save_nonfinite_diagnostics,
)
from utils import calculate_seg_loss, get_hybrid_soft_prompt_single_class_text_embedding


METRIC_FIELDS = [
    "epoch", "scope", "class_name", "n", "pixel_auc", "pixel_ap", "image_auc", "image_ap",
]
DIAGNOSTIC_FIELDS = [
    "epoch", "batch", "parameter_group", "cls_grad_norm", "seg_grad_norm", "cosine",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Phase2C deterministic VisA curriculum training")
    parser.add_argument(
        "--condition", required=True,
        choices=["A_prime", "B", "C", "P", "P_LoRA_only"],
        help="Training condition name",
    )
    parser.add_argument(
        "--diagnostic-batch-size", type=int, default=1,
        help="batch size for post-epoch gradient diagnostics; kept separate to limit VRAM",
    )
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--hybrid-alpha-max", required=True, type=float)
    parser.add_argument("--train-manifest", default="splits/visa_train_seed42.csv")
    parser.add_argument("--val-manifest", default="splits/visa_val_seed42.csv")
    parser.add_argument("--split-metadata", default="splits/visa_split_seed42_metadata.json")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--metric-thresholds", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--bf16", action="store_true",
        help="use CUDA autocast with bfloat16 (without gradient scaling)",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print resolved config only")
    # Engineering-only smoke-test controls.  Default None means full execution.
    # A smoke run is NOT a scientific result and must use a distinct save-path.
    parser.add_argument(
        "--max-train-batches", type=int, default=None,
        help="[SMOKE/DEBUG] stop training loop after this many batches per epoch",
    )
    parser.add_argument(
        "--max-val-batches", type=int, default=None,
        help="[SMOKE/DEBUG] stop validation loop after this many batches per class",
    )
    return parser.parse_args()


def build_model(config, device):
    clip_model = create_model(
        model_name="ViT-L-14-336", img_size=config["img_size"], device=device,
        pretrained="openai", require_pretrained=True,
    )
    if config["grad_checkpointing"]:
        clip_model.set_grad_checkpointing(True)
    clip_model.eval()
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=config["n_groups"],
        image_adapt_weight=0.2,
        conv_lora_rank=8,
        conv_lora_alpha=2.0,
        conv_kernel_size_list=[3, 5],
        text_adapt_weight=config["text_adapt_weight"],
        lora_rank=16,
        lora_alpha=2.0,
        dfg_mode=config["dfg_mode"],
        dfg_attn_dim=config["dfg_attn_dim"],
        dfg_attn_tau=config["dfg_attn_tau"],
        use_ss2d_dfg=config["use_ss2d_dfg"],
        dfg_gamma_max=config["dfg_gamma_max"],
        dfg_ss2d_fusion=config["dfg_ss2d_fusion"],
        dfg_beta=config["dfg_beta"],
        dfg_beta_schedule=config["dfg_beta_schedule"],
        dfg_beta_target=config["dfg_beta_target"],
        dfg_beta_current=config["dfg_beta"],
        use_soft_prompt=False,
        soft_prompt_ctx_len=config["soft_prompt_ctx_len"],
        soft_prompt_init="phrase",
        soft_prompt_init_phrase="a photo of a",
    ).to(device)
    model.eval()
    model.use_hybrid_soft_prompt = True
    model.prompt_mode = "hybrid"
    model.hybrid_alpha_current = 0.0
    model.hybrid_alpha_max = config["hybrid_alpha_max"]
    model.soft_prompt_freeze_epochs = config["soft_prompt_freeze_epochs"]
    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.soft_prompt.requires_grad_(False)
    return model


def build_optimizer(model, config):
    optimizer = torch.optim.Adam([
        {"name": "text_adapter", "params": model.text_adapter.parameters(), "lr": config["text_lr"]},
        {"name": "image_adapter", "params": model.image_adapter.parameters(), "lr": config["image_lr"]},
        {
            "name": "soft_prompt", "params": model.soft_prompt.parameters(), "lr": 0.0,
            "constant_lr": config["soft_prompt_lr"],
        },
    ])
    return optimizer, StepLR(optimizer, step_size=1, gamma=config["lr_gamma"])


def text_features_for_batch(model, class_names, device, dataset_name="VisA", include_regularizers=True):
    cache = {}
    kg_losses = []
    k_losses = []
    for class_name in sorted(set(class_names)):
        if include_regularizers:
            embeddings, kg_loss, _, components = get_hybrid_soft_prompt_single_class_text_embedding(
                model, dataset_name, class_name, device, return_kg=True, return_components=True
            )
            k_loss, _ = compute_hybrid_k_regularization(
                model, components["hard_text"], components["soft_text"], model.hybrid_alpha_current
            )
            kg_losses.append(kg_loss)
            k_losses.append(k_loss)
        else:
            embeddings, _, _ = get_hybrid_soft_prompt_single_class_text_embedding(
                model, dataset_name, class_name, device, return_kg=True
            )
        cache[class_name] = embeddings
    features = torch.stack([cache[name] for name in class_names], dim=0).permute(1, 0, 2, 3)
    zero = torch.zeros((), device=device)
    return (
        features,
        torch.stack(kg_losses).mean() if kg_losses else zero,
        torch.stack(k_losses).mean() if k_losses else zero,
    )


def forward_losses(model, batch, device, include_regularizers=True):
    image = batch["image"].to(device)
    mask = batch["mask"].to(device)
    label = batch["label"].to(device)
    text_features, kg_loss, k_loss = text_features_for_batch(
        model, list(batch["class_name"]), device, include_regularizers=include_regularizers
    )
    seg_tokens, det_tokens = model(image)
    seg_features = torch.stack(seg_tokens, dim=0)
    det_features = torch.stack(det_tokens, dim=0)

    # Keep the large CLIP forward in AMP, but compute the much smaller
    # vision--text logits and losses in FP32.  By epoch 6 the hybrid prompt
    # reaches alpha=0.20; writing these reductions/softmax logits in FP16 can
    # overflow on T4 even while every trainable parameter is still finite.
    # Disabling autocast here avoids BF16's severe T4 slowdown without making
    # the ViT encoder run in FP32.
    with torch.autocast(device_type=device.type, enabled=False):
        text_features_fp32 = text_features.float()
        cls_pred = torch.stack([
            torch.matmul(
                det_features[index].float().unsqueeze(1), text_features_fp32[index]
            ).squeeze(1)
            for index in range(det_features.shape[0])
        ], dim=0).mean(dim=0)
        cls_loss = F.cross_entropy(cls_pred, label)
        seg_pred = model.vision_text_fusion_gate_seg(
            seg_features.float(), text_features_fp32
        )
        seg_loss = calculate_seg_loss(seg_pred, mask)
    return cls_loss, seg_loss, kg_loss, k_loss


def _safe_metric(metric):
    try:
        return float(metric.compute().item() * 100.0)
    except (RuntimeError, ValueError):
        return float("nan")


@torch.no_grad()
def validate_visa(model, dataset, device, epoch, batch_size, num_workers, thresholds):
    by_class = defaultdict(list)
    for index, record in enumerate(dataset.meta):
        by_class[record["class_name"]].append(index)
    rows = []
    for class_name in sorted(by_class):
        loader = DataLoader(
            Subset(dataset, by_class[class_name]), batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
        )
        pixel_auc = BinaryAUROC(thresholds=thresholds)
        pixel_ap = BinaryAveragePrecision(thresholds=thresholds)
        image_auc = BinaryAUROC(thresholds=thresholds)
        image_ap = BinaryAveragePrecision(thresholds=thresholds)
        n = 0
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device).to(torch.int32)
            label = batch["label"].to(torch.int32)
            text, _, _ = text_features_for_batch(
                model, list(batch["class_name"]), device, include_regularizers=False
            )
            seg_tokens, det_tokens = model(image)
            seg_features = torch.stack(seg_tokens)
            det_features = torch.stack(det_tokens)
            cls_logits = torch.stack([
                torch.matmul(det_features[index].unsqueeze(1), text[index]).squeeze(1)
                for index in range(det_features.shape[0])
            ]).mean(0)
            cls_score = F.softmax(cls_logits, dim=1)[:, 1]
            seg_score = model.vision_text_fusion_gate_seg(
                seg_features, text, test_mode=True, domain=DOMAINS["VisA"]
            )
            pixel_auc.update(seg_score.detach().flatten().cpu(), mask.flatten().cpu())
            pixel_ap.update(seg_score.detach().flatten().cpu(), mask.flatten().cpu())
            image_auc.update(cls_score.detach().cpu(), label.cpu())
            image_ap.update(cls_score.detach().cpu(), label.cpu())
            n += len(label)
        rows.append({
            "epoch": epoch, "scope": "category", "class_name": class_name, "n": n,
            "pixel_auc": _safe_metric(pixel_auc), "pixel_ap": _safe_metric(pixel_ap),
            "image_auc": _safe_metric(image_auc), "image_ap": _safe_metric(image_ap),
        })
    finite_rows = [
        row for row in rows
        if all(math.isfinite(row[key]) for key in ("pixel_auc", "pixel_ap", "image_auc", "image_ap"))
    ]
    if not finite_rows:
        raise RuntimeError("No VisA category has valid validation metrics")
    macro = {
        "epoch": epoch, "scope": "macro", "class_name": "__macro__", "n": sum(row["n"] for row in rows)
    }
    for key in ("pixel_auc", "pixel_ap", "image_auc", "image_ap"):
        macro[key] = float(np.mean([row[key] for row in finite_rows]))
    return rows + [macro]


def diagnostic_batches(dataset, id_batches):
    index_by_id = {record["sample_id"]: index for index, record in enumerate(dataset.meta)}
    return [default_collate([dataset[index_by_id[sample_id]] for sample_id in ids]) for ids in id_batches]


def checkpoint_payload(model, config, epoch):
    return {
        "epoch": epoch,
        "condition": config["condition"],
        "seed": config["seed"],
        "n_groups": model.n_groups,
        "dfg_mode": model.dfg_mode,
        "dfg_attn_dim": model.dfg_attn_dim,
        "dfg_attn_tau": model.dfg_attn_tau,
        "use_ss2d_dfg": model.use_ss2d_dfg,
        "dfg_gamma_max": model.dfg_gamma_max,
        "dfg_ss2d_fusion": model.dfg_ss2d_fusion,
        "dfg_beta": model.dfg_beta,
        "dfg_beta_schedule": config["dfg_beta_schedule"],
        "dfg_beta_target": config["dfg_beta_target"],
        "prompt_mode": "hybrid",
        "use_soft_prompt": False,
        "use_hybrid_soft_prompt": True,
        "soft_prompt_ctx_len": config["soft_prompt_ctx_len"],
        "hybrid_alpha_current": model.hybrid_alpha_current,
        "hybrid_alpha_max": config["hybrid_alpha_max"],
        "soft_prompt_freeze_epochs": config["soft_prompt_freeze_epochs"],
        "grad_clip_norm": config["grad_clip_norm"],
        "lambda_kg": config["lambda_kg"],
        "lambda_k": config["lambda_k"],
        "text_adapter": model.text_adapter.state_dict(),
        "image_adapter": model.image_adapter.state_dict(),
        "soft_prompt": model.soft_prompt.state_dict(),
    }


def train_phase2c(args):
    config = phase2c_config(args.condition, args.save_path, args.hybrid_alpha_max)
    config["batch_size"] = args.batch_size
    config["num_workers"] = args.num_workers
    config["amp"] = not args.no_amp
    config["bf16"] = args.bf16
    config["diagnostic_batch_size"] = args.diagnostic_batch_size
    # Determine the manual loss scaling factor for PCGrad to prevent FP16 underflow.
    # - BF16 uses 1.0 (no scaling needed)
    # - FP16 AMP uses 65536.0 (prevents underflow)
    # - FP32 uses 1.0 (no scaling needed)
    if config["bf16"]:
        scale_factor = 1.0
    elif config["amp"]:
        scale_factor = 8192.0
    else:
        scale_factor = 1.0
    config["pcgrad_scale_factor"] = scale_factor

    # Record smoke-test limits in config so any output is clearly marked.
    max_train_batches = getattr(args, "max_train_batches", None)
    max_val_batches = getattr(args, "max_val_batches", None)
    if max_train_batches is not None or max_val_batches is not None:
        config["smoke_test"] = True
        config["max_train_batches"] = max_train_batches
        config["max_val_batches"] = max_val_batches
    if config["pcgrad_enabled"]:
        config["training_commit_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    if args.no_amp and args.bf16:
        raise ValueError("--bf16 cannot be combined with --no-amp")
    if args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return
    for required in (args.train_manifest, args.val_manifest, args.split_metadata):
        if not Path(required).is_file():
            raise FileNotFoundError(f"Required Phase2C split artifact missing: {required}")

    output = Path(args.save_path)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(args.split_metadata, output / "split_metadata.json")
    logging.basicConfig(
        filename=output / "train.log", level=logging.INFO,
        format="%(asctime)s %(filename)s %(lineno)d: %(message)s",
    )
    logger = logging.getLogger("phase2c")
    logger.info("config=%s", config)
    if config["pcgrad_enabled"]:
        if config["bf16"]:
            logger.info("PCGrad precision mode: native BF16")
        elif config["amp"]:
            logger.info(
                "PCGrad precision mode: FP16 with manual scaling (scale_factor=%s)",
                config["pcgrad_scale_factor"],
            )
        else:
            logger.info("PCGrad precision mode: FP32")

    seed_everything(config["seed"])
    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() and args.cuda_device >= 0 else "cpu"
    )
    if config["bf16"] and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("--bf16 requires a CUDA device with native bfloat16 support")
    model = build_model(config, device)
    optimizer, scheduler = build_optimizer(model, config)
    train_dataset = get_text_and_image_dataset("VisA", config["img_size"], "train", args.train_manifest)
    diagnostic_dataset = get_text_and_image_dataset("VisA", config["img_size"], "val", args.train_manifest)
    val_dataset = get_text_and_image_dataset("VisA", config["img_size"], "val", args.val_manifest)
    sampler = EpochDeterministicSampler(train_dataset, config["seed"])
    loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], sampler=sampler,
        num_workers=config["num_workers"], pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    id_batches = persist_diagnostic_batch_ids(
        train_dataset, config["diagnostic_batch_size"], output / "diagnostic_batches.json", config["seed"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"] and not config["bf16"])
    autocast_dtype = torch.bfloat16 if config["bf16"] else torch.float16

    metric_rows = []
    for epoch in range(1, config["epochs"] + 1):
        sampler.set_epoch(epoch - 1)
        alpha = alpha_for_epoch(epoch, config["hybrid_alpha_max"], config["soft_prompt_freeze_epochs"])
        model.hybrid_alpha_current = alpha
        frozen = epoch <= config["soft_prompt_freeze_epochs"]
        model.soft_prompt.requires_grad_(not frozen)
        apply_soft_prompt_lr_policy(optimizer, frozen)
        beta = get_dfg_beta_for_epoch(
            max(1, epoch - config["activation_delay_epochs"]),
            config["dfg_beta_schedule"],
            config["dfg_beta_target"], config["dfg_beta"]
        )
        model.set_dfg_beta(beta)
        losses = []
        non_finite_loss_skips = 0
        non_finite_grad_skips = 0
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Phase2C {config['condition']} epoch {epoch}")):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=config["amp"]):
                cls_loss, seg_loss, kg_loss, k_loss = forward_losses(model, batch, device)
                # Explicit weighted losses (coefficients are currently 1.0;
                # the named variables survive coefficient changes without
                # silently breaking PCGrad's other_loss computation).
                cls_loss_weighted = cls_loss
                seg_loss_weighted = seg_loss
                loss = (
                    cls_loss_weighted
                    + seg_loss_weighted
                    + config["lambda_kg"] * kg_loss
                    + config["lambda_k"] * k_loss
                )
            if not torch.isfinite(loss):
                # A non-finite forward pass has not modified parameters yet.
                # Preserve diagnostics and skip this deterministic batch, as the
                # main training entrypoint does, instead of discarding an entire
                # long run on the first transient FP16 overflow.
                non_finite_loss_skips += 1
                diag_path = save_nonfinite_diagnostics(
                    save_path=str(output),
                    epoch_one_based=epoch,
                    batch_idx=batch_idx,
                    non_finite_loss_skips=non_finite_loss_skips,
                    model=model,
                    tensors={
                        "loss": loss,
                        "cls_loss": cls_loss,
                        "seg_loss": seg_loss,
                        "kg_loss": kg_loss,
                        "k_loss": k_loss,
                    },
                    metadata={
                        "condition": config["condition"],
                        "alpha": alpha,
                        "beta": beta,
                        "pcgrad_scale_factor": config["pcgrad_scale_factor"],
                        "class_names": list(batch["class_name"]),
                        "labels": batch["label"].detach().cpu().tolist(),
                    },
                )
                logger.warning(
                    "Non-finite loss at epoch %d batch %d (skip %d): "
                    "loss=%s cls=%s seg=%s kg=%s k=%s; diagnostics=%s",
                    epoch, batch_idx, non_finite_loss_skips,
                    bool(torch.isfinite(loss).item()),
                    bool(torch.isfinite(cls_loss).item()),
                    bool(torch.isfinite(seg_loss).item()),
                    bool(torch.isfinite(kg_loss).item()),
                    bool(torch.isfinite(k_loss).item()),
                    diag_path,
                )
                optimizer.zero_grad(set_to_none=True)
                if non_finite_loss_skips > config["non_finite_loss_abort_threshold"]:
                    raise RuntimeError(
                        "Aborting Phase2C because non-finite-loss skips exceeded "
                        f"{config['non_finite_loss_abort_threshold']} at epoch {epoch}. "
                        f"Latest diagnostics: {diag_path}"
                    )
                continue
            optimizer.zero_grad(set_to_none=True)
            if config["pcgrad_enabled"]:
                # apply_pcgrad populates .grad for all trainable parameters.
                # Do NOT call loss.backward() after this point.
                apply_pcgrad(
                    loss,
                    cls_loss_weighted,
                    seg_loss_weighted,
                    model,
                    config["pcgrad_groups"],
                    config["pcgrad_epsilon"],
                    scale_factor=config.get("pcgrad_scale_factor", 1.0),
                )
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            if has_non_finite_grad(optimizer):
                if config["pcgrad_enabled"] and config["amp"] and not config["bf16"]:
                    old_scale = config["pcgrad_scale_factor"]
                    new_scale = max(1.0, old_scale * 0.5)
                    config["pcgrad_scale_factor"] = new_scale
                    optimizer.zero_grad(set_to_none=True)
                    logger.warning(
                        "Epoch %d, batch %d: Non-finite gradient detected. Skipping optimizer step and reducing pcgrad_scale_factor from %s to %s.",
                        epoch, batch_idx, old_scale, new_scale
                    )
                    non_finite_grad_skips += 1
                    continue
                else:
                    raise RuntimeError(f"Non-finite Phase2C gradient at epoch {epoch}")
            clip_module_grad(model.image_adapter, config["grad_clip_norm"])
            clip_module_grad(model.text_adapter, config["grad_clip_norm"])
            if not frozen:
                clip_module_grad(model.soft_prompt, config["grad_clip_norm"])
            if config["pcgrad_enabled"]:
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            bad_param_name, _ = first_nonfinite_trainable_parameter(model)
            if bad_param_name is not None:
                raise RuntimeError(
                    f"Non-finite trainable parameter '{bad_param_name}' after optimizer step "
                    f"at epoch {epoch}, batch {batch_idx}."
                )
            losses.append(float(loss.item()))
        scheduler.step()
        apply_soft_prompt_lr_policy(optimizer, frozen)
        checkpoint = checkpoints / f"adapter_{epoch}.pth"
        torch.save(checkpoint_payload(model, config, epoch), checkpoint)

        fixed_batches = diagnostic_batches(diagnostic_dataset, id_batches)
        # The diagnostic pass has the same batch size as training. Release
        # allocator-reserved blocks before its fresh autograd graphs are built.
        if device.type == "cuda":
            torch.cuda.empty_cache()

        def diagnostic_loss_builder(batch):
            # Keep diagnostics in the same AMP mode as the training forward
            # pass; otherwise a BF16 run would unexpectedly build FP32
            # activations here and can exceed the GPU budget after an epoch.
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=config["amp"]):
                return forward_losses(model, batch, device, include_regularizers=False)[:2]

        diagnostics = run_gradient_diagnostics(
            model, optimizer, scheduler, fixed_batches, diagnostic_loss_builder,
            epoch,
        )
        append_csv(output / "gradient_diagnostics.csv", diagnostics, DIAGNOSTIC_FIELDS)
        if config["pcgrad_enabled"]:
            # pcgrad_diagnostics.csv covers only groups that receive PCGrad.
            pcgrad_rows = run_pcgrad_diagnostics(
                model, optimizer, scheduler, fixed_batches,
                diagnostic_loss_builder, epoch,
                config["pcgrad_groups"],
                config["pcgrad_epsilon"],
            )
            append_csv(output / "pcgrad_diagnostics.csv", pcgrad_rows, PCGRAD_DIAGNOSTIC_FIELDS)
        validation = validate_visa(
            model, val_dataset, device, epoch, config["batch_size"],
            config["num_workers"], args.metric_thresholds,
        )
        append_csv(output / "visa_val_metrics.csv", validation, METRIC_FIELDS)
        metric_rows.extend(validation)
        macro_rows = [row for row in metric_rows if row["scope"] == "macro"]
        if len(macro_rows) >= 3:
            selection_rows = [
                {**row, "checkpoint": f"checkpoints/adapter_{row['epoch']}.pth"} for row in macro_rows
            ]
            write_selection(selection_rows, output / "selection.json")
        logger.info(
            "epoch=%d alpha=%s beta=%s mean_loss=%s validation_macro=%s "
            "non_finite_loss_skips=%d non_finite_grad_skips=%d",
            epoch, alpha, beta, float(np.mean(losses)), validation[-1],
            non_finite_loss_skips, non_finite_grad_skips,
        )


def main():
    train_phase2c(parse_args())


if __name__ == "__main__":
    main()
