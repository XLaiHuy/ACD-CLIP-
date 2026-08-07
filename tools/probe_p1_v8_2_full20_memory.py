#!/usr/bin/env python3
"""Controlled 6-Microbatch Train Memory Probe for P1-v8.2.

Executes exactly 6 microbatches (1 optimizer-equivalent accumulation window)
with batch_size=1, grad_accum_steps=6, precision=bf16, img_size=518, Candidate-1 parameters.
Measures peak GPU reserved/allocated memory, host RSS, available RAM, and finite gradients.
"""
from __future__ import annotations

import json
import os
import sys
import time

import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset import get_text_and_image_dataset
from model.adapter import ACDCLIP
from model.clip import create_model
from model.h6.losses import (
    active_role_balanced_router_loss,
    actual_local_residual_loss,
    build_semantic_roles,
    factor_specific_residual_role_loss,
)
from utils import calculate_seg_loss, get_phase2b_global_text_features

_CONFIG_PATH = "configs/phase4/p1_v8_2_candidate1.json"
_OUTPUT_PATH = "runs/phase4/p1_v8_2_full20_prelaunch_audit/train_memory_probe.json"


def get_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_available_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 * 1024 * 1024)


def main():
    print("=== Starting Controlled Train Memory Probe (6 Microbatches = 1 Accumulation Window) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[ERROR] CUDA is not available. Cannot perform GPU memory probe.")
        sys.exit(1)

    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)

    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)

    # 1. Reset CUDA peak memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    start_rss = get_rss_mb()
    start_available_ram = get_available_ram_gb()
    total_gpu_mem_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)

    # 2. Model init
    precision = cfg.get("precision", "bf16")
    torch_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    clip_model = create_model(
        "ViT-L-14-336",
        img_size=cfg["img_size"],
        device=device,
        pretrained="openai",
        require_pretrained=True,
        precision=precision,
    )
    clip_model.set_grad_checkpointing(True)
    clip_model.eval()

    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=cfg["n_groups"],
        lora_rank=16,
        lora_alpha=2.0,
        conv_lora_rank=8,
        conv_lora_alpha=2.0,
        conv_kernel_size_list=[3, 5],
        dfg_mode=cfg["dfg_mode"],
        dfg_attn_dim=256,
        dfg_attn_tau=cfg["dfg_attn_tau"],
        use_ss2d_dfg=cfg["use_ss2d_dfg"],
        dfg_ss2d_fusion=cfg["dfg_ss2d_fusion"],
        dfg_beta=0.10,
        h6_progress=1,
    )
    model.train()
    model.to(device)
    model.clipmodel.eval()

    # Freeze rho
    rho_raw = dict(model.named_parameters()).get("h6.rho.raw", None)
    if rho_raw is not None:
        rho_raw.requires_grad_(False)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16"))

    # 3. Dataset & DataLoader
    dataset = get_text_and_image_dataset(cfg.get("dataset", "VisA"), img_size=cfg["img_size"], stage="train")
    if isinstance(dataset, dict):
        dataset = torch.utils.data.ConcatDataset(list(dataset.values()))

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    microbatch_records = []
    optimizer_step_occurred = False

    t0 = time.time()
    for batch_idx, batch_data in enumerate(loader, start=1):
        if batch_idx > 6:
            break

        image = batch_data["image"].to(device, non_blocking=True)
        mask = batch_data["mask"].to(device, non_blocking=True)
        label = batch_data["label"].to(device, non_blocking=True)
        local_mask_valid = batch_data.get("local_mask_valid", torch.ones_like(mask)).to(device, non_blocking=True)
        class_names = list(batch_data["class_name"])

        autocast_ctx = torch.autocast("cuda", dtype=torch_dtype)
        with autocast_ctx:
            visual_output = model(image, return_phase4_features=True)
            h6_batch = model.h6.build_batch(
                model, cfg.get("dataset", "VisA"), class_names, visual_output, hybrid_alpha=1.0
            )
            seg_features = torch.stack(visual_output["seg_tokens"], dim=0)
            det_features = torch.stack(visual_output["det_tokens"], dim=0)
            text_global = get_phase2b_global_text_features(
                model, cfg.get("dataset", "VisA"), class_names, device,
                use_hybrid_soft_prompt=False, use_soft_prompt=True,
            ).to(dtype=det_features.dtype)

            cls_pred = torch.stack([
                torch.matmul(det_features[level].unsqueeze(1), text_global[level]).squeeze(1)
                for level in range(cfg["n_groups"])
            ], dim=0).mean(dim=0)
            cls_loss = F.cross_entropy(cls_pred.float(), label)

            seg_pred, base_group_logits, base_abnormal_minus_normal = (
                model.vision_text_fusion_gate_seg(
                    seg_features, text_global, img_size=cfg["img_size"],
                    h6_patch_logits=h6_batch["h6_logits"], return_details=True,
                )
            )
            base_abnormal_minus_normal = base_abnormal_minus_normal.detach()
            seg_loss = calculate_seg_loss(seg_pred.float(), mask.float())
            task_loss = cls_loss + seg_loss

            P = h6_batch["h6_logits"].shape[-1]
            q_role, hard_role, mask_coverage, local_valid_patch, local_valid_image = build_semantic_roles(
                mask, label, patch_count=P, local_mask_valid=local_mask_valid,
            )

            h6_route = active_role_balanced_router_loss(
                h6_batch["dense_probabilities"], q_role, hard_role, local_valid_patch
            )
            h6_factor_role = factor_specific_residual_role_loss(
                h6_batch["rho_scaled_factor_correction"], q_role, hard_role,
                mask_coverage, local_valid_patch, base_abnormal_minus_normal,
            )
            h6_actual_local = actual_local_residual_loss(
                h6_batch["rho_scaled_actual_correction"], q_role, hard_role,
                mask_coverage, local_valid_patch, base_abnormal_minus_normal,
            )

            total_loss = (
                task_loss +
                cfg["lambda_h6_route"] * h6_route +
                cfg["lambda_h6_factor_role"] * h6_factor_role +
                cfg["lambda_h6_actual_local"] * h6_actual_local
            )

        assert torch.isfinite(total_loss), f"Non-finite loss at microbatch {batch_idx}"

        # Scaled backward
        scaler.scale(total_loss / 6.0).backward()

        if batch_idx == 6:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_occurred = True

        allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        rss_mb = get_rss_mb()

        rec = {
            "microbatch": batch_idx,
            "loss": float(total_loss.item()),
            "allocated_mb": float(allocated_mb),
            "reserved_mb": float(reserved_mb),
            "peak_mb": float(peak_mb),
            "rss_mb": float(rss_mb),
        }
        microbatch_records.append(rec)
        print(f"  Microbatch {batch_idx}/6: loss={total_loss.item():.4f} | VRAM allocated={allocated_mb:.1f}MB reserved={reserved_mb:.1f}MB peak={peak_mb:.1f}MB | RSS={rss_mb:.1f}MB")

    t1 = time.time()

    peak_reserved_mb = max(r["reserved_mb"] for r in microbatch_records)
    peak_allocated_mb = max(r["allocated_mb"] for r in microbatch_records)
    peak_gpu_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)

    gpu_headroom_pct = (1.0 - (peak_gpu_reserved_mb / total_gpu_mem_mb)) * 100.0
    end_rss_mb = get_rss_mb()
    end_available_ram_gb = get_available_ram_gb()

    probe_result = {
        "device_name": torch.cuda.get_device_name(device),
        "total_gpu_mem_mb": float(total_gpu_mem_mb),
        "microbatches": microbatch_records,
        "optimizer_step_occurred": optimizer_step_occurred,
        "peak_allocated_mb": float(peak_allocated_mb),
        "peak_reserved_mb": float(peak_reserved_mb),
        "peak_gpu_reserved_mb": float(peak_gpu_reserved_mb),
        "gpu_headroom_pct": float(gpu_headroom_pct),
        "start_rss_mb": float(start_rss),
        "end_rss_mb": float(end_rss_mb),
        "start_available_ram_gb": float(start_available_ram),
        "end_available_ram_gb": float(end_available_ram_gb),
        "probe_duration_sec": float(t1 - t0),
        "train_gate_passed": bool(gpu_headroom_pct >= 15.0 and optimizer_step_occurred),
    }

    with open(_OUTPUT_PATH, "w") as f:
        json.dump(probe_result, f, indent=2)

    print(f"\n=== Controlled Train Memory Probe Completed ===")
    print(f"Total GPU Memory: {total_gpu_mem_mb:.1f} MB")
    print(f"Peak Reserved GPU Memory: {peak_gpu_reserved_mb:.1f} MB ({peak_gpu_reserved_mb/total_gpu_mem_mb*100:.1f}%)")
    print(f"GPU Headroom: {gpu_headroom_pct:.2f}% (Requirement >= 15.0%)")
    print(f"Host RSS: start={start_rss:.1f}MB, end={end_rss_mb:.1f}MB")
    print(f"Optimizer Step Occurred: {optimizer_step_occurred}")
    print(f"Probe Result Saved to: {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
