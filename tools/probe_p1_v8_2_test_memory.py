#!/usr/bin/env python3
"""Controlled Test Memory Probe for P1-v8.2.

Evaluates 2 test samples with Candidate-1 model pipeline, measures peak GPU and host RAM,
and verifies metric buffer release.
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

from dataset import DOMAINS, get_text_and_image_dataset
from model.adapter import ACDCLIP
from model.clip import create_model
from utils import get_phase2b_global_text_features


_CONFIG_PATH = "configs/phase4/p1_v8_2_candidate1.json"
_OUTPUT_PATH = "runs/phase4/p1_v8_2_full20_prelaunch_audit/test_memory_probe.json"


def get_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_available_ram_gb() -> float:
    return psutil.virtual_memory().available / (1024 * 1024 * 1024)


def main():
    print("=== Starting Controlled Test Memory Probe (2 Test Samples) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[ERROR] CUDA is not available. Cannot perform GPU test memory probe.")
        sys.exit(1)

    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)

    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    start_rss = get_rss_mb()
    total_gpu_mem_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)

    precision = cfg.get("precision", "bf16")
    clip_model = create_model(
        "ViT-L-14-336",
        img_size=cfg["img_size"],
        device=device,
        pretrained="openai",
        require_pretrained=True,
        precision=precision,
    )
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
    model.eval()
    model.to(device)

    dataset = get_text_and_image_dataset(cfg.get("dataset", "VisA"), img_size=cfg["img_size"], stage="test")
    if isinstance(dataset, dict):
        first_cat = list(dataset.keys())[0]
        dataset = dataset[first_cat]

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    t0 = time.time()
    samples_processed = 0
    image_preds = []
    torch_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader, start=1):
            if batch_idx > 2:
                break

            image = batch_data["image"].to(device, non_blocking=True)
            mask = batch_data["mask"].to(device, non_blocking=True)
            class_names = list(batch_data["class_name"])

            with torch.autocast("cuda", dtype=torch_dtype):
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

                cls_preds = [
                    torch.matmul(det_features[i].unsqueeze(1), text_global[i]).squeeze(1)
                    for i in range(det_features.shape[0])
                ]
                cls_preds = torch.stack(cls_preds, dim=0).mean(dim=0)
                pred_image = F.softmax(cls_preds, dim=1)[:, 1]

                seg_pred = model.vision_text_fusion_gate_seg(
                    seg_features, text_global, test_mode=True, domain=DOMAINS.get(cfg.get("dataset", "VisA"), "VisA"),
                    h6_patch_logits=h6_batch["h6_logits"],
                )

            flat_seg = torch.flatten(seg_pred, start_dim=1)
            pmax_pred, _ = torch.max(flat_seg, dim=1)
            pred_image = pred_image * 0.9 + pmax_pred * 0.1

            image_preds.append(pred_image.cpu())
            samples_processed += 1

    t1 = time.time()

    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    gpu_headroom_pct = (1.0 - (peak_reserved_mb / total_gpu_mem_mb)) * 100.0

    # Explicit buffer release
    del image_preds, model, clip_model
    torch.cuda.empty_cache()

    end_rss_mb = get_rss_mb()
    end_allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

    test_probe_result = {
        "device_name": torch.cuda.get_device_name(device),
        "total_gpu_mem_mb": float(total_gpu_mem_mb),
        "samples_processed": samples_processed,
        "peak_allocated_mb": float(peak_allocated_mb),
        "peak_reserved_mb": float(peak_reserved_mb),
        "end_allocated_mb": float(end_allocated_mb),
        "gpu_headroom_pct": float(gpu_headroom_pct),
        "start_rss_mb": float(start_rss),
        "end_rss_mb": float(end_rss_mb),
        "probe_duration_sec": float(t1 - t0),
        "test_gate_passed": bool(gpu_headroom_pct >= 15.0 and end_allocated_mb < 50.0),
    }

    with open(_OUTPUT_PATH, "w") as f:
        json.dump(test_probe_result, f, indent=2)

    print(f"\n=== Controlled Test Memory Probe Completed ===")
    print(f"Samples Processed: {samples_processed}")
    print(f"Peak Reserved GPU Memory: {peak_reserved_mb:.1f} MB ({peak_reserved_mb/total_gpu_mem_mb*100:.1f}%)")
    print(f"GPU Headroom: {gpu_headroom_pct:.2f}% (Requirement >= 15.0%)")
    print(f"After Cleanup VRAM Allocated: {end_allocated_mb:.1f} MB")
    print(f"Probe Result Saved to: {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
