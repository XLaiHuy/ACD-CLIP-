#!/usr/bin/env python3
"""Deterministic Multi-Batch No-Step Calibration (Iteration C)

NO OPTIMIZER — NO PARAMETER UPDATES — NO LOSS.BACKWARD ACCUMULATION

Supports gradient accumulation window grouping (grad_accum_steps = 6)
to smooth out microbatch normal/anomaly variance across windowed iterations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
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
    get_desired_correction,
)
from utils import calculate_seg_loss, get_phase2b_global_text_features

_RUN_DIR = "runs/phase4/p1_v8_2_iteration/C_calibration"
_CONFIG_PATH = "configs/phase4/p1_v8_2_candidate1.json"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quantiles_dict(t: torch.Tensor | list[float] | np.ndarray) -> dict:
    if isinstance(t, torch.Tensor):
        arr = t.detach().float().flatten().cpu().numpy()
    elif isinstance(t, list):
        arr = np.array(t, dtype=np.float32)
    else:
        arr = t.astype(np.float32)
    if len(arr) == 0:
        return {"min": 0.0, "p01": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def grad_probe(loss: torch.Tensor, params: list, label: str) -> dict:
    try:
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True
        )
        valid = [g for g in grads if g is not None]
        finite = [g for g in valid if torch.isfinite(g).all()]
        nonzero = [g for g in finite if g.abs().max() > 0]
        l2 = torch.cat([g.detach().flatten() for g in finite]).norm().item() if finite else 0.0
        maxabs = max(g.abs().max().item() for g in finite) if finite else 0.0
        return {
            "total_params": len(grads),
            "valid": len(valid),
            "finite": len(finite),
            "nonzero": len(nonzero),
            "l2_norm": l2,
            "max_abs": maxabs,
            "connected": len(nonzero) > 0,
            "grads": grads,
        }
    except Exception as e:
        return {"connected": False, "error": str(e), "grads": []}


def grad_cosine(grads1: list, grads2: list) -> float | None:
    pairs = []
    for g1, g2 in zip(grads1, grads2):
        if g1 is not None and g2 is not None:
            pairs.append((g1.detach().flatten().float(), g2.detach().flatten().float()))
    if not pairs:
        return None
    v1 = torch.cat([p[0] for p in pairs])
    v2 = torch.cat([p[1] for p in pairs])
    n1 = v1.norm()
    n2 = v2.norm()
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(torch.dot(v1, v2) / (n1 * n2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=120, help="Total microbatch images")
    parser.add_argument("--grad-accum-steps", type=int, default=6, help="Gradient accumulation window size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=_CONFIG_PATH)
    parser.add_argument("--output-dir", type=str, default=_RUN_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config) as f:
        cfg = json.load(f)
    canonical_json = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical_json.encode()).hexdigest()

    G = cfg["n_groups"]
    rho_cfg = cfg["rho_values"]
    T = cfg["h6_logit_temperature"]
    correction_max = cfg["correction_max"]
    correction_epsilon = cfg["correction_epsilon"]
    dataset_name = cfg.get("dataset", "VisA")

    # 1. Model init
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
        n_groups=G,
        lora_rank=16,
        lora_alpha=2.0,
        conv_lora_rank=8,
        conv_lora_alpha=2.0,
        conv_kernel_size_list=[3, 5],
        dfg_mode=cfg.get("dfg_mode", "attn"),
        dfg_attn_dim=256,
        dfg_attn_tau=cfg.get("dfg_attn_tau", 8.0),
        use_ss2d_dfg=cfg.get("use_ss2d_dfg", True),
        dfg_ss2d_fusion=cfg.get("dfg_ss2d_fusion", "weight_residual"),
        dfg_beta=0.1,
        h6_progress=1,
    )
    model.train()
    model.to(device)
    model.clipmodel.eval()

    # Freeze rho
    rho_raw = dict(model.named_parameters()).get("h6.rho.raw", None)
    if rho_raw is not None:
        rho_raw.requires_grad_(False)

    # 2. Dataset & DataLoader (Deterministic)
    dataset = get_text_and_image_dataset(dataset_name, img_size=cfg["img_size"], stage="train")
    if isinstance(dataset, dict):
        dataset = torch.utils.data.ConcatDataset(list(dataset.values()))

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # Deterministic order
        drop_last=True,
    )

    probe_batches = {1, 6, 12, 30, 60, 120}
    sample_manifest = []
    batch_metrics = []
    gradient_probe_results = {}

    router_params = list(model.h6.router.parameters())
    core_params = list(model.h6.semantic_core.parameters())
    img_adapter_params = list(model.image_adapter.parameters())

    print(f"=== Iteration C Calibration ({args.num_images} images, grad_accum_steps={args.grad_accum_steps}) ===")
    t_start = time.time()

    total_images_processed = 0
    all_role_counts = [0, 0, 0, 0]

    for batch_idx, batch_data in enumerate(loader, start=1):
        if batch_idx > args.num_images:
            break

        image = batch_data["image"].to(device)
        mask = batch_data["mask"].to(device)
        label = batch_data["label"].to(device)
        local_mask_valid = batch_data.get("local_mask_valid", torch.ones_like(mask)).to(device)
        class_names = list(batch_data["class_name"])
        B = image.shape[0]
        total_images_processed += B

        sample_manifest.append({
            "batch_idx": batch_idx,
            "class_names": class_names,
            "label": label.tolist(),
            "has_mask_positive": (mask > 0).any().item(),
        })

        autocast_ctx = torch.autocast(
            "cuda" if device.type == "cuda" else "cpu",
            dtype=torch_dtype,
        )

        with autocast_ctx:
            visual_output = model(image, return_phase4_features=True)
            h6_batch = model.h6.build_batch(
                model, dataset_name, class_names, visual_output, hybrid_alpha=1.0
            )
            seg_features = torch.stack(visual_output["seg_tokens"], dim=0)
            det_features = torch.stack(visual_output["det_tokens"], dim=0)
            text_global = get_phase2b_global_text_features(
                model, dataset_name, class_names, device,
                use_hybrid_soft_prompt=False,
                use_soft_prompt=True,
            ).to(dtype=det_features.dtype)

            cls_pred = torch.stack([
                torch.matmul(det_features[level].unsqueeze(1), text_global[level]).squeeze(1)
                for level in range(G)
            ], dim=0).mean(dim=0)
            cls_loss = F.cross_entropy(cls_pred.float(), label)

            seg_pred, base_group_logits, base_abnormal_minus_normal = (
                model.vision_text_fusion_gate_seg(
                    seg_features, text_global, img_size=cfg["img_size"],
                    h6_patch_logits=h6_batch["h6_logits"],
                    return_details=True,
                )
            )
            base_abnormal_minus_normal = base_abnormal_minus_normal.detach()
            seg_loss = calculate_seg_loss(seg_pred.float(), mask.float())
            task_loss = cls_loss + seg_loss

        P = h6_batch["h6_logits"].shape[-1]
        M = h6_batch["dense_probabilities"].shape[-1]

        q_role, hard_role, mask_coverage, local_valid_patch, local_valid_image = build_semantic_roles(
            mask, label, patch_count=P, local_mask_valid=local_mask_valid,
            core_threshold=cfg["role_morphology_config"]["core_threshold"],
            boundary_threshold=cfg["role_morphology_config"]["boundary_threshold"],
        )
        role_counts = [(hard_role == m).sum().item() for m in range(M)]
        for m in range(M):
            all_role_counts[m] += role_counts[m]

        h6_route = active_role_balanced_router_loss(
            h6_batch["dense_probabilities"], q_role, hard_role, local_valid_patch
        )
        h6_factor_role = factor_specific_residual_role_loss(
            h6_batch["rho_scaled_factor_correction"], q_role, hard_role,
            mask_coverage, local_valid_patch, base_abnormal_minus_normal,
            correction_max=correction_max, epsilon=correction_epsilon,
        )
        h6_actual_local = actual_local_residual_loss(
            h6_batch["rho_scaled_actual_correction"], q_role, hard_role,
            mask_coverage, local_valid_patch, base_abnormal_minus_normal,
            correction_max=correction_max, epsilon=correction_epsilon,
        )

        rsc = h6_batch["rho_scaled_actual_correction"].detach().float()
        rsf = h6_batch["rho_scaled_factor_correction"].detach().float()
        desired = get_desired_correction(
            mask_coverage, base_abnormal_minus_normal,
            correction_max=correction_max, epsilon=correction_epsilon,
        )

        dense_probs = h6_batch["dense_probabilities"].detach().float()
        router_entropy = -(dense_probs.clamp_min(1e-8).log() * dense_probs).sum(dim=-1).mean().item()
        router_usage = dense_probs.mean(dim=(0, 1, 2)).cpu().tolist()

        batch_record = {
            "batch_idx": batch_idx,
            "batch_size": B,
            "role_counts": role_counts,
            "task_loss": float(task_loss.item()),
            "cls_loss": float(cls_loss.item()),
            "seg_loss": float(seg_loss.item()),
            "route_loss_raw": float(h6_route.item()),
            "factor_role_loss_raw": float(h6_factor_role.item()),
            "actual_local_loss_raw": float(h6_actual_local.item()),
            "router_entropy": router_entropy,
            "router_usage": router_usage,
            "actual_correction_quantiles": quantiles_dict(rsc),
            "factor_correction_quantiles": quantiles_dict(rsf),
            "desired_correction_quantiles": quantiles_dict(desired),
            "saturation_rate": float((rsc.flatten().abs() >= 1.0 * 0.95).float().mean().item()),
        }
        batch_metrics.append(batch_record)

        if batch_idx in probe_batches:
            task_f = task_loss.float()
            route_f = h6_route.float()
            factor_f = h6_factor_role.float()
            actual_f = h6_actual_local.float()

            p_route_router = grad_probe(route_f, router_params, f"B{batch_idx}_route→router")
            p_factor_core = grad_probe(factor_f, core_params, f"B{batch_idx}_factor→core")
            p_actual_router = grad_probe(actual_f, router_params, f"B{batch_idx}_actual→router")
            p_actual_core = grad_probe(actual_f, core_params, f"B{batch_idx}_actual→core")
            p_task_adapter = grad_probe(task_f, img_adapter_params, f"B{batch_idx}_task→adapter")
            p_aux_adapter = grad_probe(route_f + factor_f + actual_f, img_adapter_params, f"B{batch_idx}_aux→adapter")
            cos_task_aux_adapter = grad_cosine(p_task_adapter["grads"], p_aux_adapter["grads"])

            gradient_probe_results[f"batch_{batch_idx}"] = {
                "route_to_router": {k: v for k, v in p_route_router.items() if k != "grads"},
                "factor_to_core": {k: v for k, v in p_factor_core.items() if k != "grads"},
                "actual_to_router": {k: v for k, v in p_actual_router.items() if k != "grads"},
                "actual_to_core": {k: v for k, v in p_actual_core.items() if k != "grads"},
                "task_to_adapter": {k: v for k, v in p_task_adapter.items() if k != "grads"},
                "aux_to_adapter": {k: v for k, v in p_aux_adapter.items() if k != "grads"},
                "cos_task_aux_adapter": cos_task_aux_adapter,
            }

        if batch_idx % 20 == 0 or batch_idx == args.num_images:
            print(f"  [Image {batch_idx}/{args.num_images}] task={task_loss.item():.4f} "
                  f"route={h6_route.item():.4f} factor={h6_factor_role.item():.6f} actual={h6_actual_local.item():.6f}")

    t_end = time.time()
    print(f"[COMPLETE] {args.num_images} images processed in {t_end-t_start:.1f}s")

    # -------------------------------------------------------------------------
    # 3. Window-Level Metric Aggregation (grad_accum_steps = 6)
    # -------------------------------------------------------------------------
    K = args.grad_accum_steps
    num_windows = total_images_processed // K
    window_metrics = []

    for w_idx in range(num_windows):
        w_batches = batch_metrics[w_idx * K : (w_idx + 1) * K]
        w_task = np.mean([b["task_loss"] for b in w_batches])
        w_route = np.mean([b["route_loss_raw"] for b in w_batches])
        w_factor = np.mean([b["factor_role_loss_raw"] for b in w_batches])
        w_actual = np.mean([b["actual_local_loss_raw"] for b in w_batches])
        window_metrics.append({
            "window_idx": w_idx + 1,
            "task_loss": float(w_task),
            "route_loss_raw": float(w_route),
            "factor_role_loss_raw": float(w_factor),
            "actual_local_loss_raw": float(w_actual),
        })

    # Window-level loss vectors
    w_task_arr = np.array([w["task_loss"] for w in window_metrics])
    w_route_arr = np.array([w["route_loss_raw"] for w in window_metrics])
    w_factor_arr = np.array([w["factor_role_loss_raw"] for w in window_metrics])
    w_actual_arr = np.array([w["actual_local_loss_raw"] for w in window_metrics])

    target_route_share = 0.015
    target_factor_share = 0.020
    target_actual_share = 0.015

    mean_task_w = float(np.mean(w_task_arr))
    mean_route_w = float(np.mean(w_route_arr))
    mean_factor_w = float(np.mean(w_factor_arr))
    mean_actual_w = float(np.mean(w_actual_arr))

    lambda_route_mean = (target_route_share * mean_task_w / mean_route_w) if mean_route_w > 1e-12 else 0.0
    lambda_factor_mean = (target_factor_share * mean_task_w / mean_factor_w) if mean_factor_w > 1e-12 else 0.0
    lambda_actual_mean = (target_actual_share * mean_task_w / mean_actual_w) if mean_actual_w > 1e-12 else 0.0

    # Split-half window stability analysis (first half of windows vs second half)
    n_half_w = num_windows // 2
    h1_w_task = np.mean(w_task_arr[:n_half_w])
    h2_w_task = np.mean(w_task_arr[n_half_w:])

    h1_route_lam = (target_route_share * h1_w_task / np.mean(w_route_arr[:n_half_w]))
    h2_route_lam = (target_route_share * h2_w_task / np.mean(w_route_arr[n_half_w:]))

    h1_factor_lam = (target_factor_share * h1_w_task / np.mean(w_factor_arr[:n_half_w]))
    h2_factor_lam = (target_factor_share * h2_w_task / np.mean(w_factor_arr[n_half_w:]))

    h1_actual_lam = (target_actual_share * h1_w_task / np.mean(w_actual_arr[:n_half_w]))
    h2_actual_lam = (target_actual_share * h2_w_task / np.mean(w_actual_arr[n_half_w:]))

    diff_route_pct = abs(h1_route_lam - h2_route_lam) / max(lambda_route_mean, 1e-8) * 100.0
    diff_factor_pct = abs(h1_factor_lam - h2_factor_lam) / max(lambda_factor_mean, 1e-8) * 100.0
    diff_actual_pct = abs(h1_actual_lam - h2_actual_lam) / max(lambda_actual_mean, 1e-8) * 100.0

    # Weighted shares using window-level selected lambdas
    w_route_shares = lambda_route_mean * w_route_arr
    w_factor_shares = lambda_factor_mean * w_factor_arr
    w_actual_shares = lambda_actual_mean * w_actual_arr
    total_aux_w = w_route_shares + w_factor_shares + w_actual_shares
    total_share_w = total_aux_w / w_task_arr * 100.0

    # Save manifest
    with open(os.path.join(args.output_dir, "sample_manifest.json"), "w") as f:
        json.dump(sample_manifest, f, indent=2)

    # Save batch_metrics.jsonl
    with open(os.path.join(args.output_dir, "batch_metrics.jsonl"), "w") as f:
        for bm in batch_metrics:
            f.write(json.dumps(bm) + "\n")

    # Save gradient metrics
    with open(os.path.join(args.output_dir, "gradient_metrics.json"), "w") as f:
        json.dump(gradient_probe_results, f, indent=2)

    calib_summary = {
        "num_images": total_images_processed,
        "grad_accum_steps": K,
        "num_windows": num_windows,
        "config_hash": config_hash,
        "all_role_counts": all_role_counts,
        "raw_window_losses": {
            "task": quantiles_dict(w_task_arr),
            "route": quantiles_dict(w_route_arr),
            "factor_role": quantiles_dict(w_factor_arr),
            "actual_local": quantiles_dict(w_actual_arr),
        },
        "selected_lambdas": {
            "lambda_route": lambda_route_mean,
            "lambda_factor_role": lambda_factor_mean,
            "lambda_actual_local": lambda_actual_mean,
        },
        "split_half_stability": {
            "route_h1": float(h1_route_lam), "route_h2": float(h2_route_lam), "diff_route_pct": float(diff_route_pct),
            "factor_h1": float(h1_factor_lam), "factor_h2": float(h2_factor_lam), "diff_factor_pct": float(diff_factor_pct),
            "actual_h1": float(h1_actual_lam), "actual_h2": float(h2_actual_lam), "diff_actual_pct": float(diff_actual_pct),
            "stable_under_20pct": bool(max(diff_route_pct, diff_factor_pct, diff_actual_pct) <= 20.0),
        },
        "weighted_shares_pct": {
            "mean_total_share": float(np.mean(total_share_w)),
            "median_total_share": float(np.median(total_share_w)),
            "p05_total_share": float(np.percentile(total_share_w, 5)),
            "p95_total_share": float(np.percentile(total_share_w, 95)),
            "max_total_share": float(np.max(total_share_w)),
        },
    }

    with open(os.path.join(args.output_dir, "calibration_summary.json"), "w") as f:
        json.dump(calib_summary, f, indent=2)

    # -------------------------------------------------------------------------
    # 4. Evaluate Calibration Gates
    # -------------------------------------------------------------------------
    all_finite = np.isfinite(w_task_arr).all() and np.isfinite(w_route_arr).all() and np.isfinite(w_factor_arr).all()
    required_probe_keys = ["route_to_router", "factor_to_core", "actual_to_core"]
    grad_ok = all(b_probes[req_k]["connected"] for b_probes in gradient_probe_results.values() for req_k in required_probe_keys)
    role_ok = (all_role_counts[0] > 0) and ((all_role_counts[1] + all_role_counts[2] + all_role_counts[3]) > 0)

    mean_share = float(np.mean(total_share_w))
    p95_share = float(np.percentile(total_share_w, 95))
    shares_ok = (4.0 <= mean_share <= 6.0) and (p95_share <= 8.5)
    stable_ok = calib_summary["split_half_stability"]["stable_under_20pct"]

    if not all_finite:
        decision = "FIX_NUMERICAL_STABILITY"
    elif not grad_ok:
        decision = "FIX_GRADIENT_REACHABILITY"
    elif not role_ok:
        decision = "FIX_ROLE_SUPPORT"
    elif not stable_ok or not shares_ok:
        decision = "FIX_LOSS_CALIBRATION"
    else:
        decision = "READY_FOR_ITERATION_D"

    print(f"\nPART II DECISION: {decision}")

    # If decision is READY_FOR_ITERATION_D, update resolved config file!
    if decision == "READY_FOR_ITERATION_D":
        cfg["calibration_decision"] = "READY_FOR_ITERATION_D"
        cfg["lambda_h6_route"] = lambda_route_mean
        cfg["lambda_h6_factor_role"] = lambda_factor_mean
        cfg["lambda_h6_actual_local"] = lambda_actual_mean
        cfg["grad_accum_steps"] = K

        with open(args.config, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[CONFIG UPDATED] Approved resolved config written to {args.config}")

    # -------------------------------------------------------------------------
    # Write CALIBRATION_REPORT.md
    # -------------------------------------------------------------------------
    report_md = f"""# Iteration C Calibration Report (Windowed Accumulation = {K})

**Dataset**: {dataset_name}  
**Images**: {total_images_processed} ({num_windows} windows of {K} microbatches)  
**Config Hash**: `{config_hash}`  
**Decision**: `{decision}`  

## 1. Summary Statistics (Window-Level)

| Metric | Task Loss | Route Loss | Factor Role Loss | Actual Local Loss |
|---|---|---|---|---|
| **Mean** | {quantiles_dict(w_task_arr)['p50']:.4f} | {quantiles_dict(w_route_arr)['p50']:.4f} | {quantiles_dict(w_factor_arr)['p50']:.6f} | {quantiles_dict(w_actual_arr)['p50']:.6f} |
| **P05** | {quantiles_dict(w_task_arr)['p05']:.4f} | {quantiles_dict(w_route_arr)['p05']:.4f} | {quantiles_dict(w_factor_arr)['p05']:.6f} | {quantiles_dict(w_actual_arr)['p05']:.6f} |
| **P95** | {quantiles_dict(w_task_arr)['p95']:.4f} | {quantiles_dict(w_route_arr)['p95']:.4f} | {quantiles_dict(w_factor_arr)['p95']:.6f} | {quantiles_dict(w_actual_arr)['p95']:.6f} |

## 2. Calibrated Lambda Coefficients

- `lambda_route` (1.5% target): `{lambda_route_mean:.6f}`
- `lambda_factor_role` (2.0% target): `{lambda_factor_mean:.6f}`
- `lambda_actual_local` (1.5% target): `{lambda_actual_mean:.6f}`

### Split-Half Window Stability Analysis (First 10 vs Second 10 Windows)
- Route Lambda H1/H2: `{h1_route_lam:.6f}` / `{h2_route_lam:.6f}` (diff: `{diff_route_pct:.2f}%`)
- Factor Lambda H1/H2: `{h1_factor_lam:.6f}` / `{h2_factor_lam:.6f}` (diff: `{diff_factor_pct:.2f}%`)
- Actual Lambda H1/H2: `{h1_actual_lam:.6f}` / `{h2_actual_lam:.6f}` (diff: `{diff_actual_pct:.2f}%`)
- **Stability Gate (<=20% diff)**: `{'PASSED' if stable_ok else 'FAILED'}`

## 3. Weighted Auxiliary Share Distribution

- Mean Total Auxiliary Share: `{mean_share:.2f}%` (target 5.0%, operational range 4.0%–6.0%)
- P95 Total Auxiliary Share: `{p95_share:.2f}%` (operational max <= 8.5%)

## 4. Role Support Breakdown

- Role 0 (Normal): `{all_role_counts[0]}` patches
- Role 1 (Outside Anomaly): `{all_role_counts[1]}` patches
- Role 2 (Core Anomaly): `{all_role_counts[2]}` patches
- Role 3 (Boundary Anomaly): `{all_role_counts[3]}` patches
- **Role Support Gate**: `{'PASSED' if role_ok else 'FAILED'}`

## 5. Decision & Approved State

Final Decision: `{decision}`
"""
    with open(os.path.join(args.output_dir, "CALIBRATION_REPORT.md"), "w") as f:
        f.write(report_md)

    print(f"[REPORT SAVED] {os.path.join(args.output_dir, 'CALIBRATION_REPORT.md')}")


if __name__ == "__main__":
    main()
