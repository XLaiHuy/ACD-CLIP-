#!/usr/bin/env python3
"""B7 Source-Exact One-Batch Dry Run for H6 Progress 1 v8.2 Candidate 1.

ONE_BATCH_SANITY_ESTIMATE_ONLY — NO OPTIMIZER STEP — NOT FOR TRAINING

This script mirrors the exact forward-pass pattern from train.py
for train_h6_progress1, but:
  1. Runs exactly ONE batch only.
  2. Does not call optimizer.step().
  3. Uses torch.autograd.grad for gradient probes (does not mutate .grad).
  4. Reports correction capacity, role statistics, and gradient connectivity.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset import get_text_and_image_dataset
from model.clip import create_model
from model.adapter import ACDCLIP
from utils import get_phase2b_global_text_features, calculate_seg_loss
from model.h6.losses import (
    active_role_balanced_router_loss,
    actual_local_residual_loss,
    build_semantic_roles,
    factor_specific_residual_role_loss,
    get_desired_correction,
)

_CONFIG_PATH = "configs/phase4/p1_v8_2_candidate1.json"
_RUN_DIR = "runs/phase4/p1_v8_2_iteration/B_final"


def load_and_validate_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    # Strict validation
    required = [
        "schema_version", "n_groups", "rho_values", "rho_trainable",
        "local_factor_mode", "correction_max", "h6_logit_temperature",
        "correction_epsilon", "residual_loss_beta", "role_morphology_config",
    ]
    for field in required:
        if field not in cfg:
            raise ValueError(f"[STRICT] Missing required config field: {field}")
    rho = cfg["rho_values"]
    if any(v == 0.0 for v in rho):
        raise ValueError(f"[STRICT] rho_values must not contain 0; got {rho}")
    if len(rho) != cfg["n_groups"]:
        raise ValueError(f"[STRICT] rho_values length {len(rho)} != n_groups {cfg['n_groups']}")
    if cfg.get("rho_trainable", True):
        raise ValueError("[STRICT] rho_trainable must be false")
    switches = cfg.get("Candidate-1_objective_switches", {})
    for key in ["load_bias", "balance", "cluster", "functional_diversity",
                "router_teacher", "center_losses", "experts"]:
        if switches.get(key, False):
            raise ValueError(f"[STRICT] {key} must be disabled in Candidate-1")
    return cfg


def quantiles(t: torch.Tensor) -> dict:
    t = t.detach().float().flatten().cpu()
    if t.numel() == 0:
        return {"min": float("nan"), "p01": float("nan"), "p05": float("nan"),
                "p50": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    qs = torch.quantile(t, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99])).tolist()
    return {"min": t.min().item(), "p01": qs[0], "p05": qs[1],
            "p50": qs[2], "p95": qs[3], "p99": qs[4], "max": t.max().item()}



def grad_probe(loss: torch.Tensor, params: list, label: str) -> dict:
    """Probe gradient connectivity without mutating .grad."""
    try:
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True
        )
        valid = [g for g in grads if g is not None]
        finite = [g for g in valid if torch.isfinite(g).all()]
        nonzero = [g for g in finite if g.abs().max() > 0]
        l2 = torch.cat([g.detach().flatten() for g in finite]).norm().item() if finite else 0.0
        maxabs = max(g.abs().max().item() for g in finite) if finite else 0.0
        print(f"  [{label}] total={len(grads)} valid={len(valid)} finite={len(finite)} "
              f"nonzero={len(nonzero)} L2={l2:.6f} maxabs={maxabs:.8f}")
        return {
            "total_params": len(grads),
            "valid": len(valid),
            "finite": len(finite),
            "nonzero": len(nonzero),
            "l2_norm": l2,
            "max_abs": maxabs,
            "connected": len(nonzero) > 0,
        }
    except Exception as e:
        print(f"  [{label}] ERROR: {e}")
        return {"connected": False, "error": str(e)}


def main():
    print("=" * 70)
    print("B7 ONE-BATCH DRY RUN")
    print("ONE_BATCH_SANITY_ESTIMATE_ONLY — NOT_FOR_TRAINING")
    print("=" * 70)

    os.makedirs(_RUN_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    cfg = load_and_validate_config(_CONFIG_PATH)
    canonical_json = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical_json.encode()).hexdigest()

    G = cfg["n_groups"]
    rho_cfg = cfg["rho_values"]
    T = cfg["h6_logit_temperature"]
    correction_max = cfg["correction_max"]
    correction_epsilon = cfg["correction_epsilon"]
    residual_beta = cfg["residual_loss_beta"]
    core_threshold = cfg["role_morphology_config"]["core_threshold"]
    boundary_threshold = cfg["role_morphology_config"]["boundary_threshold"]
    dataset_name = cfg.get("dataset", "VisA")

    theoretical_capacity = 2.0 * T * rho_cfg[0]
    print(f"[CONFIG] hash={config_hash[:16]}..., n_groups={G}, dataset={dataset_name}")
    print(f"[CONFIG] T={T}, rho={rho_cfg}")
    print(f"[CAPACITY] theoretical ±{theoretical_capacity:.4f} | correction_max={correction_max}")
    if correction_max > theoretical_capacity:
        print(f"[WARN] correction_max={correction_max} exceeds theoretical capacity {theoretical_capacity:.4f}")
    else:
        print(f"[OK] correction_max={correction_max} <= theoretical capacity {theoretical_capacity:.4f}")

    # -------------------------------------------------------------------------
    # Model initialization (fresh OpenAI)
    # -------------------------------------------------------------------------
    t0 = time.time()
    precision = cfg.get("precision", "bf16")
    torch_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    print(f"\n[INIT] Loading fresh ViT-L-14-336 (OpenAI) ...")
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
    print(f"[INIT] Model ready in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------------
    # Freeze rho
    # -------------------------------------------------------------------------
    rho_raw = dict(model.named_parameters()).get("h6.rho.raw", None)
    if rho_raw is not None:
        rho_raw.requires_grad_(False)
    actual_rho = model.h6.rho_values().detach().cpu()
    rho_frozen = (rho_raw is None) or (not rho_raw.requires_grad)
    print(f"[RHO] values={actual_rho.tolist()}, requires_grad=False={rho_frozen}")
    assert torch.all(actual_rho > 0), f"rho must be positive, got {actual_rho}"
    assert rho_frozen, "rho.raw must be frozen (requires_grad=False)"

    # -------------------------------------------------------------------------
    # One batch from canonical source-training data
    # -------------------------------------------------------------------------
    torch.manual_seed(cfg.get("seed", 0))
    dataset = get_text_and_image_dataset(dataset_name, img_size=cfg["img_size"], stage="train")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=cfg.get("batch_size", 1), shuffle=True, drop_last=True)
    batch = next(iter(loader))

    image = batch["image"].to(device)
    mask = batch["mask"].to(device)
    label = batch["label"].to(device)
    local_mask_valid = batch.get("local_mask_valid", torch.ones_like(mask)).to(device)
    class_names = list(batch["class_name"])
    B = image.shape[0]
    print(f"\n[BATCH] B={B}, classes={class_names}, labels={label.tolist()}")
    print(f"[BATCH] image={image.shape}, mask={mask.shape}")

    # -------------------------------------------------------------------------
    # Forward pass (mirrors train.py:1286-1344)
    # -------------------------------------------------------------------------
    t_fwd = time.time()
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
            use_hybrid_soft_prompt=False,  # hard_anchor mode
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
        # As in production: detach base after capture
        base_abnormal_minus_normal = base_abnormal_minus_normal.detach()
        seg_loss = calculate_seg_loss(seg_pred.float(), mask.float())
        task_loss = cls_loss + seg_loss

    t_fwd_end = time.time()

    # Shape info
    P = h6_batch["h6_logits"].shape[-1]
    M = h6_batch["dense_probabilities"].shape[-1]
    print(f"\n[SHAPES] G={G}, B={B}, P={P}, M={M}")

    # Tensor finiteness checks
    for key in ["factor_patch_logits", "h6_logits", "dense_probabilities",
                "rho_scaled_factor_correction", "rho_scaled_actual_correction",
                "factor_bank"]:
        t = h6_batch.get(key)
        if isinstance(t, torch.Tensor):
            ok = torch.isfinite(t).all().item()
            print(f"  {'[OK]' if ok else '[FAIL]'} {key}: finite={ok}, shape={tuple(t.shape)}")

    # -------------------------------------------------------------------------
    # Semantic roles
    # -------------------------------------------------------------------------
    q_role, hard_role, mask_coverage, local_valid_patch, local_valid_image = build_semantic_roles(
        mask, label, patch_count=P, local_mask_valid=local_mask_valid,
        core_threshold=core_threshold, boundary_threshold=boundary_threshold,
    )
    role_counts = [(hard_role == m).sum().item() for m in range(M)]
    print(f"\n[ROLES] counts by role: {role_counts}")
    print(f"[ROLES] local_valid_image: {local_valid_image.tolist()}")
    print(f"[ROLES] valid_patches: {local_valid_patch.sum().item()}/{B*P}")

    # -------------------------------------------------------------------------
    # Losses
    # -------------------------------------------------------------------------
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

    losses = {
        "task": task_loss.item(), "cls": cls_loss.item(), "seg": seg_loss.item(),
        "route": h6_route.item(), "factor_role": h6_factor_role.item(),
        "actual_local": h6_actual_local.item(),
    }
    print(f"\n[LOSSES]")
    for name, val in losses.items():
        finite = math.isfinite(val)
        print(f"  {'[OK]' if finite else '[FAIL]'} {name} = {val:.6f}")
    assert all(math.isfinite(v) for v in losses.values()), "All losses must be finite"

    # -------------------------------------------------------------------------
    # Correction capacity audit
    # -------------------------------------------------------------------------
    rsc = h6_batch["rho_scaled_actual_correction"].detach().float()
    rsf = h6_batch["rho_scaled_factor_correction"].detach().float()
    desired = get_desired_correction(
        mask_coverage, base_abnormal_minus_normal,
        correction_max=correction_max, epsilon=correction_epsilon,
    )
    q_actual = quantiles(rsc)
    q_factor = quantiles(rsf)
    q_desired = quantiles(desired)
    saturation_hi = (rsc.flatten().abs() >= theoretical_capacity * 0.95).float().mean().item()
    print(f"\n[CAPACITY AUDIT] theoretical ±{theoretical_capacity:.4f}")
    print(f"  actual_correction: {q_actual}")
    print(f"  factor_correction: {q_factor}")
    print(f"  desired_correction: {q_desired}")
    print(f"  saturation rate (|c|>=0.95*cap): {saturation_hi:.4f}")

    # Clamp rate by role
    clamp_rates = {}
    for role_idx in range(M):
        role_mask = (hard_role == role_idx) & local_valid_patch
        if role_mask.any():
            role_actual = rsc[role_mask.unsqueeze(0).expand(G, B, P)].flatten()
            sat = (role_actual.abs() >= theoretical_capacity * 0.95).float().mean().item()
            clamp_rates[f"role_{role_idx}"] = sat
    print(f"  clamp_rate_by_role: {clamp_rates}")

    # -------------------------------------------------------------------------
    # Router diagnostics
    # -------------------------------------------------------------------------
    dense_probs = h6_batch["dense_probabilities"].detach().float()
    router_entropy = -(dense_probs.clamp_min(1e-8).log() * dense_probs).sum(dim=-1).mean().item()
    router_usage = dense_probs.mean(dim=(0, 1, 2)).cpu().tolist()
    print(f"\n[ROUTER] entropy={router_entropy:.4f}, usage={[f'{u:.4f}' for u in router_usage]}")

    # Factor pairwise cosine (anomaly directions)
    if isinstance(h6_batch.get("factor_bank"), torch.Tensor):
        fb = h6_batch["factor_bank"].detach().float()  # [G, B, M, D, S]
        abn_dir = F.normalize((fb[..., 1] - fb[..., 0]), dim=-1)
        k = abn_dir[0, 0]  # [M, D]
        cos_mat = torch.mm(k, k.t())  # [M, M]
        n = M
        off = cos_mat[~torch.eye(n, dtype=torch.bool)]
        print(f"[FACTORS] anomaly-direction pairwise cosine: mean={off.abs().mean():.4f} max={off.abs().max():.4f}")

    # -------------------------------------------------------------------------
    # Gradient probes (retain_graph, no .grad mutation)
    # -------------------------------------------------------------------------
    print(f"\n[GRADIENTS] Computing gradient probes (retain_graph=True)...")
    router_params = list(model.h6.router.parameters())
    core_params = list(model.h6.semantic_core.parameters())
    img_adapter_params = list(model.image_adapter.parameters())

    # All losses must be computed in float for autograd
    h6_route_f = h6_route.float()
    h6_factor_f = h6_factor_role.float()
    h6_actual_f = h6_actual_local.float()
    task_f = task_loss.float()

    grad_results = {}
    grad_results["route_to_router"] = grad_probe(h6_route_f, router_params, "route→router")
    grad_results["factor_to_core"] = grad_probe(h6_factor_f, core_params, "factor→semantic_core")
    grad_results["actual_to_router"] = grad_probe(h6_actual_f, router_params, "actual→router")
    grad_results["actual_to_core"] = grad_probe(h6_actual_f, core_params, "actual→semantic_core")
    grad_results["task_to_adapter"] = grad_probe(task_f, img_adapter_params, "task→image_adapter")

    # rho must have no gradient
    if rho_raw is not None:
        print(f"  [rho] requires_grad={rho_raw.requires_grad}, grad={'None (OK)' if rho_raw.grad is None else 'NOT None (FAIL)'}")

    # -------------------------------------------------------------------------
    # Lambda estimates (ONE_BATCH_SANITY_ESTIMATE_ONLY)
    # -------------------------------------------------------------------------
    task_v, route_v, factor_v, actual_v = losses["task"], losses["route"], losses["factor_role"], losses["actual_local"]
    lambda_route = (0.015 * task_v / route_v) if route_v > 1e-9 else None
    lambda_factor = (0.020 * task_v / factor_v) if factor_v > 1e-9 else None
    lambda_actual = (0.015 * task_v / actual_v) if actual_v > 1e-9 else None
    print(f"\n[LAMBDA] ONE_BATCH_SANITY_ESTIMATE_ONLY — NOT_FOR_TRAINING")
    print(f"  lambda_route (1.5% target): {lambda_route}")
    print(f"  lambda_factor_role (2.0% target): {lambda_factor}")
    print(f"  lambda_actual_local (1.5% target): {lambda_actual}")

    # -------------------------------------------------------------------------
    # Performance
    # -------------------------------------------------------------------------
    fwd_time = t_fwd_end - t_fwd
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0
    print(f"\n[PERF] Forward: {fwd_time:.2f}s, GPU peak: {gpu_mem:.0f} MB")

    # -------------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------------
    report = {
        "ONE_BATCH_SANITY_ESTIMATE_ONLY": True,
        "NOT_FOR_TRAINING": True,
        "config_hash": config_hash,
        "G": G, "B": B, "P": P, "M": M,
        "rho_values": actual_rho.tolist(),
        "rho_frozen": rho_frozen,
        "h6_logit_temperature": T,
        "theoretical_capacity": theoretical_capacity,
        "correction_max": correction_max,
        "losses": losses,
        "gradients": grad_results,
        "correction_capacity": {
            "actual_correction_quantiles": q_actual,
            "factor_correction_quantiles": q_factor,
            "desired_correction_quantiles": q_desired,
            "saturation_rate": saturation_hi,
            "clamp_rate_by_role": clamp_rates,
        },
        "router": {
            "entropy": router_entropy,
            "mean_usage": router_usage,
        },
        "role_counts": role_counts,
        "lambda_estimates": {
            "ONE_BATCH_SANITY_ESTIMATE_ONLY": True,
            "NOT_FOR_TRAINING": True,
            "lambda_route": lambda_route,
            "lambda_factor_role": lambda_factor,
            "lambda_actual_local": lambda_actual,
        },
        "perf": {"fwd_time_s": fwd_time, "gpu_peak_mb": gpu_mem},
    }

    report_path = os.path.join(_RUN_DIR, "B7_one_batch_dry_run.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[SAVED] {report_path}")

    # -------------------------------------------------------------------------
    # Gate check
    # -------------------------------------------------------------------------
    required_connected = [
        "route_to_router", "factor_to_core",
        "actual_to_router", "actual_to_core", "task_to_adapter",
    ]
    disconnected = [k for k in required_connected if not grad_results.get(k, {}).get("connected", False)]
    if disconnected:
        print(f"\n[WARN] Required gradient paths disconnected: {disconnected}")
        print(f"\nDECISION: FIX_GRADIENT_REACHABILITY")
    else:
        print(f"\n[OK] All required gradient paths are finite and nonzero")
        print(f"\nPART I GATE PASSED")

    print("\n" + "=" * 70)
    print("ONE_BATCH_SANITY_ESTIMATE_ONLY — COMPLETE")
    print("=" * 70)
    return report


if __name__ == "__main__":
    main()
