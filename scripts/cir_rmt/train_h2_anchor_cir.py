#!/usr/bin/env python3
"""Train the exact historical H2 arm and its Anchor/CIR extensions.

The historical H2 repository is imported read-only.  This runner owns only
the extension branches and checkpoint plumbing; it never edits historical
source files.  R uses the historical native path, RA adds a train-only image
parameter anchor, and RCA adds the frozen CIR-V2 peer transport to the
training segmentation path.  All checkpoints are evaluated natively with
alpha=0 by the historical evaluator.
"""
from __future__ import annotations

import argparse
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
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "exact_h2_anchor_cir_master_v1.json"
DEFAULT_H2_REPO = Path("/home/ai4/caohuy/ACD-CLIP-base-new-phase1-h2-anchor-cir-20260901")


def _load_h2_modules(h2_repo: Path) -> dict[str, Any]:
    """Import all historical model/data helpers from the isolated H2 tree."""
    h2_repo = h2_repo.resolve()
    if not h2_repo.is_dir():
        raise FileNotFoundError(f"H2 repository/worktree not found: {h2_repo}")
    if str(h2_repo) not in sys.path:
        sys.path.insert(0, str(h2_repo))
    if str(ROOT) not in sys.path:
        sys.path.insert(1, str(ROOT))
    from dataset import TextAndImageDataset  # type: ignore
    from model.adapter import ACDCLIP  # type: ignore
    from model.clip import create_model  # type: ignore
    from train import (  # type: ignore
        compute_hybrid_k_regularization,
        get_dfg_beta_for_epoch,
        get_hybrid_alpha_for_epoch,
        get_optimizer_lr,
    )
    from utils import (  # type: ignore
        calculate_seg_loss,
        get_hybrid_soft_prompt_single_class_text_embedding,
    )
    return {
        "TextAndImageDataset": TextAndImageDataset,
        "ACDCLIP": ACDCLIP,
        "create_model": create_model,
        "compute_hybrid_k_regularization": compute_hybrid_k_regularization,
        "get_dfg_beta_for_epoch": get_dfg_beta_for_epoch,
        "get_hybrid_alpha_for_epoch": get_hybrid_alpha_for_epoch,
        "get_optimizer_lr": get_optimizer_lr,
        "calculate_seg_loss": calculate_seg_loss,
        "get_hybrid_soft_prompt_single_class_text_embedding": get_hybrid_soft_prompt_single_class_text_embedding,
    }


def _load_current_cir_primitives(h2_repo: Path) -> dict[str, Any]:
    """Load current CIR helpers after historical ``model`` imports finish.

    H2 and the current repository both expose a top-level ``model`` package.
    H2 must win while its classes are imported, but the current CIR package
    must win when its runtime is imported.  H2 class/function objects already
    retain their historical globals, so removing only H2's cached model
    modules before importing the current package is safe and keeps the two
    implementations isolated in one process.
    """
    h2_root = h2_repo.resolve()
    h2_model_names = [
        name for name, module in list(sys.modules.items())
        if (name == "model" or name.startswith("model."))
        and str(getattr(module, "__file__", "")).startswith(str(h2_root))
    ]
    for name in sorted(h2_model_names, key=len, reverse=True):
        sys.modules.pop(name, None)
    sys.path[:] = [entry for entry in sys.path if Path(entry or os.curdir).resolve() != h2_root]
    root_string = str(ROOT)
    sys.path[:] = [entry for entry in sys.path if entry != root_string]
    sys.path.insert(0, root_string)
    from tools.cir_rmt.core import cir_logits_from_native_weights, peer_delta_from_native_margins
    from tools.cir_rmt.parameter_anchor import ImageParameterAnchor

    return {
        "cir_logits_from_native_weights": cir_logits_from_native_weights,
        "peer_delta_from_native_margins": peer_delta_from_native_margins,
        "ImageParameterAnchor": ImageParameterAnchor,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNKNOWN"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def capture_rng() -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        # Historical DataLoader did not receive an explicit generator.  Its
        # sampler/worker seeds came from the process-global torch RNG.
        "dataloader_generator_state": torch.get_rng_state().clone(),
        "dataloader_generator_kind": "implicit_process_global_torch_rng_historical_h2",
    }


def restore_rng(payload: Mapping[str, Any]) -> None:
    if "python_random_state" in payload:
        random.setstate(payload["python_random_state"])
    if "numpy_random_state" in payload:
        np.random.set_state(payload["numpy_random_state"])
    if "torch_cpu_rng_state" in payload:
        torch.set_rng_state(payload["torch_cpu_rng_state"])
    if torch.cuda.is_available() and payload.get("torch_cuda_rng_state_all"):
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"refusing to overwrite symlink checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def build_model(cfg: Mapping[str, Any], modules: Mapping[str, Any], device: torch.device) -> Any:
    create_model = modules["create_model"]
    model_class = modules["ACDCLIP"]
    clip_model = create_model(
        model_name=str(cfg["model_name"]),
        img_size=int(cfg["img_size"]),
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    if bool(cfg["grad_checkpointing"]):
        clip_model.set_grad_checkpointing(True)
    clip_model.eval()
    model = model_class(
        clip_model=clip_model,
        n_groups=int(cfg["n_groups"]),
        image_adapt_weight=float(cfg["image_adapt_weight"]),
        conv_lora_rank=int(cfg["conv_lora_rank"]),
        conv_lora_alpha=float(cfg["conv_lora_alpha"]),
        conv_kernel_size_list=list(cfg["conv_kernel_size_list"]),
        text_adapt_weight=float(cfg["text_adapt_weight"]),
        lora_rank=int(cfg["lora_rank"]),
        lora_alpha=float(cfg["lora_alpha"]),
        dfg_mode=str(cfg["dfg_mode"]),
        dfg_attn_dim=int(cfg["dfg_attn_dim"]),
        dfg_attn_tau=float(cfg["dfg_attn_tau"]),
        use_ss2d_dfg=bool(cfg["use_ss2d_dfg"]),
        dfg_gamma_max=float(cfg["dfg_gamma_max"]),
        dfg_ss2d_fusion=str(cfg["dfg_ss2d_fusion"]),
        dfg_beta=float(cfg["dfg_beta"]),
        dfg_beta_schedule=str(cfg["dfg_beta_schedule"]),
        dfg_beta_target=float(cfg["dfg_beta_target"]),
        dfg_beta_current=float(cfg["dfg_beta"]),
        dfg_weight_residual_fp32=bool(cfg["dfg_weight_residual_fp32"]),
        # Historical H2 constructed the soft-prompt module with this flag,
        # then enabled its hybrid schedule on the model instance.
        use_soft_prompt=True,
        soft_prompt_ctx_len=int(cfg["soft_prompt_ctx_len"]),
        soft_prompt_init=str(cfg["soft_prompt_init"]),
        soft_prompt_init_phrase=str(cfg["soft_prompt_init_phrase"]),
    ).to(device)
    model.eval()
    model.use_soft_prompt = True
    model.use_hybrid_soft_prompt = True
    model.prompt_mode = "hybrid"
    model.hybrid_alpha_current = 0.0
    model.hybrid_alpha_max = float(cfg["hybrid_alpha_max"])
    model.soft_prompt_freeze_epochs = int(cfg["soft_prompt_freeze_epochs"])
    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.soft_prompt.requires_grad_(False)
    return model


def make_optimizer(model: Any, cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    # Preserve historical H2 group order: text, image, soft prompt.
    return torch.optim.Adam([
        {"name": "text_adapter", "params": model.text_adapter.parameters(), "lr": float(cfg["text_lr"])},
        {"name": "image_adapter", "params": model.image_adapter.parameters(), "lr": float(cfg["image_lr"])},
        {"name": "soft_prompt", "params": model.soft_prompt.parameters(), "lr": 0.0, "constant_lr": float(cfg["soft_prompt_lr"])},
    ])


def set_epoch_state(model: Any, optimizer: torch.optim.Optimizer, cfg: Mapping[str, Any], modules: Mapping[str, Any], epoch: int) -> tuple[float, float, bool]:
    alpha = modules["get_hybrid_alpha_for_epoch"](
        int(epoch), float(cfg["hybrid_alpha_max"]), int(cfg["soft_prompt_freeze_epochs"])
    )
    frozen = int(epoch) <= int(cfg["soft_prompt_freeze_epochs"])
    model.hybrid_alpha_current = float(alpha)
    model.soft_prompt.requires_grad_(not frozen)
    model.text_adapter.requires_grad_(True)
    for group in optimizer.param_groups:
        if group.get("name") == "soft_prompt":
            group["lr"] = 0.0 if frozen else float(group["constant_lr"])
    beta = modules["get_dfg_beta_for_epoch"](
        int(epoch), str(cfg["dfg_beta_schedule"]), float(cfg["dfg_beta_target"]), float(cfg["dfg_beta"])
    )
    model.set_dfg_beta(float(beta))
    return float(alpha), float(beta), bool(frozen)


def load_anchor(path: Path, model: Any, device: torch.device, anchor_class: Any) -> Any:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("image_adapter"), Mapping):
        raise ValueError(f"anchor checkpoint lacks image_adapter state: {path}")
    reference = {
        str(name): value.detach().float().cpu().clone()
        for name, value in payload["image_adapter"].items()
        if isinstance(value, torch.Tensor)
    }
    expected = {str(name): value for name, value in model.image_adapter.named_parameters()}
    if set(reference) != set(expected):
        raise ValueError("H2 anchor image_adapter parameter identity mismatch")
    for name, value in expected.items():
        if tuple(value.shape) != tuple(reference[name].shape):
            raise ValueError(f"H2 anchor shape mismatch for {name}")
    return anchor_class(
        reference,
        checkpoint_sha256=sha256_file(path),
        epoch=int(payload.get("epoch", -1)),
        config_sha256=None,
        device=device,
    )


def h2_dfg_weights(model: Any, img_feat: torch.Tensor, group_text: torch.Tensor, group_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact historical H2 attention weights for one stage."""
    v_gap = img_feat.mean(dim=1)
    v_ss2d = None
    if model.use_ss2d_dfg:
        v_ss2d = model.image_adapter["dfg_ss2d_branches"][group_index](img_feat)
    text_normal = group_text[..., 0]
    text_abnormal = group_text[..., 1]
    k_normal = model.image_adapter["vision_text_k"][group_index](text_normal)
    k_abnormal = model.image_adapter["vision_text_k"][group_index](text_abnormal)
    scale = (model.dfg_attn_dim ** 0.5) * model.dfg_attn_tau
    if model.use_ss2d_dfg and model.dfg_ss2d_fusion == "weight_residual":
        q_gap = model.image_adapter["vision_text_q"][group_index](v_gap)
        q_ss2d = model.image_adapter["vision_text_q"][group_index](v_ss2d)
        if model.dfg_weight_residual_fp32:
            q_gap_for_attn = q_gap.float()
            q_ss2d_for_attn = q_ss2d.float()
            k_normal_for_attn = k_normal.float()
            k_abnormal_for_attn = k_abnormal.float()
        else:
            q_gap_for_attn = q_gap
            q_ss2d_for_attn = q_ss2d
            k_normal_for_attn = k_normal
            k_abnormal_for_attn = k_abnormal
        scores_gap_normal = torch.einsum("bd,bnd->bn", q_gap_for_attn, k_normal_for_attn) / scale
        scores_gap_abnormal = torch.einsum("bd,bnd->bn", q_gap_for_attn, k_abnormal_for_attn) / scale
        scores_ss2d_normal = torch.einsum("bd,bnd->bn", q_ss2d_for_attn, k_normal_for_attn) / scale
        scores_ss2d_abnormal = torch.einsum("bd,bnd->bn", q_ss2d_for_attn, k_abnormal_for_attn) / scale
        weights_normal = (1.0 - model.dfg_beta) * F.softmax(scores_gap_normal, dim=1)
        weights_normal = weights_normal + model.dfg_beta * F.softmax(scores_ss2d_normal, dim=1)
        weights_abnormal = (1.0 - model.dfg_beta) * F.softmax(scores_gap_abnormal, dim=1)
        weights_abnormal = weights_abnormal + model.dfg_beta * F.softmax(scores_ss2d_abnormal, dim=1)
        return weights_normal.to(dtype=text_normal.dtype), weights_abnormal.to(dtype=text_abnormal.dtype)
    q = model.image_adapter["vision_text_q"][group_index](v_gap)
    weights_normal = F.softmax(torch.einsum("bd,bnd->bn", q, k_normal) / scale, dim=1)
    weights_abnormal = F.softmax(torch.einsum("bd,bnd->bn", q, k_abnormal) / scale, dim=1)
    return weights_normal, weights_abnormal


def h2_native_weights_logits(model: Any, seg_features: torch.Tensor, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    group_text = text_features.permute(1, 0, 2, 3)
    weights = []
    logits = []
    for stage in range(int(seg_features.shape[0])):
        normal, abnormal = h2_dfg_weights(model, seg_features[stage], group_text, stage)
        weights.append(torch.stack([normal, abnormal], dim=-1))
        fused_normal = torch.einsum("bn,bnd->bd", normal, group_text[..., 0])
        fused_abnormal = torch.einsum("bn,bnd->bd", abnormal, group_text[..., 1])
        fused = torch.stack([F.normalize(fused_normal, dim=-1), F.normalize(fused_abnormal, dim=-1)], dim=-1)
        logits.append(torch.matmul(10.0 * seg_features[stage], fused))
    return torch.stack(weights, dim=0), torch.stack(logits, dim=0)


def logits_to_training_probability(logits: torch.Tensor, img_size: int) -> torch.Tensor:
    maps = logits.permute(0, 1, 3, 2).reshape(logits.shape[0], logits.shape[1], 2, 37, 37)
    maps = F.interpolate(maps.reshape(-1, 2, 37, 37), size=(int(img_size), int(img_size)), mode="bilinear", align_corners=True)
    maps = maps.reshape(logits.shape[0], logits.shape[1], 2, int(img_size), int(img_size))
    return F.softmax(maps.mean(dim=0), dim=1)


def logits_to_deployment_probability(logits: torch.Tensor, img_size: int, domain: str) -> torch.Tensor:
    kernel, sigma = ((7, 1.0) if str(domain) == "Industrial" else (9, 1.5))
    maps = []
    for stage in range(int(logits.shape[0])):
        stage_map = logits[stage].permute(0, 2, 1).reshape(logits.shape[1], 2, 37, 37)
        stage_map = gaussian_blur2d(stage_map, (kernel, kernel), (sigma, sigma))
        maps.append(F.interpolate(stage_map, size=(int(img_size), int(img_size)), mode="bilinear", align_corners=True))
    return F.softmax(torch.stack(maps, dim=0).mean(dim=0), dim=1)


def cir_training_segmentation(model: Any, seg_features: torch.Tensor, text_features: torch.Tensor, cfg: Mapping[str, Any], core: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    native_weights, native_logits = h2_native_weights_logits(model, seg_features, text_features)
    group_text = text_features.permute(1, 0, 2, 3)
    visual = F.normalize(seg_features.float(), dim=-1)
    prompts = F.normalize(group_text.float(), dim=-2)
    similarities = torch.einsum("sbpd,bgdc->sbpgc", visual, prompts)
    group_margins = similarities[..., 1] - similarities[..., 0]
    delta, stats = core["peer_delta_from_native_margins"](
        seg_features.detach(), group_margins.detach(),
        peer_count=int(cfg["rmt_peer_count"]),
        spatial_radius=int(cfg["rmt_spatial_radius"]),
        eps=float(cfg["rmt_eps"]),
        mad_constant=float(cfg["rmt_mad_constant"]),
    )
    configured_score_mode = str(cfg.get("rmt_score_mode", "exact_score_space")).lower()
    score_mode = "optimized" if configured_score_mode == "exact_score_space" else configured_score_mode
    cir_logits, native_score_logits = core["cir_logits_from_native_weights"](
        seg_features, group_text, native_weights, delta,
        float(cfg["rmt_training_alpha"]),
        score_mode=score_mode,
        eps=float(cfg["rmt_eps"]),
        transport_direction=str(cfg["rmt_transport_direction"]),
    )
    values = {
        "peer_valid_fraction": float(stats["valid"].float().mean().detach().item()),
        "peer_candidate_mean": float(stats["candidate_count"].float().mean().detach().item()),
        "mad_mean": float(stats["mad"].float().mean().detach().item()),
        "z_abs_p95": float(stats["z"].float().abs().reshape(-1).quantile(0.95).detach().item()),
        "delta_abs_mean": float(delta.float().abs().mean().detach().item()),
        "delta_saturation_fraction": float((delta.float().abs() > 0.95).float().mean().detach().item()),
        "native_cir_score_max_abs_diff": float((native_logits.float() - native_score_logits.float()).abs().max().detach().item()),
        "transport_l1": float((cir_logits.float() - native_score_logits.float()).abs().mean().detach().item()),
    }
    if not bool(stats["valid"].all()):
        raise RuntimeError("H2 RCA encountered an invalid K=8 peer set")
    if not torch.isfinite(cir_logits).all():
        raise FloatingPointError("H2 RCA CIR logits are non-finite")
    return logits_to_training_probability(cir_logits, int(cfg["img_size"])), values


def checkpoint_payload(
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: StepLR,
    scaler: torch.cuda.amp.GradScaler,
    cfg: Mapping[str, Any],
    config_sha: str,
    arm: str,
    epoch: int,
    global_step: int,
    anchor: Any | None,
    anchor_lambda: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "h2_contract_sha256": str(config_sha),
        "git_sha": current_git_sha(),
        "architecture_freeze_sha256": str(cfg["architecture_freeze_sha256"]),
        "source_dataset": str(cfg["source_dataset"]),
        "source_root": str(Path(cfg["source_root"]).resolve()),
        "source_manifest": str((Path(cfg["h2_repo_path"]).resolve() / cfg["manifest_path"]).resolve()),
        "source_manifest_sha256": sha256_file((Path(cfg["h2_repo_path"]).resolve() / cfg["manifest_path"]).resolve()),
        "clip_asset": str((Path(cfg["h2_repo_path"]).resolve() / "model" / "ViT-L-14-336px.pt").resolve()),
        "clip_asset_sha256": sha256_file((Path(cfg["h2_repo_path"]).resolve() / "model" / "ViT-L-14-336px.pt").resolve()),
        "arm": str(arm),
        "n_groups": int(cfg["n_groups"]),
        "dfg_mode": str(cfg["dfg_mode"]),
        "dfg_attn_dim": int(cfg["dfg_attn_dim"]),
        "dfg_attn_tau": float(cfg["dfg_attn_tau"]),
        "use_ss2d_dfg": bool(cfg["use_ss2d_dfg"]),
        "dfg_gamma_max": float(cfg["dfg_gamma_max"]),
        "dfg_ss2d_fusion": str(cfg["dfg_ss2d_fusion"]),
        "dfg_beta": float(cfg["dfg_beta"]),
        "dfg_beta_schedule": str(cfg["dfg_beta_schedule"]),
        "dfg_beta_target": float(cfg["dfg_beta_target"]),
        "dfg_beta_current": float(model.dfg_beta),
        "dfg_weight_residual_fp32": bool(cfg["dfg_weight_residual_fp32"]),
        "prompt_mode": "hybrid",
        "use_soft_prompt": True,
        "use_hybrid_soft_prompt": True,
        "soft_prompt_ctx_len": int(cfg["soft_prompt_ctx_len"]),
        "soft_prompt_init": str(cfg["soft_prompt_init"]),
        "soft_prompt_init_phrase": str(cfg["soft_prompt_init_phrase"]),
        "hybrid_alpha_current": float(model.hybrid_alpha_current),
        "hybrid_alpha_max": float(cfg["hybrid_alpha_max"]),
        "soft_prompt_freeze_epochs": int(cfg["soft_prompt_freeze_epochs"]),
        "lambda_kg": float(cfg["lambda_kg"]),
        "lambda_k": float(cfg["lambda_k"]),
        "k_reg_detached_wk": True,
        "k_reg_per_stage": True,
        "grad_clip_norm": float(cfg["grad_clip_norm"]),
        "amp_enabled": bool(cfg["amp"]),
        "precision": "amp_mixed_historical_h2",
        "tf32": str(cfg["tf32"]),
        "anchor": anchor.metadata(float(anchor_lambda)) if anchor is not None else {"enabled": False, "lambda_anchor": float(anchor_lambda)},
        "cir_training": bool(arm == "RCA"),
        "deployment_alpha": 0.0,
        "image_adapter": model.image_adapter.state_dict(),
        "text_adapter": model.text_adapter.state_dict(),
        "soft_prompt": model.soft_prompt.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": list(history),
    }
    payload.update(capture_rng())
    return payload


def restore_payload(model: Any, optimizer: torch.optim.Optimizer, scheduler: StepLR, scaler: torch.cuda.amp.GradScaler, payload: Mapping[str, Any]) -> None:
    model.image_adapter.load_state_dict(payload["image_adapter"])
    model.text_adapter.load_state_dict(payload["text_adapter"])
    model.soft_prompt.load_state_dict(payload["soft_prompt"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    if payload.get("scaler_state"):
        scaler.load_state_dict(payload["scaler_state"])
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", 0.0))
    model.dfg_beta = float(payload.get("dfg_beta_current", model.dfg_beta))
    restore_rng(payload)


def make_loader(cfg: Mapping[str, Any], modules: Mapping[str, Any], h2_repo: Path) -> DataLoader:
    dataset = modules["TextAndImageDataset"](
        str(Path(cfg["source_root"]).resolve()),
        str((h2_repo / "dataset" / "hub" / "VisA.jsonl").resolve()),
        int(cfg["img_size"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
    )


def make_e0(args: argparse.Namespace, cfg: Mapping[str, Any], config_sha: str, modules: Mapping[str, Any], anchor_class: Any) -> None:
    seed_everything(int(cfg["seed"]))
    device = torch.device(args.device)
    model = build_model(cfg, modules, device)
    optimizer = make_optimizer(model, cfg)
    scheduler = StepLR(optimizer, step_size=1, gamma=float(cfg["lr_gamma"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["amp"]))
    payload = checkpoint_payload(
        model, optimizer, scheduler, scaler, cfg, config_sha, "E0", 0, 0, None, 0.0, []
    )
    payload["snapshot_kind"] = "exact_h2_initialization_only"
    payload["dataloader_contract"] = "historical H2 implicit global RNG; no explicit DataLoader generator"
    atomic_torch_save(payload, Path(args.output))
    write_json(Path(args.output).with_suffix(".identity.json"), {
        "status": "PASS",
        "snapshot_kind": payload["snapshot_kind"],
        "path": str(Path(args.output).resolve()),
        "sha256": sha256_file(Path(args.output)),
        "git_sha": current_git_sha(),
        "arch_id": str(cfg["arch_id"]),
        "architecture_version": int(cfg["architecture_version"]),
        "architecture_freeze_sha256": str(cfg["architecture_freeze_sha256"]),
        "h2_repo": str(Path(cfg["h2_repo_path"]).resolve()),
        "h2_commit": cfg["h2_repo_commit"],
        "config_sha256": config_sha,
        "seed": int(cfg["seed"]),
        "epoch": 0,
        "learned_h2_e10_weights_included": False,
        "optimizer_state_present": True,
        "scheduler_state": scheduler.state_dict(),
        "amp": bool(cfg["amp"]),
    })
    print(f"E0_SNAPSHOT={Path(args.output).resolve()}")
    print(f"E0_SHA256={sha256_file(Path(args.output))}")


def train_arm(args: argparse.Namespace, cfg: Mapping[str, Any], config_sha: str, modules: Mapping[str, Any], anchor_class: Any, core: Mapping[str, Any]) -> None:
    arm = str(args.arm).upper()
    if arm not in {"R", "RA", "RCA"}:
        raise ValueError(f"unsupported arm: {arm}")
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "train.log"
    logging.basicConfig(filename=str(log_path), level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(f"h2_master_{arm}")
    device = torch.device(args.device)
    e0_path = Path(args.e0).resolve()
    if not e0_path.is_file():
        raise FileNotFoundError(e0_path)
    e0 = torch.load(e0_path, map_location="cpu", weights_only=False)
    if e0.get("snapshot_kind") != "exact_h2_initialization_only":
        raise ValueError("E0 snapshot is not the exact-H2 initialization snapshot")
    if e0.get("h2_contract_sha256") != config_sha:
        raise ValueError("E0/config SHA mismatch")
    seed_everything(int(cfg["seed"]))
    model = build_model(cfg, modules, device)
    optimizer = make_optimizer(model, cfg)
    scheduler = StepLR(optimizer, step_size=1, gamma=float(cfg["lr_gamma"]))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["amp"]))
    anchor = None
    if arm in {"RA", "RCA"}:
        if args.anchor_checkpoint is None:
            raise ValueError("RA/RCA require --anchor-checkpoint")
        anchor = load_anchor(Path(args.anchor_checkpoint).resolve(), model, device, anchor_class)
    resume_path = Path(args.resume).resolve() if args.resume else None
    if resume_path is not None:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("arm") != arm or payload.get("h2_contract_sha256") != config_sha:
            raise ValueError("resume payload identity mismatch")
        restore_payload(model, optimizer, scheduler, scaler, payload)
        history = list(payload.get("history", []))
        start_epoch = int(payload["epoch"]) + 1
    else:
        restore_payload(model, optimizer, scheduler, scaler, e0)
        history = []
        start_epoch = 1
    loader = make_loader(cfg, modules, Path(cfg["h2_repo_path"]).resolve())
    max_epoch = int(args.max_epoch or (int(cfg["r_candidate_epochs"][0]) if arm == "R" else cfg["epochs"]))
    candidates = set(int(x) for x in (cfg["r_candidate_epochs"] if arm == "R" else cfg["candidate_epochs"]))
    h2_root = Path(cfg["h2_repo_path"]).resolve()
    source_manifest = (h2_root / cfg["manifest_path"]).resolve()
    clip_asset = (h2_root / "model" / "ViT-L-14-336px.pt").resolve()
    manifest_path = run_root / "run_manifest.json"
    manifest = {
        "status": "RUNNING",
        "arm": arm,
        "git_sha": current_git_sha(),
        "arch_id": str(cfg["arch_id"]),
        "architecture_version": int(cfg["architecture_version"]),
        "architecture_freeze_sha256": str(cfg["architecture_freeze_sha256"]),
        "h2_repo_path": str(h2_root),
        "h2_repo_commit": cfg["h2_repo_commit"],
        "h2_contract_sha256": config_sha,
        "config_sha256": config_sha,
        "source": str(cfg["source_dataset"]),
        "source_root": str(Path(cfg["source_root"]).resolve()),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "clip_asset": str(clip_asset),
        "clip_asset_sha256": sha256_file(clip_asset),
        "seed": int(cfg["seed"]),
        "e0_path": str(e0_path),
        "e0_sha256": sha256_file(e0_path),
        "anchor": anchor.metadata(float(cfg["anchor_lambda"])) if anchor is not None else {"enabled": False, "lambda_anchor": 0.0},
        "cir_training": bool(arm == "RCA"),
        "deployment_alpha": 0.0,
        "start_epoch": start_epoch,
        "max_epoch": max_epoch,
        "candidate_epochs": sorted(candidates),
        "optimizer": {
            "class": "torch.optim.Adam",
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "param_group_order": ["text_adapter", "image_adapter", "soft_prompt"],
            "base_lrs": {"text_adapter": float(cfg["text_lr"]), "image_adapter": float(cfg["image_lr"]), "soft_prompt": float(cfg["soft_prompt_lr"])},
        },
        "scheduler": {
            "class": "torch.optim.lr_scheduler.StepLR",
            "step_size": 1,
            "gamma": float(cfg["lr_gamma"]),
            "timing": "after_epoch_before_candidate_save",
        },
        "loader": {
            "batch_size": int(cfg["batch_size"]),
            "effective_batch_size": int(cfg["effective_batch_size"]),
            "num_workers": int(cfg["num_workers"]),
            "pin_memory": bool(cfg["pin_memory"]),
            "persistent_workers": bool(cfg["persistent_workers"]),
            "prefetch_factor": int(cfg["prefetch_factor"]),
        },
        "precision": {
            "amp": bool(cfg["amp"]),
            "amp_dtype": str(cfg["amp_dtype"]),
            "tf32": str(cfg["tf32"]),
            "grad_scaler": True,
        },
    }
    write_json(manifest_path, manifest)
    logger.info("arm=%s h2_commit=%s config_sha=%s start_epoch=%d max_epoch=%d", arm, cfg["h2_repo_commit"], config_sha, start_epoch, max_epoch)
    global_step = int(history[-1]["global_step"]) if history else int(e0.get("global_step", 0))
    started = time.perf_counter()
    for epoch in range(start_epoch, max_epoch + 1):
        alpha, beta, soft_frozen = set_epoch_state(model, optimizer, cfg, modules, epoch)
        loss_values: list[float] = []
        cls_values: list[float] = []
        seg_values: list[float] = []
        kg_values: list[float] = []
        k_values: list[float] = []
        anchor_values: list[float] = []
        rmt_rows: list[dict[str, float]] = []
        nonfinite_loss = 0
        nonfinite_grad = 0
        epoch_started = time.perf_counter()
        for batch_idx, input_data in enumerate(tqdm(loader, desc=f"{arm} E{epoch:02d}", leave=False)):
            image = input_data["image"].to(device, non_blocking=True)
            mask = input_data["mask"].to(device, non_blocking=True)
            label = input_data["label"].to(device, non_blocking=True)
            class_names = input_data["class_name"]
            text_by_class: dict[str, torch.Tensor] = {}
            kg_losses: list[torch.Tensor] = []
            k_losses: list[torch.Tensor] = []
            for class_name in list(set(class_names)):
                text_embedding_levels, kg_loss_class, _soft_stats, components = modules["get_hybrid_soft_prompt_single_class_text_embedding"](
                    model, "VisA", class_name, device, return_kg=True, return_components=True
                )
                k_loss_class, _k_stats = modules["compute_hybrid_k_regularization"](
                    model, components["hard_text"], components["soft_text"], float(alpha)
                )
                text_by_class[class_name] = text_embedding_levels
                kg_losses.append(kg_loss_class)
                k_losses.append(k_loss_class)
            text_features = torch.stack([text_by_class[class_name] for class_name in class_names], dim=0).permute(1, 0, 2, 3)
            kg_loss = torch.stack(kg_losses).mean() if kg_losses else torch.zeros((), device=device)
            k_loss = torch.stack(k_losses).mean() if k_losses else torch.zeros((), device=device)
            with torch.cuda.amp.autocast(enabled=bool(cfg["amp"])):
                seg_tokens, det_tokens = model(image)
                seg_features = torch.stack(seg_tokens, dim=0)
                det_features = torch.stack(det_tokens, dim=0)
                cls_pred = torch.stack([
                    torch.matmul(det_features[i].unsqueeze(1), text_features[i]).squeeze(1)
                    for i in range(det_features.shape[0])
                ], dim=0).mean(dim=0)
                cls_loss = F.cross_entropy(cls_pred, label)
                if arm == "RCA":
                    seg_pred, rmt_values = cir_training_segmentation(model, seg_features, text_features, cfg, core)
                    rmt_rows.append(rmt_values)
                else:
                    seg_pred = model.vision_text_fusion_gate_seg(seg_features, text_features)
                seg_loss = modules["calculate_seg_loss"](seg_pred, mask)
                base_loss = cls_loss + seg_loss + float(cfg["lambda_kg"]) * kg_loss + float(cfg["lambda_k"]) * k_loss
                anchor_loss = anchor.loss(model.image_adapter) if anchor is not None else base_loss.detach() * 0.0
                total_loss = base_loss + float(cfg["anchor_lambda"]) * anchor_loss
            if not torch.isfinite(total_loss).all():
                nonfinite_loss += 1
                optimizer.zero_grad(set_to_none=True)
                if nonfinite_loss > int(cfg["non_finite_loss_abort_threshold"]):
                    raise RuntimeError(f"non-finite H2 extension loss exceeded threshold at arm={arm} epoch={epoch}")
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            bad_grad = any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for group in optimizer.param_groups for parameter in group["params"]
            )
            if bad_grad:
                nonfinite_grad += 1
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue
            nn.utils.clip_grad_norm_(model.image_adapter.parameters(), float(cfg["grad_clip_norm"]))
            nn.utils.clip_grad_norm_(model.text_adapter.parameters(), float(cfg["grad_clip_norm"]))
            if not soft_frozen:
                nn.utils.clip_grad_norm_(model.soft_prompt.parameters(), float(cfg["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            loss_values.append(float(total_loss.detach().float().item()))
            cls_values.append(float(cls_loss.detach().float().item()))
            seg_values.append(float(seg_loss.detach().float().item()))
            kg_values.append(float(kg_loss.detach().float().item()))
            k_values.append(float(k_loss.detach().float().item()))
            anchor_values.append(float(anchor_loss.detach().float().item()))
        scheduler.step()
        if soft_frozen:
            for group in optimizer.param_groups:
                if group.get("name") == "soft_prompt":
                    group["lr"] = 0.0
        else:
            for group in optimizer.param_groups:
                if group.get("name") == "soft_prompt":
                    group["lr"] = float(group["constant_lr"])
        rmt_mean = {
            key: float(np.mean([row[key] for row in rmt_rows]))
            for key in rmt_rows[0]
        } if rmt_rows else {}
        row: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "mean_loss": float(np.mean(loss_values)) if loss_values else None,
            "mean_cls": float(np.mean(cls_values)) if cls_values else None,
            "mean_seg": float(np.mean(seg_values)) if seg_values else None,
            "mean_kg": float(np.mean(kg_values)) if kg_values else None,
            "mean_k": float(np.mean(k_values)) if k_values else None,
            "mean_anchor": float(np.mean(anchor_values)) if anchor_values else None,
            "alpha": alpha,
            "beta": beta,
            "soft_prompt_frozen": soft_frozen,
            "nonfinite_loss_skips": nonfinite_loss,
            "nonfinite_grad_skips": nonfinite_grad,
            "batches": len(loss_values),
            "elapsed_seconds": time.perf_counter() - epoch_started,
            "image_lr": float(optimizer.param_groups[1]["lr"]),
            "text_lr": float(optimizer.param_groups[0]["lr"]),
            "soft_prompt_lr": float(optimizer.param_groups[2]["lr"]),
            "scheduler_last_epoch": int(scheduler.last_epoch),
            "scheduler_step_count": int(scheduler._step_count),
            **rmt_mean,
        }
        history.append(row)
        payload = checkpoint_payload(model, optimizer, scheduler, scaler, cfg, config_sha, arm, epoch, global_step, anchor, float(cfg["anchor_lambda"]) if anchor is not None else 0.0, history)
        atomic_torch_save(payload, run_root / "last.pth")
        if epoch in candidates:
            atomic_torch_save(payload, run_root / f"adapter_{epoch}.pth")
        write_csv(run_root / "training_telemetry.csv", list(history[0].keys()), history)
        write_json(run_root / "PROGRESS.json", {"status": "RUNNING", "arm": arm, "epoch": epoch, "global_step": global_step, "candidate_saved": epoch in candidates, "history": history})
        logger.info("epoch=%d row=%s", epoch, row)
        print(f"{arm}_E{epoch}=PASS batches={len(loss_values)} global_step={global_step} image_lr={row['image_lr']:.9g}")
    manifest.update({
        "status": "COMPLETED",
        "completed_epoch": max_epoch,
        "global_step": global_step,
        "history": history,
        "train_wall_seconds": time.perf_counter() - started,
        "checkpoint_sha256": {str(epoch): sha256_file(run_root / f"adapter_{epoch}.pth") for epoch in sorted(candidates) if (run_root / f"adapter_{epoch}.pth").is_file()},
    })
    write_json(manifest_path, manifest)
    write_json(run_root / "PROGRESS.json", {"status": "COMPLETED", "arm": arm, "epoch": max_epoch, "global_step": global_step, "history": history})
    print(f"{arm}_STATUS=COMPLETED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("make-e0", "train"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm", choices=("R", "RA", "RCA"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--e0", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--anchor-checkpoint", type=Path)
    parser.add_argument("--max-epoch", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    config_sha = sha256_file(args.config)
    h2_repo = Path(cfg.get("h2_repo_path", DEFAULT_H2_REPO)).resolve()
    modules = _load_h2_modules(h2_repo)
    current_cir = _load_current_cir_primitives(h2_repo)
    core = {
        "cir_logits_from_native_weights": current_cir["cir_logits_from_native_weights"],
        "peer_delta_from_native_margins": current_cir["peer_delta_from_native_margins"],
    }
    if args.mode == "make-e0":
        if args.output is None:
            raise ValueError("--output is required for make-e0")
        make_e0(args, cfg, config_sha, modules, current_cir["ImageParameterAnchor"])
    else:
        if args.arm is None or args.run_root is None or args.e0 is None:
            raise ValueError("train requires --arm, --run-root, and --e0")
        train_arm(args, cfg, config_sha, modules, current_cir["ImageParameterAnchor"], core)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
