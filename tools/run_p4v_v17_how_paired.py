#!/usr/bin/env python3
"""Fresh full-TRAIN Phase2B common warmup plus paired ungated K=1 HOW probe."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import train as phase2b
from audit_p4_k1_oracle_utility import _sha256
from run_p4v_short64 import build
from utils import calculate_seg_loss, configure_canonical_fp32, get_phase2b_global_text_features

SEED = 1707
WARMUP_EPOCHS = 5
ACTIVE_UPDATES = 32
ETA = 1.0


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def prepare_phase2b(model):
    model.requires_grad_(False)
    model.clipmodel.eval()
    model.image_adapter.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.soft_prompt.requires_grad_(False)
    model.h6.requires_grad_(False)
    model.use_hybrid_soft_prompt = True
    model.use_soft_prompt = False
    model.prompt_mode = "hybrid"
    model.hybrid_alpha_current = 0.0
    model.hybrid_alpha_max = 0.2
    model.soft_prompt_freeze_epochs = 3


def optimizer(model):
    return torch.optim.Adam([
        {"name": "text_adapter", "params": model.text_adapter.parameters(), "lr": 5e-4},
        {"name": "image_adapter", "params": model.image_adapter.parameters(), "lr": 1e-3},
        {"name": "soft_prompt", "params": model.soft_prompt.parameters(), "lr": 0.0, "constant_lr": 5e-5},
    ])


def full_loader(config, generator_state=None):
    dataset = phase2b.get_text_and_image_dataset("VisA", config["img_size"], "train")
    generator = phase2b.make_dataloader_generator(SEED)
    if generator_state is not None:
        generator.set_state(generator_state)
    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=phase2b.seed_worker, generator=generator)
    return dataset, loader, generator


def set_epoch6_phase2b(model, opt):
    epoch = WARMUP_EPOCHS + 1
    model.hybrid_alpha_current = phase2b.get_hybrid_alpha_for_epoch(epoch, .2, 3)
    model.soft_prompt.requires_grad_(True)
    phase2b.apply_soft_prompt_lr_policy(opt, False)
    model.set_dfg_beta(phase2b.get_dfg_beta_for_epoch(epoch, "warmup010", .10, .10))


def base_main(model, image, mask, label, classes, config):
    visual = model(image, return_phase4_features=True)
    features = torch.stack(visual["seg_tokens"])
    text = get_phase2b_global_text_features(model, "VisA", list(classes), image.device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
    pred, logits, _ = model.vision_text_fusion_gate_seg(features, text, img_size=config["img_size"], return_details=True)
    det = torch.stack(visual["det_tokens"])
    cls = torch.stack([det[group].unsqueeze(1).matmul(text[group]).squeeze(1) for group in range(model.n_groups)]).mean(0)
    return calculate_seg_loss(pred.float(), mask) + F.cross_entropy(cls.float(), label), pred, logits, visual, features, text


def isolated_how(model, visual, features, text, mask, config):
    detached = {key: ([item.detach() for item in value] if isinstance(value, list) else value.detach()) for key, value in visual.items()}
    state = model.h6.phase4v_state_code(model, detached)["semantic_code"]
    adapted, anchors = [], []
    for group in range(model.n_groups):
        gate = torch.ones(features[group].shape[:2], device=features.device, dtype=features.dtype)
        out = model.h6.phase4v_adapt(features[group].detach(), state, gate, enabled=True, semantic_conditioning=True, spatial_gating=False)
        adapted.append(out)
        with torch.no_grad():
            group_text = text.detach().permute(1, 0, 2, 3)
            weights = model.compute_dfg_weights(features[group].detach(), group_text, group)
            anchors.append(model.apply_dfg_weights(group_text, weights["normal"], weights["abnormal"]).detach())
    corr_logits = []
    for group, out in enumerate(adapted):
        patches = out["adapted"]; side = math.isqrt(patches.shape[1])
        corr_logits.append((10 * patches).matmul(anchors[group]).permute(0, 2, 1).view(patches.shape[0], 2, side, side))
    corr = F.softmax(torch.stack([F.interpolate(value, size=config["img_size"], mode="bilinear", align_corners=True) for value in corr_logits]).mean(0), dim=1)
    return calculate_seg_loss(corr.float(), mask), corr, adapted


def grad_norm(module):
    values = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return 0.0 if not values else float(torch.stack(values).norm())


def max_abs(left, right):
    return float((left.detach().float() - right.detach().float()).abs().max())


def zero_step(model, config, output: Path):
    model.h6.conditional_semantic_core.requires_grad_(True)
    model.h6.visual_adapter.requires_grad_(True)
    _, loader, _ = full_loader(config)
    raw = next(iter(loader)); device = next(model.parameters()).device
    image, mask, label, classes = raw["image"].to(device).float(), raw["mask"].to(device).float(), raw["label"].to(device), list(raw["class_name"])
    model.train(); model.clipmodel.eval(); base_loss, base_pred, _, visual, features, text = base_main(model, image, mask, label, classes, config)
    detached = {key: ([item.detach() for item in value] if isinstance(value, list) else value.detach()) for key, value in visual.items()}
    state = model.h6.phase4v_state_code(model, detached)["semantic_code"]
    ones = [torch.ones(features[group].shape[:2], device=device) for group in range(model.n_groups)]
    zero_outputs = [model.h6.phase4v_adapt(features[group].detach(), state, ones[group], enabled=True, semantic_conditioning=True, spatial_gating=False, lambda_override=0.0) for group in range(model.n_groups)]
    zero_pred = model.vision_text_fusion_gate_seg(torch.stack([out["adapted"] for out in zero_outputs]), text, img_size=config["img_size"])
    delta_zero_outputs = [model.h6.phase4v_adapt(features[group].detach(), state, ones[group], enabled=True, semantic_conditioning=True, spatial_gating=False, force_delta_zero=True) for group in range(model.n_groups)]
    delta_zero_pred = model.vision_text_fusion_gate_seg(torch.stack([out["adapted"] for out in delta_zero_outputs]), text, img_size=config["img_size"])
    corr_loss, corr_pred, adapted = isolated_how(model, visual, features, text, mask, config)
    model.zero_grad(set_to_none=True); corr_loss.backward(retain_graph=True)
    corr_base = grad_norm(model.image_adapter) + grad_norm(model.text_adapter) + grad_norm(model.soft_prompt)
    corr_cops, corr_visual = grad_norm(model.h6.conditional_semantic_core), grad_norm(model.h6.visual_adapter)
    model.zero_grad(set_to_none=True); base_loss.backward()
    base_main_grad = grad_norm(model.image_adapter) + grad_norm(model.text_adapter)
    delta = torch.stack([out["delta_v"] for out in adapted]); correction = torch.stack([out["correction"] for out in adapted])
    checks = {"how_off_exact": max_abs(base_pred, base_pred), "lambda_zero_exact": max_abs(zero_pred, base_pred), "delta_zero_exact": max_abs(delta_zero_pred, base_pred), "corr_base_grad_exact_zero": corr_base == 0.0, "base_main_grad_positive": base_main_grad > 0.0, "cops_grad_positive": corr_cops > 0.0, "visual_grad_positive": corr_visual > 0.0, "router_grad_zero": grad_norm(model.h6.router) == 0.0, "factor_grad_zero": grad_norm(model.h6.semantic_core) == 0.0, "act_absent": model.h6.act_head is None, "finite": bool(torch.isfinite(base_loss) and torch.isfinite(corr_loss) and torch.isfinite(corr_pred).all())}
    exact = (checks["how_off_exact"], checks["lambda_zero_exact"], checks["delta_zero_exact"])
    passed = all(value == 0.0 for value in exact) and all(value is True for key, value in checks.items() if key not in {"how_off_exact", "lambda_zero_exact", "delta_zero_exact"})
    report = {"decision": "V1_7_HOW_ZERO_STEP_PASS" if passed else "V1_7_HOW_ZERO_STEP_FAIL", "checks": checks, "gradients": {"correction_base": corr_base, "correction_cops": corr_cops, "correction_visual": corr_visual, "base_main": base_main_grad}, "geometry": {"raw_delta_norm": float(delta.detach().float().norm(dim=-1).mean()), "ungated_correction_norm": float(correction.detach().float().norm(dim=-1).mean()), "correction_to_base": float(correction.detach().float().norm(dim=-1).mean() / features.detach().float().norm(dim=-1).mean())}, "provenance": {"optimizer_steps": 0, "precision": "strict FP32", "train_dataset": "full VisA TRAIN", "seed": SEED}}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return passed


def clone(config, state, optimizer_state, device):
    model = build(config, device); prepare_phase2b(model)
    model.image_adapter.load_state_dict(state["image_adapter"]); model.text_adapter.load_state_dict(state["text_adapter"]); model.soft_prompt.load_state_dict(state["soft_prompt"]); model.h6.load_state_dict(state["h6"])
    model.h6.conditional_semantic_core.requires_grad_(True); model.h6.visual_adapter.requires_grad_(True)
    opt = optimizer(model); opt.load_state_dict(copy.deepcopy(optimizer_state)); set_epoch6_phase2b(model, opt)
    opt.add_param_group({"name": "conditional_semantic_core", "params": model.h6.conditional_semantic_core.parameters(), "lr": 1e-4})
    opt.add_param_group({"name": "visual_adapter", "params": model.h6.visual_adapter.parameters(), "lr": 1e-4})
    return model, opt

def active_steps(model, opt, loader, config, how):
    model.train(); model.clipmodel.eval(); rows = []
    for step, raw in zip(range(ACTIVE_UPDATES), loader):
        device = next(model.parameters()).device
        image, mask, label, classes = raw["image"].to(device).float(), raw["mask"].to(device).float(), raw["label"].to(device), list(raw["class_name"])
        opt.zero_grad(set_to_none=True); base_loss, _, _, visual, features, text = base_main(model, image, mask, label, classes, config)
        if how: how_loss, _, _ = isolated_how(model, visual, features, text, mask, config)
        else: how_loss = torch.zeros_like(base_loss)
        total = base_loss + ETA * how_loss; total.backward(); torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]], 1.0); opt.step()
        rows.append({"step": step + 1, "base_loss": float(base_loss.detach()), "how_loss": float(how_loss.detach()), "how_to_base": float((how_loss / base_loss.detach().clamp_min(1e-6)).detach()), "finite": bool(torch.isfinite(total))})
    return rows


def common_warmup(config, device, output: Path):
    fresh = build(config, device); prepare_phase2b(fresh)
    dataset, loader, generator = full_loader(config)
    warm_opt = optimizer(fresh); scheduler = StepLR(warm_opt, step_size=1, gamma=.9)
    common_dir = output / "common_warmup"
    common_dir.mkdir(parents=True, exist_ok=True)
    phase2b.train(fresh, "VisA", loader, warm_opt, scheduler, device, WARMUP_EPOCHS, str(common_dir), phase2b.logging.getLogger("p4v_v17"), use_amp=False, dfg_beta_schedule="warmup010", dfg_beta_target=.10, dfg_beta=.10, lambda_kg=.01, lambda_k=.002, hybrid_alpha_max=.2, soft_prompt_freeze_epochs=3, grad_clip_norm=1.0)
    state = {"image_adapter": fresh.image_adapter.state_dict(), "text_adapter": fresh.text_adapter.state_dict(), "soft_prompt": fresh.soft_prompt.state_dict(), "h6": fresh.h6.state_dict()}
    common_path = common_dir / "common_state.pth"; common_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**state, "optimizer": warm_opt.state_dict(), "generator_state": generator.get_state(), "epoch": WARMUP_EPOCHS}, common_path)
    report = {"decision": "V1_7_COMMON_WARMUP_COMPLETE_PENDING_READINESS", "protocol": {"initialization": "fresh OpenAI CLIP only", "train_dataset_size": len(dataset), "steps_per_epoch": len(loader), "warmup_epochs": WARMUP_EPOCHS, "seed": SEED, "precision": "strict FP32", "config_sha256": _sha256(args.config)}, "local_checkpoint": str(common_path), "readiness_checkpoint": str(common_dir / f"adapter_{WARMUP_EPOCHS}.pth")}
    (common_dir / "COMMON_WARMUP_TRAIN.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def load_common(output: Path):
    path = output / "common_warmup" / "common_state.pth"
    if not path.is_file():
        raise FileNotFoundError(f"common warmup state required: {path}")
    return path, torch.load(path, map_location="cpu", weights_only=False)


def existing_common(config, device, output: Path, source_path: Path, config_path: Path):
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    required = ("image_adapter", "text_adapter", "soft_prompt")
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"fresh checkpoint missing required Phase2B state: {missing}")
    model = build(config, device); prepare_phase2b(model)
    model.image_adapter.load_state_dict(source["image_adapter"])
    model.text_adapter.load_state_dict(source["text_adapter"])
    model.soft_prompt.load_state_dict(source["soft_prompt"])
    dataset, loader, generator = full_loader(config)
    warm_opt = optimizer(model)
    common_dir = output / "common_warmup"; common_dir.mkdir(parents=True, exist_ok=True)
    state = {"image_adapter": model.image_adapter.state_dict(), "text_adapter": model.text_adapter.state_dict(), "soft_prompt": model.soft_prompt.state_dict(), "h6": model.h6.state_dict()}
    common_path = common_dir / "common_state.pth"
    torch.save({**state, "optimizer": warm_opt.state_dict(), "generator_state": generator.get_state(), "epoch": int(source.get("epoch", 0)), "source_checkpoint": str(source_path)}, common_path)
    report = {"decision": "PHASE2B_BASE_HEALTHY_SELECTED", "protocol": {"source": "existing fresh V1.7 checkpoint; no Phase2B retraining", "selected_epoch": int(source.get("epoch", 0)), "train_dataset_size": len(dataset), "steps_per_epoch": len(loader), "seed": SEED, "precision": "strict FP32", "optimizer_history": "not stored in source checkpoint; initialized at selected state", "source_checkpoint": str(source_path), "source_checkpoint_sha256": _sha256(source_path), "config_sha256": _sha256(config_path)}, "common_state": str(common_path)}
    (common_dir / "COMMON_STATE_FROM_EXISTING_FRESH.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def zero_step_from_common(config, device, output: Path):
    common_path, checkpoint = load_common(output)
    state = {key: checkpoint[key] for key in ("image_adapter", "text_adapter", "soft_prompt", "h6")}
    model, _ = clone(config, state, checkpoint["optimizer"], device)
    if not zero_step(model, config, output / "V1_7_HOW_ZERO_STEP.json"):
        raise RuntimeError("V1.7 HOW zero-step from selected common state failed")
def loss_preflight(config, device, output: Path):
    common_path, checkpoint = load_common(output)
    state = {key: checkpoint[key] for key in ("image_adapter", "text_adapter", "soft_prompt", "h6")}
    model, opt = clone(config, state, checkpoint["optimizer"], device)
    _, loader, _ = full_loader(config, checkpoint["generator_state"])
    raw = next(iter(loader)); image, mask, label, classes = raw["image"].to(device).float(), raw["mask"].to(device).float(), raw["label"].to(device), list(raw["class_name"])
    model.train(); model.clipmodel.eval(); base_loss, _, _, visual, features, text = base_main(model, image, mask, label, classes, config); how_loss, _, _ = isolated_how(model, visual, features, text, mask, config)
    report = {"decision": "V1_7_HOW_LOSS_SCALE_RECORDED", "base_loss": float(base_loss.detach()), "how_loss": float(how_loss.detach()), "how_to_base": float((how_loss / base_loss.detach().clamp_min(1e-6)).detach()), "eta": ETA, "common_state": str(common_path), "optimizer_steps": 0, "finite": bool(torch.isfinite(base_loss) and torch.isfinite(how_loss))}
    (output / "HOW_LOSS_PREFLIGHT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def paired_train(config, device, output: Path, config_path: Path):
    preflight_path = output / "HOW_LOSS_PREFLIGHT.json"
    if not preflight_path.is_file():
        raise RuntimeError("run --loss-preflight-only and inspect HOW_LOSS_PREFLIGHT.json before paired training")
    common_path, checkpoint = load_common(output)
    state = {key: checkpoint[key] for key in ("image_adapter", "text_adapter", "soft_prompt", "h6")}
    base, base_opt = clone(config, state, checkpoint["optimizer"], device); dataset, base_loader, _ = full_loader(config, checkpoint["generator_state"])
    base_rows = active_steps(base, base_opt, base_loader, config, False)
    torch.save({"image_adapter": base.image_adapter.state_dict(), "text_adapter": base.text_adapter.state_dict(), "soft_prompt": base.soft_prompt.state_dict(), "h6": base.h6.state_dict()}, output / "base_adapter_state.pth")
    del base, base_opt; torch.cuda.empty_cache()
    how, how_opt = clone(config, state, checkpoint["optimizer"], device); _, how_loader, _ = full_loader(config, checkpoint["generator_state"])
    how_rows = active_steps(how, how_opt, how_loader, config, True)
    torch.save({"image_adapter": how.image_adapter.state_dict(), "text_adapter": how.text_adapter.state_dict(), "soft_prompt": how.soft_prompt.state_dict(), "h6": how.h6.state_dict()}, output / "how_adapter_state.pth")
    report = {"decision": "V1_7_HOW_PAIRED_TRAIN_COMPLETE", "protocol": {"initialization": "fresh OpenAI CLIP only", "train_dataset_size": len(dataset), "steps_per_epoch": len(base_loader), "warmup_epochs": WARMUP_EPOCHS, "active_updates": ACTIVE_UPDATES, "seed": SEED, "precision": "strict FP32", "g_train": 1, "eta": ETA, "config_sha256": _sha256(config_path)}, "finite": {"base": all(row["finite"] for row in base_rows), "how": all(row["finite"] for row in how_rows)}, "loss_scale": {"preflight": json.loads(preflight_path.read_text()), "first_how_to_base": how_rows[0]["how_to_base"], "mean_how_to_base": sum(row["how_to_base"] for row in how_rows) / len(how_rows)}, "local_checkpoints": [str(common_path), str(output / "base_adapter_state.pth"), str(output / "how_adapter_state.pth")]}
    (output / "V1_7_HOW_PAIRED_TRAIN.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("runs/phase4/k1/short64_seed0_attempt5/config.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase4v/v1_7/how"))
    parser.add_argument("--existing-fresh-checkpoint", type=Path, default=Path("runs/phase4v/v1_7/readiness_full/adapter_5.pth"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--zero-step-only", action="store_true")
    mode.add_argument("--existing-common-only", action="store_true")
    mode.add_argument("--zero-step-from-common", action="store_true")
    mode.add_argument("--common-warmup-only", action="store_true")
    mode.add_argument("--loss-preflight-only", action="store_true")
    mode.add_argument("--paired-from-common", action="store_true")
    args = parser.parse_args(); configure_canonical_fp32(); seed_everything(SEED); config = json.loads(args.config.read_text()); device = torch.device("cuda:0")
    if args.zero_step_only:
        fresh = build(config, device); prepare_phase2b(fresh)
        if not zero_step(fresh, config, args.output / "V1_7_HOW_ZERO_STEP.json"):
            raise RuntimeError("V1.7 HOW zero-step failed")
    elif args.existing_common_only:
        existing_common(config, device, args.output, args.existing_fresh_checkpoint, args.config)
    elif args.zero_step_from_common:
        zero_step_from_common(config, device, args.output)
    elif args.common_warmup_only:
        common_warmup(config, device, args.output)
    elif args.loss_preflight_only:
        loss_preflight(config, device, args.output)
    else:
        paired_train(config, device, args.output, args.config)


if __name__ == "__main__":
    main()
