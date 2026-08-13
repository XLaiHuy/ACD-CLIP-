#!/usr/bin/env python3
"""Inference-only matched RAW/Q capture from a faithful fresh Phase4 seed state."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import get_text_and_image_dataset
from model.adapter import ACDCLIP
from model.checkpoint_utils import h6_config_from_checkpoint
from model.clip import create_model
from model.h6.utility_routing import build_patch_targets, utility_teacher
from tools.audit_p1_v84a_post300 import _IndexedDataset, _state_hash
from utils import get_phase2b_global_text_features, make_dataloader_generator, seed_worker


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fresh_model(checkpoint: dict, device: torch.device) -> ACDCLIP:
    config = checkpoint["phase2b_config"]
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=int(checkpoint.get("img_size", 518)),
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    if bool(config.get("grad_checkpointing", True)):
        clip_model.set_grad_checkpointing(True)
    h6_kwargs = {f"h6_{name}": value for name, value in h6_config_from_checkpoint(checkpoint).items()}
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=int(checkpoint["n_groups"]),
        image_adapt_weight=float(config["image_adapt_weight"]),
        conv_lora_rank=int(config["conv_lora_rank"]),
        conv_lora_alpha=float(config["conv_lora_alpha"]),
        conv_kernel_size_list=list(config["conv_kernel_size_list"]),
        text_adapt_weight=float(config["text_adapt_weight"]),
        lora_rank=int(config["lora_rank"]),
        lora_alpha=float(config["lora_alpha"]),
        dfg_mode=checkpoint["dfg_mode"],
        dfg_attn_dim=int(checkpoint["dfg_attn_dim"]),
        dfg_attn_tau=float(checkpoint["dfg_attn_tau"]),
        use_ss2d_dfg=bool(checkpoint["use_ss2d_dfg"]),
        dfg_gamma_max=float(checkpoint["dfg_gamma_max"]),
        dfg_ss2d_fusion=checkpoint["dfg_ss2d_fusion"],
        dfg_beta=float(checkpoint["dfg_beta"]),
        dfg_beta_schedule=checkpoint["dfg_beta_schedule"],
        dfg_beta_target=float(checkpoint["dfg_beta_target"]),
        dfg_beta_current=float(checkpoint["dfg_beta"]),
        use_soft_prompt=bool(checkpoint["use_soft_prompt"]),
        soft_prompt_ctx_len=int(checkpoint["soft_prompt_ctx_len"]),
        soft_prompt_init=checkpoint["soft_prompt_init"],
        soft_prompt_init_phrase=checkpoint["soft_prompt_init_phrase"],
        **h6_kwargs,
    ).to(device)
    model.eval()
    model.clipmodel.eval()
    model.prompt_mode = "hybrid"
    model.use_soft_prompt = False
    model.use_hybrid_soft_prompt = True
    model.hybrid_alpha_current = 0.0  # exact epoch-one schedule state
    model.h6_global_text_mode = checkpoint["global_text_mode"]
    model.set_dfg_beta(float(checkpoint["dfg_beta"]))
    model.h6.set_epoch(1)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = torch.load(args.source_capture, map_location="cpu", weights_only=False)
    wanted = sorted({int(value) for value in source["image_id"].tolist()})
    if wanted != list(range(len(wanted))):
        raise RuntimeError(f"source image support is not a contiguous prefix: {wanted[:8]}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = json.loads(args.config.read_text())
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    model = fresh_model(checkpoint, device)
    state_before = _state_hash(model)
    dataset = _IndexedDataset(get_text_and_image_dataset("VisA", int(checkpoint["img_size"]), "train"))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
        worker_init_fn=seed_worker, generator=make_dataloader_generator(args.seed),
    )
    rows: dict[str, list[torch.Tensor]] = {
        key: [] for key in (
            "image_id", "group_index", "patch_index", "region", "target", "teacher_probability",
            "utility_gap", "utility_informative", "utility_valid", "raw_router_input", "production_q", "logits",
        )
    }
    with torch.inference_mode():
        for image_id, sample in enumerate(loader):
            if image_id >= len(wanted):
                break
            selected = source["image_id"].long() == image_id
            image = sample["image"].to(device, non_blocking=True)
            mask = sample["mask"].to(device, non_blocking=True)
            local_valid = sample["local_mask_valid"].to(device, non_blocking=True)
            classes = [sample["class_name"][0]]
            visual = model(image, return_phase4_features=True)
            h6 = model.h6.build_batch(model, "VisA", classes, visual, hybrid_alpha=0.0, update_load_bias=False)
            seg_features = torch.stack(visual["seg_tokens"], dim=0)
            text_global = get_phase2b_global_text_features(
                model, "VisA", classes, device, use_hybrid_soft_prompt=True, use_soft_prompt=False
            ).to(dtype=seg_features.dtype)
            _, _, base = model.vision_text_fusion_gate_seg(
                seg_features, text_global, img_size=int(checkpoint["img_size"]),
                h6_patch_logits=h6["h6_logits"], return_details=True,
            )
            patch_count = int(h6["factor_residual_logits"].shape[2])
            targets, utility_valid = build_patch_targets(mask, patch_count, local_valid)
            epsilon = float(config["h6_exploration_end"])
            utility = utility_teacher(
                base.detach(), h6["factor_residual_logits"], targets, utility_valid,
                rho=0.05,
                denominator_floor=float(config["h6_utility_denominator_floor"]),
                tau_utility=float(config["h6_tau_utility"]),
                factor_tau_utility=float(config["h6_factor_tau_utility"]),
                router_tau_utility=float(config["h6_router_tau_utility"]),
                epsilon=epsilon,
                gain_threshold=float(config["h6_utility_gain_threshold"]),
                router_gain_threshold=float(config["h6_router_gain_threshold"]),
                entropy_threshold=float(config["h6_utility_entropy_threshold"]),
                router_confidence_mode=config["h6_router_confidence_mode"],
                router_margin_rel_threshold=float(config["h6_router_margin_rel_threshold"]),
                router_target_mode=config["h6_router_target_mode"],
                role_topology=config["h6_role_topology"],
                role_teacher_scale=float(config["h6_role_teacher_scale"]),
                routed_probabilities=h6["prediction_probabilities"],
            )
            group = source["group_index"][selected].long().to(device)
            patch = source["patch_index"][selected].long().to(device)
            count = int(group.numel())
            rows["image_id"].append(torch.full((count,), image_id, dtype=torch.int64))
            rows["group_index"].append(group.cpu())
            rows["patch_index"].append(patch.cpu())
            rows["region"].append((targets[0, patch] >= 0.5).to(torch.uint8).cpu())
            rows["target"].append(targets[0, patch].float().cpu())
            rows["teacher_probability"].append(utility["q_utility"][group, 0, patch].float().cpu())
            rows["utility_gap"].append(utility["role_gap"][group, 0, patch].float().cpu())
            rows["utility_informative"].append(utility["informative"][group, 0, patch].to(torch.uint8).cpu())
            rows["utility_valid"].append(utility["valid"][group, 0, patch].to(torch.uint8).cpu())
            rows["raw_router_input"].append(h6["router_patch_features"][group, 0, patch].float().cpu())
            rows["production_q"].append(h6["queries"][group, 0, patch].float().cpu())
            rows["logits"].append(h6["prediction_logits"][group, 0, patch].float().cpu())
    capture = {key: torch.cat(values) for key, values in rows.items()}
    for key in ("image_id", "group_index", "patch_index"):
        if not torch.equal(capture[key], source[key].long()):
            raise RuntimeError(f"source index mismatch in {key}")
    state_after = _state_hash(model)
    if state_before != state_after:
        raise RuntimeError("inference changed fresh model state")
    output = {
        "state": "fresh_seed0_epoch1_initialization",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_capture": str(args.source_capture.resolve()),
        "seed": args.seed,
        "source_commit": checkpoint.get("git_sha"),
        "fresh_parameter_state_hash": state_before,
        "metadata": {
            "images": len(wanted), "patches": int(capture["image_id"].numel()),
            "optimizer_steps": 0, "backward": False, "model_state_unchanged": True,
            "tf32_disabled": not torch.backends.cuda.matmul.allow_tf32,
        },
        **capture,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()), "metadata": output["metadata"],
        "raw_shape": list(capture["raw_router_input"].shape), "q_shape": list(capture["production_q"].shape),
    }, indent=2))


if __name__ == "__main__":
    main()
