#!/usr/bin/env python3
"""Build a bounded Tier-3 patch bank from real training images only.

The saved rows are exactly ``PatchRouter.router_input_features(seg_tokens)``:
FP32, L2-normalized [G, B, P, 768] patch tokens before the router's local or
global query transforms.  This script deliberately never synthesizes patches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset import get_text_and_image_dataset
from model.adapter import ACDCLIP
from model.checkpoint_utils import h6_config_from_checkpoint, load_adapter_checkpoint
from model.clip import create_model
from model.h6.cluster_responsibility import balanced_kmeans, tensor_sha256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _h6_args(config: dict) -> dict:
    """Map checkpoint metadata to the ACDCLIP public construction arguments."""
    keys = (
        "num_factors", "top_k", "bank_dim", "router_dim", "router_temperature",
        "router_soft_epochs", "sparse_transition_epochs", "load_bias_enabled",
        "load_bias_momentum", "load_bias_step", "load_bias_max", "vae_hidden_dim",
        "vae_latent_dim", "vae_class_ratio", "slot_init_enabled", "slot_init_scale",
        "slot_init_seed_offset", "factor_grad_diagnostics_enabled", "late_factor_identity_enabled",
        "factor_id_scale", "factor_id_max_ratio", "factor_generator_specialization_enabled",
        "factor_head_init_scale", "factor_local_dynamic_mix", "router_query_mode",
        "router_query_global_weight", "router_local_bypass_scale", "router_local_bypass_max_ratio",
        "router_local_projection_seed_offset", "router_key_anchor_enabled", "router_key_anchor_seed_offset",
        "router_key_adaptation_initial_ratio", "router_key_adaptation_max_ratio",
        "factor_context_anchor_enabled", "factor_context_anchor_seed_offset",
        "factor_context_adaptation_initial_ratio", "factor_context_adaptation_max_ratio",
        "factor_identity_tangent_projection_enabled", "lambda_dynamic_mean_anchor",
        "dynamic_mean_anchor_min_cosine", "dynamic_mean_anchor_start_epoch",
        "dynamic_mean_anchor_warmup_epochs", "router_teacher_mode", "progress_version",
        "expert_enabled", "expert_bottleneck", "expert_fofs_seed_offset", "expert_state_condition_scale",
        "expert_scale_target", "expert_scale_start_epoch", "expert_scale_warmup_epochs",
        "expert_max_relative_ratio", "prediction_routing", "cluster_responsibility_enabled",
        "cluster_temperature",
    )
    aliases = {
        "factor_grad_diagnostics_enabled": "h6_factor_grad_diagnostics",
        "lambda_dynamic_mean_anchor": "lambda_h6_dynamic_mean_anchor",
        "prediction_routing": "h6_prediction_routing",
        "cluster_responsibility_enabled": "h6_cluster_responsibility",
        "cluster_temperature": "h6_cluster_temperature",
    }
    return {aliases.get(key, f"h6_{key}"): config[key] for key in keys if key in config}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_patches", type=int, default=20_000)
    parser.add_argument("--max_cluster_initializations", type=int, default=3)
    parser.add_argument(
        "--verify_checkpoint_only",
        action="store_true",
        help="Construct and reload the model, then verify Tier-3 buffers without reading data.",
    )
    args = parser.parse_args()
    if not 1 <= args.max_patches <= 20_000:
        raise ValueError("--max_patches must be in [1, 20000]")
    if not 1 <= args.max_cluster_initializations <= 3:
        raise ValueError("--max_cluster_initializations must be in [1, 3]")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = h6_config_from_checkpoint(checkpoint)
    if config is None:
        raise ValueError("Tier-3 patch-bank construction requires a Phase-4 checkpoint")
    clip_model = create_model(args.model_name, args.img_size, device=device, pretrained="openai", require_pretrained=True)
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=int(checkpoint.get("n_groups", config["n_groups"])),
        dfg_mode=checkpoint.get("dfg_mode", "mlp"),
        dfg_attn_dim=int(checkpoint.get("dfg_attn_dim", 256)),
        dfg_attn_tau=float(checkpoint.get("dfg_attn_tau", 4.0)),
        use_ss2d_dfg=bool(checkpoint.get("use_ss2d_dfg", False)),
        dfg_gamma_max=float(checkpoint.get("dfg_gamma_max", 0.2)),
        dfg_ss2d_fusion=checkpoint.get("dfg_ss2d_fusion", "feature_residual"),
        dfg_beta=float(checkpoint.get("dfg_beta", 0.10)),
        h6_progress=1,
        **_h6_args(config),
    ).to(device)
    load_adapter_checkpoint(model, checkpoint)
    if args.verify_checkpoint_only:
        if not model.h6.cluster_ready:
            raise RuntimeError("checkpoint reload did not materialize Tier-3 centroids")
        print(json.dumps({
            "checkpoint": str(args.checkpoint),
            "cluster_centroids_shape": list(model.h6.cluster_centroids.shape),
            "cluster_identity_shape": list(model.h6.cluster_identity.shape),
            "cluster_identity_tied": bool(model.h6.semantic_core.tier3_cluster_identity_enabled),
        }, sort_keys=True))
        return
    model.eval()
    dataset = get_text_and_image_dataset(args.dataset, args.img_size, "train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for item in loader:
            visual_output = model(item["image"].to(device), return_phase4_features=True)
            router_input = model.h6.router.router_input_features(visual_output["seg_tokens"])
            remaining = args.max_patches - sum(part.shape[0] for part in rows)
            if remaining <= 0:
                break
            rows.append(router_input.reshape(-1, router_input.shape[-1]).detach().cpu()[:remaining])
    features = torch.cat(rows, dim=0)
    if features.shape[0] < 80:
        raise RuntimeError("real training stream yielded fewer than 80 patches; cannot form four reliable clusters")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_path = args.output_dir / "patch_bank.pt"
    patch_meta = {
        "tier": 3,
        "representation": "PatchRouter.router_input_features(seg_tokens): FP32 L2-normalized [G,B,P,768]",
        "dataset": args.dataset,
        "split": "train",
        "seed": args.seed,
        "max_patches": args.max_patches,
        "patch_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": _sha256(args.checkpoint),
        "patch_bank_sha256": tensor_sha256(features),
    }
    torch.save({"features": features, "metadata": patch_meta}, patch_path)
    (args.output_dir / "patch_bank_meta.json").write_text(json.dumps(patch_meta, indent=2, sort_keys=True) + "\n")
    centroids, _, report = balanced_kmeans(
        features, num_clusters=4, seed=args.seed,
        max_initializations=args.max_cluster_initializations,
    )
    centroid_meta = {**patch_meta, "centroid_sha256": tensor_sha256(centroids)}
    torch.save({"centroids": centroids, "metadata": centroid_meta}, args.output_dir / "cluster_centroids.pt")
    (args.output_dir / "cluster_report.json").write_text(
        json.dumps({**report, "metadata": centroid_meta}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
