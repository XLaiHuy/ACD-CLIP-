#!/usr/bin/env python3
"""Read-only semantic and gradient audit for a completed Tier-3 checkpoint.

The audit never calls an optimizer or writes a checkpoint.  It uses the exact
training dataset/preprocessing class with a fixed seed and measures whether the
saved centroid targets correlate with patch-mask semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset import get_text_and_image_dataset
from model.adapter import ACDCLIP
from model.checkpoint_utils import h6_config_from_checkpoint, load_adapter_checkpoint
from model.clip import create_model
from model.h6.cluster_responsibility import cluster_responsibility_loss
from tools.build_tier3_patch_bank import _h6_args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm(values: list[torch.Tensor]) -> float:
    values = [value.detach().float().pow(2).sum() for value in values if value is not None]
    return float(torch.stack(values).sum().sqrt().item()) if values else 0.0


def _stratified_indices(dataset, per_label: int, seed: int) -> list[int]:
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for index, metadata in enumerate(dataset.meta):
        by_label[int(metadata["label"])].append(index)
    rng = random.Random(seed)
    selected: list[int] = []
    for label in (0, 1):
        rng.shuffle(by_label[label])
        selected.extend(by_label[label][:per_label])
    if not selected:
        raise RuntimeError("the selected training set is empty")
    rng.shuffle(selected)
    return selected


def _patch_coverage(mask: torch.Tensor, groups: int, patches: int) -> torch.Tensor:
    side = int(round(math.sqrt(patches)))
    if side * side != patches:
        raise ValueError(f"cannot map {patches} patches to a square mask grid")
    coverage = F.interpolate(mask.float(), size=(side, side), mode="area").reshape(mask.shape[0], patches)
    return coverage.unsqueeze(0).expand(groups, -1, -1).reshape(-1)


def _construct_model(checkpoint: dict, config: dict, device: torch.device, args) -> ACDCLIP:
    clip_model = create_model(
        args.model_name, args.img_size, device=device, pretrained="openai", require_pretrained=True,
    )
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
    if not model.h6.cluster_ready:
        raise RuntimeError("Tier-3 checkpoint has no materialized centroid buffer")
    # Match the completed Tier-3 training graph.  In particular, frozen CLIP
    # parameters must not retain a full visual/text backward graph during this
    # read-only gradient measurement.
    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.h6.requires_grad_(True)
    model.soft_prompt.requires_grad_(False)
    model.eval()
    return model


def _batch_payload(model: ACDCLIP, item: dict, dataset_name: str, temperature: float):
    device = next(model.parameters()).device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        visual_output = model(item["image"].to(device), return_phase4_features=True)
        h6_batch = model.h6.build_batch(
            model, dataset_name, list(item["class_name"]), visual_output,
            hybrid_alpha=0.0, update_load_bias=False,
        )
        raw_loss, q_cluster, diagnostics = cluster_responsibility_loss(
            h6_batch["router_patch_features"], model.h6.cluster_centroids,
            h6_batch["dense_probabilities"], temperature,
        )
    return visual_output, h6_batch, raw_loss, q_cluster, diagnostics


def _checkpoint_log_summary(run_dir: Path) -> list[dict]:
    rows = []
    for epoch in (1, 2, 3):
        path = run_dir / "diagnostics" / f"epoch_{epoch:03d}.json"
        payload = json.loads(path.read_text())
        losses = payload["loss_components"]
        weighted = float(losses["cluster_resp_weighted"])
        total = float(losses["total"])
        rows.append({
            "epoch": epoch,
            "raw_l_resp": float(losses["cluster_resp_raw"]),
            "effective_coefficient": float(losses["cluster_resp_weight"]),
            "weighted_l_resp": weighted,
            "total_loss": total,
            "weighted_fraction_of_total": weighted / total,
            "target_entropy": float(losses["cluster_target_entropy"]),
            "router_entropy": float(losses["cluster_router_entropy"]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--dataset", default="Brain")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--per_label", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda_device", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.per_label <= 0:
        raise ValueError("--batch_size and --per_label must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = h6_config_from_checkpoint(checkpoint)
    if config is None or not config.get("cluster_responsibility_enabled", False):
        raise ValueError("a Tier-3 Phase-4 checkpoint is required")
    coefficient = float(config["cluster_loss_weight"])
    temperature = float(config["cluster_temperature"])
    if coefficient <= 0.0:
        raise ValueError("checkpoint has no active Tier-3 responsibility coefficient")

    model = _construct_model(checkpoint, config, device, args)
    dataset = get_text_and_image_dataset(args.dataset, args.img_size, "train")
    selected_indices = _stratified_indices(dataset, args.per_label, args.seed)
    loader = DataLoader(Subset(dataset, selected_indices), batch_size=args.batch_size, shuffle=False, num_workers=0)

    cluster_mass = torch.zeros(model.h6.num_factors, dtype=torch.float64)
    anomaly_patch_mass = torch.zeros_like(cluster_mass)
    normal_patch_mass = torch.zeros_like(cluster_mass)
    coverage_sum = torch.zeros(model.h6.num_factors, dtype=torch.float64)
    inside_coverage_sum = torch.zeros(model.h6.num_factors, dtype=torch.float64)
    hard_assignment_count = torch.zeros(model.h6.num_factors, dtype=torch.long)
    factor_anomaly_sum = torch.zeros(model.h6.num_factors, model.h6.num_factors, dtype=torch.float64)
    source_label_count = torch.zeros(model.h6.num_factors, 2, dtype=torch.float64)
    q_variation: list[torch.Tensor] = []
    gradient_audit = None
    first_batch_responsibility = None
    patches_per_image_per_level = None

    for batch_index, item in enumerate(loader):
        if batch_index == 0:
            visual_output, h6_batch, raw_loss, q_cluster, diagnostics = _batch_payload(
                model, item, args.dataset, temperature,
            )
        else:
            with torch.no_grad():
                visual_output, h6_batch, raw_loss, q_cluster, diagnostics = _batch_payload(
                    model, item, args.dataset, temperature,
                )
        if batch_index == 0:
            groups = {
                "router": list(model.h6.router.parameters()),
                "semantic_slots": [model.h6.semantic_core.concept_slots],
                "factor_id_embedding": [model.h6.semantic_core.factor_id_embedding],
                "factor_id_projection": list(model.h6.semantic_core.factor_id_projection.parameters()),
                "semantic_level_projectors": list(model.h6.semantic_core.level_projectors.parameters()),
                "image_adapter": list(model.image_adapter.parameters()),
                "text_adapter": list(model.text_adapter.parameters()),
            }
            unique = []
            seen = set()
            for parameters in groups.values():
                for parameter in parameters:
                    if parameter.requires_grad and id(parameter) not in seen:
                        unique.append(parameter)
                        seen.add(id(parameter))
            grads = torch.autograd.grad(raw_loss * coefficient, unique, retain_graph=False, allow_unused=True)
            by_id = {id(parameter): grad for parameter, grad in zip(unique, grads)}
            gradient_audit = {
                name: _norm([by_id.get(id(parameter)) for parameter in parameters if parameter.requires_grad])
                for name, parameters in groups.items()
            }
            first_batch_responsibility = {
                "raw_l_resp": float(raw_loss.detach().cpu()),
                "weighted_l_resp": float((raw_loss * coefficient).detach().cpu()),
                "target_usage": diagnostics["cluster_target_usage"].cpu().tolist(),
                "router_usage": diagnostics["cluster_router_usage"].cpu().tolist(),
                "target_entropy": float(diagnostics["cluster_target_entropy"].cpu()),
                "router_entropy": float(diagnostics["cluster_router_entropy"].cpu()),
            }
        with torch.no_grad():
            features = h6_batch["router_patch_features"]
            groups_n, batch_n, patches_n, _ = features.shape
            patches_per_image_per_level = patches_n
            coverage = _patch_coverage(item["mask"].to(device), groups_n, patches_n)
            image_labels = item["label"].to(device).view(1, batch_n, 1).expand(groups_n, -1, patches_n).reshape(-1)
            q_flat = q_cluster.reshape(-1, model.h6.num_factors).double()
            assignment = q_flat.argmax(dim=-1)
            q_variation.append(q_cluster.reshape(-1, model.h6.num_factors).float().std(dim=0, unbiased=False).cpu())
            patches = F.normalize(torch.stack(visual_output["seg_tokens"], dim=0).float(), dim=-1)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                # [G,B,P,1,D] against [G,B,1,M,D,2] yields one anomaly
                # logit per patch and factor without using router averaging.
                factor_logits = model.h6.h6_logit(
                    patches.unsqueeze(3), h6_batch["active_factor_bank"].unsqueeze(2)
                )
            # h6_logit already returns abnormal-minus-normal evidence; it has
            # no remaining paired-state dimension.
            factor_anomaly = factor_logits.reshape(-1, model.h6.num_factors)
            for cluster in range(model.h6.num_factors):
                weights = q_flat[:, cluster]
                mass = weights.sum()
                if float(mass) <= 0.0:
                    continue
                anomaly = coverage > 0.0
                cluster_mass[cluster] += mass.cpu()
                anomaly_patch_mass[cluster] += weights[anomaly].sum().cpu()
                normal_patch_mass[cluster] += weights[~anomaly].sum().cpu()
                coverage_sum[cluster] += (weights * coverage.double()).sum().cpu()
                inside_coverage_sum[cluster] += (weights[anomaly] * coverage[anomaly].double()).sum().cpu()
                hard_assignment_count[cluster] += (assignment == cluster).sum().cpu()
                factor_anomaly_sum[cluster] += (weights[:, None] * factor_anomaly.double()).sum(dim=0).cpu()
                for label in (0, 1):
                    source_label_count[cluster, label] += weights[image_labels == label].sum().cpu()
        del visual_output, h6_batch, raw_loss, q_cluster, diagnostics
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cluster_rows = []
    for cluster in range(model.h6.num_factors):
        mass = float(cluster_mass[cluster])
        if mass == 0.0:
            raise RuntimeError(f"cluster {cluster} received no audit patches")
        cluster_rows.append({
            "cluster": cluster,
            "effective_patch_mass": mass,
            "hard_argmax_patch_count": int(hard_assignment_count[cluster]),
            "anomaly_patch_fraction": float(anomaly_patch_mass[cluster] / mass),
            "normal_patch_fraction": float(normal_patch_mass[cluster] / mass),
            "mean_mask_coverage": float(coverage_sum[cluster] / mass),
            "mean_inside_mask_coverage": float(inside_coverage_sum[cluster] / anomaly_patch_mass[cluster].clamp_min(1e-12)),
            "mean_outside_mask_coverage": float(1.0 - coverage_sum[cluster] / mass),
            "source_image_label_distribution": {
                "normal_effective_patch_mass": float(source_label_count[cluster, 0]),
                "anomaly_effective_patch_mass": float(source_label_count[cluster, 1]),
            },
            "dataset_distribution": {args.dataset: mass},
            "factor_anomaly_logit_mean": (factor_anomaly_sum[cluster] / mass).tolist(),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "audit_kind": "read_only_tier3_checkpoint_and_cluster_semantics",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "dataset": args.dataset,
        "training_preprocessing": "TextAndImageDataset train transform with fixed seed",
        "sample": {
            "seed": args.seed,
            "selected_images": len(selected_indices),
            "per_label_requested": args.per_label,
            "selected_image_labels": dict(Counter(int(dataset.meta[index]["label"]) for index in selected_indices)),
            "patches_per_image_per_level": patches_per_image_per_level,
        },
        "responsibility": {
            "configured_coefficient": coefficient,
            "temperature": temperature,
            "formula": "total_loss includes configured_coefficient * KL(q_cluster || dense_router_probs)",
            "schedule": "none; the coefficient is the direct --h6_lambda_cluster_resp checkpoint value",
            "epoch_log_means": _checkpoint_log_summary(args.run_dir),
            "first_audit_batch": first_batch_responsibility,
            "weighted_l_resp_gradient_norm": gradient_audit,
        },
        "cluster_semantics": {
            "assignment": "soft detached q_cluster weighting against saved centroids; hard argmax count is diagnostic only",
            "anomaly_patch": "downsampled training mask coverage > 0 at the 37x37 router grid",
            "mean_outside_mask_coverage": "1 - mean_mask_coverage over assigned patches",
            "q_cluster_std_by_factor": torch.stack(q_variation).mean(dim=0).tolist(),
            "clusters": cluster_rows,
        },
    }
    (args.output_dir / "tier3_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
