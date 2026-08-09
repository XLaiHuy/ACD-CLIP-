#!/usr/bin/env python3
"""Optimizer-step-matched, no-step loss-semantics audit for P1-v8.3.

The audit consumes natural seed-0 VisA training samples in consecutive groups
of six.  Every counterfactual loss is differentiated through the same forward
graph.  It never constructs an optimizer and never writes ``parameter.grad``.
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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from dataset import get_text_and_image_dataset
from model.h6.utility_routing import (
    build_patch_targets,
    effective_number_utility_factor_loss,
    support_normalized_utility_router_loss,
    utility_factor_loss,
    utility_router_loss,
    utility_teacher,
)
from tools.audit_p1_v83_semantics import _model_from_checkpoint
from train import h6_drift_parameter_groups
from utils import calculate_seg_loss, get_phase2b_global_text_features


FACTOR_FORMULAS = {
    "F0_region_mean_50_50": None,
    "F1_effective_beta_0.99": 0.99,
    "F1_effective_beta_0.999": 0.999,
    "F1_effective_beta_0.9999": 0.9999,
}
ROUTER_FORMULAS = ("R0_informative_mean", "R1_valid_support_mean")
GROUP_NAMES = ("shared_semantic", "state_path", "text_lora", "vae_class", "router")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameter_hash(parameters: list[torch.nn.Parameter]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def effective_number_factor_loss(
    payload: dict[str, torch.Tensor], y_patch: torch.Tensor, beta: float
) -> torch.Tensor:
    """One patch-weighted valid mean using inverse effective region counts."""
    return effective_number_utility_factor_loss(payload, y_patch, beta=beta)


def support_aware_router_loss(
    dense_probabilities: torch.Tensor, payload: dict[str, torch.Tensor]
) -> torch.Tensor:
    """FixMatch-style masked CE divided by all valid patch support."""
    return support_normalized_utility_router_loss(dense_probabilities, payload)


def _factor_region_stats(payload: dict[str, torch.Tensor], y_patch: torch.Tensor) -> dict:
    per_patch = (
        payload["responsibility"].detach() * payload["loss_per_factor"].detach()
    ).sum(dim=-1)
    targets = y_patch.unsqueeze(0).expand_as(per_patch)
    valid = payload["valid"]
    result = {}
    for name, region in (
        ("normal", valid & (targets < 0.5)),
        ("anomaly", valid & (targets >= 0.5)),
    ):
        values = per_patch[region]
        result[name] = {
            "count_group_patches": int(values.numel()),
            "loss_sum": float(values.sum().item()) if values.numel() else 0.0,
            "loss_mean": float(values.mean().item()) if values.numel() else None,
        }
    return result


def _layout(model) -> tuple[dict, list[torch.nn.Parameter], dict[int, list[tuple[str, int, int]]]]:
    source_groups = h6_drift_parameter_groups(model)
    groups = {
        "shared_semantic": [p for p in source_groups["shared_semantic"] if p.requires_grad],
        "state_path": [p for p in source_groups["state_path"] if p.requires_grad],
        "text_lora": [p for p in source_groups["text_lora"] if p.requires_grad],
        "vae_class": [p for p in source_groups["vae_class_path"] if p.requires_grad],
        "router": [p for p in source_groups["router"] if p.requires_grad],
    }
    unique, seen = [], set()
    for parameters in groups.values():
        for parameter in parameters:
            if id(parameter) not in seen:
                unique.append(parameter)
                seen.add(id(parameter))
    slices: dict[int, list[tuple[str, int, int]]] = {}
    for group_name, parameters in groups.items():
        offset = 0
        for parameter in parameters:
            end = offset + parameter.numel()
            slices.setdefault(id(parameter), []).append((group_name, offset, end))
            offset = end
    metadata = {
        name: {
            "tensor_count": len(parameters),
            "parameter_count": sum(parameter.numel() for parameter in parameters),
        }
        for name, parameters in groups.items()
    }
    return metadata, unique, slices


def _blank_window_vectors(components: tuple[str, ...], group_meta: dict) -> dict:
    return {
        component: {
            group: torch.zeros(meta["parameter_count"], dtype=torch.float32)
            for group, meta in group_meta.items()
        }
        for component in components
    }


def _accumulate_component(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    slices: dict[int, list[tuple[str, int, int]]],
    destination: dict[str, torch.Tensor],
    *,
    retain_graph: bool,
    divisor: float,
) -> None:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        flat = gradient.detach().float().reshape(-1).cpu() / float(divisor)
        for group_name, start, end in slices[id(parameter)]:
            destination[group_name][start:end].add_(flat)


def _geometry(vectors: dict[str, dict[str, torch.Tensor]]) -> dict:
    result = {}
    factor_names = tuple(FACTOR_FORMULAS)
    for group_name in GROUP_NAMES:
        main = vectors["main"][group_name]
        main_norm = float(main.norm().item())
        entry = {"main_norm": main_norm, "factors": {}, "routers": {}, "pairs": {}}
        for name in factor_names:
            vector = vectors[name][group_name]
            norm = float(vector.norm().item())
            dot = float(torch.dot(main, vector).item())
            entry["factors"][name] = {
                "norm": norm,
                "to_main": None if main_norm <= 1e-12 else norm / main_norm,
                "dot_main": dot,
                "cos_main": None if main_norm <= 1e-12 or norm <= 1e-12 else dot / (main_norm * norm),
            }
        for name in ROUTER_FORMULAS:
            vector = vectors[name][group_name]
            norm = float(vector.norm().item())
            dot = float(torch.dot(main, vector).item())
            entry["routers"][name] = {
                "norm": norm,
                "to_main": None if main_norm <= 1e-12 else norm / main_norm,
                "dot_main": dot,
                "cos_main": None if main_norm <= 1e-12 or norm <= 1e-12 else dot / (main_norm * norm),
            }
        for factor_name in factor_names:
            factor = vectors[factor_name][group_name]
            for router_name in ROUTER_FORMULAS:
                router = vectors[router_name][group_name]
                entry["pairs"][f"{factor_name}+{router_name}"] = {
                    "dot_factor_router": float(torch.dot(factor, router).item())
                }
        result[group_name] = entry
    return result


def _batch_losses(model, checkpoint: dict, sample: dict, device: torch.device) -> tuple[dict, dict]:
    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    label = sample["label"].reshape(1).to(device)
    local_valid = sample.get("local_mask_valid", torch.ones_like(sample["mask"]))
    local_valid = local_valid.unsqueeze(0).to(device)
    class_names = [sample["class_name"]]

    visual = model(image, return_phase4_features=True)
    h6_batch = model.h6.build_batch(
        model,
        "VisA",
        class_names,
        visual,
        hybrid_alpha=float(checkpoint["hybrid_alpha_current"]),
        update_load_bias=False,
    )
    seg_features = torch.stack(visual["seg_tokens"], dim=0)
    det_features = torch.stack(visual["det_tokens"], dim=0)
    text_global = get_phase2b_global_text_features(
        model,
        "VisA",
        class_names,
        device,
        use_hybrid_soft_prompt=True,
        use_soft_prompt=False,
    ).to(dtype=det_features.dtype)
    cls_pred = torch.stack(
        [
            torch.matmul(det_features[level].unsqueeze(1), text_global[level]).squeeze(1)
            for level in range(model.n_groups)
        ],
        dim=0,
    ).mean(dim=0)
    cls_loss = F.cross_entropy(cls_pred.float(), label)
    seg_pred, _, base_margin = model.vision_text_fusion_gate_seg(
        seg_features,
        text_global,
        img_size=int(checkpoint["img_size"]),
        h6_patch_logits=h6_batch["h6_logits"],
        return_details=True,
    )
    task_loss = cls_loss + calculate_seg_loss(seg_pred.float(), mask.float())
    patch_count = int(h6_batch["factor_patch_logits"].shape[2])
    y_patch, valid_patch = build_patch_targets(mask, patch_count, local_valid)
    payload = utility_teacher(
        base_margin.detach(),
        h6_batch["factor_patch_logits"],
        y_patch,
        valid_patch,
        rho=0.05,
        denominator_floor=0.10,
        tau_utility=0.05,
        epsilon=0.15,
        gain_threshold=0.02,
        entropy_threshold=0.98,
    )
    losses = {"main": task_loss, "F0_region_mean_50_50": utility_factor_loss(payload, y_patch)}
    for name, beta in FACTOR_FORMULAS.items():
        if beta is not None:
            losses[name] = effective_number_factor_loss(payload, y_patch, beta)
    losses["R0_informative_mean"] = utility_router_loss(h6_batch["dense_probabilities"], payload)
    losses["R1_valid_support_mean"] = support_aware_router_loss(h6_batch["dense_probabilities"], payload)

    physical_valid = valid_patch
    physical_anomaly = physical_valid & (y_patch >= 0.5)
    informative = payload["informative"]
    stats = {
        "dataset_index": int(sample["dataset_index"]),
        "class_name": sample["class_name"],
        "image_label": int(label.item()),
        "normal_patch_count": int((physical_valid & ~physical_anomaly).sum().item()),
        "anomaly_patch_count": int(physical_anomaly.sum().item()),
        "valid_patch_count": int(physical_valid.sum().item()),
        "informative_group_patch_count": int(informative.sum().item()),
        "valid_group_patch_count": int(payload["valid"].sum().item()),
        "informative_physical_patch_count": int(informative.any(dim=0).sum().item()),
        "losses": {name: float(loss.detach().item()) for name, loss in losses.items()},
        "factor_regions": _factor_region_stats(payload, y_patch),
    }
    return losses, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/p1_v83_dev/root_cause_optimizer_audit")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-windows", type=int, default=24)
    parser.add_argument("--max-windows", type=int, default=96)
    parser.add_argument("--required-window-types", type=int, default=3)
    args = parser.parse_args()
    if args.min_windows < 24 or args.max_windows < args.min_windows or args.max_windows > 96:
        raise ValueError("window bounds must satisfy 24 <= min <= max <= 96")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _seed_everything(args.seed)
    device = torch.device("cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = _model_from_checkpoint(checkpoint, device)
    model.clipmodel.set_grad_checkpointing(True)
    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.soft_prompt.requires_grad_(False)
    model.h6.requires_grad_(True)
    model.h6.rho.raw.requires_grad_(False)
    model.eval()
    model.clipmodel.eval()

    group_meta, parameters, slices = _layout(model)
    before_hash = _parameter_hash(parameters)
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("audit requires clear parameter.grad fields")

    dataset = get_text_and_image_dataset("VisA", int(checkpoint["img_size"]), "train")
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    components = ("main", *tuple(FACTOR_FORMULAS), *ROUTER_FORMULAS)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    windows = []
    category_counts = {"normal_only": 0, "anomaly_containing": 0, "router_informative": 0}
    cursor = 0

    while len(windows) < args.max_windows:
        vectors = _blank_window_vectors(components, group_meta)
        microbatches = []
        loss_sums = {name: 0.0 for name in components}
        for _ in range(6):
            dataset_index = order[cursor]
            cursor += 1
            sample = dataset[dataset_index]
            sample["dataset_index"] = dataset_index
            losses, stats = _batch_losses(model, checkpoint, sample, device)
            microbatches.append(stats)
            for name in components:
                loss_sums[name] += float(losses[name].detach().item()) / 6.0
            for component_index, name in enumerate(components):
                _accumulate_component(
                    losses[name],
                    parameters,
                    slices,
                    vectors[name],
                    retain_graph=component_index < len(components) - 1,
                    divisor=6.0,
                )
            del losses

        anomaly_count = sum(item["anomaly_patch_count"] for item in microbatches)
        normal_count = sum(item["normal_patch_count"] for item in microbatches)
        informative_count = sum(item["informative_group_patch_count"] for item in microbatches)
        valid_count = anomaly_count + normal_count
        valid_group_count = sum(item["valid_group_patch_count"] for item in microbatches)
        categories = {
            "normal_only": anomaly_count == 0,
            "anomaly_containing": anomaly_count > 0,
            "router_informative": informative_count > 0,
        }
        for name, present in categories.items():
            category_counts[name] += int(present)
        window = {
            "window_index": len(windows) + 1,
            "sample_order_start": cursor - 6,
            "sample_order_end": cursor - 1,
            "dataset_indices": [item["dataset_index"] for item in microbatches],
            "categories": categories,
            "counts": {
                "normal_patch_count": normal_count,
                "anomaly_patch_count": anomaly_count,
                "valid_patch_count": valid_count,
                "anomaly_fraction": anomaly_count / valid_count if valid_count else 0.0,
                "informative_group_patch_count": informative_count,
                "valid_group_patch_count": valid_group_count,
                "informative_fraction": informative_count / valid_group_count if valid_group_count else 0.0,
            },
            "losses": loss_sums,
            "microbatches": microbatches,
            "geometry": _geometry(vectors),
        }
        windows.append(window)
        progress = {
            "status": "RUNNING",
            "windows_completed": len(windows),
            "category_counts": category_counts,
            "last_window": window["window_index"],
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json(output_dir / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        enough_types = all(
            count >= args.required_window_types for count in category_counts.values()
        )
        if len(windows) >= args.min_windows and enough_types:
            break

    after_hash = _parameter_hash(parameters)
    grad_fields_clear = all(parameter.grad is None for parameter in parameters)
    summary = {
        "schema_version": 1,
        "status": "PASS" if before_hash == after_hash and grad_fields_clear else "FAIL",
        "audit": "optimizer_step_matched_no_step_counterfactual",
        "source_head": os.popen("git rev-parse HEAD").read().strip(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "seed": args.seed,
        "ordering": "torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))",
        "batch_size": 1,
        "grad_accum_steps": 6,
        "precision": "fp32",
        "tf32": False,
        "amp": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "windows_completed": len(windows),
        "category_counts": category_counts,
        "required_window_types_each": args.required_window_types,
        "parameter_groups": group_meta,
        "factor_formulas": {
            name: ({"kind": "current_region_mean_50_50"} if beta is None else {
                "kind": "inverse_effective_number_patch_weighted_valid_mean", "beta": beta
            })
            for name, beta in FACTOR_FORMULAS.items()
        },
        "router_formulas": {
            "R0_informative_mean": "mean CE over informative group-patches",
            "R1_valid_support_mean": "sum masked CE divided by all valid group-patches",
        },
        "state_integrity": {
            "parameter_hash_before": before_hash,
            "parameter_hash_after": after_hash,
            "parameter_state_unchanged": before_hash == after_hash,
            "parameter_grad_fields_clear": grad_fields_clear,
        },
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json(output_dir / "optimizer_windows.json", {"summary": summary, "windows": windows})
    _write_json(output_dir / "audit_summary.json", summary)
    _write_json(output_dir / "progress.json", {**summary, "status": summary["status"]})
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
