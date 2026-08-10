#!/usr/bin/env python3
"""Fresh-init, six-microbatch, no-step gradient calibration for P1-v8.4-A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
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
from model.adapter import ACDCLIP
from model.checkpoint_utils import h6_config_from_checkpoint
from model.clip import create_model
from model.h6.utility_routing import (
    act_teacher,
    build_patch_targets,
    effective_number_act_loss,
    effective_number_utility_factor_loss,
    support_normalized_utility_router_loss,
    utility_teacher,
)
from train import h6_drift_parameter_groups
from utils import calculate_seg_loss, get_phase2b_global_text_features


COMPONENTS = ("main", "factor", "router", "act")
GROUPS = (
    "shared_semantic", "state_path", "text_lora", "vae_class",
    "router", "act_head", "image_adapter",
)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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


def _fresh_model(protocol: dict, device: torch.device, seed: int) -> ACDCLIP:
    _seed(seed)
    config = protocol["phase2b_config"]
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=int(protocol["img_size"]),
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    h6_config = h6_config_from_checkpoint(protocol)
    h6_config["progress_version"] = "P1-v8.4-A"
    h6_kwargs = {f"h6_{name}": value for name, value in h6_config.items()}
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=int(protocol["n_groups"]),
        image_adapt_weight=float(config["image_adapt_weight"]),
        conv_lora_rank=int(config["conv_lora_rank"]),
        conv_lora_alpha=float(config["conv_lora_alpha"]),
        conv_kernel_size_list=list(config["conv_kernel_size_list"]),
        text_adapt_weight=float(config["text_adapt_weight"]),
        lora_rank=int(config["lora_rank"]),
        lora_alpha=float(config["lora_alpha"]),
        dfg_mode=protocol["dfg_mode"],
        dfg_attn_dim=int(protocol["dfg_attn_dim"]),
        dfg_attn_tau=float(protocol["dfg_attn_tau"]),
        use_ss2d_dfg=bool(protocol["use_ss2d_dfg"]),
        dfg_gamma_max=float(protocol["dfg_gamma_max"]),
        dfg_ss2d_fusion=protocol["dfg_ss2d_fusion"],
        dfg_beta=float(protocol["dfg_beta"]),
        dfg_beta_schedule=protocol["dfg_beta_schedule"],
        dfg_beta_target=float(protocol["dfg_beta_target"]),
        dfg_beta_current=float(protocol["dfg_beta"]),
        use_soft_prompt=bool(protocol["use_soft_prompt"]),
        soft_prompt_ctx_len=int(protocol["soft_prompt_ctx_len"]),
        soft_prompt_init=protocol["soft_prompt_init"],
        soft_prompt_init_phrase=protocol["soft_prompt_init_phrase"],
        **h6_kwargs,
    ).to(device)
    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.soft_prompt.requires_grad_(False)
    model.h6.requires_grad_(True)
    model.h6.rho.raw.requires_grad_(False)
    model.clipmodel.set_grad_checkpointing(True)
    model.train()
    model.clipmodel.eval()
    model.prompt_mode = "h6_dynamic"
    model.use_soft_prompt = False
    model.use_hybrid_soft_prompt = True
    model.hybrid_alpha_current = float(protocol["hybrid_alpha_current"])
    model.h6_global_text_mode = "phase2b_hybrid"
    model.set_dfg_beta(float(protocol["dfg_beta"]))
    model.h6.set_epoch(1)
    return model


def _layout(model: ACDCLIP):
    source = h6_drift_parameter_groups(model)
    groups = {
        "shared_semantic": [p for p in source["shared_semantic"] if p.requires_grad],
        "state_path": [p for p in source["state_path"] if p.requires_grad],
        "text_lora": [p for p in source["text_lora"] if p.requires_grad],
        "vae_class": [p for p in source["vae_class_path"] if p.requires_grad],
        "router": [p for p in source["router"] if p.requires_grad],
        "act_head": [p for p in source["act_head"] if p.requires_grad],
        "image_adapter": [p for p in model.image_adapter.parameters() if p.requires_grad],
    }
    unique, seen = [], set()
    for parameters in groups.values():
        for parameter in parameters:
            if id(parameter) not in seen:
                unique.append(parameter)
                seen.add(id(parameter))
    locations: dict[int, list[tuple[str, int, int]]] = {}
    metadata = {}
    for group, parameters in groups.items():
        offset = 0
        for parameter in parameters:
            end = offset + parameter.numel()
            locations.setdefault(id(parameter), []).append((group, offset, end))
            offset = end
        metadata[group] = {"tensor_count": len(parameters), "parameter_count": offset}
    return metadata, unique, locations


def _vectors(metadata: dict) -> dict:
    return {
        component: {
            group: torch.zeros(entry["parameter_count"], dtype=torch.float32)
            for group, entry in metadata.items()
        }
        for component in COMPONENTS
    }


def _accumulate(loss, parameters, locations, destination, *, retain_graph):
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        flat = gradient.detach().float().reshape(-1).cpu() / 6.0
        for group, start, end in locations[id(parameter)]:
            destination[group][start:end].add_(flat)


def _losses(
    model,
    sample: dict,
    device: torch.device,
    *,
    act_gain_threshold: float,
):
    image = sample["image"].unsqueeze(0).to(device)
    mask = sample["mask"].unsqueeze(0).to(device)
    label = sample["label"].reshape(1).to(device)
    local_valid = sample.get("local_mask_valid", torch.ones_like(sample["mask"]))
    local_valid = local_valid.unsqueeze(0).to(device)
    class_names = [sample["class_name"]]
    visual = model(image, return_phase4_features=True)
    h6_batch = model.h6.build_batch(
        model, "VisA", class_names, visual,
        hybrid_alpha=model.hybrid_alpha_current, update_load_bias=False,
    )
    seg_features = torch.stack(visual["seg_tokens"], dim=0)
    det_features = torch.stack(visual["det_tokens"], dim=0)
    text_global = get_phase2b_global_text_features(
        model, "VisA", class_names, device,
        use_hybrid_soft_prompt=True, use_soft_prompt=False,
    ).to(dtype=det_features.dtype)
    cls_pred = torch.stack([
        torch.matmul(det_features[level].unsqueeze(1), text_global[level]).squeeze(1)
        for level in range(model.n_groups)
    ], dim=0).mean(dim=0)
    cls_loss = F.cross_entropy(cls_pred.float(), label)
    seg_pred, _, base_margin = model.vision_text_fusion_gate_seg(
        seg_features, text_global, img_size=518,
        h6_patch_logits=h6_batch["h6_logits"], return_details=True,
    )
    main = cls_loss + calculate_seg_loss(seg_pred.float(), mask.float())
    patch_count = int(h6_batch["factor_residual_logits"].shape[2])
    y_patch, valid = build_patch_targets(mask, patch_count, local_valid)
    utility = utility_teacher(
        base_margin.detach(), h6_batch["factor_residual_logits"], y_patch, valid,
        rho=0.05, denominator_floor=0.10, tau_utility=0.05,
        epsilon=0.15, gain_threshold=0.02, entropy_threshold=0.98,
        routed_probabilities=h6_batch["prediction_probabilities"],
    )
    act = act_teacher(utility, gain_threshold=act_gain_threshold)
    losses = {
        "main": main,
        "factor": effective_number_utility_factor_loss(utility, y_patch, beta=0.999),
        "router": support_normalized_utility_router_loss(
            h6_batch["dense_probabilities"], utility
        ),
        "act": effective_number_act_loss(
            h6_batch["act_logits"], act, y_patch, beta=0.999
        ),
    }
    physical_anomaly = valid & (y_patch >= 0.5)
    return losses, {
        "normal_patch_count": int((valid & ~physical_anomaly).sum().item()),
        "anomaly_patch_count": int(physical_anomaly.sum().item()),
        "act_positive_group_patch_count": int(act["positive"].sum().item()),
        "act_negative_group_patch_count": int(act["negative"].sum().item()),
        "act_ambiguous_group_patch_count": int(act["ambiguous"].sum().item()),
        "router_informative_group_patch_count": int(utility["informative"].sum().item()),
        "losses": {name: float(value.detach().item()) for name, value in losses.items()},
    }


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float | None:
    denominator = float(first.norm().item() * second.norm().item())
    return None if denominator <= 1e-12 else float(torch.dot(first, second).item() / denominator)


def _window_geometry(vectors: dict) -> dict:
    result = {}
    for group in GROUPS:
        main = vectors["main"][group]
        main_norm = float(main.norm().item())
        components = {}
        for component in ("factor", "router", "act"):
            vector = vectors[component][group]
            norm = float(vector.norm().item())
            components[component] = {
                "raw_norm": norm,
                "raw_norm_to_main": None if main_norm <= 1e-12 else norm / main_norm,
                "cos_main": _cosine(main, vector),
            }
        result[group] = {"main_norm": main_norm, "components": components}
    return result


def _quantiles(values: list[float]) -> dict:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": len(values),
        "median": float(torch.quantile(tensor, 0.50).item()),
        "p75": float(torch.quantile(tensor, 0.75).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
        "max": float(tensor.max().item()),
    }


def _select_act_lambda(windows: list[dict]) -> tuple[float, dict]:
    ratios = []
    for window in windows:
        # At the preregistered zero initialization the ACT linear weights
        # intentionally block ACT-loss gradients into patch features. The ACT
        # head itself is the nonzero common main/ACT parameter group at step 0.
        ratio = window["geometry"]["act_head"]["components"]["act"]["raw_norm_to_main"]
        if ratio is not None and ratio > 0.0 and math.isfinite(ratio):
            ratios.append(ratio)
    if not ratios:
        raise RuntimeError("ACT has no nonzero common-window gradient on the image adapter")
    raw = _quantiles(ratios)
    # Analytic stability envelope, not a training sweep: never exceed main in
    # an observed window, keep p95 at or below half of main, and cap at 1.
    selected = min(1.0, 1.0 / raw["max"], 0.5 / raw["p95"])
    if not math.isfinite(selected) or selected <= 0.0:
        raise RuntimeError("ACT lambda cannot be calibrated to a positive stable value")
    weighted = _quantiles([selected * value for value in ratios])
    return selected, {
        "group": "act_head",
        "raw_norm_to_main": raw,
        "selected_weighted_norm_to_main": weighted,
        "selection_formula": "min(1, 1/raw_max, 0.5/raw_p95)",
        "stability_contract": "observed max<=1 and p95<=0.5 of main",
        "initial_feature_path_note": (
            "ACT-to-image-adapter raw gradient is exactly zero at zero-initialized "
            "linear weights; it becomes reachable after the head leaves zero"
        ),
    }


def _summaries(windows: list[dict], act_lambda: float) -> dict:
    weights = {"factor": 0.03, "router": 0.10, "act": act_lambda}
    output = {"weights": weights, "groups": {}}
    for group in GROUPS:
        component_summary = {}
        combined_ratios = []
        for component, weight in weights.items():
            raw_ratios, weighted_ratios, cosines = [], [], []
            for window in windows:
                entry = window["geometry"][group]
                ratio = entry["components"][component]["raw_norm_to_main"]
                cosine = entry["components"][component]["cos_main"]
                if ratio is not None:
                    raw_ratios.append(ratio)
                    weighted_ratios.append(weight * ratio)
                if cosine is not None:
                    cosines.append(cosine)
            component_summary[component] = {
                "raw_norm_to_main": _quantiles(raw_ratios) if raw_ratios else None,
                "weighted_norm_to_main": _quantiles(weighted_ratios) if weighted_ratios else None,
                "cos_main": _quantiles(cosines) if cosines else None,
            }
        for window in windows:
            vectors = window["_vectors"]
            main = vectors["main"][group]
            main_norm = float(main.norm().item())
            combined = sum(
                weights[name] * vectors[name][group] for name in ("factor", "router", "act")
            )
            if main_norm > 1e-12:
                combined_ratios.append(float(combined.norm().item()) / main_norm)
        output["groups"][group] = {
            "components": component_summary,
            "true_combined_weighted_aux_to_main": (
                _quantiles(combined_ratios) if combined_ratios else None
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol-checkpoint", type=Path,
        default=Path("runs/p1_v83_dev/corrected_300b_primary_anchored_attempt1/adapter_1.pth"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/p1_v83_dev/v84a_gradient_calibration"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--windows", type=int, default=24)
    parser.add_argument("--act-gain-threshold", type=float, default=0.0)
    args = parser.parse_args()
    if args.windows < 24 or args.windows > 50:
        raise ValueError("natural calibration requires 24..50 six-microbatch windows")
    if args.act_gain_threshold < 0.0:
        raise ValueError("--act-gain-threshold must be non-negative")
    if (args.output_dir / "calibration_summary.json").exists():
        raise FileExistsError("refusing to overwrite completed gradient calibration")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    protocol = torch.load(args.protocol_checkpoint, map_location="cpu")
    model = _fresh_model(protocol, device, args.seed)
    metadata, parameters, locations = _layout(model)
    before_hash = _parameter_hash(parameters)
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("calibration requires clear parameter.grad fields")
    dataset = get_text_and_image_dataset("VisA", 518, "train")
    order = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(args.seed)
    ).tolist()
    started = time.monotonic()
    windows = []
    cursor = 0
    for window_index in range(args.windows):
        vectors = _vectors(metadata)
        counts = {
            "normal_patch_count": 0, "anomaly_patch_count": 0,
            "act_positive_group_patch_count": 0, "act_negative_group_patch_count": 0,
            "act_ambiguous_group_patch_count": 0,
            "router_informative_group_patch_count": 0,
        }
        losses_mean = {name: 0.0 for name in COMPONENTS}
        indices = []
        for _ in range(6):
            index = order[cursor]
            cursor += 1
            indices.append(index)
            losses, stats = _losses(
                model,
                dataset[index],
                device,
                act_gain_threshold=args.act_gain_threshold,
            )
            for name in COMPONENTS:
                losses_mean[name] += float(losses[name].detach().item()) / 6.0
            for key in counts:
                counts[key] += stats[key]
            for component_index, name in enumerate(COMPONENTS):
                _accumulate(
                    losses[name], parameters, locations, vectors[name],
                    retain_graph=component_index < len(COMPONENTS) - 1,
                )
            del losses
        window = {
            "window": window_index + 1,
            "dataset_indices": indices,
            "counts": counts,
            "losses": losses_mean,
            "geometry": _window_geometry(vectors),
            "_vectors": vectors,
        }
        windows.append(window)
        progress = {
            "status": "RUNNING", "windows_completed": len(windows),
            "elapsed_seconds": time.monotonic() - started,
        }
        _write_json(args.output_dir / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    act_lambda, act_selection = _select_act_lambda(windows)
    gradient_summary = _summaries(windows, act_lambda)
    label_totals = {
        key: sum(window["counts"][key] for window in windows)
        for key in (
            "normal_patch_count", "anomaly_patch_count",
            "act_positive_group_patch_count", "act_negative_group_patch_count",
            "act_ambiguous_group_patch_count", "router_informative_group_patch_count",
        )
    }
    label_totals["normal_group_patch_count"] = (
        label_totals["normal_patch_count"] * int(model.n_groups)
    )
    label_totals["anomaly_group_patch_count"] = (
        label_totals["anomaly_patch_count"] * int(model.n_groups)
    )
    label_totals["valid_group_patch_count"] = (
        label_totals["normal_group_patch_count"]
        + label_totals["anomaly_group_patch_count"]
    )
    label_totals["act_positive_fraction"] = (
        label_totals["act_positive_group_patch_count"]
        / max(label_totals["valid_group_patch_count"], 1)
    )
    label_totals["act_negative_fraction"] = (
        label_totals["act_negative_group_patch_count"]
        / max(label_totals["valid_group_patch_count"], 1)
    )
    label_totals["act_ambiguous_fraction"] = (
        label_totals["act_ambiguous_group_patch_count"]
        / max(label_totals["valid_group_patch_count"], 1)
    )
    after_hash = _parameter_hash(parameters)
    integrity = before_hash == after_hash and all(p.grad is None for p in parameters)
    serializable_windows = [
        {key: value for key, value in window.items() if key != "_vectors"}
        for window in windows
    ]
    summary = {
        "status": "PASS" if integrity else "FAIL",
        "audit": "fresh_p1_v84a_natural_six_microbatch_no_step_gradient_calibration",
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "protocol_checkpoint": str(args.protocol_checkpoint.resolve()),
        "protocol_checkpoint_sha256": _sha256(args.protocol_checkpoint),
        "initialization": "fresh_openai_clip_seed0_no_checkpoint_weights_loaded",
        "progress_version": "P1-v8.4-A",
        "dataset": "VisA/train",
        "seed": args.seed,
        "image_size": 518,
        "batch_size": 1,
        "gradient_accumulation_steps": 6,
        "gradient_checkpointing": True,
        "windows": args.windows,
        "microbatches_per_window": 6,
        "optimizer_constructed": False,
        "backward_executed": False,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "parameter_mutation": False,
        "precision": "fp32",
        "tf32": False,
        "amp": False,
        "rho": 0.05,
        "act_teacher_utility_object": "g_route",
        "parameter_groups": metadata,
        "selected_lambda_factor": 0.03,
        "selected_lambda_router": 0.10,
        "selected_lambda_act": act_lambda,
        "act_gain_threshold": args.act_gain_threshold,
        "act_label_statistics": label_totals,
        "act_lambda_calibration": act_selection,
        "gradient_summary": gradient_summary,
        "gradient_safety_contract": {
            "raw_gradient_ratios_finite": True,
            "weighted_max_le_one": (
                gradient_summary["groups"]["act_head"]["components"]["act"]
                ["weighted_norm_to_main"]["max"] <= 1.0
            ),
            "weighted_p95_le_half": (
                gradient_summary["groups"]["act_head"]["components"]["act"]
                ["weighted_norm_to_main"]["p95"] <= 0.5
            ),
        },
        "state_integrity": {
            "parameter_hash_before": before_hash,
            "parameter_hash_after": after_hash,
            "unchanged": integrity,
            "grad_fields_clear": all(p.grad is None for p in parameters),
        },
        "runtime_seconds": time.monotonic() - started,
    }
    _write_json(args.output_dir / "optimizer_windows.json", {
        "summary": summary, "windows": serializable_windows,
    })
    _write_json(args.output_dir / "calibration_summary.json", summary)
    _write_json(args.output_dir / "progress.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
