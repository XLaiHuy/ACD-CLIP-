#!/usr/bin/env python3
"""Inference-only Stage 1.5 attribution for the trained K1 checkpoint."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import get_text_and_image_dataset
from model.adapter import ACDCLIP
from model.checkpoint_utils import load_adapter_checkpoint
from model.clip import create_model
from utils import (
    configure_canonical_fp32,
    get_phase2b_global_text_features,
    make_dataloader_generator,
    seed_worker,
)


CONDITIONS = {
    "FULL": (1.0, 1.0),
    "STATE_ONLY": (1.0, 0.0),
    "CLASS_ONLY": (0.0, 1.0),
}


def _mean(values):
    values = torch.cat(values).float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
    }


def _pearson(x, y):
    x, y = torch.cat(x).float(), torch.cat(y).float()
    if x.numel() < 2 or x.std(unbiased=False) == 0 or y.std(unbiased=False) == 0:
        return None
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1])


def _condition_metrics(store):
    result = {}
    for condition, regions in store.items():
        base = {region: _mean(values) for region, values in regions["base"].items()}
        dynamic = {region: _mean(values) for region, values in regions["dynamic"].items()}
        final = {region: _mean(values) for region, values in regions["final"].items()}
        delta = {region: _mean(values) for region, values in regions["delta"].items()}
        normal_abs = delta["normal"]["mean"]
        anomaly_abs = delta["anomaly"]["mean"]
        result[condition] = {
            "bce": {"base": base, "dynamic": dynamic, "final": final},
            "gain_base_minus_final": {
                region: base[region]["mean"] - final[region]["mean"]
                for region in ("normal", "anomaly", "all")
            },
            "residual": {
                "mean": _mean(regions["delta_signed"])["mean"],
                "std": _mean(regions["delta_signed"])["std"],
                "mean_abs_delta_normal": normal_abs,
                "mean_abs_delta_anomaly": anomaly_abs,
                "max_abs_delta": float(torch.cat(regions["delta"]["all"]).max()),
                "positive_fraction": float(torch.cat(regions["delta_signed"]).gt(0).float().mean()),
                "negative_fraction": float(torch.cat(regions["delta_signed"]).lt(0).float().mean()),
                "selectivity_ratio": anomaly_abs / max(normal_abs, 1e-12),
                "mean_signed_delta_normal": _mean(regions["delta_signed_normal"])["mean"],
                "mean_signed_delta_anomaly": _mean(regions["delta_signed_anomaly"])["mean"],
            },
            "confidence_correlation": {
                "abs_delta_vs_base_anomaly_probability_normal": _pearson(
                    regions["confidence_normal"], regions["delta_normal"]
                ),
                "abs_delta_vs_base_anomaly_probability_anomaly": _pearson(
                    regions["confidence_anomaly"], regions["delta_anomaly"]
                ),
            },
        }
    return result


def _decision(metrics):
    gain = {name: value["gain_base_minus_final"] for name, value in metrics.items()}
    useful = lambda name: gain[name]["anomaly"] > 0.0
    safe = lambda name: gain[name]["normal"] >= 0.0
    if useful("STATE_ONLY") and safe("STATE_ONLY") and (
        not useful("CLASS_ONLY") or not safe("CLASS_ONLY")
    ) and gain["FULL"]["normal"] < gain["STATE_ONLY"]["normal"]:
        return "VAE_CLASS_NOT_JUSTIFIED", "K1b = CoPS STATE-only conditional semantic residual"
    if useful("CLASS_ONLY") and safe("CLASS_ONLY") and not safe("STATE_ONLY"):
        return "COPS_STATE_CURRENT_FORM_NOT_JUSTIFIED", "inspect a simpler deterministic CoPS STATE formulation before retraining"
    if useful("STATE_ONLY") and safe("STATE_ONLY") and useful("CLASS_ONLY") and safe("CLASS_ONLY") and (
        gain["FULL"]["normal"] < min(gain["STATE_ONLY"]["normal"], gain["CLASS_ONLY"]["normal"])
    ):
        return "STATE_CLASS_INTERACTION_BOTTLENECK", "inspect state/class vector geometry before one minimal interaction-only hypothesis"
    if useful("STATE_ONLY") and gain["STATE_ONLY"]["normal"] < 0.0 and useful("CLASS_ONLY") and gain["CLASS_ONLY"]["normal"] < 0.0:
        return "K1_SELECTIVITY_BOTTLENECK", "Stage 1.6 K1 plus exact-base NO-OP selectivity gate"
    if not any(useful(name) for name in ("FULL", "STATE_ONLY", "CLASS_ONLY")):
        return "K1_SEMANTIC_DIRECTION_NOT_SUPPORTED", "stop the semantic-factor line; do not add K2"
    return "K1_COMPONENT_ATTRIBUTION_MIXED", "no follow-up training is authorized until the mixed component evidence is reviewed"


def _build_model(config, checkpoint, device):
    parameters = inspect.signature(ACDCLIP.__init__).parameters
    kwargs = {
        name: config[name]
        for name, parameter in parameters.items()
        if name not in {"self", "clip_model", "kwargs"}
        and parameter.kind is not inspect.Parameter.VAR_KEYWORD
        and name in config
    }
    kwargs.update({
        "dfg_beta_current": 0.0,
        "dfg_weight_residual_fp32": True,
        "h6_role_topology": config["h6_role_topology"],
        "h6_role_teacher_scale": config["h6_role_teacher_scale"],
        "h6_intrinsic_factor_responsibility": config["h6_intrinsic_factor_responsibility"],
        "h6_prediction_routing": config["h6_prediction_routing"],
        "diagnostics_mode": config["h6_diagnostics_mode"],
        "diagnostics_interval": config["h6_diagnostics_interval"],
        "h6_cluster_responsibility": config["h6_cluster_responsibility"],
        "h6_cluster_temperature": config["h6_cluster_temperature"],
        "h6_router_boundary_mode": config["h6_router_boundary_mode"],
        "h6_router_boundary_trust_scale": config["h6_router_boundary_trust_scale"],
    })
    clip = create_model(
        config["model_name"], img_size=config["img_size"], device=device,
        pretrained="openai", require_pretrained=True, precision="fp32",
    )
    if config["grad_checkpointing"]:
        clip.set_grad_checkpointing(True)
    model = ACDCLIP(clip_model=clip, **kwargs).to(device)
    model.use_soft_prompt = False
    model.use_hybrid_soft_prompt = True
    model.prompt_mode = "h6_dynamic"
    load_adapter_checkpoint(model, checkpoint)
    model.set_dfg_beta(0.0)
    model.eval()
    model.clipmodel.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/phase4/k1/short64_seed0_attempt5"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase4/k1/stage1_5_attribution.json"))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text())
    checkpoint = torch.load(run_dir / "adapter_1.pth", map_location="cpu", weights_only=False)
    if checkpoint.get("git_sha") not in {None, "5b28204f998e7ebb5f5067f59c9eb74b3ad4fd75"}:
        raise RuntimeError("checkpoint provenance does not match Commit A base")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1.5 attribution requires CUDA")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    configure_canonical_fp32()
    device = torch.device(f"cuda:{config['cuda_device']}")
    model = _build_model(config, checkpoint, device)
    dataset = get_text_and_image_dataset("VisA", config["img_size"], "train")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=config["num_workers"],
        pin_memory=config["pin_memory"], worker_init_fn=seed_worker,
        generator=make_dataloader_generator(config["seed"]),
    )
    regions = ("normal", "anomaly", "all")
    store = {
        condition: {name: {region: [] for region in regions} for name in ("base", "dynamic", "final")}
        | {"delta": {region: [] for region in regions}, "delta_signed": [], "delta_signed_normal": [], "delta_signed_anomaly": [], "confidence_normal": [], "confidence_anomaly": [], "delta_normal": [], "delta_anomaly": []}
        for condition in (*CONDITIONS, "BASE")
    }
    semantic = {"dynamic_base_cosine": [], "state_delta_raw_l2": [], "class_delta_raw_l2": [], "state_class_cosine": [], "state_applied_l2": [], "class_applied_l2": []}
    geometry = {}
    with torch.inference_mode():
        for batch_index, raw in enumerate(loader, start=1):
            if batch_index > 64:
                break
            if batch_index not in {1, 32, 64}:
                continue
            image = raw["image"].to(device=device, dtype=torch.float32, non_blocking=True)
            mask = raw["mask"].to(device=device, dtype=torch.float32, non_blocking=True)
            valid_mask = raw["local_mask_valid"].to(device=device, non_blocking=True)
            class_names = list(raw["class_name"])
            visual = model(image, return_phase4_features=True)
            base_text = get_phase2b_global_text_features(
                model, "VisA", class_names, device, use_hybrid_soft_prompt=True, use_soft_prompt=False
            ).float()
            batch_map = {
                name: model.h6.build_batch(model, "VisA", class_names, visual, hybrid_alpha=0.0,
                    base_text_features=base_text, state_scale=state_scale, class_scale=class_scale)
                for name, (state_scale, class_scale) in CONDITIONS.items()
            }
            full = batch_map["FULL"]
            saved = torch.load(run_dir / "k1_fixed_train_probes" / f"batch_{batch_index:03d}.pt", map_location="cpu", weights_only=False)
            geometry[str(batch_index)] = {
                "class_names": class_names,
                "mask_max_abs_error_vs_saved_probe": float((mask.cpu() - saved["mask"]).abs().max()),
                "valid_mask_max_abs_error_vs_saved_probe": float((valid_mask.cpu() - saved["local_mask_valid"]).abs().max()),
            }
            patches = full["base_group_logits"].shape[2]
            side = int(patches ** 0.5)
            target = F.adaptive_avg_pool2d(mask, (side, side)).flatten(1).clamp(0.0, 1.0)
            valid = F.adaptive_avg_pool2d(valid_mask.float(), (side, side)).flatten(1) >= 1.0 - 1e-6
            target = target.unsqueeze(0).expand(model.n_groups, -1, -1)
            valid = valid.unsqueeze(0).expand_as(target)
            region_masks = {"all": valid, "normal": valid & (target < 0.5), "anomaly": valid & (target >= 0.5)}
            semantic["dynamic_base_cosine"].append(F.cosine_similarity(full["dynamic_text"].float(), base_text.permute(1, 0, 2, 3).float(), dim=2).reshape(-1).cpu())
            state = full["state_delta_raw"].float()
            klass = full["class_delta_raw"].float().unsqueeze(1).expand_as(state)
            semantic["state_delta_raw_l2"].append(state.norm(dim=-1).reshape(-1).cpu())
            semantic["class_delta_raw_l2"].append(klass.norm(dim=-1).reshape(-1).cpu())
            semantic["state_class_cosine"].append(F.cosine_similarity(state, klass, dim=-1).reshape(-1).cpu())
            semantic["state_applied_l2"].append((full["gamma_state"] * full["state_delta"].float()).norm(dim=-1).reshape(-1).cpu())
            semantic["class_applied_l2"].append((full["gamma_class"] * full["class_delta"].float()).norm(dim=-1).reshape(-1).cpu())
            for condition in (*CONDITIONS, "BASE"):
                batch = batch_map.get(condition)
                base_logits = full["base_group_logits"]
                if condition == "BASE":
                    dynamic_abnormal = base_logits[..., 1]
                    final_logits = base_logits
                else:
                    dynamic_abnormal = batch["dynamic_abnormal_logits"]
                    final_logits = batch["final_group_logits"]
                base_logit = base_logits[..., 1] - base_logits[..., 0]
                dynamic_logit = dynamic_abnormal - base_logits[..., 0]
                final_logit = final_logits[..., 1] - final_logits[..., 0]
                delta = final_logits[..., 1] - base_logits[..., 1]
                posterior = torch.sigmoid(base_logit)
                for region, region_mask in region_masks.items():
                    for name, logits in (("base", base_logit), ("dynamic", dynamic_logit), ("final", final_logit)):
                        store[condition][name][region].append(F.binary_cross_entropy_with_logits(logits[region_mask], target[region_mask], reduction="none").cpu())
                    store[condition]["delta"][region].append(delta[region_mask].abs().cpu())
                store[condition]["delta_signed"].append(delta[valid].cpu())
                store[condition]["delta_signed_normal"].append(delta[region_masks["normal"]].cpu())
                store[condition]["delta_signed_anomaly"].append(delta[region_masks["anomaly"]].cpu())
                for region in ("normal", "anomaly"):
                    region_mask = region_masks[region]
                    store[condition][f"confidence_{region}"].append(posterior[region_mask].cpu())
                    store[condition][f"delta_{region}"].append(delta[region_mask].abs().cpu())
    metrics = _condition_metrics(store)
    decision, next_hypothesis = _decision(metrics)
    report = {
        "decision": decision,
        "next_authorized_hypothesis": next_hypothesis,
        "inference_only": True,
        "checkpoint": str(run_dir / "adapter_1.pth"),
        "checkpoint_git_sha": checkpoint.get("git_sha"),
        "probe_batches": [1, 32, 64],
        "geometry_replay": geometry,
        "conditions": metrics,
        "semantic_geometry": {name: _mean(values) for name, values in semantic.items()},
        "invariants": {
            "fp32": True,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "amp": False,
            "rho": [float(value) for value in model.h6.rho_values().cpu()],
            "legacy_router_trainable": any(parameter.requires_grad for parameter in model.h6.router.parameters()),
            "legacy_factor_core_trainable": any(parameter.requires_grad for parameter in model.h6.semantic_core.parameters()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    compact = {name: {"normal_gain": value["gain_base_minus_final"]["normal"], "anomaly_gain": value["gain_base_minus_final"]["anomaly"], "selectivity": value["residual"]["selectivity_ratio"]} for name, value in metrics.items()}
    print(json.dumps({"decision": decision, "next_authorized_hypothesis": next_hypothesis, "conditions": compact}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
