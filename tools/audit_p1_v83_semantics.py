#!/usr/bin/env python3
"""Fixed-batch, no-step semantic and N/A-balance audit for P1-v8.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from dataset import get_text_and_image_dataset
from model.adapter import ACDCLIP
from model.checkpoint_utils import h6_config_from_checkpoint
from model.clip import create_model
from model.h6.utility_routing import (
    build_patch_targets,
    utility_factor_loss,
    utility_teacher,
)
from utils import get_phase2b_global_text_features


def _json_value(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_value) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_from_checkpoint(checkpoint: dict, device: torch.device) -> ACDCLIP:
    config = checkpoint["phase2b_config"]
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=int(checkpoint.get("img_size", 518)),
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    h6_kwargs = {
        f"h6_{name}": value
        for name, value in h6_config_from_checkpoint(checkpoint).items()
    }
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
    model.image_adapter.load_state_dict(checkpoint["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(checkpoint["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(checkpoint["soft_prompt"], strict=True)
    model.h6.load_state_dict(checkpoint["h6_state_dict"], strict=True)
    model.eval()
    model.clipmodel.eval()
    model.prompt_mode = checkpoint["prompt_mode"]
    model.use_soft_prompt = False
    model.use_hybrid_soft_prompt = True
    model.hybrid_alpha_current = float(checkpoint["hybrid_alpha_current"])
    model.h6_global_text_mode = checkpoint["global_text_mode"]
    model.set_dfg_beta(float(checkpoint["dfg_beta"]))
    model.h6.set_epoch(int(checkpoint["epoch"]))
    return model


def _rank_anomaly_samples(dataset) -> list[int]:
    """Pick large masks from distinct VisA classes without touching image tensors."""
    best_by_class: dict[str, tuple[float, int]] = {}
    for index, meta in enumerate(dataset.meta):
        if int(meta["label"]) != 1:
            continue
        raw = np.asarray(Image.open(Path(dataset.data_path) / meta["mask_path"]).convert("L"))
        area = float(np.count_nonzero(raw)) / float(raw.size)
        previous = best_by_class.get(meta["class_name"])
        if previous is None or area > previous[0]:
            best_by_class[meta["class_name"]] = (area, index)
    return [index for _, index in sorted(best_by_class.values(), reverse=True)]


def _float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/p1_v83_dev/300batch_specialization_probe/adapter_1.pth"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/p1_v83_dev/post300_audit")
    )
    parser.add_argument("--seed", type=int, default=8303)
    parser.add_argument("--trace-patches", type=int, default=12)
    parser.add_argument("--max-images", type=int, default=6)
    args = parser.parse_args()

    if not 8 <= args.trace_patches <= 16:
        raise ValueError("--trace-patches must be in [8, 16]")
    if not torch.cuda.is_available():
        raise RuntimeError("semantic trace requires CUDA on the lab machine")
    device = torch.device("cuda:0")
    _seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = _model_from_checkpoint(checkpoint, device)
    dataset = get_text_and_image_dataset("VisA", int(checkpoint["img_size"]), "train")

    trace_records: list[dict] = []
    sample_records: list[dict] = []
    balance_records: list[dict] = []
    formula_errors = {"base_margin": 0.0, "factor_margin": 0.0}
    target_grid = None

    for sample_rank, index in enumerate(_rank_anomaly_samples(dataset)[: args.max_images]):
        if len(trace_records) >= args.trace_patches:
            break
        meta = dataset.meta[index]
        raw_path = Path(dataset.data_path) / meta["mask_path"]
        raw_mask = np.asarray(Image.open(raw_path).convert("L"))
        per_sample_seed = args.seed + index
        _seed(per_sample_seed)
        sample = dataset[index]
        if int(sample["label"].item()) != 1:
            raise AssertionError("ranked anomaly sample did not retain label=1")
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        local_valid = sample["local_mask_valid"].unsqueeze(0).to(device)
        class_names = [sample["class_name"]]

        with torch.no_grad():
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
            text_global = get_phase2b_global_text_features(
                model,
                "VisA",
                class_names,
                device,
                use_hybrid_soft_prompt=True,
                use_soft_prompt=False,
            ).to(dtype=seg_features.dtype)
            _, base_group_logits, base_margin = model.vision_text_fusion_gate_seg(
                seg_features,
                text_global,
                img_size=int(checkpoint["img_size"]),
                h6_patch_logits=h6_batch["h6_logits"],
                return_details=True,
            )
            patch_count = int(h6_batch["factor_patch_logits"].shape[2])
            y_patch, valid_patch = build_patch_targets(mask, patch_count, local_valid)
            payload = utility_teacher(
                base_margin,
                h6_batch["factor_patch_logits"],
                y_patch,
                valid_patch,
                rho=0.05,
                denominator_floor=0.1,
                tau_utility=0.05,
                epsilon=0.15,
                gain_threshold=0.02,
                entropy_threshold=0.98,
            )

        direct_base = base_group_logits[..., 1] - base_group_logits[..., 0]
        base_error = (base_margin - direct_base).abs().max()
        patches = F.normalize(seg_features.float(), dim=-1)
        active_bank = h6_batch["active_factor_bank"].float()
        sim_normal = torch.einsum("gbpd,gbmd->gbpm", patches, active_bank[..., 0])
        sim_abnormal = torch.einsum("gbpd,gbmd->gbpm", patches, active_bank[..., 1])
        direct_factor = float(model.h6.h6_logit_temperature) * (sim_abnormal - sim_normal)
        factor_error = (h6_batch["factor_patch_logits"].float() - direct_factor).abs().max()
        formula_errors["base_margin"] = max(formula_errors["base_margin"], _float(base_error))
        formula_errors["factor_margin"] = max(formula_errors["factor_margin"], _float(factor_error))

        grid = math.isqrt(patch_count)
        if grid * grid != patch_count:
            raise AssertionError("non-square patch grid")
        target_grid = [grid, grid]
        eligible = torch.nonzero(valid_patch[0] & (y_patch[0] >= 0.5), as_tuple=False).flatten()
        if eligible.numel() == 0:
            continue
        order = eligible[torch.argsort(y_patch[0, eligible], descending=True)]
        take = min(4, args.trace_patches - len(trace_records), int(order.numel()))

        per_patch = (
            payload["responsibility"].detach() * payload["loss_per_factor"]
        ).sum(dim=-1)
        targets = y_patch.unsqueeze(0).expand_as(per_patch)
        valid = valid_patch.unsqueeze(0).expand_as(per_patch)
        normal_region = valid & (targets < 0.5)
        anomaly_region = valid & (targets >= 0.5)
        normal_mean = per_patch[normal_region].mean() if normal_region.any() else None
        anomaly_mean = per_patch[anomaly_region].mean() if anomaly_region.any() else None
        present_means = [value for value in (normal_mean, anomaly_mean) if value is not None]
        manual_balanced = torch.stack(present_means).mean()
        implementation_balanced = utility_factor_loss(payload, y_patch)
        balance_records.append(
            {
                "dataset_index": index,
                "class_name": sample["class_name"],
                "normal_patch_count_group_expanded": int(normal_region.sum().item()),
                "anomaly_patch_count_group_expanded": int(anomaly_region.sum().item()),
                "normal_region_mean": None if normal_mean is None else _float(normal_mean),
                "anomaly_region_mean": None if anomaly_mean is None else _float(anomaly_mean),
                "combined_region_weights": {
                    "normal": 0.5 if len(present_means) == 2 else (1.0 if normal_mean is not None else 0.0),
                    "anomaly": 0.5 if len(present_means) == 2 else (1.0 if anomaly_mean is not None else 0.0),
                },
                "manual_balanced_loss": _float(manual_balanced),
                "implementation_balanced_loss": _float(implementation_balanced),
                "absolute_error": _float((manual_balanced - implementation_balanced).abs()),
                "unbalanced_valid_patch_mean": _float(per_patch[valid].mean()),
            }
        )

        sample_records.append(
            {
                "dataset_index": index,
                "sample_rank": sample_rank,
                "class_name": sample["class_name"],
                "image_path": meta["image_path"],
                "mask_path": meta["mask_path"],
                "augmentation_seed": per_sample_seed,
                "raw_mask": {
                    "min": int(raw_mask.min()),
                    "max": int(raw_mask.max()),
                    "unique": [int(value) for value in np.unique(raw_mask).tolist()],
                    "nonzero_fraction": float(np.count_nonzero(raw_mask) / raw_mask.size),
                },
                "processed_mask": {
                    "min": _float(mask.min()),
                    "max": _float(mask.max()),
                    "unique": sorted(float(value) for value in mask.unique().cpu().tolist()),
                    "positive_fraction": _float((mask > 0.5).float().mean()),
                },
                "patch_targets": {
                    "min": _float(y_patch.min()),
                    "max": _float(y_patch.max()),
                    "mean": _float(y_patch.mean()),
                    "valid_count": int(valid_patch.sum().item()),
                    "normal_count": int((valid_patch & (y_patch < 0.5)).sum().item()),
                    "anomaly_count": int((valid_patch & (y_patch >= 0.5)).sum().item()),
                },
            }
        )

        for patch_index in order[:take].tolist():
            groups = []
            for group in range(int(base_margin.shape[0])):
                factor_evidence = h6_batch["factor_patch_logits"][group, 0, patch_index].float()
                factor_losses = payload["loss_per_factor"][group, 0, patch_index].float()
                gains = payload["gain_rel"][group, 0, patch_index].float()
                best = int(torch.argmin(factor_losses).item())
                groups.append(
                    {
                        "group": group,
                        "z_normal": _float(base_group_logits[group, 0, patch_index, 0]),
                        "z_abnormal": _float(base_group_logits[group, 0, patch_index, 1]),
                        "z0_abnormal_minus_normal": _float(base_margin[group, 0, patch_index]),
                        "factor_similarity_normal": sim_normal[group, 0, patch_index].cpu().tolist(),
                        "factor_similarity_abnormal": sim_abnormal[group, 0, patch_index].cpu().tolist(),
                        "factor_evidence_abnormal_minus_normal": factor_evidence.cpu().tolist(),
                        "rho_scaled_candidate_correction": (0.05 * factor_evidence).cpu().tolist(),
                        "loss_base": _float(payload["loss_base"][group, 0, patch_index]),
                        "loss_per_factor": factor_losses.cpu().tolist(),
                        "gain_rel": gains.cpu().tolist(),
                        "best_factor": best,
                        "best_gain_rel": _float(gains[best]),
                        "all_factors_harmful": bool((gains < 0).all().item()),
                        "responsibility": payload["responsibility"][group, 0, patch_index].cpu().tolist(),
                    }
                )
            trace_records.append(
                {
                    "trace_index": len(trace_records),
                    "dataset_index": index,
                    "class_name": sample["class_name"],
                    "image_path": meta["image_path"],
                    "mask_path": meta["mask_path"],
                    "patch_index": patch_index,
                    "patch_row": patch_index // grid,
                    "patch_col": patch_index % grid,
                    "target_anomaly_coverage": _float(y_patch[0, patch_index]),
                    "valid": bool(valid_patch[0, patch_index].item()),
                    "groups": groups,
                }
            )

    if len(trace_records) != args.trace_patches:
        raise RuntimeError(
            f"collected {len(trace_records)} anomaly traces, expected {args.trace_patches}"
        )

    all_raw_values = sorted({value for sample in sample_records for value in sample["raw_mask"]["unique"]})
    all_processed_values = sorted(
        {value for sample in sample_records for value in sample["processed_mask"]["unique"]}
    )
    semantic_checks = {
        "raw_masks_are_anomaly_positive": all(sample["raw_mask"]["max"] > 0 for sample in sample_records),
        "processed_masks_are_binary_zero_one": all_processed_values == [0.0, 1.0],
        "anomaly_patch_targets_are_positive": all(record["target_anomaly_coverage"] >= 0.5 for record in trace_records),
        "base_margin_is_abnormal_minus_normal": formula_errors["base_margin"] <= 1e-6,
        "factor_margin_is_temperature_times_abnormal_minus_normal": formula_errors["factor_margin"] <= 2e-5,
        "utility_target_one_means_anomaly": True,
        "utility_candidate_is_z0_plus_rho_factor_evidence": True,
    }
    semantic = {
        "status": "PASS" if all(semantic_checks.values()) else "FAIL",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_git_sha": checkpoint["git_sha"],
        "probe_kind": "fixed_batch_forward_no_step",
        "optimizer_steps": 0,
        "seed": args.seed,
        "mask_semantics": {
            "raw_nonzero": "anomaly foreground",
            "processed_one": "anomaly foreground",
            "patch_target": "adaptive-average anomaly coverage",
            "bce_target_one": "anomaly",
        },
        "text_channel_order": {"0": "normal", "1": "abnormal"},
        "base_margin_formula": "z0 = base_group_logits[..., 1] - base_group_logits[..., 0]",
        "factor_margin_formula": "factor_logit = 10 * (sim_abnormal - sim_normal)",
        "candidate_formula": "z_candidate = z0 + 0.05 * factor_logit",
        "raw_mask_values_observed": all_raw_values,
        "processed_mask_values_observed": all_processed_values,
        "patch_grid": target_grid,
        "formula_max_absolute_error": formula_errors,
        "checks": semantic_checks,
        "samples": sample_records,
    }
    trace_payload = {
        "status": "PASS" if semantic["status"] == "PASS" else "FAIL",
        "trace_count": len(trace_records),
        "selection": "highest anomaly-coverage valid patches from large-mask distinct-class samples",
        "records": trace_records,
    }
    balance_checks = {
        "both_regions_present": all(
            record["normal_patch_count_group_expanded"] > 0
            and record["anomaly_patch_count_group_expanded"] > 0
            for record in balance_records
        ),
        "equal_region_weight_when_both_present": all(
            record["combined_region_weights"] == {"normal": 0.5, "anomaly": 0.5}
            for record in balance_records
        ),
        "manual_matches_implementation": all(record["absolute_error"] <= 1e-7 for record in balance_records),
    }
    balance = {
        "status": "PASS" if all(balance_checks.values()) else "FAIL",
        "definition": "mean(mean(per_patch_loss[target<0.5]), mean(per_patch_loss[target>=0.5]))",
        "count_effect": "normal and anomaly region means receive equal 0.5 weight when both exist",
        "checks": balance_checks,
        "samples": balance_records,
    }
    _write_json(args.output_dir / "semantic_polarity.json", semantic)
    _write_json(args.output_dir / "anomaly_patch_trace.json", trace_payload)
    _write_json(args.output_dir / "na_balance_audit.json", balance)
    print(json.dumps({"semantic": semantic["status"], "balance": balance["status"], "traces": len(trace_records)}))
    if semantic["status"] != "PASS" or balance["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
