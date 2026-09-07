#!/usr/bin/env python3
"""Run the preregistered THBR Part-A red-team audit.

This script performs source-only endpoint tail/boundary analysis and a fixed
16-batch gradient redundancy audit.  It does not update model parameters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
os.sys.path.insert(0, str(REPO))

from dataset import CLASS_NAMES, get_text_and_image_dataset
from h2_clean.contract import EpochWorkerInit, make_dataloader_generator, sha256_file
from h2_clean.precision import PrecisionPolicy
from h2_clean.stage_fusion import H2_EQUAL_STAGE_FUSION_WEIGHTS, fuse_stage_logits
from model.adapter import ACDCLIP
from model.clip import create_model
from scripts.evaluate_h2_fixed_fusion_r2_source import (
    ENDPOINT_SUBSET,
    SPLIT_IDENTITY,
    binary_metrics,
    load_endpoint_selection,
    stage_logits_from_features,
)
from train import compute_hybrid_k_regularization, get_hybrid_alpha_for_epoch, get_dfg_beta_for_epoch
from utils import BinaryDiceLoss, FocalLoss, get_hybrid_soft_prompt_single_class_text_embedding


IMG = 518
SEED = 0
START_EPOCH = 10
START_GLOBAL_STEP = 3607
MAX_ATTEMPTS = 500
GRADIENT_AUDIT_BATCHES = 16
ANCHOR_LAMBDA = 0.0021633926715180626
ANCHOR_RHO = 0.10
PROTOCOL_ID = "H2_THBR_R1"
E10 = Path("/workspace/h2_safe_anchor_e20_medical_selected/adapter_10.pth")
E10_SHA256 = "64b72dc3d1155285c826781bee4c5970bd45218e95b21675fd19d9a6b2ab54a7"
FREEZE = Path("/workspace/h2_late_convlora_freeze_r1/A_LATE_FREEZE_R1_STAGE23_CONVLORA/final.pth")
FREEZE_SHA256 = "4a5f3fd0e54b53f8cc78060a4092e3d1240792f2809d7ebfe57e9340789fc9f8"
GRADBUDGET = Path("/workspace/h2_late_convlora_gradbudget_bounded_r1/A_GRADBUDGET_SHORT_R1_CANDIDATE/final.pth")
GRADBUDGET_SHA256 = "8b54f7e2e77e02f055cf73ae5c403681ff61cf3aeb82863cfe212d1ae499c41f"
ARM_E10 = "A_SAFE_ANCHOR_E10"
ARM_FREEZE = "A_LATE_CONVLORA_FREEZE_R1"
ARM_GRADBUDGET = "A_GRADBUDGET_R1_DESCRIPTIVE_INVALID"

MANIFEST_JSON = REPO / "audit/H2_THBR_R1_ATTEMPT_MANIFEST.json"
SCORE_MD = REPO / "audit/H2_THBR_R1_SCORE_SEMANTICS.md"
PREMISE_CSV = REPO / "audit/H2_THBR_R1_HARD_BACKGROUND_PREMISE.csv"
PREMISE_JSON = REPO / "audit/H2_THBR_R1_HARD_BACKGROUND_PREMISE.json"
BOUNDARY_CSV = REPO / "audit/H2_THBR_R1_BOUNDARY_AUDIT.csv"
BOUNDARY_JSON = REPO / "audit/H2_THBR_R1_BOUNDARY_AUDIT.json"
REDUNDANCY_CSV = REPO / "audit/H2_THBR_R1_GRADIENT_REDUNDANCY.csv"
REDUNDANCY_JSON = REPO / "audit/H2_THBR_R1_GRADIENT_REDUNDANCY.json"
CALIBRATION_JSON = REPO / "audit/H2_THBR_R1_LAMBDA_CALIBRATION.json"
AUDIT_DECISION_JSON = REPO / "audit/H2_THBR_R1_AUDIT_DECISION.json"
AUDIT_DECISION_MD = REPO / "audit/H2_THBR_R1_AUDIT_DECISION.md"


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rng_from_payload(payload: dict) -> dict:
    return {key: payload[key] for key in (
        "python_random_state", "numpy_random_state", "torch_cpu_rng_state", "torch_cuda_rng_state_all",
    )}


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    if torch.cuda.is_available() and state["torch_cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


def validate_start() -> tuple[dict, dict]:
    if sha256_file(E10) != E10_SHA256:
        raise RuntimeError("E10 SHA256 mismatch")
    payload = torch.load(E10, map_location="cpu", weights_only=False)
    required = {"model_state", "optimizer_state", "scheduler_state", "scaler_state", "resolved_scientific_config", "resolved_operational_config", "dataloader_generator_state", "python_random_state", "numpy_random_state", "torch_cpu_rng_state", "torch_cuda_rng_state_all"}
    if not required.issubset(payload):
        raise RuntimeError(f"E10 is not a full-state checkpoint: {sorted(required - set(payload))}")
    if (payload["epoch"], payload["global_step"]) != (START_EPOCH, START_GLOBAL_STEP):
        raise RuntimeError("E10 epoch/global step mismatch")
    expected = {
        "precision": "fp16", "amp_enabled": True, "gradscaler_enabled": True,
        "tf32_enabled": False, "seed": 0, "anchor_gradient_budget": True,
        "anchor_family_budget": ANCHOR_RHO, "lambda_kg": .01, "lambda_k": .002,
        "use_hybrid_soft_prompt": True, "use_soft_prompt": False,
        "use_ss2d_dfg": True, "dfg_mode": "attn",
        "precision_protocol": "HISTORICAL_MIXED_FP16_FP32_V1",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"E10 identity mismatch {key}: {payload.get(key)!r} != {value!r}")
    science = payload["resolved_scientific_config"]
    operational = payload["resolved_operational_config"]
    if science.get("anchor_lambda") != ANCHOR_LAMBDA or science.get("batch_size") != 6 or science.get("img_size") != IMG:
        raise RuntimeError("E10 Safe-Anchor/base configuration mismatch")
    if operational.get("num_workers") != 6 or not operational.get("pin_memory"):
        raise RuntimeError("E10 DataLoader identity mismatch")
    summary = json.loads((REPO / "audit/H2_SAFE_ANCHOR_E20_TRAINING_SUMMARY.json").read_text())
    row = next(row for row in summary["checkpoints"] if int(row["epoch"]) == START_EPOCH)
    if not row["numerical_validity"] or int(row["nonfinite_loss_skips"]) or int(row["nonfinite_grad_skips"]):
        raise RuntimeError("tracked E10 numerical validity failed")
    return payload, {
        "checkpoint": str(E10), "checkpoint_sha256": E10_SHA256,
        "epoch": START_EPOCH, "global_step": START_GLOBAL_STEP,
        "numerical_validity": True, "scientific_config": science,
        "operational_config": operational, "tracked_summary": row,
    }


def make_model(payload: dict, device: torch.device, *, training_graph: bool = False) -> ACDCLIP:
    clip = create_model("ViT-L-14-336", img_size=IMG, device=device, pretrained="openai", require_pretrained=True)
    if training_graph:
        clip.set_grad_checkpointing(True)
    model = ACDCLIP(
        clip_model=clip, n_groups=3, image_adapt_weight=.2, text_adapt_weight=.2,
        conv_lora_rank=8, conv_lora_alpha=2., conv_kernel_size_list=[3, 5],
        lora_rank=16, lora_alpha=2., dfg_mode="attn", dfg_attn_dim=256,
        dfg_attn_tau=8., use_ss2d_dfg=True, dfg_gamma_max=.2,
        dfg_ss2d_fusion="weight_residual", dfg_beta=.1,
        dfg_beta_schedule="warmup010", dfg_beta_target=.1,
        dfg_beta_current=float(payload["dfg_beta_current"]), dfg_weight_residual_fp32=True,
        stage_fusion_weights=H2_EQUAL_STAGE_FUSION_WEIGHTS,
        use_soft_prompt=False, soft_prompt_ctx_len=4,
        soft_prompt_init="phrase", soft_prompt_init_phrase="a photo of a",
    ).to(device).eval()
    for module in model.modules():
        if hasattr(module, "enable_fp16_numerical_islands"):
            module.enable_fp16_numerical_islands = False
    state = payload["model_state"]
    model.image_adapter.load_state_dict(state["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(state["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(state["soft_prompt"], strict=True)
    model.prompt_mode = "hybrid"
    model.use_hybrid_soft_prompt = True
    model.use_soft_prompt = False
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", .2))
    model.hybrid_alpha_max = .2
    model.soft_prompt_freeze_epochs = 3
    model.requires_grad_(False)
    if training_graph:
        model.image_adapter.requires_grad_(True)
        model.text_adapter.requires_grad_(True)
        model.soft_prompt.requires_grad_(True)
    return model


def make_loader(dataset, epoch: int):
    generator = make_dataloader_generator(SEED)
    generator.manual_seed(SEED + 104729 * int(epoch))
    worker = EpochWorkerInit(SEED)
    worker.set_epoch(epoch)
    return DataLoader(
        dataset, batch_size=6, shuffle=True, num_workers=6, pin_memory=True,
        generator=generator, worker_init_fn=worker, persistent_workers=False,
        prefetch_factor=2,
    )


def batch_identity(attempt: int, epoch: int, batch_idx: int, batch, image, mask, label) -> dict:
    return {
        "attempt_index": int(attempt), "epoch": int(epoch), "batch": int(batch_idx),
        "file_names": json.dumps(list(batch["file_name"]), separators=(",", ":")),
        "image_sha256": tensor_hash(image), "mask_sha256": tensor_hash(mask),
        "labels": json.dumps(label.detach().cpu().tolist(), separators=(",", ":")),
    }


IDENTITY_KEYS = ("attempt_index", "epoch", "batch", "file_names", "image_sha256", "mask_sha256", "labels")


def collect_manifest(payload: dict) -> list[dict]:
    restore_rng_state(rng_from_payload(payload))
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    rows = []
    for epoch in range(START_EPOCH + 1, START_EPOCH + 10):
        for batch_idx, batch in enumerate(make_loader(dataset, epoch)):
            image = batch["image"].to("cuda:0")
            mask = batch["mask"].to("cuda:0")
            label = batch["label"].to("cuda:0")
            rows.append(batch_identity(len(rows), epoch, batch_idx, batch, image, mask, label))
            if len(rows) == MAX_ATTEMPTS:
                return rows
    raise RuntimeError("failed to materialize 500 source attempts")


def write_score_semantics() -> None:
    SCORE_MD.write_text(
        "# H2 THBR R1 score semantics\n\n"
        "The canonical training segmentation tensor is `seg_pred` returned by `ACDCLIP.vision_text_fusion_gate_seg` in `train.py` with `test_mode=False`, `cir_training=False`, and equal stage fusion. It has shape `[B, 2, 518, 518]` and is a post-fusion softmax probability map produced from bilinearly resized native 37x37 stage logits. The THBR anomaly score is `seg_pred[:, 1, :, :]`, the abnormal-class probability.\n\n"
        "The existing focal loss consumes the full two-channel `seg_pred` probability tensor directly; it does not apply another softmax. Normal Dice consumes `seg_pred[:, 0, :, :]` against `1-mask`, and abnormal Dice consumes `seg_pred[:, 1, :, :]` against `mask`. The existing task segmentation loss is focal + normal Dice + abnormal Dice.\n\n"
        "For the frozen endpoint audit, the repository's existing evaluator semantics are preserved: each native stage score is formed from the same vision/text features, Gaussian-blurred with the existing 7x7 industrial kernel, bilinearly resized to 518x518, equally fused in logit space, and softmaxed; endpoint THBR diagnostics therefore use the same post-fusion abnormal probability score space as the bounded diagnostics. No separate score head is introduced.\n"
    )


def scalar_stats(values: np.ndarray, quantiles=(.90, .95, .99, .995)) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not x.size:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "max": float("nan"), **{f"p{str(q * 100).replace('.', '')}": float("nan") for q in quantiles}}
    result = {"count": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)), "max": float(x.max())}
    for q in quantiles:
        result[f"p{str(q * 100).replace('.', '')}"] = float(np.quantile(x, q))
    return result


def region_stats(score: np.ndarray, mask: np.ndarray) -> dict:
    boundary, interior, near, far = morphology(mask.astype(bool))
    return {
        "positive": scalar_stats(score[mask.astype(bool)], quantiles=(.95, .99)),
        "interior": scalar_stats(score[interior], quantiles=(.95, .99)),
        "boundary": scalar_stats(score[boundary], quantiles=(.95, .99)),
        "background": scalar_stats(score[~mask.astype(bool)]),
        "near_background": scalar_stats(score[near]),
        "far_background": scalar_stats(score[far]),
    }


def morphology(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Exact existing bounded-diagnostic semantics: no new radius or tuning.
    from scipy import ndimage
    structure = np.ones((7, 7), dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    dilated = ndimage.binary_dilation(mask, structure=structure)
    return mask & ~eroded, eroded, dilated & ~mask, ~dilated


def top_mass(values: np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(x.sum())
    result = {}
    for fraction in (.001, .005, .01, .05):
        count = max(1, int(math.ceil(fraction * x.size)))
        result[str(fraction)] = float(np.sort(x)[-count:].sum() / total) if total > 0 else float("nan")
    return result


def top_p_composition(score: np.ndarray, mask: np.ndarray) -> dict:
    boundary, interior, near, far = morphology(mask.astype(bool))
    p = int(mask.astype(bool).sum())
    if p <= 0:
        return {"P": 0, "anomaly_fraction": float("nan"), "anomaly_boundary_fraction": float("nan"), "near_background_fraction": float("nan"), "far_background_fraction": float("nan")}
    flat_order = np.argsort(-score.reshape(-1), kind="stable")[:p]
    def frac(region):
        return float(region.reshape(-1)[flat_order].mean())
    return {"P": p, "anomaly_fraction": frac(mask.astype(bool)), "anomaly_boundary_fraction": frac(boundary), "near_background_fraction": frac(near), "far_background_fraction": frac(far)}


def endpoint_model(payload: dict, device: torch.device, beta: float, alpha: float) -> ACDCLIP:
    model = make_model(payload, device)
    model.dfg_beta = beta
    model.hybrid_alpha_current = alpha
    model.stage_fusion_weights = H2_EQUAL_STAGE_FUSION_WEIGHTS
    return model


def audit_endpoints(start_payload: dict, device: torch.device) -> tuple[dict, list[dict], dict]:
    selected, by_category = load_endpoint_selection()
    paths = [(ARM_E10, E10, .1, .2), (ARM_FREEZE, FREEZE, .1, .2), (ARM_GRADBUDGET, GRADBUDGET, 0.0, 0.0)]
    endpoint_results = {}
    premise_rows = []
    boundary_rows = []
    for arm, path, beta, alpha in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        model = endpoint_model(start_payload, device, beta, alpha)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model_state", payload)
        model.image_adapter.load_state_dict(state["image_adapter"], strict=True)
        model.text_adapter.load_state_dict(state["text_adapter"], strict=True)
        model.soft_prompt.load_state_dict(state["soft_prompt"], strict=True)
        datasets = get_text_and_image_dataset("VisA", IMG, "test")
        score_by_name = {}
        mask_by_name = {}
        for category in CLASS_NAMES["VisA"]:
            dataset = datasets[category]
            index_by_name = {meta["image_path"]: i for i, meta in enumerate(dataset.meta)}
            indices = [index_by_name[row["file_name"]] for row in by_category[category]]
            loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False, num_workers=0)
            text, _, _ = get_hybrid_soft_prompt_single_class_text_embedding(model, "VisA", category, device, return_kg=False)
            with torch.no_grad():
                for batch in loader:
                    image = batch["image"].to(device)
                    with PrecisionPolicy("fp16").autocast(device):
                        seg_tokens, _ = model(image)
                        vision = torch.stack(seg_tokens)
                        logits = stage_logits_from_features(model, vision, text)
                        fused = fuse_stage_logits(logits, H2_EQUAL_STAGE_FUSION_WEIGHTS)
                        scores = F.softmax(fused, dim=1)[:, 1].float().cpu().numpy()
                    masks = batch["mask"][:, 0].numpy().astype(np.uint8)
                    for name, score, mask in zip(batch["file_name"], scores, masks):
                        score_by_name[name] = score
                        mask_by_name[name] = mask
        expected = [row["file_name"] for row in selected]
        if list(score_by_name) != expected:
            raise RuntimeError(f"endpoint ordering mismatch for {arm}")
        all_scores = np.concatenate([score_by_name[name].reshape(-1) for name in expected])
        all_masks = np.concatenate([mask_by_name[name].reshape(-1) for name in expected])
        arm_rows = []
        arm_boundary = []
        aggregate_mass = {str(f): {"numerator": 0.0, "denominator": 0.0} for f in (.001, .005, .01, .05)}
        top_compositions = []
        for row in selected:
            if row["label_type"] != "anomaly":
                continue
            name = row["file_name"]
            score = score_by_name[name]
            mask = mask_by_name[name]
            regions = region_stats(score, mask)
            pcomp = top_p_composition(score, mask)
            bg = score[~mask.astype(bool)]
            boundary, interior, near, far = morphology(mask.astype(bool))
            concentration = top_mass(bg)
            for frac, value in concentration.items():
                count = max(1, int(math.ceil(float(frac) * bg.size)))
                aggregate_mass[frac]["numerator"] += float(np.sort(bg)[-count:].sum())
                aggregate_mass[frac]["denominator"] += float(bg.sum())
            record = {"arm": arm, "category": row["category"], "file_name": name, "P": int(mask.astype(bool).sum())}
            for region, stats in regions.items():
                for key, value in stats.items():
                    record[f"{region}_{key}"] = value
            for frac, value in concentration.items():
                record[f"background_top_{frac}_mass_fraction"] = value
            record.update({f"top_P_{key}": value for key, value in pcomp.items()})
            arm_rows.append(record)
            top_background = np.argsort(-bg, kind="stable")[:max(1, int(math.ceil(.01 * bg.size)))]
            near_bg_values = bg[near[~mask.astype(bool)]] if near.any() else np.empty(0)
            # The top-background indices are indexed in the flattened background array.
            bg_near = near.reshape(-1)[mask.reshape(-1) == 0]
            bg_far = far.reshape(-1)[mask.reshape(-1) == 0]
            arm_boundary.append({
                "arm": arm, "category": row["category"], "file_name": name,
                "background_count": int(bg.size),
                "top_background_1pct_count": int(top_background.size),
                "top_background_1pct_near_fraction": float(bg_near[top_background].mean()) if top_background.size else float("nan"),
                "top_background_1pct_far_fraction": float(bg_far[top_background].mean()) if top_background.size else float("nan"),
                "top_background_1pct_near_score_mass_fraction": float(bg[top_background][bg_near[top_background]].sum() / max(float(bg[top_background].sum()), 1e-30)) if top_background.size and bg_near[top_background].any() else 0.0,
                "near_background_count": int(near.sum()), "far_background_count": int(far.sum()),
                "near_background_fraction_of_background": float(near.sum() / max(1, (~mask.astype(bool)).sum())),
            })
            top_compositions.append(pcomp)
        endpoint_results[arm] = {
            "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
            "sample_count": len(expected), "anomalous_image_count": len(arm_rows),
            "anomaly_image_records": arm_rows,
            "aggregate_background_mass_concentration": {frac: value["numerator"] / max(value["denominator"], 1e-30) for frac, value in aggregate_mass.items()},
            "mean_per_image_background_mass_concentration": {frac: float(np.mean([r[f"background_top_{frac}_mass_fraction"] for r in arm_rows])) for frac in aggregate_mass},
            "top_P_composition_mean": {key: float(np.nanmean([r[f"top_P_{key}"] for r in arm_rows])) for key in ("anomaly_fraction", "anomaly_boundary_fraction", "near_background_fraction", "far_background_fraction")},
            "top_P_composition_median": {key: float(np.nanmedian([r[f"top_P_{key}"] for r in arm_rows])) for key in ("anomaly_fraction", "anomaly_boundary_fraction", "near_background_fraction", "far_background_fraction")},
            "pixel_metrics": binary_metrics(all_scores, all_masks),
            "endpoint_finite": bool(np.isfinite(all_scores).all()),
        }
        premise_rows.extend(arm_rows)
        boundary_rows.extend(arm_boundary)
        del model
        torch.cuda.empty_cache()
    json_dump(PREMISE_JSON, {
        "protocol_id": PROTOCOL_ID, "scope": "frozen 96-image VisA endpoint; anomalous-image regional audit; no target/Medical/MVTec",
        "endpoint_subset_sha256": sha256_file(ENDPOINT_SUBSET), "endpoint_sample_count": 96,
        "anomalous_image_count_per_arm": 48, "arms": endpoint_results,
        "morphology": "existing H2 7x7 erosion/dilation definitions; no new radius",
        "background_mass_definition": "sum of top ceil(fraction * background_count) anomaly scores divided by total background score mass",
        "top_P_definition": "P equals ground-truth anomaly-pixel count per anomalous image; top P predicted anomaly-score pixels",
    })
    with PREMISE_CSV.open("w", newline="") as handle:
        fields = sorted({key for row in premise_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(premise_rows)
    with BOUNDARY_CSV.open("w", newline="") as handle:
        fields = sorted({key for row in boundary_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(boundary_rows)
    json_dump(BOUNDARY_JSON, {
        "protocol_id": PROTOCOL_ID, "morphology": "existing H2 7x7 erosion/dilation definitions",
        "scope": "48 anomalous images per arm on frozen VisA endpoint cohort",
        "records": boundary_rows,
        "aggregate": {
            arm: {
                "mean_top_background_1pct_near_fraction": float(np.nanmean([r["top_background_1pct_near_fraction"] for r in boundary_rows if r["arm"] == arm])),
                "median_top_background_1pct_near_fraction": float(np.nanmedian([r["top_background_1pct_near_fraction"] for r in boundary_rows if r["arm"] == arm])),
                "mean_top_background_1pct_far_fraction": float(np.nanmean([r["top_background_1pct_far_fraction"] for r in boundary_rows if r["arm"] == arm])),
                "mean_near_background_fraction_of_background": float(np.nanmean([r["near_background_fraction_of_background"] for r in boundary_rows if r["arm"] == arm])),
            } for arm in (ARM_E10, ARM_FREEZE, ARM_GRADBUDGET)
        },
    })
    return endpoint_results, premise_rows, boundary_rows


def load_training_batch(payload: dict, manifest: list[dict], attempt: int):
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    target = manifest[attempt]
    loader = make_loader(dataset, target["epoch"])
    for batch_idx, batch in enumerate(loader):
        if batch_idx != target["batch"]:
            continue
        image = batch["image"].to("cuda:0")
        mask = batch["mask"].to("cuda:0")
        label = batch["label"].to("cuda:0")
        current = batch_identity(attempt, target["epoch"], batch_idx, batch, image, mask, label)
        if current != target:
            raise RuntimeError("fixed batch manifest mismatch")
        return batch, image, mask, label
    raise RuntimeError("manifest batch unavailable")


def build_training_losses(model: ACDCLIP, image, mask, label, class_names, device, policy: PrecisionPolicy, focal: FocalLoss, dice: BinaryDiceLoss, include_thbr: bool = True):
    by_class, kg_losses, k_losses = {}, [], []
    for class_name in sorted(set(class_names)):
        text, kg, _, components = get_hybrid_soft_prompt_single_class_text_embedding(model, "VisA", class_name, device, return_kg=True, return_components=True)
        k, _ = compute_hybrid_k_regularization(model, components["hard_text"], components["soft_text"], model.hybrid_alpha_current)
        by_class[class_name] = text; kg_losses.append(kg); k_losses.append(k)
    text = torch.stack([by_class[name] for name in class_names]).permute(1, 0, 2, 3)
    kg_loss = torch.stack(kg_losses).mean(); k_loss = torch.stack(k_losses).mean()
    with policy.autocast(device):
        seg_tokens, det_tokens = model(image)
        seg_features = torch.stack(seg_tokens); det_features = torch.stack(det_tokens)
        cls_pred = torch.stack([torch.matmul(det_features[i].unsqueeze(1), text[i]).squeeze(1) for i in range(3)]).mean(0)
        cls_loss = F.cross_entropy(cls_pred, label)
        seg_pred = model.vision_text_fusion_gate_seg(seg_features, text, test_mode=False, cir_training=False)
        focal_loss = focal(seg_pred, mask)
        normal_dice = dice(seg_pred[:, 0], 1 - mask)
        abnormal_dice = dice(seg_pred[:, 1], mask)
        seg_loss = focal_loss + normal_dice + abnormal_dice
        existing_task = cls_loss + seg_loss + .01 * kg_loss + .002 * k_loss
        thbr = thbr_raw(seg_pred[:, 1], mask) if include_thbr else torch.zeros((), device=device, dtype=seg_pred.dtype)
    return {
        "existing_task": existing_task, "cls": cls_loss, "focal": focal_loss,
        "normal_dice": normal_dice, "abnormal_dice": abnormal_dice,
        "seg_pred": seg_pred, "thbr": thbr,
    }


def matched_pairs(score: torch.Tensor, mask: torch.Tensor):
    anomaly = mask > .5
    background = ~anomaly
    if int(anomaly.sum()) == 0 or int(background.sum()) == 0:
        return None
    k = min(int(anomaly.sum()), int(background.sum()))
    hard, _ = torch.topk(score[background], k=k, largest=True, sorted=True)
    weak, _ = torch.topk(score[anomaly], k=k, largest=False, sorted=True)
    return hard, weak


def thbr_raw(anomaly_score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    for score, sample_mask in zip(anomaly_score, mask[:, 0]):
        pairs = matched_pairs(score, sample_mask)
        if pairs is not None:
            hard, weak = pairs
            losses.append(F.softplus(hard - weak).mean())
    if not losses:
        return anomaly_score.sum() * 0.0
    return torch.stack(losses).mean()


def grad_vector(loss: torch.Tensor, parameters: list[torch.Tensor], retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(g.float().reshape(-1) if g is not None else torch.zeros_like(p, dtype=torch.float32).reshape(-1)) for g, p in zip(grads, parameters)])


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = float(left.norm().cpu() * right.norm().cpu())
    return float((left.dot(right).cpu() / denominator)) if denominator else float("nan")


def finite_median(values) -> float:
    """Median of defined cosine observations; NaN means no defined observation."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def gradient_redundancy(payload: dict, manifest: list[dict]) -> tuple[list[dict], dict]:
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    restore_rng_state(rng_from_payload(payload))
    model = make_model(payload, device, training_graph=True)
    model.hybrid_alpha_current = get_hybrid_alpha_for_epoch(START_EPOCH + 1, .2, 3)
    model.dfg_beta = get_dfg_beta_for_epoch(START_EPOCH + 1, "warmup010", .1, .1)
    params = [p for p in model.image_adapter.parameters() if p.requires_grad]
    focal = FocalLoss(); dice = BinaryDiceLoss()
    rows = []; ratios = []
    for attempt in range(GRADIENT_AUDIT_BATCHES):
        batch, image, mask, label = load_training_batch(payload, manifest, attempt)
        losses = build_training_losses(model, image, mask, label, batch["class_name"], device, policy, focal, dice, include_thbr=True)
        order = [("cls", losses["cls"]), ("focal", losses["focal"]), ("normal_dice", losses["normal_dice"]), ("abnormal_dice", losses["abnormal_dice"]), ("thbr", losses["thbr"]), ("existing_task", losses["existing_task"])]
        vectors = {}
        for index, (name, loss) in enumerate(order):
            # Keep the graph through the pixel-gradient audit; the final
            # focal-to-score gradient call frees it after all vector metrics.
            vectors[name] = grad_vector(loss, params, retain_graph=True)
        task_norm = float(vectors["existing_task"].norm().cpu())
        thbr_norm = float(vectors["thbr"].norm().cpu())
        ratio = thbr_norm / (task_norm + 1e-12)
        ratios.append(ratio)
        pixel_grad = torch.autograd.grad(losses["focal"], losses["seg_pred"], retain_graph=False, allow_unused=False)[0][:, 1].abs().float().detach().cpu().numpy()
        focal_overlap = []
        violation = []
        for score, sample_mask, pg in zip(losses["seg_pred"][:, 1].detach().float().cpu().numpy(), mask[:, 0].detach().cpu().numpy(), pixel_grad):
            pairs = matched_pairs(torch.from_numpy(score), torch.from_numpy(sample_mask))
            if pairs is None:
                continue
            hard, weak = pairs
            anomaly = sample_mask > .5; background = ~anomaly; k = len(hard)
            hard_values = score[background]
            hard_indices = np.argsort(-hard_values, kind="stable")[:k]
            top_grad_indices = np.argsort(-pg.reshape(-1), kind="stable")[:k]
            bg_flat_indices = np.flatnonzero(background.reshape(-1))
            focal_overlap.append(float(np.intersect1d(bg_flat_indices[hard_indices], top_grad_indices).size / max(1, k)))
            violation.append(float((hard.detach().numpy() > weak.detach().numpy()).mean()))
        cos_focal = cosine(vectors["thbr"], vectors["focal"])
        cos_normal = cosine(vectors["thbr"], vectors["normal_dice"])
        cos_abnormal = cosine(vectors["thbr"], vectors["abnormal_dice"])
        cos_existing = cosine(vectors["thbr"], vectors["existing_task"])
        row = {"attempt_index": attempt, **{key: manifest[attempt][key] for key in IDENTITY_KEYS}, "cls_grad_norm": float(vectors["cls"].norm().cpu()), "focal_grad_norm": float(vectors["focal"].norm().cpu()), "normal_dice_grad_norm": float(vectors["normal_dice"].norm().cpu()), "abnormal_dice_grad_norm": float(vectors["abnormal_dice"].norm().cpu()), "thbr_grad_norm": thbr_norm, "existing_task_grad_norm": task_norm, "thbr_to_existing_task_ratio": ratio, "thbr_cosine_to_focal": cos_focal, "thbr_cosine_to_normal_dice": cos_normal, "thbr_cosine_to_abnormal_dice": cos_abnormal, "thbr_cosine_to_existing_task": cos_existing, "thbr_cosine_to_focal_defined": bool(np.isfinite(cos_focal)), "thbr_cosine_to_normal_dice_defined": bool(np.isfinite(cos_normal)), "thbr_cosine_to_abnormal_dice_defined": bool(np.isfinite(cos_abnormal)), "thbr_cosine_to_existing_task_defined": bool(np.isfinite(cos_existing)), "focal_top_k_overlap_with_thbr_hard_background": float(np.mean(focal_overlap)) if focal_overlap else float("nan"), "thbr_matched_violation_fraction": float(np.mean(violation)) if violation else float("nan"), "finite": bool(all(torch.isfinite(v).all().item() for v in vectors.values())), "anomalous_samples": int((mask[:, 0] > .5).any(dim=(1, 2)).sum().item())}
        rows.append(row)
        del losses, vectors
        torch.cuda.empty_cache()
    valid = [row for row in rows if row["finite"] and np.isfinite(row["thbr_to_existing_task_ratio"])]
    R = float(np.median([row["thbr_to_existing_task_ratio"] for row in valid])) if valid else float("nan")
    lambda_thbr = float(.05 / R) if np.isfinite(R) and R > 0 else float("nan")
    with REDUNDANCY_CSV.open("w", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    summary = {
        "protocol_id": PROTOCOL_ID, "batch_count": len(rows), "valid_batch_count": len(valid),
        "rows": rows, "aggregation": {
            "thbr_grad_norm_median": float(np.median([r["thbr_grad_norm"] for r in valid])),
            "existing_task_grad_norm_median": float(np.median([r["existing_task_grad_norm"] for r in valid])),
            "thbr_to_existing_task_ratio_median": R,
            "thbr_cosine_to_focal_median": finite_median([r["thbr_cosine_to_focal"] for r in valid]),
            "thbr_cosine_to_normal_dice_median": finite_median([r["thbr_cosine_to_normal_dice"] for r in valid]),
            "thbr_cosine_to_abnormal_dice_median": finite_median([r["thbr_cosine_to_abnormal_dice"] for r in valid]),
            "thbr_cosine_to_existing_task_median": finite_median([r["thbr_cosine_to_existing_task"] for r in valid]),
            "thbr_cosine_defined_counts": {
                "focal": int(sum(r["thbr_cosine_to_focal_defined"] for r in valid)),
                "normal_dice": int(sum(r["thbr_cosine_to_normal_dice_defined"] for r in valid)),
                "abnormal_dice": int(sum(r["thbr_cosine_to_abnormal_dice_defined"] for r in valid)),
                "existing_task": int(sum(r["thbr_cosine_to_existing_task_defined"] for r in valid)),
            },
            "zero_gradient_counts": {
                "focal": int(sum(r["focal_grad_norm"] == 0.0 for r in valid)),
                "normal_dice": int(sum(r["normal_dice_grad_norm"] == 0.0 for r in valid)),
                "abnormal_dice": int(sum(r["abnormal_dice_grad_norm"] == 0.0 for r in valid)),
                "thbr": int(sum(r["thbr_grad_norm"] == 0.0 for r in valid)),
            },
            "focal_top_k_overlap_with_thbr_hard_background_mean": float(np.nanmean([r["focal_top_k_overlap_with_thbr_hard_background"] for r in valid])),
        },
        "score_space": "training seg_pred[:,1] post-fusion abnormal probability",
        "no_optimizer_update": True,
    }
    json_dump(REDUNDANCY_JSON, summary)
    calibration = {
        "protocol_id": PROTOCOL_ID, "batch_count": len(valid),
        "R_median_thbr_over_existing_task_gradient": R,
        "lambda_thbr": lambda_thbr,
        "target_auxiliary_effective_gradient_ratio": .05,
        "formula": "lambda_thbr = 0.05 / median(||grad(L_thbr_raw)||/(||grad(L_existing_task)||+1e-12))",
        "batch_identities": [{key: row[key] for key in IDENTITY_KEYS} for row in rows],
        "gradient_norms": [{key: row[key] for key in ("attempt_index", "thbr_grad_norm", "existing_task_grad_norm", "thbr_to_existing_task_ratio")} for row in rows],
        "finite_and_stable": bool(valid and np.isfinite(R) and R > 0 and np.isfinite(lambda_thbr)),
    }
    json_dump(CALIBRATION_JSON, calibration)
    del model
    torch.cuda.empty_cache()
    return rows, calibration


def audit_decision(endpoint_results: dict, redundancy: dict, calibration: dict) -> dict:
    e10 = endpoint_results[ARM_E10]; freeze = endpoint_results[ARM_FREEZE]; grad = endpoint_results[ARM_GRADBUDGET]
    boundary = json.loads(BOUNDARY_JSON.read_text())
    near_fractions = [boundary["aggregate"][arm]["mean_top_background_1pct_near_fraction"] for arm in (ARM_E10, ARM_FREEZE, ARM_GRADBUDGET)]
    # Audit-only rubric: tail is supported when high-background pixels are
    # concentrated in a small top tail and materially enter top-P predictions.
    top1 = [e10["aggregate_background_mass_concentration"]["0.01"], freeze["aggregate_background_mass_concentration"]["0.01"], grad["aggregate_background_mass_concentration"]["0.01"]]
    topP_near = [e10["top_P_composition_mean"]["near_background_fraction"], freeze["top_P_composition_mean"]["near_background_fraction"], grad["top_P_composition_mean"]["near_background_fraction"]]
    hard_tail_supported = bool(np.mean(top1) > .01 and np.mean(topP_near) > .0)
    max_near = max(near_fractions)
    min_near = min(near_fractions)
    boundary_risk = "DOMINANT" if min_near >= .80 else ("MATERIAL" if min_near >= .50 else "LOW")
    red = redundancy["aggregation"]
    overlap = red["focal_top_k_overlap_with_thbr_hard_background_mean"]
    cos_focal = red["thbr_cosine_to_focal_median"]
    focal_redundancy = "NEAR_EQUIVALENT" if cos_focal >= .95 and overlap >= .80 else ("MATERIAL" if cos_focal >= .70 or overlap >= .50 else "LOW")
    authorized = hard_tail_supported and boundary_risk != "DOMINANT" and focal_redundancy != "NEAR_EQUIVALENT" and calibration["finite_and_stable"]
    decision = {
        "protocol_id": PROTOCOL_ID,
        "audit_scope": "source-only frozen 96-image VisA endpoint plus fixed 16 source gradient batches",
        "hard_background_tail_premise": "SUPPORTED" if hard_tail_supported else "NOT_SUPPORTED",
        "boundary_artifact_risk": boundary_risk,
        "focal_redundancy": focal_redundancy,
        "thbr_training_authorized": bool(authorized),
        "rubric": {
            "hard_tail_support": "mean top-1% background score-mass concentration > uniform 1% baseline and nonzero mean near-background fraction in top-P predictions across audited arms",
            "boundary_dominant": "minimum arm mean top-background-1% near-boundary fraction >= 0.80",
            "boundary_material": "minimum arm mean top-background-1% near-boundary fraction >= 0.50",
            "focal_near_equivalent": "median THBR/focal parameter-gradient cosine >= 0.95 and focal top-K overlap with THBR hard background >= 0.80",
            "focal_material": "median cosine >= 0.70 or overlap >= 0.50",
        },
        "endpoint_evidence": {
            "safe_anchor_e10": {"pixel_metrics": e10["pixel_metrics"], "top1_background_mass": top1[0], "top_P_near_fraction": topP_near[0]},
            "late_freeze": {"pixel_metrics": freeze["pixel_metrics"], "top1_background_mass": top1[1], "top_P_near_fraction": topP_near[1]},
            "gradbudget_descriptive_invalid": {"pixel_metrics": grad["pixel_metrics"], "top1_background_mass": top1[2], "top_P_near_fraction": topP_near[2]},
        },
        "gradient_redundancy_evidence": red,
        "boundary_near_fraction_by_arm": dict(zip((ARM_E10, ARM_FREEZE, ARM_GRADBUDGET), near_fractions)),
        "prior_gradbudget_numerical_status": "INVALID_NUMERICAL",
        "no_optimizer_update_in_audit": True,
        "no_target_inference": True,
    }
    json_dump(AUDIT_DECISION_JSON, decision)
    AUDIT_DECISION_MD.write_text(
        f"# {PROTOCOL_ID} audit decision\n\n"
        f"- `HARD_BACKGROUND_TAIL_PREMISE={decision['hard_background_tail_premise']}`\n"
        f"- `BOUNDARY_ARTIFACT_RISK={boundary_risk}`\n"
        f"- `FOCAL_REDUNDANCY={focal_redundancy}`\n"
        f"- `THBR_TRAINING_AUTHORIZED={'YES' if authorized else 'NO'}`\n\n"
        "The audit is source-only. It uses the frozen 96-image VisA endpoint cohort, existing 7x7 boundary/near/far semantics, and 16 fixed E10-state source batches without optimizer updates. The GradBudget endpoint is descriptive evidence only and remains numerically invalid under its prior paired-run decision.\n\n"
        f"Audit rubric values: top-background-1% mass concentrations={top1}; top-P near-background fractions={topP_near}; mean top-background-1% near-boundary fractions={near_fractions}; THBR/focal gradient cosine median={cos_focal}; focal top-K overlap={overlap}.\n\n"
        + ("The audit authorizes exactly one bounded THBR candidate/control screen.\n" if authorized else "The audit rejects THBR training; no candidate screen is authorized.\n")
        + "\nNo target inference or automatic follow-up is authorized.\n"
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "audit", "gradients"), default="all")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    payload, identity = validate_start()
    write_score_semantics()
    manifest = collect_manifest(payload)
    json_dump(MANIFEST_JSON, {
        "protocol_id": PROTOCOL_ID, "dataset": "VisA", "split": "train", "attempt_count": len(manifest),
        "identity_fields": list(IDENTITY_KEYS), "start_checkpoint": identity,
        "loader": {"batch_size": 6, "shuffle": True, "num_workers": 6, "pin_memory": True, "persistent_workers": False, "generator_seed_formula": "seed + 104729 * epoch", "worker_seed_formula": "seed + 1000003 * epoch + worker_id"},
        "attempts": manifest, "no_optimizer_update": True,
    })
    if args.stage in ("all", "audit"):
        endpoint_results, _, _ = audit_endpoints(payload, torch.device("cuda:0"))
    else:
        endpoint_results = json.loads(PREMISE_JSON.read_text())["arms"]
    if args.stage in ("all", "gradients"):
        rows, calibration = gradient_redundancy(payload, manifest)
    else:
        calibration = json.loads(CALIBRATION_JSON.read_text())
        rows = json.loads(REDUNDANCY_JSON.read_text())["rows"]
    if args.stage == "all":
        audit_decision(endpoint_results, json.loads(REDUNDANCY_JSON.read_text()), calibration)


if __name__ == "__main__":
    main()
