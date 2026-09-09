#!/usr/bin/env python3
"""Evaluate TTA-I/H/F on a deterministic held-out VisA category gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import BaseSingleClassDataset
from dataset.info import CLASS_NAMES, DATA_PATH, DOMAINS
from h2_clean.contract import sha256_file
from h2_clean.exact_metrics import ExactBinaryAccumulator
from model.adapter import ACDCLIP
from model.clip import create_model
from utils import get_multiple_adapted_text_embedding


IMG_SIZE = 518
CHECKPOINT_SHA256 = "727dc4813db1ef0c5a6db3a4cf15e916ad1413dcf629913358419d2734edecfe"
CLIP_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
VAL_CLASSES = ("candle", "macaroni1", "pcb3", "pipe_fryum")
POLICIES = {
    "TTA-I": ("identity",),
    "TTA-H": ("identity", "hflip"),
    "TTA-F": ("identity", "hflip", "vflip", "hvflip"),
}


def transform_image(image: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return image
    if name == "hflip":
        return torch.flip(image, dims=(-1,))
    if name == "vflip":
        return torch.flip(image, dims=(-2,))
    if name == "hvflip":
        return torch.flip(image, dims=(-2, -1))
    raise ValueError(f"unknown transform {name!r}")


def inverse_map(value: torch.Tensor, name: str) -> torch.Tensor:
    return transform_image(value, name)


def build_model(device: torch.device, checkpoint: Path) -> tuple[ACDCLIP, dict]:
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 does not match canonical A15")
    if sha256_file(Path("model/ViT-L-14-336px.pt")) != CLIP_SHA256:
        raise ValueError("CLIP SHA256 does not match canonical artifact")
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=IMG_SIZE,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=3,
        lora_rank=16,
        lora_alpha=2.0,
        conv_lora_rank=8,
        conv_lora_alpha=2.0,
        conv_kernel_size_list=[3, 5],
        dfg_mode="attn",
        dfg_attn_dim=256,
        dfg_attn_tau=8.0,
        use_ss2d_dfg=True,
        dfg_gamma_max=0.2,
        dfg_ss2d_fusion="weight_residual",
        dfg_beta=0.1,
        dfg_beta_schedule="warmup010",
        dfg_beta_target=0.1,
        dfg_beta_current=0.1,
    ).to(device)
    model.eval()
    checkpoint_payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.image_adapter.load_state_dict(checkpoint_payload["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(checkpoint_payload["text_adapter"], strict=True)
    if checkpoint_payload.get("use_hybrid_soft_prompt") or checkpoint_payload.get("prompt_mode") == "hybrid":
        model.prompt_mode = "hybrid"
        model.use_hybrid_soft_prompt = True
        model.use_soft_prompt = False
        if "soft_prompt" in checkpoint_payload:
            model.soft_prompt.load_state_dict(checkpoint_payload["soft_prompt"], strict=True)
        model.hybrid_alpha_current = float(checkpoint_payload.get("hybrid_alpha_current", 0.2))
    model.dfg_weight_residual_fp32 = bool(checkpoint_payload.get("dfg_weight_residual_fp32", True))
    model.set_dfg_beta(float(checkpoint_payload.get("dfg_beta_current", checkpoint_payload.get("dfg_beta", 0.1))))
    return model, checkpoint_payload


def forward_logits_and_map(
        model,
        text_features: torch.Tensor,
        image: torch.Tensor,
        dataset_name: str,
):
    seg_tokens, det_tokens = model(image)
    seg_features = torch.stack(seg_tokens, dim=0)
    det_features = torch.stack(det_tokens, dim=0)
    batch_size = seg_features.shape[1]
    epoch_text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1, 1)
    cls_logits = torch.stack([
        torch.matmul(det_features[group].unsqueeze(1), epoch_text_features[group]).squeeze(1)
        for group in range(det_features.shape[0])
    ], dim=0).mean(dim=0)
    seg_pred = model.vision_text_fusion_gate_seg(
        seg_features,
        epoch_text_features,
        test_mode=True,
        domain=DOMAINS[dataset_name],
    )
    return cls_logits, seg_pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    if tuple(VAL_CLASSES) != tuple(sorted(set(VAL_CLASSES))):
        raise ValueError("held-out categories must be unique and sorted")
    if not set(VAL_CLASSES).issubset(CLASS_NAMES["VisA"]):
        raise ValueError("held-out category is not a VisA class")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, checkpoint_payload = build_model(device, args.checkpoint)
    with torch.no_grad():
        text_embeddings = get_multiple_adapted_text_embedding(model, "VisA", device)

    accumulators = {
        policy: {
            cls: ExactBinaryAccumulator(args.output.parent / ".spool" / policy / cls)
            for cls in VAL_CLASSES
        }
        for policy in POLICIES
    }
    image_labels = {policy: {cls: [] for cls in VAL_CLASSES} for policy in POLICIES}
    image_scores = {policy: {cls: [] for cls in VAL_CLASSES} for policy in POLICIES}
    start = time.perf_counter()
    with torch.no_grad():
        for class_name in VAL_CLASSES:
            dataset = BaseSingleClassDataset(
                DATA_PATH["VisA"], "./dataset/hub/VisA.jsonl", IMG_SIZE, class_name
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                transformed = {}
                for transform_name in ("identity", "hflip", "vflip", "hvflip"):
                    transformed[transform_name] = forward_logits_and_map(
                        model, text_embeddings[class_name], transform_image(image, transform_name), "VisA"
                    )
                for policy, transforms_for_policy in POLICIES.items():
                    maps = torch.stack([
                        inverse_map(transformed[name][1], name) for name in transforms_for_policy
                    ], dim=0).mean(dim=0)
                    logits = torch.stack([
                        transformed[name][0] for name in transforms_for_policy
                    ], dim=0).mean(dim=0)
                    scores = F.softmax(logits, dim=1)[:, 1]
                    if not torch.isfinite(maps).all() or not torch.isfinite(scores).all():
                        raise ValueError(f"non-finite output in {policy}/{class_name}")
                    accumulators[policy][class_name].update(maps, mask)
                    image_labels[policy][class_name].append(label.detach().cpu())
                    image_scores[policy][class_name].append(scores.detach().cpu())
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    results = {}
    for policy, by_class in accumulators.items():
        per_class = {}
        for class_name, accumulator in by_class.items():
            try:
                pixel_auc, pixel_ap = accumulator.compute()
            finally:
                accumulator.cleanup()
            labels = torch.cat(image_labels[policy][class_name]).flatten()
            scores = torch.cat(image_scores[policy][class_name]).flatten()
            if labels.min() != labels.max():
                from torchmetrics.functional import auroc, average_precision
                image_auc = float(auroc(scores, labels, task="binary").item() * 100)
                image_ap = float(average_precision(scores, labels, task="binary").item() * 100)
            else:
                image_auc = image_ap = 0.0
            per_class[class_name] = {
                "pixel_auc": float(pixel_auc * 100),
                "pixel_ap": float(pixel_ap * 100),
                "image_auc": image_auc,
                "image_ap": image_ap,
            }
        macro = {
            key: sum(row[key] for row in per_class.values()) / len(per_class)
            for key in ("pixel_auc", "pixel_ap", "image_auc", "image_ap")
        }
        results[policy] = {
            "transforms": list(POLICIES[policy]),
            "latency_multiplier": len(POLICIES[policy]),
            "per_class": per_class,
            "macro": macro,
        }
    payload = {
        "protocol": "R3_SOURCE_CATEGORY_HELD_OUT_V1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "held_out_categories": list(VAL_CLASSES),
        "runtime_seconds": time.perf_counter() - start,
        "finite_outputs": True,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "runtime_seconds": payload["runtime_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
