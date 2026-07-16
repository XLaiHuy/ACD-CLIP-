#!/usr/bin/env python3
"""Thin frozen-checkpoint wrapper around phase2b_anchor_diagnosis."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch

from dataset import get_text_and_image_dataset
from phase2b_anchor_diagnosis import (
    IMAGE_DATASETS,
    PIXEL_DATASETS,
    aggregate_rows,
    build_model,
    build_text_cache,
    evaluate_dataset,
    load_checkpoint,
    prepare_dataset,
    write_csv,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def evaluator_args(args):
    return argparse.Namespace(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cuda_device=args.cuda_device,
        img_size=args.img_size,
        pixel_stride=args.pixel_stride,
        max_samples=args.max_samples,
        max_samples_per_label=None,
        metric_thresholds=None,
        model_name="ViT-L-14-336",
        n_groups=3,
        lora_rank=16,
        lora_alpha=2.0,
        conv_lora_rank=8,
        conv_lora_alpha=2.0,
        conv_kernel_size_list=[3, 5],
        soft_prompt_ctx_len=4,
        soft_prompt_init="phrase",
        soft_prompt_init_phrase="a photo of a",
        dfg_mode="attn",
        dfg_attn_dim=256,
        dfg_attn_tau=8.0,
        use_ss2d_dfg=True,
        dfg_gamma_max=0.2,
        dfg_ss2d_fusion="weight_residual",
        dfg_beta=0.10,
        dfg_beta_schedule="warmup010",
        dfg_beta_target=0.10,
        fixed_prompt_config="current_shared",
        prompt_configs=["current_shared"],
        fixed_score_rule="cls_only",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint = Path(args.checkpoint).resolve()
    if (output_dir / "complete").exists():
        raise RuntimeError(f"Refusing to overwrite completed output: {output_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_sha = sha256_file(checkpoint)
    if actual_sha != args.expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {checkpoint}: {actual_sha}")
    with open(checkpoint, "rb") as handle:
        if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Unresolved Git LFS pointer: {checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=False)
    config = evaluator_args(args)
    device = torch.device(
        f"cuda:{config.cuda_device}" if torch.cuda.is_available() and config.cuda_device >= 0 else "cpu"
    )
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata_summary = {
        key: value for key, value in metadata.items()
        if key not in {"image_adapter", "text_adapter", "soft_prompt"}
    }
    metadata_summary["state_dict_keys"] = [
        key for key in ("image_adapter", "text_adapter", "soft_prompt") if key in metadata
    ]
    with (output_dir / "checkpoint_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata_summary, handle, default=str, indent=2, sort_keys=True)
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    resolved = vars(config) | {
        "state": args.state,
        "checkpoint": str(checkpoint),
        "sha256": actual_sha,
        "device": str(device),
        "image_score": "cls_only",
        "tta": False,
        "gaussian_sigma": 1.5,
        "gaussian_kernel": 9,
    }
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)

    model = build_model(config, device)
    model.eval()
    epoch = load_checkpoint(model, checkpoint, config, device)
    raw_rows, pixel_rows = [], []
    with torch.no_grad():
        for dataset_name in PIXEL_DATASETS:
            datasets = get_text_and_image_dataset(dataset_name, config.img_size, "test")
            text_cache = build_text_cache(model, dataset_name, list(datasets), device, "current_shared")
            for class_name, dataset in datasets.items():
                class_raw, pixel_row = evaluate_dataset(
                    model, dataset_name, class_name, prepare_dataset(dataset, config), text_cache,
                    device, config, epoch, "current_shared"
                )
                raw_rows.extend(class_raw)
                pixel_rows.append(pixel_row)

    aggregate_rows_out, image_rows = aggregate_rows(
        raw_rows, pixel_rows, IMAGE_DATASETS, ["cls_only"]
    )
    for row in pixel_rows:
        row["metric_type"] = "pixel"
    for row in image_rows:
        row["metric_type"] = "image"
    dataset_rows = pixel_rows + image_rows
    write_csv(
        output_dir / "dataset_metrics.csv", dataset_rows,
        ["metric_type", "dataset", "epoch", "prompt_config", "score_rule", "pixel_auc", "pixel_ap", "image_auc", "image_ap"],
    )
    with (output_dir / "dataset_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_rows, handle, indent=2, sort_keys=True)
    write_csv(
        output_dir / "image_score_raw_predictions.csv", raw_rows,
        ["dataset", "epoch", "prompt_config", "file_name", "label", "cls_score", "max_pixel", "top1pct_pixel"],
    )
    write_csv(
        output_dir / "macro_metrics.csv", aggregate_rows_out,
        ["epoch", "prompt_config", "score_rule", "pixel_auc_6", "pixel_ap_6", "image_auc_3", "image_ap_3", "image_n"],
    )
    if len(pixel_rows) != 6 or len(image_rows) != 3:
        raise RuntimeError(f"Expected 6 pixel and 3 image rows, got {len(pixel_rows)} and {len(image_rows)}")
    values = [v for row in dataset_rows for v in row.values() if isinstance(v, float)]
    if any(not math.isfinite(value) or value < 0 or value > 100 for value in values):
        raise RuntimeError("Metric outside [0, 100] or non-finite")
    (output_dir / "complete").write_text("complete\n", encoding="utf-8")


if __name__ == "__main__":
    main()
