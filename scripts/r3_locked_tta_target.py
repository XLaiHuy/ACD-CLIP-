#!/usr/bin/env python3
"""Evaluate the pre-locked TTA-F policy on the frozen target datasets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc, average_precision

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import get_text_and_image_dataset
from dataset.info import CLASS_NAMES, DOMAINS
from h2_clean.exact_metrics import ExactBinaryAccumulator
from h2_clean.contract import sha256_file
from scripts.r3_source_tta import (
    CHECKPOINT_SHA256,
    IMG_SIZE,
    build_model,
    forward_logits_and_map,
    inverse_map,
    transform_image,
)


POLICY = ("identity", "hflip", "vflip", "hvflip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", type=str, default=CHECKPOINT_SHA256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    if sha256_file(args.checkpoint) != args.expected_sha256:
        raise ValueError("target checkpoint SHA256 does not match expected value")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, checkpoint_payload = build_model(device, args.checkpoint, args.expected_sha256)
    results = {}
    start = time.perf_counter()
    for dataset_name in ("Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir", "MVTec"):
        with torch.no_grad():
            text_embeddings = __import__("utils").get_multiple_adapted_text_embedding(model, dataset_name, device)
        dataset_results = {}
        for class_name, image_dataset in get_text_and_image_dataset(dataset_name, IMG_SIZE, "test").items():
            spool = ExactBinaryAccumulator(
                args.output.parent / ".spool" / dataset_name / class_name
            )
            labels_all = []
            scores_all = []
            loader = DataLoader(
                image_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for batch in loader:
                    image = batch["image"].to(device, non_blocking=True)
                    mask = batch["mask"].to(device, non_blocking=True)
                    label = batch["label"].to(device, non_blocking=True)
                    transformed = {
                        name: forward_logits_and_map(
                            model,
                            text_embeddings[class_name],
                            transform_image(image, name),
                            dataset_name,
                        )
                        for name in POLICY
                    }
                    seg_pred = torch.stack(
                        [inverse_map(transformed[name][1], name) for name in POLICY], dim=0
                    ).mean(dim=0)
                    logits = torch.stack([transformed[name][0] for name in POLICY], dim=0).mean(dim=0)
                    cls_score = F.softmax(logits, dim=1)[:, 1]
                    if DOMAINS[dataset_name] == "Medical":
                        image_score = cls_score
                    else:
                        pmax = seg_pred.flatten(start_dim=1).amax(dim=1)
                        image_score = cls_score * 0.9 + pmax * 0.1
                    if not torch.isfinite(seg_pred).all() or not torch.isfinite(image_score).all():
                        raise ValueError(f"non-finite TTA-F output in {dataset_name}/{class_name}")
                    spool.update(seg_pred, mask)
                    labels_all.append(label.detach().cpu())
                    scores_all.append(image_score.detach().cpu())
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            try:
                pixel_auc, pixel_ap = spool.compute()
            finally:
                spool.cleanup()
            labels = torch.cat(labels_all).flatten()
            scores = torch.cat(scores_all).flatten()
            if labels.min() != labels.max():
                image_auc = float(auroc(scores, labels, task="binary").item() * 100)
                image_ap = float(average_precision(scores, labels, task="binary").item() * 100)
            else:
                image_auc = image_ap = 0.0
            dataset_results[class_name] = {
                "pixel_auc": float(pixel_auc * 100),
                "pixel_ap": float(pixel_ap * 100),
                "image_auc": image_auc,
                "image_ap": image_ap,
            }
        macro = {
            key: sum(row[key] for row in dataset_results.values()) / len(dataset_results)
            for key in ("pixel_auc", "pixel_ap", "image_auc", "image_ap")
        }
        results[dataset_name] = {"per_class": dataset_results, "macro": macro}
    payload = {
        "protocol": "R3_LOCKED_TTA_F_TARGET_V1",
        "policy": list(POLICY),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "runtime_seconds": time.perf_counter() - start,
        "finite_outputs": True,
        "target_selection_or_tuning": False,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "runtime_seconds": payload["runtime_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
