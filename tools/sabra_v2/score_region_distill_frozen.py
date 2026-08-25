"""Score frozen native/P27 held maps with zero fitting, teacher, or selection steps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.sabra.data import EXPECTED_VISA_CLASSES, VisaEvaluationDataset, read_visa_metadata
from tools.sabra_car.r0_direction import exact_metrics
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.region_cache import atomic_write_json, sha256_file
from tools.sabra_v2.train_region_distill import ROOT


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-class", choices=EXPECTED_VISA_CLASSES, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    payload = torch.load(args.predictions, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "P27_IMMUTABLE_HELD_PREDICTIONS_V1":
        raise RuntimeError("wrong immutable prediction schema")
    if payload.get("held_class") != args.held_class or payload.get("gt_used") is not False:
        raise RuntimeError("scoring requires GT-free predictions for exactly the requested held class")
    if payload.get("mask_reads") != 0:
        raise RuntimeError("prediction stage accessed a held mask")
    inventory = loco_inventory(read_visa_metadata(args.metadata), args.held_class)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(inventory.held_rows):
        raise RuntimeError("held prediction count does not match immutable held inventory")
    by_path = {str(record["image_path"]): record for record in records}
    if len(by_path) != len(records):
        raise RuntimeError("held predictions contain duplicate image paths")

    native_scores: list[np.ndarray] = []
    p27_scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    mask_file_reads = 0
    loader = DataLoader(VisaEvaluationDataset(inventory.held_rows, args.visa_root), batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        image_path = str(batch["image_path"][0])
        record = by_path.get(image_path)
        if record is None:
            raise RuntimeError(f"missing frozen prediction for {image_path}")
        native = record.get("native_abnormal_probability")
        p27 = record.get("p27_abnormal_probability")
        if not isinstance(native, torch.Tensor) or tuple(native.shape) != (518, 518):
            raise RuntimeError("native prediction must be a [518,518] tensor")
        if not isinstance(p27, torch.Tensor) or tuple(p27.shape) != (518, 518):
            raise RuntimeError("P27 prediction must be a [518,518] tensor")
        native_scores.append(native.numpy().astype(np.float32, copy=False).reshape(-1))
        p27_scores.append(p27.numpy().astype(np.float32, copy=False).reshape(-1))
        labels.append(batch["mask"][0, 0].numpy().astype(np.uint8, copy=False).reshape(-1))
        mask_file_reads += int(batch["label"][0].item())
    native_metrics = exact_metrics(np.concatenate(native_scores), np.concatenate(labels))
    p27_metrics = exact_metrics(np.concatenate(p27_scores), np.concatenate(labels))
    result = {
        "schema_version": "P27_HELD_METRICS_V1",
        "held_class": args.held_class,
        "prediction_sha256": sha256_file(args.predictions),
        "fit_or_teacher_steps": 0,
        "native_metrics": native_metrics,
        "p27_metrics": p27_metrics,
        "delta": {key: p27_metrics[key] - native_metrics[key] for key in ("pAP", "pAUROC")},
        "held_mask_file_reads_after_prediction_freeze": mask_file_reads,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "p27_held_metrics.json"
    atomic_write_json(output_path, result)
    return {"metrics_path": str(output_path), **result}


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
