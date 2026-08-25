"""Score immutable GT-free held predictions; this stage never fits P27 parameters."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.sabra.data import EXPECTED_VISA_CLASSES, VisaEvaluationDataset, read_visa_metadata
from tools.sabra_car.r0_direction import exact_metrics
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.train_region_distill import ROOT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-class", choices=EXPECTED_VISA_CLASSES, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    prediction_payload = torch.load(args.predictions, map_location="cpu", weights_only=True)
    if prediction_payload.get("held_class") != args.held_class or prediction_payload.get("gt_used") is not False:
        raise RuntimeError("scoring requires GT-free predictions for exactly the requested held class")
    inventory = loco_inventory(read_visa_metadata(args.metadata), args.held_class)
    records = prediction_payload.get("records")
    if not isinstance(records, list) or len(records) != len(inventory.held_rows):
        raise RuntimeError("held prediction count does not match immutable held inventory")
    by_path = {str(record["image_path"]): record for record in records}
    if len(by_path) != len(records):
        raise RuntimeError("held predictions contain duplicate image paths")
    native_scores: list[np.ndarray] = []
    p27_scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    loader = DataLoader(VisaEvaluationDataset(inventory.held_rows, args.visa_root), batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        image_path = str(batch["image_path"][0])
        record = by_path.get(image_path)
        if record is None:
            raise RuntimeError(f"missing frozen prediction for {image_path}")
        native_probability = record.get("native_abnormal_probability")
        p27_probability = record.get("p27_abnormal_probability")
        if any(not isinstance(value, torch.Tensor) or tuple(value.shape) != (518, 518) for value in (native_probability, p27_probability)):
            raise RuntimeError("held prediction must contain native and P27 [518,518] probability maps")
        native_scores.append(native_probability.numpy().astype(np.float32, copy=False).reshape(-1))
        p27_scores.append(p27_probability.numpy().astype(np.float32, copy=False).reshape(-1))
        labels.append(batch["mask"][0, 0].numpy().astype(np.uint8, copy=False).reshape(-1))
    flat_labels = np.concatenate(labels)
    native_metrics = exact_metrics(np.concatenate(native_scores), flat_labels)
    p27_metrics = exact_metrics(np.concatenate(p27_scores), flat_labels)
    result = {
        "held_class": args.held_class,
        "prediction_sha256": _sha256(args.predictions),
        "fit_or_teacher_steps": 0,
        "native_metrics": native_metrics,
        "p27_metrics": p27_metrics,
        "delta": {key: p27_metrics[key] - native_metrics[key] for key in ("pAP", "pAUROC")},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "p27_held_metrics.json"
    if output_path.exists():
        raise RuntimeError("immutable held metric artifact already exists")
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {"metrics_path": str(output_path), **result}


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
