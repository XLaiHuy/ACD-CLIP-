"""Score immutable P30 predictions only after the prediction-freeze gate."""
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
from tools.sabra_v2.p30_contract import P30_PREREGISTRATION_PATH, P30_UUID, load_and_audit_p30_preregistration
from tools.sabra_v2.region_cache import atomic_write_json, sha256_file
from tools.sabra_v2.train_region_distill import ROOT


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-class", choices=EXPECTED_VISA_CLASSES, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--p30-prereg-sha", required=True)
    parser.add_argument("--p30-uuid", default=P30_UUID)
    parser.add_argument("--stage", choices=("smoke", "one_class", "subset", "full"), default="full")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.p30_uuid != P30_UUID:
        raise RuntimeError("P30 UUID does not match the frozen preregistration")
    load_and_audit_p30_preregistration(P30_PREREGISTRATION_PATH, args.p30_prereg_sha)
    payload = torch.load(args.predictions, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "P30_IMMUTABLE_HELD_PREDICTIONS_V1"
        or payload.get("held_class") != args.held_class
        or payload.get("gt_used") is not False
        or payload.get("mask_reads") != 0
        or payload.get("p30_preregistration_sha256") != args.p30_prereg_sha
        or payload.get("p30_uuid") != args.p30_uuid
    ):
        raise RuntimeError("P30 scoring requires matching GT-free immutable predictions")
    inventory = loco_inventory(read_visa_metadata(args.metadata), args.held_class)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(inventory.held_rows):
        raise RuntimeError("P30 frozen inventory mismatch")
    by_path = {str(record["image_path"]): record for record in records}
    if len(by_path) != len(records):
        raise RuntimeError("P30 duplicate frozen paths")
    native: list[np.ndarray] = []
    p30: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    mask_reads = 0
    for batch in DataLoader(VisaEvaluationDataset(inventory.held_rows, args.visa_root), batch_size=1, shuffle=False, num_workers=0):
        record = by_path.get(str(batch["image_path"][0]))
        if record is None:
            raise RuntimeError("missing P30 frozen prediction")
        for key, collection in (("native_abnormal_probability", native), ("p30_abnormal_probability", p30)):
            value = record.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (518, 518):
                raise RuntimeError("P30 frozen map shape mismatch")
            collection.append(value.numpy().astype(np.float32, copy=False).reshape(-1))
        labels.append(batch["mask"][0, 0].numpy().astype(np.uint8, copy=False).reshape(-1))
        mask_reads += int(batch["label"][0].item())
    native_metrics = exact_metrics(np.concatenate(native), np.concatenate(labels))
    p30_metrics = exact_metrics(np.concatenate(p30), np.concatenate(labels))
    result: dict[str, object] = {
        "schema_version": "P30_HELD_METRICS_V1",
        "held_class": args.held_class,
        "stage": args.stage,
        "p30_uuid": args.p30_uuid,
        "prediction_sha256": sha256_file(args.predictions),
        "fit_or_teacher_steps": 0,
        "native_metrics": native_metrics,
        "p30_metrics": p30_metrics,
        "delta": {key: p30_metrics[key] - native_metrics[key] for key in ("pAP", "pAUROC")},
        "held_mask_file_reads_after_prediction_freeze": mask_reads,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "p30_held_metrics.json"
    if output_path.exists():
        raise RuntimeError("P30 score output already exists; refusing a second score attempt")
    atomic_write_json(output_path, result)
    return {"metrics_path": str(output_path), **result}


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
