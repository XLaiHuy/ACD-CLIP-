"""Produce GT-free held-fold P27 predictions only after a frozen fold checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tools.sabra.data import EXPECTED_VISA_CLASSES, VisaEvidenceDataset, read_visa_metadata
from tools.sabra_v2.audit_region_distill import PROTOCOL_PATH, audit_protocol
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.student_forward import assert_frozen_phase2b, forward_region_student
from tools.sabra_v2.train_region_distill import ROOT, _load_frozen_phase2b


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-class", choices=EXPECTED_VISA_CLASSES, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--p26-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    audit_protocol(json.loads(PROTOCOL_PATH.read_text()))
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("status") != "FOLD_TRAINING_COMPLETE":
        raise RuntimeError("held prediction rejects engineering-smoke or incomplete adapter checkpoints")
    if checkpoint.get("held_class") != args.held_class:
        raise RuntimeError("adapter checkpoint held class mismatch")
    inventory = loco_inventory(read_visa_metadata(args.metadata), args.held_class)
    device = torch.device(args.device)
    phase2b, config = _load_frozen_phase2b(args.p26_checkpoint, args.clip_asset, device)
    adapter = RegionResidualAdapter().to(device)
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()
    assert_frozen_phase2b(phase2b, adapter)
    loader = DataLoader(VisaEvidenceDataset(inventory.held_rows, args.visa_root), batch_size=args.batch_size, shuffle=False, num_workers=0)
    from model.phase2b_runtime import forward_phase2b

    records: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            frozen = forward_phase2b(phase2b, batch["image"], batch["class_name"], device, config, domain="Industrial", require_grad=False)
            student = forward_region_student(adapter, frozen.seg_features, frozen.native_logits)
            for index, image_path in enumerate(batch["image_path"]):
                records.append({"image_path": image_path, "class_name": args.held_class, "abnormal_probability": student.deployed_probability[index, 1].detach().cpu()})
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "p27_held_predictions.pt"
    torch.save({"held_class": args.held_class, "gt_used": False, "records": records}, output_path)
    return {"prediction_path": str(output_path), "records": len(records), "gt_used": False}


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
