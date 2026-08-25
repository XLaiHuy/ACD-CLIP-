"""Create immutable GT-free native/P27 held predictions from validated Tier A."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata
from tools.sabra_v2.audit_region_distill import PROTOCOL_PATH, audit_protocol
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import CacheProvenance, TierADataset, atomic_write_json, sha256_file
from tools.sabra_v2.student_forward import forward_region_student
from tools.sabra_v2.train_region_distill import ROOT


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-class", choices=EXPECTED_VISA_CLASSES, required=True)
    parser.add_argument("--visa-root", type=Path, required=True, help="provenance-only; no file below this root is opened")
    parser.add_argument("--p26-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--execution-base-sha", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--prefetch-factor", type=int, choices=(2, 4), default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--non-blocking", action="store_true")
    return parser


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def run(args: argparse.Namespace) -> dict[str, object]:
    audit_protocol(json.loads(PROTOCOL_PATH.read_text()))
    verify_p26_parent(args.p26_checkpoint, args.clip_asset, ROOT / "configs/phase2b_canonical_v1.json")
    execution_base = str(args.execution_base_sha or _head())
    provenance = CacheProvenance(execution_base, sha256_file(args.metadata))
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("status") != "FOLD_TRAINING_COMPLETE":
        raise RuntimeError("held prediction rejects engineering-smoke or incomplete adapter checkpoints")
    if checkpoint.get("held_class") != args.held_class:
        raise RuntimeError("adapter checkpoint held class mismatch")
    if checkpoint.get("cache_provenance") != provenance.as_dict():
        raise RuntimeError("adapter checkpoint cache provenance mismatch")
    inventory = loco_inventory(read_visa_metadata(args.metadata), args.held_class)
    dataset = TierADataset(inventory.held_rows, args.cache_root, provenance)
    loader_kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
    }
    if args.num_workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    loader = DataLoader(dataset, **loader_kwargs)
    device = torch.device(args.device)
    adapter = RegionResidualAdapter().to(device)
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()

    records: list[dict[str, object]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            seg_features = batch["seg_features"].permute(1, 0, 2, 3).to(
                device=device, dtype=torch.float32, non_blocking=args.non_blocking
            )
            native_logits = batch["native_logits"].permute(1, 0, 2, 3).to(
                device=device, dtype=torch.float32, non_blocking=args.non_blocking
            )
            student = forward_region_student(adapter, seg_features, native_logits)
            for index, image_path in enumerate(batch["image_path"]):
                records.append(
                    {
                        "image_path": image_path,
                        "class_name": args.held_class,
                        "native_abnormal_probability": student.native_probability[index, 1].detach().cpu(),
                        "p27_abnormal_probability": student.deployed_probability[index, 1].detach().cpu(),
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "p27_held_predictions.pt"
    payload = {
        "schema_version": "P27_IMMUTABLE_HELD_PREDICTIONS_V1",
        "held_class": args.held_class,
        "gt_used": False,
        "mask_reads": 0,
        "cache_provenance": provenance.as_dict(),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "records": records,
    }
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    output_path.chmod(0o444)
    result = {
        "prediction_path": str(output_path),
        "prediction_sha256": sha256_file(output_path),
        "held_class": args.held_class,
        "records": len(records),
        "gt_used": False,
        "mask_reads": 0,
        "prediction_seconds": time.perf_counter() - started,
        "completion_status": "COMPLETE",
    }
    atomic_write_json(args.output / "PREDICTION_COMPLETE.json", result)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
