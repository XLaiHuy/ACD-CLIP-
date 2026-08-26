"""Freeze GT-free P30 held predictions from exact Tier-A cache tensors."""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
import uuid
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.p30_contract import (
    P30_PREREGISTRATION_PATH,
    P30_UUID,
    load_and_audit_p30_preregistration,
    p30_cache_provenance,
    p30_preregistration_hash,
)
from tools.sabra_v2.region_cache import TierADataset, atomic_write_json, sha256_file
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
    parser.add_argument("--p30-execution-base-sha", required=True)
    parser.add_argument("--p30-prereg-sha", required=True)
    parser.add_argument("--p30-uuid", default=P30_UUID)
    parser.add_argument("--stage", choices=("smoke", "one_class", "subset", "full"), default="full")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size != 1:
        raise RuntimeError("P30 held prediction batch size must remain exactly one")
    if args.p30_uuid != P30_UUID:
        raise RuntimeError("P30 UUID does not match the frozen preregistration")
    load_and_audit_p30_preregistration(P30_PREREGISTRATION_PATH, args.p30_prereg_sha)
    verify_p26_parent(args.p26_checkpoint, args.clip_asset, ROOT / "configs/phase2b_canonical_v1.json")
    provenance = p30_cache_provenance(args.metadata)
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=True)
    expected_hash = p30_preregistration_hash(P30_PREREGISTRATION_PATH)
    expected_status = "ENGINEERING_SMOKE_ONLY" if args.stage == "smoke" else "FOLD_TRAINING_COMPLETE"
    if (
        checkpoint.get("schema_version") != "P30_REGION_ADAPTER_CHECKPOINT_V1"
        or checkpoint.get("status") != expected_status
        or checkpoint.get("held_class") != args.held_class
        or checkpoint.get("stage") != args.stage
        or checkpoint.get("cache_provenance") != provenance.as_dict()
        or checkpoint.get("p30_execution_base_sha") != args.p30_execution_base_sha
        or checkpoint.get("p30_preregistration_sha256") != expected_hash
        or checkpoint.get("p30_uuid") != args.p30_uuid
    ):
        raise RuntimeError("P30 adapter checkpoint provenance mismatch")
    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, args.held_class)
    dataset = TierADataset(inventory.held_rows, args.cache_root, provenance)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    from tools.sabra_v2.region_adapter import RegionResidualAdapter

    adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    adapter.eval()
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch in loader:
            seg = batch["seg_features"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            native = batch["native_logits"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            student = forward_region_student(adapter, seg, native)
            records.append({
                "image_path": str(batch["image_path"][0]),
                "class_name": args.held_class,
                "native_abnormal_probability": student.native_probability[0, 1].detach().cpu(),
                "p30_abnormal_probability": student.deployed_probability[0, 1].detach().cpu(),
                "p30_region_residual": student.region_residual[:, 0].detach().cpu(),
            })
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "p30_held_predictions.pt"
    completion_path = args.output / "PREDICTION_COMPLETE.json"
    if output_path.exists() or completion_path.exists():
        raise RuntimeError("P30 prediction output already exists; refusing a second prediction attempt")
    payload = {
        "schema_version": "P30_IMMUTABLE_HELD_PREDICTIONS_V1",
        "held_class": args.held_class,
        "gt_used": False,
        "mask_reads": 0,
        "cache_provenance": provenance.as_dict(),
        "p30_execution_base_sha": args.p30_execution_base_sha,
        "p30_preregistration_sha256": expected_hash,
        "p30_uuid": args.p30_uuid,
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "records": records,
    }
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    output_path.chmod(0o444)
    result = {
        "schema_version": "P30_PREDICTION_COMPLETE_V1",
        "prediction_path": str(output_path),
        "prediction_sha256": sha256_file(output_path),
        "held_class": args.held_class,
        "stage": args.stage,
        "p30_uuid": args.p30_uuid,
        "records": len(records),
        "gt_used": False,
        "mask_reads": 0,
        "prediction_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "completion_status": "COMPLETE",
    }
    atomic_write_json(completion_path, result)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
