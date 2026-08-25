"""Train one frozen-schedule P27 LOCO fold entirely from validated exact caches."""
from __future__ import annotations

import argparse
import json
import os
import random
import resource
import statistics
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.phase2b_runtime import configure_canonical_fp32
from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata
from tools.sabra_v2.audit_region_distill import PROTOCOL_PATH, audit_protocol
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import CacheProvenance, CachedSourceDataset, atomic_write_json, sha256_file
from tools.sabra_v2.student_forward import forward_region_student
from tools.sabra_v2.train_region_distill import ROOT
from utils import calculate_seg_loss


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-class", choices=EXPECTED_VISA_CLASSES, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--p26-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--execution-base-sha", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None, help="engineering-only cap")
    parser.add_argument("--engineering-smoke", action="store_true")
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_steps is not None and (not args.engineering_smoke or args.max_steps <= 0):
        raise RuntimeError("max-steps is available only for positive engineering-smoke runs")
    if not args.engineering_smoke and (args.epochs, args.batch_size, args.learning_rate) != (20, 1, 1e-3):
        raise RuntimeError("scientific training requires exactly 20 epochs, batch size 1, and learning rate 0.001")
    audit_protocol(json.loads(PROTOCOL_PATH.read_text()))
    config_path = ROOT / "configs/phase2b_canonical_v1.json"
    verify_p26_parent(args.p26_checkpoint, args.clip_asset, config_path)
    inventory = loco_inventory(read_visa_metadata(args.metadata), args.held_class)
    execution_base = str(args.execution_base_sha or _head())
    provenance = CacheProvenance(execution_base, sha256_file(args.metadata))
    source_dataset = CachedSourceDataset(inventory.fit_rows, args.held_class, args.cache_root, provenance)

    configure_canonical_fp32()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Canonical deployment's Gaussian blur differentiates through CUDA
    # reflection padding, whose backward has no declared deterministic kernel.
    # Warn-only is the strongest available policy; parity bounds this path.
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    adapter = RegionResidualAdapter().to(device)
    if any(not parameter.requires_grad for parameter in adapter.parameters()):
        raise RuntimeError("the full RegionResidualAdapter must remain trainable")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
        "generator": generator,
    }
    if args.num_workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    source_loader = DataLoader(source_dataset, **loader_kwargs)

    steps = 0
    step_seconds: list[float] = []
    last_loss: float | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started_training = time.perf_counter()
    adapter.train()
    for _epoch in range(args.epochs):
        for batch in source_loader:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started_step = time.perf_counter()
            source_mask = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=args.non_blocking)
            teacher_region = batch["teacher_region"].to(device=device, dtype=torch.float32, non_blocking=args.non_blocking)
            seg_features = batch["seg_features"].permute(1, 0, 2, 3).to(
                device=device, dtype=torch.float32, non_blocking=args.non_blocking
            )
            native_logits = batch["native_logits"].permute(1, 0, 2, 3).to(
                device=device, dtype=torch.float32, non_blocking=args.non_blocking
            )
            student = forward_region_student(adapter, seg_features, native_logits)
            distillation = F.smooth_l1_loss(
                student.region_residual,
                teacher_region.unsqueeze(0).expand(3, -1, -1, -1),
            )
            localization = calculate_seg_loss(student.deployed_probability, source_mask)
            loss = distillation + localization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in adapter.parameters()):
                raise FloatingPointError("missing or non-finite P27 adapter gradient")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - started_step)
            steps += 1
            last_loss = float(loss.detach().cpu())
            if args.max_steps is not None and steps >= args.max_steps:
                break
        if args.max_steps is not None and steps >= args.max_steps:
            break

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "p27_region_adapter.pt"
    payload = {
        "schema_version": "P27_REGION_ADAPTER_CHECKPOINT_V1",
        "status": "ENGINEERING_SMOKE_ONLY" if args.engineering_smoke else "FOLD_TRAINING_COMPLETE",
        "held_class": args.held_class,
        "state_dict": adapter.state_dict(),
        "steps": steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "cache_provenance": provenance.as_dict(),
        "p26_checkpoint_sha256": provenance.p26_sha256,
        "clip_asset_sha256": provenance.clip_sha256,
        "config_sha256": provenance.config_sha256,
        "phase2b_optimization_steps": 0,
        "clip_optimization_steps": 0,
    }
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_path)
    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "held_class": args.held_class,
        "fit_records": len(inventory.fit_rows),
        "held_records_not_read": len(inventory.held_rows),
        "held_gt_reads": 0,
        "held_mask_reads": 0,
        "steps": steps,
        "last_loss": last_loss,
        "status": payload["status"],
        "training_seconds": time.perf_counter() - started_training,
        "median_step_seconds": statistics.median(step_seconds) if step_seconds else None,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "phase2b_optimization_steps": 0,
        "clip_optimization_steps": 0,
    }
    atomic_write_json(args.output / "TRAINING_COMPLETE.json", result)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
