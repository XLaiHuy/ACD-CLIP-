"""Train one P30 LOCO fold from frozen P27 Tier-A/Tier-B cache tensors."""
from __future__ import annotations

import argparse
import math
import random
import resource
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from model.phase2b_runtime import configure_canonical_fp32
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
from tools.sabra_v2.p30_objective import p30_directional_loss
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import CachedSourceDataset, atomic_write_json, sha256_file
from tools.sabra_v2.train_region_distill import ROOT


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-class", choices=EXPECTED_VISA_CLASSES, required=True)
    parser.add_argument("--visa-root", type=Path, required=True, help="provenance-only; no held files are opened")
    parser.add_argument("--p26-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--p30-execution-base-sha", required=True)
    parser.add_argument("--p30-prereg-sha", required=True)
    parser.add_argument("--p30-uuid", default=P30_UUID)
    parser.add_argument("--stage", choices=("smoke", "one_class", "subset", "full"), default="full")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--prefetch-factor", type=int, choices=(2, 4), default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--non-blocking", action="store_true")
    return parser


def _finite_gradient_stats(adapter: RegionResidualAdapter) -> dict[str, Any]:
    per_parameter: dict[str, Any] = {}
    nonfinite_count = 0
    total_elements = 0
    total_zero = 0
    for name, parameter in adapter.named_parameters():
        if parameter.grad is None:
            per_parameter[name] = {
                "norm": 0.0,
                "median_abs": 0.0,
                "max_abs": 0.0,
                "zero_fraction": 1.0,
                "nonfinite_count": int(parameter.numel()),
            }
            nonfinite_count += int(parameter.numel())
            total_elements += int(parameter.numel())
            total_zero += int(parameter.numel())
            continue
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        current_nonfinite = int((~finite).sum().detach().cpu())
        nonfinite_count += current_nonfinite
        total_elements += int(gradient.numel())
        total_zero += int((gradient == 0).sum().detach().cpu())
        finite_values = gradient[finite].abs()
        if finite_values.numel():
            median_abs = float(finite_values.median().detach().cpu())
            max_abs = float(finite_values.max().detach().cpu())
            norm = float(torch.linalg.vector_norm(gradient[finite]).detach().cpu())
        else:
            median_abs = math.nan
            max_abs = math.nan
            norm = math.nan
        per_parameter[name] = {
            "norm": norm,
            "median_abs": median_abs,
            "max_abs": max_abs,
            "zero_fraction": float((gradient == 0).float().mean().detach().cpu()),
            "nonfinite_count": current_nonfinite,
        }
    return {
        "per_parameter": per_parameter,
        "global_zero_fraction": total_zero / total_elements if total_elements else 1.0,
        "nonfinite_count": nonfinite_count,
    }


def _summarize_gradient_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sampled_steps": [], "sample_count": 0}
    names = tuple(samples[0]["per_parameter"])
    per_parameter: dict[str, Any] = {}
    for name in names:
        values = [sample["per_parameter"][name] for sample in samples]
        per_parameter[name] = {
            "norm_min": min(float(value["norm"]) for value in values),
            "norm_median": statistics.median(float(value["norm"]) for value in values),
            "norm_max": max(float(value["norm"]) for value in values),
            "median_abs_median": statistics.median(float(value["median_abs"]) for value in values),
            "max_abs_max": max(float(value["max_abs"]) for value in values),
            "zero_fraction_mean": statistics.fmean(float(value["zero_fraction"]) for value in values),
            "nonfinite_count_max": max(int(value["nonfinite_count"]) for value in values),
        }
    return {
        "sampled_steps": [int(sample["step"]) for sample in samples],
        "sample_count": len(samples),
        "global_zero_fraction_mean": statistics.fmean(float(sample["global_zero_fraction"]) for sample in samples),
        "nonfinite_count_max": max(int(sample["nonfinite_count"]) for sample in samples),
        "per_parameter": per_parameter,
    }


def _parameter_delta(initial: dict[str, torch.Tensor], adapter: RegionResidualAdapter) -> dict[str, float]:
    squared = 0.0
    max_abs = 0.0
    for name, parameter in adapter.named_parameters():
        difference = parameter.detach().cpu() - initial[name]
        squared += float(difference.square().sum())
        max_abs = max(max_abs, float(difference.abs().max()))
    return {"l2": math.sqrt(squared), "max_abs": max_abs}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_steps is not None and (not args.engineering_smoke or args.max_steps <= 0):
        raise RuntimeError("max-steps is available only for positive engineering-smoke runs")
    if args.engineering_smoke:
        if args.epochs <= 0 or args.batch_size != 1 or args.learning_rate <= 0 or args.seed != 0:
            raise RuntimeError("P30 smoke requires positive epochs, batch size 1, positive lr, and seed 0")
    elif (args.epochs, args.batch_size, args.learning_rate, args.seed) != (20, 1, 1e-3, 0):
        raise RuntimeError("scientific P30 training requires exactly 20 epochs, batch size 1, lr 0.001, seed 0")
    if args.p30_uuid != P30_UUID:
        raise RuntimeError("P30 UUID does not match the frozen preregistration")
    prereg = load_and_audit_p30_preregistration(P30_PREREGISTRATION_PATH, args.p30_prereg_sha)
    verify_p26_parent(args.p26_checkpoint, args.clip_asset, ROOT / "configs/phase2b_canonical_v1.json")
    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, args.held_class)
    provenance = p30_cache_provenance(args.metadata)
    source_dataset = CachedSourceDataset(inventory.fit_rows, args.held_class, args.cache_root, provenance)
    configure_canonical_fp32()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    if any(not parameter.requires_grad for parameter in adapter.parameters()):
        raise RuntimeError("all P30 adapter parameters must remain trainable")
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
    initial_state = {name: parameter.detach().cpu().clone() for name, parameter in adapter.named_parameters()}
    steps = 0
    step_seconds: list[float] = []
    sampled_gradients: list[dict[str, Any]] = []
    last_terms: dict[str, Any] | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    adapter.train()
    for _epoch in range(args.epochs):
        for batch in source_loader:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started_step = time.perf_counter()
            seg_features = batch["seg_features"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32, non_blocking=args.non_blocking)
            teacher_region = batch["teacher_region"].to(device=device, dtype=torch.float32, non_blocking=args.non_blocking)
            if teacher_region.requires_grad:
                raise RuntimeError("cached P30 teacher unexpectedly requires grad")
            student_region = adapter(seg_features)
            terms = p30_directional_loss(student_region, teacher_region)
            optimizer.zero_grad(set_to_none=True)
            terms.total.backward()
            gradient_stats = _finite_gradient_stats(adapter)
            if gradient_stats["nonfinite_count"]:
                raise FloatingPointError("non-finite P30 adapter gradient")
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            steps += 1
            step_seconds.append(time.perf_counter() - started_step)
            if steps == 1 or steps % 1000 == 0:
                gradient_stats["step"] = steps
                sampled_gradients.append(gradient_stats)
            last_terms = {
                "total": float(terms.total.detach().cpu()),
                "directional": float(terms.directional.detach().cpu()),
                "valid_count": terms.valid_count,
            }
            if args.max_steps is not None and steps >= args.max_steps:
                break
        if args.max_steps is not None and steps >= args.max_steps:
            break
    if not steps:
        raise RuntimeError("P30 training produced zero steps")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "p30_region_adapter.pt"
    completion_path = args.output / "TRAINING_COMPLETE.json"
    if checkpoint_path.exists() or completion_path.exists():
        raise RuntimeError("P30 training output already exists; refusing a second fold attempt")
    state_dict = {name: parameter.detach().cpu() for name, parameter in adapter.named_parameters()}
    payload = {
        "schema_version": "P30_REGION_ADAPTER_CHECKPOINT_V1",
        "status": "ENGINEERING_SMOKE_ONLY" if args.engineering_smoke else "FOLD_TRAINING_COMPLETE",
        "held_class": args.held_class,
        "state_dict": state_dict,
        "steps": steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "stage": args.stage,
        "objective": prereg["objective"]["name"],
        "p30_uuid": args.p30_uuid,
        "p30_preregistration_sha256": p30_preregistration_hash(P30_PREREGISTRATION_PATH),
        "p30_execution_base_sha": args.p30_execution_base_sha,
        "cache_provenance": provenance.as_dict(),
        "p26_checkpoint_sha256": provenance.p26_sha256,
        "clip_asset_sha256": provenance.clip_sha256,
        "config_sha256": provenance.config_sha256,
        "phase2b_optimization_steps": 0,
        "clip_optimization_steps": 0,
        "teacher_trainable": False,
        "source_mask_used_by_loss": False,
    }
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    parameter_delta = _parameter_delta(initial_state, adapter)
    result = {
        "schema_version": "P30_TRAINING_COMPLETE_V1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "held_class": args.held_class,
        "stage": args.stage,
        "p30_uuid": args.p30_uuid,
        "p30_preregistration_sha256": p30_preregistration_hash(P30_PREREGISTRATION_PATH),
        "p30_execution_base_sha": args.p30_execution_base_sha,
        "fit_records": len(inventory.fit_rows),
        "held_records_not_read": len(inventory.held_rows),
        "held_gt_reads": 0,
        "held_mask_reads": 0,
        "source_mask_used_by_loss": False,
        "steps": steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "last_terms": last_terms,
        "status": payload["status"],
        "training_seconds": time.perf_counter() - started,
        "median_step_seconds": statistics.median(step_seconds),
        "mean_step_seconds": statistics.fmean(step_seconds),
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "phase2b_optimization_steps": 0,
        "clip_optimization_steps": 0,
        "optimizer_steps": steps,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "teacher_trainable": False,
        "teacher_parameter_delta": 0.0,
        "student_parameter_delta": parameter_delta,
        "gradient_health": _summarize_gradient_samples(sampled_gradients),
    }
    atomic_write_json(completion_path, result)
    return result


def main() -> None:
    print(__import__("json").dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
