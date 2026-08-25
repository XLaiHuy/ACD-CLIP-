"""Short engineering-only benchmark and real-frozen-output cache parity audit."""
from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.phase2b_runtime import forward_phase2b
from tools.sabra.data import VisaEvaluationDataset, read_visa_metadata
from tools.sabra_v2.correction_teacher import build_source_teacher_region_target
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import CachedRegionDataset, validate_region_cache
from tools.sabra_v2.student_forward import forward_region_student, materialize_frozen_inputs
from tools.sabra_v2.train_region_distill import (
    ROOT,
    _build_exact_cache,
    _cache_provenance,
    _load_frozen_phase2b,
    _seed_everything,
)
from utils import calculate_seg_loss


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-class", default="candle")
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--p26-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loss(adapter: RegionResidualAdapter, features: torch.Tensor, logits: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor):
    student = forward_region_student(adapter, features, logits)
    distillation = F.smooth_l1_loss(student.region_residual, teacher.unsqueeze(0).expand(3, -1, -1, -1))
    localization = calculate_seg_loss(student.deployed_probability, mask)
    return student, distillation, localization, distillation + localization


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    device = torch.device("cuda")
    all_rows = read_visa_metadata(ROOT / "dataset/hub/VisA.jsonl")
    inventory = loco_inventory(all_rows, args.held_class)
    rows = inventory.fit_rows[: args.samples]
    phase2b, config = _load_frozen_phase2b(args.p26_checkpoint, args.clip_asset, device)
    provenance = _cache_provenance(args, rows)
    build = _build_exact_cache(args, phase2b, config, rows, provenance, device)
    cache = CachedRegionDataset(validate_region_cache(args.cache_dir, provenance, rows))
    raw_loader = DataLoader(VisaEvaluationDataset(rows, args.visa_root), batch_size=1, shuffle=False, num_workers=0)

    raw_batch = next(iter(raw_loader))
    _synchronize(device)
    start = time.perf_counter()
    frozen = forward_phase2b(phase2b, raw_batch["image"], raw_batch["class_name"], device, config, domain="Industrial", require_grad=False)
    _synchronize(device)
    uncached_forward_seconds = time.perf_counter() - start
    source_mask = raw_batch["mask"].to(device=device, dtype=torch.float32)
    teacher = build_source_teacher_region_target(frozen.native_logits, source_mask)
    direct_features, direct_logits = materialize_frozen_inputs(frozen.seg_features, frozen.native_logits)

    for name in ("seg_features", "native_logits", "teacher_region", "source_mask"):
        with (args.cache_dir / f"{name}.bin").open("rb") as handle:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    start = time.perf_counter()
    sample = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in cache[0].items()}
    cached_read_seconds = time.perf_counter() - start
    cached_features = sample["seg_features"].unsqueeze(1).to(device)
    cached_logits = sample["native_logits"].unsqueeze(1).to(device)
    cached_teacher = sample["teacher_region"].to(device)
    cached_mask = sample["source_mask"].unsqueeze(0).to(device)
    tensor_parity = {
        "seg_features": torch.equal(cached_features, direct_features),
        "native_logits": torch.equal(cached_logits, direct_logits),
        "teacher_region": torch.equal(cached_teacher, teacher),
        "source_mask": torch.equal(cached_mask, source_mask),
    }

    _seed_everything(args.seed)
    uncached_adapter = RegionResidualAdapter().to(device)
    cached_adapter = copy.deepcopy(uncached_adapter)
    uncached_optimizer = torch.optim.AdamW(uncached_adapter.parameters(), lr=1e-3)
    cached_optimizer = torch.optim.AdamW(cached_adapter.parameters(), lr=1e-3)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    start = time.perf_counter()
    uncached_frozen = forward_phase2b(phase2b, raw_batch["image"], raw_batch["class_name"], device, config, domain="Industrial", require_grad=False)
    uncached_mask = raw_batch["mask"].to(device=device, dtype=torch.float32)
    uncached_teacher = build_source_teacher_region_target(uncached_frozen.native_logits, uncached_mask)
    uncached_features, uncached_logits = materialize_frozen_inputs(uncached_frozen.seg_features, uncached_frozen.native_logits)
    uncached_student, uncached_distill, uncached_localize, uncached_total = _loss(uncached_adapter, uncached_features, uncached_logits, uncached_teacher, uncached_mask)
    uncached_optimizer.zero_grad(set_to_none=True); uncached_total.backward(); uncached_optimizer.step()
    _synchronize(device)
    uncached_step_seconds = time.perf_counter() - start
    uncached_gpu = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    start = time.perf_counter()
    cached_student, cached_distill, cached_localize, cached_total = _loss(cached_adapter, cached_features, cached_logits, cached_teacher, cached_mask)
    cached_optimizer.zero_grad(set_to_none=True); cached_total.backward(); cached_optimizer.step()
    _synchronize(device)
    cached_step_seconds = time.perf_counter() - start
    cached_gpu = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }

    gradient_errors = {}
    parameter_errors = {}
    for (name, direct_parameter), (_, cached_parameter) in zip(uncached_adapter.named_parameters(), cached_adapter.named_parameters()):
        gradient_errors[name] = float((direct_parameter.grad - cached_parameter.grad).abs().max())
        parameter_errors[name] = float((direct_parameter - cached_parameter).abs().max())
    max_gradient_error = max(gradient_errors.values())
    max_parameter_error = max(parameter_errors.values())
    tolerance = 1e-7
    source_counts = {held: len(loco_inventory(all_rows, held).fit_rows) for held in dict.fromkeys(row["class_name"] for row in all_rows)}
    total_source_samples = sum(source_counts.values())
    cache_bytes_per_sample = build["bytes"] / len(rows)
    result = {
        "label": "MEASURED ENGINEERING",
        "parity": {
            "tensor_exact": tensor_parity,
            "student_forward_exact": torch.equal(uncached_student.deployed_probability, cached_student.deployed_probability),
            "distillation_loss_exact": torch.equal(uncached_distill, cached_distill),
            "localization_loss_exact": torch.equal(uncached_localize, cached_localize),
            "total_loss_exact": torch.equal(uncached_total, cached_total),
            "max_gradient_abs_error": max_gradient_error,
            "max_optimizer_parameter_abs_error": max_parameter_error,
            "cuda_parity_tolerance": tolerance,
            "gradient_and_step_within_tolerance": max(max_gradient_error, max_parameter_error) <= tolerance,
            "tolerance_justification": "Frozen adaptive_average_pool2d CUDA backward is nondeterministic in torch 2.5.1; exact inputs/forward/loss are required and backward drift is bounded independently.",
        },
        "cache": {
            **build,
            "dtype": "float32",
            "format": "memory-mapped raw fixed-shape tensors with SHA-256 manifest",
            "bytes_per_sample": cache_bytes_per_sample,
            "estimated_bytes_per_fold": {held: int(count * cache_bytes_per_sample) for held, count in source_counts.items()},
        },
        "timing": {
            "uncached_forward_seconds": uncached_forward_seconds,
            "uncached_full_step_seconds": uncached_step_seconds,
            "cached_read_seconds": cached_read_seconds,
            "cached_training_step_seconds": cached_step_seconds,
            "measured_step_speedup": uncached_step_seconds / cached_step_seconds,
            "cached_read_throughput_bytes_per_second": cache_bytes_per_sample / max(cached_read_seconds, 1e-12),
        },
        "memory": {"uncached_gpu": uncached_gpu, "cached_gpu": cached_gpu, "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "projection": {
            "label": "PROJECTED ENGINEERING",
            "source_samples_across_12_folds": total_source_samples,
            "cache_build_seconds_12_folds": build["seconds_per_sample"] * total_source_samples,
            "cached_training_seconds_12_folds": cached_step_seconds * 20 * total_source_samples,
            "total_seconds_12_folds": (build["seconds_per_sample"] + 20 * cached_step_seconds) * total_source_samples,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
