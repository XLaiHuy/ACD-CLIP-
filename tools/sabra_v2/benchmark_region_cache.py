"""Bounded real-P26 exactness and uncached-versus-cached P27 benchmark."""
from __future__ import annotations

import argparse
import gc
import json
import resource
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


from tools.sabra.data import VisaEvaluationDataset, read_visa_metadata
from tools.sabra_v2.correction_teacher import build_source_teacher_region_target
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import CacheProvenance, CachedSourceDataset, atomic_write_json, sha256_file
from tools.sabra_v2.student_forward import forward_region_student, materialize_frozen_inputs
from tools.sabra_v2.train_region_distill import ROOT, _load_frozen_phase2b
from utils import calculate_seg_loss


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-class", default="candle")
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--p26-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.maximum(left.abs(), right.abs()).clamp_min(torch.finfo(left.dtype).tiny)
    return float(((left - right).abs() / denominator).max().item())


def _step(
    adapter: RegionResidualAdapter,
    optimizer: torch.optim.Optimizer,
    seg: torch.Tensor,
    native: torch.Tensor,
    mask: torch.Tensor,
    teacher: torch.Tensor,
) -> tuple[float, dict[str, torch.Tensor]]:
    student = forward_region_student(adapter, seg, native)
    distillation = F.smooth_l1_loss(student.region_residual, teacher.unsqueeze(0).expand(3, -1, -1, -1))
    localization = calculate_seg_loss(student.deployed_probability, mask)
    loss = distillation + localization
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {name: parameter.grad.detach().cpu().clone() for name, parameter in adapter.named_parameters()}
    optimizer.step()
    return float(loss.detach().cpu()), gradients


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples < 3:
        raise ValueError("benchmark requires at least three median samples")
    device = torch.device(args.device)
    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, args.held_class)
    selected = list(inventory.fit_rows[: args.samples + 1])
    direct_dataset = VisaEvaluationDataset(selected, args.visa_root)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
    provenance = CacheProvenance(head, sha256_file(args.metadata))
    cached_dataset = CachedSourceDataset(selected, args.held_class, args.cache_root, provenance)

    torch.manual_seed(0)
    direct_adapter = RegionResidualAdapter().to(device)
    initial_state = {name: value.detach().cpu().clone() for name, value in direct_adapter.state_dict().items()}
    direct_optimizer = torch.optim.AdamW(direct_adapter.parameters(), lr=1e-3)
    phase2b, config = _load_frozen_phase2b(args.p26_checkpoint, args.clip_asset, device)
    from model.phase2b_runtime import forward_phase2b

    direct_total: list[float] = []
    frozen_seconds: list[float] = []
    teacher_seconds: list[float] = []
    adapter_seconds: list[float] = []
    direct_losses: list[float] = []
    frozen_cpu: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    final_direct_gradients: dict[str, torch.Tensor] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index in range(args.samples + 1):
        _sync(device)
        total_started = time.perf_counter()
        item = direct_dataset[index]
        _sync(device)
        started = time.perf_counter()
        frozen = forward_phase2b(
            phase2b,
            item["image"].unsqueeze(0),
            [item["class_name"]],
            device,
            config,
            domain="Industrial",
            require_grad=False,
        )
        _sync(device)
        frozen_elapsed = time.perf_counter() - started
        mask = item["mask"].unsqueeze(0).to(device)
        started = time.perf_counter()
        teacher = build_source_teacher_region_target(frozen.native_logits, mask)
        _sync(device)
        teacher_elapsed = time.perf_counter() - started
        seg, native = materialize_frozen_inputs(frozen.seg_features, frozen.native_logits)
        started = time.perf_counter()
        torch.use_deterministic_algorithms(True, warn_only=True)
        loss, gradients = _step(direct_adapter, direct_optimizer, seg, native, mask, teacher)
        torch.use_deterministic_algorithms(False)
        _sync(device)
        adapter_elapsed = time.perf_counter() - started
        total_elapsed = time.perf_counter() - total_started
        frozen_cpu.append((seg.cpu(), native.cpu(), mask.cpu(), teacher.cpu()))
        if index:
            direct_total.append(total_elapsed)
            frozen_seconds.append(frozen_elapsed)
            teacher_seconds.append(teacher_elapsed)
            adapter_seconds.append(adapter_elapsed)
            direct_losses.append(loss)
            final_direct_gradients = gradients
    direct_state = {name: value.detach().cpu().clone() for name, value in direct_adapter.state_dict().items()}
    peak_uncached_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_uncached_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    del phase2b, direct_adapter, direct_optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    cached_adapter = RegionResidualAdapter().to(device)
    cached_adapter.load_state_dict(initial_state)
    cached_optimizer = torch.optim.AdamW(cached_adapter.parameters(), lr=1e-3)
    cached_total: list[float] = []
    cache_read_seconds: list[float] = []
    cached_losses: list[float] = []
    max_abs = {"seg_features": 0.0, "native_logits": 0.0, "teacher_region": 0.0, "source_mask": 0.0}
    max_rel = dict(max_abs)
    exact = {key: True for key in max_abs}
    final_cached_gradients: dict[str, torch.Tensor] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(True, warn_only=True)
    for index in range(args.samples + 1):
        _sync(device)
        total_started = time.perf_counter()
        started = time.perf_counter()
        item = cached_dataset[index]
        read_elapsed = time.perf_counter() - started
        cached_values = {
            "seg_features": item["seg_features"].unsqueeze(1),
            "native_logits": item["native_logits"].unsqueeze(1),
            "source_mask": item["mask"].unsqueeze(0),
            "teacher_region": item["teacher_region"].unsqueeze(0),
        }
        direct_values = {
            "seg_features": frozen_cpu[index][0],
            "native_logits": frozen_cpu[index][1],
            "source_mask": frozen_cpu[index][2],
            "teacher_region": frozen_cpu[index][3],
        }
        for name in max_abs:
            difference = (direct_values[name] - cached_values[name]).abs()
            max_abs[name] = max(max_abs[name], float(difference.max().item()))
            max_rel[name] = max(max_rel[name], _relative_error(direct_values[name], cached_values[name]))
            exact[name] = exact[name] and torch.equal(direct_values[name], cached_values[name])
        seg = cached_values["seg_features"].to(device)
        native = cached_values["native_logits"].to(device)
        mask = cached_values["source_mask"].to(device)
        teacher = cached_values["teacher_region"].to(device)
        loss, gradients = _step(cached_adapter, cached_optimizer, seg, native, mask, teacher)
        _sync(device)
        total_elapsed = time.perf_counter() - total_started
        if index:
            cached_total.append(total_elapsed)
            cache_read_seconds.append(read_elapsed)
            cached_losses.append(loss)
            final_cached_gradients = gradients
    peak_cached_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_cached_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0

    loss_exact = direct_losses == cached_losses
    loss_max_abs = max(abs(left - right) for left, right in zip(direct_losses, cached_losses))
    loss_tolerance_pass = loss_max_abs <= 1e-6
    gradient_exact = all(torch.equal(final_direct_gradients[name], final_cached_gradients[name]) for name in final_direct_gradients)
    optimizer_exact = all(torch.equal(direct_state[name], cached_adapter.state_dict()[name].cpu()) for name in direct_state)
    gradient_max_abs = max(
        float((final_direct_gradients[name] - final_cached_gradients[name]).abs().max().item())
        for name in final_direct_gradients
    )
    optimizer_max_abs = max(
        float((direct_state[name] - cached_adapter.state_dict()[name].cpu()).abs().max().item())
        for name in direct_state
    )
    gradient_tolerance_pass = gradient_max_abs <= 1e-6
    optimizer_tolerance_pass = optimizer_max_abs <= 1e-6
    uncached_median = statistics.median(direct_total)
    cached_median = statistics.median(cached_total)
    full_record_count = 2162
    full_training_steps = full_record_count * 11 * 20
    projected_cache_build = statistics.median(frozen_seconds) * full_record_count
    projected_training = cached_median * full_training_steps
    result = {
        "schema_version": "P27_ENGINEERING_BENCHMARK_V1",
        "sample_count": args.samples,
        "exactness": {
            "tensor_exact": exact,
            "max_absolute_difference": max_abs,
            "max_relative_difference": max_rel,
            "total_loss_exact": loss_exact,
            "total_loss_max_absolute_difference": loss_max_abs,
            "total_loss_tolerance": 1e-6,
            "total_loss_tolerance_pass": loss_tolerance_pass,
            "adapter_gradient_exact": gradient_exact,
            "optimizer_step_exact": optimizer_exact,
            "adapter_gradient_max_absolute_difference": gradient_max_abs,
            "optimizer_state_max_absolute_difference": optimizer_max_abs,
            "adapter_gradient_tolerance": 1e-6,
            "optimizer_state_tolerance": 1e-6,
            "adapter_gradient_tolerance_pass": gradient_tolerance_pass,
            "optimizer_state_tolerance_pass": optimizer_tolerance_pass,
            "non_exact_reason": "CUDA reflection_pad2d and adaptive_avg_pool2d backward use atomic accumulation and have no deterministic implementation" if not gradient_exact else None,
            "status": "PASS" if all(exact.values()) and loss_tolerance_pass and gradient_tolerance_pass and optimizer_tolerance_pass else "FAIL",
        },
        "timings": {
            "uncached_step_seconds": direct_total,
            "uncached_median_seconds_per_step": uncached_median,
            "cached_step_seconds": cached_total,
            "cached_median_seconds_per_step": cached_median,
            "measured_speedup": uncached_median / cached_median,
            "frozen_forward_median_seconds_per_sample": statistics.median(frozen_seconds),
            "cache_read_median_seconds_per_sample": statistics.median(cache_read_seconds),
            "teacher_build_median_seconds_per_sample": statistics.median(teacher_seconds),
            "adapter_forward_backward_step_median_seconds": statistics.median(adapter_seconds),
        },
        "projection": {
            "full_training_steps": full_training_steps,
            "full_tier_a_build_seconds": projected_cache_build,
            "full_training_seconds": projected_training,
            "full_tier_a_plus_training_hours": (projected_cache_build + projected_training) / 3600.0,
        },
        "memory": {
            "uncached_peak_gpu_allocated_bytes": peak_uncached_allocated,
            "uncached_peak_gpu_reserved_bytes": peak_uncached_reserved,
            "cached_peak_gpu_allocated_bytes": peak_cached_allocated,
            "cached_peak_gpu_reserved_bytes": peak_cached_reserved,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    atomic_write_json(args.output, result)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
