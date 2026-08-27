"""Engineering-only P32 cached qualification; never runs scientific scoring.

This path intentionally exposes only the frozen source/fit cache.  It does
not accept held inputs, model checkpoints, or a scientific execution marker.
The optimizer steps performed here are engineering smoke/profile steps after
the P32 preregistration freeze, not a scientific Stage 2 attempt.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from model.phase2b_runtime import configure_canonical_fp32
from tools.sabra.data import read_visa_metadata
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p29_contract import p29_cache_provenance
from tools.sabra_v2.p32_objective import (
    P32_OBJECTIVE_NAME,
    P32_PREREGISTRATION_SHA256,
    p32_functional_margin_components,
    p32_objective_contract,
)
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import CachedSourceDataset, atomic_write_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_METADATA = ROOT / "dataset/hub/VisA.jsonl"
DEFAULT_HELD_CLASS = "candle"
P32_EPOCHS = 20
P32_BATCH_SIZE = 1
P32_LEARNING_RATE = 0.001
P32_SEED = 0
P32_WARMUP_STEPS = 5
P32_PROFILE_STEPS = 40
P30R1_SPEED_PROFILE = ROOT / "research/sabra_v2/region_distill/P30R1_SPEED_PROFILE.json"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--held-class", default=DEFAULT_HELD_CLASS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _finite_gradient_audit(adapter: RegionResidualAdapter) -> dict[str, Any]:
    gradients: list[torch.Tensor] = []
    missing = 0
    nonfinite = 0
    for parameter in adapter.parameters():
        if parameter.grad is None:
            missing += parameter.numel()
            continue
        gradient = parameter.grad.detach()
        gradients.append(gradient.reshape(-1))
        nonfinite += int((~torch.isfinite(gradient)).sum().item())
    if not gradients:
        flat = torch.zeros(1, dtype=torch.float32)
    else:
        flat = torch.cat(gradients)
    return {
        "missing_gradient_elements": missing,
        "nonfinite_count": nonfinite,
        "l2": float(torch.linalg.vector_norm(flat).cpu()),
        "max_abs": float(flat.abs().max().cpu()),
        "finite": nonfinite == 0 and missing == 0 and bool(torch.isfinite(flat).all()),
    }


def _parameter_delta(initial: dict[str, torch.Tensor], adapter: RegionResidualAdapter) -> dict[str, float]:
    squared = 0.0
    max_abs = 0.0
    for name, parameter in adapter.named_parameters():
        difference = parameter.detach().cpu() - initial[name]
        squared += float(difference.square().sum().item())
        max_abs = max(max_abs, float(difference.abs().max().item()))
    return {"l2": math.sqrt(squared), "max_abs": max_abs}


def _make_loader(
    metadata: Path,
    cache_root: Path,
    held_class: str,
) -> tuple[DataLoader, dict[str, Any]]:
    rows = read_visa_metadata(metadata)
    inventory = loco_inventory(rows, held_class)
    if held_class != DEFAULT_HELD_CLASS:
        raise RuntimeError("P32 engineering qualification is locked to candle")
    if (len(inventory.fit_rows), len(inventory.held_rows)) != (1962, 200):
        raise RuntimeError("P32 candle LOCO inventory changed")
    provenance = p29_cache_provenance(metadata)
    dataset = CachedSourceDataset(
        inventory.fit_rows,
        held_class,
        cache_root,
        provenance,
        load_source_mask=False,
        load_native_logits=False,
    )
    generator = torch.Generator().manual_seed(P32_SEED)
    loader = DataLoader(
        dataset,
        batch_size=P32_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )
    audit = {
        "metadata": str(metadata),
        "metadata_sha256": sha256_file(metadata),
        "cache_root": str(cache_root.resolve()),
        "cache_provenance": provenance.as_dict(),
        "held_class": held_class,
        "fit_records": len(inventory.fit_rows),
        "held_records_not_read": len(inventory.held_rows),
        "source_mask_loaded": False,
        "native_logits_loaded": False,
        "dataset_length": len(dataset),
    }
    return loader, audit


def _next_cached_batch(loader: DataLoader, iterator: Iterable[Any] | None) -> tuple[dict[str, Any], Iterable[Any]]:
    if iterator is None:
        iterator = iter(loader)
    try:
        return next(iterator), iterator  # type: ignore[arg-type]
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator  # type: ignore[arg-type]


def _materialize_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    forbidden = {"mask", "native_logits"}.intersection(batch)
    if forbidden:
        raise RuntimeError(f"forbidden cached fields reached P32 objective: {sorted(forbidden)}")
    if "seg_features" not in batch or "teacher_region" not in batch:
        raise RuntimeError("P32 cached batch is missing segmentation features or teacher region")
    seg_features = batch["seg_features"].permute(1, 0, 2, 3).to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )
    teacher_region = batch["teacher_region"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=False,
    )
    if teacher_region.requires_grad:
        raise RuntimeError("cached teacher unexpectedly requires gradients")
    return seg_features, teacher_region


def _train_step(
    adapter: RegionResidualAdapter,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    iterator: Iterable[Any] | None,
    device: torch.device,
    *,
    measure: bool,
) -> tuple[dict[str, Any], Iterable[Any]]:
    _sync(device)
    input_started = time.perf_counter()
    batch, iterator = _next_cached_batch(loader, iterator)
    data_loader_seconds = time.perf_counter() - input_started
    transfer_started = time.perf_counter()
    seg_features, teacher_region = _materialize_batch(batch, device)
    _sync(device)
    cache_tensor_transfer_seconds = time.perf_counter() - transfer_started
    input_seconds = data_loader_seconds + cache_tensor_transfer_seconds

    if device.type == "cuda" and measure:
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        objective_start = torch.cuda.Event(enable_timing=True)
        objective_end = torch.cuda.Event(enable_timing=True)
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        forward_start.record()
    else:
        forward_started = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    student_region = adapter(seg_features)
    if device.type == "cuda" and measure:
        forward_end.record()
        objective_start.record()
    else:
        forward_seconds = time.perf_counter() - forward_started
        objective_started = time.perf_counter()
    loss, student_effect, teacher_effect = p32_functional_margin_components(student_region, teacher_region)
    if device.type != "cuda" or not measure:
        objective_seconds = time.perf_counter() - objective_started
    if device.type == "cuda" and measure:
        objective_end.record()
        backward_start.record()
    else:
        backward_started = time.perf_counter()
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("P32 objective produced a non-finite loss")
    loss.backward()
    gradient_audit = _finite_gradient_audit(adapter)
    if not gradient_audit["finite"]:
        raise FloatingPointError(f"P32 objective produced an unhealthy gradient: {gradient_audit}")
    optimizer.step()
    if device.type == "cuda" and measure:
        backward_end.record()
        _sync(device)
        forward_seconds = float(forward_start.elapsed_time(forward_end)) / 1000.0
        objective_seconds = float(objective_start.elapsed_time(objective_end)) / 1000.0
        backward_seconds = float(backward_start.elapsed_time(backward_end)) / 1000.0
    else:
        backward_seconds = time.perf_counter() - backward_started
    # P30R1's frozen timing excludes DataLoader wait after a batch is yielded.
    # Keep both quantities so cache/I/O latency is not misattributed to math.
    comparable_step_seconds = (
        cache_tensor_transfer_seconds + forward_seconds + objective_seconds + backward_seconds
    )
    end_to_end_step_seconds = input_seconds + forward_seconds + objective_seconds + backward_seconds
    row = {
        "data_loader_seconds": data_loader_seconds,
        "cache_tensor_transfer_seconds": cache_tensor_transfer_seconds,
        "input_cache_seconds": input_seconds,
        "forward_seconds": forward_seconds,
        "objective_seconds": objective_seconds,
        "backward_optimizer_seconds": backward_seconds,
        "step_seconds": comparable_step_seconds,
        "end_to_end_step_seconds": end_to_end_step_seconds,
        "loss": float(loss.detach().cpu()),
        "student_effect_shape": list(student_effect.shape),
        "teacher_effect_shape": list(teacher_effect.shape),
        "loss_finite": True,
        "gradient": gradient_audit,
        "teacher_requires_grad": bool(teacher_region.requires_grad),
        "teacher_effect_requires_grad": bool(teacher_effect.requires_grad),
    }
    return row, iterator


def _summary(rows: list[dict[str, Any]], *, warmup_steps: int, label: str) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"empty P32 {label} profile")

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    step = values("step_seconds")
    end_to_end = values("end_to_end_step_seconds")
    objective = values("objective_seconds")
    input_cache = values("input_cache_seconds")
    data_loader = values("data_loader_seconds")
    cache_transfer = values("cache_tensor_transfer_seconds")
    forward = values("forward_seconds")
    backward = values("backward_optimizer_seconds")
    return {
        "label": label,
        "warmup_steps": warmup_steps,
        "measured_steps": len(rows),
        "optimizer_steps": len(rows),
        "median_step_seconds": statistics.median(step),
        "p90_step_seconds": float(torch.quantile(torch.tensor(step, dtype=torch.float64), 0.90)),
        "mean_step_seconds": statistics.fmean(step),
        "median_end_to_end_step_seconds": statistics.median(end_to_end),
        "p90_end_to_end_step_seconds": float(torch.quantile(torch.tensor(end_to_end, dtype=torch.float64), 0.90)),
        "mean_end_to_end_step_seconds": statistics.fmean(end_to_end),
        "input_cache_median_seconds": statistics.median(input_cache),
        "data_loader_median_seconds": statistics.median(data_loader),
        "cache_tensor_transfer_median_seconds": statistics.median(cache_transfer),
        "forward_median_seconds": statistics.median(forward),
        "objective_median_seconds": statistics.median(objective),
        "backward_optimizer_median_seconds": statistics.median(backward),
        "objective_fraction_of_step_median": statistics.median(objective) / statistics.median(step),
        "finite": all(bool(row["loss_finite"]) and bool(row["gradient"]["finite"]) for row in rows),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _write_checkpoint(
    path: Path,
    adapter: RegionResidualAdapter,
    *,
    optimizer_steps: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": "P32_ENGINEERING_CHECKPOINT_V1",
        "status": "ENGINEERING_QUALIFICATION_ONLY",
        "protocol_id": "P32",
        "objective": P32_OBJECTIVE_NAME,
        "objective_count": 1,
        "preregistration_sha256": P32_PREREGISTRATION_SHA256,
        "optimizer_steps": optimizer_steps,
        "state_dict": {name: value.detach().cpu() for name, value in adapter.named_parameters()},
        "teacher_trainable": False,
        "scientific_execution_marker": None,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or path.exists():
        raise RuntimeError(f"refusing to overwrite P32 engineering checkpoint: {path}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": payload["schema_version"],
        "strict_state_dict_reload": False,
    }


def _strict_reload_and_probe(
    checkpoint: Path,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "P32_ENGINEERING_CHECKPOINT_V1":
        raise RuntimeError("P32 engineering checkpoint schema mismatch")
    if payload.get("preregistration_sha256") != P32_PREREGISTRATION_SHA256:
        raise RuntimeError("P32 engineering checkpoint preregistration hash mismatch")
    restored = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    restored.load_state_dict(payload["state_dict"], strict=True)
    restored.eval()
    iterator: Iterable[Any] | None = None
    batch, iterator = _next_cached_batch(loader, iterator)
    seg_features, teacher_region = _materialize_batch(batch, device)
    with torch.no_grad():
        residual = restored(seg_features)
    if tuple(residual.shape) != (3, 1, 9, 9) or not bool(torch.isfinite(residual).all().item()):
        raise RuntimeError("strictly reloaded adapter failed the cached forward probe")
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "strict_state_dict_reload": True,
        "probe": "one cached Tier-A fit batch, adapter-only residual forward",
        "probe_shape": list(residual.shape),
        "probe_finite": True,
        "probe_teacher_requires_grad": bool(teacher_region.requires_grad),
        "probe_device": str(device),
    }


def _baseline_comparison(profile: dict[str, Any]) -> dict[str, Any]:
    baseline = _json(P30R1_SPEED_PROFILE)
    reference = baseline["warmed_profile"]
    reference_step = float(reference["median_step_seconds"])
    reference_objective = float(reference["objective_median_seconds"])
    observed_step = float(profile["median_step_seconds"])
    observed_objective = float(profile["objective_median_seconds"])
    return {
        "reference_artifact": str(P30R1_SPEED_PROFILE),
        "reference_protocol": "P30R1",
        "reference_warmed_profile_steps": int(reference["measured_steps"]),
        "reference_median_step_seconds": reference_step,
        "reference_objective_median_seconds": reference_objective,
        "observed_median_step_seconds": observed_step,
        "observed_objective_median_seconds": observed_objective,
        "end_to_end_overhead_percent_vs_P30R1": 100.0 * (observed_step / reference_step - 1.0),
        "objective_only_overhead_percent_vs_P30R1": 100.0 * (observed_objective / reference_objective - 1.0),
        "comparison_caveat": "comparison is against the frozen cached P30R1 CUDA profile on the same host; input/cache time is reported separately",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"P32 engineering output is non-empty: {args.output_root}")
    if not args.metadata.is_file() or not args.cache_root.is_dir():
        raise RuntimeError("P32 metadata and frozen cache root must exist")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")

    configure_canonical_fp32()
    random.seed(P32_SEED)
    torch.manual_seed(P32_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(P32_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    loader, data_audit = _make_loader(args.metadata, args.cache_root, args.held_class)
    adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    if any(not parameter.requires_grad for parameter in adapter.parameters()):
        raise RuntimeError("all P32 adapter parameters must remain trainable for engineering smoke")
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=P32_LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
    )
    initial_state = {name: value.detach().cpu().clone() for name, value in adapter.named_parameters()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    iterator: Iterable[Any] | None = None
    smoke_row, iterator = _train_step(adapter, optimizer, loader, iterator, device, measure=True)
    smoke_delta = _parameter_delta(initial_state, adapter)
    if smoke_delta["l2"] <= 0.0:
        raise RuntimeError("P32 cached engineering smoke did not update the student")
    data_audit["smoke_batch_fields"] = ["seg_features", "teacher_region"]
    data_audit["smoke_held_GT_read_count"] = 0
    data_audit["smoke_held_mask_read_count"] = 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_root / "p32_engineering_adapter.pt"
    checkpoint_info = _write_checkpoint(checkpoint, adapter, optimizer_steps=1)
    reload_info = _strict_reload_and_probe(checkpoint, loader, device)

    micro_rows: list[dict[str, Any]] = []
    for _ in range(5):
        row, iterator = _train_step(adapter, optimizer, loader, iterator, device, measure=True)
        micro_rows.append(row)
    micro = _summary(micro_rows, warmup_steps=0, label="5_step_microprofile")

    for _ in range(P32_WARMUP_STEPS):
        _unused_row, iterator = _train_step(adapter, optimizer, loader, iterator, device, measure=False)
    profile_rows: list[dict[str, Any]] = []
    for _ in range(P32_PROFILE_STEPS):
        row, iterator = _train_step(adapter, optimizer, loader, iterator, device, measure=True)
        profile_rows.append(row)
    profile = _summary(profile_rows, warmup_steps=P32_WARMUP_STEPS, label="40_step_warmed_profile")
    comparison = _baseline_comparison(profile)

    engineering_steps = 1 + len(micro_rows) + P32_WARMUP_STEPS + len(profile_rows)
    result = {
        "schema_version": "P32_ENGINEERING_RUN_V1",
        "status": "ENGINEERING_QUALIFICATION_ONLY",
        "protocol_id": "P32",
        "objective": p32_objective_contract(),
        "preregistration_sha256": P32_PREREGISTRATION_SHA256,
        "device": str(device),
        "training_schedule": {
            "epochs_frozen_for_future_science": P32_EPOCHS,
            "batch_size": P32_BATCH_SIZE,
            "learning_rate": P32_LEARNING_RATE,
            "seed": P32_SEED,
            "engineering_optimizer_steps": engineering_steps,
            "scientific_training_runs": 0,
        },
        "data_access_audit": data_audit,
        "model_forward_audit": {
            "new_clip_forwards": 0,
            "new_phase2b_forwards": 0,
            "new_teacher_forwards": 0,
            "adapter_forwards": engineering_steps + 1,
            "cached_student_only": True,
            "cache_rebuilds": 0,
            "held_GT_read_count": 0,
            "held_mask_read_count": 0,
        },
        "smoke": {
            "status": "PASS",
            "optimizer_steps": 1,
            "loss": smoke_row["loss"],
            "loss_finite": smoke_row["loss_finite"],
            "gradient": smoke_row["gradient"],
            "student_parameter_delta": smoke_delta,
            "frozen_parameter_delta": 0.0,
            "teacher_requires_grad": smoke_row["teacher_requires_grad"],
            "teacher_effect_requires_grad": smoke_row["teacher_effect_requires_grad"],
            "student_effect_shape": smoke_row["student_effect_shape"],
            "teacher_effect_shape": smoke_row["teacher_effect_shape"],
            "checkpoint": checkpoint_info,
            "strict_reload": reload_info,
        },
        "microprofile_5_step": micro,
        "warmed_profile_40_step": profile,
        "speed_comparison": comparison,
        "memory": {
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "retained_graph": False,
            "duplicate_teacher_tensor": False,
        },
        "scientific_safety": {
            "new_scientific_stage2_attempts": 0,
            "new_stage3_attempts": 0,
            "full_runs": 0,
            "held_result_tuning_iterations": 0,
            "new_clip_forwards": 0,
            "new_phase2b_forwards": 0,
            "cache_rebuilds": 0,
            "scientific_uuid_created": False,
            "execution_marker_created": False,
            "new_scientific_held_predictions": 0,
        },
        "gates": {
            "import_compile": "PASS",
            "objective_smoke": "PASS",
            "production_reference_parity": "covered by focused tests",
            "cached_batch_forward": "PASS",
            "backward": "PASS",
            "optimizer_step": "PASS",
            "checkpoint_save": "PASS",
            "checkpoint_strict_reload": "PASS",
            "microprofile": "PASS" if micro["finite"] else "ENGINEERING_STOP",
            "warmed_profile": "PASS" if profile["finite"] else "ENGINEERING_STOP",
        },
    }
    if not micro["finite"] or not profile["finite"]:
        result["status"] = "ENGINEERING_STOP"
    atomic_write_json(args.output_root / "P32_ENGINEERING_RUN.json", result)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
