"""Run an engineering-only P30R1 cached adapter path; never score science."""
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
from tools.sabra_v2.p30r1_contract import (
    P30R1_PREREGISTRATION_PATH,
    P30R1_UUID,
    load_and_audit_p30r1_preregistration,
    p30r1_cache_provenance,
    p30r1_preregistration_hash,
)
from tools.sabra_v2.p30r1_objective import (
    P30R1_OBJECTIVE_NAME,
    p30r1_teacher_relative_components,
)
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import CachedSourceDataset, atomic_write_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_VISA_ROOT = Path("/workspace/data/source/visa_unpack")
DEFAULT_P26_CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
DEFAULT_CLIP_ASSET = ROOT / "model/ViT-L-14-336px.pt"
P30R1_EPOCHS = 20
P30R1_BATCH_SIZE = 1
P30R1_LEARNING_RATE = 0.001
P30R1_SEED = 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-class", choices=EXPECTED_VISA_CLASSES, required=True)
    parser.add_argument("--visa-root", type=Path, default=DEFAULT_VISA_ROOT, help="provenance-only; no held files are opened")
    parser.add_argument("--p26-checkpoint", type=Path, default=DEFAULT_P26_CHECKPOINT)
    parser.add_argument("--clip-asset", type=Path, default=DEFAULT_CLIP_ASSET)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--execution-base-sha", required=True)
    parser.add_argument("--preregistration-sha", required=True)
    parser.add_argument("--p30r1-uuid", default=P30R1_UUID)
    parser.add_argument(
        "--stage",
        choices=("engineering_smoke", "engineering_microprofile", "engineering_profile"),
        default="engineering_smoke",
    )
    parser.add_argument("--epochs", type=int, default=P30R1_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=P30R1_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=P30R1_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=P30R1_SEED)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--prefetch-factor", type=int, choices=(2, 4), default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--non-blocking", action="store_true")
    return parser


def _finite_gradient_summary(
    adapter: RegionResidualAdapter,
    *,
    collect_details: bool,
) -> tuple[dict[str, Any], torch.Tensor]:
    per_parameter: dict[str, Any] = {}
    flattened: list[torch.Tensor] = []
    missing = 0
    for name, parameter in adapter.named_parameters():
        if parameter.grad is None:
            missing += int(parameter.numel())
            per_parameter[name] = {
                "norm": 0.0,
                "median_abs": 0.0,
                "max_abs": 0.0,
                "zero_fraction": 1.0,
                "nonfinite_count": int(parameter.numel()),
            }
            continue
        gradient = parameter.grad.detach()
        flattened.append(gradient.reshape(-1))
        if collect_details:
            finite = torch.isfinite(gradient)
            finite_values = gradient[finite].abs()
            per_parameter[name] = {
                "norm": torch.linalg.vector_norm(gradient),
                "median_abs": finite_values.median() if finite_values.numel() else torch.tensor(float("nan"), device=gradient.device),
                "max_abs": finite_values.max() if finite_values.numel() else torch.tensor(float("nan"), device=gradient.device),
                "zero_fraction": (gradient == 0).float().mean(),
                "nonfinite_count": (~finite).sum(),
            }
    if not flattened:
        flat = torch.zeros(1, device=next(adapter.parameters()).device)
    else:
        flat = torch.cat(flattened)
    summary = {
        "per_parameter": per_parameter,
        "missing_gradient_elements": missing,
        "global_zero_fraction": (flat == 0).float().mean() if collect_details else torch.tensor(0.0, device=flat.device),
        "nonfinite_count": (~torch.isfinite(flat)).sum(),
        "l2": torch.linalg.vector_norm(flat) if collect_details else torch.tensor(0.0, device=flat.device),
        "max_abs": flat.abs().max() if collect_details else torch.tensor(0.0, device=flat.device),
    }
    return summary, flat


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sampled_steps": [], "sample_count": 0}
    result: dict[str, Any] = {
        "sampled_steps": [int(sample["step"]) for sample in samples],
        "sample_count": len(samples),
        "global_zero_fraction": statistics.fmean(float(sample["global_zero_fraction"]) for sample in samples),
        "nonfinite_count_max": max(int(sample["nonfinite_count"]) for sample in samples),
        "missing_gradient_elements_max": max(int(sample["missing_gradient_elements"]) for sample in samples),
        "l2_min": min(float(sample["l2"]) for sample in samples),
        "l2_max": max(float(sample["l2"]) for sample in samples),
        "max_abs_max": max(float(sample["max_abs"]) for sample in samples),
        "per_parameter": {},
    }
    names = tuple(samples[0]["per_parameter"])
    for name in names:
        values = [sample["per_parameter"][name] for sample in samples]
        result["per_parameter"][name] = {
            "norm_min": min(float(value["norm"]) for value in values),
            "norm_median": statistics.median(float(value["norm"]) for value in values),
            "norm_max": max(float(value["norm"]) for value in values),
            "median_abs_median": statistics.median(float(value["median_abs"]) for value in values),
            "max_abs_max": max(float(value["max_abs"]) for value in values),
            "zero_fraction_mean": statistics.fmean(float(value["zero_fraction"]) for value in values),
            "nonfinite_count_max": max(int(value["nonfinite_count"]) for value in values),
        }
    return result


def _parameter_delta(initial: dict[str, torch.Tensor], adapter: RegionResidualAdapter) -> dict[str, float]:
    squared = 0.0
    max_abs = 0.0
    for name, parameter in adapter.named_parameters():
        difference = parameter.detach().cpu() - initial[name]
        squared += float(difference.square().sum())
        max_abs = max(max_abs, float(difference.abs().max()))
    return {"l2": math.sqrt(squared), "max_abs": max_abs}


def _event_elapsed_ms(events: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> list[float]:
    return [float(start.elapsed_time(end)) for start, end in events]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.p30r1_uuid != P30R1_UUID:
        raise RuntimeError("P30R1 UUID does not match the frozen preregistration")
    if args.cache_root.resolve() != DEFAULT_CACHE_ROOT.resolve():
        raise RuntimeError("P30R1 must reuse the frozen P27 cache root")
    if args.max_steps <= 0 or args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise RuntimeError("max-steps must be positive and greater than warmup-steps")
    if (args.epochs, args.batch_size, args.learning_rate, args.seed) != (
        P30R1_EPOCHS,
        P30R1_BATCH_SIZE,
        P30R1_LEARNING_RATE,
        P30R1_SEED,
    ):
        raise RuntimeError("engineering path must retain the frozen P30R1 schedule")
    if not args.metadata.is_file() or not args.cache_root.is_dir() or not args.visa_root.is_dir():
        raise RuntimeError("P30R1 metadata, cache root, and VisA root must exist")
    prereg = load_and_audit_p30r1_preregistration(P30R1_PREREGISTRATION_PATH, args.preregistration_sha)
    verify_p26_parent(args.p26_checkpoint, args.clip_asset, ROOT / "configs/phase2b_canonical_v1.json")
    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, args.held_class)
    provenance = p30r1_cache_provenance(args.metadata)
    source_dataset = CachedSourceDataset(
        inventory.fit_rows,
        args.held_class,
        args.cache_root,
        provenance,
        load_source_mask=False,
        load_native_logits=False,
    )
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
        raise RuntimeError("all RegionResidualAdapter parameters must remain trainable")
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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    cuda_forward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    cuda_objective_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    host_steps: list[float] = []
    host_forward: list[float] = []
    host_objective: list[float] = []
    sampled_gradients: list[dict[str, Any]] = []
    finite_loss_flags: list[torch.Tensor] = []
    finite_gradient_flags: list[torch.Tensor] = []
    last_loss: torch.Tensor | None = None
    exact_zero_teacher_flags: list[torch.Tensor] = []
    teacher_scale_detached = True
    steps = 0
    measured_steps = 0
    started = time.perf_counter()
    adapter.train()
    while steps < args.max_steps:
        for batch in source_loader:
            if steps >= args.max_steps:
                break
            steps += 1
            measure = steps > args.warmup_steps
            if device.type == "cuda" and measure:
                total_start = torch.cuda.Event(enable_timing=True)
                total_end = torch.cuda.Event(enable_timing=True)
                forward_start = torch.cuda.Event(enable_timing=True)
                forward_end = torch.cuda.Event(enable_timing=True)
                objective_start = torch.cuda.Event(enable_timing=True)
                objective_end = torch.cuda.Event(enable_timing=True)
                total_start.record()
                cuda_forward_events.append((forward_start, forward_end))
                cuda_objective_events.append((objective_start, objective_end))
            elif device.type != "cuda" and measure:
                step_started = time.perf_counter()
            seg_features = batch["seg_features"].permute(1, 0, 2, 3).to(
                device=device, dtype=torch.float32, non_blocking=args.non_blocking
            )
            teacher_region = batch["teacher_region"].to(
                device=device, dtype=torch.float32, non_blocking=args.non_blocking
            )
            if teacher_region.requires_grad:
                raise RuntimeError("cached P30R1 teacher unexpectedly requires grad")
            exact_zero_teacher_flags.append(torch.all(teacher_region == 0.0, dim=(1, 2)).sum().detach())
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda" and measure:
                forward_start.record()
            elif measure:
                forward_started = time.perf_counter()
            student_region = adapter(seg_features)
            if device.type == "cuda" and measure:
                forward_end.record()
                objective_start.record()
            elif measure:
                host_forward.append(time.perf_counter() - forward_started)
                objective_started = time.perf_counter()
            loss, _normalized_student, _normalized_teacher, teacher_scale = p30r1_teacher_relative_components(
                student_region,
                teacher_region,
            )
            teacher_scale_detached = teacher_scale_detached and not teacher_scale.requires_grad
            if device.type == "cuda" and measure:
                objective_end.record()
            elif measure:
                host_objective.append(time.perf_counter() - objective_started)
            last_loss = loss.detach()
            finite_loss_flags.append(torch.isfinite(loss.detach()))
            loss.backward()
            collect_gradient_details = measure and (
                measured_steps == 0
                or measured_steps + 1 == args.max_steps - args.warmup_steps
            )
            gradient_summary, flat_gradient = _finite_gradient_summary(
                adapter,
                collect_details=collect_gradient_details,
            )
            finite_gradient_flags.append(
                torch.isfinite(flat_gradient).all()
                & (gradient_summary["missing_gradient_elements"] == 0)
            )
            if collect_gradient_details:
                sampled = {
                    "step": steps,
                    "per_parameter": {},
                    "missing_gradient_elements": gradient_summary["missing_gradient_elements"],
                    "global_zero_fraction": gradient_summary["global_zero_fraction"],
                    "nonfinite_count": gradient_summary["nonfinite_count"],
                    "l2": gradient_summary["l2"],
                    "max_abs": gradient_summary["max_abs"],
                }
                for name, value in gradient_summary["per_parameter"].items():
                    sampled["per_parameter"][name] = value
                sampled_gradients.append(sampled)
            optimizer.step()
            if device.type == "cuda" and measure:
                total_end.record()
                cuda_events.append((total_start, total_end))
            elif measure:
                host_steps.append(time.perf_counter() - step_started)
            if measure:
                measured_steps += 1
        if not len(source_loader):
            raise RuntimeError("P30R1 cached source loader is empty")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        step_ms = _event_elapsed_ms(cuda_events)
        forward_ms = _event_elapsed_ms(cuda_forward_events)
        objective_ms = _event_elapsed_ms(cuda_objective_events)
    else:
        step_ms = host_steps
        forward_ms = host_forward
        objective_ms = host_objective
    if len(step_ms) != measured_steps:
        raise RuntimeError(f"timing sample mismatch: {len(step_ms)} != {measured_steps}")
    finite_loss = bool(torch.stack(finite_loss_flags).all().detach().cpu())
    finite_gradient = bool(torch.stack(finite_gradient_flags).all().detach().cpu())
    if last_loss is None:
        raise RuntimeError("P30R1 cached training produced no loss")
    final_loss = float(last_loss.cpu())
    exact_zero_teacher_count = int(torch.stack(exact_zero_teacher_flags).sum().detach().cpu())
    gradient_health = _summarize_samples(sampled_gradients)
    if (
        not finite_loss
        or not finite_gradient
        or not teacher_scale_detached
        or gradient_health.get("missing_gradient_elements_max", 0) != 0
    ):
        raise FloatingPointError("P30R1 engineering path produced non-finite values or a trainable teacher scale")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "p30r1_region_adapter.pt"
    completion_path = args.output / "P30R1_TRAINING_COMPLETE.json"
    if checkpoint_path.exists() or completion_path.exists():
        raise RuntimeError("P30R1 training output already exists; refusing overwrite")
    state_dict = {name: parameter.detach().cpu() for name, parameter in adapter.named_parameters()}
    payload = {
        "schema_version": "P30R1_REGION_ADAPTER_CHECKPOINT_V1",
        "status": "ENGINEERING_QUALIFICATION_ONLY",
        "held_class": args.held_class,
        "stage": args.stage,
        "state_dict": state_dict,
        "steps": steps,
        "measured_steps": measured_steps,
        "warmup_steps": args.warmup_steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "objective": P30R1_OBJECTIVE_NAME,
        "objective_count": 1,
        "p30r1_uuid": args.p30r1_uuid,
        "p30r1_preregistration_sha256": p30r1_preregistration_hash(P30R1_PREREGISTRATION_PATH),
        "p30r1_execution_base_sha": args.execution_base_sha,
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
    training_seconds = time.perf_counter() - started
    result = {
        "schema_version": "P30R1_TRAINING_COMPLETE_V1",
        "status": payload["status"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "held_class": args.held_class,
        "stage": args.stage,
        "p30r1_uuid": args.p30r1_uuid,
        "p30r1_preregistration_sha256": p30r1_preregistration_hash(P30R1_PREREGISTRATION_PATH),
        "p30r1_execution_base_sha": args.execution_base_sha,
        "fit_records": len(inventory.fit_rows),
        "held_records_not_read": len(inventory.held_rows),
        "held_GT_read_count": 0,
        "held_mask_read_count": 0,
        "source_mask_loaded": False,
        "native_logits_loaded": False,
        "exact_zero_teacher_records_observed": exact_zero_teacher_count,
        "teacher_scale_detached": teacher_scale_detached,
        "steps": steps,
        "measured_steps": measured_steps,
        "warmup_steps": args.warmup_steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "last_loss": final_loss,
        "loss_finite": finite_loss,
        "gradient_finite": finite_gradient,
        "training_seconds": training_seconds,
        "step_time_ms": {
            "median": float(statistics.median(step_ms)),
            "p90": float(torch.quantile(torch.tensor(step_ms, dtype=torch.float64), 0.90)),
            "mean": float(statistics.fmean(step_ms)),
            "samples": len(step_ms),
            "includes": "cache tensor transfer after DataLoader yield, adapter forward, P30R1 objective, backward, and AdamW step",
            "dataloader_wait_included": False,
        },
        "component_time_ms": {
            "forward_median": float(statistics.median(forward_ms)),
            "objective_median": float(statistics.median(objective_ms)),
            "forward_mean": float(statistics.fmean(forward_ms)),
            "objective_mean": float(statistics.fmean(objective_ms)),
            "objective_fraction_of_step_median": float(statistics.median(objective_ms) / statistics.median(step_ms)),
        },
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "phase2b_optimization_steps": 0,
        "clip_optimization_steps": 0,
        "optimizer_steps": steps,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "teacher_forward_count": 0,
        "teacher_parameter_delta": 0.0,
        "student_parameter_delta": parameter_delta,
        "gradient_health": gradient_health,
    }
    atomic_write_json(completion_path, result)
    return result


def main() -> None:
    import json

    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
