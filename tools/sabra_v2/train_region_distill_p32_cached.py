"""Run the one locked P32 candle fit from the immutable source cache."""
from __future__ import annotations

import argparse
import json
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
DEFAULT_METADATA = ROOT / "dataset/hub/VisA.jsonl"
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
HELD_CLASS = "candle"
EPOCHS = 20
BATCH_SIZE = 1
LEARNING_RATE = 0.001
BETAS = (0.9, 0.999)
OPTIMIZER_EPSILON = 1e-8
WEIGHT_DECAY = 0.01
AMSGRAD = False
SEED = 0
EXPECTED_FIT_RECORDS = 1962
EXPECTED_HELD_RECORDS = 200
EXPECTED_STEPS = EXPECTED_FIT_RECORDS * EPOCHS


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--held-class", default=HELD_CLASS)
    parser.add_argument("--attempt-uuid", required=True)
    parser.add_argument("--execution-base-sha", required=True)
    parser.add_argument("--preregistration-sha", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _finite_gradient_audit(adapter: RegionResidualAdapter) -> dict[str, Any]:
    missing = 0
    nonfinite = 0
    total = 0
    zero = 0
    squared = 0.0
    maximum = 0.0
    per_parameter: dict[str, Any] = {}
    for name, parameter in adapter.named_parameters():
        total += int(parameter.numel())
        if parameter.grad is None:
            missing += int(parameter.numel())
            per_parameter[name] = {
                "norm": 0.0,
                "max_abs": 0.0,
                "zero_fraction": 1.0,
                "nonfinite_count": int(parameter.numel()),
            }
            continue
        gradient = parameter.grad.detach()
        finite = torch.isfinite(gradient)
        current_nonfinite = int((~finite).sum().item())
        nonfinite += current_nonfinite
        zero += int((gradient == 0).sum().item())
        finite_gradient = gradient[finite]
        norm = float(torch.linalg.vector_norm(finite_gradient).item()) if finite_gradient.numel() else math.nan
        max_abs = float(finite_gradient.abs().max().item()) if finite_gradient.numel() else math.nan
        squared += norm * norm
        maximum = max(maximum, max_abs) if math.isfinite(max_abs) else math.nan
        per_parameter[name] = {
            "norm": norm,
            "max_abs": max_abs,
            "zero_fraction": float((gradient == 0).float().mean().item()),
            "nonfinite_count": current_nonfinite,
        }
    return {
        "missing_gradient_elements": missing,
        "nonfinite_count": nonfinite,
        "l2": math.sqrt(squared) if math.isfinite(squared) else math.nan,
        "max_abs": maximum,
        "global_zero_fraction": zero / total if total else 1.0,
        "finite": missing == 0 and nonfinite == 0 and math.isfinite(squared) and math.isfinite(maximum),
        "per_parameter": per_parameter,
    }


def _parameter_delta(initial: dict[str, torch.Tensor], adapter: RegionResidualAdapter) -> dict[str, float]:
    squared = 0.0
    maximum = 0.0
    for name, parameter in adapter.named_parameters():
        difference = parameter.detach().cpu() - initial[name]
        squared += float(difference.square().sum().item())
        maximum = max(maximum, float(difference.abs().max().item()))
    return {"l2": math.sqrt(squared), "max_abs": maximum}


def _summarize_gradient_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"sample_count": 0, "sampled_steps": []}
    names = tuple(samples[0]["per_parameter"])
    per_parameter: dict[str, Any] = {}
    for name in names:
        values = [sample["per_parameter"][name] for sample in samples]
        per_parameter[name] = {
            "norm_min": min(float(value["norm"]) for value in values),
            "norm_median": statistics.median(float(value["norm"]) for value in values),
            "norm_max": max(float(value["norm"]) for value in values),
            "max_abs_max": max(float(value["max_abs"]) for value in values),
            "zero_fraction_mean": statistics.fmean(float(value["zero_fraction"]) for value in values),
            "nonfinite_count_max": max(int(value["nonfinite_count"]) for value in values),
        }
    return {
        "sample_count": len(samples),
        "sampled_steps": [int(sample["step"]) for sample in samples],
        "global_zero_fraction_mean": statistics.fmean(float(sample["global_zero_fraction"]) for sample in samples),
        "l2_min": min(float(sample["l2"]) for sample in samples),
        "l2_max": max(float(sample["l2"]) for sample in samples),
        "max_abs_max": max(float(sample["max_abs"]) for sample in samples),
        "per_parameter": per_parameter,
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_and_loader(args: argparse.Namespace) -> tuple[DataLoader, Any, dict[str, Any]]:
    if args.held_class != HELD_CLASS:
        raise RuntimeError("P32 scientific training is locked to the candle fold")
    if args.preregistration_sha != P32_PREREGISTRATION_SHA256:
        raise RuntimeError("P32 preregistration hash mismatch")
    if args.cache_root.resolve() != DEFAULT_CACHE_ROOT.resolve():
        raise RuntimeError("P32 scientific training must reuse /workspace/p27r1_cache_v1")
    if not args.metadata.is_file() or not args.cache_root.is_dir():
        raise RuntimeError("P32 metadata or frozen cache root is missing")
    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, HELD_CLASS)
    if len(inventory.fit_rows) != EXPECTED_FIT_RECORDS or len(inventory.held_rows) != EXPECTED_HELD_RECORDS:
        raise RuntimeError("P32 candle LOCO inventory changed")
    provenance = p29_cache_provenance(args.metadata)
    dataset = CachedSourceDataset(
        inventory.fit_rows,
        HELD_CLASS,
        args.cache_root,
        provenance,
        load_source_mask=False,
        load_native_logits=False,
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )
    return loader, provenance, {
        "metadata": str(args.metadata),
        "metadata_sha256": sha256_file(args.metadata),
        "cache_root": str(args.cache_root.resolve()),
        "cache_provenance": provenance.as_dict(),
        "held_class": HELD_CLASS,
        "fit_records": EXPECTED_FIT_RECORDS,
        "held_records_not_read": EXPECTED_HELD_RECORDS,
        "source_mask_loaded": False,
        "native_logits_loaded": False,
        "dataset_length": len(dataset),
    }


def _checkpoint_payload(
    adapter: RegionResidualAdapter,
    *,
    args: argparse.Namespace,
    provenance: Any,
    steps: int,
) -> dict[str, Any]:
    return {
        "schema_version": "P32_REGION_ADAPTER_CHECKPOINT_V1",
        "status": "FOLD_TRAINING_COMPLETE",
        "protocol_id": "P32",
        "held_class": HELD_CLASS,
        "attempt_uuid": args.attempt_uuid,
        "scientific_execution_base_sha": args.execution_base_sha,
        "preregistration_sha256": args.preregistration_sha,
        "objective": P32_OBJECTIVE_NAME,
        "objective_count": 1,
        "objective_contract": p32_objective_contract(),
        "state_dict": {name: value.detach().cpu() for name, value in adapter.named_parameters()},
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "betas": list(BETAS),
            "epsilon": OPTIMIZER_EPSILON,
            "weight_decay": WEIGHT_DECAY,
            "amsgrad": AMSGRAD,
        },
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "optimizer_steps": steps,
        "cache_provenance": provenance.as_dict(),
        "teacher_trainable": False,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
        "source_mask_used_by_loss": False,
        "scientific_execution_marker": None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"P32 training output is non-empty: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    loader, provenance, data_audit = _validate_and_loader(args)

    configure_canonical_fp32()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    if any(not parameter.requires_grad for parameter in adapter.parameters()):
        raise RuntimeError("all RegionResidualAdapter parameters must remain trainable")
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        eps=OPTIMIZER_EPSILON,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )
    initial_state = {name: parameter.detach().cpu().clone() for name, parameter in adapter.named_parameters()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    step_seconds: list[float] = []
    gradient_samples: list[dict[str, Any]] = []
    nonfinite_loss_count = 0
    nonfinite_gradient_count = 0
    missing_gradient_count = 0
    last_loss: torch.Tensor | None = None
    last_student_effect_shape: list[int] | None = None
    last_teacher_effect_shape: list[int] | None = None
    teacher_detached = True
    steps = 0
    for _epoch in range(EPOCHS):
        epoch_steps = 0
        for batch in loader:
            step_started = time.perf_counter()
            if set(batch).intersection({"mask", "native_logits", "label"}):
                raise RuntimeError("forbidden held/source field reached the P32 objective")
            if set(batch) != {"class_name", "image_path", "sample_id", "index", "seg_features", "teacher_region"}:
                raise RuntimeError(f"unexpected P32 cached batch fields: {sorted(batch)}")
            seg_features = batch["seg_features"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            teacher_region = batch["teacher_region"].to(device=device, dtype=torch.float32)
            if teacher_region.requires_grad:
                raise RuntimeError("cached teacher unexpectedly requires gradients")
            optimizer.zero_grad(set_to_none=True)
            student_region = adapter(seg_features)
            loss, student_effect, teacher_effect = p32_functional_margin_components(student_region, teacher_region)
            loss_finite = bool(torch.isfinite(loss.detach()).item())
            if not loss_finite:
                nonfinite_loss_count += 1
                raise FloatingPointError("P32 objective produced a non-finite loss")
            loss.backward()
            gradient = _finite_gradient_audit(adapter)
            missing_gradient_count += int(gradient["missing_gradient_elements"])
            nonfinite_gradient_count += int(gradient["nonfinite_count"])
            if not gradient["finite"]:
                raise FloatingPointError(f"P32 objective produced an unhealthy gradient: {gradient}")
            optimizer.step()
            steps += 1
            epoch_steps += 1
            step_seconds.append(time.perf_counter() - step_started)
            if steps == 1 or steps % 1000 == 0 or steps == EXPECTED_STEPS:
                sample = dict(gradient)
                sample["step"] = steps
                gradient_samples.append(sample)
            last_loss = loss.detach()
            last_student_effect_shape = list(student_effect.shape)
            last_teacher_effect_shape = list(teacher_effect.shape)
            teacher_detached = teacher_detached and not teacher_effect.requires_grad
        if epoch_steps != EXPECTED_FIT_RECORDS:
            raise RuntimeError(f"P32 epoch {steps // EXPECTED_FIT_RECORDS + 1} yielded {epoch_steps} steps")
    if steps != EXPECTED_STEPS:
        raise RuntimeError(f"P32 optimizer step count mismatch: {steps} != {EXPECTED_STEPS}")
    if last_loss is None or last_student_effect_shape != [BATCH_SIZE, 518, 518] or last_teacher_effect_shape != [BATCH_SIZE, 518, 518]:
        raise RuntimeError("P32 training did not produce the frozen effect shapes")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    final_loss = float(last_loss.cpu().item())
    student_delta = _parameter_delta(initial_state, adapter)
    if student_delta["l2"] <= 0.0:
        raise RuntimeError("P32 student parameters did not update")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "p32_region_adapter.pt"
    completion_path = args.output / "P32_TRAINING_COMPLETE.json"
    payload = _checkpoint_payload(adapter, args=args, provenance=provenance, steps=steps)
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    checkpoint_path.chmod(0o444)
    result: dict[str, Any] = {
        "schema_version": "P32_TRAINING_COMPLETE_V1",
        "status": "FOLD_TRAINING_COMPLETE",
        "protocol_id": "P32",
        "held_class": HELD_CLASS,
        "attempt_uuid": args.attempt_uuid,
        "scientific_execution_base_sha": args.execution_base_sha,
        "preregistration_sha256": args.preregistration_sha,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "fit_records": EXPECTED_FIT_RECORDS,
        "held_records_not_read": EXPECTED_HELD_RECORDS,
        "held_gt_reads": 0,
        "held_mask_reads": 0,
        "source_mask_loaded": False,
        "native_logits_loaded": False,
        "cache_rebuilt": False,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "betas": list(BETAS),
        "optimizer_epsilon": OPTIMIZER_EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "amsgrad": AMSGRAD,
        "seed": SEED,
        "objective": P32_OBJECTIVE_NAME,
        "objective_count": 1,
        "optimizer_steps": steps,
        "expected_optimizer_steps": EXPECTED_STEPS,
        "last_loss": final_loss,
        "loss_finite": nonfinite_loss_count == 0 and math.isfinite(final_loss),
        "gradient_finite": nonfinite_gradient_count == 0 and missing_gradient_count == 0,
        "nonfinite_loss_count": nonfinite_loss_count,
        "nonfinite_gradient_count": nonfinite_gradient_count,
        "missing_gradient_elements_total": missing_gradient_count,
        "teacher_detached": teacher_detached,
        "teacher_parameter_delta": 0.0,
        "student_parameter_delta": student_delta,
        "training_seconds": training_seconds,
        "step_time_seconds": {
            "median": statistics.median(step_seconds),
            "p90": _quantile(step_seconds, 0.90),
            "mean": statistics.fmean(step_seconds),
            "samples": len(step_seconds),
            "timing_note": "host wall time around cached batch, adapter, objective, backward, and AdamW; CUDA synchronized at completion",
        },
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
        "adapter_forwards": steps,
        "gradient_health_samples": _summarize_gradient_samples(gradient_samples),
        "data_access_audit": data_audit,
        "cache_provenance": provenance.as_dict(),
        "objective_contract": p32_objective_contract(),
    }
    if not result["loss_finite"] or not result["gradient_finite"] or not teacher_detached:
        raise FloatingPointError("P32 completed with an invalid finite/gradient/detach audit")
    atomic_write_json(completion_path, result)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
