"""Run the one locked P33 candle fit from the immutable source cache."""
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

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.phase2b_runtime import configure_canonical_fp32
from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p29_contract import p29_cache_provenance
from tools.sabra_v2.p33_objective import (
    P33_OBJECTIVE_NAME,
    P33_PREREGISTRATION_SHA256,
    p33_actionability_components,
    p33_objective_contract,
)
from tools.sabra_v2.p29_contract import CORRECTION_SCALE
from tools.sabra_v2.p32_objective import deployed_margin_effect
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


def _weight_summary(weight: torch.Tensor) -> dict[str, Any]:
    values = weight.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    return {
        "n": int(values.numel()),
        "mean": float(values.mean()),
        "q50": float(torch.quantile(values, 0.50)),
        "q90": float(torch.quantile(values, 0.90)),
        "q95": float(torch.quantile(values, 0.95)),
        "q99": float(torch.quantile(values, 0.99)),
        "min": float(values.min()),
        "max": float(values.max()),
        "exact_zero_fraction": float((values == 0).float().mean()),
        "near_zero_fraction_le_1e-6": float((values <= 1e-6).float().mean()),
        "strong_active_fraction_ge_0.5": float((values >= 0.5).float().mean()),
        "finite": bool(torch.isfinite(values).all()),
        "bounded": bool((values >= 0).all() and (values <= 1).all()),
    }


def _source_actionability_summary(cache_root: Path) -> dict[str, Any]:
    """Summarize the locked candle source Tier-B actionability without held access."""
    unique: dict[str, torch.Tensor] = {}
    shard = cache_root / "tier_b" / HELD_CLASS
    manifest_path = shard / "manifest.json"
    teacher_path = shard / "teacher_region.npy"
    if not manifest_path.is_file() or not teacher_path.is_file():
        raise RuntimeError("locked candle Tier-B source cache is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_ids = list(manifest["sample_ids"])
    teacher = np.load(teacher_path, mmap_mode="r", allow_pickle=False)
    if tuple(teacher.shape) != (len(sample_ids), 9, 9):
        raise RuntimeError("locked candle Tier-B teacher shape changed")
    for index, sample_id in enumerate(sample_ids):
        if str(sample_id).split(":", 1)[0] == HELD_CLASS:
            raise RuntimeError("held candle sample reached source-only actionability summary")
        value = torch.from_numpy(np.array(teacher[index], copy=True)).to(dtype=torch.float32)
        if sample_id in unique and not torch.equal(unique[sample_id], value):
            raise RuntimeError("duplicate Tier-B source teacher values disagree")
        unique.setdefault(sample_id, value)
    if not unique:
        raise RuntimeError("frozen Tier-B source cache is empty")
    values = torch.stack([unique[key] for key in sorted(unique)])
    total = 0
    total_sum = 0.0
    minimum = 1.0
    maximum = 0.0
    exact_zero = 0
    near_zero = 0
    strong_active = 0
    finite = True
    bounded = True
    sampled: list[np.ndarray] = []
    total_elements = int(values.shape[0] * 518 * 518)
    sample_stride = max(1, math.ceil(total_elements / 200_000))
    seen = 0
    with torch.no_grad():
        for start in range(0, values.shape[0], 8):
            teacher_chunk = values[start : start + 8]
            teacher_effect = deployed_margin_effect(teacher_chunk)
            weight = (teacher_effect.abs() / CORRECTION_SCALE).clamp(0.0, 1.0)
            flat = weight.cpu().numpy().reshape(-1)
            total += int(flat.size)
            total_sum += float(flat.sum(dtype=np.float64))
            minimum = min(minimum, float(flat.min()))
            maximum = max(maximum, float(flat.max()))
            exact_zero += int(np.count_nonzero(flat == 0.0))
            near_zero += int(np.count_nonzero(flat <= 1e-6))
            strong_active += int(np.count_nonzero(flat >= 0.5))
            finite = finite and bool(np.isfinite(flat).all())
            bounded = bounded and bool(((flat >= 0.0) & (flat <= 1.0)).all())
            first = (-seen) % sample_stride
            sampled.append(flat[first::sample_stride])
            seen += flat.size
    sample = np.concatenate(sampled) if sampled else np.zeros(0, dtype=np.float32)
    if total == 0 or not sample.size:
        raise RuntimeError("source-only actionability summary is empty")
    quantiles = np.quantile(sample, (0.50, 0.90, 0.95, 0.99), method="linear")
    weight_summary = {
        "n": total,
        "mean": total_sum / total,
        "q50": float(quantiles[0]),
        "q90": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "q99": float(quantiles[3]),
        "min": minimum,
        "max": maximum,
        "exact_zero_fraction": exact_zero / total,
        "near_zero_fraction_le_1e-6": near_zero / total,
        "strong_active_fraction_ge_0.5": strong_active / total,
        "finite": finite,
        "bounded": bounded,
        "quantile_method": "deterministic systematic sample",
        "quantile_sample_size": int(sample.size),
        "chunk_size": 8,
    }
    return {
        "source_cache_tier_b_shards": 1,
        "source_exposures": len(sample_ids),
        "source_unique_samples": len(unique),
        "duplicate_exposures_not_counted": len(sample_ids) - len(unique),
        "weight_summary": weight_summary,
        "held_reads": 0,
        "new_neural_forwards": 0,
        "cache_rebuilds": 0,
        "category_specific_rule": False,
    }


def _validate_and_loader(args: argparse.Namespace) -> tuple[DataLoader, Any, dict[str, Any]]:
    if args.held_class != HELD_CLASS:
        raise RuntimeError("P33 scientific training is locked to the candle fold")
    if args.preregistration_sha != P33_PREREGISTRATION_SHA256:
        raise RuntimeError("P33 preregistration hash mismatch")
    if args.cache_root.resolve() != DEFAULT_CACHE_ROOT.resolve():
        raise RuntimeError("P33 scientific training must reuse /workspace/p27r1_cache_v1")
    if not args.metadata.is_file() or not args.cache_root.is_dir():
        raise RuntimeError("P33 metadata or frozen cache root is missing")
    rows = read_visa_metadata(args.metadata)
    if tuple(sorted({str(row["class_name"]) for row in rows})) != tuple(sorted(EXPECTED_VISA_CLASSES)):
        raise RuntimeError("unexpected VisA class inventory")
    inventory = loco_inventory(rows, HELD_CLASS)
    if len(inventory.fit_rows) != EXPECTED_FIT_RECORDS or len(inventory.held_rows) != EXPECTED_HELD_RECORDS:
        raise RuntimeError("P33 candle LOCO inventory changed")
    provenance = p29_cache_provenance(args.metadata)
    dataset = CachedSourceDataset(
        inventory.fit_rows,
        HELD_CLASS,
        args.cache_root,
        provenance,
        load_source_mask=False,
        load_native_logits=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=torch.Generator().manual_seed(SEED),
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


def _checkpoint_payload(adapter: RegionResidualAdapter, args: argparse.Namespace, provenance: Any, steps: int) -> dict[str, Any]:
    return {
        "schema_version": "P33_REGION_ADAPTER_CHECKPOINT_V1",
        "status": "FOLD_TRAINING_COMPLETE",
        "protocol_id": "P33",
        "held_class": HELD_CLASS,
        "attempt_uuid": args.attempt_uuid,
        "scientific_execution_base_sha": args.execution_base_sha,
        "preregistration_sha256": args.preregistration_sha,
        "objective": P33_OBJECTIVE_NAME,
        "objective_count": 1,
        "objective_contract": p33_objective_contract(),
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
        raise RuntimeError(f"P33 training output is non-empty: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    loader, provenance, data_audit = _validate_and_loader(args)
    source_actionability = _source_actionability_summary(args.cache_root)

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
    actionability_samples: list[dict[str, Any]] = []
    nonfinite_loss_count = 0
    nonfinite_gradient_count = 0
    missing_gradient_count = 0
    last_loss: torch.Tensor | None = None
    last_student_effect_shape: list[int] | None = None
    last_teacher_effect_shape: list[int] | None = None
    last_weight_shape: list[int] | None = None
    teacher_detached = True
    weight_detached = True
    steps = 0
    for epoch in range(EPOCHS):
        epoch_steps = 0
        for batch in loader:
            step_started = time.perf_counter()
            expected_fields = {"class_name", "image_path", "sample_id", "index", "seg_features", "teacher_region"}
            if set(batch).intersection({"mask", "native_logits", "label"}):
                raise RuntimeError("forbidden held/source field reached the P33 objective")
            if set(batch) != expected_fields:
                raise RuntimeError(f"unexpected P33 cached batch fields: {sorted(batch)}")
            seg_features = batch["seg_features"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            teacher_region = batch["teacher_region"].to(device=device, dtype=torch.float32)
            if teacher_region.requires_grad:
                raise RuntimeError("cached teacher unexpectedly requires gradients")
            optimizer.zero_grad(set_to_none=True)
            student_region = adapter(seg_features)
            loss, student_effect, teacher_effect, weight = p33_actionability_components(student_region, teacher_region)
            if not bool(torch.isfinite(loss.detach()).item()):
                nonfinite_loss_count += 1
                raise FloatingPointError("P33 objective produced a non-finite loss")
            loss.backward()
            gradient = _finite_gradient_audit(adapter)
            missing_gradient_count += int(gradient["missing_gradient_elements"])
            nonfinite_gradient_count += int(gradient["nonfinite_count"])
            if not gradient["finite"]:
                raise FloatingPointError(f"P33 objective produced an unhealthy gradient: {gradient}")
            optimizer.step()
            steps += 1
            epoch_steps += 1
            step_seconds.append(time.perf_counter() - step_started)
            if steps == 1 or steps % 1000 == 0 or steps == EXPECTED_STEPS:
                gradient_sample = dict(gradient)
                gradient_sample["step"] = steps
                gradient_samples.append(gradient_sample)
                actionability_sample = _weight_summary(weight)
                actionability_sample["step"] = steps
                actionability_samples.append(actionability_sample)
            last_loss = loss.detach()
            last_student_effect_shape = list(student_effect.shape)
            last_teacher_effect_shape = list(teacher_effect.shape)
            last_weight_shape = list(weight.shape)
            teacher_detached = teacher_detached and not teacher_effect.requires_grad
            weight_detached = weight_detached and not weight.requires_grad
        if epoch_steps != EXPECTED_FIT_RECORDS:
            raise RuntimeError(f"P33 epoch {epoch + 1} yielded {epoch_steps} steps")
    if steps != EXPECTED_STEPS:
        raise RuntimeError(f"P33 optimizer step count mismatch: {steps} != {EXPECTED_STEPS}")
    if (
        last_loss is None
        or last_student_effect_shape != [BATCH_SIZE, 518, 518]
        or last_teacher_effect_shape != [BATCH_SIZE, 518, 518]
        or last_weight_shape != [BATCH_SIZE, 518, 518]
    ):
        raise RuntimeError("P33 training did not produce the frozen effect/weight shapes")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    final_loss = float(last_loss.cpu().item())
    student_delta = _parameter_delta(initial_state, adapter)
    if student_delta["l2"] <= 0.0:
        raise RuntimeError("P33 student parameters did not update")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "p33_region_adapter.pt"
    completion_path = args.output / "P33_TRAINING_COMPLETE.json"
    payload = _checkpoint_payload(adapter, args, provenance, steps)
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint_path)
    checkpoint_path.chmod(0o444)
    result: dict[str, Any] = {
        "schema_version": "P33_TRAINING_COMPLETE_V1",
        "status": "FOLD_TRAINING_COMPLETE",
        "protocol_id": "P33",
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
        "objective": P33_OBJECTIVE_NAME,
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
        "weight_detached": weight_detached,
        "teacher_parameter_delta": 0.0,
        "student_parameter_delta": student_delta,
        "training_seconds": training_seconds,
        "step_time_seconds": {
            "median": statistics.median(step_seconds),
            "p90": _quantile(step_seconds, 0.90),
            "mean": statistics.fmean(step_seconds),
            "samples": len(step_seconds),
            "timing_note": "host wall time around cached batch, adapter, P33 objective, backward, and AdamW; CUDA synchronized at completion",
        },
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
        "adapter_forwards": steps,
        "gradient_health_samples": _summarize_gradient_samples(gradient_samples),
        "actionability_weight_samples": actionability_samples,
        "source_only_actionability": source_actionability,
        "data_access_audit": data_audit,
        "cache_provenance": provenance.as_dict(),
        "objective_contract": p33_objective_contract(),
    }
    if not result["loss_finite"] or not result["gradient_finite"] or not teacher_detached or not weight_detached:
        raise FloatingPointError("P33 completed with an invalid finite/gradient/detach audit")
    atomic_write_json(completion_path, result)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
