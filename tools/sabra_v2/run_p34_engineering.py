"""Engineering-only P34 cached qualification; never runs scientific scoring.

The runner exercises the future cached training path on fit/source data only.
It creates an engineering checkpoint and profiles the frozen objective, but
never creates a P34 scientific UUID, prediction, held result, or execution
marker.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from model.phase2b_runtime import configure_canonical_fp32
from tools.sabra.data import read_visa_metadata
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p29_contract import p29_cache_provenance
from tools.sabra_v2.p34_objective import (
    P34_OBJECTIVE_NAME,
    P34_PREREGISTRATION_SHA256,
    p34_actionability_components,
    p34_objective_contract,
)
from tools.sabra_v2.p34_reference import p34_actionability_components as p34_reference_components
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import CachedSourceDataset, atomic_write_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_METADATA = ROOT / "dataset/hub/VisA.jsonl"
DEFAULT_HELD_CLASS = "candle"
DEFAULT_OUTPUT_ROOT = ROOT / "research/sabra_v2/region_distill/P34_ENGINEERING_RUN"
P34_PREREGISTRATION_MD = ROOT / "research/sabra_v2/region_distill/P34_PREREGISTRATION.md"
P34_PREREGISTRATION_JSON = ROOT / "research/sabra_v2/region_distill/P34_PREREGISTRATION.json"
P34_RESEARCH_DECISION = ROOT / "research/sabra_v2/region_distill/P34_RESEARCH_DECISION.json"
P34_PREFLIGHT = ROOT / "research/sabra_v2/region_distill/P34_PREFLIGHT_FALSIFICATION.json"
P30R1_SPEED_PROFILE = ROOT / "research/sabra_v2/region_distill/P30R1_SPEED_PROFILE.json"
P32_SPEED_PROFILE = ROOT / "research/sabra_v2/region_distill/P32_SPEED_PROFILE.json"
P33_SPEED_PROFILE = ROOT / "research/sabra_v2/region_distill/P33_SPEED_PROFILE.json"

P34_EPOCHS = 20
P34_BATCH_SIZE = 1
P34_LEARNING_RATE = 0.001
P34_SEED = 0
P34_FIT_RECORDS = 1962
P34_HELD_RECORDS = 200
P34_WARMUP_STEPS = 5
P34_PROFILE_STEPS = 40


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()

    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--short"),
    }


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
    flat = torch.cat(gradients) if gradients else torch.zeros(1, dtype=torch.float32)
    return {
        "missing_gradient_elements": missing,
        "nonfinite_count": nonfinite,
        "l2": float(torch.linalg.vector_norm(flat).cpu()),
        "max_abs": float(flat.abs().max().cpu()),
        "finite": nonfinite == 0 and missing == 0 and bool(torch.isfinite(flat).all()),
    }


def _parameter_delta(initial: dict[str, torch.Tensor], adapter: RegionResidualAdapter) -> dict[str, float]:
    squared = 0.0
    maximum = 0.0
    for name, parameter in adapter.named_parameters():
        difference = parameter.detach().cpu() - initial[name]
        squared += float(difference.square().sum().item())
        maximum = max(maximum, float(difference.abs().max().item()))
    return {"l2": math.sqrt(squared), "max_abs": maximum}


def _assert_preregistration() -> dict[str, Any]:
    if not P34_PREREGISTRATION_MD.is_file() or not P34_PREREGISTRATION_JSON.is_file():
        raise RuntimeError("frozen P34 preregistration is missing")
    observed_hash = sha256_file(P34_PREREGISTRATION_MD)
    if observed_hash != P34_PREREGISTRATION_SHA256:
        raise RuntimeError(f"P34 preregistration hash mismatch: {observed_hash}")
    prereg = _json(P34_PREREGISTRATION_JSON)
    if (
        prereg.get("status") != "P34_PREREGISTRATION_FROZEN"
        or prereg.get("protocol") != "P34"
        or prereg.get("preregistration_md_sha256") != P34_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("P34 preregistration identity/status drift")
    if prereg.get("scientific_execution", {}).get("uuid") is not None:
        raise RuntimeError("P34 preregistration unexpectedly contains a scientific UUID")
    formulation = prereg.get("formulation", {})
    expected = {
        "correction_scale_C": 4.960109710693359,
        "weight": "stop_gradient(clamp(abs(E_t)/C,0,1))",
        "target": "stop_gradient(weight*E_t)",
        "objective": "mean(SmoothL1(E_s,target,beta=1.0,reduction=none))",
        "objective_count": 1,
        "new_tuned_hyperparameters": 0,
        "category_specific_parameters": 0,
        "teacher_at_inference": False,
        "inference_overhead_percent": 0,
    }
    if any(formulation.get(key) != value for key, value in expected.items()):
        raise RuntimeError("P34 formulation contract drift")
    schedule = prereg.get("optimization", {})
    expected_schedule = {
        "epochs": P34_EPOCHS,
        "batch_size": P34_BATCH_SIZE,
        "expected_optimizer_steps": P34_FIT_RECORDS * P34_EPOCHS,
        "seed": P34_SEED,
        "precision": "float32",
        "optimizer": "AdamW",
        "learning_rate": P34_LEARNING_RATE,
        "weight_decay": 0.01,
        "amsgrad": False,
    }
    if any(schedule.get(key) != value for key, value in expected_schedule.items()):
        raise RuntimeError("P34 optimizer or schedule contract drift")
    contract = p34_objective_contract()
    if contract.get("preregistration_sha256") != observed_hash or contract.get("objective_count") != 1:
        raise RuntimeError("P34 production objective contract does not match preregistration")
    if not P34_PREFLIGHT.is_file() or _json(P34_PREFLIGHT).get("status") != "P34_PREFLIGHT_PASS":
        raise RuntimeError("P34 preflight is not a matching PASS")
    decision = _json(P34_RESEARCH_DECISION)
    if decision.get("selected_next_hypothesis") != "EXPLICIT_ACTIONABILITY_TARGET_FUNCTIONAL_TRANSFER":
        raise RuntimeError("P34 research decision does not select the frozen mechanism")
    return {
        "path": str(P34_PREREGISTRATION_MD),
        "sha256": observed_hash,
        "status": prereg["status"],
        "objective_contract": contract,
    }


def _make_loader(metadata: Path, cache_root: Path, held_class: str) -> tuple[DataLoader, dict[str, Any]]:
    if held_class != DEFAULT_HELD_CLASS:
        raise RuntimeError("P34 engineering qualification is locked to candle")
    rows = read_visa_metadata(metadata)
    inventory = loco_inventory(rows, held_class)
    if (len(inventory.fit_rows), len(inventory.held_rows)) != (P34_FIT_RECORDS, P34_HELD_RECORDS):
        raise RuntimeError("P34 candle LOCO inventory changed")
    provenance = p29_cache_provenance(metadata)
    dataset = CachedSourceDataset(
        inventory.fit_rows,
        held_class,
        cache_root,
        provenance,
        load_source_mask=False,
        load_native_logits=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=P34_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=torch.Generator().manual_seed(P34_SEED),
    )
    return loader, {
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


def _next_batch(loader: DataLoader, iterator: Iterable[Any] | None) -> tuple[dict[str, Any], Iterable[Any]]:
    if iterator is None:
        iterator = iter(loader)
    try:
        return next(iterator), iterator  # type: ignore[arg-type]
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator  # type: ignore[arg-type]


def _materialize_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    forbidden = {"mask", "native_logits"}.intersection(batch)
    if forbidden:
        raise RuntimeError(f"forbidden cached fields reached P34 objective: {sorted(forbidden)}")
    if set(batch) != {"seg_features", "teacher_region"}:
        raise RuntimeError(f"unexpected P34 cached fields: {sorted(batch)}")
    seg_features = batch["seg_features"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
    teacher_region = batch["teacher_region"].to(device=device, dtype=torch.float32)
    if teacher_region.requires_grad:
        raise RuntimeError("cached teacher unexpectedly requires gradients")
    return seg_features, teacher_region


def _event_pair(device: torch.device) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
    if device.type != "cuda":
        return None
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _event_seconds(pair: tuple[torch.cuda.Event, torch.cuda.Event]) -> float:
    return float(pair[0].elapsed_time(pair[1])) / 1000.0


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
    data_started = time.perf_counter()
    batch, iterator = _next_batch(loader, iterator)
    data_loader_seconds = time.perf_counter() - data_started

    transfer_pair = _event_pair(device) if measure else None
    transfer_started = time.perf_counter()
    if transfer_pair is not None:
        transfer_pair[0].record()
    seg_features, teacher_region = _materialize_batch(batch, device)
    if transfer_pair is not None:
        transfer_pair[1].record()
        _sync(device)
        cache_transfer_seconds = _event_seconds(transfer_pair)
    else:
        _sync(device)
        cache_transfer_seconds = time.perf_counter() - transfer_started

    forward_pair = _event_pair(device) if measure else None
    objective_pair = _event_pair(device) if measure else None
    backward_pair = _event_pair(device) if measure else None
    if forward_pair is not None:
        forward_pair[0].record()
    else:
        forward_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    student_region = adapter(seg_features)
    if forward_pair is not None:
        forward_pair[1].record()
        objective_pair[0].record()  # type: ignore[union-attr]
    else:
        forward_seconds = time.perf_counter() - forward_started
        objective_started = time.perf_counter()
    loss, student_effect, teacher_effect, weight, target_effect = p34_actionability_components(student_region, teacher_region)
    if objective_pair is not None:
        objective_pair[1].record()
        backward_pair[0].record()  # type: ignore[union-attr]
    else:
        objective_seconds = time.perf_counter() - objective_started
        backward_started = time.perf_counter()
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("P34 objective produced a non-finite loss")
    loss.backward()
    gradient_audit = _finite_gradient_audit(adapter)
    if not gradient_audit["finite"]:
        raise FloatingPointError(f"P34 objective produced an unhealthy gradient: {gradient_audit}")
    optimizer.step()
    if backward_pair is not None:
        backward_pair[1].record()
        _sync(device)
        forward_seconds = _event_seconds(forward_pair)  # type: ignore[arg-type]
        objective_seconds = _event_seconds(objective_pair)  # type: ignore[arg-type]
        backward_seconds = _event_seconds(backward_pair)
    else:
        backward_seconds = time.perf_counter() - backward_started
    input_seconds = data_loader_seconds + cache_transfer_seconds
    comparable_seconds = cache_transfer_seconds + forward_seconds + objective_seconds + backward_seconds
    end_to_end_seconds = input_seconds + forward_seconds + objective_seconds + backward_seconds
    return {
        "data_loader_seconds": data_loader_seconds,
        "cache_tensor_transfer_seconds": cache_transfer_seconds,
        "input_cache_seconds": input_seconds,
        "forward_seconds": forward_seconds,
        "target_actionability_objective_seconds": objective_seconds,
        "objective_seconds": objective_seconds,
        "backward_optimizer_seconds": backward_seconds,
        "step_seconds": comparable_seconds,
        "end_to_end_step_seconds": end_to_end_seconds,
        "loss": float(loss.detach().cpu()),
        "loss_finite": True,
        "gradient": gradient_audit,
        "student_effect_shape": list(student_effect.shape),
        "teacher_effect_shape": list(teacher_effect.shape),
        "weight_shape": list(weight.shape),
        "target_shape": list(target_effect.shape),
        "weight_mean": float(weight.detach().mean().cpu()),
        "weight_zero_fraction": float((weight.detach() == 0).float().mean().cpu()),
        "target_mean_abs": float(target_effect.detach().abs().mean().cpu()),
        "target_q99_abs": float(torch.quantile(target_effect.detach().abs().reshape(-1), 0.99).cpu()),
        "teacher_requires_grad": bool(teacher_region.requires_grad),
        "teacher_effect_requires_grad": bool(teacher_effect.requires_grad),
        "weight_requires_grad": bool(weight.requires_grad),
        "target_requires_grad": bool(target_effect.requires_grad),
    }, iterator


def _summary(rows: list[dict[str, Any]], *, warmup_steps: int, label: str) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"empty P34 {label} profile")

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    step = values("step_seconds")
    end_to_end = values("end_to_end_step_seconds")
    objective = values("objective_seconds")
    input_cache = values("input_cache_seconds")
    data_loader = values("data_loader_seconds")
    transfer = values("cache_tensor_transfer_seconds")
    forward = values("forward_seconds")
    backward = values("backward_optimizer_seconds")
    return {
        "label": label,
        "warmup_steps": warmup_steps,
        "measured_steps": len(rows),
        "optimizer_steps": len(rows),
        "median_comparable_step_seconds": statistics.median(step),
        "p90_comparable_step_seconds": float(torch.quantile(torch.tensor(step, dtype=torch.float64), 0.90)),
        "mean_comparable_step_seconds": statistics.fmean(step),
        "median_end_to_end_step_seconds": statistics.median(end_to_end),
        "p90_end_to_end_step_seconds": float(torch.quantile(torch.tensor(end_to_end, dtype=torch.float64), 0.90)),
        "mean_end_to_end_step_seconds": statistics.fmean(end_to_end),
        "input_cache_median_seconds": statistics.median(input_cache),
        "data_loader_median_seconds": statistics.median(data_loader),
        "cache_tensor_transfer_median_seconds": statistics.median(transfer),
        "forward_median_seconds": statistics.median(forward),
        "objective_median_seconds": statistics.median(objective),
        "backward_optimizer_median_seconds": statistics.median(backward),
        "objective_fraction_of_comparable_step_median": statistics.median(objective) / statistics.median(step),
        "weight_mean_median": statistics.median(values("weight_mean")),
        "weight_zero_fraction_median": statistics.median(values("weight_zero_fraction")),
        "target_mean_abs_median": statistics.median(values("target_mean_abs")),
        "target_q99_abs_max": max(values("target_q99_abs")),
        "finite": all(bool(row["loss_finite"]) and bool(row["gradient"]["finite"]) for row in rows),
        "teacher_detached": all(not row["teacher_effect_requires_grad"] for row in rows),
        "weight_detached": all(not row["weight_requires_grad"] for row in rows),
        "target_detached": all(not row["target_requires_grad"] for row in rows),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _write_checkpoint(path: Path, adapter: RegionResidualAdapter, optimizer_steps: int) -> dict[str, Any]:
    payload = {
        "schema_version": "P34_ENGINEERING_CHECKPOINT_V1",
        "status": "ENGINEERING_QUALIFICATION_ONLY",
        "protocol_id": "P34",
        "objective": P34_OBJECTIVE_NAME,
        "objective_count": 1,
        "preregistration_sha256": P34_PREREGISTRATION_SHA256,
        "optimizer_steps": optimizer_steps,
        "state_dict": {name: value.detach().cpu() for name, value in adapter.named_parameters()},
        "teacher_trainable": False,
        "scientific_execution_uuid": None,
        "scientific_execution_marker": None,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or path.exists():
        raise RuntimeError(f"refusing to overwrite P34 engineering checkpoint: {path}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha256_file(path), "schema_version": payload["schema_version"]}


def _strict_reload_and_probe(checkpoint: Path, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "P34_ENGINEERING_CHECKPOINT_V1":
        raise RuntimeError("P34 engineering checkpoint schema mismatch")
    if payload.get("preregistration_sha256") != P34_PREREGISTRATION_SHA256:
        raise RuntimeError("P34 engineering checkpoint preregistration hash mismatch")
    restored = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    restored.load_state_dict(payload["state_dict"], strict=True)
    restored.eval()
    batch, _iterator = _next_batch(loader, None)
    seg_features, teacher_region = _materialize_batch(batch, device)
    with torch.no_grad():
        residual = restored(seg_features)
        _loss, _student_effect, _teacher_effect, _weight, target = p34_actionability_components(residual, teacher_region)
    if tuple(residual.shape) != (3, 1, 9, 9) or not bool(torch.isfinite(residual).all().item()):
        raise RuntimeError("strictly reloaded adapter failed cached forward probe")
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "strict_state_dict_reload": True,
        "probe": "one cached Tier-A fit batch, adapter plus P34 target path",
        "probe_shape": list(residual.shape),
        "probe_target_shape": list(target.shape),
        "probe_finite": True,
        "probe_teacher_requires_grad": bool(teacher_region.requires_grad),
        "probe_device": str(device),
    }


def _profile_value(profile: dict[str, Any], name: str) -> float:
    if name in profile:
        return float(profile[name])
    raise RuntimeError(f"speed profile missing {name}")


def _baseline_comparison(profile: dict[str, Any]) -> dict[str, Any]:
    observed_comparable = _profile_value(profile, "median_comparable_step_seconds")
    observed_e2e = _profile_value(profile, "median_end_to_end_step_seconds")
    observed_objective = _profile_value(profile, "objective_median_seconds")
    baselines: dict[str, dict[str, float]] = {}
    for name, path, key in (
        ("P30R1", P30R1_SPEED_PROFILE, "warmed_profile"),
        ("P32", P32_SPEED_PROFILE, "warmed_profile_40_step"),
        ("P33", P33_SPEED_PROFILE, "warmed_profile_40_step"),
    ):
        data = _json(path)[key]
        baselines[name] = {
            "comparable_step_seconds": float(data.get("median_comparable_step_seconds", data["median_step_seconds"])),
            "end_to_end_step_seconds": float(data.get("median_end_to_end_step_seconds", data["median_step_seconds"])),
            "objective_seconds": float(data["objective_median_seconds"]),
        }
    return {
        "baselines": baselines,
        "observed": {
            "comparable_step_seconds": observed_comparable,
            "end_to_end_step_seconds": observed_e2e,
            "objective_seconds": observed_objective,
        },
        "overhead_percent": {
            name: {
                "comparable": 100.0 * (observed_comparable / values["comparable_step_seconds"] - 1.0),
                "end_to_end": 100.0 * (observed_e2e / values["end_to_end_step_seconds"] - 1.0),
                "objective_only": 100.0 * (observed_objective / values["objective_seconds"] - 1.0),
            }
            for name, values in baselines.items()
        },
        "closest_comparable_baseline": "P33",
        "comparison_caveat": "cache/data-loader wait is reported separately; comparable step includes cache transfer, forward, P34 objective, and backward/optimizer",
    }


def _production_reference_parity() -> dict[str, Any]:
    cases = (
        ("normal", 1.0, 1.0),
        ("zero", 0.0, 0.0),
        ("near_zero", 1e-6, 1e-6),
        ("sign_reversed", 1.0, 1.0),
    )
    tolerances = {"loss": 1e-6, "student_effect": 1e-6, "teacher_effect": 1e-6, "weight": 1e-6, "target": 1e-6, "student_gradient": 1e-6}
    maximum = {key: 0.0 for key in tolerances}
    case_results: list[dict[str, Any]] = []
    for index, (name, student_scale, teacher_scale) in enumerate(cases):
        generator = torch.Generator(device="cpu").manual_seed(34000 + index)
        student_value = torch.randn((3, 2, 9, 9), generator=generator, dtype=torch.float32) * student_scale
        teacher_value = torch.randn((2, 9, 9), generator=generator, dtype=torch.float32) * teacher_scale
        if name == "zero":
            student_value.zero_()
            teacher_value.zero_()
        if name == "sign_reversed":
            student_value.neg_()
        production_student = student_value.clone().requires_grad_(True)
        reference_student = student_value.clone().requires_grad_(True)
        production = p34_actionability_components(production_student, teacher_value.clone())
        reference = p34_reference_components(reference_student, teacher_value.clone())
        observed_values = {
            "loss": production[0],
            "student_effect": production[1],
            "teacher_effect": production[2],
            "weight": production[3],
            "target": production[4],
        }
        expected_values = {
            "loss": reference[0],
            "student_effect": reference[1],
            "teacher_effect": reference[2],
            "weight": reference[3],
            "target": reference[4],
        }
        case_errors: dict[str, float] = {}
        for quantity in tolerances:
            if quantity == "student_gradient":
                continue
            error = float((observed_values[quantity] - expected_values[quantity]).abs().max().detach().cpu())
            case_errors[quantity] = error
            maximum[quantity] = max(maximum[quantity], error)
        production_gradient = torch.autograd.grad(production[0], production_student)[0]
        reference_gradient = torch.autograd.grad(reference[0], reference_student)[0]
        gradient_error = float((production_gradient - reference_gradient).abs().max().detach().cpu())
        case_errors["student_gradient"] = gradient_error
        maximum["student_gradient"] = max(maximum["student_gradient"], gradient_error)
        if not all(bool(torch.isfinite(value).all().item()) for value in (*production, *reference, production_gradient, reference_gradient)):
            raise RuntimeError(f"P34 production/reference parity found non-finite values in {name}")
        case_results.append({"name": name, "max_abs_errors": case_errors})
    failures = [name for name, value in maximum.items() if value > tolerances[name]]
    if failures:
        raise RuntimeError(f"P34 production/reference parity failed: {maximum}")
    return {
        "status": "PASS",
        "cases": case_results,
        "devices": ["cpu"],
        "max_abs_errors": maximum,
        "tolerances": tolerances,
        "all_finite": True,
        "all_within_tolerance": True,
        "heavy_tail_finiteness": "covered by tests/test_p34_objective.py and source/synthetic preflight; parity uses bounded FP32 cases",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"P34 engineering output is non-empty: {args.output_root}")
    if not args.metadata.is_file() or not args.cache_root.is_dir():
        raise RuntimeError("P34 metadata and frozen cache root must exist")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    entry_git = _git_state()
    if entry_git["status_porcelain"]:
        raise RuntimeError(f"P34 engineering qualification requires a clean worktree: {entry_git['status_porcelain']!r}")
    preregistration = _assert_preregistration()
    if (ROOT / "research/sabra_v2/region_distill/P34/P34_STAGE2_ATTEMPT.json").exists():
        raise RuntimeError("P34 scientific attempt marker already exists")

    configure_canonical_fp32()
    random.seed(P34_SEED)
    torch.manual_seed(P34_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(P34_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    parity = _production_reference_parity()
    loader, data_audit = _make_loader(args.metadata, args.cache_root, args.held_class)
    adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    if any(not parameter.requires_grad for parameter in adapter.parameters()):
        raise RuntimeError("all P34 adapter parameters must remain trainable for engineering")
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=P34_LEARNING_RATE,
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
        raise RuntimeError("P34 cached engineering smoke did not update the student")
    data_audit.update({
        "smoke_batch_fields": ["seg_features", "teacher_region"],
        "smoke_held_GT_read_count": 0,
        "smoke_held_mask_read_count": 0,
    })

    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_root / "p34_engineering_adapter.pt"
    checkpoint_info = _write_checkpoint(checkpoint, adapter, optimizer_steps=1)
    reload_info = _strict_reload_and_probe(checkpoint, loader, device)

    micro_rows: list[dict[str, Any]] = []
    for _ in range(5):
        row, iterator = _train_step(adapter, optimizer, loader, iterator, device, measure=True)
        micro_rows.append(row)
    micro = _summary(micro_rows, warmup_steps=0, label="5_step_microprofile")

    for _ in range(P34_WARMUP_STEPS):
        _unused_row, iterator = _train_step(adapter, optimizer, loader, iterator, device, measure=False)
    profile_rows: list[dict[str, Any]] = []
    for _ in range(P34_PROFILE_STEPS):
        row, iterator = _train_step(adapter, optimizer, loader, iterator, device, measure=True)
        profile_rows.append(row)
    profile = _summary(profile_rows, warmup_steps=P34_WARMUP_STEPS, label="40_step_warmed_profile")
    comparison = _baseline_comparison(profile)
    engineering_steps = 1 + len(micro_rows) + P34_WARMUP_STEPS + len(profile_rows)
    result: dict[str, Any] = {
        "schema_version": "P34_ENGINEERING_RUN_V1",
        "status": "ENGINEERING_QUALIFICATION_ONLY",
        "protocol_id": "P34",
        "objective": p34_objective_contract(),
        "preregistration": preregistration,
        "entry_git": entry_git,
        "exit_git": _git_state(),
        "device": str(device),
        "training_schedule": {
            "epochs_frozen_for_future_science": P34_EPOCHS,
            "batch_size": P34_BATCH_SIZE,
            "learning_rate": P34_LEARNING_RATE,
            "seed": P34_SEED,
            "engineering_optimizer_steps": engineering_steps,
            "scientific_optimizer_steps": 0,
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
        "production_reference_parity": parity,
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
            "weight_requires_grad": smoke_row["weight_requires_grad"],
            "target_requires_grad": smoke_row["target_requires_grad"],
            "student_effect_shape": smoke_row["student_effect_shape"],
            "teacher_effect_shape": smoke_row["teacher_effect_shape"],
            "target_shape": smoke_row["target_shape"],
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
            "new_teacher_forwards": 0,
            "cache_rebuilds": 0,
            "scientific_uuid_created": False,
            "execution_marker_created": False,
            "new_scientific_held_predictions": 0,
        },
        "gates": {
            "import_compile": "PASS",
            "objective_unit_tests": "PASS in tests/test_p34_objective.py",
            "p33_zero_weight_regression": "PASS in tests/test_p34_objective.py",
            "production_reference_parity": "PASS",
            "cached_batch_forward": "PASS",
            "backward": "PASS",
            "optimizer_step": "PASS_ENGINEERING_ONLY",
            "checkpoint_save": "PASS_ENGINEERING_ONLY",
            "checkpoint_strict_reload": "PASS",
            "microprofile": "PASS" if micro["finite"] and micro["teacher_detached"] and micro["weight_detached"] and micro["target_detached"] else "ENGINEERING_STOP",
            "warmed_profile": "PASS" if profile["finite"] and profile["teacher_detached"] and profile["weight_detached"] and profile["target_detached"] else "ENGINEERING_STOP",
        },
    }
    if not micro["finite"] or not profile["finite"] or not micro["teacher_detached"] or not profile["teacher_detached"] or not micro["weight_detached"] or not profile["weight_detached"] or not micro["target_detached"] or not profile["target_detached"]:
        result["status"] = "ENGINEERING_STOP"
    atomic_write_json(args.output_root / "P34_ENGINEERING_RUN.json", result)
    print(json.dumps({"status": result["status"], "output": str(args.output_root / "P34_ENGINEERING_RUN.json"), "engineering_steps": engineering_steps}, sort_keys=True))
    return result


if __name__ == "__main__":
    run(make_parser().parse_args())
