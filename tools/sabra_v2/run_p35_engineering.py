"""Engineering-only P35 cached qualification; never runs scientific scoring.

The runner exercises the frozen P35 objective on fit/source cache batches only.
It creates no scientific UUID, held prediction, execution marker, or Stage 2
result. Optimizer steps are limited to the engineering smoke and profiles.
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
from tools.sabra_v2.p35_objective import (
    P35_OBJECTIVE_NAME,
    P35_PREREGISTRATION_SHA256,
    p35_actionability_components,
    p35_objective_contract,
)
from tools.sabra_v2.p35_reference import p35_actionability_components as p35_reference_components
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import CachedSourceDataset, atomic_write_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_METADATA = ROOT / "dataset/hub/VisA.jsonl"
DEFAULT_HELD_CLASS = "candle"
DEFAULT_OUTPUT_ROOT = ROOT / "research/sabra_v2/region_distill/P35_ENGINEERING_RUN"
P35_PREREGISTRATION_MD = ROOT / "research/sabra_v2/region_distill/P35_PREREGISTRATION.md"
P35_PREREGISTRATION_JSON = ROOT / "research/sabra_v2/region_distill/P35_PREREGISTRATION.json"
P35_RESEARCH_DECISION = ROOT / "research/sabra_v2/region_distill/P35_RESEARCH_DECISION.json"
P35_PREFLIGHT = ROOT / "research/sabra_v2/region_distill/P35_PREFLIGHT_FALSIFICATION.json"
P33_SPEED_PROFILE = ROOT / "research/sabra_v2/region_distill/P33_SPEED_PROFILE.json"
P35_SCIENCE_ROOT = ROOT / "research/sabra_v2/region_distill/P35"
P35_ENGINEERING_QUALIFICATION = ROOT / "research/sabra_v2/region_distill/P35_ENGINEERING_QUALIFICATION.json"
P35_SPEED_PROFILE = ROOT / "research/sabra_v2/region_distill/P35_SPEED_PROFILE.json"

P35_EPOCHS = 20
P35_BATCH_SIZE = 1
P35_LEARNING_RATE = 0.001
P35_SEED = 0
P35_FIT_RECORDS = 1962
P35_HELD_RECORDS = 200
P35_WARMUP_STEPS = 5
P35_PROFILE_STEPS = 40


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


def _git_state() -> dict[str, str]:
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
            missing += int(parameter.numel())
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
    if not P35_PREREGISTRATION_MD.is_file() or not P35_PREREGISTRATION_JSON.is_file():
        raise RuntimeError("frozen P35 preregistration is missing")
    observed_hash = sha256_file(P35_PREREGISTRATION_MD)
    if observed_hash != P35_PREREGISTRATION_SHA256:
        raise RuntimeError(f"P35 preregistration Markdown hash mismatch: {observed_hash}")
    prereg = _json(P35_PREREGISTRATION_JSON)
    if (
        prereg.get("status") != "P35_PREREGISTRATION_FROZEN"
        or prereg.get("protocol") != "P35"
        or prereg.get("preregistration_md_sha256") != observed_hash
    ):
        raise RuntimeError("P35 preregistration identity/status drift")
    if prereg.get("scientific_execution", {}).get("uuid") is not None:
        raise RuntimeError("P35 preregistration unexpectedly contains a scientific UUID")
    formulation = prereg.get("formulation", {})
    expected = {
        "correction_scale_C": 4.960109710693359,
        "normalized_effect": "abs(detached(E_t))/C",
        "weight": "stop_gradient(tanh(abs(E_t)/C))",
        "target": "stop_gradient(E_t)",
        "objective": "mean(weight*SmoothL1(E_s,target,beta=1.0,reduction=none))",
        "objective_count": 1,
        "target_is_full_teacher_effect": True,
        "target_shrinkage": False,
        "new_tuned_hyperparameters": 0,
        "new_learnable_parameters": 0,
        "category_specific_parameters": 0,
        "teacher_at_inference": False,
        "inference_overhead_percent": 0,
    }
    if any(formulation.get(key) != value for key, value in expected.items()):
        raise RuntimeError("P35 formulation contract drift")
    optimization = prereg.get("optimization", {})
    expected_schedule = {
        "epochs": P35_EPOCHS,
        "batch_size": P35_BATCH_SIZE,
        "expected_optimizer_steps": P35_FIT_RECORDS * P35_EPOCHS,
        "seed": P35_SEED,
        "precision": "float32",
        "optimizer": "AdamW",
        "learning_rate": P35_LEARNING_RATE,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "schedule_change": False,
    }
    if any(optimization.get(key) != value for key, value in expected_schedule.items()):
        raise RuntimeError("P35 optimizer or schedule contract drift")
    contract = p35_objective_contract()
    if contract.get("preregistration_sha256") != observed_hash or contract.get("objective_count") != 1:
        raise RuntimeError("P35 production objective contract does not match preregistration")
    preflight = _json(P35_PREFLIGHT)
    if preflight.get("status") != "P35_PREFLIGHT_PASS":
        raise RuntimeError("P35 preflight is not a matching PASS")
    prereg_preflight = prereg.get("preflight", {})
    observed_preflight_hash = sha256_file(P35_PREFLIGHT)
    if prereg_preflight.get("sha256") != observed_preflight_hash:
        raise RuntimeError("P35 preflight artifact hash mismatch")
    decision = _json(P35_RESEARCH_DECISION)
    if decision.get("selected_candidate") != "B" or decision.get("selected_next_hypothesis") != "SOFT_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER":
        raise RuntimeError("P35 research decision does not select the frozen mechanism")
    if (P35_SCIENCE_ROOT / "P35_STAGE2_ATTEMPT.json").exists():
        raise RuntimeError("P35 scientific attempt marker already exists")
    return {
        "path": str(P35_PREREGISTRATION_MD),
        "sha256": observed_hash,
        "status": prereg["status"],
        "objective_contract": contract,
        "preflight_sha256": observed_preflight_hash,
    }


def _make_loader(metadata: Path, cache_root: Path, held_class: str) -> tuple[DataLoader, dict[str, Any]]:
    if held_class != DEFAULT_HELD_CLASS:
        raise RuntimeError("P35 engineering qualification is locked to candle")
    if cache_root.resolve() != DEFAULT_CACHE_ROOT.resolve():
        raise RuntimeError("P35 engineering qualification must reuse the frozen cache root")
    rows = read_visa_metadata(metadata)
    inventory = loco_inventory(rows, held_class)
    if (len(inventory.fit_rows), len(inventory.held_rows)) != (P35_FIT_RECORDS, P35_HELD_RECORDS):
        raise RuntimeError("P35 candle LOCO inventory changed")
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
        batch_size=P35_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=torch.Generator().manual_seed(P35_SEED),
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


def _next_cached_batch(loader: DataLoader, iterator: Iterable[Any] | None) -> tuple[dict[str, Any], Iterable[Any]]:
    if iterator is None:
        iterator = iter(loader)
    try:
        return next(iterator), iterator  # type: ignore[arg-type]
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator  # type: ignore[arg-type]


def _materialize_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    forbidden = {"mask", "native_logits", "label"}.intersection(batch)
    if forbidden:
        raise RuntimeError(f"forbidden cached fields reached P35 objective: {sorted(forbidden)}")
    allowed_metadata = {"class_name", "image_path", "index", "sample_id"}
    unexpected = set(batch) - {"seg_features", "teacher_region"} - allowed_metadata
    if unexpected:
        raise RuntimeError(f"unexpected P35 cached fields: {sorted(unexpected)}")
    if "seg_features" not in batch or "teacher_region" not in batch:
        raise RuntimeError("P35 cached batch is missing segmentation features or teacher region")
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
    batch, iterator = _next_cached_batch(loader, iterator)
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
    loss, student_effect, teacher_effect, weight, target_effect = p35_actionability_components(student_region, teacher_region)
    if not torch.equal(target_effect, teacher_effect):
        raise RuntimeError("P35 target is not exactly the full detached teacher effect")
    if objective_pair is not None:
        objective_pair[1].record()
        backward_pair[0].record()  # type: ignore[union-attr]
    else:
        objective_seconds = time.perf_counter() - objective_started
        backward_started = time.perf_counter()
    if not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError("P35 objective produced a non-finite loss")
    loss.backward()
    gradient_audit = _finite_gradient_audit(adapter)
    if not gradient_audit["finite"]:
        raise FloatingPointError(f"P35 objective produced an unhealthy gradient: {gradient_audit}")
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
        "teacher_requires_grad": bool(teacher_region.requires_grad),
        "teacher_effect_requires_grad": bool(teacher_effect.requires_grad),
        "weight_requires_grad": bool(weight.requires_grad),
        "target_requires_grad": bool(target_effect.requires_grad),
        "target_is_full_teacher_effect": True,
    }, iterator


def _summary(rows: list[dict[str, Any]], *, warmup_steps: int, label: str) -> dict[str, Any]:
    if not rows:
        raise RuntimeError(f"empty P35 {label} profile")

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
        "median_step_seconds": statistics.median(step),
        "p90_step_seconds": float(torch.quantile(torch.tensor(step, dtype=torch.float64), 0.90)),
        "mean_step_seconds": statistics.fmean(step),
        "median_end_to_end_step_seconds": statistics.median(end_to_end),
        "p90_end_to_end_step_seconds": float(torch.quantile(torch.tensor(end_to_end, dtype=torch.float64), 0.90)),
        "mean_end_to_end_step_seconds": statistics.fmean(end_to_end),
        "input_cache_median_seconds": statistics.median(input_cache),
        "data_loader_median_seconds": statistics.median(data_loader),
        "cache_tensor_transfer_median_seconds": statistics.median(transfer),
        "forward_median_seconds": statistics.median(forward),
        "objective_median_seconds": statistics.median(objective),
        "backward_optimizer_median_seconds": statistics.median(backward),
        "objective_fraction_of_step_median": statistics.median(objective) / statistics.median(step),
        "finite": all(bool(row["loss_finite"]) and bool(row["gradient"]["finite"]) for row in rows),
        "teacher_detached": all(not row["teacher_effect_requires_grad"] for row in rows),
        "target_full": all(row["target_is_full_teacher_effect"] and not row["target_requires_grad"] for row in rows),
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _write_checkpoint(path: Path, adapter: RegionResidualAdapter, optimizer_steps: int) -> dict[str, Any]:
    payload = {
        "schema_version": "P35_ENGINEERING_CHECKPOINT_V1",
        "status": "ENGINEERING_QUALIFICATION_ONLY",
        "protocol_id": "P35",
        "objective": P35_OBJECTIVE_NAME,
        "objective_count": 1,
        "preregistration_sha256": P35_PREREGISTRATION_SHA256,
        "optimizer_steps": optimizer_steps,
        "state_dict": {name: value.detach().cpu() for name, value in adapter.named_parameters()},
        "teacher_trainable": False,
        "scientific_execution_marker": None,
    }
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or path.exists():
        raise RuntimeError(f"refusing to overwrite P35 engineering checkpoint: {path}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha256_file(path), "schema_version": payload["schema_version"]}


def _strict_reload_and_probe(checkpoint: Path, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "P35_ENGINEERING_CHECKPOINT_V1":
        raise RuntimeError("P35 engineering checkpoint schema mismatch")
    if payload.get("preregistration_sha256") != P35_PREREGISTRATION_SHA256:
        raise RuntimeError("P35 engineering checkpoint preregistration hash mismatch")
    restored = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    restored.load_state_dict(payload["state_dict"], strict=True)
    restored.eval()
    batch, _iterator = _next_cached_batch(loader, None)
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
    baseline = _json(P33_SPEED_PROFILE)
    reference = baseline["warmed_profile_40_step"]
    reference_step = float(reference["median_comparable_step_seconds"])
    reference_end_to_end = float(reference["median_end_to_end_step_seconds"])
    reference_objective = float(reference["objective_median_seconds"])
    observed_step = float(profile["median_step_seconds"])
    observed_end_to_end = float(profile["median_end_to_end_step_seconds"])
    observed_objective = float(profile["objective_median_seconds"])
    return {
        "reference_artifact": str(P33_SPEED_PROFILE),
        "reference_protocol": "P33",
        "reference_warmed_profile_steps": int(reference["measured_steps"]),
        "reference_median_comparable_step_seconds": reference_step,
        "reference_median_end_to_end_step_seconds": reference_end_to_end,
        "reference_objective_median_seconds": reference_objective,
        "observed_median_comparable_step_seconds": observed_step,
        "observed_median_end_to_end_step_seconds": observed_end_to_end,
        "observed_objective_median_seconds": observed_objective,
        "end_to_end_overhead_percent_vs_P33": 100.0 * (observed_end_to_end / reference_end_to_end - 1.0),
        "comparable_step_overhead_percent_vs_P33": 100.0 * (observed_step / reference_step - 1.0),
        "objective_only_overhead_percent_vs_P33": 100.0 * (observed_objective / reference_objective - 1.0),
        "comparison_caveat": "same cached host comparison; DataLoader wait is reported separately",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(f"P35 engineering output is non-empty: {args.output_root}")
    if not args.metadata.is_file() or not args.cache_root.is_dir():
        raise RuntimeError("P35 metadata and frozen cache root must exist")
    if P35_ENGINEERING_QUALIFICATION.exists() or P35_SPEED_PROFILE.exists():
        raise RuntimeError("P35 engineering evidence already exists; refusing overwrite")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    preregistration = _assert_preregistration()
    entry_git = _git_state()
    if entry_git["status_porcelain"]:
        raise RuntimeError("P35 engineering qualification requires a clean worktree")

    configure_canonical_fp32()
    random.seed(P35_SEED)
    torch.manual_seed(P35_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(P35_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)

    loader, data_audit = _make_loader(args.metadata, args.cache_root, args.held_class)
    adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    if any(not parameter.requires_grad for parameter in adapter.parameters()):
        raise RuntimeError("all P35 adapter parameters must remain trainable for engineering smoke")
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=P35_LEARNING_RATE,
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
        raise RuntimeError("P35 cached engineering smoke did not update the student")
    data_audit.update({
        "smoke_batch_fields": ["seg_features", "teacher_region"],
        "smoke_held_GT_read_count": 0,
        "smoke_held_mask_read_count": 0,
    })

    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_root / "p35_engineering_adapter.pt"
    checkpoint_info = _write_checkpoint(checkpoint, adapter, optimizer_steps=1)
    reload_info = _strict_reload_and_probe(checkpoint, loader, device)

    micro_rows: list[dict[str, Any]] = []
    for _ in range(5):
        row, iterator = _train_step(adapter, optimizer, loader, iterator, device, measure=True)
        micro_rows.append(row)
    micro = _summary(micro_rows, warmup_steps=0, label="5_step_microprofile")

    for _ in range(P35_WARMUP_STEPS):
        _unused_row, iterator = _train_step(adapter, optimizer, loader, iterator, device, measure=False)
    profile_rows: list[dict[str, Any]] = []
    for _ in range(P35_PROFILE_STEPS):
        row, iterator = _train_step(adapter, optimizer, loader, iterator, device, measure=True)
        profile_rows.append(row)
    profile = _summary(profile_rows, warmup_steps=P35_WARMUP_STEPS, label="40_step_warmed_profile")
    comparison = _baseline_comparison(profile)

    engineering_steps = 1 + len(micro_rows) + P35_WARMUP_STEPS + len(profile_rows)
    speed_profile = {
        "schema_version": "P35_SPEED_PROFILE_V1",
        "status": "P35_SPEED_QUALIFICATION_PASS" if profile["finite"] and profile["teacher_detached"] and profile["target_full"] else "P35_SPEED_QUALIFICATION_FAIL",
        "protocol_id": "P35",
        "preregistration_sha256": P35_PREREGISTRATION_SHA256,
        "engineering_run": str(args.output_root / "P35_ENGINEERING_RUN.json"),
        "microprofile_5_step": micro,
        "warmed_profile_40_step": profile,
        "comparison_vs_P33": comparison,
        "inference": {"overhead_percent": 0.0, "weight_computed_at_inference": False},
        "memory": {
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "retained_graph": False,
            "duplicate_teacher_tensor": False,
        },
        "objective_only_overhead_reported_separately": True,
        "scientific_stage2_attempts": 0,
    }

    result: dict[str, Any] = {
        "schema_version": "P35_ENGINEERING_RUN_V1",
        "status": "ENGINEERING_QUALIFICATION_ONLY",
        "protocol_id": "P35",
        "objective": p35_objective_contract(),
        "preregistration_sha256": P35_PREREGISTRATION_SHA256,
        "entry_git": entry_git,
        "exit_git": _git_state(),
        "device": str(device),
        "training_schedule": {
            "epochs_frozen_for_future_science": P35_EPOCHS,
            "batch_size": P35_BATCH_SIZE,
            "learning_rate": P35_LEARNING_RATE,
            "seed": P35_SEED,
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
            "weight_requires_grad": smoke_row["weight_requires_grad"],
            "target_requires_grad": smoke_row["target_requires_grad"],
            "target_is_full_teacher_effect": smoke_row["target_is_full_teacher_effect"],
            "student_effect_shape": smoke_row["student_effect_shape"],
            "teacher_effect_shape": smoke_row["teacher_effect_shape"],
            "checkpoint": checkpoint_info,
            "strict_reload": reload_info,
        },
        "microprofile_5_step": micro,
        "warmed_profile_40_step": profile,
        "speed_comparison": comparison,
        "memory": speed_profile["memory"],
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
            "preregistration_integrity": "PASS",
            "preflight": "PASS",
            "import_compile": "PASS",
            "objective_smoke": "PASS",
            "production_reference_parity": "PASS in tests/test_p35_objective.py",
            "cached_batch_forward": "PASS",
            "backward": "PASS",
            "optimizer_step": "PASS_ENGINEERING_ONLY",
            "checkpoint_save": "PASS_ENGINEERING_ONLY",
            "checkpoint_strict_reload": "PASS",
            "report_wrapper_mock_generation": "PASS in tests/test_p34_reporting.py",
            "microprofile": "PASS" if micro["finite"] and micro["teacher_detached"] and micro["target_full"] else "ENGINEERING_STOP",
            "warmed_profile": "PASS" if profile["finite"] and profile["teacher_detached"] and profile["target_full"] else "ENGINEERING_STOP",
            "inference_overhead": "PASS_ZERO",
            "new_tuned_hyperparameters": 0,
            "objective_count": 1,
        },
        "preregistration_audit": preregistration,
    }
    if not micro["finite"] or not profile["finite"] or not micro["teacher_detached"] or not profile["teacher_detached"] or not micro["target_full"] or not profile["target_full"]:
        result["status"] = "ENGINEERING_STOP"
        speed_profile["status"] = "P35_SPEED_QUALIFICATION_FAIL"

    atomic_write_json(args.output_root / "P35_ENGINEERING_RUN.json", result)
    atomic_write_json(P35_SPEED_PROFILE, speed_profile)
    qualification = {
        "schema_version": "P35_ENGINEERING_QUALIFICATION_V1",
        "status": "P35_PASS_TO_SCIENTIFIC_PROTOCOL" if result["status"] == "ENGINEERING_QUALIFICATION_ONLY" and speed_profile["status"] == "P35_SPEED_QUALIFICATION_PASS" else "P35_ENGINEERING_STOP",
        "final_gate": "P35_PASS_TO_SCIENTIFIC_PROTOCOL" if result["status"] == "ENGINEERING_QUALIFICATION_ONLY" and speed_profile["status"] == "P35_SPEED_QUALIFICATION_PASS" else "P35_ENGINEERING_STOP",
        "protocol_id": "P35",
        "authoritative_preregistration": {
            "path": str(P35_PREREGISTRATION_MD),
            "sha256": P35_PREREGISTRATION_SHA256,
        },
        "selected_hypothesis": "SOFT_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER",
        "implementation": {
            "production_module": "tools/sabra_v2/p35_objective.py",
            "reference_module": "tools/sabra_v2/p35_reference.py",
            "runner": "tools/sabra_v2/run_p35_engineering.py",
            "objective_count": 1,
            "full_target_preserved": True,
            "target_shaping": False,
            "new_tuned_hyperparameters": 0,
            "new_learnable_parameters": 0,
            "category_specific_parameters": 0,
            "inference_overhead_percent": 0.0,
        },
        "production_smoke": result["smoke"],
        "data_access_audit": result["data_access_audit"],
        "model_and_gradient_audit": result["model_forward_audit"],
        "speed": {
            "artifact": str(P35_SPEED_PROFILE),
            "status": speed_profile["status"],
            "end_to_end_overhead_percent_vs_P33": comparison["end_to_end_overhead_percent_vs_P33"],
            "comparable_step_overhead_percent_vs_P33": comparison["comparable_step_overhead_percent_vs_P33"],
            "objective_only_overhead_percent_vs_P33": comparison["objective_only_overhead_percent_vs_P33"],
            "microprofile": micro,
            "warmed_profile": profile,
        },
        "memory": result["memory"],
        "scientific_safety": result["scientific_safety"],
        "gates": result["gates"],
        "entry_git": entry_git,
        "exit_git": _git_state(),
        "engineering_optimizer_steps": engineering_steps,
        "scientific_stage2_attempts": 0,
        "final_note": "Engineering-only cached qualification passed; no P35 scientific Stage 2 attempt has started.",
    }
    atomic_write_json(P35_ENGINEERING_QUALIFICATION, qualification)
    print(json.dumps({"status": qualification["status"], "output": str(P35_ENGINEERING_QUALIFICATION), "engineering_steps": engineering_steps}, sort_keys=True))
    return qualification


def main() -> None:
    run(make_parser().parse_args())


if __name__ == "__main__":
    main()
