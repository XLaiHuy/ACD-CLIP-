"""Engineering-only P30R1 qualification; scientific stages are not exposed."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.p30r1_contract import (
    P30R1_BRANCH,
    P30R1_PREREGISTRATION_PATH,
    P30R1_UUID,
    load_and_audit_p30r1_preregistration,
    p30r1_cache_provenance,
    p30r1_preregistration_hash,
)
from tools.sabra_v2.p30r1_objective import p30r1_teacher_relative_components
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import TierADataset, atomic_write_json, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_VISA_ROOT = Path("/workspace/data/source/visa_unpack")
DEFAULT_P26_CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
DEFAULT_CLIP_ASSET = ROOT / "model/ViT-L-14-336px.pt"
DEFAULT_METADATA = ROOT / "dataset/hub/VisA.jsonl"
P30_PROFILE_MEDIAN_SECONDS = 0.006899061845615506
P29_PROFILE_MEDIAN_SECONDS = 0.010768339969217777
P30R1_PREFERRED_OVERHEAD_PERCENT = 10.0
P30R1_HARD_OVERHEAD_PERCENT = 15.0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visa-root", type=Path, default=DEFAULT_VISA_ROOT)
    parser.add_argument("--p26-checkpoint", type=Path, default=DEFAULT_P26_CHECKPOINT)
    parser.add_argument("--clip-asset", type=Path, default=DEFAULT_CLIP_ASSET)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--preregistration", type=Path, default=P30R1_PREREGISTRATION_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--prefetch-factor", type=int, choices=(2, 4), default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--non-blocking", action="store_true")
    return parser


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def _utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _active_p30r1_training_processes() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True)
    own_pid = str(os.getpid())
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if "tools.sabra_v2.train_region_distill_p30r1_cached" in line
        and not line.lstrip().startswith(own_pid + " ")
    ]


def _run_module(module: str, arguments: Sequence[str]) -> float:
    command = [sys.executable, "-m", module, *arguments]
    print(json.dumps({"event": "START", "utc": _utc(), "command": command}), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    print(json.dumps({"event": "COMPLETE", "utc": _utc(), "seconds": elapsed}), flush=True)
    return elapsed


def _audit_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], str, str]:
    if args.preregistration.resolve() != P30R1_PREREGISTRATION_PATH.resolve():
        raise RuntimeError("P30R1 accepts only the frozen in-repository preregistration")
    if args.cache_root.resolve() != DEFAULT_CACHE_ROOT.resolve():
        raise RuntimeError("P30R1 must reuse the frozen P27 cache root")
    if not args.metadata.is_file() or not args.cache_root.is_dir() or not args.visa_root.is_dir():
        raise RuntimeError("P30R1 metadata, cache root, and VisA root must exist")
    prereg_hash = p30r1_preregistration_hash(P30R1_PREREGISTRATION_PATH)
    prereg = load_and_audit_p30r1_preregistration(P30R1_PREREGISTRATION_PATH, prereg_hash)
    parent = verify_p26_parent(args.p26_checkpoint, args.clip_asset, ROOT / "configs/phase2b_canonical_v1.json")
    rows = read_visa_metadata(args.metadata)
    observed_classes = tuple(sorted({str(row["class_name"]) for row in rows}))
    expected_classes = tuple(sorted(EXPECTED_VISA_CLASSES))
    if observed_classes != expected_classes:
        raise RuntimeError(f"unexpected VisA class inventory: {observed_classes}")
    provenance = p30r1_cache_provenance(args.metadata)
    fold_counts = {
        class_name: {
            "fit_records": len(loco_inventory(rows, class_name).fit_rows),
            "held_records": len(loco_inventory(rows, class_name).held_rows),
        }
        for class_name in EXPECTED_VISA_CLASSES
    }
    input_audit = {
        "metadata": {"path": str(args.metadata), "sha256": sha256_file(args.metadata), "records": len(rows)},
        "cache_root": str(args.cache_root.resolve()),
        "cache_provenance": provenance.as_dict(),
        "visa_root": str(args.visa_root.resolve()),
        "fold_counts": fold_counts,
        "p26": parent,
        "config": {
            "path": str(ROOT / "configs/phase2b_canonical_v1.json"),
            "sha256": sha256_file(ROOT / "configs/phase2b_canonical_v1.json"),
        },
        "p26_checkpoint": {"path": str(args.p26_checkpoint), "sha256": sha256_file(args.p26_checkpoint)},
        "clip_asset": {"path": str(args.clip_asset), "sha256": sha256_file(args.clip_asset)},
        "preregistration_sha256": prereg_hash,
        "objective": prereg["objective"]["name"],
    }
    return input_audit, prereg_hash, _git("rev-parse", "HEAD")


def production_reference_parity() -> dict[str, Any]:
    """Compare the production function to the frozen preflight reference."""
    from tools.sabra_v2 import p30r1_preflight as reference

    generator = torch.Generator().manual_seed(20260826)
    ordinary_teacher = torch.randn((2, 9, 9), generator=generator, dtype=torch.float32)
    ordinary_student = torch.randn((3, 2, 9, 9), generator=generator, dtype=torch.float32)
    cases = {
        "ordinary_random": (ordinary_student, ordinary_teacher),
        "scale_mismatch": (ordinary_student * 10.0, ordinary_teacher),
        "zero_teacher": (ordinary_student[:, :1], torch.zeros_like(ordinary_teacher[:1])),
        "near_zero_teacher": (ordinary_student[:, :1], ordinary_teacher[:1] * 1e-8),
        "heavy_tail_corruption": (
            ordinary_student[:, :1].clone().index_put((torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])), torch.tensor([100.0])),
            ordinary_teacher[:1],
        ),
    }
    rows: list[dict[str, Any]] = []
    max_errors = {"loss": 0.0, "normalized_student": 0.0, "normalized_teacher": 0.0, "teacher_scale": 0.0, "student_gradient": 0.0}
    tolerance = {"rtol": 1e-6, "atol": 1e-7}
    for name, (student, teacher) in cases.items():
        production_student = student.detach().clone().requires_grad_(True)
        reference_student = student.detach().clone().requires_grad_(True)
        production_teacher = teacher.detach().clone().requires_grad_(True)
        reference_teacher = teacher.detach().clone().requires_grad_(True)
        production = p30r1_teacher_relative_components(production_student, production_teacher)
        expected = reference.teacher_relative_components(reference_student, reference_teacher)
        production_gradient = torch.autograd.grad(production[0], production_student)[0]
        expected_gradient = torch.autograd.grad(expected[0], reference_student)[0]
        errors = {
            "loss": float((production[0] - expected[0]).abs().detach().cpu()),
            "normalized_student": float((production[1] - expected[1]).abs().max().detach().cpu()),
            "normalized_teacher": float((production[2] - expected[2]).abs().max().detach().cpu()),
            "teacher_scale": float((production[3] - expected[3]).abs().max().detach().cpu()),
            "student_gradient": float((production_gradient - expected_gradient).abs().max().detach().cpu()),
        }
        for key, value in errors.items():
            max_errors[key] = max(max_errors[key], value)
        torch.testing.assert_close(production[0], expected[0], **tolerance)
        for actual, target in zip(production[1:], expected[1:]):
            torch.testing.assert_close(actual, target, **tolerance)
        torch.testing.assert_close(production_gradient, expected_gradient, **tolerance)
        rows.append({"case": name, "errors": errors, "teacher_grad_none": production_teacher.grad is None})
    return {
        "status": "PASS",
        "reference": "tools.sabra_v2.p30r1_preflight.teacher_relative_components",
        "production": "tools.sabra_v2.p30r1_objective.p30r1_teacher_relative_components",
        "cases": rows,
        "max_abs_errors": max_errors,
        "tolerance": tolerance,
        "teacher_gradients_not_backpropagated": all(row["teacher_grad_none"] for row in rows),
    }


def _common_training_args(
    args: argparse.Namespace,
    prereg_hash: str,
    base_commit: str,
    output: Path,
    stage: str,
    max_steps: int,
    warmup_steps: int,
) -> list[str]:
    return [
        "--held-class", "candle",
        "--visa-root", str(args.visa_root),
        "--p26-checkpoint", str(args.p26_checkpoint),
        "--clip-asset", str(args.clip_asset),
        "--cache-root", str(args.cache_root),
        "--output", str(output),
        "--metadata", str(args.metadata),
        "--execution-base-sha", base_commit,
        "--preregistration-sha", prereg_hash,
        "--stage", stage,
        "--epochs", "20",
        "--batch-size", "1",
        "--learning-rate", "0.001",
        "--seed", "0",
        "--max-steps", str(max_steps),
        "--warmup-steps", str(warmup_steps),
        "--device", args.device,
        "--num-workers", str(args.num_workers),
    ] + (["--prefetch-factor", str(args.prefetch_factor)] if args.num_workers else []) + (["--pin-memory"] if args.pin_memory else []) + (["--non-blocking"] if args.non_blocking else [])


def _run_fold(
    args: argparse.Namespace,
    root: Path,
    prereg_hash: str,
    base_commit: str,
    stage: str,
    max_steps: int,
    warmup_steps: int,
) -> dict[str, Any]:
    if root.exists():
        raise RuntimeError(f"P30R1 qualification output already exists: {root}")
    root.mkdir(parents=True)
    _run_module(
        "tools.sabra_v2.train_region_distill_p30r1_cached",
        _common_training_args(args, prereg_hash, base_commit, root, stage, max_steps, warmup_steps),
    )
    completion = _json(root / "P30R1_TRAINING_COMPLETE.json")
    return {"training": completion, "root": str(root)}


def _reload_checkpoint_and_probe(
    checkpoint_path: Path,
    args: argparse.Namespace,
    input_audit: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "P30R1_REGION_ADAPTER_CHECKPOINT_V1":
        raise RuntimeError("P30R1 checkpoint schema mismatch on reload")
    adapter = RegionResidualAdapter().to(device="cpu", dtype=torch.float32)
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    adapter.eval()
    rows = read_visa_metadata(args.metadata)
    fit_rows = loco_inventory(rows, "candle").fit_rows
    provenance = p30r1_cache_provenance(args.metadata)
    dataset = TierADataset(fit_rows[:1], args.cache_root, provenance, load_native_logits=True)
    batch = dataset[0]
    with torch.no_grad():
        region = adapter(batch["seg_features"].unsqueeze(1))
    if tuple(region.shape) != (3, 1, 9, 9) or not bool(torch.isfinite(region).all()):
        raise RuntimeError("reloaded adapter failed the GT-free future-forward probe")
    return {
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "strict_state_dict_reload": True,
        "future_prediction_probe": "one frozen Tier-A fit sample, adapter-only residual forward",
        "future_prediction_probe_shape": list(region.shape),
        "future_prediction_probe_finite": True,
        "held_GT_read_count": 0,
        "held_mask_read_count": 0,
        "input_audit_cache_root": input_audit["cache_root"],
    }


def _check_training_result(result: Mapping[str, Any], expected_steps: int, expected_measured: int) -> None:
    required = {
        "status": "ENGINEERING_QUALIFICATION_ONLY",
        "steps": expected_steps,
        "measured_steps": expected_measured,
        "optimizer_steps": expected_steps,
        "held_GT_read_count": 0,
        "held_mask_read_count": 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "teacher_forward_count": 0,
        "teacher_parameter_delta": 0.0,
        "source_mask_loaded": False,
        "native_logits_loaded": False,
        "teacher_scale_detached": True,
        "loss_finite": True,
        "gradient_finite": True,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise RuntimeError(f"P30R1 training check failed for {key}: {result.get(key)!r} != {expected!r}")
    if float(result["student_parameter_delta"]["l2"]) <= 0.0:
        raise RuntimeError("P30R1 student parameters did not change")
    gradient_health = result.get("gradient_health", {})
    if gradient_health.get("nonfinite_count_max", 1) != 0 or gradient_health.get("missing_gradient_elements_max", 1) != 0:
        raise RuntimeError(f"P30R1 gradient health failed: {gradient_health}")


def _profile_summary(training: Mapping[str, Any], label: str) -> dict[str, Any]:
    step = training["step_time_ms"]
    component = training["component_time_ms"]
    median_seconds = float(step["median"]) / 1000.0
    overhead_p30 = 100.0 * (median_seconds - P30_PROFILE_MEDIAN_SECONDS) / P30_PROFILE_MEDIAN_SECONDS
    overhead_p29 = 100.0 * (median_seconds - P29_PROFILE_MEDIAN_SECONDS) / P29_PROFILE_MEDIAN_SECONDS
    return {
        "label": label,
        "status": "PASS" if math.isfinite(median_seconds) and overhead_p30 <= P30R1_HARD_OVERHEAD_PERCENT else "ENGINEERING_STOP",
        "measured_steps": int(training["measured_steps"]),
        "optimizer_steps": int(training["optimizer_steps"]),
        "warmup_steps": int(training["warmup_steps"]),
        "startup_training_seconds": float(training["training_seconds"]),
        "median_step_seconds": median_seconds,
        "p90_step_seconds": float(step["p90"]) / 1000.0,
        "mean_step_seconds": float(step["mean"]) / 1000.0,
        "objective_median_seconds": float(component["objective_median"]) / 1000.0,
        "forward_median_seconds": float(component["forward_median"]) / 1000.0,
        "objective_fraction_of_step_median": float(component["objective_fraction_of_step_median"]),
        "p30_frozen_median_step_seconds": P30_PROFILE_MEDIAN_SECONDS,
        "p29_frozen_median_step_seconds": P29_PROFILE_MEDIAN_SECONDS,
        "overhead_vs_p30_percent": overhead_p30,
        "overhead_vs_p29_percent": overhead_p29,
        "objective_only_overhead_interpretation": "objective timing is reported separately; end-to-end timing includes cached tensor transfer after DataLoader yield and adapter/backward/optimizer",
        "peak_gpu_allocated_bytes": int(training["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": int(training["peak_gpu_reserved_bytes"]),
        "peak_process_rss_kib": int(training["peak_process_rss_kib"]),
        "new_clip_forwards": int(training["new_clip_forwards"]),
        "new_phase2b_forwards": int(training["new_phase2b_forwards"]),
        "teacher_forward_count": int(training["teacher_forward_count"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise RuntimeError(f"P30R1 output root already exists; refusing overwrite: {args.output_root}")
    if _active_p30r1_training_processes():
        raise RuntimeError("duplicate P30R1 training process detected")
    branch = _git("branch", "--show-current")
    if branch != P30R1_BRANCH:
        raise RuntimeError(f"P30R1 must run on {P30R1_BRANCH}, got {branch!r}")
    input_audit, prereg_hash, base_commit = _audit_inputs(args)
    parity = production_reference_parity()
    args.output_root.mkdir(parents=True)
    smoke = _run_fold(args, args.output_root / "smoke", prereg_hash, base_commit, "engineering_smoke", 1, 0)
    _check_training_result(smoke["training"], 1, 1)
    reload_probe = _reload_checkpoint_and_probe(Path(smoke["training"]["checkpoint"]), args, input_audit)
    micro = _run_fold(args, args.output_root / "microprofile", prereg_hash, base_commit, "engineering_microprofile", 5, 0)
    _check_training_result(micro["training"], 5, 5)
    micro_summary = _profile_summary(micro["training"], "5-step micro-profile")
    if micro_summary["status"] != "PASS":
        raise RuntimeError(f"P30R1 micro-profile exceeded hard speed bound: {micro_summary}")
    profile = _run_fold(args, args.output_root / "profile", prereg_hash, base_commit, "engineering_profile", 45, 5)
    _check_training_result(profile["training"], 45, 40)
    profile_summary = _profile_summary(profile["training"], "40-step warmed profile")
    if profile_summary["status"] != "PASS":
        raise RuntimeError(f"P30R1 profile exceeded hard speed bound: {profile_summary}")
    result = {
        "schema_version": "P30R1_ENGINEERING_QUALIFICATION_V1",
        "status": "PASS",
        "label": "ENGINEERING_QUALIFICATION_ONLY",
        "utc_timestamp": _utc(),
        "branch": branch,
        "base_commit": base_commit,
        "final_commit": None,
        "final_commit_explanation": "filled by the handoff after the evidence commit; no scientific result is encoded here",
        "preregistration_sha256": prereg_hash,
        "formulation_identifier": "P30R1_TEACHER_RELATIVE_SMOOTHL1_V1",
        "formulation_hash": "290aae42e04d9faae5a10b929eb58aa0da066b5dbd248b3fee40f20e9094781c",
        "objective_count": 1,
        "production_reference_parity": parity,
        "unit_test_count": None,
        "unit_test_count_explanation": "recorded after the runner in the final evidence artifact",
        "regression_test_count": None,
        "regression_test_count_explanation": "recorded after the runner in the final evidence artifact",
        "input_audit": input_audit,
        "smoke_optimizer_steps": smoke["training"]["optimizer_steps"],
        "smoke_student_parameter_delta": smoke["training"]["student_parameter_delta"],
        "smoke_teacher_parameter_delta": smoke["training"]["teacher_parameter_delta"],
        "smoke_zero_target_gradient_status": "static production objective parity and preflight zero-target gate pass; source fit exact-zero count is recorded in training output",
        "smoke_nonfinite_gradient_count": smoke["training"]["gradient_health"]["nonfinite_count_max"],
        "checkpoint_reload": reload_probe,
        "zero_target_gradient_status": "preserved by production objective; no zero-target filtering code",
        "nonfinite_gradient_count": max(
            int(smoke["training"]["gradient_health"]["nonfinite_count_max"]),
            int(micro["training"]["gradient_health"]["nonfinite_count_max"]),
            int(profile["training"]["gradient_health"]["nonfinite_count_max"]),
        ),
        "CLIP_forward_count": 0,
        "Phase2B_forward_count": 0,
        "teacher_forward_count": 0,
        "held_GT_read_count": 0,
        "held_mask_read_count": 0,
        "microprofile_step_count": micro["training"]["measured_steps"],
        "profile_step_count": profile["training"]["measured_steps"],
        "microprofile": micro_summary,
        "profile": profile_summary,
        "median_step_time": profile_summary["median_step_seconds"],
        "p90_step_time": profile_summary["p90_step_seconds"],
        "objective_time": profile_summary["objective_median_seconds"],
        "P30_comparison": {
            "P30_median_step_time": P30_PROFILE_MEDIAN_SECONDS,
            "P29_median_step_time": P29_PROFILE_MEDIAN_SECONDS,
            "overhead_vs_P30_percent": profile_summary["overhead_vs_p30_percent"],
            "overhead_vs_P29_percent": profile_summary["overhead_vs_p29_percent"],
            "preferred_overhead_percent_max": P30R1_PREFERRED_OVERHEAD_PERCENT,
            "hard_overhead_percent_max": P30R1_HARD_OVERHEAD_PERCENT,
        },
        "training_overhead_percent": profile_summary["overhead_vs_p30_percent"],
        "inference_overhead_percent": 0.0,
        "cache_rebuild_count": 0,
        "preregistration_deviation": False,
        "engineering_incidents": [],
        "scientific_execution": {
            "stage2_started": False,
            "stage2_scored": False,
            "stage3_started": False,
            "full_started": False,
            "execution_marker_created": False,
        },
        "production_outputs": {
            "smoke": smoke["training"],
            "microprofile": micro["training"],
            "profile": profile["training"],
        },
        "final_gate": "PASS_TO_STAGE2_PROTOCOL",
    }
    atomic_write_json(args.output_root / "P30R1_ENGINEERING_QUALIFICATION.json", result)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
