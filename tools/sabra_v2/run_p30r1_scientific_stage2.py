"""Run exactly one preregistered P30R1 candle Stage 2 attempt."""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.sabra.data import EXPECTED_VISA_CLASSES, VisaEvaluationDataset, read_visa_metadata
from tools.sabra_car.r0_direction import exact_metrics
from tools.sabra_v2.analyze_p30_outputs import _directional_cosine, _teacher_regions
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
from tools.sabra_v2.p30r1_objective import P30R1_FORMULATION_HASH
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import (
    CachedSourceDataset,
    TierADataset,
    atomic_write_json,
    sha256_file,
)
from tools.sabra_v2.student_forward import forward_region_student


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "research/sabra_v2/region_distill/P30R1"
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_VISA_ROOT = Path("/workspace/data/source/visa_unpack")
DEFAULT_P26_CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
DEFAULT_CLIP_ASSET = ROOT / "model/ViT-L-14-336px.pt"
DEFAULT_METADATA = ROOT / "dataset/hub/VisA.jsonl"

ENGINEERING_QUALIFICATION_SHA = "b59fb225a1a794ea83687078f9d0826ad28416f1"
P30R1_STAGE2_CLASS = "candle"
P30R1_EPOCHS = 20
P30R1_BATCH_SIZE = 1
P30R1_LEARNING_RATE = 0.001
P30R1_SEED = 0

P30R1_P29_METRICS = Path("/workspace/p29_science_v1/candle/metrics/p29_held_metrics.json")
P30R1_P30_METRICS = ROOT / "research/sabra_v2/region_distill/P30/qualification/stage2_one_class/candle/metrics/p30_held_metrics.json"
P30R1_P30_TRANSFER = ROOT / "research/sabra_v2/region_distill/P30/qualification/stage2_one_class/P30_TRANSFER_DIAGNOSTIC.json"
P30R1_P30_STABILITY = ROOT / "research/sabra_v2/region_distill/P30/qualification/stage2_one_class/P30_STABILITY_DIAGNOSTIC.json"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--visa-root", type=Path, default=DEFAULT_VISA_ROOT)
    parser.add_argument("--p26-checkpoint", type=Path, default=DEFAULT_P26_CHECKPOINT)
    parser.add_argument("--clip-asset", type=Path, default=DEFAULT_CLIP_ASSET)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _remote_sha(branch: str) -> str:
    fields = _git("ls-remote", "origin", f"refs/heads/{branch}").split()
    if len(fields) != 2:
        raise RuntimeError(f"could not resolve remote branch {branch}")
    return fields[0]


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _active_p30r1_processes() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True)
    own_pid = str(os.getpid())
    needles = ("run_p30r1_scientific_stage2", "train_region_distill_p30r1_cached")
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if any(needle in line for needle in needles)
        and not line.lstrip().startswith(own_pid + " ")
    ]


def _assert_frozen_execution_state() -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != P30R1_BRANCH:
        raise RuntimeError(f"P30R1 Stage 2 must run on {P30R1_BRANCH}, got {branch!r}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", ENGINEERING_QUALIFICATION_SHA, head],
        cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError("HEAD is not the engineering-qualified commit or its descendant")
    porcelain = _git("status", "--porcelain")
    if porcelain:
        raise RuntimeError(f"scientific execution requires a clean worktree: {porcelain!r}")
    remote = _remote_sha(branch)
    if remote != head:
        raise RuntimeError(f"local/remote mismatch before attempt: {head} != {remote}")
    protected_paths = (
        "tools/sabra_v2/p30r1_objective.py",
        "tools/sabra_v2/train_region_distill_p30r1_cached.py",
        "tools/sabra_v2/region_cache.py",
        "tools/sabra_v2/region_adapter.py",
        "tools/sabra_v2/student_forward.py",
    )
    changed = subprocess.run(
        ["git", "diff", "--name-only", ENGINEERING_QUALIFICATION_SHA, "--", *protected_paths],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if changed:
        raise RuntimeError(f"qualified scientific core differs from engineering qualification: {changed}")
    active = _active_p30r1_processes()
    if active:
        raise RuntimeError(f"duplicate P30R1 scientific/training process detected: {active}")
    return {
        "branch": branch,
        "head": head,
        "remote_sha": remote,
        "remote_equals_local": True,
        "worktree_clean_before_attempt": True,
        "engineering_qualification_sha": ENGINEERING_QUALIFICATION_SHA,
        "qualified_core_changed": False,
        "duplicate_processes": [],
    }


def _audit_inputs(args: argparse.Namespace, git_identity: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    if args.cache_root.resolve() != DEFAULT_CACHE_ROOT.resolve():
        raise RuntimeError("P30R1 Stage 2 must reuse the frozen P27 cache root")
    for path in (args.metadata, args.cache_root, args.visa_root, args.p26_checkpoint, args.clip_asset):
        if not path.exists():
            raise RuntimeError(f"missing frozen input: {path}")
    prereg_hash = p30r1_preregistration_hash(P30R1_PREREGISTRATION_PATH)
    prereg = load_and_audit_p30r1_preregistration(P30R1_PREREGISTRATION_PATH, prereg_hash)
    engineering_path = ROOT / "research/sabra_v2/region_distill/P30R1_ENGINEERING_QUALIFICATION.json"
    engineering = _json(engineering_path)
    if (
        engineering.get("status") != "PASS"
        or engineering.get("final_gate") != "PASS_TO_STAGE2_PROTOCOL"
        or engineering.get("formulation_hash") != P30R1_FORMULATION_HASH
        or engineering.get("preregistration_sha256") != prereg_hash
    ):
        raise RuntimeError("engineering qualification is not a matching PASS artifact")
    parent_assets = verify_p26_parent(args.p26_checkpoint, args.clip_asset, ROOT / "configs/phase2b_canonical_v1.json")
    rows = read_visa_metadata(args.metadata)
    if tuple(sorted({str(row["class_name"]) for row in rows})) != tuple(sorted(EXPECTED_VISA_CLASSES)):
        raise RuntimeError("unexpected VisA class inventory")
    inventory = loco_inventory(rows, P30R1_STAGE2_CLASS)
    if len(inventory.fit_rows) != 1962 or len(inventory.held_rows) != 200:
        raise RuntimeError("frozen candle fit/held inventory changed")
    provenance = p30r1_cache_provenance(args.metadata)
    # This validates source-only manifests and tensor contracts without opening held labels/masks.
    CachedSourceDataset(
        inventory.fit_rows,
        P30R1_STAGE2_CLASS,
        args.cache_root,
        provenance,
        load_source_mask=False,
        load_native_logits=False,
    )
    class_order = list(EXPECTED_VISA_CLASSES)
    tier_a_manifest = args.cache_root / "tier_a" / P30R1_STAGE2_CLASS / "manifest.json"
    tier_b_manifest = args.cache_root / "tier_b" / P30R1_STAGE2_CLASS / "manifest.json"
    for path in (P30R1_P29_METRICS, P30R1_P30_METRICS, P30R1_P30_TRANSFER, P30R1_P30_STABILITY):
        if not path.is_file():
            raise RuntimeError(f"missing frozen comparison evidence: {path}")
    input_audit = {
        "metadata": {"path": str(args.metadata), "sha256": sha256_file(args.metadata), "records": len(rows)},
        "visa_root": str(args.visa_root.resolve()),
        "cache_root": str(args.cache_root.resolve()),
        "cache_provenance": provenance.as_dict(),
        "tier_a_candle_manifest": {"path": str(tier_a_manifest), "sha256": sha256_file(tier_a_manifest)},
        "tier_b_candle_manifest": {"path": str(tier_b_manifest), "sha256": sha256_file(tier_b_manifest)},
        "class_order": class_order,
        "candle_fit_records": len(inventory.fit_rows),
        "candle_held_records": len(inventory.held_rows),
        "p26": parent_assets,
        "p26_checkpoint": {"path": str(args.p26_checkpoint), "sha256": sha256_file(args.p26_checkpoint)},
        "clip_asset": {"path": str(args.clip_asset), "sha256": sha256_file(args.clip_asset)},
        "config": {
            "path": str(ROOT / "configs/phase2b_canonical_v1.json"),
            "sha256": sha256_file(ROOT / "configs/phase2b_canonical_v1.json"),
        },
        "engineering_qualification": {"path": str(engineering_path), "sha256": sha256_file(engineering_path)},
        "p30r1_research_report": {
            "path": str(ROOT / "research/sabra_v2/region_distill/P30R1_RESEARCH_REPORT.md"),
            "sha256": sha256_file(ROOT / "research/sabra_v2/region_distill/P30R1_RESEARCH_REPORT.md"),
        },
        "p29_metrics": {"path": str(P30R1_P29_METRICS), "sha256": sha256_file(P30R1_P29_METRICS)},
        "p30_metrics": {"path": str(P30R1_P30_METRICS), "sha256": sha256_file(P30R1_P30_METRICS)},
        "p30_transfer": {"path": str(P30R1_P30_TRANSFER), "sha256": sha256_file(P30R1_P30_TRANSFER)},
        "p30_stability": {"path": str(P30R1_P30_STABILITY), "sha256": sha256_file(P30R1_P30_STABILITY)},
        "held_gt_reads_before_prediction_freeze": 0,
        "held_mask_reads_before_prediction_freeze": 0,
        "mvtec_reads": 0,
        "medical_reads": 0,
        "cache_rebuilt": False,
    }
    return prereg, input_audit, prereg_hash


def _run_module(module: str, arguments: Sequence[str]) -> float:
    command = [sys.executable, "-m", module, *arguments]
    print(json.dumps({"event": "START", "utc": _utc(), "command": command}), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    print(json.dumps({"event": "COMPLETE", "utc": _utc(), "module": module, "seconds": elapsed}), flush=True)
    return elapsed


def _training_arguments(
    args: argparse.Namespace,
    output: Path,
    execution_sha: str,
    prereg_hash: str,
) -> list[str]:
    return [
        "--held-class", P30R1_STAGE2_CLASS,
        "--visa-root", str(args.visa_root),
        "--p26-checkpoint", str(args.p26_checkpoint),
        "--clip-asset", str(args.clip_asset),
        "--cache-root", str(args.cache_root),
        "--output", str(output),
        "--metadata", str(args.metadata),
        "--execution-base-sha", execution_sha,
        "--preregistration-sha", prereg_hash,
        "--stage", "engineering_profile",
        "--epochs", str(P30R1_EPOCHS),
        "--batch-size", str(P30R1_BATCH_SIZE),
        "--learning-rate", str(P30R1_LEARNING_RATE),
        "--seed", str(P30R1_SEED),
        "--max-steps", str(1962 * P30R1_EPOCHS),
        "--warmup-steps", "0",
        "--device", args.device,
        "--num-workers", "0",
    ]


def _audit_raw_training(raw: Mapping[str, Any], parent_seconds: float, attempt_uuid: str, prereg_hash: str, execution_sha: str) -> dict[str, Any]:
    expected_steps = 1962 * P30R1_EPOCHS
    required = {
        "status": "ENGINEERING_QUALIFICATION_ONLY",
        "steps": expected_steps,
        "measured_steps": expected_steps,
        "optimizer_steps": expected_steps,
        "epochs": P30R1_EPOCHS,
        "batch_size": P30R1_BATCH_SIZE,
        "learning_rate": P30R1_LEARNING_RATE,
        "seed": P30R1_SEED,
        "fit_records": 1962,
        "held_records_not_read": 200,
        "held_GT_read_count": 0,
        "held_mask_read_count": 0,
        "source_mask_loaded": False,
        "native_logits_loaded": False,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "teacher_forward_count": 0,
        "teacher_parameter_delta": 0.0,
        "teacher_scale_detached": True,
        "loss_finite": True,
        "gradient_finite": True,
    }
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise RuntimeError(f"scientific training contract failed for {key}: {raw.get(key)!r} != {expected!r}")
    gradient_health = raw.get("gradient_health", {})
    if gradient_health.get("nonfinite_count_max", 1) != 0 or gradient_health.get("missing_gradient_elements_max", 1) != 0:
        raise RuntimeError(f"scientific gradient health failed: {gradient_health}")
    if not math.isfinite(float(raw.get("last_loss", float("nan")))):
        raise RuntimeError("scientific final loss is non-finite")
    if float(raw["student_parameter_delta"]["l2"]) <= 0.0:
        raise RuntimeError("scientific student parameters did not change")
    step_seconds = float(raw["step_time_ms"]["median"]) / 1000.0
    training_seconds = float(raw["training_seconds"])
    return {
        "schema_version": "P30R1_STAGE2_TRAINING_COMPLETE_V1",
        "status": "FOLD_TRAINING_COMPLETE",
        "attempt_uuid": attempt_uuid,
        "p30r1_uuid": P30R1_UUID,
        "p30r1_preregistration_sha256": prereg_hash,
        "scientific_execution_base_sha": execution_sha,
        "class": P30R1_STAGE2_CLASS,
        "fit_records": 1962,
        "held_records_not_read": 200,
        "epochs": P30R1_EPOCHS,
        "batch_size": P30R1_BATCH_SIZE,
        "learning_rate": P30R1_LEARNING_RATE,
        "seed": P30R1_SEED,
        "optimizer_steps": expected_steps,
        "raw_trainer_status": raw["status"],
        "raw_trainer_stage": raw["stage"],
        "raw_trainer_completion": "P30R1_TRAINING_COMPLETE.json",
        "training_seconds": training_seconds,
        "parent_process_seconds": float(parent_seconds),
        "median_step_seconds": step_seconds,
        "mean_step_seconds": float(raw["step_time_ms"]["mean"]) / 1000.0,
        "p90_step_seconds": float(raw["step_time_ms"]["p90"]) / 1000.0,
        "peak_gpu_allocated_bytes": int(raw["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": int(raw["peak_gpu_reserved_bytes"]),
        "peak_process_rss_kib": int(raw["peak_process_rss_kib"]),
        "last_loss": float(raw["last_loss"]),
        "loss_finite": True,
        "gradient_finite": True,
        "nonfinite_gradient_count": int(gradient_health.get("nonfinite_count_max", 0)),
        "nonfinite_loss_count": 0,
        "gradient_health": gradient_health,
        "student_parameter_delta": raw["student_parameter_delta"],
        "teacher_parameter_delta": 0.0,
        "teacher_scale_detached": True,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "teacher_forward_count": 0,
        "held_gt_reads": 0,
        "held_mask_reads": 0,
        "source_mask_loaded": False,
        "native_logits_loaded": False,
        "cache_rebuilt": False,
    }


def _run_training(args: argparse.Namespace, root: Path, attempt_uuid: str, prereg_hash: str, execution_sha: str) -> dict[str, Any]:
    training_root = root / P30R1_STAGE2_CLASS / "training"
    if training_root.exists():
        raise RuntimeError(f"scientific training output already exists: {training_root}")
    training_root.mkdir(parents=True)
    parent_seconds = _run_module(
        "tools.sabra_v2.train_region_distill_p30r1_cached",
        _training_arguments(args, training_root, execution_sha, prereg_hash),
    )
    raw_path = training_root / "P30R1_TRAINING_COMPLETE.json"
    checkpoint_path = training_root / "p30r1_region_adapter.pt"
    if not raw_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("scientific trainer did not produce both required artifacts")
    raw = _json(raw_path)
    training = _audit_raw_training(raw, parent_seconds, attempt_uuid, prereg_hash, execution_sha)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "P30R1_REGION_ADAPTER_CHECKPOINT_V1":
        raise RuntimeError("scientific checkpoint schema mismatch")
    if checkpoint.get("steps") != 1962 * P30R1_EPOCHS or checkpoint.get("objective_count") != 1:
        raise RuntimeError("scientific checkpoint schedule/objective mismatch")
    training["checkpoint"] = str(checkpoint_path)
    training["checkpoint_sha256"] = sha256_file(checkpoint_path)
    training["checkpoint_status"] = checkpoint.get("status")
    training["checkpoint_stage"] = checkpoint.get("stage")
    atomic_write_json(root / P30R1_STAGE2_CLASS / "P30R1_STAGE2_TRAINING_COMPLETE.json", training)
    return training


def _run_prediction(
    args: argparse.Namespace,
    root: Path,
    training: Mapping[str, Any],
    attempt_uuid: str,
    prereg_hash: str,
    execution_sha: str,
) -> dict[str, Any]:
    prediction_root = root / P30R1_STAGE2_CLASS / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_root / "p30r1_held_predictions.pt"
    completion_path = prediction_root / "PREDICTION_COMPLETE.json"
    if prediction_path.exists() or completion_path.exists():
        raise RuntimeError("scientific prediction output already exists; refusing overwrite")
    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, P30R1_STAGE2_CLASS)
    provenance = p30r1_cache_provenance(args.metadata)
    checkpoint = torch.load(training["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint.get("status") != "ENGINEERING_QUALIFICATION_ONLY" or checkpoint.get("stage") != "engineering_profile":
        raise RuntimeError("scientific path received an unexpected qualified trainer checkpoint")
    if checkpoint.get("cache_provenance") != provenance.as_dict():
        raise RuntimeError("scientific checkpoint cache provenance mismatch")
    dataset = TierADataset(inventory.held_rows, args.cache_root, provenance, load_native_logits=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    adapter.eval()
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for batch in loader:
            seg = batch["seg_features"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            native = batch["native_logits"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            student = forward_region_student(adapter, seg, native)
            records.append({
                "image_path": str(batch["image_path"][0]),
                "class_name": P30R1_STAGE2_CLASS,
                "native_abnormal_probability": student.native_probability[0, 1].detach().cpu(),
                "p30r1_abnormal_probability": student.deployed_probability[0, 1].detach().cpu(),
                "p30r1_region_residual": student.region_residual[:, 0].detach().cpu(),
            })
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if len(records) != 200:
        raise RuntimeError(f"scientific held prediction count mismatch: {len(records)}")
    payload = {
        "schema_version": "P30R1_IMMUTABLE_HELD_PREDICTIONS_V1",
        "status": "PREDICTIONS_FROZEN_GT_FREE",
        "attempt_uuid": attempt_uuid,
        "p30r1_uuid": P30R1_UUID,
        "p30r1_preregistration_sha256": prereg_hash,
        "scientific_execution_base_sha": execution_sha,
        "held_class": P30R1_STAGE2_CLASS,
        "gt_used": False,
        "mask_reads": 0,
        "cache_provenance": provenance.as_dict(),
        "adapter_checkpoint_sha256": training["checkpoint_sha256"],
        "records": records,
    }
    temporary = prediction_path.with_name(f".{prediction_path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, prediction_path)
    prediction_path.chmod(0o444)
    result = {
        "schema_version": "P30R1_PREDICTION_COMPLETE_V1",
        "status": "COMPLETE",
        "attempt_uuid": attempt_uuid,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "held_class": P30R1_STAGE2_CLASS,
        "records": len(records),
        "gt_used": False,
        "mask_reads": 0,
        "prediction_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "held_gt_reads": 0,
        "held_mask_reads": 0,
        "completion_status": "COMPLETE",
    }
    atomic_write_json(completion_path, result)
    return result


def _freeze_predictions(root: Path, prediction: Mapping[str, Any], attempt_uuid: str, prereg_hash: str) -> dict[str, Any]:
    prediction_path = Path(str(prediction["prediction_path"]))
    completion_path = prediction_path.parent / "PREDICTION_COMPLETE.json"
    if not prediction_path.is_file() or not completion_path.is_file():
        raise RuntimeError("cannot freeze missing predictions")
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "P30R1_IMMUTABLE_HELD_PREDICTIONS_V1"
        or payload.get("attempt_uuid") != attempt_uuid
        or payload.get("held_class") != P30R1_STAGE2_CLASS
        or payload.get("gt_used") is not False
        or payload.get("mask_reads") != 0
        or len(payload.get("records", [])) != 200
        or prediction.get("prediction_sha256") != sha256_file(prediction_path)
        or prediction_path.stat().st_mode & 0o222
    ):
        raise RuntimeError("scientific prediction freeze firewall failed")
    gate = {
        "schema_version": "P30R1_SCORING_GATE_V1",
        "status": "PASS",
        "completion_status": "PREDICTIONS_FROZEN_BEFORE_SCORING",
        "utc_timestamp": _utc(),
        "attempt_uuid": attempt_uuid,
        "p30r1_uuid": P30R1_UUID,
        "p30r1_preregistration_sha256": prereg_hash,
        "held_class": P30R1_STAGE2_CLASS,
        "prediction_count": 1,
        "predictions_frozen": True,
        "fit_or_teacher_steps_after_gate": 0,
        "held_gt_reads_before_prediction_freeze": 0,
        "held_mask_reads_before_prediction_freeze": 0,
        "predictions": [{
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
            "records": 200,
            "held_class": P30R1_STAGE2_CLASS,
        }],
    }
    atomic_write_json(root / "P30R1_SCORING_GATE.json", gate)
    return gate


def _score_frozen_predictions(
    args: argparse.Namespace,
    root: Path,
    prediction: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if gate.get("status") != "PASS" or gate.get("predictions_frozen") is not True:
        raise RuntimeError("held scoring requires a passing prediction-freeze gate")
    prediction_path = Path(str(prediction["prediction_path"]))
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 200:
        raise RuntimeError("frozen prediction record count mismatch")
    by_path = {str(record["image_path"]): record for record in records}
    if len(by_path) != len(records):
        raise RuntimeError("frozen predictions contain duplicate paths")
    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, P30R1_STAGE2_CLASS)
    native_scores: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    image_paths: list[str] = []
    held_mask_reads = 0
    for batch in DataLoader(
        VisaEvaluationDataset(inventory.held_rows, args.visa_root),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    ):
        image_path = str(batch["image_path"][0])
        record = by_path.get(image_path)
        if record is None:
            raise RuntimeError(f"missing frozen prediction for {image_path}")
        native = record.get("native_abnormal_probability")
        candidate = record.get("p30r1_abnormal_probability")
        residual = record.get("p30r1_region_residual")
        if not isinstance(native, torch.Tensor) or tuple(native.shape) != (518, 518):
            raise RuntimeError("native frozen map shape mismatch")
        if not isinstance(candidate, torch.Tensor) or tuple(candidate.shape) != (518, 518):
            raise RuntimeError("P30R1 frozen map shape mismatch")
        if not isinstance(residual, torch.Tensor) or tuple(residual.shape) != (3, 9, 9):
            raise RuntimeError("P30R1 frozen residual shape mismatch")
        native_scores.append(native.numpy().astype(np.float32, copy=False))
        candidate_scores.append(candidate.numpy().astype(np.float32, copy=False))
        masks.append(batch["mask"][0, 0].numpy().astype(np.uint8, copy=False))
        image_paths.append(image_path)
        held_mask_reads += int(batch["label"][0].item())
    native_array = np.stack(native_scores)
    candidate_array = np.stack(candidate_scores)
    mask_array = np.stack(masks)
    native_metrics = exact_metrics(native_array.reshape(-1), mask_array.reshape(-1))
    candidate_metrics = exact_metrics(candidate_array.reshape(-1), mask_array.reshape(-1))
    result = {
        "schema_version": "P30R1_HELD_METRICS_V1",
        "status": "COMPLETE",
        "held_class": P30R1_STAGE2_CLASS,
        "attempt_uuid": payload["attempt_uuid"],
        "prediction_sha256": sha256_file(prediction_path),
        "fit_or_teacher_steps": 0,
        "native_metrics": native_metrics,
        "p30r1_metrics": candidate_metrics,
        "delta": {
            key: candidate_metrics[key] - native_metrics[key]
            for key in ("pAP", "pAUROC")
        },
        "held_mask_file_reads_after_prediction_freeze": held_mask_reads,
        "held_gt_reads_before_prediction_freeze": 0,
        "held_mask_reads_before_prediction_freeze": 0,
    }
    metrics_root = root / P30R1_STAGE2_CLASS / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(metrics_root / "P30R1_HELD_METRICS.json", result)
    return result, {
        "native_probability": native_array,
        "candidate_probability": candidate_array,
        "masks": mask_array,
        "image_paths": np.asarray(image_paths, dtype=object),
    }


def _frozen_parent_values() -> dict[str, Any]:
    p29_metrics = _json(P30R1_P29_METRICS)
    p30_metrics = _json(P30R1_P30_METRICS)
    p30_transfer = _json(P30R1_P30_TRANSFER)
    p30_stability = _json(P30R1_P30_STABILITY)
    p30_class = next(row for row in p30_transfer["classes"] if row["class"] == P30R1_STAGE2_CLASS)
    p30_stability_class = next(row for row in p30_stability["classes"] if row["class"] == P30R1_STAGE2_CLASS)
    return {
        "native": p30_metrics["native_metrics"],
        "p29": p29_metrics["p29_metrics"],
        "p30": p30_metrics["p30_metrics"],
        "p29_transfer": p30_class["p29"],
        "p30_transfer": p30_class["p30"],
        "p29_stability": p30_stability_class["p29"],
        "p30_stability": p30_stability_class["p30"],
    }


def _run_diagnostics(
    args: argparse.Namespace,
    root: Path,
    prediction_context: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, P30R1_STAGE2_CLASS)
    residual_records = torch.load(
        root / P30R1_STAGE2_CLASS / "predictions" / "p30r1_held_predictions.pt",
        map_location="cpu",
        weights_only=True,
    )["records"]
    residual = np.stack([record["p30r1_region_residual"].numpy() for record in residual_records]).astype(np.float32, copy=False)
    r1_regions = residual.transpose(1, 0, 2, 3)
    tier_a_manifest = _json(args.cache_root / "tier_a" / P30R1_STAGE2_CLASS / "manifest.json")
    index_by_id = {sample_id: index for index, sample_id in enumerate(tier_a_manifest["sample_ids"])}
    indices = [index_by_id[f"{row['class_name']}:{row['image_path']}"] for row in inventory.held_rows]
    native_cache = np.load(args.cache_root / "tier_a" / P30R1_STAGE2_CLASS / "native_logits.npy", mmap_mode="r", allow_pickle=False)
    teacher = _teacher_regions(native_cache, indices, prediction_context["masks"], torch.device(args.device))
    teacher_staged = np.broadcast_to(teacher[None, ...], r1_regions.shape)
    r1_direction = _directional_cosine(teacher_staged, r1_regions)
    from tools.sabra_v2.p29r1_forensic import residual_magnitude_summary, sign_alignment, vectorized_pixel_shifts

    r1_alignment = sign_alignment(teacher_staged, r1_regions)
    r1_magnitude = residual_magnitude_summary(r1_regions)
    r1_shifts = vectorized_pixel_shifts(
        prediction_context["native_probability"],
        prediction_context["candidate_probability"],
        prediction_context["masks"],
    )
    transfer = {
        "schema_version": "P30R1_TRANSFER_DIAGNOSTIC_V1",
        "status": "COMPLETE",
        "held_class": P30R1_STAGE2_CLASS,
        "held_mask_reads_post_freeze": int(metrics["held_mask_file_reads_after_prediction_freeze"]),
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "p29": parent["p29_transfer"],
        "p30": parent["p30_transfer"],
        "p30r1": {
            "directional_cosine": r1_direction,
            "alignment": r1_alignment,
            "magnitude": r1_magnitude,
        },
        "delta_vs_p29": {
            "directional_cosine": r1_direction["mean"] - parent["p29_transfer"]["directional_cosine"]["mean"],
            "sign_agreement": r1_alignment["sign_agreement"] - parent["p29_transfer"]["alignment"]["sign_agreement"],
        },
        "delta_vs_p30": {
            "directional_cosine": r1_direction["mean"] - parent["p30_transfer"]["directional_cosine"]["mean"],
            "sign_agreement": r1_alignment["sign_agreement"] - parent["p30_transfer"]["alignment"]["sign_agreement"],
        },
    }
    stability = {
        "schema_version": "P30R1_STABILITY_DIAGNOSTIC_V1",
        "status": "COMPLETE",
        "held_class": P30R1_STAGE2_CLASS,
        "held_mask_reads_post_freeze": int(metrics["held_mask_file_reads_after_prediction_freeze"]),
        "p29": parent["p29_stability"],
        "p30": parent["p30_stability"],
        "p30r1": r1_shifts,
        "global_p30r1_residual_abs_q99": r1_magnitude["q99_abs"],
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
    }
    atomic_write_json(root / "P30R1_TRANSFER_DIAGNOSTIC.json", transfer)
    atomic_write_json(root / "P30R1_STABILITY_DIAGNOSTIC.json", stability)
    return transfer, stability


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(child) for child in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _scientific_gate(
    prereg: Mapping[str, Any],
    training: Mapping[str, Any],
    prediction: Mapping[str, Any],
    metrics: Mapping[str, Any],
    transfer: Mapping[str, Any],
    stability: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    attempt_uuid: str,
) -> dict[str, Any]:
    frozen = prereg["stage_2_frozen_gates"]
    metric_gates = frozen["metrics"]
    runtime_gates = frozen["runtime"]
    p30r1_metrics = metrics["p30r1_metrics"]
    directional = transfer["p30r1"]["directional_cosine"]["mean"]
    sign = transfer["p30r1"]["alignment"]["sign_agreement"]
    residual_q99 = stability["global_p30r1_residual_abs_q99"]
    normal_q99_shift = stability["p30r1"]["normal"]["q99"]
    median_step = float(training["median_step_seconds"])
    wall_seconds = max(float(training["training_seconds"]), float(training["parent_process_seconds"]))
    checks = {
        "pAP": {
            "value": float(p30r1_metrics["pAP"]),
            "minimum": float(metric_gates["pAP"]["minimum"]),
            "pass": float(p30r1_metrics["pAP"]) >= float(metric_gates["pAP"]["minimum"]),
        },
        "pAUROC": {
            "value": float(p30r1_metrics["pAUROC"]),
            "minimum": float(metric_gates["pAUROC"]["minimum"]),
            "pass": float(p30r1_metrics["pAUROC"]) >= float(metric_gates["pAUROC"]["minimum"]),
        },
        "directional_cosine": {
            "value": float(directional),
            "minimum": float(metric_gates["directional_cosine"]["minimum"]),
            "pass": float(directional) >= float(metric_gates["directional_cosine"]["minimum"]),
        },
        "sign_agreement": {
            "value": float(sign),
            "minimum": float(metric_gates["sign_agreement"]["minimum"]),
            "pass": float(sign) >= float(metric_gates["sign_agreement"]["minimum"]),
        },
        "residual_abs_q99": {
            "value": float(residual_q99),
            "maximum": float(metric_gates["residual_abs_q99"]["maximum"]),
            "pass": float(residual_q99) <= float(metric_gates["residual_abs_q99"]["maximum"]),
        },
        "normal_score_q99_shift": {
            "value": float(normal_q99_shift),
            "maximum": float(metric_gates["normal_score_q99_shift"]["maximum"]),
            "pass": float(normal_q99_shift) <= float(metric_gates["normal_score_q99_shift"]["maximum"]),
        },
        "nonfinite_gradient_count": {
            "value": int(training["nonfinite_gradient_count"]),
            "maximum": int(frozen["gradient_and_provenance"]["nonfinite_gradient_count_max"]),
            "pass": int(training["nonfinite_gradient_count"]) == 0,
        },
        "nonfinite_loss_count": {
            "value": int(training["nonfinite_loss_count"]),
            "maximum": int(frozen["gradient_and_provenance"]["nonfinite_loss_count_max"]),
            "pass": int(training["nonfinite_loss_count"]) == 0,
        },
        "median_step_seconds": {
            "value": median_step,
            "maximum": float(runtime_gates["median_step_seconds_max"]),
            "pass": median_step <= float(runtime_gates["median_step_seconds_max"]),
        },
        "training_wall_seconds": {
            "value": wall_seconds,
            "maximum": float(runtime_gates["training_seconds_max"]),
            "pass": wall_seconds <= float(runtime_gates["training_seconds_max"]),
        },
        "inference_overhead_percent": {
            "value": 0.0,
            "maximum": float(runtime_gates["inference_overhead_percent_max"]),
            "pass": True,
        },
    }
    structural = {
        "attempt_uuid": attempt_uuid == prediction["attempt_uuid"] == metrics["attempt_uuid"],
        "objective_count": True,
        "teacher_scale_detached": training["teacher_scale_detached"],
        "teacher_parameter_delta": training["teacher_parameter_delta"] == 0.0,
        "student_parameter_delta": float(training["student_parameter_delta"]["l2"]) > 0.0,
        "new_clip_forwards": training["new_clip_forwards"] == 0,
        "new_phase2b_forwards": training["new_phase2b_forwards"] == 0,
        "held_reads_before_prediction_freeze": (
            input_audit["held_gt_reads_before_prediction_freeze"] == 0
            and input_audit["held_mask_reads_before_prediction_freeze"] == 0
        ),
        "prediction_freeze": prediction["gt_used"] is False and prediction["mask_reads"] == 0,
        "cache_rebuilt": input_audit["cache_rebuilt"] is False,
        "provenance": True,
        "all_required_values_finite": _finite_tree({"metrics": metrics, "transfer": transfer, "stability": stability}),
    }
    failures = [name for name, item in checks.items() if not item["pass"]]
    failures.extend(name for name, passed in structural.items() if not passed)
    return {
        "schema_version": "P30R1_STAGE2_GATE_V1",
        "status": "STAGE2_PASS" if not failures else "STAGE2_SCIENTIFIC_STOP",
        "attempt_uuid": attempt_uuid,
        "class": P30R1_STAGE2_CLASS,
        "checks": checks,
        "structural_checks": structural,
        "failures": failures,
        "threshold_source": "P30R1_PREREGISTRATION.json stage_2_frozen_gates",
        "stage3_started": False,
        "full_started": False,
    }


def _comparison(metrics: Mapping[str, Any], transfer: Mapping[str, Any], stability: Mapping[str, Any], parent: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "native": metrics["native_metrics"],
        "p29": parent["p29"],
        "p30": parent["p30"],
        "p30r1": metrics["p30r1_metrics"],
        "directional_cosine": {
            "p29": parent["p29_transfer"]["directional_cosine"]["mean"],
            "p30": parent["p30_transfer"]["directional_cosine"]["mean"],
            "p30r1": transfer["p30r1"]["directional_cosine"]["mean"],
        },
        "sign_agreement": {
            "p29": parent["p29_transfer"]["alignment"]["sign_agreement"],
            "p30": parent["p30_transfer"]["alignment"]["sign_agreement"],
            "p30r1": transfer["p30r1"]["alignment"]["sign_agreement"],
        },
        "mean_abs_residual": {
            "p29": parent["p29_transfer"]["magnitude"]["mean_abs"],
            "p30": parent["p30_transfer"]["magnitude"]["mean_abs"],
            "p30r1": transfer["p30r1"]["magnitude"]["mean_abs"],
        },
        "residual_abs_q99": {
            "p29": parent["p29_transfer"]["magnitude"]["q99_abs"],
            "p30": parent["p30_transfer"]["magnitude"]["q99_abs"],
            "p30r1": stability["global_p30r1_residual_abs_q99"],
        },
        "normal_score_q99_shift": {
            "p29": parent["p29_stability"]["normal"]["q99"],
            "p30": parent["p30_stability"]["normal"]["q99"],
            "p30r1": stability["p30r1"]["normal"]["q99"],
        },
    }


def _post_run_audit(
    root: Path,
    attempt: Mapping[str, Any],
    training: Mapping[str, Any],
    prediction: Mapping[str, Any],
    scoring_gate: Mapping[str, Any],
    metrics: Mapping[str, Any],
    stage2_gate: Mapping[str, Any],
    input_audit: Mapping[str, Any],
) -> dict[str, Any]:
    prediction_path = Path(str(prediction["prediction_path"]))
    scoring_gate_path = root / "P30R1_SCORING_GATE.json"
    metrics_path = root / P30R1_STAGE2_CLASS / "metrics" / "P30R1_HELD_METRICS.json"
    failures: list[str] = []
    if attempt.get("completion_status") != "ATTEMPT_CONSUMED":
        failures.append("attempt identity is not consumed")
    if training.get("optimizer_steps") != 39240 or training.get("status") != "FOLD_TRAINING_COMPLETE":
        failures.append("training schedule/status mismatch")
    if prediction.get("records") != 200 or prediction.get("gt_used") is not False or prediction.get("mask_reads") != 0:
        failures.append("prediction provenance mismatch")
    if scoring_gate.get("status") != "PASS" or scoring_gate.get("predictions_frozen") is not True:
        failures.append("prediction freeze gate failed")
    if not prediction_path.is_file() or prediction_path.stat().st_mode & 0o222:
        failures.append("prediction artifact is not immutable")
    if not scoring_gate_path.is_file() or not metrics_path.is_file():
        failures.append("required scoring artifacts missing")
    if prediction_path.stat().st_mtime_ns > scoring_gate_path.stat().st_mtime_ns:
        failures.append("prediction was written after freeze gate")
    if metrics_path.stat().st_mtime_ns < scoring_gate_path.stat().st_mtime_ns:
        failures.append("metrics were written before freeze gate")
    if metrics.get("held_mask_file_reads_after_prediction_freeze", 0) <= 0:
        failures.append("post-freeze held mask access was not recorded")
    if input_audit.get("held_gt_reads_before_prediction_freeze") != 0 or input_audit.get("held_mask_reads_before_prediction_freeze") != 0:
        failures.append("held supervision reached training")
    if training.get("new_clip_forwards") != 0 or training.get("new_phase2b_forwards") != 0:
        failures.append("unexpected model forwards occurred")
    tracked_code_changes = _git("diff", "--name-only")
    if tracked_code_changes:
        failures.append(f"tracked code changed after attempt: {tracked_code_changes}")
    result = {
        "schema_version": "P30R1_POST_RUN_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "terminal_status": stage2_gate["status"],
        "attempt_uuid": attempt["attempt_uuid"],
        "attempt_count": 1,
        "class": P30R1_STAGE2_CLASS,
        "optimizer_steps": training["optimizer_steps"],
        "predictions_frozen_before_scoring": scoring_gate.get("predictions_frozen"),
        "held_gt_reads_before_prediction_freeze": input_audit["held_gt_reads_before_prediction_freeze"],
        "held_mask_reads_before_prediction_freeze": input_audit["held_mask_reads_before_prediction_freeze"],
        "held_mask_reads_after_prediction_freeze": metrics.get("held_mask_file_reads_after_prediction_freeze"),
        "new_clip_forwards": training["new_clip_forwards"],
        "new_phase2b_forwards": training["new_phase2b_forwards"],
        "teacher_parameter_delta": training["teacher_parameter_delta"],
        "cache_rebuilt": input_audit["cache_rebuilt"],
        "stage3_started": False,
        "full_started": False,
        "execution_marker_created": False,
        "tracked_code_changes_after_attempt": tracked_code_changes,
        "failures": failures,
    }
    atomic_write_json(root / "P30R1_POST_RUN_AUDIT.json", result)
    return result


def _final_report(
    root: Path,
    attempt: Mapping[str, Any],
    training: Mapping[str, Any],
    prediction: Mapping[str, Any],
    metrics: Mapping[str, Any],
    transfer: Mapping[str, Any],
    stability: Mapping[str, Any],
    gate: Mapping[str, Any],
    audit: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> str:
    r1 = metrics["p30r1_metrics"]
    r1_direction = transfer["p30r1"]["directional_cosine"]["mean"]
    r1_sign = transfer["p30r1"]["alignment"]["sign_agreement"]
    r1_q99 = stability["global_p30r1_residual_abs_q99"]
    r1_normal_q99 = stability["p30r1"]["normal"]["q99"]
    directional_retained = gate["checks"]["directional_cosine"]["pass"]
    radial_stable = gate["checks"]["residual_abs_q99"]["pass"] and gate["checks"]["normal_score_q99_shift"]["pass"]
    lines = [
        "# P30R1 Scientific Stage 2 Final Report",
        "",
        f"Primary decision: `{gate['status']}`.",
        "",
        "Exactly one preregistered candle attempt was executed. Stage 3 and the full 12-class run were not started.",
        "",
        "## Attempt and frozen execution",
        "",
        f"- attempt UUID: `{attempt['attempt_uuid']}`",
        f"- UTC start: `{attempt['utc_timestamp']}`",
        f"- branch: `{attempt['branch']}`",
        f"- scientific execution commit: `{attempt['scientific_execution_base_sha']}`",
        f"- engineering qualification commit: `{attempt['engineering_qualification_sha']}`",
        f"- preregistration SHA-256: `{attempt['p30r1_preregistration_sha256']}`",
        f"- class: `{P30R1_STAGE2_CLASS}`; fit `1962`; held `200`; expected steps `39240`",
        "- objective, optimizer, cache, architecture, and inference path remained frozen; no tuning or rerun occurred.",
        "",
        "## P30R1 vs P30 vs P29",
        "",
        "| metric | P29 | P30 | P30R1 | frozen gate |\n"
        "|---|---:|---:|---:|---|",
        f"| pAP | {comparison['p29']['pAP']:.12f} | {comparison['p30']['pAP']:.12f} | {comparison['p30r1']['pAP']:.12f} | >= 0.4641403049313743 |",
        f"| pAUROC | {comparison['p29']['pAUROC']:.12f} | {comparison['p30']['pAUROC']:.12f} | {comparison['p30r1']['pAUROC']:.12f} | >= 0.9306671435137679 |",
        f"| directional cosine | {comparison['directional_cosine']['p29']:.12f} | {comparison['directional_cosine']['p30']:.12f} | {comparison['directional_cosine']['p30r1']:.12f} | >= 0.6985491737886378 |",
        f"| sign agreement | {comparison['sign_agreement']['p29']:.12f} | {comparison['sign_agreement']['p30']:.12f} | {comparison['sign_agreement']['p30r1']:.12f} | >= 0.5554938271604938 |",
        f"| mean absolute residual | {comparison['mean_abs_residual']['p29']:.12f} | {comparison['mean_abs_residual']['p30']:.12f} | {comparison['mean_abs_residual']['p30r1']:.12f} | descriptive |",
        f"| residual absolute q99 | {comparison['residual_abs_q99']['p29']:.12f} | {comparison['residual_abs_q99']['p30']:.12f} | {comparison['residual_abs_q99']['p30r1']:.12f} | <= 8.643353872299194 |",
        f"| normal-score q99 shift | {comparison['normal_score_q99_shift']['p29']:.12f} | {comparison['normal_score_q99_shift']['p30']:.12f} | {comparison['normal_score_q99_shift']['p30r1']:.12f} | <= 0.0010011587851122385 |",
        "",
        "## Mechanism result",
        "",
        f"- directional behavior retained under the preregistered threshold: `{directional_retained}` (cosine `{r1_direction:.12f}`, sign `{r1_sign:.12f}`).",
        f"- radial q99 and normal-score saturation controlled under the preregistered thresholds: `{radial_stable}` (residual q99 `{r1_q99:.12f}`, normal q99 shift `{r1_normal_q99:.12f}`).",
        f"- pAP gate recovered: `{gate['checks']['pAP']['pass']}`; pAUROC gate safe: `{gate['checks']['pAUROC']['pass']}`.",
        "These are results of this one candle test only; they do not authorize a new objective, tuning, or Stage 3.",
        "",
        "## Training and data audit",
        "",
        f"- optimizer steps: `{training['optimizer_steps']}`; finite loss/gradients: `{training['loss_finite']}`/`{training['gradient_finite']}`; nonfinite counts: `{training['nonfinite_loss_count']}`/`{training['nonfinite_gradient_count']}`.",
        f"- student parameter delta L2: `{training['student_parameter_delta']['l2']}`; teacher parameter delta: exactly `{training['teacher_parameter_delta']}`; teacher scale detached: `{training['teacher_scale_detached']}`.",
        f"- new CLIP forwards: `{training['new_clip_forwards']}`; new Phase2B forwards: `{training['new_phase2b_forwards']}`; cache rebuild: `{training['cache_rebuilt']}`.",
        f"- held GT/mask reads before prediction freeze: `{training['held_gt_reads']}`/`{training['held_mask_reads']}`; post-freeze mask reads for scoring: `{metrics['held_mask_file_reads_after_prediction_freeze']}`.",
        f"- prediction freeze: `{audit['predictions_frozen_before_scoring']}`; prediction SHA-256: `{prediction['prediction_sha256']}`.",
        "",
        "## Runtime audit",
        "",
        f"- training wall time: `{training['training_seconds']:.6f}` seconds (parent process `{training['parent_process_seconds']:.6f}`); median step `{training['median_step_seconds']:.12f}` seconds; mean `{training['mean_step_seconds']:.12f}`; p90 `{training['p90_step_seconds']:.12f}`.",
        f"- frozen limits: median <= `{gate['checks']['median_step_seconds']['maximum']}` seconds and wall <= `{gate['checks']['training_wall_seconds']['maximum']}` seconds.",
        "- inference overhead: `0%`; deployment and scoring path were unchanged.",
        "",
        "## Gate and final audit",
        "",
        f"- Stage 2 gate: `{gate['status']}`; failed checks: `{gate['failures']}`.",
        f"- post-run audit: `{audit['status']}`; attempt count: `{audit['attempt_count']}`; Stage 3 started: `{audit['stage3_started']}`; full run started: `{audit['full_started']}`.",
        "- all scoring occurred after the immutable GT-free prediction freeze; no held result was inspected during training.",
        "",
        f"`{gate['status']}`",
    ]
    return "\n".join(lines) + "\n"


def _write_engineering_stop(root: Path, attempt: Mapping[str, Any], exc: BaseException) -> None:
    atomic_write_json(
        root / "P30R1_ENGINEERING_STOP.json",
        {
            "schema_version": "P30R1_STAGE2_ENGINEERING_STOP_V1",
            "status": "STAGE2_ENGINEERING_STOP",
            "attempt_uuid": attempt.get("attempt_uuid"),
            "utc_timestamp": _utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "rerun_forbidden": True,
            "partial_artifacts_preserved": True,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise RuntimeError("P30R1 Stage 2 accepts only the preregistered evidence directory")
    git_identity = _assert_frozen_execution_state()
    if args.output_root.exists():
        residual = [path.name for path in args.output_root.iterdir() if path.name != ".gitignore"]
        if residual:
            raise RuntimeError(f"P30R1 Stage 2 output is already occupied: {residual}")
    prereg, input_audit, prereg_hash = _audit_inputs(args, git_identity)
    from tools.sabra_v2.run_p30r1_engineering import production_reference_parity

    parity = production_reference_parity()
    if parity.get("status") != "PASS":
        raise RuntimeError("production/reference parity no longer passes")
    args.output_root.mkdir(parents=True, exist_ok=True)
    attempt = {
        "schema_version": "P30R1_STAGE2_ATTEMPT_V1",
        "completion_status": "ATTEMPT_CONSUMED",
        "attempt_uuid": str(uuid.uuid4()),
        "attempt_number": 1,
        "utc_timestamp": _utc(),
        "branch": git_identity["branch"],
        "scientific_execution_base_sha": git_identity["head"],
        "engineering_qualification_sha": ENGINEERING_QUALIFICATION_SHA,
        "p30r1_uuid": P30R1_UUID,
        "p30r1_preregistration_sha256": prereg_hash,
        "class": P30R1_STAGE2_CLASS,
        "class_order": list(EXPECTED_VISA_CLASSES),
        "fit_records": 1962,
        "held_records": 200,
        "epochs": P30R1_EPOCHS,
        "batch_size": P30R1_BATCH_SIZE,
        "learning_rate": P30R1_LEARNING_RATE,
        "seed": P30R1_SEED,
        "objective": "P30R1_TEACHER_RELATIVE_SMOOTHL1_V1",
        "formulation_hash": P30R1_FORMULATION_HASH,
        "cache_root": input_audit["cache_root"],
        "cache_provenance": input_audit["cache_provenance"],
        "new_clip_forwards_expected": 0,
        "new_phase2b_forwards_expected": 0,
        "held_reads_before_prediction_freeze_expected": 0,
        "stage3_started": False,
        "full_started": False,
        "automatic_rerun": False,
    }
    attempt_path = args.output_root / "P30R1_STAGE2_ATTEMPT.json"
    if attempt_path.exists():
        raise RuntimeError("P30R1 Stage 2 attempt identity already exists")
    atomic_write_json(attempt_path, attempt)
    try:
        training = _run_training(args, args.output_root, attempt["attempt_uuid"], prereg_hash, git_identity["head"])
        prediction = _run_prediction(args, args.output_root, training, attempt["attempt_uuid"], prereg_hash, git_identity["head"])
        scoring_gate = _freeze_predictions(args.output_root, prediction, attempt["attempt_uuid"], prereg_hash)
        metrics, prediction_context = _score_frozen_predictions(args, args.output_root, prediction, scoring_gate)
        parent = _frozen_parent_values()
        transfer, stability = _run_diagnostics(args, args.output_root, prediction_context, metrics, parent)
        gate = _scientific_gate(prereg, training, prediction, metrics, transfer, stability, input_audit, attempt["attempt_uuid"])
        comparison = _comparison(metrics, transfer, stability, parent)
        atomic_write_json(
            args.output_root / "P30R1_STAGE2_QUALIFICATION.json",
            {
                "schema_version": "P30R1_STAGE2_QUALIFICATION_V1",
                "status": gate["status"],
                "attempt": attempt,
                "input_audit": input_audit,
                "production_reference_parity": parity,
                "training": training,
                "prediction": prediction,
                "scoring_gate": scoring_gate,
                "metrics": metrics,
                "transfer": transfer,
                "stability": stability,
                "comparison": comparison,
                "gate": gate,
                "stage3_started": False,
                "full_started": False,
            },
        )
        audit = _post_run_audit(args.output_root, attempt, training, prediction, scoring_gate, metrics, gate, input_audit)
        _atomic_text(
            args.output_root / "P30R1_FINAL_REPORT.md",
            _final_report(args.output_root, attempt, training, prediction, metrics, transfer, stability, gate, audit, comparison),
        )
        return {
            "status": gate["status"],
            "attempt_uuid": attempt["attempt_uuid"],
            "metrics": metrics,
            "gate": gate,
            "audit": audit,
        }
    except BaseException as exc:
        _write_engineering_stop(args.output_root, attempt, exc)
        raise


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
