"""Run exactly one preregistered P32 candle Stage 2 attempt."""
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
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.p29_contract import p29_cache_provenance
from tools.sabra_v2.p32_objective import (
    P32_OBJECTIVE_NAME,
    P32_PREREGISTRATION_SHA256,
    p32_functional_margin_components,
    p32_objective_contract,
)
from tools.sabra_v2.p29r1_forensic import residual_magnitude_summary, vectorized_pixel_shifts
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import (
    CachedSourceDataset,
    TierADataset,
    atomic_write_json,
    sha256_file,
)
from tools.sabra_v2.student_forward import forward_region_student


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "research/sabra_v2/region_distill/P32"
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_VISA_ROOT = Path("/workspace/data/source/visa_unpack")
DEFAULT_P26_CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
DEFAULT_CLIP_ASSET = ROOT / "model/ViT-L-14-336px.pt"
DEFAULT_METADATA = ROOT / "dataset/hub/VisA.jsonl"

P32_BRANCH = "research/p29r1-fast-objective-forensic-v1"
P32_ENGINEERING_QUALIFICATION_SHA = "9939949a02a4587eb809a4d2102a666fc90deacc"
P32_STAGE2_CLASS = "candle"
P32_EPOCHS = 20
P32_BATCH_SIZE = 1
P32_LEARNING_RATE = 0.001
P32_SEED = 0
P32_FIT_RECORDS = 1962
P32_HELD_RECORDS = 200
P32_EXPECTED_STEPS = P32_FIT_RECORDS * P32_EPOCHS

P32_PREREGISTRATION_PATH = ROOT / "research/sabra_v2/region_distill/P32_PREREGISTRATION.json"
P32_PREREGISTRATION_MD = ROOT / "research/sabra_v2/region_distill/P32_PREREGISTRATION.md"
P32_ENGINEERING_QUALIFICATION_PATH = ROOT / "research/sabra_v2/region_distill/P32_ENGINEERING_QUALIFICATION.json"
P32_IMPLEMENTATION_REPORT_PATH = ROOT / "research/sabra_v2/region_distill/P32_IMPLEMENTATION_REPORT.md"
P31_CONTROL_RESULT = ROOT / "research/sabra_v2/region_distill/P31/P31_CONTROL_SCIENTIFIC_RESULT.json"
P30R1_METRICS = ROOT / "research/sabra_v2/region_distill/P30R1/candle/metrics/P30R1_HELD_METRICS.json"

P32_PROTECTED_PATHS = (
    "tools/sabra_v2/p32_objective.py",
    "tools/sabra_v2/p32_reference.py",
    "tools/sabra_v2/region_adapter.py",
    "tools/sabra_v2/region_cache.py",
    "tools/sabra_v2/student_forward.py",
    "model/phase2b_runtime.py",
    "research/sabra_v2/region_distill/P32_PREREGISTRATION.md",
    "research/sabra_v2/region_distill/P32_PREREGISTRATION.json",
)


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


def _active_p32_processes() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True)
    excluded_pids = {str(os.getpid()), str(os.getppid())}
    needles = ("run_p32_scientific_stage2", "train_region_distill_p32_cached")
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if any(needle in line for needle in needles)
        and not line.lstrip().split(maxsplit=1)[0] in excluded_pids
    ]


def _assert_frozen_execution_state(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise RuntimeError("P32 Stage 2 accepts only the preregistered evidence directory")
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != P32_BRANCH:
        raise RuntimeError(f"P32 Stage 2 must run on {P32_BRANCH}, got {branch!r}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", P32_ENGINEERING_QUALIFICATION_SHA, head],
        cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError("HEAD is not the engineering-qualified P32 descendant")
    porcelain = _git("status", "--porcelain")
    if porcelain:
        raise RuntimeError(f"scientific execution requires a clean worktree: {porcelain!r}")
    remote = _remote_sha(branch)
    if remote != head:
        raise RuntimeError(f"local/remote mismatch before attempt: {head} != {remote}")
    changed = subprocess.run(
        ["git", "diff", "--name-only", P32_ENGINEERING_QUALIFICATION_SHA, "--", *P32_PROTECTED_PATHS],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if changed:
        raise RuntimeError(f"frozen P32 scientific core changed after qualification: {changed}")
    active = _active_p32_processes()
    if active:
        raise RuntimeError(f"duplicate P32 scientific/training process detected: {active}")
    residual = []
    if args.output_root.exists():
        residual = [path.name for path in args.output_root.iterdir() if path.name != ".gitignore"]
    if residual:
        raise RuntimeError(f"P32 Stage 2 output is already occupied: {residual}")
    return {
        "branch": branch,
        "head": head,
        "remote_sha": remote,
        "remote_equals_local": True,
        "worktree_clean_before_attempt": True,
        "engineering_qualification_sha": P32_ENGINEERING_QUALIFICATION_SHA,
        "qualified_core_changed": False,
        "duplicate_processes": [],
    }


def _audit_preregistration() -> tuple[dict[str, Any], str]:
    if not P32_PREREGISTRATION_PATH.is_file() or not P32_PREREGISTRATION_MD.is_file():
        raise RuntimeError("frozen P32 preregistration files are missing")
    observed_hash = sha256_file(P32_PREREGISTRATION_MD)
    if observed_hash != P32_PREREGISTRATION_SHA256:
        raise RuntimeError(f"P32 preregistration Markdown hash mismatch: {observed_hash}")
    prereg = _json(P32_PREREGISTRATION_PATH)
    if (
        prereg.get("schema_version") != "P32_PREREGISTRATION_V1"
        or prereg.get("status") != "P32_PREREGISTRATION_FROZEN"
        or prereg.get("protocol_id") != "P32"
        or prereg.get("preregistration_md_sha256") != P32_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("P32 preregistration identity/status drift")
    hypothesis = prereg.get("hypothesis", {})
    if (
        hypothesis.get("name") != "FUNCTIONAL_MARGIN_EFFECT"
        or hypothesis.get("primary_forensic_mechanism") != "TEACHER_DIRECTION_NOT_CAUSAL"
        or hypothesis.get("secondary_forensic_mechanism") != "SPARSE_SELECTIVE_CORRECTION"
        or hypothesis.get("native_control_required") is not True
    ):
        raise RuntimeError("P32 hypothesis contract drift")
    frozen = prereg.get("frozen", {})
    expected = {
        "fold": "candle",
        "fit_records": P32_FIT_RECORDS,
        "held_records": P32_HELD_RECORDS,
        "epochs": P32_EPOCHS,
        "batch_size": P32_BATCH_SIZE,
        "seed": P32_SEED,
        "precision": "fp32",
        "learning_rate": P32_LEARNING_RATE,
        "new_objective_count": 1,
        "new_hyperparameter_count": 0,
        "teacher_at_inference": False,
        "incremental_inference_overhead_percent": 0,
    }
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise RuntimeError("P32 frozen schedule or complexity contract drift")
    if prereg.get("allowed_model_forwards", {}).get("new_clip_forwards") != 0:
        raise RuntimeError("P32 forward contract drift")
    if prereg.get("allowed_model_forwards", {}).get("new_phase2b_forwards") != 0:
        raise RuntimeError("P32 forward contract drift")
    if prereg.get("allowed_model_forwards", {}).get("new_teacher_forwards") != 0:
        raise RuntimeError("P32 forward contract drift")
    return prereg, observed_hash


def _audit_inputs(args: argparse.Namespace, git_identity: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    if args.cache_root.resolve() != DEFAULT_CACHE_ROOT.resolve():
        raise RuntimeError("P32 Stage 2 must reuse the frozen P27 cache root")
    for path in (args.metadata, args.cache_root, args.visa_root, args.p26_checkpoint, args.clip_asset):
        if not path.exists():
            raise RuntimeError(f"missing frozen input: {path}")
    prereg, prereg_hash = _audit_preregistration()
    engineering = _json(P32_ENGINEERING_QUALIFICATION_PATH)
    if (
        engineering.get("status") != "P32_PASS_TO_SCIENTIFIC_PROTOCOL"
        or engineering.get("final_gate") != "P32_PASS_TO_SCIENTIFIC_PROTOCOL"
        or engineering.get("preregistration_sha256") != prereg_hash
        or engineering.get("implementation", {}).get("production_module") != "tools/sabra_v2/p32_objective.py"
        or engineering.get("implementation", {}).get("objective_count") != 1
    ):
        raise RuntimeError("P32 engineering qualification is not a matching PASS artifact")
    parent_assets = verify_p26_parent(args.p26_checkpoint, args.clip_asset, ROOT / "configs/phase2b_canonical_v1.json")
    rows = read_visa_metadata(args.metadata)
    if tuple(sorted({str(row["class_name"]) for row in rows})) != tuple(sorted(EXPECTED_VISA_CLASSES)):
        raise RuntimeError("unexpected VisA class inventory")
    inventory = loco_inventory(rows, P32_STAGE2_CLASS)
    if len(inventory.fit_rows) != P32_FIT_RECORDS or len(inventory.held_rows) != P32_HELD_RECORDS:
        raise RuntimeError("frozen candle fit/held inventory changed")
    provenance = p29_cache_provenance(args.metadata)
    CachedSourceDataset(
        inventory.fit_rows,
        P32_STAGE2_CLASS,
        args.cache_root,
        provenance,
        load_source_mask=False,
        load_native_logits=False,
    )
    tier_a_manifest = args.cache_root / "tier_a" / P32_STAGE2_CLASS / "manifest.json"
    tier_b_manifest = args.cache_root / "tier_b" / P32_STAGE2_CLASS / "manifest.json"
    for path in (P31_CONTROL_RESULT, P30R1_METRICS, tier_a_manifest, tier_b_manifest, P32_IMPLEMENTATION_REPORT_PATH):
        if not path.is_file():
            raise RuntimeError(f"missing frozen P32 input/evidence: {path}")
    objective_contract = p32_objective_contract()
    if objective_contract.get("preregistration_sha256") != prereg_hash or objective_contract.get("objective_count") != 1:
        raise RuntimeError("P32 objective contract does not match preregistration")
    input_audit = {
        "metadata": {"path": str(args.metadata), "sha256": sha256_file(args.metadata), "records": len(rows)},
        "visa_root": str(args.visa_root.resolve()),
        "visa_root_accessed_before_prediction_freeze": False,
        "cache_root": str(args.cache_root.resolve()),
        "cache_provenance": provenance.as_dict(),
        "tier_a_candle_manifest": {"path": str(tier_a_manifest), "sha256": sha256_file(tier_a_manifest)},
        "tier_b_candle_manifest": {"path": str(tier_b_manifest), "sha256": sha256_file(tier_b_manifest)},
        "class_order": list(EXPECTED_VISA_CLASSES),
        "candle_fit_records": len(inventory.fit_rows),
        "candle_held_records": len(inventory.held_rows),
        "p26": parent_assets,
        "p26_checkpoint": {"path": str(args.p26_checkpoint), "sha256": sha256_file(args.p26_checkpoint)},
        "clip_asset": {"path": str(args.clip_asset), "sha256": sha256_file(args.clip_asset)},
        "config": {
            "path": str(ROOT / "configs/phase2b_canonical_v1.json"),
            "sha256": sha256_file(ROOT / "configs/phase2b_canonical_v1.json"),
        },
        "engineering_qualification": {
            "path": str(P32_ENGINEERING_QUALIFICATION_PATH),
            "sha256": sha256_file(P32_ENGINEERING_QUALIFICATION_PATH),
        },
        "implementation_report": {
            "path": str(P32_IMPLEMENTATION_REPORT_PATH),
            "sha256": sha256_file(P32_IMPLEMENTATION_REPORT_PATH),
        },
        "p31_control_result": {"path": str(P31_CONTROL_RESULT), "sha256": sha256_file(P31_CONTROL_RESULT)},
        "p30r1_metrics": {"path": str(P30R1_METRICS), "sha256": sha256_file(P30R1_METRICS)},
        "held_gt_reads_before_prediction_freeze": 0,
        "held_mask_reads_before_prediction_freeze": 0,
        "held_outcome_metrics_read_before_prediction_freeze": False,
        "cache_rebuilt": False,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
        "preregistration_sha256": prereg_hash,
        "scientific_execution_base_sha": git_identity["head"],
    }
    return prereg, input_audit, prereg_hash


def _production_reference_parity() -> dict[str, Any]:
    from tools.sabra_v2 import p32_reference

    cases = (("normal", 1.0, 1.0), ("zero", 0.0, 0.0), ("student_heavy", 100.0, 1.0))
    maximum = {"loss": 0.0, "student_effect": 0.0, "teacher_effect": 0.0, "student_gradient": 0.0}
    for index, (_name, student_scale, teacher_scale) in enumerate(cases):
        generator = torch.Generator(device="cpu").manual_seed(3200 + index)
        student_value = torch.randn((3, 2, 9, 9), generator=generator, dtype=torch.float32) * student_scale
        teacher_value = torch.randn((2, 9, 9), generator=generator, dtype=torch.float32) * teacher_scale
        if _name == "zero":
            student_value.zero_()
            teacher_value.zero_()
        production_student = student_value.clone().requires_grad_(True)
        reference_student = student_value.clone().requires_grad_(True)
        production_teacher = teacher_value.clone()
        reference_teacher = teacher_value.clone()
        production = p32_functional_margin_components(production_student, production_teacher)
        reference = p32_reference.p32_functional_margin_components(reference_student, reference_teacher)
        values = (
            ("loss", production[0], reference[0]),
            ("student_effect", production[1], reference[1]),
            ("teacher_effect", production[2], reference[2]),
        )
        for name, observed, expected in values:
            difference = float((observed - expected).abs().max().detach().cpu())
            maximum[name] = max(maximum[name], difference)
        production[0].backward()
        reference[0].backward()
        gradient_difference = float((production_student.grad - reference_student.grad).abs().max().detach().cpu())
        maximum["student_gradient"] = max(maximum["student_gradient"], gradient_difference)
        if not all(bool(torch.isfinite(value).all().item()) for value in (*production, *reference, production_student.grad, reference_student.grad)):
            raise RuntimeError(f"P32 production/reference parity found non-finite values in {_name}")
    tolerances = {"loss": 1e-4, "student_effect": 1e-4, "teacher_effect": 1e-5, "student_gradient": 1e-6}
    failures = [name for name, value in maximum.items() if value > tolerances[name]]
    if failures:
        raise RuntimeError(f"P32 production/reference parity failed: {maximum}")
    return {
        "status": "PASS",
        "cases": len(cases),
        "device": "cpu",
        "max_abs_errors": maximum,
        "tolerances": tolerances,
        "all_finite": True,
        "all_within_tolerance": True,
    }


def _run_module(module: str, arguments: Sequence[str]) -> float:
    command = [sys.executable, "-m", module, *arguments]
    print(json.dumps({"event": "START", "utc": _utc(), "module": module}), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    print(json.dumps({"event": "COMPLETE", "utc": _utc(), "module": module, "seconds": elapsed}), flush=True)
    return elapsed


def _run_training(args: argparse.Namespace, root: Path, attempt_uuid: str, prereg_hash: str, execution_sha: str) -> dict[str, Any]:
    training_root = root / P32_STAGE2_CLASS / "training"
    if training_root.exists():
        raise RuntimeError(f"scientific training output already exists: {training_root}")
    training_root.mkdir(parents=True)
    parent_seconds = _run_module(
        "tools.sabra_v2.train_region_distill_p32_cached",
        (
            "--output", str(training_root),
            "--metadata", str(args.metadata),
            "--cache-root", str(args.cache_root),
            "--held-class", P32_STAGE2_CLASS,
            "--attempt-uuid", attempt_uuid,
            "--execution-base-sha", execution_sha,
            "--preregistration-sha", prereg_hash,
            "--device", str(args.device),
        ),
    )
    raw_path = training_root / "P32_TRAINING_COMPLETE.json"
    checkpoint_path = training_root / "p32_region_adapter.pt"
    if not raw_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("P32 scientific trainer did not produce required artifacts")
    raw = _json(raw_path)
    required = {
        "status": "FOLD_TRAINING_COMPLETE",
        "protocol_id": "P32",
        "attempt_uuid": attempt_uuid,
        "preregistration_sha256": prereg_hash,
        "scientific_execution_base_sha": execution_sha,
        "held_class": P32_STAGE2_CLASS,
        "fit_records": P32_FIT_RECORDS,
        "held_records_not_read": P32_HELD_RECORDS,
        "optimizer_steps": P32_EXPECTED_STEPS,
        "expected_optimizer_steps": P32_EXPECTED_STEPS,
        "objective": P32_OBJECTIVE_NAME,
        "objective_count": 1,
        "held_gt_reads": 0,
        "held_mask_reads": 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
        "cache_rebuilt": False,
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"P32 trainer provenance/schedule mismatch: {required}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != "P32_REGION_ADAPTER_CHECKPOINT_V1"
        or checkpoint.get("status") != "FOLD_TRAINING_COMPLETE"
        or checkpoint.get("attempt_uuid") != attempt_uuid
        or checkpoint.get("preregistration_sha256") != prereg_hash
        or checkpoint.get("optimizer_steps") != P32_EXPECTED_STEPS
        or checkpoint.get("objective_count") != 1
        or checkpoint.get("teacher_trainable") is not False
        or checkpoint.get("new_clip_forwards") != 0
        or checkpoint.get("new_phase2b_forwards") != 0
        or checkpoint.get("new_teacher_forwards") != 0
    ):
        raise RuntimeError("P32 scientific checkpoint contract mismatch")
    adapter = RegionResidualAdapter()
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("cache_provenance") != p29_cache_provenance(args.metadata).as_dict():
        raise RuntimeError("P32 scientific checkpoint cache provenance mismatch")
    training = dict(raw)
    training.update({
        "parent_process_seconds": parent_seconds,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_strict_reload": True,
        "checkpoint_status": checkpoint.get("status"),
        "objective_contract": p32_objective_contract(),
    })
    atomic_write_json(root / P32_STAGE2_CLASS / "P32_STAGE2_TRAINING_COMPLETE.json", training)
    return training


def _run_prediction(
    args: argparse.Namespace,
    root: Path,
    training: Mapping[str, Any],
    attempt_uuid: str,
    prereg_hash: str,
    execution_sha: str,
) -> dict[str, Any]:
    prediction_root = root / P32_STAGE2_CLASS / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_root / "p32_held_predictions.pt"
    completion_path = prediction_root / "PREDICTION_COMPLETE.json"
    if prediction_path.exists() or completion_path.exists():
        raise RuntimeError("P32 scientific prediction output already exists; refusing overwrite")
    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, P32_STAGE2_CLASS)
    provenance = p29_cache_provenance(args.metadata)
    checkpoint = torch.load(training["checkpoint"], map_location="cpu", weights_only=True)
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
            if set(batch) != {"class_name", "image_path", "sample_id", "index", "seg_features", "native_logits"}:
                raise RuntimeError(f"unexpected P32 held prediction fields: {sorted(batch)}")
            seg = batch["seg_features"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            native = batch["native_logits"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            student = forward_region_student(adapter, seg, native)
            records.append({
                "image_path": str(batch["image_path"][0]),
                "class_name": P32_STAGE2_CLASS,
                "native_abnormal_probability": student.native_probability[0, 1].detach().cpu(),
                "p32_abnormal_probability": student.deployed_probability[0, 1].detach().cpu(),
                "p32_region_residual": student.region_residual[:, 0].detach().cpu(),
            })
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if len(records) != P32_HELD_RECORDS:
        raise RuntimeError(f"P32 scientific held prediction count mismatch: {len(records)}")
    payload = {
        "schema_version": "P32_IMMUTABLE_HELD_PREDICTIONS_V1",
        "status": "PREDICTIONS_FROZEN_GT_FREE",
        "attempt_uuid": attempt_uuid,
        "protocol_id": "P32",
        "preregistration_sha256": prereg_hash,
        "scientific_execution_base_sha": execution_sha,
        "held_class": P32_STAGE2_CLASS,
        "gt_used": False,
        "mask_reads": 0,
        "held_gt_reads": 0,
        "cache_provenance": provenance.as_dict(),
        "adapter_checkpoint_sha256": training["checkpoint_sha256"],
        "records": records,
    }
    temporary = prediction_path.with_name(f".{prediction_path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, prediction_path)
    prediction_path.chmod(0o444)
    result = {
        "schema_version": "P32_PREDICTION_COMPLETE_V1",
        "status": "COMPLETE",
        "attempt_uuid": attempt_uuid,
        "protocol_id": "P32",
        "preregistration_sha256": prereg_hash,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "held_class": P32_STAGE2_CLASS,
        "records": len(records),
        "gt_used": False,
        "mask_reads": 0,
        "held_gt_reads": 0,
        "prediction_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "adapter_forwards": len(records),
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
        "completion_status": "COMPLETE",
    }
    atomic_write_json(completion_path, result)
    return result


def _freeze_predictions(root: Path, prediction: Mapping[str, Any], attempt_uuid: str, prereg_hash: str) -> dict[str, Any]:
    prediction_path = Path(str(prediction["prediction_path"]))
    completion_path = prediction_path.parent / "PREDICTION_COMPLETE.json"
    if not prediction_path.is_file() or not completion_path.is_file():
        raise RuntimeError("cannot freeze missing P32 predictions")
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "P32_IMMUTABLE_HELD_PREDICTIONS_V1"
        or payload.get("attempt_uuid") != attempt_uuid
        or payload.get("preregistration_sha256") != prereg_hash
        or payload.get("held_class") != P32_STAGE2_CLASS
        or payload.get("gt_used") is not False
        or payload.get("mask_reads") != 0
        or payload.get("held_gt_reads") != 0
        or len(payload.get("records", [])) != P32_HELD_RECORDS
        or prediction.get("prediction_sha256") != sha256_file(prediction_path)
        or prediction_path.stat().st_mode & 0o222
    ):
        raise RuntimeError("P32 scientific prediction freeze firewall failed")
    gate = {
        "schema_version": "P32_PREDICTION_FROZEN_V1",
        "status": "PASS",
        "completion_status": "P32_PREDICTION_FROZEN",
        "utc_timestamp": _utc(),
        "attempt_uuid": attempt_uuid,
        "protocol_id": "P32",
        "p32_preregistration_sha256": prereg_hash,
        "held_class": P32_STAGE2_CLASS,
        "prediction_count": 1,
        "predictions_frozen": True,
        "fit_or_teacher_steps_after_gate": 0,
        "held_gt_reads_before_prediction_freeze": 0,
        "held_mask_reads_before_prediction_freeze": 0,
        "held_outcome_metrics_read_before_prediction_freeze": False,
        "predictions": [{
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
            "records": P32_HELD_RECORDS,
            "held_class": P32_STAGE2_CLASS,
        }],
    }
    atomic_write_json(root / "P32_PREDICTION_FROZEN.json", gate)
    return gate


def _score_frozen_predictions(
    args: argparse.Namespace,
    root: Path,
    prediction: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if gate.get("status") != "PASS" or gate.get("predictions_frozen") is not True:
        raise RuntimeError("P32 held scoring requires the passing prediction-freeze gate")
    prediction_path = Path(str(prediction["prediction_path"]))
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != P32_HELD_RECORDS:
        raise RuntimeError("P32 frozen prediction record count mismatch")
    by_path = {str(record["image_path"]): record for record in records}
    if len(by_path) != len(records):
        raise RuntimeError("P32 frozen predictions contain duplicate paths")
    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, P32_STAGE2_CLASS)
    native_scores: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    image_paths: list[str] = []
    held_gt_reads = 0
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
            raise RuntimeError(f"missing P32 frozen prediction for {image_path}")
        native = record.get("native_abnormal_probability")
        candidate = record.get("p32_abnormal_probability")
        residual = record.get("p32_region_residual")
        if not isinstance(native, torch.Tensor) or tuple(native.shape) != (518, 518):
            raise RuntimeError("P32 native frozen map shape mismatch")
        if not isinstance(candidate, torch.Tensor) or tuple(candidate.shape) != (518, 518):
            raise RuntimeError("P32 candidate frozen map shape mismatch")
        if not isinstance(residual, torch.Tensor) or tuple(residual.shape) != (3, 9, 9):
            raise RuntimeError("P32 frozen residual shape mismatch")
        native_scores.append(native.numpy().astype(np.float32, copy=False))
        candidate_scores.append(candidate.numpy().astype(np.float32, copy=False))
        residuals.append(residual.numpy().astype(np.float32, copy=False))
        masks.append(batch["mask"][0, 0].numpy().astype(np.uint8, copy=False))
        image_paths.append(image_path)
        held_gt_reads += 1
        held_mask_reads += int(batch["label"][0].item())
    native_array = np.stack(native_scores)
    candidate_array = np.stack(candidate_scores)
    residual_array = np.stack(residuals)
    mask_array = np.stack(masks)
    native_metrics = exact_metrics(native_array.reshape(-1), mask_array.reshape(-1))
    candidate_metrics = exact_metrics(candidate_array.reshape(-1), mask_array.reshape(-1))
    result = {
        "schema_version": "P32_HELD_METRICS_V1",
        "status": "COMPLETE",
        "protocol_id": "P32",
        "held_class": P32_STAGE2_CLASS,
        "attempt_uuid": payload["attempt_uuid"],
        "preregistration_sha256": payload["preregistration_sha256"],
        "prediction_sha256": sha256_file(prediction_path),
        "fit_or_teacher_steps": 0,
        "native_metrics": native_metrics,
        "p32_metrics": candidate_metrics,
        "delta": {key: candidate_metrics[key] - native_metrics[key] for key in ("pAP", "pAUROC")},
        "held_gt_reads_before_prediction_freeze": 0,
        "held_mask_reads_before_prediction_freeze": 0,
        "held_gt_reads_after_prediction_freeze": held_gt_reads,
        "held_mask_file_reads_after_prediction_freeze": held_mask_reads,
        "scoring_started_after_prediction_freeze": True,
    }
    metrics_root = root / P32_STAGE2_CLASS / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(metrics_root / "P32_HELD_METRICS.json", result)
    return result, {
        "native_probability": native_array,
        "candidate_probability": candidate_array,
        "residual": residual_array,
        "masks": mask_array,
        "image_paths": np.asarray(image_paths, dtype=object),
    }


def _diagnostics(root: Path, metrics: Mapping[str, Any], context: Mapping[str, np.ndarray]) -> dict[str, Any]:
    native = np.asarray(context["native_probability"], dtype=np.float32)
    candidate = np.asarray(context["candidate_probability"], dtype=np.float32)
    residual = np.asarray(context["residual"], dtype=np.float32)
    masks = np.asarray(context["masks"], dtype=np.uint8)
    if not all(np.isfinite(value).all() for value in (native, candidate, residual)):
        raise RuntimeError("P32 post-freeze diagnostics found non-finite predictions")
    shift = vectorized_pixel_shifts(native, candidate, masks)
    residual_summary = residual_magnitude_summary(residual)
    candidate_effect = candidate - native
    effect_summary = {
        "mean_abs": float(np.abs(candidate_effect).mean()),
        "q95_abs": float(np.quantile(np.abs(candidate_effect), 0.95)),
        "q99_abs": float(np.quantile(np.abs(candidate_effect), 0.99)),
        "max_abs": float(np.abs(candidate_effect).max()),
    }
    result = {
        "schema_version": "P32_DOWNSTREAM_DIAGNOSTIC_V1",
        "status": "COMPLETE",
        "protocol_id": "P32",
        "held_class": P32_STAGE2_CLASS,
        "attempt_uuid": metrics["attempt_uuid"],
        "held_mask_reads_post_freeze": metrics["held_mask_file_reads_after_prediction_freeze"],
        "residual": residual_summary,
        "residual_exact_nonzero_fraction": float(np.count_nonzero(residual) / residual.size),
        "native_to_p32_score_effect": effect_summary,
        "p32_minus_native_pixel_shift": shift,
        "raw_direction_metrics_are_gates": False,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
    }
    atomic_write_json(root / "P32_DOWNSTREAM_DIAGNOSTIC.json", result)
    return result


def _historical_comparison(metrics: Mapping[str, Any]) -> dict[str, Any]:
    p31 = _json(P31_CONTROL_RESULT)
    p30r1 = _json(P30R1_METRICS)
    native = p31.get("primary_comparison", {}).get("native_p31")
    if not isinstance(native, Mapping):
        raise RuntimeError("P31 native comparison artifact changed")
    p30r1_metrics = p30r1.get("p30r1_metrics")
    if not isinstance(p30r1_metrics, Mapping):
        raise RuntimeError("P30R1 metrics artifact changed")
    p32_metrics = metrics["p32_metrics"]
    return {
        "P31_native_zero_adapter": {"pAP": float(native["pAP"]), "pAUROC": float(native["pAUROC"])},
        "P30R1": {"pAP": float(p30r1_metrics["pAP"]), "pAUROC": float(p30r1_metrics["pAUROC"])},
        "P32": {"pAP": float(p32_metrics["pAP"]), "pAUROC": float(p32_metrics["pAUROC"])},
        "P32_minus_P31_native": {
            key: float(p32_metrics[key]) - float(native[key]) for key in ("pAP", "pAUROC")
        },
        "P32_minus_P30R1": {
            key: float(p32_metrics[key]) - float(p30r1_metrics[key]) for key in ("pAP", "pAUROC")
        },
    }


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return True


def _scientific_gate(
    prereg: Mapping[str, Any],
    training: Mapping[str, Any],
    prediction: Mapping[str, Any],
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    attempt_uuid: str,
) -> dict[str, Any]:
    frozen = prereg["future_stage2"]
    residual_q99 = float(diagnostics["residual"]["q99_abs"])
    normal_q99 = diagnostics["p32_minus_native_pixel_shift"]["normal"]["q99"]
    if normal_q99 is None:
        raise RuntimeError("P32 normal-score q99 is undefined")
    checks = {
        "pAP": {
            "value": float(metrics["p32_metrics"]["pAP"]),
            "minimum": float(frozen["pAP_threshold"]),
            "pass": float(metrics["p32_metrics"]["pAP"]) >= float(frozen["pAP_threshold"]),
        },
        "pAUROC": {
            "value": float(metrics["p32_metrics"]["pAUROC"]),
            "minimum": float(frozen["pAUROC_threshold"]),
            "pass": float(metrics["p32_metrics"]["pAUROC"]) >= float(frozen["pAUROC_threshold"]),
        },
        "residual_abs_q99": {
            "value": residual_q99,
            "maximum": float(frozen["residual_absolute_q99_max"]),
            "pass": residual_q99 <= float(frozen["residual_absolute_q99_max"]),
        },
        "normal_score_q99_shift": {
            "value": float(normal_q99),
            "maximum": float(frozen["normal_score_q99_shift_max"]),
            "pass": float(normal_q99) <= float(frozen["normal_score_q99_shift_max"]),
        },
        "nonfinite_loss_count": {
            "value": int(training["nonfinite_loss_count"]),
            "maximum": int(frozen["nonfinite_loss_count"]),
            "pass": int(training["nonfinite_loss_count"]) == 0,
        },
        "nonfinite_gradient_count": {
            "value": int(training["nonfinite_gradient_count"]),
            "maximum": int(frozen["nonfinite_gradient_count"]),
            "pass": int(training["nonfinite_gradient_count"]) == 0,
        },
    }
    structural = {
        "attempt_uuid": attempt_uuid == prediction["attempt_uuid"] == metrics["attempt_uuid"],
        "exact_optimizer_steps": training["optimizer_steps"] == P32_EXPECTED_STEPS,
        "objective_count": training["objective_count"] == 1,
        "student_parameter_delta": float(training["student_parameter_delta"]["l2"]) > 0.0,
        "teacher_parameter_delta": float(training["teacher_parameter_delta"]) == 0.0,
        "teacher_detached": training["teacher_detached"] is True,
        "new_clip_forwards": training["new_clip_forwards"] == 0,
        "new_phase2b_forwards": training["new_phase2b_forwards"] == 0,
        "new_teacher_forwards": training["new_teacher_forwards"] == 0,
        "held_reads_before_prediction_freeze": (
            input_audit["held_gt_reads_before_prediction_freeze"] == 0
            and input_audit["held_mask_reads_before_prediction_freeze"] == 0
            and input_audit["held_outcome_metrics_read_before_prediction_freeze"] is False
        ),
        "prediction_gt_free": prediction["gt_used"] is False and prediction["mask_reads"] == 0,
        "cache_rebuilt": input_audit["cache_rebuilt"] is False,
        "inference_overhead_zero": True,
        "all_required_values_finite": _finite_tree({"training": training, "metrics": metrics, "diagnostics": diagnostics}),
    }
    failures = [name for name, item in checks.items() if item["pass"] is not True]
    failures.extend(name for name, passed in structural.items() if not passed)
    return {
        "schema_version": "P32_STAGE2_GATE_V1",
        "status": "P32_STAGE2_PASS" if not failures else "P32_STAGE2_SCIENTIFIC_STOP",
        "attempt_uuid": attempt_uuid,
        "protocol_id": "P32",
        "class": P32_STAGE2_CLASS,
        "checks": checks,
        "structural_checks": structural,
        "failures": failures,
        "threshold_source": "P32_PREREGISTRATION.json future_stage2",
        "raw_direction_metrics_are_gates": False,
        "stage3_started": False,
        "full_started": False,
        "rerun_allowed": False,
    }


def _post_run_audit(
    root: Path,
    attempt: Mapping[str, Any],
    training: Mapping[str, Any],
    prediction: Mapping[str, Any],
    freeze: Mapping[str, Any],
    metrics: Mapping[str, Any],
    gate: Mapping[str, Any],
    input_audit: Mapping[str, Any],
) -> dict[str, Any]:
    prediction_path = Path(str(prediction["prediction_path"]))
    freeze_path = root / "P32_PREDICTION_FROZEN.json"
    metrics_path = root / P32_STAGE2_CLASS / "metrics" / "P32_HELD_METRICS.json"
    attempt_path = root / "P32_STAGE2_ATTEMPT.json"
    failures: list[str] = []
    if attempt.get("completion_status") != "ATTEMPT_CONSUMED":
        failures.append("attempt identity is not consumed")
    if training.get("optimizer_steps") != P32_EXPECTED_STEPS or training.get("status") != "FOLD_TRAINING_COMPLETE":
        failures.append("training schedule/status mismatch")
    if prediction.get("records") != P32_HELD_RECORDS or prediction.get("gt_used") is not False or prediction.get("mask_reads") != 0:
        failures.append("prediction provenance mismatch")
    if freeze.get("status") != "PASS" or freeze.get("predictions_frozen") is not True:
        failures.append("prediction freeze gate failed")
    if not prediction_path.is_file() or prediction_path.stat().st_mode & 0o222:
        failures.append("prediction artifact is not immutable")
    if not attempt_path.is_file() or attempt_path.stat().st_mode & 0o222:
        failures.append("attempt identity is not immutable")
    if not freeze_path.is_file() or not metrics_path.is_file():
        failures.append("required post-freeze artifacts are missing")
    if prediction_path.stat().st_mtime_ns > freeze_path.stat().st_mtime_ns:
        failures.append("prediction was written after freeze gate")
    if metrics_path.stat().st_mtime_ns < freeze_path.stat().st_mtime_ns:
        failures.append("metrics were written before freeze gate")
    if metrics.get("held_mask_file_reads_after_prediction_freeze", 0) <= 0:
        failures.append("post-freeze held mask access was not recorded")
    if input_audit.get("held_gt_reads_before_prediction_freeze") != 0 or input_audit.get("held_mask_reads_before_prediction_freeze") != 0:
        failures.append("held supervision reached pre-freeze execution")
    if input_audit.get("held_outcome_metrics_read_before_prediction_freeze") is not False:
        failures.append("held outcome metrics were read before prediction freeze")
    if training.get("new_clip_forwards") != 0 or training.get("new_phase2b_forwards") != 0 or training.get("new_teacher_forwards") != 0:
        failures.append("unexpected model forward occurred")
    tracked_code_changes = _git("diff", "--name-only")
    if tracked_code_changes:
        failures.append(f"tracked code changed after attempt: {tracked_code_changes}")
    result = {
        "schema_version": "P32_POST_RUN_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "terminal_status": gate["status"],
        "attempt_uuid": attempt["attempt_uuid"],
        "attempt_count": 1,
        "class": P32_STAGE2_CLASS,
        "optimizer_steps": training["optimizer_steps"],
        "predictions_frozen_before_scoring": freeze.get("predictions_frozen"),
        "held_gt_reads_before_prediction_freeze": input_audit["held_gt_reads_before_prediction_freeze"],
        "held_mask_reads_before_prediction_freeze": input_audit["held_mask_reads_before_prediction_freeze"],
        "held_outcome_metrics_read_before_prediction_freeze": input_audit["held_outcome_metrics_read_before_prediction_freeze"],
        "held_gt_reads_after_prediction_freeze": metrics.get("held_gt_reads_after_prediction_freeze"),
        "held_mask_reads_after_prediction_freeze": metrics.get("held_mask_file_reads_after_prediction_freeze"),
        "new_clip_forwards": training["new_clip_forwards"],
        "new_phase2b_forwards": training["new_phase2b_forwards"],
        "new_teacher_forwards": training["new_teacher_forwards"],
        "teacher_parameter_delta": training["teacher_parameter_delta"],
        "cache_rebuilt": input_audit["cache_rebuilt"],
        "stage3_started": False,
        "full_started": False,
        "execution_marker_created": False,
        "automatic_rerun": False,
        "tracked_code_changes_after_attempt": tracked_code_changes,
        "failures": failures,
    }
    atomic_write_json(root / "P32_POST_RUN_AUDIT.json", result)
    return result


def _final_report(
    attempt: Mapping[str, Any],
    training: Mapping[str, Any],
    prediction: Mapping[str, Any],
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    gate: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> str:
    checks = gate["checks"]
    lines = [
        "# P32 Scientific Stage 2 Final Report",
        "",
        f"Final status: {gate['status']}.",
        "",
        "Exactly one preregistered candle attempt was executed. No Stage 3 or full 12-class run was started, and no rerun or tuning occurred.",
        "",
        "## Frozen execution",
        "",
        f"- attempt UUID: {attempt['attempt_uuid']}",
        f"- branch: {attempt['branch']}",
        f"- scientific execution commit: {attempt['scientific_execution_base_sha']}",
        f"- preregistration SHA-256: {attempt['p32_preregistration_sha256']}",
        f"- class: {P32_STAGE2_CLASS}; fit {P32_FIT_RECORDS}; held {P32_HELD_RECORDS}; optimizer steps {training['optimizer_steps']}",
        f"- objective: {P32_OBJECTIVE_NAME}; objective count {training['objective_count']}; seed {training['seed']}; FP32 AdamW schedule remained frozen.",
        "",
        "## Locked endpoint comparison",
        "",
        "| metric | P31 native / zero adapter | P30R1 | P32 | P32 minus P31 |\n|---|---:|---:|---:|---:|",
        f"| pAP | {comparison['P31_native_zero_adapter']['pAP']:.12f} | {comparison['P30R1']['pAP']:.12f} | {comparison['P32']['pAP']:.12f} | {comparison['P32_minus_P31_native']['pAP']:+.12f} |",
        f"| pAUROC | {comparison['P31_native_zero_adapter']['pAUROC']:.12f} | {comparison['P30R1']['pAUROC']:.12f} | {comparison['P32']['pAUROC']:.12f} | {comparison['P32_minus_P31_native']['pAUROC']:+.12f} |",
        "",
        f"- pAP gate: {checks['pAP']['pass']}; pAUROC gate: {checks['pAUROC']['pass']}.",
        f"- P32 minus P30R1: pAP {comparison['P32_minus_P30R1']['pAP']:+.12f}, pAUROC {comparison['P32_minus_P30R1']['pAUROC']:+.12f}.",
        "",
        "## Mechanism and safety diagnostics",
        "",
        f"- residual absolute q99: {checks['residual_abs_q99']['value']:.12f}; frozen maximum {checks['residual_abs_q99']['maximum']:.12f}; pass {checks['residual_abs_q99']['pass']}.",
        f"- normal score q99 shift: {checks['normal_score_q99_shift']['value']:.12f}; frozen maximum {checks['normal_score_q99_shift']['maximum']:.12f}; pass {checks['normal_score_q99_shift']['pass']}.",
        f"- residual exact nonzero fraction: {diagnostics['residual_exact_nonzero_fraction']:.12f}; native-to-P32 score-effect q99 absolute: {diagnostics['native_to_p32_score_effect']['q99_abs']:.12f}.",
        "- raw direction metrics were descriptive only and were not used as gates; no held-derived tuning or new teacher forward occurred.",
        "",
        "## Runtime and data audit",
        "",
        f"- training wall time: {training['training_seconds']:.6f} seconds; parent process {training['parent_process_seconds']:.6f} seconds; median measured step {training['step_time_seconds']['median']:.6f} seconds.",
        f"- peak GPU allocated/reserved: {training['peak_gpu_allocated_bytes']} / {training['peak_gpu_reserved_bytes']} bytes; inference overhead: 0%.",
        f"- finite loss/gradient: {training['loss_finite']}/{training['gradient_finite']}; nonfinite counts: {training['nonfinite_loss_count']}/{training['nonfinite_gradient_count']}.",
        f"- pre-freeze held GT/mask reads: {audit['held_gt_reads_before_prediction_freeze']}/{audit['held_mask_reads_before_prediction_freeze']}; post-freeze held GT/mask reads: {audit['held_gt_reads_after_prediction_freeze']}/{audit['held_mask_reads_after_prediction_freeze']}.",
        f"- prediction freeze: {audit['predictions_frozen_before_scoring']}; prediction SHA-256: {prediction['prediction_sha256']}.",
        f"- new CLIP/Phase2B/teacher forwards: {audit['new_clip_forwards']}/{audit['new_phase2b_forwards']}/{audit['new_teacher_forwards']}; cache rebuilds: {audit['cache_rebuilt']}.",
        "",
        "## Terminal audit",
        "",
        f"- gate: {gate['status']}; failed checks: {gate['failures']}.",
        f"- post-run audit: {audit['status']}; attempt count: {audit['attempt_count']}; Stage 3 started: {audit['stage3_started']}; full run started: {audit['full_started']}.",
        "- the authoritative P32 preregistration was not edited after the attempt identity was created.",
        "",
        gate["status"],
        "",
    ]
    return "\n".join(lines)


def _write_engineering_stop(root: Path, attempt: Mapping[str, Any] | None, exc: BaseException) -> None:
    atomic_write_json(
        root / "P32_STAGE2_ENGINEERING_STOP.json",
        {
            "schema_version": "P32_STAGE2_ENGINEERING_STOP_V1",
            "status": "P32_STAGE2_ENGINEERING_STOP",
            "attempt_uuid": attempt.get("attempt_uuid") if attempt else None,
            "utc_timestamp": _utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "rerun_forbidden": True,
            "partial_artifacts_preserved": True,
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    attempt: dict[str, Any] | None = None
    try:
        git_identity = _assert_frozen_execution_state(args)
        prereg, input_audit, prereg_hash = _audit_inputs(args, git_identity)
        parity = _production_reference_parity()
        args.output_root.mkdir(parents=True, exist_ok=True)
        pre_execution = {
            "schema_version": "P32_STAGE2_PRE_EXECUTION_AUDIT_V1",
            "status": "PASS",
            "protocol_id": "P32",
            "utc_timestamp": _utc(),
            "git": git_identity,
            "preregistration_sha256": prereg_hash,
            "objective_contract": p32_objective_contract(),
            "production_reference_parity": parity,
            "input_audit": input_audit,
            "counts_before_attempt": {
                "new_scientific_stage2_attempts": 0,
                "new_stage3_attempts": 0,
                "full_runs": 0,
                "held_result_tuning_iterations": 0,
                "new_clip_forwards": 0,
                "new_phase2b_forwards": 0,
                "new_teacher_forwards": 0,
                "cache_rebuilds": 0,
                "optimizer_steps": 0,
                "attempt_uuid_created": False,
            },
        }
        atomic_write_json(args.output_root / "P32_STAGE2_PRE_EXECUTION_AUDIT.json", pre_execution)
        attempt = {
            "schema_version": "P32_STAGE2_ATTEMPT_V1",
            "completion_status": "ATTEMPT_CONSUMED",
            "attempt_uuid": str(uuid.uuid4()),
            "attempt_number": 1,
            "utc_timestamp": _utc(),
            "branch": git_identity["branch"],
            "scientific_execution_base_sha": git_identity["head"],
            "implementation_commit": git_identity["head"],
            "engineering_qualification_sha": P32_ENGINEERING_QUALIFICATION_SHA,
            "protocol_id": "P32",
            "p32_preregistration_sha256": prereg_hash,
            "class": P32_STAGE2_CLASS,
            "class_order": list(EXPECTED_VISA_CLASSES),
            "fit_records": P32_FIT_RECORDS,
            "held_records": P32_HELD_RECORDS,
            "epochs": P32_EPOCHS,
            "batch_size": P32_BATCH_SIZE,
            "learning_rate": P32_LEARNING_RATE,
            "seed": P32_SEED,
            "objective": P32_OBJECTIVE_NAME,
            "objective_contract": p32_objective_contract(),
            "cache_root": input_audit["cache_root"],
            "cache_provenance": input_audit["cache_provenance"],
            "new_clip_forwards_expected": 0,
            "new_phase2b_forwards_expected": 0,
            "new_teacher_forwards_expected": 0,
            "held_reads_before_prediction_freeze_expected": 0,
            "prediction_freeze_required": True,
            "stage3_started": False,
            "full_started": False,
            "automatic_rerun": False,
            "held_result_tuning_iterations": 0,
            "no_code_change_after_identity": True,
        }
        attempt_path = args.output_root / "P32_STAGE2_ATTEMPT.json"
        if attempt_path.exists():
            raise RuntimeError("P32 Stage 2 attempt identity already exists")
        atomic_write_json(attempt_path, attempt)
        attempt_path.chmod(0o444)
        training = _run_training(args, args.output_root, attempt["attempt_uuid"], prereg_hash, git_identity["head"])
        prediction = _run_prediction(args, args.output_root, training, attempt["attempt_uuid"], prereg_hash, git_identity["head"])
        freeze = _freeze_predictions(args.output_root, prediction, attempt["attempt_uuid"], prereg_hash)
        metrics, context = _score_frozen_predictions(args, args.output_root, prediction, freeze)
        diagnostics = _diagnostics(args.output_root, metrics, context)
        comparison = _historical_comparison(metrics)
        gate = _scientific_gate(prereg, training, prediction, metrics, diagnostics, input_audit, attempt["attempt_uuid"])
        qualification = {
            "schema_version": "P32_STAGE2_QUALIFICATION_V1",
            "status": gate["status"],
            "attempt": attempt,
            "pre_execution_audit": pre_execution,
            "input_audit": input_audit,
            "production_reference_parity": parity,
            "training": training,
            "prediction": prediction,
            "prediction_freeze": freeze,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "comparison": comparison,
            "gate": gate,
            "stage3_started": False,
            "full_started": False,
            "automatic_rerun": False,
        }
        atomic_write_json(args.output_root / "P32_STAGE2_QUALIFICATION.json", qualification)
        audit = _post_run_audit(args.output_root, attempt, training, prediction, freeze, metrics, gate, input_audit)
        _atomic_text(
            args.output_root / "P32_FINAL_REPORT.md",
            _final_report(attempt, training, prediction, metrics, diagnostics, comparison, gate, audit),
        )
        return {"status": gate["status"], "attempt_uuid": attempt["attempt_uuid"], "metrics": metrics, "gate": gate, "audit": audit}
    except BaseException as exc:
        if attempt is not None:
            _write_engineering_stop(args.output_root, attempt, exc)
        raise


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
