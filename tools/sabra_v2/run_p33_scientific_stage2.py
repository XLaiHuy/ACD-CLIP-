"""Run exactly one preregistered P33 candle Scientific Stage 2 attempt."""
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
from tools.sabra_v2.p33_objective import (
    P33_OBJECTIVE_NAME,
    P33_PREREGISTRATION_SHA256,
    p33_actionability_components,
    p33_objective_contract,
)
from tools.sabra_v2.p33_reference import p33_actionability_components as p33_reference_components
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
DEFAULT_OUTPUT_ROOT = ROOT / "research/sabra_v2/region_distill/P33"
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_VISA_ROOT = Path("/workspace/data/source/visa_unpack")
DEFAULT_P26_CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
DEFAULT_CLIP_ASSET = ROOT / "model/ViT-L-14-336px.pt"
DEFAULT_METADATA = ROOT / "dataset/hub/VisA.jsonl"

P33_BRANCH = "research/p29r1-fast-objective-forensic-v1"
P33_ENGINEERING_QUALIFICATION_COMMIT = "eef6877c692fcb0102c7fcab686d68c4a28f39f4"
P33_STAGE2_CLASS = "candle"
P33_EPOCHS = 20
P33_BATCH_SIZE = 1
P33_LEARNING_RATE = 0.001
P33_SEED = 0
P33_FIT_RECORDS = 1962
P33_HELD_RECORDS = 200
P33_EXPECTED_STEPS = P33_FIT_RECORDS * P33_EPOCHS
P33_SUPPORT_THRESHOLD = 0.0496010971069336

P33_PREREGISTRATION_PATH = ROOT / "research/sabra_v2/region_distill/P33_PREREGISTRATION.json"
P33_PREREGISTRATION_MD = ROOT / "research/sabra_v2/region_distill/P33_PREREGISTRATION.md"
P33_RESEARCH_DECISION = ROOT / "research/sabra_v2/region_distill/P33_RESEARCH_DECISION.json"
P33_PREFLIGHT = ROOT / "research/sabra_v2/region_distill/P33_PREFLIGHT_FALSIFICATION.json"
P33_ENGINEERING_QUALIFICATION = ROOT / "research/sabra_v2/region_distill/P33_ENGINEERING_QUALIFICATION.json"
P33_IMPLEMENTATION_REPORT = ROOT / "research/sabra_v2/region_distill/P33_IMPLEMENTATION_REPORT.md"
P33_SPEED_PROFILE = ROOT / "research/sabra_v2/region_distill/P33_SPEED_PROFILE.json"
P33_FORENSIC_ANALYSIS = ROOT / "research/sabra_v2/region_distill/P33_FORENSIC_ANALYSIS.json"

P31_CONTROL_RESULT = ROOT / "research/sabra_v2/region_distill/P31/P31_CONTROL_SCIENTIFIC_RESULT.json"
P30R1_METRICS = ROOT / "research/sabra_v2/region_distill/P30R1/candle/metrics/P30R1_HELD_METRICS.json"
P30R1_PREDICTIONS = ROOT / "research/sabra_v2/region_distill/P30R1/candle/predictions/p30r1_held_predictions.pt"
P32_METRICS = ROOT / "research/sabra_v2/region_distill/P32/candle/metrics/P32_HELD_METRICS.json"

P33_EXECUTION_HARNESS_PATHS = {
    "tools/sabra_v2/run_p33_scientific_stage2.py",
    "tools/sabra_v2/train_region_distill_p33_cached.py",
}


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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _active_p33_processes() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True)
    excluded = {str(os.getpid()), str(os.getppid())}
    needles = ("run_p33_scientific_stage2", "train_region_distill_p33_cached")
    active: list[str] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or fields[0] in excluded:
            continue
        if any(needle in fields[1] for needle in needles):
            active.append(line.strip())
    return active


def _assert_frozen_execution_state(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise RuntimeError("P33 Stage 2 accepts only the preregistered evidence directory")
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != P33_BRANCH:
        raise RuntimeError(f"P33 Stage 2 must run on {P33_BRANCH}, got {branch!r}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", P33_ENGINEERING_QUALIFICATION_COMMIT, head], cwd=ROOT
    ).returncode != 0:
        raise RuntimeError("HEAD is not the engineering-qualified P33 descendant")
    porcelain = _git("status", "--porcelain")
    if porcelain:
        raise RuntimeError(f"scientific execution requires a clean worktree: {porcelain!r}")
    remote = _remote_sha(branch)
    if remote != head:
        raise RuntimeError(f"local/remote mismatch before attempt: {head} != {remote}")
    changed = _git("diff", "--name-only", P33_ENGINEERING_QUALIFICATION_COMMIT, "--").splitlines()
    unexpected = sorted(set(changed) - P33_EXECUTION_HARNESS_PATHS)
    if unexpected:
        raise RuntimeError(f"P33 frozen scientific implementation changed: {unexpected}")
    active = _active_p33_processes()
    if active:
        raise RuntimeError(f"duplicate P33 scientific/training process detected: {active}")
    if args.output_root.exists():
        residual = [path.name for path in args.output_root.iterdir() if path.name != ".gitignore"]
        if residual:
            raise RuntimeError(f"P33 Stage 2 output is already occupied: {residual}")
    if (ROOT / "research/sabra_v2/region_distill/P33/P33_STAGE2_ATTEMPT.json").exists():
        raise RuntimeError("a P33 scientific attempt marker already exists")
    return {
        "branch": branch,
        "head": head,
        "remote_sha": remote,
        "remote_equals_local": True,
        "worktree_clean_before_attempt": True,
        "engineering_qualification_commit": P33_ENGINEERING_QUALIFICATION_COMMIT,
        "execution_harness_descendant_paths": sorted(P33_EXECUTION_HARNESS_PATHS),
        "unexpected_core_changes": [],
        "duplicate_processes": [],
        "attempt_count_before": 0,
    }


def _audit_preregistration() -> tuple[dict[str, Any], str]:
    if not P33_PREREGISTRATION_PATH.is_file() or not P33_PREREGISTRATION_MD.is_file():
        raise RuntimeError("frozen P33 preregistration files are missing")
    observed_hash = sha256_file(P33_PREREGISTRATION_MD)
    if observed_hash != P33_PREREGISTRATION_SHA256:
        raise RuntimeError(f"P33 preregistration Markdown hash mismatch: {observed_hash}")
    prereg = _json(P33_PREREGISTRATION_PATH)
    if (
        prereg.get("schema_version") != "P33_PREREGISTRATION_V1"
        or prereg.get("status") != "P33_PREREGISTRATION_FROZEN"
        or prereg.get("protocol_id") != "P33"
        or prereg.get("execution_authorized") is not False
        or prereg.get("scientific_uuid") is not None
        or prereg.get("execution_marker") is not None
        or prereg.get("preregistration_md_sha256") != P33_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("P33 preregistration identity/status drift")
    hypothesis = prereg.get("hypothesis", {})
    if (
        hypothesis.get("name") != "CONTINUOUS_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER"
        or hypothesis.get("primary_forensic_mechanism") != "LOST_SELECTIVITY_DENSE_MICRO_CORRECTION"
        or hypothesis.get("native_control_required") is not True
    ):
        raise RuntimeError("P33 hypothesis contract drift")
    mechanism = prereg.get("mechanism", {})
    if (
        mechanism.get("inherited_correction_scale_C") != 4.960109710693359
        or mechanism.get("objective") != "mean(weight * SmoothL1(student_effect, stop_gradient(teacher_effect), beta=1.0, reduction=none))"
        or mechanism.get("weight") != "stop_gradient(clamp(abs(teacher_effect)/C, min=0, max=1))"
        or mechanism.get("target_shrinkage") is not False
        or mechanism.get("hard_threshold") is not False
        or mechanism.get("auxiliary_terms") != []
    ):
        raise RuntimeError("P33 mathematical formulation drift")
    frozen = prereg.get("frozen", {})
    expected = {
        "fold": "candle",
        "fit_records": P33_FIT_RECORDS,
        "held_records": P33_HELD_RECORDS,
        "epochs": P33_EPOCHS,
        "batch_size": P33_BATCH_SIZE,
        "seed": P33_SEED,
        "precision": "fp32",
        "learning_rate": P33_LEARNING_RATE,
        "new_objective_count": 1,
        "new_hyperparameter_count": 0,
        "new_learnable_parameter_count": 0,
        "teacher_at_inference": False,
        "incremental_inference_overhead_percent": 0,
    }
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise RuntimeError("P33 frozen schedule or complexity contract drift")
    forwards = prereg.get("allowed_model_forwards", {})
    if any(forwards.get(key) != 0 for key in ("new_clip_forwards", "new_phase2b_forwards", "new_teacher_forwards")):
        raise RuntimeError("P33 forward contract drift")
    contract = p33_objective_contract()
    if contract.get("preregistration_sha256") != observed_hash or contract.get("objective_count") != 1:
        raise RuntimeError("P33 production objective contract drift")
    return prereg, observed_hash


def _production_reference_parity() -> dict[str, Any]:
    cases = (
        ("normal", 1.0, 1.0),
        ("zero", 0.0, 0.0),
        ("student_heavy", 100.0, 1.0),
        ("near_zero_teacher", 1.0, 0.01),
    )
    maximum = {"loss": 0.0, "student_effect": 0.0, "teacher_effect": 0.0, "weight": 0.0, "student_gradient": 0.0}
    for index, (name, student_scale, teacher_scale) in enumerate(cases):
        generator = torch.Generator(device="cpu").manual_seed(33000 + index)
        student_value = torch.randn((3, 2, 9, 9), generator=generator, dtype=torch.float32) * student_scale
        teacher_value = torch.randn((2, 9, 9), generator=generator, dtype=torch.float32) * teacher_scale
        if name == "zero":
            student_value.zero_()
            teacher_value.zero_()
        production_student = student_value.clone().requires_grad_(True)
        reference_student = student_value.clone().requires_grad_(True)
        production = p33_actionability_components(production_student, teacher_value.clone())
        reference = p33_reference_components(reference_student, teacher_value.clone())
        values = (
            ("loss", production[0], reference[0]),
            ("student_effect", production[1], reference[1]),
            ("teacher_effect", production[2], reference[2]),
            ("weight", production[3], reference[3]),
        )
        for quantity, observed, expected in values:
            maximum[quantity] = max(maximum[quantity], float((observed - expected).abs().max().detach().cpu()))
        production[0].backward()
        reference[0].backward()
        maximum["student_gradient"] = max(
            maximum["student_gradient"],
            float((production_student.grad - reference_student.grad).abs().max().detach().cpu()),
        )
        if not all(bool(torch.isfinite(value).all().item()) for value in (*production, *reference, production_student.grad, reference_student.grad)):
            raise RuntimeError(f"P33 production/reference parity found non-finite values in {name}")
    tolerances = {"loss": 1e-4, "student_effect": 1e-4, "teacher_effect": 1e-5, "weight": 1e-5, "student_gradient": 1e-6}
    failures = [name for name, value in maximum.items() if value > tolerances[name]]
    if failures:
        raise RuntimeError(f"P33 production/reference parity failed: {maximum}")
    return {
        "status": "PASS",
        "cases": len(cases),
        "device": "cpu",
        "max_abs_errors": maximum,
        "tolerances": tolerances,
        "all_finite": True,
        "all_within_tolerance": True,
    }


def _audit_inputs(args: argparse.Namespace, git_identity: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    if args.cache_root.resolve() != DEFAULT_CACHE_ROOT.resolve():
        raise RuntimeError("P33 Stage 2 must reuse the frozen P27 cache root")
    for path in (args.metadata, args.cache_root, args.visa_root, args.p26_checkpoint, args.clip_asset):
        if not path.exists():
            raise RuntimeError(f"missing frozen input: {path}")
    prereg, prereg_hash = _audit_preregistration()
    if not P33_PREFLIGHT.is_file() or _json(P33_PREFLIGHT).get("status") != "P33_PREFLIGHT_PASS":
        raise RuntimeError("P33 preflight artifact is not a matching PASS")
    engineering = _json(P33_ENGINEERING_QUALIFICATION)
    if (
        engineering.get("status") != "P33_PASS_TO_SCIENTIFIC_PROTOCOL"
        or engineering.get("final_gate") != "P33_PASS_TO_SCIENTIFIC_PROTOCOL"
        or engineering.get("preregistration_sha256") != prereg_hash
        or engineering.get("implementation", {}).get("production_module") != "tools/sabra_v2/p33_objective.py"
        or engineering.get("implementation", {}).get("objective_count") != 1
    ):
        raise RuntimeError("P33 engineering qualification is not a matching PASS artifact")
    parent_assets = verify_p26_parent(args.p26_checkpoint, args.clip_asset, ROOT / "configs/phase2b_canonical_v1.json")
    rows = read_visa_metadata(args.metadata)
    if tuple(sorted({str(row["class_name"]) for row in rows})) != tuple(sorted(EXPECTED_VISA_CLASSES)):
        raise RuntimeError("unexpected VisA class inventory")
    inventory = loco_inventory(rows, P33_STAGE2_CLASS)
    if len(inventory.fit_rows) != P33_FIT_RECORDS or len(inventory.held_rows) != P33_HELD_RECORDS:
        raise RuntimeError("frozen candle fit/held inventory changed")
    provenance = p29_cache_provenance(args.metadata)
    CachedSourceDataset(
        inventory.fit_rows,
        P33_STAGE2_CLASS,
        args.cache_root,
        provenance,
        load_source_mask=False,
        load_native_logits=False,
    )
    tier_a_manifest = args.cache_root / "tier_a" / P33_STAGE2_CLASS / "manifest.json"
    tier_b_manifest = args.cache_root / "tier_b" / P33_STAGE2_CLASS / "manifest.json"
    inherited = (P31_CONTROL_RESULT, P30R1_METRICS, P32_METRICS, P33_RESEARCH_DECISION, P33_FORENSIC_ANALYSIS)
    for path in (tier_a_manifest, tier_b_manifest, P33_IMPLEMENTATION_REPORT, P33_SPEED_PROFILE, *inherited):
        if not path.is_file():
            raise RuntimeError(f"missing frozen P33 input/evidence: {path}")
    p33_contract = p33_objective_contract()
    if p33_contract.get("preregistration_sha256") != prereg_hash or p33_contract.get("objective_count") != 1:
        raise RuntimeError("P33 objective contract does not match preregistration")
    input_audit: dict[str, Any] = {
        "metadata": {"path": str(args.metadata), "sha256": sha256_file(args.metadata), "records": len(rows)},
        "visa_root": str(args.visa_root.resolve()),
        "visa_root_accessed_before_prediction_freeze": False,
        "cache_root": str(args.cache_root.resolve()),
        "cache_provenance": provenance.as_dict(),
        "tier_a_candle_manifest": {"path": str(tier_a_manifest), "sha256": sha256_file(tier_a_manifest)},
        "tier_b_candle_manifest": {"path": str(tier_b_manifest), "sha256": sha256_file(tier_b_manifest)},
        "class_order": list(EXPECTED_VISA_CLASSES),
        "candle_fit_records": P33_FIT_RECORDS,
        "candle_held_records": P33_HELD_RECORDS,
        "p26": parent_assets,
        "p26_checkpoint": {"path": str(args.p26_checkpoint), "sha256": sha256_file(args.p26_checkpoint)},
        "clip_asset": {"path": str(args.clip_asset), "sha256": sha256_file(args.clip_asset)},
        "config": {"path": str(ROOT / "configs/phase2b_canonical_v1.json"), "sha256": sha256_file(ROOT / "configs/phase2b_canonical_v1.json")},
        "engineering_qualification": {"path": str(P33_ENGINEERING_QUALIFICATION), "sha256": sha256_file(P33_ENGINEERING_QUALIFICATION)},
        "implementation_report": {"path": str(P33_IMPLEMENTATION_REPORT), "sha256": sha256_file(P33_IMPLEMENTATION_REPORT)},
        "speed_profile": {"path": str(P33_SPEED_PROFILE), "sha256": sha256_file(P33_SPEED_PROFILE)},
        "p31_control_result": {"path": str(P31_CONTROL_RESULT), "sha256": sha256_file(P31_CONTROL_RESULT)},
        "p30r1_metrics": {"path": str(P30R1_METRICS), "sha256": sha256_file(P30R1_METRICS)},
        "p32_metrics": {"path": str(P32_METRICS), "sha256": sha256_file(P32_METRICS)},
        "p30r1_predictions": {"path": str(P30R1_PREDICTIONS), "sha256": sha256_file(P30R1_PREDICTIONS)},
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


def _run_module(module: str, arguments: Sequence[str]) -> float:
    command = [sys.executable, "-m", module, *arguments]
    print(json.dumps({"event": "START", "utc": _utc(), "module": module}), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    print(json.dumps({"event": "COMPLETE", "utc": _utc(), "module": module, "seconds": elapsed}), flush=True)
    return elapsed


def _run_training(args: argparse.Namespace, root: Path, attempt_uuid: str, prereg_hash: str, execution_sha: str) -> dict[str, Any]:
    training_root = root / P33_STAGE2_CLASS / "training"
    if training_root.exists():
        raise RuntimeError(f"scientific training output already exists: {training_root}")
    training_root.mkdir(parents=True)
    parent_seconds = _run_module(
        "tools.sabra_v2.train_region_distill_p33_cached",
        (
            "--output", str(training_root),
            "--metadata", str(args.metadata),
            "--cache-root", str(args.cache_root),
            "--held-class", P33_STAGE2_CLASS,
            "--attempt-uuid", attempt_uuid,
            "--execution-base-sha", execution_sha,
            "--preregistration-sha", prereg_hash,
            "--device", str(args.device),
        ),
    )
    raw_path = training_root / "P33_TRAINING_COMPLETE.json"
    checkpoint_path = training_root / "p33_region_adapter.pt"
    if not raw_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("P33 scientific trainer did not produce required artifacts")
    raw = _json(raw_path)
    required = {
        "status": "FOLD_TRAINING_COMPLETE",
        "protocol_id": "P33",
        "attempt_uuid": attempt_uuid,
        "preregistration_sha256": prereg_hash,
        "scientific_execution_base_sha": execution_sha,
        "held_class": P33_STAGE2_CLASS,
        "fit_records": P33_FIT_RECORDS,
        "held_records_not_read": P33_HELD_RECORDS,
        "optimizer_steps": P33_EXPECTED_STEPS,
        "expected_optimizer_steps": P33_EXPECTED_STEPS,
        "objective": P33_OBJECTIVE_NAME,
        "objective_count": 1,
        "held_gt_reads": 0,
        "held_mask_reads": 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
        "cache_rebuilt": False,
        "teacher_detached": True,
        "weight_detached": True,
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"P33 trainer provenance/schedule mismatch: {required}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != "P33_REGION_ADAPTER_CHECKPOINT_V1"
        or checkpoint.get("status") != "FOLD_TRAINING_COMPLETE"
        or checkpoint.get("attempt_uuid") != attempt_uuid
        or checkpoint.get("preregistration_sha256") != prereg_hash
        or checkpoint.get("optimizer_steps") != P33_EXPECTED_STEPS
        or checkpoint.get("objective_count") != 1
        or checkpoint.get("teacher_trainable") is not False
        or checkpoint.get("new_clip_forwards") != 0
        or checkpoint.get("new_phase2b_forwards") != 0
        or checkpoint.get("new_teacher_forwards") != 0
    ):
        raise RuntimeError("P33 scientific checkpoint contract mismatch")
    adapter = RegionResidualAdapter()
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("cache_provenance") != p29_cache_provenance(args.metadata).as_dict():
        raise RuntimeError("P33 scientific checkpoint cache provenance mismatch")
    training = dict(raw)
    training.update({
        "parent_process_seconds": parent_seconds,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_strict_reload": True,
        "checkpoint_status": checkpoint.get("status"),
        "objective_contract": p33_objective_contract(),
    })
    atomic_write_json(root / P33_STAGE2_CLASS / "P33_STAGE2_TRAINING_COMPLETE.json", training)
    return training


def _run_prediction(
    args: argparse.Namespace,
    root: Path,
    training: Mapping[str, Any],
    attempt_uuid: str,
    prereg_hash: str,
    execution_sha: str,
) -> dict[str, Any]:
    prediction_root = root / P33_STAGE2_CLASS / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_root / "p33_held_predictions.pt"
    completion_path = prediction_root / "PREDICTION_COMPLETE.json"
    if prediction_path.exists() or completion_path.exists():
        raise RuntimeError("P33 scientific prediction output already exists; refusing overwrite")
    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, P33_STAGE2_CLASS)
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
            expected_fields = {"class_name", "image_path", "sample_id", "index", "seg_features", "native_logits"}
            if set(batch) != expected_fields:
                raise RuntimeError(f"unexpected P33 held prediction fields: {sorted(batch)}")
            seg = batch["seg_features"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            native = batch["native_logits"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            student = forward_region_student(adapter, seg, native)
            records.append({
                "image_path": str(batch["image_path"][0]),
                "class_name": P33_STAGE2_CLASS,
                "native_abnormal_probability": student.native_probability[0, 1].detach().cpu(),
                "p33_abnormal_probability": student.deployed_probability[0, 1].detach().cpu(),
                "p33_region_residual": student.region_residual[:, 0].detach().cpu(),
            })
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if len(records) != P33_HELD_RECORDS:
        raise RuntimeError(f"P33 scientific held prediction count mismatch: {len(records)}")
    payload = {
        "schema_version": "P33_IMMUTABLE_HELD_PREDICTIONS_V1",
        "status": "PREDICTIONS_FROZEN_GT_FREE",
        "attempt_uuid": attempt_uuid,
        "protocol_id": "P33",
        "preregistration_sha256": prereg_hash,
        "scientific_execution_base_sha": execution_sha,
        "held_class": P33_STAGE2_CLASS,
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
        "schema_version": "P33_PREDICTION_COMPLETE_V1",
        "status": "COMPLETE",
        "attempt_uuid": attempt_uuid,
        "protocol_id": "P33",
        "preregistration_sha256": prereg_hash,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "held_class": P33_STAGE2_CLASS,
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
        raise RuntimeError("cannot freeze missing P33 predictions")
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "P33_IMMUTABLE_HELD_PREDICTIONS_V1"
        or payload.get("attempt_uuid") != attempt_uuid
        or payload.get("preregistration_sha256") != prereg_hash
        or payload.get("held_class") != P33_STAGE2_CLASS
        or payload.get("gt_used") is not False
        or payload.get("mask_reads") != 0
        or payload.get("held_gt_reads") != 0
        or len(payload.get("records", [])) != P33_HELD_RECORDS
        or prediction.get("prediction_sha256") != sha256_file(prediction_path)
        or prediction_path.stat().st_mode & 0o222
    ):
        raise RuntimeError("P33 scientific prediction freeze firewall failed")
    gate = {
        "schema_version": "P33_PREDICTION_FROZEN_V1",
        "status": "PASS",
        "completion_status": "P33_PREDICTION_FROZEN",
        "utc_timestamp": _utc(),
        "attempt_uuid": attempt_uuid,
        "protocol_id": "P33",
        "p33_preregistration_sha256": prereg_hash,
        "held_class": P33_STAGE2_CLASS,
        "prediction_count": 1,
        "predictions_frozen": True,
        "fit_or_teacher_steps_after_gate": 0,
        "held_gt_reads_before_prediction_freeze": 0,
        "held_mask_reads_before_prediction_freeze": 0,
        "held_outcome_metrics_read_before_prediction_freeze": False,
        "predictions": [{
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
            "records": P33_HELD_RECORDS,
            "held_class": P33_STAGE2_CLASS,
        }],
    }
    atomic_write_json(root / "P33_PREDICTION_FROZEN.json", gate)
    return gate


def _score_frozen_predictions(
    args: argparse.Namespace,
    root: Path,
    prediction: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if gate.get("status") != "PASS" or gate.get("predictions_frozen") is not True:
        raise RuntimeError("P33 held scoring requires the passing prediction-freeze gate")
    prediction_path = Path(str(prediction["prediction_path"]))
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != P33_HELD_RECORDS:
        raise RuntimeError("P33 frozen prediction record count mismatch")
    by_path = {str(record["image_path"]): record for record in records}
    if len(by_path) != len(records):
        raise RuntimeError("P33 frozen predictions contain duplicate paths")
    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, P33_STAGE2_CLASS)
    native_scores: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    image_paths: list[str] = []
    held_gt_reads = 0
    held_mask_reads = 0
    scoring_started = time.perf_counter()
    for batch in DataLoader(
        VisaEvaluationDataset(inventory.held_rows, args.visa_root),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    ):
        image_path = str(batch["image_path"][0])
        record = by_path.get(image_path)
        if record is None:
            raise RuntimeError(f"missing P33 frozen prediction for {image_path}")
        native = record.get("native_abnormal_probability")
        candidate = record.get("p33_abnormal_probability")
        residual = record.get("p33_region_residual")
        if not isinstance(native, torch.Tensor) or tuple(native.shape) != (518, 518):
            raise RuntimeError("P33 native frozen map shape mismatch")
        if not isinstance(candidate, torch.Tensor) or tuple(candidate.shape) != (518, 518):
            raise RuntimeError("P33 candidate frozen map shape mismatch")
        if not isinstance(residual, torch.Tensor) or tuple(residual.shape) != (3, 9, 9):
            raise RuntimeError("P33 frozen residual shape mismatch")
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
        "schema_version": "P33_HELD_METRICS_V1",
        "status": "COMPLETE",
        "protocol_id": "P33",
        "held_class": P33_STAGE2_CLASS,
        "attempt_uuid": payload["attempt_uuid"],
        "preregistration_sha256": payload["preregistration_sha256"],
        "prediction_sha256": sha256_file(prediction_path),
        "fit_or_teacher_steps": 0,
        "native_metrics": native_metrics,
        "p33_metrics": candidate_metrics,
        "delta": {key: candidate_metrics[key] - native_metrics[key] for key in ("pAP", "pAUROC")},
        "held_gt_reads_before_prediction_freeze": 0,
        "held_mask_reads_before_prediction_freeze": 0,
        "held_gt_reads_after_prediction_freeze": held_gt_reads,
        "held_mask_file_reads_after_prediction_freeze": held_mask_reads,
        "scoring_started_after_prediction_freeze": True,
        "scoring_seconds": time.perf_counter() - scoring_started,
    }
    metrics_root = root / P33_STAGE2_CLASS / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(metrics_root / "P33_HELD_METRICS.json", result)
    return result, {
        "native_probability": native_array,
        "candidate_probability": candidate_array,
        "residual": residual_array,
        "masks": mask_array,
        "image_paths": np.asarray(image_paths, dtype=object),
    }


def _concentration(values: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    total = float(absolute.sum())
    if total == 0.0:
        return {"effective_support": 0.0, "effective_support_fraction": 0.0, "gini": 0.0, "top_1_percent_mass": 0.0, "top_5_percent_mass": 0.0, "top_10_percent_mass": 0.0}
    ordered = np.sort(absolute)[::-1]
    effective = total * total / float(np.sum(absolute * absolute))
    n = ordered.size
    ascending = ordered[::-1]
    weighted = float(np.sum(np.arange(1, n + 1, dtype=np.float64) * ascending))
    gini = 2.0 * weighted / (n * total) - (n + 1.0) / n
    return {
        "effective_support": effective,
        "effective_support_fraction": effective / n,
        "gini": gini,
        "top_1_percent_mass": float(ordered[: max(1, math.ceil(0.01 * n))].sum() / total),
        "top_5_percent_mass": float(ordered[: max(1, math.ceil(0.05 * n))].sum() / total),
        "top_10_percent_mass": float(ordered[: max(1, math.ceil(0.10 * n))].sum() / total),
    }


def _support_overlap(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=bool).reshape(-1)
    right = np.asarray(right, dtype=bool).reshape(-1)
    intersection = left & right
    union = left | right
    return {
        "left_active_fraction": float(left.mean()),
        "right_active_fraction": float(right.mean()),
        "intersection_fraction": float(intersection.mean()),
        "union_fraction": float(union.mean()),
        "jaccard": float(intersection.sum() / union.sum()) if union.any() else 1.0,
        "left_contained_in_right": float(intersection.sum() / left.sum()) if left.any() else None,
        "right_contained_in_left": float(intersection.sum() / right.sum()) if right.any() else None,
    }


def _held_enrichment(delta: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(delta)
    area = float(masks.mean())
    result: dict[str, Any] = {"anomaly_area_fraction": area}
    for threshold in (P33_SUPPORT_THRESHOLD, 1e-4):
        active = absolute > threshold
        result[str(threshold)] = {
            "active_fraction": float(active.mean()),
            "anomaly_fraction_of_active": float(masks[active].mean()) if active.any() else 0.0,
            "anomaly_enrichment": float(masks[active].mean() / area) if active.any() and area else None,
        }
    total = float(absolute.sum())
    anomaly_mass = float(absolute[masks.astype(bool)].sum())
    result["absolute_effect_mass_fraction_in_anomaly"] = anomaly_mass / total if total else 0.0
    result["absolute_effect_mass_enrichment"] = (anomaly_mass / total) / area if total and area else None
    return result


def _diagnostics(
    root: Path,
    metrics: Mapping[str, Any],
    context: Mapping[str, np.ndarray],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    native = np.asarray(context["native_probability"], dtype=np.float32)
    candidate = np.asarray(context["candidate_probability"], dtype=np.float32)
    residual = np.asarray(context["residual"], dtype=np.float32)
    masks = np.asarray(context["masks"], dtype=np.uint8)
    if not all(np.isfinite(value).all() for value in (native, candidate, residual)):
        raise RuntimeError("P33 post-freeze diagnostics found non-finite predictions")
    candidate_effect = candidate - native
    shift = vectorized_pixel_shifts(native, candidate, masks)
    residual_abs = np.abs(residual)
    coordinate_support = residual_abs > P33_SUPPORT_THRESHOLD
    p30_payload = torch.load(P30R1_PREDICTIONS, map_location="cpu", weights_only=True)
    p30_records = p30_payload.get("records", [])
    if len(p30_records) != P33_HELD_RECORDS:
        raise RuntimeError("frozen P30R1 prediction count changed")
    p30_residual = np.stack([record["p30r1_region_residual"].numpy() for record in p30_records]).astype(np.float32)
    p30_support = np.abs(p30_residual) > P33_SUPPORT_THRESHOLD
    actionability = training.get("source_only_actionability", {})
    result: dict[str, Any] = {
        "schema_version": "P33_DOWNSTREAM_DIAGNOSTIC_V1",
        "status": "COMPLETE",
        "protocol_id": "P33",
        "held_class": P33_STAGE2_CLASS,
        "attempt_uuid": metrics["attempt_uuid"],
        "held_mask_reads_post_freeze": metrics["held_mask_file_reads_after_prediction_freeze"],
        "residual": residual_magnitude_summary(residual),
        "residual_exact_nonzero_fraction": float(np.count_nonzero(residual) / residual.size),
        "residual_support": {
            "threshold": P33_SUPPORT_THRESHOLD,
            "fraction_abs_gt_threshold": float(coordinate_support.mean()),
            "concentration": _concentration(residual),
        },
        "native_to_p33_score_effect": {
            "mean_abs": float(np.abs(candidate_effect).mean()),
            "median_abs": float(np.quantile(np.abs(candidate_effect), 0.50)),
            "q90_abs": float(np.quantile(np.abs(candidate_effect), 0.90)),
            "q95_abs": float(np.quantile(np.abs(candidate_effect), 0.95)),
            "q99_abs": float(np.quantile(np.abs(candidate_effect), 0.99)),
            "max_abs": float(np.abs(candidate_effect).max()),
            "near_zero_fraction_le_1e-10": float(np.mean(np.abs(candidate_effect) <= 1e-10)),
            "near_zero_fraction_le_1e-8": float(np.mean(np.abs(candidate_effect) <= 1e-8)),
            "concentration": _concentration(candidate_effect),
        },
        "p33_minus_native_pixel_shift": shift,
        "support_overlap_vs_p30r1": _support_overlap(p30_support, coordinate_support),
        "held_descriptive_enrichment": _held_enrichment(candidate_effect, masks),
        "source_only_actionability": actionability,
        "raw_direction_metrics_are_gates": False,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
    }
    atomic_write_json(root / "P33_DOWNSTREAM_DIAGNOSTIC.json", result)
    return result


def _historical_comparison(metrics: Mapping[str, Any]) -> dict[str, Any]:
    p31 = _json(P31_CONTROL_RESULT)
    p30r1 = _json(P30R1_METRICS)
    p32 = _json(P32_METRICS)
    native = p31.get("primary_comparison", {}).get("native_p31")
    p30_metrics = p30r1.get("p30r1_metrics")
    p32_metrics = p32.get("p32_metrics")
    if not isinstance(native, Mapping) or not isinstance(p30_metrics, Mapping) or not isinstance(p32_metrics, Mapping):
        raise RuntimeError("historical comparison artifact schema changed")
    p33_metrics = metrics["p33_metrics"]
    output: dict[str, Any] = {
        "P31_native_zero_adapter": {"pAP": float(native["pAP"]), "pAUROC": float(native["pAUROC"])},
        "P30R1": {"pAP": float(p30_metrics["pAP"]), "pAUROC": float(p30_metrics["pAUROC"])},
        "P32": {"pAP": float(p32_metrics["pAP"]), "pAUROC": float(p32_metrics["pAUROC"])},
        "P33": {"pAP": float(p33_metrics["pAP"]), "pAUROC": float(p33_metrics["pAUROC"])},
    }
    for method in ("P30R1", "P32", "P33"):
        output[f"{method}_minus_P31_native"] = {key: output[method][key] - output["P31_native_zero_adapter"][key] for key in ("pAP", "pAUROC")}
    output["P33_minus_P30R1"] = {key: output["P33"][key] - output["P30R1"][key] for key in ("pAP", "pAUROC")}
    output["P33_minus_P32"] = {key: output["P33"][key] - output["P32"][key] for key in ("pAP", "pAUROC")}
    output["native_consistency"] = {
        key: float(metrics["native_metrics"][key]) - output["P31_native_zero_adapter"][key] for key in ("pAP", "pAUROC")
    }
    return output


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
    freeze: Mapping[str, Any],
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    attempt_uuid: str,
) -> dict[str, Any]:
    frozen = prereg["future_stage2"]
    normal_q99 = diagnostics["p33_minus_native_pixel_shift"]["normal"]["q99"]
    if normal_q99 is None:
        raise RuntimeError("P33 normal-score q99 is undefined")
    checks = {
        "pAP": {
            "value": float(metrics["p33_metrics"]["pAP"]),
            "minimum": float(frozen["pAP_threshold"]),
            "pass": float(metrics["p33_metrics"]["pAP"]) >= float(frozen["pAP_threshold"]),
        },
        "pAUROC": {
            "value": float(metrics["p33_metrics"]["pAUROC"]),
            "minimum": float(frozen["pAUROC_threshold"]),
            "pass": float(metrics["p33_metrics"]["pAUROC"]) >= float(frozen["pAUROC_threshold"]),
        },
        "residual_abs_q99": {
            "value": float(diagnostics["residual"]["q99_abs"]),
            "maximum": float(frozen["residual_absolute_q99_max"]),
            "pass": float(diagnostics["residual"]["q99_abs"]) <= float(frozen["residual_absolute_q99_max"]),
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
        "attempt_uuid": attempt_uuid == prediction["attempt_uuid"] == metrics["attempt_uuid"] == freeze["attempt_uuid"],
        "exact_optimizer_steps": training["optimizer_steps"] == P33_EXPECTED_STEPS,
        "objective_count": training["objective_count"] == 1,
        "student_parameter_delta": float(training["student_parameter_delta"]["l2"]) > 0.0,
        "teacher_parameter_delta": float(training["teacher_parameter_delta"]) == 0.0,
        "teacher_detached": training["teacher_detached"] is True,
        "weight_detached": training["weight_detached"] is True,
        "new_clip_forwards": training["new_clip_forwards"] == 0,
        "new_phase2b_forwards": training["new_phase2b_forwards"] == 0,
        "new_teacher_forwards": training["new_teacher_forwards"] == 0,
        "held_reads_before_prediction_freeze": (
            input_audit["held_gt_reads_before_prediction_freeze"] == 0
            and input_audit["held_mask_reads_before_prediction_freeze"] == 0
            and input_audit["held_outcome_metrics_read_before_prediction_freeze"] is False
        ),
        "prediction_gt_free": prediction["gt_used"] is False and prediction["mask_reads"] == 0,
        "predictions_frozen": freeze["predictions_frozen"] is True,
        "cache_rebuilt": input_audit["cache_rebuilt"] is False,
        "inference_overhead_zero": True,
        "stage3_not_started": True,
        "full_run_not_started": True,
        "automatic_rerun": False,
        "all_required_values_finite": _finite_tree({"training": training, "metrics": metrics, "diagnostics": diagnostics}),
    }
    failures = [name for name, item in checks.items() if item["pass"] is not True]
    failures.extend(name for name, passed in structural.items() if not passed)
    return {
        "schema_version": "P33_STAGE2_GATE_V1",
        "status": "P33_STAGE2_PASS" if not failures else "P33_STAGE2_SCIENTIFIC_STOP",
        "attempt_uuid": attempt_uuid,
        "protocol_id": "P33",
        "class": P33_STAGE2_CLASS,
        "checks": checks,
        "structural_checks": structural,
        "failures": failures,
        "threshold_source": "P33_PREREGISTRATION.json future_stage2",
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
    freeze_path = root / "P33_PREDICTION_FROZEN.json"
    metrics_path = root / P33_STAGE2_CLASS / "metrics" / "P33_HELD_METRICS.json"
    attempt_path = root / "P33_STAGE2_ATTEMPT.json"
    failures: list[str] = []
    if attempt.get("completion_status") != "ATTEMPT_CONSUMED":
        failures.append("attempt identity is not consumed")
    if training.get("optimizer_steps") != P33_EXPECTED_STEPS or training.get("status") != "FOLD_TRAINING_COMPLETE":
        failures.append("training schedule/status mismatch")
    if prediction.get("records") != P33_HELD_RECORDS or prediction.get("gt_used") is not False or prediction.get("mask_reads") != 0:
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
        "schema_version": "P33_POST_RUN_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "terminal_status": gate["status"],
        "attempt_uuid": attempt["attempt_uuid"],
        "attempt_count": 1,
        "class": P33_STAGE2_CLASS,
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
        "automatic_rerun": False,
        "rerun_count": 0,
        "tracked_code_changes_after_attempt": tracked_code_changes,
        "failures": failures,
    }
    atomic_write_json(root / "P33_POST_RUN_AUDIT.json", result)
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
    p30_support = diagnostics["support_overlap_vs_p30r1"]["left_active_fraction"]
    p33_support = diagnostics["support_overlap_vs_p30r1"]["right_active_fraction"]
    p33_weight = diagnostics["source_only_actionability"].get("weight_summary", {})
    lines = [
        "# P33 Scientific Stage 2 Final Report",
        "",
        f"Final status: {gate['status']}.",
        "",
        "Exactly one preregistered candle attempt was executed. No rerun, tuning, Stage 3, subset expansion, or full run occurred.",
        "",
        "## Frozen execution",
        "",
        f"- attempt UUID: {attempt['attempt_uuid']}",
        f"- branch: {attempt['branch']}",
        f"- scientific execution commit: {attempt['scientific_execution_base_sha']}",
        f"- engineering qualification commit: {attempt['engineering_qualification_commit']}",
        f"- preregistration SHA-256: {attempt['p33_preregistration_sha256']}",
        f"- class/split: {P33_STAGE2_CLASS}; fit {P33_FIT_RECORDS}; held {P33_HELD_RECORDS}; optimizer steps {training['optimizer_steps']}",
        f"- objective: {P33_OBJECTIVE_NAME}; objective count {training['objective_count']}; seed {training['seed']}; FP32 AdamW schedule remained frozen.",
        "",
        "## Locked endpoint comparison",
        "",
        "| metric | P31/native | P30R1 | P32 | P33 | P33 minus native | P33 minus P32 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| pAP | {comparison['P31_native_zero_adapter']['pAP']:.12f} | {comparison['P30R1']['pAP']:.12f} | {comparison['P32']['pAP']:.12f} | {comparison['P33']['pAP']:.12f} | {comparison['P33_minus_P31_native']['pAP']:+.12f} | {comparison['P33_minus_P32']['pAP']:+.12f} |",
        f"| pAUROC | {comparison['P31_native_zero_adapter']['pAUROC']:.12f} | {comparison['P30R1']['pAUROC']:.12f} | {comparison['P32']['pAUROC']:.12f} | {comparison['P33']['pAUROC']:.12f} | {comparison['P33_minus_P31_native']['pAUROC']:+.12f} | {comparison['P33_minus_P32']['pAUROC']:+.12f} |",
        "",
        f"- P33 pAP gate: {checks['pAP']['pass']}; pAUROC gate: {checks['pAUROC']['pass']}.",
        "- P31/native, P30R1, and P32 values are frozen historical comparators; no historical result was used to alter P33 gates.",
        "",
        "## Selectivity and actionability result",
        "",
        f"- inherited-threshold residual support: P30R1 {p30_support:.12f}; P33 {p33_support:.12f}; P32 reference was 0.871481481481.",
        f"- P33/P30R1 support Jaccard: {diagnostics['support_overlap_vs_p30r1']['jaccard']:.12f}; P33 containment of P30R1: {diagnostics['support_overlap_vs_p30r1']['left_contained_in_right']!s}.",
        f"- P33 residual effective support fraction: {diagnostics['residual_support']['concentration']['effective_support_fraction']:.12f}; Gini: {diagnostics['residual_support']['concentration']['gini']:.12f}; top-10% mass: {diagnostics['residual_support']['concentration']['top_10_percent_mass']:.12f}.",
        f"- source-only actionability weights: mean {p33_weight.get('mean')}; median {p33_weight.get('q50')}; q90 {p33_weight.get('q90')}; q95 {p33_weight.get('q95')}; q99 {p33_weight.get('q99')}; exact-zero fraction {p33_weight.get('exact_zero_fraction')}; range [{p33_weight.get('min')}, {p33_weight.get('max')}].",
        f"- candidate score-effect q99 abs: {diagnostics['native_to_p33_score_effect']['q99_abs']:.12f}; normal-score q99 shift: {checks['normal_score_q99_shift']['value']:.12f}.",
        "- The mechanism is selective transfer through a bounded training-only weight; raw teacher-vector fidelity was not an objective or gate.",
        "",
        "## Mechanism answers",
        "",
        f"- Did P33 reduce intervention density relative to P32? {'yes' if p33_support < 0.871481481481 else 'no'} under the inherited residual-support diagnostic.",
        f"- Did it retain P30R1 support? {'yes' if diagnostics['support_overlap_vs_p30r1']['left_contained_in_right'] is not None and diagnostics['support_overlap_vs_p30r1']['left_contained_in_right'] > 0.5 else 'no/partial'} descriptively.",
        f"- Did pAP improve relative to P32? {'yes' if comparison['P33_minus_P32']['pAP'] > 0 else 'no'}.",
        f"- Did pAUROC recover toward or exceed native? {'yes' if comparison['P33']['pAUROC'] >= comparison['P32']['pAUROC'] else 'no'} toward native; {'exceeded native' if comparison['P33']['pAUROC'] >= comparison['P31_native_zero_adapter']['pAUROC'] else 'did not exceed native'}.",
        f"- Did radial/tail and normal-score behavior remain safe? {'yes' if checks['residual_abs_q99']['pass'] and checks['normal_score_q99_shift']['pass'] else 'no'} under the preregistered gates.",
        "- Did actionability help without raw direction fidelity? This is interpreted only through the locked detection endpoints and descriptive selectivity diagnostics; no direction metric was optimized.",
        "",
        "## Runtime and audit",
        "",
        f"- training wall time: {training['training_seconds']:.6f} seconds; parent process time: {training['parent_process_seconds']:.6f} seconds; median step {training['step_time_seconds']['median']:.6f} seconds; prediction time {prediction['prediction_seconds']:.6f} seconds; scoring time {metrics['scoring_seconds']:.6f} seconds.",
        f"- peak GPU allocated/reserved during training: {training['peak_gpu_allocated_bytes']} / {training['peak_gpu_reserved_bytes']} bytes.",
        f"- finite loss/gradient: {training['loss_finite']}/{training['gradient_finite']}; nonfinite counts: {training['nonfinite_loss_count']}/{training['nonfinite_gradient_count']}.",
        f"- student parameter delta L2: {training['student_parameter_delta']['l2']}; teacher/frozen parameter delta: {training['teacher_parameter_delta']}.",
        f"- prediction frozen before scoring: {audit['predictions_frozen_before_scoring']}; prediction SHA-256: {prediction['prediction_sha256']}.",
        f"- held GT/mask reads before freeze: {audit['held_gt_reads_before_prediction_freeze']}/{audit['held_mask_reads_before_prediction_freeze']}; after freeze: {audit['held_gt_reads_after_prediction_freeze']}/{audit['held_mask_reads_after_prediction_freeze']}.",
        f"- new CLIP/Phase2B/teacher forwards: {audit['new_clip_forwards']}/{audit['new_phase2b_forwards']}/{audit['new_teacher_forwards']}; cache rebuilds: {audit['cache_rebuilt']}; reruns: {audit['automatic_rerun']}.",
        "",
        "## Terminal audit",
        "",
        f"- scientific gate: {gate['status']}; failed checks: {gate['failures']}.",
        f"- post-run audit: {audit['status']}; attempt count: {audit['attempt_count']}; Stage 3 started: {audit['stage3_started']}; full run started: {audit['full_started']}.",
        "- authoritative P33 preregistration was not edited after attempt identity creation.",
        "",
        gate["status"],
        "",
    ]
    return "\n".join(lines)


def _write_engineering_stop(root: Path, attempt: Mapping[str, Any] | None, exc: BaseException) -> None:
    atomic_write_json(
        root / "P33_STAGE2_ENGINEERING_STOP.json",
        {
            "schema_version": "P33_STAGE2_ENGINEERING_STOP_V1",
            "status": "P33_STAGE2_ENGINEERING_STOP",
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
            "schema_version": "P33_STAGE2_PRE_EXECUTION_AUDIT_V1",
            "status": "PASS",
            "protocol_id": "P33",
            "utc_timestamp": _utc(),
            "git": git_identity,
            "preregistration_sha256": prereg_hash,
            "objective_contract": p33_objective_contract(),
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
        atomic_write_json(args.output_root / "P33_STAGE2_PRE_EXECUTION_AUDIT.json", pre_execution)
        attempt = {
            "schema_version": "P33_STAGE2_ATTEMPT_V1",
            "completion_status": "ATTEMPT_CONSUMED",
            "attempt_uuid": str(uuid.uuid4()),
            "attempt_number": 1,
            "utc_timestamp": _utc(),
            "branch": git_identity["branch"],
            "scientific_execution_base_sha": git_identity["head"],
            "implementation_commit": git_identity["head"],
            "engineering_qualification_commit": P33_ENGINEERING_QUALIFICATION_COMMIT,
            "protocol_id": "P33",
            "p33_preregistration_sha256": prereg_hash,
            "class": P33_STAGE2_CLASS,
            "split": "locked candle LOCO fit/held split",
            "class_order": list(EXPECTED_VISA_CLASSES),
            "fit_records": P33_FIT_RECORDS,
            "held_records": P33_HELD_RECORDS,
            "epochs": P33_EPOCHS,
            "batch_size": P33_BATCH_SIZE,
            "learning_rate": P33_LEARNING_RATE,
            "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.01, "amsgrad": False},
            "seed": P33_SEED,
            "expected_optimizer_steps": P33_EXPECTED_STEPS,
            "objective": P33_OBJECTIVE_NAME,
            "objective_contract": p33_objective_contract(),
            "cache_root": input_audit["cache_root"],
            "cache_provenance": input_audit["cache_provenance"],
            "frozen_checkpoint_identities": {"p26_checkpoint": input_audit["p26_checkpoint"]},
            "inherited_artifact_hashes": {
                key: input_audit[key] for key in ("engineering_qualification", "implementation_report", "speed_profile", "p31_control_result", "p30r1_metrics", "p32_metrics", "p30r1_predictions")
            },
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
        attempt_path = args.output_root / "P33_STAGE2_ATTEMPT.json"
        if attempt_path.exists():
            raise RuntimeError("P33 Stage 2 attempt identity already exists")
        atomic_write_json(attempt_path, attempt)
        attempt_path.chmod(0o444)
        training = _run_training(args, args.output_root, attempt["attempt_uuid"], prereg_hash, git_identity["head"])
        prediction = _run_prediction(args, args.output_root, training, attempt["attempt_uuid"], prereg_hash, git_identity["head"])
        freeze = _freeze_predictions(args.output_root, prediction, attempt["attempt_uuid"], prereg_hash)
        metrics, context = _score_frozen_predictions(args, args.output_root, prediction, freeze)
        diagnostics = _diagnostics(args.output_root, metrics, context, training)
        comparison = _historical_comparison(metrics)
        gate = _scientific_gate(prereg, training, prediction, freeze, metrics, diagnostics, input_audit, attempt["attempt_uuid"])
        qualification = {
            "schema_version": "P33_STAGE2_QUALIFICATION_V1",
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
        atomic_write_json(args.output_root / "P33_STAGE2_QUALIFICATION.json", qualification)
        audit = _post_run_audit(args.output_root, attempt, training, prediction, freeze, metrics, gate, input_audit)
        _atomic_text(args.output_root / "P33_FINAL_REPORT.md", _final_report(attempt, training, prediction, metrics, diagnostics, comparison, gate, audit))
        return {"status": gate["status"], "attempt_uuid": attempt["attempt_uuid"], "metrics": metrics, "gate": gate, "audit": audit}
    except BaseException as exc:
        if attempt is not None:
            _write_engineering_stop(args.output_root, attempt, exc)
        raise


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
