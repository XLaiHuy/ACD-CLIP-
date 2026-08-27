"""Run exactly one preregistered P34 candle Scientific Stage 2 attempt."""
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
from tools.sabra_v2.p29r1_forensic import residual_magnitude_summary, vectorized_pixel_shifts
from tools.sabra_v2.p34_objective import (
    P34_OBJECTIVE_NAME,
    P34_PREREGISTRATION_SHA256,
    p34_actionability_components,
    p34_objective_contract,
)
from tools.sabra_v2.p34_reference import p34_actionability_components as p34_reference_components
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import (
    CachedSourceDataset,
    TierADataset,
    atomic_write_json,
    sha256_file,
)
from tools.sabra_v2.student_forward import forward_region_student


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "research/sabra_v2/region_distill/P34"
DEFAULT_VISA_ROOT = Path("/workspace/data/source/visa_unpack")
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_P26_CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
DEFAULT_CLIP_ASSET = ROOT / "model/ViT-L-14-336px.pt"
DEFAULT_METADATA = ROOT / "dataset/hub/VisA.jsonl"
P34_BRANCH = "research/p29r1-fast-objective-forensic-v1"
P34_ENGINEERING_QUALIFICATION_COMMIT = "81bb9ef3896bc4723bda9488b0a63a7a93cf2e33"
P34_CLASS = "candle"
P34_EPOCHS = 20
P34_BATCH_SIZE = 1
P34_SEED = 0
P34_LEARNING_RATE = 0.001
P34_BETAS = (0.9, 0.999)
P34_OPTIMIZER_EPSILON = 1e-8
P34_WEIGHT_DECAY = 0.01
P34_AMSGRAD = False
P34_FIT_RECORDS = 1962
P34_HELD_RECORDS = 200
P34_EXPECTED_STEPS = P34_FIT_RECORDS * P34_EPOCHS
P34_C = 4.960109710693359
# This decimal is the exact frozen JSON gate value; do not recompute it with
# a different floating-point rounding path.
P34_MECHANISM_EPSILON = 0.0496010971069336

P34_PREREGISTRATION_JSON = ROOT / "research/sabra_v2/region_distill/P34_PREREGISTRATION.json"
P34_PREREGISTRATION_MD = ROOT / "research/sabra_v2/region_distill/P34_PREREGISTRATION.md"
P34_RESEARCH_DECISION = ROOT / "research/sabra_v2/region_distill/P34_RESEARCH_DECISION.json"
P34_PREFLIGHT = ROOT / "research/sabra_v2/region_distill/P34_PREFLIGHT_FALSIFICATION.json"
P34_IMPLEMENTATION_REPORT = ROOT / "research/sabra_v2/region_distill/P34_IMPLEMENTATION_REPORT.md"
P34_ENGINEERING_QUALIFICATION = ROOT / "research/sabra_v2/region_distill/P34_ENGINEERING_QUALIFICATION.json"
P34_SPEED_PROFILE = ROOT / "research/sabra_v2/region_distill/P34_SPEED_PROFILE.json"

P31_CONTROL_RESULT = ROOT / "research/sabra_v2/region_distill/P31/P31_CONTROL_SCIENTIFIC_RESULT.json"
P30R1_METRICS = ROOT / "research/sabra_v2/region_distill/P30R1/candle/metrics/P30R1_HELD_METRICS.json"
P30R1_PREDICTIONS = ROOT / "research/sabra_v2/region_distill/P30R1/candle/predictions/p30r1_held_predictions.pt"
P32_METRICS = ROOT / "research/sabra_v2/region_distill/P32/candle/metrics/P32_HELD_METRICS.json"
P32_PREDICTIONS = ROOT / "research/sabra_v2/region_distill/P32/candle/predictions/p32_held_predictions.pt"
P33_METRICS = ROOT / "research/sabra_v2/region_distill/P33/candle/metrics/P33_HELD_METRICS.json"
P33_PREDICTIONS = ROOT / "research/sabra_v2/region_distill/P33/candle/predictions/p33_held_predictions.pt"

P34_EXECUTION_HARNESS_PATHS = {
    "tools/sabra_v2/run_p34_scientific_stage2.py",
    "tools/sabra_v2/train_region_distill_p34_cached.py",
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


def _immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)
    path.chmod(0o444)


def _immutable_torch(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    path.chmod(0o444)


def _active_p34_processes() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True)
    excluded = {str(os.getpid()), str(os.getppid())}
    needles = ("run_p34_scientific_stage2", "train_region_distill_p34_cached")
    active: list[str] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or fields[0] in excluded:
            continue
        command = fields[1]
        if ("ps -eo" in command or "rg " in command) and not command.startswith("python"):
            continue
        if any(needle in command for needle in needles):
            active.append(line.strip())
    return active


def _assert_frozen_execution_state(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.resolve() != DEFAULT_OUTPUT_ROOT.resolve():
        raise RuntimeError("P34 Stage 2 accepts only the preregistered evidence directory")
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    if branch != P34_BRANCH:
        raise RuntimeError(f"P34 Stage 2 must run on {P34_BRANCH}, got {branch!r}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", P34_ENGINEERING_QUALIFICATION_COMMIT, head],
        cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError("HEAD is not an engineering-qualified P34 descendant")
    porcelain = _git("status", "--porcelain")
    if porcelain:
        raise RuntimeError(f"scientific execution requires a clean worktree: {porcelain!r}")
    remote = _remote_sha(branch)
    if remote != head:
        raise RuntimeError(f"local/remote mismatch before attempt: {head} != {remote}")
    changed = _git("diff", "--name-only", P34_ENGINEERING_QUALIFICATION_COMMIT, "--").splitlines()
    unexpected = sorted(set(changed) - P34_EXECUTION_HARNESS_PATHS)
    if unexpected:
        raise RuntimeError(f"P34 frozen implementation changed outside execution harness: {unexpected}")
    active = _active_p34_processes()
    if active:
        raise RuntimeError(f"duplicate P34 scientific/training process detected: {active}")
    if args.output_root.exists():
        residual = [path.name for path in args.output_root.iterdir() if path.name != ".gitignore"]
        if residual:
            raise RuntimeError(f"P34 Stage 2 output is already occupied: {residual}")
    if (ROOT / "research/sabra_v2/region_distill/P34/P34_STAGE2_ATTEMPT.json").exists():
        raise RuntimeError("a P34 scientific attempt marker already exists")
    return {
        "branch": branch,
        "head": head,
        "remote_sha": remote,
        "remote_equals_local": True,
        "worktree_clean_before_attempt": True,
        "engineering_qualification_commit": P34_ENGINEERING_QUALIFICATION_COMMIT,
        "execution_harness_descendant_paths": sorted(P34_EXECUTION_HARNESS_PATHS),
        "unexpected_core_changes": [],
        "duplicate_processes": [],
        "attempt_count_before": 0,
    }


def _audit_preregistration() -> tuple[dict[str, Any], str]:
    if not P34_PREREGISTRATION_JSON.is_file() or not P34_PREREGISTRATION_MD.is_file():
        raise RuntimeError("frozen P34 preregistration files are missing")
    observed_hash = sha256_file(P34_PREREGISTRATION_MD)
    if observed_hash != P34_PREREGISTRATION_SHA256:
        raise RuntimeError(f"P34 preregistration Markdown hash mismatch: {observed_hash}")
    prereg = _json(P34_PREREGISTRATION_JSON)
    if (
        prereg.get("schema") != "P34_PREREGISTRATION_V1"
        or prereg.get("status") != "P34_PREREGISTRATION_FROZEN"
        or prereg.get("protocol") != "P34"
        or prereg.get("preregistration_md_sha256") != P34_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("P34 preregistration identity/status drift")
    execution = prereg.get("scientific_execution", {})
    if (
        execution.get("attempts_at_freeze") != 0
        or execution.get("uuid") is not None
        or execution.get("execution_marker") is not None
        or execution.get("held_tuning_iterations") != 0
    ):
        raise RuntimeError("P34 preregistration already contains scientific execution state")
    if prereg.get("selected_hypothesis") != "EXPLICIT_ACTIONABILITY_TARGET_FUNCTIONAL_TRANSFER":
        raise RuntimeError("P34 selected hypothesis drift")
    formulation = prereg.get("formulation", {})
    expected_formulation = {
        "correction_scale_C": P34_C,
        "weight": "stop_gradient(clamp(abs(E_t)/C,0,1))",
        "target": "stop_gradient(weight*E_t)",
        "objective": "mean(SmoothL1(E_s,target,beta=1.0,reduction=none))",
        "smooth_l1_beta": 1.0,
        "objective_count": 1,
        "auxiliary_terms": [],
        "teacher_detached": True,
        "weight_detached": True,
        "target_detached": True,
        "student_self_normalized": False,
        "radial_identifiable": True,
        "hard_gate": False,
        "sparsity_regularizer": False,
        "learned_gate": False,
        "category_specific_parameters": 0,
        "new_tuned_hyperparameters": 0,
        "new_learnable_parameters": 0,
        "teacher_at_inference": False,
        "inference_overhead_percent": 0,
    }
    if any(formulation.get(key) != value for key, value in expected_formulation.items()):
        raise RuntimeError("P34 mathematical formulation drift")
    tensor_contract = prereg.get("tensor_contract", {})
    expected_tensors = {
        "student_region": "[3,B,9,9] float32",
        "teacher_region": "[B,9,9] float32",
        "student_effect": "[B,518,518] float32",
        "teacher_effect": "[B,518,518] float32",
        "weight": "[B,518,518] float32 detached [0,1]",
        "target": "[B,518,518] float32 detached",
    }
    if any(tensor_contract.get(key) != value for key, value in expected_tensors.items()):
        raise RuntimeError("P34 tensor contract drift")
    optimization = prereg.get("optimization", {})
    expected_optimization = {
        "epochs": P34_EPOCHS,
        "batch_size": P34_BATCH_SIZE,
        "expected_optimizer_steps": P34_EXPECTED_STEPS,
        "seed": P34_SEED,
        "precision": "float32",
        "optimizer": "AdamW",
        "learning_rate": P34_LEARNING_RATE,
        "betas": list(P34_BETAS),
        "epsilon": P34_OPTIMIZER_EPSILON,
        "weight_decay": P34_WEIGHT_DECAY,
        "amsgrad": P34_AMSGRAD,
        "schedule_change": False,
    }
    if any(optimization.get(key) != value for key, value in expected_optimization.items()):
        raise RuntimeError("P34 optimizer or schedule contract drift")
    data = prereg.get("data", {})
    if (
        data.get("cache_root") != str(DEFAULT_CACHE_ROOT)
        or data.get("split") != "LOCO candle"
        or data.get("fit_records") != P34_FIT_RECORDS
        or data.get("held_records") != P34_HELD_RECORDS
        or data.get("held_read_order") != "only after P34_PREDICTION_FROZEN"
    ):
        raise RuntimeError("P34 data contract drift")
    forwards = prereg.get("architecture", {})
    if forwards.get("inference_path") != "existing native plus adapter deployment path":
        raise RuntimeError("P34 inference path drift")
    gates = prereg.get("future_scientific_gates", {})
    required_gate_values = {
        "pap_minimum": 0.5141403049313743,
        "pauroc_minimum": 0.9806671435137679,
        "global_residual_abs_q99_max": 8.643353872299194,
        "normal_score_effect_q99_shift_max": 0.0010011587851122385,
        "nonfinite_loss_count": 0,
        "nonfinite_gradient_count": 0,
        "mechanism_epsilon": "C/100",
        "mechanism_epsilon_value": P34_MECHANISM_EPSILON,
        "active_fraction_max_relative_to_p33": 0.999074074074,
        "effective_support_fraction_max_relative_to_p33": 0.962760408648,
        "gini_min_relative_to_p33": 0.069176234345,
        "mechanism_gate_is_not_p30r1_support_target": True,
        "all_gates_required": True,
    }
    if any(gates.get(key) != value for key, value in required_gate_values.items()):
        raise RuntimeError("P34 future scientific gate drift")
    contract = p34_objective_contract()
    if contract.get("preregistration_sha256") != observed_hash or contract.get("objective_count") != 1:
        raise RuntimeError("P34 production objective contract does not match preregistration")
    return prereg, observed_hash


def _audit_inputs(
    args: argparse.Namespace, git_identity: Mapping[str, Any], prereg_hash: str
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    if args.cache_root.resolve() != DEFAULT_CACHE_ROOT.resolve():
        raise RuntimeError("P34 Stage 2 must reuse the frozen P27 cache root")
    config = ROOT / "configs/phase2b_canonical_v1.json"
    for path in (args.metadata, args.cache_root, args.visa_root, args.p26_checkpoint, args.clip_asset, config):
        if not path.exists():
            raise RuntimeError(f"missing frozen input: {path}")
    parent_assets = verify_p26_parent(args.p26_checkpoint, args.clip_asset, config)
    rows = read_visa_metadata(args.metadata)
    if tuple(sorted({str(row["class_name"]) for row in rows})) != tuple(sorted(EXPECTED_VISA_CLASSES)):
        raise RuntimeError("unexpected VisA class inventory")
    inventory = loco_inventory(rows, P34_CLASS)
    if len(inventory.fit_rows) != P34_FIT_RECORDS or len(inventory.held_rows) != P34_HELD_RECORDS:
        raise RuntimeError("frozen candle fit/held inventory changed")
    provenance = p29_cache_provenance(args.metadata)
    CachedSourceDataset(
        inventory.fit_rows,
        P34_CLASS,
        args.cache_root,
        provenance,
        load_source_mask=False,
        load_native_logits=False,
    )
    tier_a = args.cache_root / "tier_a" / P34_CLASS / "manifest.json"
    tier_b = args.cache_root / "tier_b" / P34_CLASS / "manifest.json"
    frozen_paths = (
        P34_RESEARCH_DECISION,
        P34_PREFLIGHT,
        P34_IMPLEMENTATION_REPORT,
        P34_ENGINEERING_QUALIFICATION,
        P34_SPEED_PROFILE,
        P31_CONTROL_RESULT,
        P30R1_METRICS,
        P30R1_PREDICTIONS,
        P32_METRICS,
        P32_PREDICTIONS,
        P33_METRICS,
        P33_PREDICTIONS,
        tier_a,
        tier_b,
    )
    for path in frozen_paths:
        if not path.is_file():
            raise RuntimeError(f"missing frozen P34 input/evidence: {path}")
    preflight = _json(P34_PREFLIGHT)
    if (
        preflight.get("status") != "P34_PREFLIGHT_PASS"
        or preflight.get("held_reads") != 0
        or preflight.get("new_clip_forwards") != 0
        or preflight.get("new_phase2b_forwards") != 0
        or preflight.get("new_teacher_forwards") != 0
    ):
        raise RuntimeError("P34 source/synthetic preflight is not a matching PASS")
    engineering = _json(P34_ENGINEERING_QUALIFICATION)
    if (
        engineering.get("status") != "P34_PASS_TO_SCIENTIFIC_PROTOCOL"
        or engineering.get("final_gate") != "P34_PASS_TO_SCIENTIFIC_PROTOCOL"
        or engineering.get("preregistration_sha256") != prereg_hash
        or engineering.get("implementation", {}).get("production_module") != "tools/sabra_v2/p34_objective.py"
        or engineering.get("implementation", {}).get("objective_count") != 1
        or engineering.get("scientific_safety", {}).get("scientific_uuid_created") is not False
    ):
        raise RuntimeError("P34 engineering qualification is not a matching PASS artifact")
    decision = _json(P34_RESEARCH_DECISION)
    if (
        decision.get("status") != "P34_RESEARCH_DECISION_COMPLETE"
        or decision.get("selected_next_hypothesis") != "EXPLICIT_ACTIONABILITY_TARGET_FUNCTIONAL_TRANSFER"
    ):
        raise RuntimeError("P34 research decision is not a matching frozen decision")
    input_audit: dict[str, Any] = {
        "metadata": {"path": str(args.metadata), "sha256": sha256_file(args.metadata), "records": len(rows)},
        "visa_root": str(args.visa_root.resolve()),
        "visa_root_accessed_before_prediction_freeze": False,
        "cache_root": str(args.cache_root.resolve()),
        "cache_provenance": provenance.as_dict(),
        "tier_a_candle_manifest": {"path": str(tier_a), "sha256": sha256_file(tier_a)},
        "tier_b_candle_manifest": {"path": str(tier_b), "sha256": sha256_file(tier_b)},
        "class_order": list(EXPECTED_VISA_CLASSES),
        "candle_fit_records": P34_FIT_RECORDS,
        "candle_held_records": P34_HELD_RECORDS,
        "p26": parent_assets,
        "p26_checkpoint": {"path": str(args.p26_checkpoint), "sha256": sha256_file(args.p26_checkpoint)},
        "clip_asset": {"path": str(args.clip_asset), "sha256": sha256_file(args.clip_asset)},
        "config": {"path": str(config), "sha256": sha256_file(config)},
        "p34_research_decision": {"path": str(P34_RESEARCH_DECISION), "sha256": sha256_file(P34_RESEARCH_DECISION)},
        "p34_preflight": {"path": str(P34_PREFLIGHT), "sha256": sha256_file(P34_PREFLIGHT)},
        "p34_preregistration_json": {"path": str(P34_PREREGISTRATION_JSON), "sha256": sha256_file(P34_PREREGISTRATION_JSON)},
        "p34_engineering_qualification": {"path": str(P34_ENGINEERING_QUALIFICATION), "sha256": sha256_file(P34_ENGINEERING_QUALIFICATION)},
        "p34_implementation_report": {"path": str(P34_IMPLEMENTATION_REPORT), "sha256": sha256_file(P34_IMPLEMENTATION_REPORT)},
        "p34_speed_profile": {"path": str(P34_SPEED_PROFILE), "sha256": sha256_file(P34_SPEED_PROFILE)},
        "p31_control_result": {"path": str(P31_CONTROL_RESULT), "sha256": sha256_file(P31_CONTROL_RESULT)},
        "p30r1_metrics": {"path": str(P30R1_METRICS), "sha256": sha256_file(P30R1_METRICS)},
        "p30r1_predictions": {"path": str(P30R1_PREDICTIONS), "sha256": sha256_file(P30R1_PREDICTIONS)},
        "p32_metrics": {"path": str(P32_METRICS), "sha256": sha256_file(P32_METRICS)},
        "p32_predictions": {"path": str(P32_PREDICTIONS), "sha256": sha256_file(P32_PREDICTIONS)},
        "p33_metrics": {"path": str(P33_METRICS), "sha256": sha256_file(P33_METRICS)},
        "p33_predictions": {"path": str(P33_PREDICTIONS), "sha256": sha256_file(P33_PREDICTIONS)},
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
    return input_audit, inventory, provenance, preflight


def _production_reference_parity() -> dict[str, Any]:
    cases = (
        ("normal", 1.0, 1.0),
        ("zero", 0.0, 0.0),
        ("near_zero", 1e-4, 1e-4),
        ("sign_reversed", 1.0, -1.0),
    )
    maximum = {key: 0.0 for key in ("loss", "student_effect", "teacher_effect", "weight", "target", "student_gradient")}
    for index, (name, student_scale, teacher_scale) in enumerate(cases):
        generator = torch.Generator(device="cpu").manual_seed(34000 + index)
        student_value = torch.randn((3, 2, 9, 9), generator=generator, dtype=torch.float32) * student_scale
        teacher_value = torch.randn((2, 9, 9), generator=generator, dtype=torch.float32) * teacher_scale
        if name == "zero":
            student_value.zero_()
            teacher_value.zero_()
        if name == "sign_reversed":
            teacher_value = -student_value.mean(dim=0)
        production_student = student_value.clone().requires_grad_(True)
        reference_student = student_value.clone().requires_grad_(True)
        production = p34_actionability_components(production_student, teacher_value.clone())
        reference = p34_reference_components(reference_student, teacher_value.clone())
        for quantity, observed, expected in zip(
            ("loss", "student_effect", "teacher_effect", "weight", "target"), production, reference
        ):
            maximum[quantity] = max(maximum[quantity], float((observed - expected).abs().max().detach().cpu()))
        production[0].backward()
        reference[0].backward()
        maximum["student_gradient"] = max(
            maximum["student_gradient"],
            float((production_student.grad - reference_student.grad).abs().max().detach().cpu()),
        )
        values = (*production, *reference, production_student.grad, reference_student.grad)
        if not all(bool(torch.isfinite(value).all().item()) for value in values):
            raise RuntimeError(f"P34 production/reference parity found non-finite values in {name}")
    tolerances = {key: 1e-6 for key in maximum}
    failures = [name for name, value in maximum.items() if value > tolerances[name]]
    if failures:
        raise RuntimeError(f"P34 production/reference parity failed: {maximum}")
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


def _run_training(
    args: argparse.Namespace,
    root: Path,
    attempt_uuid: str,
    prereg_hash: str,
    execution_sha: str,
    provenance: Any,
) -> dict[str, Any]:
    training_root = root / P34_CLASS / "training"
    if training_root.exists():
        raise RuntimeError(f"scientific training output already exists: {training_root}")
    training_root.mkdir(parents=True)
    parent_seconds = _run_module(
        "tools.sabra_v2.train_region_distill_p34_cached",
        (
            "--output", str(training_root),
            "--metadata", str(args.metadata),
            "--cache-root", str(args.cache_root),
            "--held-class", P34_CLASS,
            "--attempt-uuid", attempt_uuid,
            "--execution-base-sha", execution_sha,
            "--preregistration-sha", prereg_hash,
            "--device", str(args.device),
        ),
    )
    raw_path = training_root / "P34_TRAINING_COMPLETE.json"
    checkpoint_path = training_root / "p34_region_adapter.pt"
    if not raw_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("P34 scientific trainer did not produce required artifacts")
    raw = _json(raw_path)
    required = {
        "status": "FOLD_TRAINING_COMPLETE",
        "protocol_id": "P34",
        "attempt_uuid": attempt_uuid,
        "preregistration_sha256": prereg_hash,
        "scientific_execution_base_sha": execution_sha,
        "held_class": P34_CLASS,
        "fit_records": P34_FIT_RECORDS,
        "held_records_not_read": P34_HELD_RECORDS,
        "optimizer_steps": P34_EXPECTED_STEPS,
        "expected_optimizer_steps": P34_EXPECTED_STEPS,
        "objective": P34_OBJECTIVE_NAME,
        "objective_count": 1,
        "held_gt_reads": 0,
        "held_mask_reads": 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
        "cache_rebuilt": False,
        "teacher_detached": True,
        "weight_detached": True,
        "target_detached": True,
        "scientific_execution_uuid": attempt_uuid,
        "scientific_execution_marker": "P34_STAGE2_ATTEMPT.json",
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"P34 trainer provenance/schedule mismatch: {required}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("schema_version") != "P34_REGION_ADAPTER_CHECKPOINT_V1"
        or checkpoint.get("status") != "FOLD_TRAINING_COMPLETE"
        or checkpoint.get("attempt_uuid") != attempt_uuid
        or checkpoint.get("preregistration_sha256") != prereg_hash
        or checkpoint.get("optimizer_steps") != P34_EXPECTED_STEPS
        or checkpoint.get("objective_count") != 1
        or checkpoint.get("teacher_trainable") is not False
        or checkpoint.get("new_clip_forwards") != 0
        or checkpoint.get("new_phase2b_forwards") != 0
        or checkpoint.get("new_teacher_forwards") != 0
        or checkpoint.get("scientific_execution_uuid") != attempt_uuid
        or checkpoint.get("scientific_execution_marker") != "P34_STAGE2_ATTEMPT.json"
    ):
        raise RuntimeError("P34 scientific checkpoint contract mismatch")
    adapter = RegionResidualAdapter()
    adapter.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("cache_provenance") != provenance.as_dict():
        raise RuntimeError("P34 scientific checkpoint cache provenance mismatch")
    training = dict(raw)
    training.update({
        "parent_process_seconds": parent_seconds,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_strict_reload": True,
        "checkpoint_status": checkpoint.get("status"),
        "objective_contract": p34_objective_contract(),
    })
    _immutable_json(root / P34_CLASS / "P34_STAGE2_TRAINING_COMPLETE.json", training)
    return training


def _run_prediction(
    args: argparse.Namespace,
    root: Path,
    training: Mapping[str, Any],
    inventory: Any,
    provenance: Any,
    attempt_uuid: str,
    prereg_hash: str,
    execution_sha: str,
) -> dict[str, Any]:
    prediction_root = root / P34_CLASS / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_root / "p34_held_predictions.pt"
    completion_path = prediction_root / "PREDICTION_COMPLETE.json"
    if prediction_path.exists() or completion_path.exists():
        raise RuntimeError("P34 scientific prediction output already exists; refusing overwrite")
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
                raise RuntimeError(f"unexpected P34 held prediction fields: {sorted(batch)}")
            seg = batch["seg_features"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            native = batch["native_logits"].permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            student = forward_region_student(adapter, seg, native)
            records.append({
                "image_path": str(batch["image_path"][0]),
                "class_name": P34_CLASS,
                "native_abnormal_probability": student.native_probability[0, 1].detach().cpu(),
                "p34_abnormal_probability": student.deployed_probability[0, 1].detach().cpu(),
                "p34_region_residual": student.region_residual[:, 0].detach().cpu(),
            })
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if len(records) != P34_HELD_RECORDS:
        raise RuntimeError(f"P34 scientific held prediction count mismatch: {len(records)}")
    payload = {
        "schema_version": "P34_IMMUTABLE_HELD_PREDICTIONS_V1",
        "status": "PREDICTIONS_FROZEN_GT_FREE",
        "attempt_uuid": attempt_uuid,
        "protocol_id": "P34",
        "preregistration_sha256": prereg_hash,
        "scientific_execution_base_sha": execution_sha,
        "held_class": P34_CLASS,
        "gt_used": False,
        "mask_reads": 0,
        "held_gt_reads": 0,
        "cache_provenance": provenance.as_dict(),
        "adapter_checkpoint_sha256": training["checkpoint_sha256"],
        "records": records,
    }
    _immutable_torch(prediction_path, payload)
    result = {
        "schema_version": "P34_PREDICTION_COMPLETE_V1",
        "status": "COMPLETE",
        "attempt_uuid": attempt_uuid,
        "protocol_id": "P34",
        "preregistration_sha256": prereg_hash,
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "held_class": P34_CLASS,
        "records": len(records),
        "prediction_shape": [P34_HELD_RECORDS, 518, 518],
        "prediction_dtype": "float32",
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
    _immutable_json(completion_path, result)
    return result


def _freeze_predictions(root: Path, prediction: Mapping[str, Any], attempt_uuid: str, prereg_hash: str) -> dict[str, Any]:
    prediction_path = Path(str(prediction["prediction_path"]))
    completion_path = prediction_path.parent / "PREDICTION_COMPLETE.json"
    if not prediction_path.is_file() or not completion_path.is_file():
        raise RuntimeError("cannot freeze missing P34 predictions")
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "P34_IMMUTABLE_HELD_PREDICTIONS_V1"
        or payload.get("attempt_uuid") != attempt_uuid
        or payload.get("preregistration_sha256") != prereg_hash
        or payload.get("held_class") != P34_CLASS
        or payload.get("gt_used") is not False
        or payload.get("mask_reads") != 0
        or payload.get("held_gt_reads") != 0
        or len(payload.get("records", [])) != P34_HELD_RECORDS
        or prediction.get("prediction_sha256") != sha256_file(prediction_path)
        or prediction_path.stat().st_mode & 0o222
    ):
        raise RuntimeError("P34 scientific prediction freeze firewall failed")
    gate = {
        "schema_version": "P34_PREDICTION_FROZEN_V1",
        "status": "PASS",
        "completion_status": "P34_PREDICTION_FROZEN",
        "utc_timestamp": _utc(),
        "attempt_uuid": attempt_uuid,
        "protocol_id": "P34",
        "p34_preregistration_sha256": prereg_hash,
        "held_class": P34_CLASS,
        "prediction_count": 1,
        "prediction_records": P34_HELD_RECORDS,
        "predictions_frozen": True,
        "fit_or_teacher_steps_after_gate": 0,
        "held_gt_reads_before_prediction_freeze": 0,
        "held_mask_reads_before_prediction_freeze": 0,
        "held_outcome_metrics_read_before_prediction_freeze": False,
        "predictions": [{
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
            "records": P34_HELD_RECORDS,
            "held_class": P34_CLASS,
        }],
    }
    _immutable_json(root / "P34_PREDICTION_FROZEN.json", gate)
    return gate


def _score_frozen_predictions(
    args: argparse.Namespace,
    root: Path,
    prediction: Mapping[str, Any],
    freeze: Mapping[str, Any],
    inventory: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if freeze.get("status") != "PASS" or freeze.get("predictions_frozen") is not True:
        raise RuntimeError("P34 held scoring requires the passing prediction-freeze gate")
    prediction_path = Path(str(prediction["prediction_path"]))
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != P34_HELD_RECORDS:
        raise RuntimeError("P34 frozen prediction record count mismatch")
    by_path = {str(record["image_path"]): record for record in records}
    if len(by_path) != len(records):
        raise RuntimeError("P34 frozen predictions contain duplicate paths")
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
            raise RuntimeError(f"missing P34 frozen prediction for {image_path}")
        native = record.get("native_abnormal_probability")
        candidate = record.get("p34_abnormal_probability")
        residual = record.get("p34_region_residual")
        if not isinstance(native, torch.Tensor) or tuple(native.shape) != (518, 518):
            raise RuntimeError("P34 native frozen map shape mismatch")
        if not isinstance(candidate, torch.Tensor) or tuple(candidate.shape) != (518, 518):
            raise RuntimeError("P34 candidate frozen map shape mismatch")
        if not isinstance(residual, torch.Tensor) or tuple(residual.shape) != (3, 9, 9):
            raise RuntimeError("P34 frozen residual shape mismatch")
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
        "schema_version": "P34_HELD_METRICS_V1",
        "status": "COMPLETE",
        "protocol_id": "P34",
        "held_class": P34_CLASS,
        "attempt_uuid": payload["attempt_uuid"],
        "preregistration_sha256": payload["preregistration_sha256"],
        "prediction_sha256": sha256_file(prediction_path),
        "fit_or_teacher_steps": 0,
        "native_metrics": native_metrics,
        "p34_metrics": candidate_metrics,
        "delta": {key: candidate_metrics[key] - native_metrics[key] for key in ("pAP", "pAUROC")},
        "held_gt_reads_before_prediction_freeze": 0,
        "held_mask_reads_before_prediction_freeze": 0,
        "held_gt_reads_after_prediction_freeze": held_gt_reads,
        "held_mask_file_reads_after_prediction_freeze": held_mask_reads,
        "scoring_started_after_prediction_freeze": True,
        "scoring_seconds": time.perf_counter() - scoring_started,
    }
    metrics_root = root / P34_CLASS / "metrics"
    metrics_root.mkdir(parents=True, exist_ok=True)
    _immutable_json(metrics_root / "P34_HELD_METRICS.json", result)
    return result, {
        "native_probability": native_array,
        "candidate_probability": candidate_array,
        "residual": residual_array,
        "masks": mask_array,
        "image_paths": np.asarray(image_paths, dtype=object),
    }


def _concentration(values: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    if absolute.size == 0:
        return {"effective_support": 0.0, "effective_support_fraction": 0.0, "gini": 0.0, "top_1_percent_mass": 0.0, "top_5_percent_mass": 0.0, "top_10_percent_mass": 0.0}
    total = float(absolute.sum())
    if total == 0.0:
        return {"effective_support": 0.0, "effective_support_fraction": 0.0, "gini": 0.0, "top_1_percent_mass": 0.0, "top_5_percent_mass": 0.0, "top_10_percent_mass": 0.0}
    ordered = np.sort(absolute)[::-1]
    effective = total * total / float(np.sum(absolute * absolute))
    ascending = ordered[::-1]
    n = ordered.size
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


def _held_enrichment(delta: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(delta)
    anomaly_area = float(masks.mean())
    active = absolute > P34_MECHANISM_EPSILON
    total = float(absolute.sum())
    anomaly_mass = float(absolute[masks.astype(bool)].sum())
    return {
        "threshold": P34_MECHANISM_EPSILON,
        "anomaly_area_fraction": anomaly_area,
        "active_fraction": float(active.mean()),
        "anomaly_fraction_of_active": float(masks[active].mean()) if active.any() else 0.0,
        "anomaly_enrichment": float(masks[active].mean() / anomaly_area) if active.any() and anomaly_area else None,
        "absolute_effect_mass_fraction_in_anomaly": anomaly_mass / total if total else 0.0,
        "absolute_effect_mass_enrichment": (anomaly_mass / total) / anomaly_area if total and anomaly_area else None,
    }


def _historical_residual_summary(path: Path, field: str, image_paths: np.ndarray) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    records = payload.get("records", [])
    if len(records) != P34_HELD_RECORDS:
        raise RuntimeError(f"historical prediction count changed: {path}")
    by_path = {str(record["image_path"]): record for record in records}
    if set(by_path) != set(str(value) for value in image_paths.tolist()):
        raise RuntimeError(f"historical prediction identity changed: {path}")
    residual = np.stack([by_path[str(image)][field].numpy() for image in image_paths]).astype(np.float32)
    if residual.shape != (P34_HELD_RECORDS, 3, 9, 9):
        raise RuntimeError(f"historical residual shape changed: {path}")
    support = np.abs(residual) > P34_MECHANISM_EPSILON
    concentration = _concentration(residual)
    return {
        "residual": residual_magnitude_summary(residual),
        "exact_nonzero_fraction": float(np.count_nonzero(residual) / residual.size),
        "active_fraction": float(support.mean()),
        "concentration": concentration,
        "effective_support_fraction": concentration["effective_support_fraction"],
        "gini": concentration["gini"],
        "top_10_percent_mass": concentration["top_10_percent_mass"],
    }


def _diagnostics(
    root: Path,
    metrics: Mapping[str, Any],
    context: Mapping[str, np.ndarray],
    training: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    native = np.asarray(context["native_probability"], dtype=np.float32)
    candidate = np.asarray(context["candidate_probability"], dtype=np.float32)
    residual = np.asarray(context["residual"], dtype=np.float32)
    masks = np.asarray(context["masks"], dtype=np.uint8)
    if not all(np.isfinite(value).all() for value in (native, candidate, residual)):
        raise RuntimeError("P34 post-freeze diagnostics found non-finite predictions")
    candidate_effect = candidate - native
    score_shift = vectorized_pixel_shifts(native, candidate, masks)
    residual_abs = np.abs(residual)
    support = residual_abs > P34_MECHANISM_EPSILON
    residual_concentration = _concentration(residual)
    source_only = preflight.get("source_only", {})
    exact = source_only.get("exact_counts", {})
    target_sample = source_only.get("sample_stats", {}).get("P34_target_wEt", {})
    training_weights = training.get("actionability_weight_samples", [])
    historical = {
        "P30R1": _historical_residual_summary(P30R1_PREDICTIONS, "p30r1_region_residual", context["image_paths"]),
        "P32": _historical_residual_summary(P32_PREDICTIONS, "p32_region_residual", context["image_paths"]),
        "P33": _historical_residual_summary(P33_PREDICTIONS, "p33_region_residual", context["image_paths"]),
    }
    historical["P34"] = {
        "active_fraction": float(support.mean()),
        "effective_support_fraction": residual_concentration["effective_support_fraction"],
        "gini": residual_concentration["gini"],
        "top_10_percent_mass": residual_concentration["top_10_percent_mass"],
    }
    result: dict[str, Any] = {
        "schema_version": "P34_DOWNSTREAM_DIAGNOSTIC_V1",
        "status": "COMPLETE",
        "protocol_id": "P34",
        "held_class": P34_CLASS,
        "attempt_uuid": metrics["attempt_uuid"],
        "mechanism_epsilon": P34_MECHANISM_EPSILON,
        "residual": {
            **residual_magnitude_summary(residual),
            "q50_abs": float(np.quantile(residual_abs, 0.50)),
            "q95_abs": float(np.quantile(residual_abs, 0.95)),
            "q99_abs": float(np.quantile(residual_abs, 0.99)),
            "max_abs": float(residual_abs.max()),
        },
        "residual_exact_nonzero_fraction": float(np.count_nonzero(residual) / residual.size),
        "residual_support": {
            "threshold": P34_MECHANISM_EPSILON,
            "active_fraction": float(support.mean()),
            "effective_support_fraction": residual_concentration["effective_support_fraction"],
            "concentration": residual_concentration,
            "gini": residual_concentration["gini"],
            "top_10_percent_mass": residual_concentration["top_10_percent_mass"],
        },
        "native_to_p34_score_effect": {
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
        "p34_minus_native_pixel_shift": score_shift,
        "held_descriptive_enrichment": _held_enrichment(candidate_effect, masks),
        "source_only_actionability": {
            "frozen_preflight_exact_counts": exact,
            "frozen_preflight_target_sample": target_sample,
            "training_batch_weight_samples": training_weights,
            "target_exact_zero_fraction": float(exact.get("target_zero", 0) / exact["pixels"]) if exact.get("pixels") else None,
            "target_near_zero_fraction": float(exact.get("target_near_zero", 0) / exact["pixels"]) if exact.get("pixels") else None,
            "weight_exact_zero_fraction": float(exact.get("weight_zero", 0) / exact["pixels"]) if exact.get("pixels") else None,
            "weight_saturated_one_fraction": float(exact.get("weight_one", 0) / exact["pixels"]) if exact.get("pixels") else None,
            "weight_gt_075_fraction": float(exact.get("weight_gt_075", 0) / exact["pixels"]) if exact.get("pixels") else None,
            "weight_gt_09_fraction": float(exact.get("weight_gt_09", 0) / exact["pixels"]) if exact.get("pixels") else None,
            "preclamp_ratio_ge_1_fraction": float(exact.get("ratio_ge_1", 0) / exact["pixels"]) if exact.get("pixels") else None,
        },
        "historical_residual_summaries": historical,
        "held_mask_reads_post_freeze": metrics["held_mask_file_reads_after_prediction_freeze"],
        "raw_direction_metrics_are_gates": False,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "new_teacher_forwards": 0,
    }
    _immutable_json(root / "P34_DOWNSTREAM_DIAGNOSTIC.json", result)
    return result


def _historical_comparison(metrics: Mapping[str, Any]) -> dict[str, Any]:
    p31 = _json(P31_CONTROL_RESULT)
    p30r1 = _json(P30R1_METRICS)
    p32 = _json(P32_METRICS)
    p33 = _json(P33_METRICS)
    native = p31.get("primary_comparison", {}).get("native_p31")
    p30_metrics = p30r1.get("p30r1_metrics")
    p32_metrics = p32.get("p32_metrics")
    p33_metrics = p33.get("p33_metrics")
    if not all(isinstance(value, Mapping) for value in (native, p30_metrics, p32_metrics, p33_metrics)):
        raise RuntimeError("historical comparison artifact schema changed")
    p34_metrics = metrics["p34_metrics"]
    output: dict[str, Any] = {
        "P31_native_zero_adapter": {"pAP": float(native["pAP"]), "pAUROC": float(native["pAUROC"])},
        "P30R1": {"pAP": float(p30_metrics["pAP"]), "pAUROC": float(p30_metrics["pAUROC"])},
        "P32": {"pAP": float(p32_metrics["pAP"]), "pAUROC": float(p32_metrics["pAUROC"])},
        "P33": {"pAP": float(p33_metrics["pAP"]), "pAUROC": float(p33_metrics["pAUROC"])},
        "P34": {"pAP": float(p34_metrics["pAP"]), "pAUROC": float(p34_metrics["pAUROC"])},
    }
    for method in ("P30R1", "P32", "P33", "P34"):
        output[f"{method}_minus_P31_native"] = {
            key: output[method][key] - output["P31_native_zero_adapter"][key] for key in ("pAP", "pAUROC")
        }
    for method in ("P32", "P33", "P34"):
        output[f"{method}_minus_P33"] = {key: output[method][key] - output["P33"][key] for key in ("pAP", "pAUROC")}
    output["P34_minus_P32"] = {key: output["P34"][key] - output["P32"][key] for key in ("pAP", "pAUROC")}
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
    frozen = prereg["future_scientific_gates"]
    normal_q99 = diagnostics["p34_minus_native_pixel_shift"]["normal"]["q99"]
    if normal_q99 is None:
        raise RuntimeError("P34 normal-score q99 is undefined")
    pap = float(metrics["p34_metrics"]["pAP"])
    pauroc = float(metrics["p34_metrics"]["pAUROC"])
    residual_q99 = float(diagnostics["residual"]["q99_abs"])
    checks = {
        "pAP": {"value": pap, "minimum": float(frozen["pap_minimum"]), "pass": pap >= float(frozen["pap_minimum"])},
        "pAUROC": {"value": pauroc, "minimum": float(frozen["pauroc_minimum"]), "pass": pauroc >= float(frozen["pauroc_minimum"])},
        "global_residual_abs_q99": {"value": residual_q99, "maximum": float(frozen["global_residual_abs_q99_max"]), "pass": residual_q99 <= float(frozen["global_residual_abs_q99_max"])},
        "normal_score_effect_q99_shift": {"value": float(normal_q99), "maximum": float(frozen["normal_score_effect_q99_shift_max"]), "pass": float(normal_q99) <= float(frozen["normal_score_effect_q99_shift_max"])},
        "nonfinite_loss_count": {"value": int(training["nonfinite_loss_count"]), "maximum": int(frozen["nonfinite_loss_count"]), "pass": int(training["nonfinite_loss_count"]) == 0},
        "nonfinite_gradient_count": {"value": int(training["nonfinite_gradient_count"]), "maximum": int(frozen["nonfinite_gradient_count"]), "pass": int(training["nonfinite_gradient_count"]) == 0},
    }
    support = diagnostics["residual_support"]
    mechanism = {
        "active_fraction_below_p33": {"value": float(support["active_fraction"]), "maximum_exclusive": float(frozen["active_fraction_max_relative_to_p33"]), "pass": float(support["active_fraction"]) < float(frozen["active_fraction_max_relative_to_p33"])},
        "effective_support_fraction_below_p33": {"value": float(support["effective_support_fraction"]), "maximum_exclusive": float(frozen["effective_support_fraction_max_relative_to_p33"]), "pass": float(support["effective_support_fraction"]) < float(frozen["effective_support_fraction_max_relative_to_p33"])},
        "gini_above_p33": {"value": float(support["gini"]), "minimum_exclusive": float(frozen["gini_min_relative_to_p33"]), "pass": float(support["gini"]) > float(frozen["gini_min_relative_to_p33"])},
    }
    structural = {
        "attempt_uuid": attempt_uuid == prediction["attempt_uuid"] == metrics["attempt_uuid"] == freeze["attempt_uuid"],
        "exact_optimizer_steps": training["optimizer_steps"] == P34_EXPECTED_STEPS,
        "objective_count": training["objective_count"] == 1,
        "student_parameter_delta": float(training["student_parameter_delta"]["l2"]) > 0.0,
        "teacher_parameter_delta": float(training["teacher_parameter_delta"]) == 0.0,
        "teacher_detached": training["teacher_detached"] is True,
        "weight_detached": training["weight_detached"] is True,
        "target_detached": training["target_detached"] is True,
        "new_clip_forwards": training["new_clip_forwards"] == 0,
        "new_phase2b_forwards": training["new_phase2b_forwards"] == 0,
        "new_teacher_forwards": training["new_teacher_forwards"] == 0,
        "held_reads_before_prediction_freeze": input_audit["held_gt_reads_before_prediction_freeze"] == 0 and input_audit["held_mask_reads_before_prediction_freeze"] == 0 and input_audit["held_outcome_metrics_read_before_prediction_freeze"] is False,
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
    failures.extend(name for name, item in mechanism.items() if item["pass"] is not True)
    failures.extend(name for name, passed in structural.items() if not passed)
    return {
        "schema_version": "P34_STAGE2_GATE_V1",
        "status": "P34_STAGE2_PASS" if not failures else "P34_STAGE2_SCIENTIFIC_STOP",
        "attempt_uuid": attempt_uuid,
        "protocol_id": "P34",
        "class": P34_CLASS,
        "checks": checks,
        "mechanism_checks": mechanism,
        "structural_checks": structural,
        "failures": failures,
        "threshold_source": "P34_PREREGISTRATION.json future_scientific_gates",
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
    freeze_path = root / "P34_PREDICTION_FROZEN.json"
    metrics_path = root / P34_CLASS / "metrics" / "P34_HELD_METRICS.json"
    attempt_path = root / "P34_STAGE2_ATTEMPT.json"
    failures: list[str] = []
    if attempt.get("completion_status") != "ATTEMPT_CONSUMED":
        failures.append("attempt identity is not consumed")
    if training.get("optimizer_steps") != P34_EXPECTED_STEPS or training.get("status") != "FOLD_TRAINING_COMPLETE":
        failures.append("training schedule/status mismatch")
    if prediction.get("records") != P34_HELD_RECORDS or prediction.get("gt_used") is not False or prediction.get("mask_reads") != 0:
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
        "schema_version": "P34_POST_RUN_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "terminal_status": gate["status"],
        "attempt_uuid": attempt["attempt_uuid"],
        "attempt_count": 1,
        "class": P34_CLASS,
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
        "adaptive_tuning": 0,
        "tracked_code_changes_after_attempt": tracked_code_changes,
        "failures": failures,
    }
    _immutable_json(root / "P34_POST_RUN_AUDIT.json", result)
    return result


def _final_report(
    final_status: str,
    attempt: Mapping[str, Any],
    training: Mapping[str, Any],
    prediction: Mapping[str, Any],
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    gate: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> str:
    source = diagnostics["source_only_actionability"]
    historical = diagnostics["historical_residual_summaries"]
    support = diagnostics["residual_support"]
    p33 = historical["P33"]
    p34 = historical["P34"] if "P34" in historical else {
        "active_fraction": support["active_fraction"],
        "effective_support_fraction": support["effective_support_fraction"],
        "gini": support["gini"],
        "top_10_percent_mass": support["top_10_percent_mass"],
    }
    normal_q99 = gate["checks"]["normal_score_effect_q99_shift"]["value"]
    lines = [
        "# P34 Scientific Stage 2 Final Report",
        "",
        f"Final status: {final_status}.",
        "",
        "Exactly one preregistered candle attempt was executed. No rerun, tuning, Stage 3, subset expansion, or full run occurred.",
        "",
        "## Frozen execution",
        "",
        f"- attempt UUID: {attempt['attempt_uuid']}",
        f"- branch: {attempt['branch']}",
        f"- scientific execution commit: {attempt['scientific_execution_base_sha']}",
        f"- engineering qualification commit: {attempt['engineering_qualification_commit']}",
        f"- preregistration SHA-256: {attempt['p34_preregistration_sha256']}",
        f"- class/split: {P34_CLASS}; fit {P34_FIT_RECORDS}; held {P34_HELD_RECORDS}; optimizer steps {training['optimizer_steps']}",
        f"- objective: {P34_OBJECTIVE_NAME}; objective count {training['objective_count']}; seed {training['seed']}; FP32 AdamW schedule remained frozen.",
        "",
        "## Locked endpoint comparison",
        "",
        "| metric | P31/native | P30R1 | P32 | P33 | P34 | P34 minus native | P34 minus P33 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| pAP | {comparison['P31_native_zero_adapter']['pAP']:.12f} | {comparison['P30R1']['pAP']:.12f} | {comparison['P32']['pAP']:.12f} | {comparison['P33']['pAP']:.12f} | {comparison['P34']['pAP']:.12f} | {comparison['P34_minus_P31_native']['pAP']:+.12f} | {comparison['P34_minus_P33']['pAP']:+.12f} |",
        f"| pAUROC | {comparison['P31_native_zero_adapter']['pAUROC']:.12f} | {comparison['P30R1']['pAUROC']:.12f} | {comparison['P32']['pAUROC']:.12f} | {comparison['P33']['pAUROC']:.12f} | {comparison['P34']['pAUROC']:.12f} | {comparison['P34_minus_P31_native']['pAUROC']:+.12f} | {comparison['P34_minus_P33']['pAUROC']:+.12f} |",
        "",
        "P31/native, P30R1, P32, and P33 are frozen historical comparators; P34 gates were not changed after the result.",
        "",
        "## Selectivity and actionability",
        "",
        "| method | active fraction | effective support fraction | Gini | top-10% mass |",
        "|---|---:|---:|---:|---:|",
        f"| P30R1 | {historical['P30R1']['active_fraction']:.12f} | {historical['P30R1']['effective_support_fraction']:.12f} | {historical['P30R1']['gini']:.12f} | {historical['P30R1']['top_10_percent_mass']:.12f} |",
        f"| P32 | {historical['P32']['active_fraction']:.12f} | {historical['P32']['effective_support_fraction']:.12f} | {historical['P32']['gini']:.12f} | {historical['P32']['top_10_percent_mass']:.12f} |",
        f"| P33 | {p33['active_fraction']:.12f} | {p33['effective_support_fraction']:.12f} | {p33['gini']:.12f} | {p33['top_10_percent_mass']:.12f} |",
        f"| P34 | {p34['active_fraction']:.12f} | {p34['effective_support_fraction']:.12f} | {p34['gini']:.12f} | {p34['top_10_percent_mass']:.12f} |",
        "",
        f"- mechanism epsilon: {P34_MECHANISM_EPSILON:.16f}; P34 residual exact-nonzero fraction: {diagnostics['residual_exact_nonzero_fraction']:.12f}.",
        f"- source-only weights: exact-zero {source['weight_exact_zero_fraction']}; saturated-one {source['weight_saturated_one_fraction']}; >0.75 {source['weight_gt_075_fraction']}; >0.9 {source['weight_gt_09_fraction']}.",
        f"- source-only shaped target: exact-zero {source['target_exact_zero_fraction']}; near-zero {source['target_near_zero_fraction']}; meaningful source target sample q50/q90/q99 {diagnostics['source_only']['frozen_preflight_target_sample'].get('q50_abs')}/{diagnostics['source_only']['frozen_preflight_target_sample'].get('q90_abs')}/{diagnostics['source_only']['frozen_preflight_target_sample'].get('q99_abs')}.",
        f"- P34 score-effect q99 abs: {diagnostics['native_to_p34_score_effect']['q99_abs']:.12f}; normal-score q99 shift: {normal_q99:.12f}.",
        "",
        "## Mechanism answers",
        "",
        f"- Explicit target shaping supplied a zero target and a restoring gradient algebraically; the frozen regression is the P34 target-vs-P33 weighting distinction.",
        f"- P34 materially reduced meaningful intervention relative to P33 under the preregistered mechanism gate: {'yes' if gate['mechanism_checks']['active_fraction_below_p33']['pass'] else 'no'}.",
        f"- High-actionability action was preserved at the source level: shaped target remains nonzero and the source weight reaches one on {source['weight_saturated_one_fraction']} of source pixels; held conditional action is descriptive only.",
        f"- All-zero/native collapse: {'not supported' if comparison['P34']['pAP'] >= comparison['P31_native_zero_adapter']['pAP'] else 'possible/unsupported by the endpoint'}; P34 pAP versus native is {comparison['P34_minus_P31_native']['pAP']:+.12f}.",
        f"- pAP gate: {gate['checks']['pAP']['pass']}; pAUROC gate: {gate['checks']['pAUROC']['pass']}; residual-tail gate: {gate['checks']['global_residual_abs_q99']['pass']}; normal-score gate: {gate['checks']['normal_score_effect_q99_shift']['pass']}.",
        "",
        "## Runtime and audit",
        "",
        f"- training wall time: {training['training_seconds']:.6f} seconds; parent process time: {training['parent_process_seconds']:.6f} seconds; median step {training['step_time_seconds']['median']:.6f} seconds; prediction time {prediction['prediction_seconds']:.6f} seconds; scoring time {metrics['scoring_seconds']:.6f} seconds.",
        f"- peak GPU allocated/reserved during training: {training['peak_gpu_allocated_bytes']} / {training['peak_gpu_reserved_bytes']} bytes; peak RSS {training['peak_process_rss_kib']} KiB.",
        f"- finite loss/gradient: {training['loss_finite']}/{training['gradient_finite']}; nonfinite counts: {training['nonfinite_loss_count']}/{training['nonfinite_gradient_count']}; student delta L2 {training['student_parameter_delta']['l2']}; frozen delta {training['teacher_parameter_delta']}.",
        f"- prediction frozen before scoring: {audit['predictions_frozen_before_scoring']}; prediction SHA-256: {prediction['prediction_sha256']}.",
        f"- held GT/mask reads before freeze: {audit['held_gt_reads_before_prediction_freeze']}/{audit['held_mask_reads_before_prediction_freeze']}; after freeze: {audit['held_gt_reads_after_prediction_freeze']}/{audit['held_mask_reads_after_prediction_freeze']}.",
        f"- new CLIP/Phase2B/teacher forwards: {audit['new_clip_forwards']}/{audit['new_phase2b_forwards']}/{audit['new_teacher_forwards']}; cache rebuilds: {audit['cache_rebuilt']}; reruns: {audit['automatic_rerun']}; Stage 3: {audit['stage3_started']}; full run: {audit['full_started']}.",
        "",
        "## Terminal audit",
        "",
        f"- scientific gate: {gate['status']}; failed checks: {gate['failures']}.",
        f"- post-run audit: {audit['status']}; attempt count: {audit['attempt_count']}; adaptive tuning: {audit['adaptive_tuning']}.",
        "- authoritative P34 preregistration was not edited after attempt identity creation.",
        "",
        final_status,
        "",
    ]
    return "\n".join(lines)


def _write_engineering_stop(root: Path, attempt: Mapping[str, Any] | None, exc: BaseException) -> None:
    _immutable_json(
        root / "P34_STAGE2_ENGINEERING_STOP.json",
        {
            "schema_version": "P34_STAGE2_ENGINEERING_STOP_V1",
            "status": "P34_STAGE2_ENGINEERING_STOP",
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
        prereg, prereg_hash = _audit_preregistration()
        input_audit, inventory, provenance, preflight = _audit_inputs(args, git_identity, prereg_hash)
        parity = _production_reference_parity()
        args.output_root.mkdir(parents=True, exist_ok=True)
        pre_execution = {
            "schema_version": "P34_STAGE2_PRE_EXECUTION_AUDIT_V1",
            "status": "PASS",
            "protocol_id": "P34",
            "utc_timestamp": _utc(),
            "git": git_identity,
            "preregistration_sha256": prereg_hash,
            "objective_contract": p34_objective_contract(),
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
        _immutable_json(args.output_root / "P34_STAGE2_PRE_EXECUTION_AUDIT.json", pre_execution)
        attempt = {
            "schema_version": "P34_STAGE2_ATTEMPT_V1",
            "completion_status": "ATTEMPT_CONSUMED",
            "attempt_uuid": str(uuid.uuid4()),
            "attempt_number": 1,
            "utc_timestamp": _utc(),
            "branch": git_identity["branch"],
            "scientific_execution_base_sha": git_identity["head"],
            "implementation_commit": git_identity["head"],
            "engineering_qualification_commit": P34_ENGINEERING_QUALIFICATION_COMMIT,
            "protocol_id": "P34",
            "p34_preregistration_sha256": prereg_hash,
            "class": P34_CLASS,
            "split": "locked candle LOCO fit/held split",
            "class_order": list(EXPECTED_VISA_CLASSES),
            "fit_records": P34_FIT_RECORDS,
            "held_records": P34_HELD_RECORDS,
            "epochs": P34_EPOCHS,
            "batch_size": P34_BATCH_SIZE,
            "learning_rate": P34_LEARNING_RATE,
            "optimizer": {"name": "AdamW", "betas": list(P34_BETAS), "epsilon": P34_OPTIMIZER_EPSILON, "weight_decay": P34_WEIGHT_DECAY, "amsgrad": P34_AMSGRAD},
            "seed": P34_SEED,
            "expected_optimizer_steps": P34_EXPECTED_STEPS,
            "objective": P34_OBJECTIVE_NAME,
            "objective_contract": p34_objective_contract(),
            "cache_root": input_audit["cache_root"],
            "cache_provenance": input_audit["cache_provenance"],
            "frozen_checkpoint_identities": {"p26_checkpoint": input_audit["p26_checkpoint"]},
            "inherited_artifact_hashes": {
                key: input_audit[key]
                for key in (
                    "p34_research_decision", "p34_preflight", "p34_preregistration_json",
                    "p34_engineering_qualification", "p34_implementation_report", "p34_speed_profile",
                    "p31_control_result", "p30r1_metrics", "p30r1_predictions", "p32_metrics",
                    "p32_predictions", "p33_metrics", "p33_predictions",
                )
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
        attempt_path = args.output_root / "P34_STAGE2_ATTEMPT.json"
        if attempt_path.exists():
            raise RuntimeError("P34 Stage 2 attempt identity already exists")
        _immutable_json(attempt_path, attempt)
        training = _run_training(args, args.output_root, attempt["attempt_uuid"], prereg_hash, git_identity["head"], provenance)
        prediction = _run_prediction(args, args.output_root, training, inventory, provenance, attempt["attempt_uuid"], prereg_hash, git_identity["head"])
        freeze = _freeze_predictions(args.output_root, prediction, attempt["attempt_uuid"], prereg_hash)
        metrics, context = _score_frozen_predictions(args, args.output_root, prediction, freeze, inventory)
        diagnostics = _diagnostics(args.output_root, metrics, context, training, preflight)
        comparison = _historical_comparison(metrics)
        gate = _scientific_gate(prereg, training, prediction, freeze, metrics, diagnostics, input_audit, attempt["attempt_uuid"])
        audit = _post_run_audit(args.output_root, attempt, training, prediction, freeze, metrics, gate, input_audit)
        final_status = gate["status"] if audit["status"] == "PASS" else "P34_STAGE2_ENGINEERING_STOP"
        qualification = {
            "schema_version": "P34_STAGE2_QUALIFICATION_V1",
            "status": final_status,
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
            "post_run_audit": audit,
            "stage3_started": False,
            "full_started": False,
            "automatic_rerun": False,
            "held_result_tuning_iterations": 0,
            "new_clip_forwards": 0,
            "new_phase2b_forwards": 0,
            "new_teacher_forwards": 0,
            "cache_rebuilds": 0,
        }
        _immutable_json(args.output_root / "P34_STAGE2_QUALIFICATION.json", qualification)
        report = _final_report(final_status, attempt, training, prediction, metrics, diagnostics, comparison, gate, audit)
        report_path = args.output_root / "P34_FINAL_REPORT.md"
        report_path.write_text(report, encoding="utf-8")
        report_path.chmod(0o444)
        return {"status": final_status, "attempt_uuid": attempt["attempt_uuid"], "metrics": metrics, "gate": gate, "audit": audit}
    except BaseException as exc:
        if attempt is not None:
            _write_engineering_stop(args.output_root, attempt, exc)
            return {"status": "P34_STAGE2_ENGINEERING_STOP", "attempt_uuid": attempt["attempt_uuid"], "error": str(exc)}
        raise


def main() -> None:
    result = run(make_parser().parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
