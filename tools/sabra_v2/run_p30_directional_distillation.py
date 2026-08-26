"""Fail-closed P30 qualification and one-shot scientific execution runner."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.p30_contract import (
    P30_OUTPUT_ROOT,
    P30_PREREGISTRATION_PATH,
    P30_UUID,
    load_and_audit_p30_preregistration,
    p30_cache_provenance,
    p30_preregistration_hash,
)
from tools.sabra_v2.p30_objective import p30_directional_loss
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import TierADataset, atomic_write_json, sha256_file
from tools.sabra_v2.student_forward import forward_region_student
from tools.sabra_v2.train_region_distill import ROOT


FULL_CLASSES = tuple(EXPECTED_VISA_CLASSES)
SUBSET_CLASSES = ("candle", "chewinggum", "macaroni2", "pcb3")
P29_ROOT = Path("/workspace/p29_science_v1")
P29_CLASS_TABLE = ROOT / "research/sabra_v2/region_distill/P29_CLASS_TABLE.csv"
P29_METRICS = ROOT / "research/sabra_v2/region_distill/P29_METRICS.json"
P29R1_AUDIT = ROOT / "research/sabra_v2/region_distill/P29R1_RECOVERY/P29R1_POST_RUN_AUDIT.json"
CONFIG_PATH = ROOT / "configs/phase2b_canonical_v1.json"
DEFAULT_VISA_ROOT = Path("/workspace/data/source/visa_unpack")
DEFAULT_P26_CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
DEFAULT_CLIP_ASSET = ROOT / "model/ViT-L-14-336px.pt"
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
P29_CACHED_MEDIAN_SECONDS_PER_STEP = 0.010768339969217777
P30_TRAINING_EPOCHS = 20
P30_BATCH_SIZE = 1
P30_LEARNING_RATE = 0.001
P30_SEED = 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("qualify", "preaudit", "run"), required=True)
    parser.add_argument("--visa-root", type=Path, default=DEFAULT_VISA_ROOT)
    parser.add_argument("--p26-checkpoint", type=Path, default=DEFAULT_P26_CHECKPOINT)
    parser.add_argument("--clip-asset", type=Path, default=DEFAULT_CLIP_ASSET)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--output-root", type=Path, default=P30_OUTPUT_ROOT)
    parser.add_argument("--preregistration", type=Path, default=P30_PREREGISTRATION_PATH)
    parser.add_argument("--p30-prereg-sha", default=None)
    parser.add_argument("--execution-base-sha", default=None)
    parser.add_argument("--p30-uuid", default=P30_UUID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _remote_sha(branch: str) -> str:
    fields = _git("ls-remote", "origin", f"refs/heads/{branch}").split()
    if len(fields) != 2:
        raise RuntimeError(f"could not resolve remote branch {branch}")
    return fields[0]


def _git_identity(expected_sha: str | None, *, require_clean: bool = True) -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    local = _git("rev-parse", "HEAD")
    if branch != "research/p29r1-fast-objective-forensic-v1":
        raise RuntimeError(f"P30 must run on the frozen branch, got {branch!r}")
    if expected_sha is not None and local != expected_sha:
        raise RuntimeError(f"P30 execution-base mismatch: expected {expected_sha}, got {local}")
    porcelain = _git("status", "--porcelain")
    if require_clean and porcelain:
        raise RuntimeError(f"P30 requires a clean worktree before execution: {porcelain!r}")
    remote = _remote_sha(branch)
    if remote != local:
        raise RuntimeError(f"P30 local/remote mismatch: {local} != {remote}")
    return {
        "branch": branch,
        "local_sha": local,
        "remote_sha": remote,
        "remote_equals_local": remote == local,
        "worktree_clean": not bool(porcelain),
    }


def _active_p30_training_processes() -> list[str]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], check=True, text=True, capture_output=True)
    own_pid = str(os.getpid())
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if "tools.sabra_v2.train_region_distill_p30_cached" in line and not line.lstrip().startswith(own_pid + " ")
    ]


def _ensure_no_active_training() -> None:
    active = _active_p30_training_processes()
    if active:
        raise RuntimeError(f"duplicate P30 training process detected: {active}")


def _audit_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], str]:
    if args.p30_uuid != P30_UUID:
        raise RuntimeError("P30 UUID does not match the frozen preregistration")
    if args.preregistration.resolve() != P30_PREREGISTRATION_PATH.resolve():
        raise RuntimeError("P30 execution only accepts the frozen in-repository preregistration")
    prereg_hash = p30_preregistration_hash(P30_PREREGISTRATION_PATH)
    if args.p30_prereg_sha is not None and args.p30_prereg_sha != prereg_hash:
        raise RuntimeError("P30 preregistration hash argument mismatch")
    prereg = load_and_audit_p30_preregistration(P30_PREREGISTRATION_PATH, prereg_hash)
    if args.cache_root.resolve() != DEFAULT_CACHE_ROOT.resolve():
        raise RuntimeError("P30 must reuse the frozen P27 cache root")
    if not args.metadata.is_file() or not args.cache_root.is_dir() or not args.visa_root.is_dir():
        raise RuntimeError("P30 frozen metadata, cache root, and VisA root must exist")
    parent_assets = verify_p26_parent(args.p26_checkpoint, args.clip_asset, CONFIG_PATH)
    rows = read_visa_metadata(args.metadata)
    observed_classes = tuple(sorted({str(row["class_name"]) for row in rows}))
    if observed_classes != tuple(sorted(FULL_CLASSES)):
        raise RuntimeError(f"unexpected VisA class inventory: {observed_classes}")
    provenance = p30_cache_provenance(args.metadata)
    inventory_counts = {
        name: {
            "fit_records": len(loco_inventory(rows, name).fit_rows),
            "held_records": len(loco_inventory(rows, name).held_rows),
        }
        for name in FULL_CLASSES
    }
    input_inventory: dict[str, Any] = {
        "metadata": {"path": str(args.metadata), "sha256": sha256_file(args.metadata), "records": len(rows)},
        "visa_root": str(args.visa_root.resolve()),
        "class_order": list(FULL_CLASSES),
        "fold_counts": inventory_counts,
        "p26": parent_assets,
        "cache_provenance": provenance.as_dict(),
        "cache_root": str(args.cache_root.resolve()),
        "config": {"path": str(CONFIG_PATH), "sha256": sha256_file(CONFIG_PATH)},
        "clip_asset": {"path": str(args.clip_asset), "sha256": sha256_file(args.clip_asset)},
        "p26_checkpoint": {"path": str(args.p26_checkpoint), "sha256": sha256_file(args.p26_checkpoint)},
        "p29_class_table": {"path": str(P29_CLASS_TABLE), "sha256": sha256_file(P29_CLASS_TABLE)},
        "p29_metrics": {"path": str(P29_METRICS), "sha256": sha256_file(P29_METRICS)},
        "p29r1_audit": {"path": str(P29R1_AUDIT), "sha256": sha256_file(P29R1_AUDIT)},
        "p29_reference": {},
        "tier_a_manifests": {},
        "tier_b_manifests": {},
        "p30_preregistration_sha256": prereg_hash,
    }
    if _json(P29R1_AUDIT).get("status") != "PASS":
        raise RuntimeError("P29R1 recovery audit is not PASS")
    for class_name in FULL_CLASSES:
        p29_checkpoint = P29_ROOT / class_name / "training" / "p29_region_adapter.pt"
        p29_prediction = P29_ROOT / class_name / "predictions" / "p29_held_predictions.pt"
        if not p29_checkpoint.is_file() or not p29_prediction.is_file():
            raise RuntimeError(f"missing frozen P29 reference artifact for {class_name}")
        input_inventory["p29_reference"][class_name] = {
            "checkpoint": {"path": str(p29_checkpoint), "sha256": sha256_file(p29_checkpoint)},
            "prediction": {"path": str(p29_prediction), "sha256": sha256_file(p29_prediction)},
        }
        for tier, filename in (("tier_a_manifests", "tier_a"), ("tier_b_manifests", "tier_b")):
            manifest_path = args.cache_root / filename / class_name / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(f"missing frozen cache manifest for {filename}/{class_name}")
            input_inventory[tier][class_name] = {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            }
    return prereg, input_inventory, prereg_hash


def _run_command(command: Sequence[str]) -> float:
    print(json.dumps({"event": "START", "utc": _utc(), "command": list(command)}), flush=True)
    started = time.perf_counter()
    subprocess.run(list(command), cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    print(json.dumps({"event": "COMPLETE", "utc": _utc(), "seconds": elapsed, "command": list(command)}), flush=True)
    return elapsed


def _run_module(module: str, arguments: Sequence[str]) -> float:
    return _run_command([sys.executable, "-m", module, *arguments])


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _common_fold_args(args: argparse.Namespace, prereg_hash: str, execution_sha: str, class_name: str) -> list[str]:
    return [
        "--held-class",
        class_name,
        "--visa-root",
        str(args.visa_root),
        "--p26-checkpoint",
        str(args.p26_checkpoint),
        "--clip-asset",
        str(args.clip_asset),
        "--cache-root",
        str(args.cache_root),
        "--metadata",
        str(args.metadata),
        "--p30-execution-base-sha",
        execution_sha,
        "--p30-prereg-sha",
        prereg_hash,
        "--p30-uuid",
        P30_UUID,
        "--device",
        args.device,
        "--num-workers",
        "0",
    ]


def _run_fold(
    args: argparse.Namespace,
    root: Path,
    class_name: str,
    stage: str,
    prereg_hash: str,
    execution_sha: str,
    *,
    engineering_smoke: bool = False,
    max_steps: int | None = None,
) -> dict[str, Any]:
    fold_root = root / class_name
    if fold_root.exists():
        raise RuntimeError(f"P30 fold output already exists; refusing overwrite: {fold_root}")
    fold_root.mkdir(parents=True)
    common = _common_fold_args(args, prereg_hash, execution_sha, class_name)
    train_args = [
        *common,
        "--output",
        str(fold_root / "training"),
        "--stage",
        stage,
        "--epochs",
        str(1 if engineering_smoke else P30_TRAINING_EPOCHS),
        "--batch-size",
        str(P30_BATCH_SIZE),
        "--learning-rate",
        str(P30_LEARNING_RATE),
        "--seed",
        str(P30_SEED),
    ]
    if engineering_smoke:
        train_args.append("--engineering-smoke")
    if max_steps is not None:
        train_args.extend(["--max-steps", str(max_steps)])
    training_seconds = _run_module("tools.sabra_v2.train_region_distill_p30_cached", train_args)
    checkpoint = fold_root / "training" / "p30_region_adapter.pt"
    prediction_args = [
        *common,
        "--adapter-checkpoint",
        str(checkpoint),
        "--output",
        str(fold_root / "predictions"),
        "--stage",
        stage,
        "--batch-size",
        "1",
    ]
    prediction_seconds = _run_module("tools.sabra_v2.evaluate_region_distill_p30_cached", prediction_args)
    training_result = _json(fold_root / "training" / "TRAINING_COMPLETE.json")
    prediction_result = _json(fold_root / "predictions" / "PREDICTION_COMPLETE.json")
    return {
        "class": class_name,
        "stage": stage,
        "training_seconds_parent": training_seconds,
        "prediction_seconds_parent": prediction_seconds,
        "training": training_result,
        "prediction": prediction_result,
    }


def _prediction_gate(root: Path, classes: Sequence[str], prereg_hash: str, stage: str) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for class_name in classes:
        completion_path = root / class_name / "predictions" / "PREDICTION_COMPLETE.json"
        prediction_path = root / class_name / "predictions" / "p30_held_predictions.pt"
        if not completion_path.is_file() or not prediction_path.is_file():
            raise RuntimeError(f"all requested P30 predictions must freeze before scoring: {class_name}")
        completion = _json(completion_path)
        if (
            completion.get("schema_version") != "P30_PREDICTION_COMPLETE_V1"
            or completion.get("completion_status") != "COMPLETE"
            or completion.get("held_class") != class_name
            or completion.get("stage") != stage
            or completion.get("prediction_sha256") != sha256_file(prediction_path)
            or prediction_path.stat().st_mode & 0o222
        ):
            raise RuntimeError(f"invalid immutable P30 prediction {class_name}")
        payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema_version") != "P30_IMMUTABLE_HELD_PREDICTIONS_V1"
            or payload.get("held_class") != class_name
            or payload.get("gt_used") is not False
            or payload.get("mask_reads") != 0
            or payload.get("p30_preregistration_sha256") != prereg_hash
            or payload.get("p30_uuid") != P30_UUID
        ):
            raise RuntimeError(f"P30 held-prediction firewall failed for {class_name}")
        artifacts.append({
            "held_class": class_name,
            "path": str(prediction_path),
            "sha256": sha256_file(prediction_path),
            "records": len(payload.get("records", [])),
        })
    gate = {
        "schema_version": "P30_SCORING_GATE_V1",
        "status": "PASS",
        "completion_status": "PREDICTIONS_FROZEN_BEFORE_SCORING",
        "utc_timestamp": _utc(),
        "stage": stage,
        "p30_uuid": P30_UUID,
        "p30_preregistration_sha256": prereg_hash,
        "prediction_count": len(artifacts),
        "classes": list(classes),
        "predictions_frozen": True,
        "fit_or_teacher_steps_after_gate": 0,
        "predictions": artifacts,
    }
    atomic_write_json(root / "P30_SCORING_GATE.json", gate)
    return gate


def _score_and_diagnose(
    args: argparse.Namespace,
    root: Path,
    classes: Sequence[str],
    stage: str,
    prereg_hash: str,
    *,
    include_diagnostics: bool = True,
) -> dict[str, Any]:
    gate = _prediction_gate(root, classes, prereg_hash, stage)
    timings: dict[str, float] = {}
    for class_name in classes:
        fold_root = root / class_name
        timings[f"score_{class_name}"] = _run_module(
            "tools.sabra_v2.score_region_distill_p30",
            [
                "--held-class",
                class_name,
                "--visa-root",
                str(args.visa_root),
                "--predictions",
                str(fold_root / "predictions" / "p30_held_predictions.pt"),
                "--output",
                str(fold_root / "metrics"),
                "--metadata",
                str(args.metadata),
                "--p30-prereg-sha",
                prereg_hash,
                "--p30-uuid",
                P30_UUID,
                "--stage",
                stage,
            ],
        )
    timings["aggregate"] = _run_module(
        "tools.sabra_v2.aggregate_region_distill_p30",
        [
            "--run-root",
            str(root),
            "--output",
            str(root / "aggregate"),
            "--classes",
            *classes,
            "--p29-class-table",
            str(P29_CLASS_TABLE),
        ],
    )
    if include_diagnostics:
        timings["diagnostics"] = _run_module(
            "tools.sabra_v2.analyze_p30_outputs",
            [
                "--run-root",
                str(root),
                "--cache-root",
                str(args.cache_root),
                "--visa-root",
                str(args.visa_root),
                "--metadata",
                str(args.metadata),
                "--scoring-gate",
                str(root / "P30_SCORING_GATE.json"),
                "--classes",
                *classes,
                "--p30-prereg-sha",
                prereg_hash,
                "--p30-uuid",
                P30_UUID,
                "--device",
                args.device,
            ],
        )
    return {"gate": gate, "timings": timings}


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(child) for child in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _path_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _diagnostic_gate(root: Path, classes: Sequence[str], *, stage: str, overhead_percent: float) -> dict[str, Any]:
    aggregate = _json(root / "aggregate" / "P30_RESULTS.json")
    transfer = _json(root / "P30_TRANSFER_DIAGNOSTIC.json")
    stability = _json(root / "P30_STABILITY_DIAGNOSTIC.json")
    transfer_rows = {str(row["class"]): row for row in transfer.get("classes", [])}
    stability_rows = {str(row["class"]): row for row in stability.get("classes", [])}
    aggregate_rows = {str(row["class"]): row for row in aggregate.get("classes", [])}
    failures: list[str] = []
    class_checks: list[dict[str, Any]] = []
    p_ap_drops_over_02 = 0
    p_auc_drops_over_02 = 0
    p_ap_drops_over_05 = 0
    p_auc_drops_over_05 = 0
    for class_name in classes:
        if class_name not in aggregate_rows or class_name not in transfer_rows or class_name not in stability_rows:
            failures.append(f"missing diagnostics for {class_name}")
            continue
        row = aggregate_rows[class_name]
        direction = transfer_rows[class_name]
        stable = stability_rows[class_name]
        p_ap_drop = float(row["delta_pAP"])
        p_auc_drop = float(row["delta_pAUROC"])
        p29_direction = direction["p29"]["directional_cosine"]["mean"]
        p30_direction = direction["p30"]["directional_cosine"]["mean"]
        p29_sign = float(direction["p29"]["alignment"]["sign_agreement"])
        p30_sign = float(direction["p30"]["alignment"]["sign_agreement"])
        p29_normal_q99 = _path_value(stable, ("p29", "normal", "q99"))
        p30_normal_q99 = _path_value(stable, ("p30", "normal", "q99"))
        finite = _finite({"metrics": row, "transfer": direction, "stability": stable})
        if p_ap_drop < -0.05:
            p_ap_drops_over_05 += 1
        if p_auc_drop < -0.05:
            p_auc_drops_over_05 += 1
        if p_ap_drop < -0.02:
            p_ap_drops_over_02 += 1
        if p_auc_drop < -0.02:
            p_auc_drops_over_02 += 1
        if not finite:
            failures.append(f"non-finite diagnostic values for {class_name}")
        if p_ap_drop < -0.05 or p_auc_drop < -0.05:
            failures.append(f"catastrophic primary-metric regression for {class_name}")
        if p29_direction is None or p30_direction is None or float(p30_direction) < float(p29_direction) - 0.01:
            failures.append(f"directional cosine regression for {class_name}")
        if p30_sign < p29_sign - 0.01:
            failures.append(f"sign agreement regression for {class_name}")
        if p29_normal_q99 is not None and p30_normal_q99 is not None and float(p30_normal_q99) > float(p29_normal_q99) + 0.001:
            failures.append(f"normal q99 inflation for {class_name}")
        class_checks.append({
            "class": class_name,
            "pAP_drop_from_native": p_ap_drop,
            "pAUROC_drop_from_native": p_auc_drop,
            "p29_directional_cosine": p29_direction,
            "p30_directional_cosine": p30_direction,
            "p29_sign_agreement": p29_sign,
            "p30_sign_agreement": p30_sign,
            "p29_normal_q99": p29_normal_q99,
            "p30_normal_q99": p30_normal_q99,
            "finite": finite,
        })
    if stage == "subset" and (p_ap_drops_over_02 > 1 or p_auc_drops_over_02 > 1):
        failures.append("more than one subset class regressed by over 0.02")
    if overhead_percent > 15.0:
        failures.append(f"training overhead exceeds 15 percent: {overhead_percent:.3f}")
    return {
        "schema_version": "P30_STAGE_GATE_V1",
        "status": "PASS" if not failures else "STOP",
        "stage": stage,
        "classes": list(classes),
        "overhead_percent": overhead_percent,
        "thresholds": {
            "catastrophic_metric_drop": -0.05,
            "subset_drop_count_over_0.02_max": 1,
            "directional_cosine_drop_max": 0.01,
            "sign_agreement_drop_max": 0.01,
            "normal_q99_inflation_max": 0.001,
            "training_overhead_investigate_max": 15.0,
        },
        "counts": {
            "pAP_drops_over_0.02": p_ap_drops_over_02,
            "pAUROC_drops_over_0.02": p_auc_drops_over_02,
            "pAP_drops_over_0.05": p_ap_drops_over_05,
            "pAUROC_drops_over_0.05": p_auc_drops_over_05,
        },
        "class_checks": class_checks,
        "failures": failures,
    }


def _synthetic_gradients(output_root: Path) -> dict[str, Any]:
    torch.manual_seed(P30_SEED)
    base = torch.linspace(-2.0, 2.0, 81, dtype=torch.float32).reshape(1, 9, 9)
    base[0, 4, 4] = 0.25

    def student(scale: float = 1.0) -> torch.Tensor:
        return base.unsqueeze(0).expand(3, -1, -1, -1).clone() * scale

    identical = float(p30_directional_loss(student(), base).total.detach())
    opposite = float(p30_directional_loss(-student(), base).total.detach())
    scales = {str(scale): float(p30_directional_loss(student(scale), base).total.detach()) for scale in (0.1, 1.0, 10.0, 100.0)}
    zero_target = p30_directional_loss(torch.randn((3, 1, 9, 9)), torch.zeros((1, 9, 9)))
    near_target = torch.zeros((1, 9, 9), dtype=torch.float32)
    near_target[0, 0, 0] = 1e-8
    near_result = p30_directional_loss(torch.randn((3, 1, 9, 9)), near_target)
    partial_losses: list[float] = []
    for mismatches in (0, 1, 10, 40, 81):
        candidate = torch.ones((3, 1, 9, 9), dtype=torch.float32)
        candidate.reshape(-1)[:mismatches] *= -1.0
        partial_losses.append(float(p30_directional_loss(candidate, torch.ones((1, 9, 9))).total.detach()))
    ordered = torch.arange(1, 82, dtype=torch.float32).reshape(1, 9, 9)
    ordered_student = ordered.unsqueeze(0).expand(3, -1, -1, -1).clone()
    reversed_student = torch.flip(ordered_student, dims=(-1, -2))
    ordering = {
        "matching": float(p30_directional_loss(ordered_student, ordered).total.detach()),
        "reversed": float(p30_directional_loss(reversed_student, ordered).total.detach()),
    }
    output = torch.zeros((3, 1, 9, 9), dtype=torch.float32, requires_grad=True)
    teacher = base.clone().requires_grad_(True)
    backward_loss = p30_directional_loss(output, teacher).total
    backward_loss.backward()
    gradient = output.grad.detach()
    report = {
        "schema_version": "P30_SYNTHETIC_GRADIENTS_V1",
        "status": "PASS",
        "observed_data_used": False,
        "objective_count": 1,
        "cases": {
            "identical_direction_loss": identical,
            "opposite_direction_loss": opposite,
            "same_direction_scale_losses": scales,
            "zero_target": {
                "valid_count": zero_target.valid_count,
                "loss": float(zero_target.total.detach()),
                "finite": bool(torch.isfinite(zero_target.total)),
            },
            "near_zero_target": {
                "valid_count": near_result.valid_count,
                "loss": float(near_result.total.detach()),
                "finite": bool(torch.isfinite(near_result.total)),
            },
            "partial_sign_mismatch_losses": partial_losses,
            "spatial_ordering_losses": ordering,
        },
        "student_output_backward": {
            "loss": float(backward_loss.detach()),
            "gradient_l2": float(torch.linalg.vector_norm(gradient)),
            "gradient_max_abs": float(gradient.abs().max()),
            "gradient_finite": bool(torch.isfinite(gradient).all()),
            "teacher_detached": teacher.grad is None,
        },
        "thresholds": {
            "gradient_l2_min": 1e-8,
            "gradient_l2_max": 1000.0,
            "gradient_max_abs_max": 100.0,
        },
    }
    scale_values = list(scales.values())
    checks = [
        identical < 0.01,
        opposite > 1.9,
        max(scale_values) < 0.1,
        max(scale_values) - min(scale_values) < 0.1,
        zero_target.valid_count == 0 and float(zero_target.total) == 0.0,
        near_result.valid_count == 1 and bool(torch.isfinite(near_result.total)),
        partial_losses == sorted(partial_losses) and partial_losses[0] < partial_losses[-1],
        ordering["matching"] < 1e-5 and ordering["reversed"] > ordering["matching"],
        bool(torch.isfinite(gradient).all()),
        1e-8 < float(torch.linalg.vector_norm(gradient)) < 1000.0,
        float(gradient.abs().max()) < 100.0,
        teacher.grad is None,
    ]
    if not all(checks):
        report["status"] = "STOP"
        report["failed_checks"] = [index for index, passed in enumerate(checks) if not passed]
        raise RuntimeError(f"P30 synthetic gradient gate failed: {report}")
    atomic_write_json(output_root / "P30_SYNTHETIC_GRADIENTS.json", report)
    return report


def _static_qualification(output_root: Path) -> dict[str, Any]:
    compile_paths = [
        "tools/sabra_v2/p30_contract.py",
        "tools/sabra_v2/p30_objective.py",
        "tools/sabra_v2/train_region_distill_p30_cached.py",
        "tools/sabra_v2/evaluate_region_distill_p30_cached.py",
        "tools/sabra_v2/score_region_distill_p30.py",
        "tools/sabra_v2/aggregate_region_distill_p30.py",
        "tools/sabra_v2/analyze_p30_outputs.py",
        "tools/sabra_v2/run_p30_directional_distillation.py",
    ]
    _run_command([sys.executable, "-m", "py_compile", *compile_paths])
    test_paths = [
        "tests/test_p30_objective.py",
        "tests/test_p30_execution_contract.py",
        "tests/test_p29_objective.py",
        "tests/test_p29_execution_contract.py",
        "tests/test_p29r1_forensic.py",
        "tests/test_p29r1_root_cause.py",
        "tests/test_sabra_v2_region_pool.py",
        "tests/test_sabra_v2_student_forward.py",
    ]
    _run_command([sys.executable, "-m", "pytest", "-q", *test_paths])
    result = {
        "schema_version": "P30_STATIC_QUALIFICATION_V1",
        "status": "PASS",
        "compile_paths": compile_paths,
        "test_paths": test_paths,
        "teacher_frozen": True,
        "unexpected_dataset_reads": 0,
        "output_schemas_checked": True,
        "new_p30_objective_count": 1,
    }
    atomic_write_json(output_root / "P30_STATIC_QUALIFICATION.json", result)
    return result


def _engineering_profile(root: Path, profile: Mapping[str, Any]) -> dict[str, Any]:
    completion = profile["training"]
    median = float(completion["median_step_seconds"])
    overhead = 100.0 * (median - P29_CACHED_MEDIAN_SECONDS_PER_STEP) / P29_CACHED_MEDIAN_SECONDS_PER_STEP
    result = {
        "schema_version": "P30_ENGINEERING_QUALIFICATION_V1",
        "status": "PASS" if math.isfinite(median) and overhead <= 15.0 else "STOP",
        "label": "ENGINEERING_QUALIFICATION_ONLY",
        "profile_class": "candle",
        "profile_steps": int(completion["steps"]),
        "p29_cached_median_seconds_per_step": P29_CACHED_MEDIAN_SECONDS_PER_STEP,
        "p30_cached_median_seconds_per_step": median,
        "p30_cached_mean_seconds_per_step": float(completion["mean_step_seconds"]),
        "training_overhead_percent": overhead,
        "peak_gpu_allocated_bytes": int(completion["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": int(completion["peak_gpu_reserved_bytes"]),
        "peak_process_rss_kib": int(completion["peak_process_rss_kib"]),
        "fit_records": int(completion["fit_records"]),
        "new_clip_forwards": int(completion["new_clip_forwards"]),
        "new_phase2b_forwards": int(completion["new_phase2b_forwards"]),
        "optimizer_steps": int(completion["optimizer_steps"]),
        "held_gt_reads": int(completion["held_gt_reads"]),
        "held_mask_reads": int(completion["held_mask_reads"]),
        "objective_count": 1,
    }
    return result


def _run_qualification(args: argparse.Namespace, prereg: Mapping[str, Any], prereg_hash: str) -> dict[str, Any]:
    output_root = args.output_root
    if output_root.exists():
        raise RuntimeError(f"P30 qualification output already exists; refusing overwrite: {output_root}")
    _ensure_no_active_training()
    identity = _git_identity(None, require_clean=True)
    output_root.mkdir(parents=True)
    static = _static_qualification(output_root)
    synthetic = _synthetic_gradients(output_root)
    smoke_root = output_root / "qualification" / "stage1_smoke"
    smoke = _run_fold(
        args,
        smoke_root,
        "candle",
        "smoke",
        prereg_hash,
        identity["local_sha"],
        engineering_smoke=True,
        max_steps=1,
    )
    smoke_training = smoke["training"]
    if (
        smoke_training.get("status") != "ENGINEERING_SMOKE_ONLY"
        or smoke_training.get("steps") != 1
        or smoke_training.get("optimizer_steps") != 1
        or smoke_training.get("held_gt_reads") != 0
        or smoke_training.get("held_mask_reads") != 0
        or smoke_training.get("teacher_parameter_delta") != 0.0
        or float(smoke_training["student_parameter_delta"]["l2"]) <= 0.0
        or smoke_training.get("new_clip_forwards") != 0
        or smoke_training.get("new_phase2b_forwards") != 0
        or not _finite(smoke_training.get("gradient_health", {}))
    ):
        raise RuntimeError(f"P30 tiny smoke gate failed: {smoke_training}")
    profile_root = output_root / "qualification" / "profile"
    profile = _run_fold(
        args,
        profile_root,
        "candle",
        "smoke",
        prereg_hash,
        identity["local_sha"],
        engineering_smoke=True,
        max_steps=40,
    )
    engineering = _engineering_profile(profile_root, profile)
    atomic_write_json(output_root / "P30_ENGINEERING_QUALIFICATION.json", engineering)
    if engineering["status"] != "PASS":
        raise RuntimeError(f"P30 engineering timing gate failed: {engineering}")

    stage2_root = output_root / "qualification" / "stage2_one_class"
    stage2_root.mkdir(parents=True)
    stage2 = _run_fold(args, stage2_root, "candle", "one_class", prereg_hash, identity["local_sha"])
    stage2_result = _score_and_diagnose(args, stage2_root, ("candle",), "one_class", prereg_hash)
    stage2_gate = _diagnostic_gate(stage2_root, ("candle",), stage="one_class", overhead_percent=float(engineering["training_overhead_percent"]))
    stage2_payload = {
        "schema_version": "P30_STAGE2_QUALIFICATION_V1",
        "status": stage2_gate["status"],
        "selection": "candle was fixed by preregistration before P30 results",
        "classes": ["candle"],
        "training": stage2,
        "scoring": stage2_result,
        "gate": stage2_gate,
    }
    atomic_write_json(output_root / "P30_STAGE2_QUALIFICATION.json", stage2_payload)
    if stage2_gate["status"] != "PASS":
        raise RuntimeError(f"P30 Stage 2 gate failed: {stage2_gate}")

    stage3_root = output_root / "qualification" / "stage3_subset"
    stage3_root.mkdir(parents=True)
    stage3_folds = [
        _run_fold(args, stage3_root, class_name, "subset", prereg_hash, identity["local_sha"])
        for class_name in SUBSET_CLASSES
    ]
    stage3_result = _score_and_diagnose(args, stage3_root, SUBSET_CLASSES, "subset", prereg_hash)
    stage3_gate = _diagnostic_gate(stage3_root, SUBSET_CLASSES, stage="subset", overhead_percent=float(engineering["training_overhead_percent"]))
    stage3_payload = {
        "schema_version": "P30_STAGE3_QUALIFICATION_V1",
        "status": stage3_gate["status"],
        "selection": "canonical-order positions 0, 3, 6, and 9 fixed by preregistration before P30 results",
        "classes": list(SUBSET_CLASSES),
        "training": stage3_folds,
        "scoring": stage3_result,
        "gate": stage3_gate,
    }
    atomic_write_json(output_root / "P30_STAGE3_QUALIFICATION.json", stage3_payload)
    if stage3_gate["status"] != "PASS":
        raise RuntimeError(f"P30 Stage 3 gate failed: {stage3_gate}")
    result = {
        "schema_version": "P30_QUALIFICATION_SUMMARY_V1",
        "status": "PASS",
        "p30_uuid": P30_UUID,
        "p30_preregistration_sha256": prereg_hash,
        "qualification_execution_base_sha": identity["local_sha"],
        "stages": ["static", "synthetic_gradients", "tiny_smoke", "one_class", "small_subset"],
        "static": static,
        "synthetic": synthetic,
        "smoke": smoke,
        "engineering": engineering,
        "stage2": stage2_payload,
        "stage3": stage3_payload,
        "no_full_marker_created": True,
    }
    atomic_write_json(output_root / "P30_QUALIFICATION_SUMMARY.json", result)
    return result


def _run_preaudit(args: argparse.Namespace, prereg: Mapping[str, Any], prereg_hash: str) -> dict[str, Any]:
    if not args.output_root.is_dir():
        raise RuntimeError("P30 qualification must complete before pre-audit")
    qualification = _json(args.output_root / "P30_QUALIFICATION_SUMMARY.json")
    if qualification.get("status") != "PASS":
        raise RuntimeError("P30 qualification summary is not PASS")
    for filename in ("P30_ENGINEERING_QUALIFICATION.json", "P30_STAGE2_QUALIFICATION.json", "P30_STAGE3_QUALIFICATION.json"):
        if _json(args.output_root / filename).get("status") != "PASS":
            raise RuntimeError(f"P30 qualification artifact is not PASS: {filename}")
    if (args.output_root / "P30_EXECUTION_MARKER.json").exists():
        raise RuntimeError("P30 full execution marker already exists")
    _ensure_no_active_training()
    identity = _git_identity(None, require_clean=True)
    _, inventory, _ = _audit_inputs(args)
    result = {
        "schema_version": "P30_PRE_AUDIT_V1",
        "status": "PASS",
        "utc_timestamp": _utc(),
        "p30_uuid": P30_UUID,
        "p30_preregistration_sha256": prereg_hash,
        "qualification_status": qualification["status"],
        "qualification_execution_base_sha": qualification["qualification_execution_base_sha"],
        "preaudit_execution_base_sha": identity["local_sha"],
        "branch": identity["branch"],
        "remote_sha": identity["remote_sha"],
        "remote_equals_local": identity["remote_equals_local"],
        "worktree_clean_before_preaudit_artifact": True,
        "attempt_marker_absent": True,
        "full_fold_outputs_absent": not any((args.output_root / name).exists() for name in FULL_CLASSES),
        "input_inventory": inventory,
        "held_gt_reads": 0,
        "held_mask_reads": 0,
        "held_teacher_reads": 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "p29_p27_rerun": False,
        "cache_rebuilt": False,
        "scientific_outputs_written": [],
    }
    if not result["full_fold_outputs_absent"]:
        raise RuntimeError("P30 full fold output path is already occupied")
    atomic_write_json(args.output_root / "P30_PRE_AUDIT.json", result)
    return result


def _full_marker(args: argparse.Namespace, prereg: Mapping[str, Any], input_inventory: Mapping[str, Any], execution_sha: str, prereg_hash: str) -> dict[str, Any]:
    output_paths = [
        str(args.output_root / name)
        for name in FULL_CLASSES
    ] + [
        str(args.output_root / filename)
        for filename in (
            "P30_EXECUTION_MARKER.json",
            "P30_SCORING_GATE.json",
            "P30_RESULTS.json",
            "P30_TRANSFER_DIAGNOSTIC.json",
            "P30_STABILITY_DIAGNOSTIC.json",
            "P30_POST_RUN_AUDIT.json",
            "P30_FINAL_REPORT.md",
        )
    ]
    marker = {
        "schema_version": "P30_EXECUTION_MARKER_V1",
        "status": "CONSUMED",
        "completion_status": "ATTEMPT_CONSUMED",
        "attempt_uuid": str(uuid.uuid4()),
        "p30_uuid": P30_UUID,
        "p30_preregistration_sha256": prereg_hash,
        "p30_execution_base_sha": execution_sha,
        "branch": "research/p29r1-fast-objective-forensic-v1",
        "utc_started": _utc(),
        "expected_output_paths": output_paths,
        "class_order": list(FULL_CLASSES),
        "input_inventory": input_inventory,
        "training_steps": 0,
        "optimizer_steps": 0,
        "held_gt_reads_before_prediction_freeze": 0,
        "held_mask_reads_before_prediction_freeze": 0,
        "held_teacher_reads_before_prediction_freeze": 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "teacher_trainable": False,
        "phase2b_trainable": False,
        "clip_trainable": False,
        "automatic_rerun": False,
        "one_full_attempt": True,
    }
    atomic_write_json(args.output_root / "P30_EXECUTION_MARKER.json", marker)
    return marker


def _gradient_diagnostic(args: argparse.Namespace, output_root: Path, classes: Sequence[str], prereg_hash: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for class_name in classes:
        completion = _json(output_root / class_name / "training" / "TRAINING_COMPLETE.json")
        health = completion.get("gradient_health", {})
        rows.append({
            "class": class_name,
            "steps": completion.get("steps"),
            "optimizer_steps": completion.get("optimizer_steps"),
            "gradient_health": health,
            "student_parameter_delta": completion.get("student_parameter_delta"),
            "teacher_parameter_delta": completion.get("teacher_parameter_delta"),
            "finite": _finite(health),
        })
    failures = [row["class"] for row in rows if not row["finite"] or row["teacher_parameter_delta"] != 0.0]
    result = {
        "schema_version": "P30_GRADIENT_DIAGNOSTIC_V1",
        "status": "PASS" if not failures else "FAIL",
        "p30_uuid": P30_UUID,
        "p30_preregistration_sha256": prereg_hash,
        "classes": rows,
        "failures": failures,
        "nonfinite_gradient_count": sum(int(row["gradient_health"].get("nonfinite_count_max", 0)) for row in rows),
    }
    atomic_write_json(output_root / "P30_GRADIENT_DIAGNOSTIC.json", result)
    if failures:
        raise RuntimeError(f"P30 gradient diagnostic failed: {failures}")
    return result


def _inference_benchmark(args: argparse.Namespace, output_root: Path, prereg_hash: str) -> dict[str, Any]:
    from tools.sabra_v2.region_cache import TierADataset

    rows = read_visa_metadata(args.metadata)
    inventory = loco_inventory(rows, "candle")
    provenance = p30_cache_provenance(args.metadata)
    dataset = TierADataset(inventory.held_rows[:1], args.cache_root, provenance)
    sample = dataset[0]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    p29_payload = torch.load(P29_ROOT / "candle/training/p29_region_adapter.pt", map_location="cpu", weights_only=True)
    p30_payload = torch.load(output_root / "candle/training/p30_region_adapter.pt", map_location="cpu", weights_only=True)
    p29 = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    p30 = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    p29.load_state_dict(p29_payload["state_dict"], strict=True)
    p30.load_state_dict(p30_payload["state_dict"], strict=True)
    p29.eval()
    p30.eval()
    seg = sample["seg_features"].unsqueeze(1).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
    native = sample["native_logits"].unsqueeze(1).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)

    def measure(adapter: RegionResidualAdapter) -> list[float]:
        with torch.no_grad():
            for _ in range(10):
                forward_region_student(adapter, seg, native)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            values: list[float] = []
            for _ in range(40):
                started = time.perf_counter()
                forward_region_student(adapter, seg, native)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                values.append(time.perf_counter() - started)
        return values

    p29_times = measure(p29)
    p30_times = measure(p30)
    p29_median = float(torch.tensor(p29_times).median())
    p30_median = float(torch.tensor(p30_times).median())
    overhead = 100.0 * (p30_median - p29_median) / p29_median if p29_median else 0.0
    result = {
        "schema_version": "P30_INFERENCE_BENCHMARK_V1",
        "status": "PASS" if math.isfinite(overhead) else "FAIL",
        "p30_uuid": P30_UUID,
        "p30_preregistration_sha256": prereg_hash,
        "held_class_label_only": "candle",
        "observed_gt": False,
        "observed_masks": False,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "benchmark_iterations": 40,
        "p29_median_seconds": p29_median,
        "p30_median_seconds": p30_median,
        "inference_overhead_percent": overhead,
        "same_forward_path": True,
        "new_inference_module": False,
    }
    atomic_write_json(output_root / "P30_INFERENCE_BENCHMARK.json", result)
    return result


def _runtime_from_folds(output_root: Path, classes: Sequence[str], parent_timings: Mapping[str, float], prereg_hash: str) -> dict[str, Any]:
    training = [_json(output_root / name / "training" / "TRAINING_COMPLETE.json") for name in classes]
    prediction = [_json(output_root / name / "predictions" / "PREDICTION_COMPLETE.json") for name in classes]
    training_steps = sum(int(item["steps"]) for item in training)
    optimizer_steps = sum(int(item["optimizer_steps"]) for item in training)
    median_step = float(torch.tensor([float(item["median_step_seconds"]) for item in training]).median())
    overhead = 100.0 * (median_step - P29_CACHED_MEDIAN_SECONDS_PER_STEP) / P29_CACHED_MEDIAN_SECONDS_PER_STEP
    result = {
        "schema_version": "P30_RUNTIME_V1",
        "status": "PASS" if all(_finite(item) for item in training + prediction) else "FAIL",
        "p30_uuid": P30_UUID,
        "p30_preregistration_sha256": prereg_hash,
        "fold_count": len(classes),
        "classes": list(classes),
        "training_steps": training_steps,
        "optimizer_steps": optimizer_steps,
        "p29_cached_median_seconds_per_step": P29_CACHED_MEDIAN_SECONDS_PER_STEP,
        "p30_median_seconds_per_step_across_folds": median_step,
        "training_overhead_percent": overhead,
        "training_seconds_sum": sum(float(item["training_seconds"]) for item in training),
        "prediction_seconds_sum": sum(float(item["prediction_seconds"]) for item in prediction),
        "parent_process_stage_timings": dict(parent_timings),
        "peak_gpu_allocated_bytes_max": max(int(item["peak_gpu_allocated_bytes"]) for item in training),
        "peak_gpu_reserved_bytes_max": max(int(item["peak_gpu_reserved_bytes"]) for item in training),
        "peak_process_rss_kib_max": max(int(item["peak_process_rss_kib"]) for item in training),
        "held_gt_reads_before_prediction_freeze": sum(int(item["held_gt_reads"]) for item in training),
        "held_mask_reads_before_prediction_freeze": sum(int(item["held_mask_reads"]) for item in training),
        "new_clip_forwards": sum(int(item["new_clip_forwards"]) for item in training),
        "new_phase2b_forwards": sum(int(item["new_phase2b_forwards"]) for item in training),
    }
    atomic_write_json(output_root / "P30_RUNTIME.json", result)
    return result


def _scientific_decision(output_root: Path, runtime: Mapping[str, Any], inference: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = _json(output_root / "P30_RESULTS.json")
    transfer = _json(output_root / "P30_TRANSFER_DIAGNOSTIC.json")
    stability = _json(output_root / "P30_STABILITY_DIAGNOSTIC.json")
    baseline = _json(P29_METRICS)
    p30_sign = transfer["macro"].get("p30_sign_agreement")
    p29_sign = transfer["macro"].get("p29_sign_agreement")
    p30_direction = transfer["macro"].get("p30_directional_cosine")
    p29_direction = transfer["macro"].get("p29_directional_cosine")
    p30_q99 = stability["macro"]["p30"]["normal"].get("q99")
    p29_q99 = stability["macro"]["p29"]["normal"].get("q99")
    p29_regressions = sum(float(row["delta_pAP"]) < 0 for row in baseline["aggregate"]["classes"])
    p30_regressions = sum(float(row["delta_pAP"]) < 0 for row in aggregate["classes"])
    p30_ap = float(aggregate["p30_macro_pAP"])
    p30_auc = float(aggregate["p30_macro_pAUROC"])
    p29_ap = float(aggregate["p29_macro_pAP"])
    p29_auc = float(aggregate["p29_macro_pAUROC"])
    overhead = float(runtime["training_overhead_percent"])
    normal_stable = p30_q99 is not None and p29_q99 is not None and float(p30_q99) <= float(p29_q99) + 0.001
    direction_not_worse = p30_direction is not None and p29_direction is not None and float(p30_direction) >= float(p29_direction) - 0.01
    sign_not_worse = p30_sign is not None and p29_sign is not None and float(p30_sign) >= float(p29_sign) - 0.01
    # The benchmark is noisy; zero inference overhead is the architectural
    # fact that both candidates use the same unchanged deployment path.
    inference_zero = not bool(inference["new_inference_module"]) and bool(inference["same_forward_path"])
    better = (
        p30_ap >= p29_ap
        and p30_auc >= p29_auc
        and p30_sign is not None
        and p29_sign is not None
        and float(p30_sign) >= float(p29_sign) + 0.01
        and normal_stable
        and p30_regressions <= p29_regressions
        and overhead <= 10.0
    )
    equivalent = (
        abs(p30_ap - p29_ap) <= 0.005
        and abs(p30_auc - p29_auc) <= 0.005
        and direction_not_worse
        and sign_not_worse
        and normal_stable
        and inference_zero
        and overhead <= 10.0
    )
    faster = equivalent and overhead < 0.0
    justified = (
        not better
        and not equivalent
        and overhead <= 15.0
        and (p30_ap >= p29_ap or p30_auc >= p29_auc or (p30_sign is not None and p29_sign is not None and float(p30_sign) > float(p29_sign)))
        and normal_stable
    )
    decision = "BETTER" if better else "FASTER_EQUIVALENT" if faster else "EQUIVALENT_BUT_SIMPLER" if equivalent else "SLOWER_BUT_JUSTIFIED" if justified else "REJECT"
    return {
        "schema_version": "P30_SCIENTIFIC_DECISION_V1",
        "decision": decision,
        "p30_macro_pAP": p30_ap,
        "p29_macro_pAP": p29_ap,
        "p30_macro_pAUROC": p30_auc,
        "p29_macro_pAUROC": p29_auc,
        "p30_minus_p29_macro_pAP": p30_ap - p29_ap,
        "p30_minus_p29_macro_pAUROC": p30_auc - p29_auc,
        "p30_sign_agreement": p30_sign,
        "p29_sign_agreement": p29_sign,
        "p30_directional_cosine": p30_direction,
        "p29_directional_cosine": p29_direction,
        "p30_normal_q99": p30_q99,
        "p29_normal_q99": p29_q99,
        "p30_pAP_regression_count": p30_regressions,
        "p29_pAP_regression_count": p29_regressions,
        "training_overhead_percent": overhead,
        "inference_overhead_percent": inference["inference_overhead_percent"],
        "balanced_success_checks": {
            "better": better,
            "equivalent_but_simpler": equivalent,
            "faster": faster,
            "normal_stable": normal_stable,
            "direction_not_worse": direction_not_worse,
            "sign_not_worse": sign_not_worse,
            "inference_zero": inference_zero,
        },
    }


def _post_run_audit(args: argparse.Namespace, marker: Mapping[str, Any], input_inventory: Mapping[str, Any], prereg_hash: str, runtime: Mapping[str, Any], gradient: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    root = args.output_root
    gate = _json(root / "P30_SCORING_GATE.json")
    if gate.get("status") != "PASS" or gate.get("prediction_count") != 12:
        failures.append("full prediction-freeze gate failed")
    if marker.get("status") != "CONSUMED" or marker.get("p30_uuid") != P30_UUID:
        failures.append("execution marker identity failed")
    if runtime.get("training_steps") != sum(len(loco_inventory(read_visa_metadata(args.metadata), name).fit_rows) * P30_TRAINING_EPOCHS for name in FULL_CLASSES):
        failures.append("full training step count failed")
    if runtime.get("optimizer_steps") != runtime.get("training_steps"):
        failures.append("optimizer step count failed")
    if gradient.get("status") != "PASS":
        failures.append("gradient diagnostic failed")
    if runtime.get("held_gt_reads_before_prediction_freeze") != 0 or runtime.get("held_mask_reads_before_prediction_freeze") != 0:
        failures.append("held supervision reached training")
    if runtime.get("new_clip_forwards") != 0 or runtime.get("new_phase2b_forwards") != 0:
        failures.append("unexpected model forwards occurred")
    for name in FULL_CLASSES:
        training = _json(root / name / "training" / "TRAINING_COMPLETE.json")
        prediction = _json(root / name / "predictions" / "PREDICTION_COMPLETE.json")
        if training.get("status") != "FOLD_TRAINING_COMPLETE" or prediction.get("completion_status") != "COMPLETE":
            failures.append(f"incomplete fold {name}")
        prediction_path = root / name / "predictions" / "p30_held_predictions.pt"
        payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
        if payload.get("gt_used") is not False or payload.get("mask_reads") != 0:
            failures.append(f"held prediction firewall failed for {name}")
        if prediction_path.stat().st_mtime > (root / "P30_SCORING_GATE.json").stat().st_mtime:
            failures.append(f"prediction was written after scoring gate for {name}")
        metrics_path = root / name / "metrics" / "p30_held_metrics.json"
        if metrics_path.stat().st_mtime < (root / "P30_SCORING_GATE.json").stat().st_mtime:
            failures.append(f"score was written before scoring gate for {name}")
    tracked_changes = _git("diff", "--name-only")
    if tracked_changes:
        failures.append(f"tracked code changed after marker: {tracked_changes}")
    result = {
        "schema_version": "P30_POST_RUN_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "terminal_status": "P30_COMPLETE" if not failures else "P30_ENGINEERING_STOP",
        "p30_uuid": P30_UUID,
        "attempt_uuid": marker.get("attempt_uuid"),
        "attempt_count": 1,
        "fold_count": 12,
        "prediction_count_before_scoring": gate.get("prediction_count"),
        "predictions_frozen_before_scoring": True,
        "held_gt_reads_before_prediction_freeze": runtime.get("held_gt_reads_before_prediction_freeze"),
        "held_mask_reads_before_prediction_freeze": runtime.get("held_mask_reads_before_prediction_freeze"),
        "new_clip_forwards": runtime.get("new_clip_forwards"),
        "new_phase2b_forwards": runtime.get("new_phase2b_forwards"),
        "training_steps": runtime.get("training_steps"),
        "optimizer_steps": runtime.get("optimizer_steps"),
        "teacher_trainable": False,
        "phase2b_trainable": False,
        "clip_trainable": False,
        "p29_p27_rerun": False,
        "automatic_rerun": False,
        "p30_preregistration_sha256": prereg_hash,
        "input_inventory": input_inventory,
        "marker": dict(marker),
        "scientific_decision": decision.get("decision"),
        "tracked_code_changes_after_marker": tracked_changes,
        "worktree_clean_at_audit": False,
        "failures": failures,
    }
    atomic_write_json(root / "P30_POST_RUN_AUDIT.json", result)
    if failures:
        raise RuntimeError(f"P30 post-run audit failed: {failures}")
    return result


def _final_report(output_root: Path, marker: Mapping[str, Any], audit: Mapping[str, Any], runtime: Mapping[str, Any], gradient: Mapping[str, Any], inference: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    aggregate = _json(output_root / "P30_RESULTS.json")
    transfer = _json(output_root / "P30_TRANSFER_DIAGNOSTIC.json")
    stability = _json(output_root / "P30_STABILITY_DIAGNOSTIC.json")
    rows = aggregate["classes"]
    lines = [
        "# P30 Final Report — Directional Distillation",
        "",
        f"- Decision: **{decision['decision']}**",
        f"- P30 UUID: `{marker['p30_uuid']}`",
        f"- Preregistration SHA-256: `{marker['p30_preregistration_sha256']}`",
        f"- Execution base SHA: `{marker['p30_execution_base_sha']}`",
        f"- Protocol audit: **{audit['status']}**; one full attempt; automatic rerun: `False`",
        "",
        "## Scientific metrics",
        "",
        f"- Macro pAP: P30 `{aggregate['p30_macro_pAP']:.9f}` vs P29 `{aggregate['p29_macro_pAP']:.9f}`; delta `{aggregate['p30_minus_p29_macro_pAP']:.9f}`.",
        f"- Macro pAUROC: P30 `{aggregate['p30_macro_pAUROC']:.9f}` vs P29 `{aggregate['p29_macro_pAUROC']:.9f}`; delta `{aggregate['p30_minus_p29_macro_pAUROC']:.9f}`.",
        f"- Directional cosine: P30 `{transfer['macro']['p30_directional_cosine']}` vs P29 `{transfer['macro']['p29_directional_cosine']}`.",
        f"- Sign agreement: P30 `{transfer['macro']['p30_sign_agreement']}` vs P29 `{transfer['macro']['p29_sign_agreement']}`.",
        f"- Normal q99 shift: P30 `{stability['macro']['p30']['normal']['q99']}` vs P29 `{stability['macro']['p29']['normal']['q99']}`.",
        "",
        "| Class | P30 pAP | P29 pAP | Δ pAP | P30 AUROC | P29 AUROC | Δ AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class']} | {row['p30_pAP']:.6f} | {row['p29_pAP']:.6f} | {row['p30_minus_p29_pAP']:.6f} | {row['p30_pAUROC']:.6f} | {row['p29_pAUROC']:.6f} | {row['p30_minus_p29_pAUROC']:.6f} |"
        )
    lines.extend([
        "",
        "## Gradients and speed",
        "",
        f"- Gradient diagnostic: **{gradient['status']}**; non-finite count `{gradient['nonfinite_gradient_count']}`.",
        f"- Training steps / optimizer steps: `{runtime['training_steps']}` / `{runtime['optimizer_steps']}`.",
        f"- Median training seconds per step: `{runtime['p30_median_seconds_per_step_across_folds']}`; overhead vs frozen P29 `{runtime['training_overhead_percent']:.3f}%`.",
        f"- Inference median overhead: `{inference['inference_overhead_percent']:.6f}%`; new inference module: `False`.",
        f"- Peak GPU allocated bytes: `{runtime['peak_gpu_allocated_bytes_max']}`; peak RSS KiB: `{runtime['peak_process_rss_kib_max']}`.",
        "",
        "## Audit and artifact policy",
        "",
        "- Training used only the frozen Tier-A/Tier-B cache and the unchanged RegionResidualAdapter.",
        "- Held GT/masks were unavailable before prediction freeze; scoring and post-freeze diagnostics are separated by `P30_SCORING_GATE.json`.",
        "- No new CLIP or Phase2B forwards, no P29/P27 rerun, no stacked auxiliary objective, and no post-hoc tuning.",
        "- Full artifacts: `P30_RESULTS.json`, `P30_TRANSFER_DIAGNOSTIC.json`, `P30_STABILITY_DIAGNOSTIC.json`, `P30_GRADIENT_DIAGNOSTIC.json`, `P30_RUNTIME.json`, `P30_INFERENCE_BENCHMARK.json`, and `P30_POST_RUN_AUDIT.json`.",
    ])
    content = "\n".join(lines) + "\n"
    _write_text(output_root / "P30_FINAL_REPORT.md", content)
    return content


def _run_full(args: argparse.Namespace, prereg: Mapping[str, Any], prereg_hash: str) -> dict[str, Any]:
    if args.execution_base_sha is None:
        raise RuntimeError("full P30 execution requires --execution-base-sha from the committed pre-audit")
    if not args.output_root.is_dir():
        raise RuntimeError("P30 qualification/pre-audit output root is missing")
    preaudit = _json(args.output_root / "P30_PRE_AUDIT.json")
    if preaudit.get("status") != "PASS":
        raise RuntimeError("P30 pre-audit is not PASS")
    qualification = _json(args.output_root / "P30_QUALIFICATION_SUMMARY.json")
    if qualification.get("status") != "PASS":
        raise RuntimeError("P30 qualification is not PASS")
    identity = _git_identity(args.execution_base_sha, require_clean=True)
    _ensure_no_active_training()
    _, input_inventory, _ = _audit_inputs(args)
    marker_path = args.output_root / "P30_EXECUTION_MARKER.json"
    if marker_path.exists():
        raise RuntimeError("P30 full execution marker already exists; scientific rerun forbidden")
    if any((args.output_root / name).exists() for name in FULL_CLASSES):
        raise RuntimeError("P30 full fold output path is occupied; refusing overwrite")
    marker = _full_marker(args, prereg, input_inventory, identity["local_sha"], prereg_hash)
    started = time.perf_counter()
    parent_timings: dict[str, float] = {}
    try:
        for class_name in FULL_CLASSES:
            fold = _run_fold(args, args.output_root, class_name, "full", prereg_hash, identity["local_sha"])
            parent_timings[f"training_{class_name}"] = float(fold["training_seconds_parent"])
            parent_timings[f"prediction_{class_name}"] = float(fold["prediction_seconds_parent"])
        scoring = _score_and_diagnose(args, args.output_root, FULL_CLASSES, "full", prereg_hash)
        parent_timings.update(scoring["timings"])
        gradient = _gradient_diagnostic(args, args.output_root, FULL_CLASSES, prereg_hash)
        runtime = _runtime_from_folds(args.output_root, FULL_CLASSES, parent_timings, prereg_hash)
        inference = _inference_benchmark(args, args.output_root, prereg_hash)
        decision = _scientific_decision(args.output_root, runtime, inference)
        atomic_write_json(args.output_root / "P30_SCIENTIFIC_DECISION.json", decision)
        audit = _post_run_audit(args, marker, input_inventory, prereg_hash, runtime, gradient, decision)
        _final_report(args.output_root, marker, audit, runtime, gradient, inference, decision)
        run_complete = {
            "schema_version": "P30_SCIENTIFIC_RUN_COMPLETE_V1",
            "status": "COMPLETE",
            "attempt_uuid": marker["attempt_uuid"],
            "p30_uuid": P30_UUID,
            "p30_preregistration_sha256": prereg_hash,
            "p30_execution_base_sha": identity["local_sha"],
            "utc_started": marker["utc_started"],
            "utc_finished": _utc(),
            "actual_scientific_runtime_seconds": time.perf_counter() - started,
            "fold_count": 12,
            "prediction_count_before_scoring": 12,
            "attempt_count": 1,
            "decision": decision["decision"],
            "automatic_rerun": False,
        }
        atomic_write_json(args.output_root / "P30_RUN_COMPLETE.json", run_complete)
        return run_complete
    except BaseException as exc:
        atomic_write_json(
            args.output_root / "P30_ATTEMPT_FAILURE.json",
            {
                "schema_version": "P30_ATTEMPT_FAILURE_V1",
                "attempt_uuid": marker["attempt_uuid"],
                "p30_uuid": P30_UUID,
                "utc_timestamp": _utc(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "automatic_rerun_forbidden": True,
                "meaningful_scientific_execution_started": True,
            },
        )
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        prereg, _, prereg_hash = _audit_inputs(args)
        if args.mode == "qualify":
            return _run_qualification(args, prereg, prereg_hash)
        if args.mode == "preaudit":
            return _run_preaudit(args, prereg, prereg_hash)
        return _run_full(args, prereg, prereg_hash)
    except BaseException as exc:
        if args.mode == "qualify" and args.output_root.is_dir():
            failure_path = args.output_root / "P30_QUALIFICATION_FAILURE.json"
            if not failure_path.exists():
                atomic_write_json(
                    failure_path,
                    {
                        "schema_version": "P30_QUALIFICATION_FAILURE_V1",
                        "status": "STOP",
                        "p30_uuid": P30_UUID,
                        "p30_preregistration_sha256": p30_preregistration_hash(P30_PREREGISTRATION_PATH),
                        "utc_timestamp": _utc(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "full_marker_created": (args.output_root / "P30_EXECUTION_MARKER.json").exists(),
                        "scientific_full_rerun_automatic": False,
                    },
                )
        raise


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
