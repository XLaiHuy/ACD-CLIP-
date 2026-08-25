"""One-shot durable P27 cache/train/predict/score runner with a scoring barrier."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata
from tools.sabra_v2.audit_region_distill import PROTOCOL_PATH, audit_protocol
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.region_cache import atomic_write_json, sha256_file
from tools.sabra_v2.train_region_distill import ROOT


EXACT_HELD_ORDER = tuple(EXPECTED_VISA_CLASSES)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--p26-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--execution-base-sha", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _remote_sha(branch: str) -> str:
    output = _git("ls-remote", "origin", f"refs/heads/{branch}")
    fields = output.split()
    if len(fields) != 2:
        raise RuntimeError(f"could not resolve remote branch {branch}")
    return fields[0]


def _run(module: str, arguments: list[str], environment: dict[str, str]) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print(json.dumps({"event": "START", "utc": _utc(), "command": command}), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print(json.dumps({"event": "COMPLETE", "utc": _utc(), "module": module}), flush=True)


def _verify_frozen_git(execution_base: str) -> tuple[str, str]:
    branch = _git("branch", "--show-current")
    local = _git("rev-parse", "HEAD")
    if local != execution_base:
        raise RuntimeError(f"execution-base mismatch: expected {execution_base}, got {local}")
    if _git("status", "--porcelain"):
        raise RuntimeError("scientific execution requires a clean worktree")
    remote = _remote_sha(branch)
    if remote != local:
        raise RuntimeError(f"remote/local mismatch: {remote} != {local}")
    return branch, remote


def _prediction_gate(run_root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for held_class in EXACT_HELD_ORDER:
        completion_path = run_root / held_class / "predictions" / "PREDICTION_COMPLETE.json"
        prediction_path = run_root / held_class / "predictions" / "p27_held_predictions.pt"
        if not completion_path.is_file() or not prediction_path.is_file():
            raise RuntimeError(f"all 12 predictions must freeze before scoring; missing {held_class}")
        completion = json.loads(completion_path.read_text())
        if completion.get("completion_status") != "COMPLETE" or completion.get("held_class") != held_class:
            raise RuntimeError(f"invalid prediction completion marker for {held_class}")
        observed_hash = sha256_file(prediction_path)
        if completion.get("prediction_sha256") != observed_hash:
            raise RuntimeError(f"prediction hash mismatch for {held_class}")
        if prediction_path.stat().st_mode & 0o222:
            raise RuntimeError(f"prediction artifact is not immutable: {prediction_path}")
        artifacts.append({"held_class": held_class, "path": str(prediction_path), "sha256": observed_hash})
    return artifacts


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    audit_protocol(protocol)
    verify_p26_parent(args.p26_checkpoint, args.clip_asset, ROOT / "configs/phase2b_canonical_v1.json")
    branch, remote = _verify_frozen_git(args.execution_base_sha)
    rows = read_visa_metadata(args.metadata)
    if tuple(sorted({str(row["class_name"]) for row in rows})) != tuple(sorted(EXACT_HELD_ORDER)):
        raise RuntimeError("exact VisA class inventory failed")
    args.run_root.mkdir(parents=True, exist_ok=True)
    attempt_path = args.run_root / "P27_ATTEMPT.json"
    if attempt_path.exists():
        raise RuntimeError("P27 attempt marker already exists; automatic rerun is forbidden")
    if any((args.run_root / class_name).exists() for class_name in EXACT_HELD_ORDER):
        raise RuntimeError("fold artifact exists before the one-shot attempt marker")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,uuid", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    attempt_uuid = str(uuid.uuid4())
    attempt = {
        "schema_version": "P27_SCIENTIFIC_ATTEMPT_V1",
        "completion_status": "ATTEMPT_CONSUMED",
        "attempt_uuid": attempt_uuid,
        "utc_timestamp": _utc(),
        "scientific_execution_base_sha": args.execution_base_sha,
        "branch": branch,
        "remote_sha": remote,
        "p27_protocol_schema": protocol["schema_version"],
        "p27_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "p26_sha256": sha256_file(args.p26_checkpoint),
        "clip_sha256": sha256_file(args.clip_asset),
        "config_sha256": sha256_file(ROOT / "configs/phase2b_canonical_v1.json"),
        "gpu": gpu,
        "visa_root": str(args.visa_root.resolve()),
        "metadata_sha256": sha256_file(args.metadata),
        "held_order": list(EXACT_HELD_ORDER),
        "cache_configuration": {
            "schema": "P27_CACHE_V1",
            "tier_a": "class-sharded numpy .npy memmap; GT-free cross-fold reuse",
            "tier_b": "fold-local numpy .npy memmap; source inventory only",
            "dtype": "float32 preserved",
            "num_workers": 0,
            "pin_memory": False,
            "non_blocking": False,
            "prefetch_factor": None,
            "cublas_workspace_config": ":4096:8",
            "seed": args.seed,
        },
        "p27_full_scientific_runs_consumed": 1,
        "mvtec_reads": 0,
        "medical_reads": 0,
    }
    atomic_write_json(attempt_path, attempt)
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    common = [
        "--visa-root", str(args.visa_root),
        "--p26-checkpoint", str(args.p26_checkpoint),
        "--clip-asset", str(args.clip_asset),
        "--cache-root", str(args.cache_root),
        "--metadata", str(args.metadata),
        "--execution-base-sha", args.execution_base_sha,
    ]
    started = time.perf_counter()
    try:
        _run("tools.sabra_v2.build_region_cache", ["--tier", "a", *common, "--device", "cuda", "--num-workers", "0"], environment)
        for held_class in EXACT_HELD_ORDER:
            fold_root = args.run_root / held_class
            _run(
                "tools.sabra_v2.build_region_cache",
                ["--tier", "b", "--held-class", held_class, *common, "--device", "cuda"],
                environment,
            )
            _run(
                "tools.sabra_v2.train_region_distill_cached",
                [
                    "--held-class", held_class,
                    *common,
                    "--output", str(fold_root / "training"),
                    "--epochs", "20",
                    "--batch-size", "1",
                    "--learning-rate", "0.001",
                    "--seed", str(args.seed),
                    "--device", "cuda",
                    "--num-workers", "0",
                ],
                environment,
            )
            _run(
                "tools.sabra_v2.evaluate_region_distill_cached",
                [
                    "--held-class", held_class,
                    *common,
                    "--adapter-checkpoint", str(fold_root / "training" / "p27_region_adapter.pt"),
                    "--output", str(fold_root / "predictions"),
                    "--batch-size", "1",
                    "--device", "cuda",
                    "--num-workers", "0",
                ],
                environment,
            )
        predictions = _prediction_gate(args.run_root)
        scoring_gate = {
            "schema_version": "P27_SCORING_GATE_V1",
            "completion_status": "PASS",
            "utc_timestamp": _utc(),
            "prediction_count": len(predictions),
            "predictions": predictions,
            "fit_or_teacher_steps_after_gate": 0,
        }
        atomic_write_json(args.run_root / "P27_SCORING_GATE.json", scoring_gate)
        for held_class in EXACT_HELD_ORDER:
            fold_root = args.run_root / held_class
            _run(
                "tools.sabra_v2.score_region_distill_frozen",
                [
                    "--held-class", held_class,
                    "--visa-root", str(args.visa_root),
                    "--predictions", str(fold_root / "predictions" / "p27_held_predictions.pt"),
                    "--output", str(fold_root / "metrics"),
                    "--metadata", str(args.metadata),
                ],
                environment,
            )
        _run(
            "tools.sabra_v2.aggregate_region_distill",
            ["--run-root", str(args.run_root), "--output", str(args.run_root / "aggregate")],
            environment,
        )
        result = {
            "schema_version": "P27_SCIENTIFIC_RUN_COMPLETE_V1",
            "completion_status": "COMPLETE",
            "attempt_uuid": attempt_uuid,
            "utc_timestamp": _utc(),
            "scientific_execution_base_sha": args.execution_base_sha,
            "actual_scientific_runtime_seconds": time.perf_counter() - started,
            "fold_count": 12,
            "prediction_count_before_scoring": 12,
            "attempt_count": 1,
            "mvtec_reads": 0,
            "medical_reads": 0,
        }
        atomic_write_json(args.run_root / "P27_RUN_COMPLETE.json", result)
        return result
    except BaseException as exc:
        atomic_write_json(
            args.run_root / "P27_ATTEMPT_FAILURE.json",
            {
                "schema_version": "P27_ATTEMPT_FAILURE_V1",
                "attempt_uuid": attempt_uuid,
                "utc_timestamp": _utc(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "automatic_rerun_forbidden": True,
            },
        )
        raise


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
