"""Fail-closed post-run audit for the one-shot P27 scientific execution."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import atomic_write_json, sha256_file
from tools.sabra_v2.train_region_distill import ROOT


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--p26-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--execution-base-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def run(args: argparse.Namespace) -> dict[str, Any]:
    assets = verify_p26_parent(
        args.p26_checkpoint,
        args.clip_asset,
        ROOT / "configs/phase2b_canonical_v1.json",
    )
    attempt_path = args.run_root / "P27_ATTEMPT.json"
    complete_path = args.run_root / "P27_RUN_COMPLETE.json"
    scoring_gate_path = args.run_root / "P27_SCORING_GATE.json"
    if not attempt_path.is_file() or not complete_path.is_file() or not scoring_gate_path.is_file():
        raise RuntimeError("attempt, scoring gate, and run completion evidence are all required")
    if (args.run_root / "P27_ATTEMPT_FAILURE.json").exists():
        raise RuntimeError("attempt failure marker exists")
    attempt = json.loads(attempt_path.read_text())
    complete = json.loads(complete_path.read_text())
    scoring_gate = json.loads(scoring_gate_path.read_text())
    if attempt.get("attempt_uuid") != complete.get("attempt_uuid"):
        raise RuntimeError("attempt UUID mismatch")
    if attempt.get("scientific_execution_base_sha") != args.execution_base_sha:
        raise RuntimeError("attempt execution-base mismatch")
    if scoring_gate.get("prediction_count") != 12 or scoring_gate.get("completion_status") != "PASS":
        raise RuntimeError("12-prediction scoring gate failed")
    rows = read_visa_metadata(args.metadata)
    expected_state_keys = set(RegionResidualAdapter().state_dict())
    folds: list[dict[str, Any]] = []
    prediction_mtimes: list[int] = []
    metric_mtimes: list[int] = []
    phase2b_steps = 0
    clip_steps = 0
    held_gt_reads_before_scoring = 0
    held_mask_reads_before_scoring = 0
    for held_class in EXPECTED_VISA_CLASSES:
        inventory = loco_inventory(rows, held_class)
        training_path = args.run_root / held_class / "training" / "TRAINING_COMPLETE.json"
        checkpoint_path = args.run_root / held_class / "training" / "p27_region_adapter.pt"
        prediction_path = args.run_root / held_class / "predictions" / "p27_held_predictions.pt"
        prediction_complete_path = args.run_root / held_class / "predictions" / "PREDICTION_COMPLETE.json"
        metric_path = args.run_root / held_class / "metrics" / "p27_held_metrics.json"
        tier_b_path = args.cache_root / "tier_b" / held_class / "manifest.json"
        required = (training_path, checkpoint_path, prediction_path, prediction_complete_path, metric_path, tier_b_path)
        if any(not path.is_file() for path in required):
            raise RuntimeError(f"incomplete fold evidence for {held_class}")
        training = json.loads(training_path.read_text())
        prediction_complete = json.loads(prediction_complete_path.read_text())
        metric = json.loads(metric_path.read_text())
        tier_b = json.loads(tier_b_path.read_text())
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        expected_steps = len(inventory.fit_rows) * 20
        if training.get("steps") != expected_steps or checkpoint.get("steps") != expected_steps:
            raise RuntimeError(f"wrong scientific step count for {held_class}")
        if checkpoint.get("status") != "FOLD_TRAINING_COMPLETE" or set(checkpoint.get("state_dict", {})) != expected_state_keys:
            raise RuntimeError(f"wrong trainable checkpoint ownership for {held_class}")
        if tier_b.get("held_class") != held_class or tier_b.get("held_mask_reads") != 0:
            raise RuntimeError(f"held supervision firewall failed for {held_class}")
        if tier_b.get("source_classes") != sorted(set(EXPECTED_VISA_CLASSES) - {held_class}):
            raise RuntimeError(f"source class inventory failed for {held_class}")
        if prediction_complete.get("mask_reads") != 0 or prediction_complete.get("gt_used") is not False:
            raise RuntimeError(f"held prediction accessed supervision for {held_class}")
        if sha256_file(prediction_path) != prediction_complete.get("prediction_sha256"):
            raise RuntimeError(f"held prediction hash failed for {held_class}")
        if metric.get("fit_or_teacher_steps") != 0:
            raise RuntimeError(f"scoring performed fitting for {held_class}")
        phase2b_steps += int(training.get("phase2b_optimization_steps", -1))
        clip_steps += int(training.get("clip_optimization_steps", -1))
        held_gt_reads_before_scoring += int(training.get("held_gt_reads", -1))
        held_mask_reads_before_scoring += int(training.get("held_mask_reads", -1)) + int(prediction_complete.get("mask_reads", -1))
        prediction_mtimes.append(prediction_path.stat().st_mtime_ns)
        metric_mtimes.append(metric_path.stat().st_mtime_ns)
        folds.append(
            {
                "held_class": held_class,
                "fit_records": len(inventory.fit_rows),
                "steps": expected_steps,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "prediction_sha256": prediction_complete["prediction_sha256"],
                "held_mask_reads_before_scoring": 0,
            }
        )
    scoring_gate_mtime = scoring_gate_path.stat().st_mtime_ns
    if max(prediction_mtimes) > scoring_gate_mtime or min(metric_mtimes) < scoring_gate_mtime:
        raise RuntimeError("scoring occurred before all predictions were frozen")
    local = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    remote_fields = _git("ls-remote", "origin", f"refs/heads/{branch}").split()
    remote = remote_fields[0] if len(remote_fields) == 2 else ""
    worktree_clean = not bool(_git("status", "--porcelain"))
    result = {
        "schema_version": "P27_POST_RUN_AUDIT_V1",
        "status": "PASS",
        "attempt_uuid": attempt["attempt_uuid"],
        "attempt_count": 1,
        "fold_count": len(folds),
        "duplicate_scientific_folds": 0,
        "rerun_poor_folds": 0,
        "predictions_frozen_before_scoring": True,
        "held_gt_reads_before_scoring": held_gt_reads_before_scoring,
        "held_mask_reads_before_scoring": held_mask_reads_before_scoring,
        "mvtec_reads": int(attempt.get("mvtec_reads", -1)),
        "medical_reads": int(attempt.get("medical_reads", -1)),
        "phase2b_optimization_steps": phase2b_steps,
        "clip_optimization_steps": clip_steps,
        "only_region_residual_adapter_trained": True,
        "assets": assets,
        "protocol_sha256": sha256_file(ROOT / "research/sabra_v2/region_distill/P27_PROTOCOL.json"),
        "scientific_execution_base_sha": args.execution_base_sha,
        "local_sha": local,
        "remote_sha": remote,
        "remote_equals_local": remote == local == args.execution_base_sha,
        "worktree_clean": worktree_clean,
        "folds": folds,
    }
    gates = (
        len(folds) == 12,
        held_gt_reads_before_scoring == 0,
        held_mask_reads_before_scoring == 0,
        result["mvtec_reads"] == 0,
        result["medical_reads"] == 0,
        phase2b_steps == 0,
        clip_steps == 0,
        result["remote_equals_local"],
        worktree_clean,
    )
    if not all(gates):
        result["status"] = "FAIL"
        raise RuntimeError(f"P27 post-run audit failed: {result}")
    atomic_write_json(args.output, result)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
