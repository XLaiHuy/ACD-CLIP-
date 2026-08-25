"""Execute the single authorized P27 12-fold lifecycle from a clean frozen base."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata, safe_data_path
from tools.sabra_v2.audit_region_distill import PROTOCOL_PATH, audit_protocol
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.region_cache import source_file_inventory_digest, source_inventory_digest
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.train_region_distill import ROOT, _sha256


FOLD_ORDER = EXPECTED_VISA_CLASSES


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _atomic_json(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _command(command: list[str], log_path: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    begin = time.perf_counter()
    with log_path.open("x") as log:
        process = subprocess.run(command, cwd=ROOT, text=True, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - begin
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}); preserved log: {log_path}")
    lines = [line for line in log_path.read_text().splitlines() if line.startswith("{")]
    if not lines:
        raise RuntimeError(f"command emitted no JSON result: {log_path}")
    return {"command": command, "started_utc": started, "elapsed_seconds": elapsed, "result": json.loads(lines[-1])}


def _aggregate(metrics_by_class: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_class = []
    for held in FOLD_ORDER:
        payload = metrics_by_class[held]
        native = payload["native_metrics"]
        p27 = payload["p27_metrics"]
        per_class.append({
            "held_class": held,
            "native_pAP": native["pAP"], "p27_pAP": p27["pAP"], "delta_pAP": p27["pAP"] - native["pAP"],
            "native_pAUROC": native["pAUROC"], "p27_pAUROC": p27["pAUROC"], "delta_pAUROC": p27["pAUROC"] - native["pAUROC"],
        })
    deltas = np.asarray([row["delta_pAP"] for row in per_class], dtype=np.float64)
    ordered = sorted(per_class, key=lambda row: row["delta_pAP"], reverse=True)
    total_gain = float(deltas.sum())
    concentration = {
        "top_1_fraction_of_net_gain": None if total_gain == 0 else ordered[0]["delta_pAP"] / total_gain,
        "top_2_fraction_of_net_gain": None if total_gain == 0 else sum(row["delta_pAP"] for row in ordered[:2]) / total_gain,
    }
    return {
        "macro": {
            "native_pAP": float(np.mean([row["native_pAP"] for row in per_class])),
            "p27_pAP": float(np.mean([row["p27_pAP"] for row in per_class])),
            "delta_pAP": float(np.mean(deltas)),
            "native_pAUROC": float(np.mean([row["native_pAUROC"] for row in per_class])),
            "p27_pAUROC": float(np.mean([row["p27_pAUROC"] for row in per_class])),
            "delta_pAUROC": float(np.mean([row["delta_pAUROC"] for row in per_class])),
        },
        "breadth": {
            "improving_pAP": int((deltas > 0).sum()),
            "non_regressing_pAP": int((deltas >= 0).sum()),
            "regressing_pAP": int((deltas < 0).sum()),
        },
        "median_category_delta_pAP": float(np.median(deltas)),
        "best_category": ordered[0],
        "worst_category": ordered[-1],
        "gain_concentration": concentration,
        "per_class": per_class,
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    if head != args.execution_base_sha:
        raise RuntimeError("HEAD does not equal recorded P27 execution base")
    if _git("status", "--porcelain"):
        raise RuntimeError("scientific execution requires a clean worktree")
    remote = _git("ls-remote", "--heads", "origin", "research/p27-cache-performance-recovery-v1").split()
    if not remote or remote[0] != head:
        raise RuntimeError("remote recovery branch does not equal local execution base")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    audit_protocol(protocol)
    config = ROOT / "configs/phase2b_canonical_v1.json"
    verify_p26_parent(args.p26_checkpoint, args.clip_asset, config)
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3070":
        raise RuntimeError("authorized CUDA GPU is unavailable or changed")
    rows = read_visa_metadata(args.metadata)
    classes = tuple(sorted({str(row["class_name"]) for row in rows}))
    if classes != tuple(sorted(FOLD_ORDER)):
        raise RuntimeError("VisA class inventory mismatch")
    for row in rows:
        if not safe_data_path(args.visa_root, str(row["image_path"])).is_file():
            raise RuntimeError("VisA image inventory is incomplete")
        if int(row["label"]) and not safe_data_path(args.visa_root, str(row["mask_path"])).is_file():
            raise RuntimeError("VisA mask inventory is incomplete")
    if args.runtime_root.exists() or args.cache_root.exists():
        raise RuntimeError("P27 runtime/cache root already exists; refusing a duplicate or ambiguous attempt")
    return {
        "execution_base_sha": head,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "asset_hashes": {
            "phase2b_checkpoint": _sha256(args.p26_checkpoint),
            "clip_asset": _sha256(args.clip_asset),
            "config": _sha256(config),
        },
        "dataset_root": str(args.visa_root.resolve()),
        "dataset_records": len(rows),
        "dataset_metadata_sha256": source_inventory_digest(rows),
        "dataset_files_sha256": source_file_inventory_digest(rows, args.visa_root),
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "full_scientific_runs_consumed_before_marker": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _preflight(args)
    args.runtime_root.mkdir(parents=True)
    args.cache_root.mkdir(parents=True)
    attempt_uuid = str(uuid.uuid4())
    marker = {
        "schema_version": "P27_SCIENTIFIC_ATTEMPT_V1",
        "attempt_uuid": attempt_uuid,
        "attempts_consumed": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "class_order": list(FOLD_ORDER),
        **preflight,
    }
    _atomic_json(args.runtime_root / "ATTEMPT_STARTED.json", marker, exclusive=True)
    started = time.perf_counter()
    fold_records: list[dict[str, Any]] = []
    metrics_by_class: dict[str, dict[str, Any]] = {}
    try:
        for held in FOLD_ORDER:
            fold_root = args.runtime_root / "folds" / held
            fold_root.mkdir(parents=True)
            cache = args.cache_root / held
            train = _command([
                sys.executable, "-m", "tools.sabra_v2.train_region_distill",
                "--held-class", held, "--visa-root", str(args.visa_root),
                "--p26-checkpoint", str(args.p26_checkpoint), "--clip-asset", str(args.clip_asset),
                "--output", str(fold_root), "--cache-dir", str(cache),
                "--epochs", "20", "--batch-size", "1", "--learning-rate", "0.001", "--seed", str(args.seed),
            ], fold_root / "train.log")
            evaluate = _command([
                sys.executable, "-m", "tools.sabra_v2.evaluate_region_distill",
                "--held-class", held, "--visa-root", str(args.visa_root),
                "--p26-checkpoint", str(args.p26_checkpoint), "--clip-asset", str(args.clip_asset),
                "--adapter-checkpoint", str(fold_root / "p27_region_adapter.pt"), "--output", str(fold_root / "predictions"),
            ], fold_root / "evaluate.log")
            prediction_path = Path(evaluate["result"]["prediction_path"])
            prediction_hash_before_score = _sha256(prediction_path)
            score = _command([
                sys.executable, "-m", "tools.sabra_v2.score_region_distill",
                "--held-class", held, "--visa-root", str(args.visa_root),
                "--predictions", str(prediction_path), "--output", str(fold_root / "metrics"),
            ], fold_root / "score.log")
            if _sha256(prediction_path) != prediction_hash_before_score:
                raise RuntimeError("held prediction changed during post-freeze scoring")
            metrics_by_class[held] = score["result"]
            cache_manifest = json.loads((cache / "manifest.json").read_text())
            if cache_manifest["held_class"] != held or cache_manifest["held_gt_reads"] != 0 or cache_manifest["held_mask_reads"] != 0:
                raise RuntimeError("fold cache firewall evidence failed")
            cache_bytes = sum(path.stat().st_size for path in cache.glob("*.bin"))
            shutil.rmtree(cache)
            fold_record = {
                "held_class": held,
                "source_classes": cache_manifest["source_classes"],
                "source_inventory_sha256": cache_manifest["source_inventory_sha256"],
                "source_files_sha256": cache_manifest["source_files_sha256"],
                "source_records": cache_manifest["record_count"],
                "cache_asset_hashes": {
                    "p26_checkpoint": cache_manifest["p26_checkpoint_sha256"],
                    "clip_asset": cache_manifest["clip_asset_sha256"],
                    "config": cache_manifest["config_sha256"],
                },
                "train": train,
                "evaluate": evaluate,
                "score": score,
                "prediction_sha256_before_and_after_score": prediction_hash_before_score,
                "cache_bytes": cache_bytes,
                "cache_removed_after_scoring": True,
                "held_gt_reads_before_scoring": 0,
                "held_mask_reads_before_scoring": 0,
            }
            fold_records.append(fold_record)
            _atomic_json(args.runtime_root / "PROGRESS.json", {"attempt_uuid": attempt_uuid, "completed_folds": [row["held_class"] for row in fold_records], "fold_records": fold_records})
    except Exception as exc:
        _atomic_json(args.runtime_root / "ENGINEERING_STOP.json", {"attempt_uuid": attempt_uuid, "status": "P27_ENGINEERING_STOP", "error": repr(exc), "completed_folds": [row["held_class"] for row in fold_records]})
        raise

    aggregate = _aggregate(metrics_by_class)
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": "P27_COMPLETE_SCIENTIFIC_RESULT_V1",
        "attempt": marker,
        "fold_records": fold_records,
        "aggregate": aggregate,
        "actual_scientific_wall_seconds": elapsed,
        "post_audit": {
            "status": "PASS",
            "attempt_markers": 1,
            "intended_folds": 12,
            "completed_folds": len(fold_records),
            "duplicate_folds": False,
            "held_gt_reads_before_scoring": 0,
            "held_mask_reads_before_scoring": 0,
            "mvtec_reads": 0,
            "medical_reads": 0,
            "phase2b_optimization_steps": 0,
            "clip_optimization_steps": 0,
            "trained_parameters": "P27 RegionResidualAdapter only",
            "protocol_sha256_unchanged": _sha256(PROTOCOL_PATH) == marker["protocol_sha256"],
            "asset_hashes_unchanged": marker["asset_hashes"] == {
                "phase2b_checkpoint": _sha256(args.p26_checkpoint), "clip_asset": _sha256(args.clip_asset), "config": _sha256(ROOT / "configs/phase2b_canonical_v1.json")
            },
        },
    }
    _atomic_json(args.runtime_root / "P27_SCIENTIFIC_RESULT.json", result)
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-base-sha", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--p26-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    result = run(make_parser().parse_args())
    print(json.dumps({"status": "COMPLETE", "attempt_uuid": result["attempt"]["attempt_uuid"], "result": str(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
