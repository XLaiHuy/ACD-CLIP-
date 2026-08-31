#!/usr/bin/env python3
"""Run one bounded, metric-neutral Medical resource-admission preflight."""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Subset

from evaluation.evaluator import evaluate_spool
from model.phase2b_runtime import build_phase2b_frozen, configure_canonical_fp32
from scripts.cir_rmt.eval_full import (
    _RssMonitor,
    _shutdown_loader,
    _target_dataset,
)
from tools.cir_rmt.corrective_eval import _evaluate_model, _sha256_file
from tools.cir_rmt.identity import config_sha256, load_cir_config


ROOT = Path(__file__).resolve().parents[2]


def _rss_bytes() -> int:
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _open_fd_count() -> int:
    return len(list(Path("/proc/self/fd").iterdir()))


def _children() -> list[int]:
    try:
        import psutil

        return sorted(int(child.pid) for child in psutil.Process().children(recursive=True))
    except Exception:
        return []


def _gpu_snapshot(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"free_bytes": None, "total_bytes": None, "allocated_bytes": 0, "reserved_bytes": 0}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _balanced_indices(dataset: object, per_label: int = 12) -> list[int]:
    selected = {0: [], 1: []}
    for index in range(len(dataset)):  # type: ignore[arg-type]
        label = int(dataset[index]["label"])  # type: ignore[index]
        if label in selected and len(selected[label]) < per_label:
            selected[label].append(index)
        if all(len(values) == per_label for values in selected.values()):
            break
    if any(len(values) < per_label for values in selected.values()):
        raise RuntimeError(f"could not form balanced preflight sample: { {key: len(value) for key, value in selected.items()} }")
    return selected[0] + selected[1]


def _metric(spool: object) -> dict[str, float | None]:
    result = evaluate_spool(spool, allow_undefined_image_metrics=True)
    return {key: None if value is None else float(value) for key, value in result["macro"].items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--medical-root", type=Path, required=True)
    parser.add_argument("--cir-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spool-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    spool_root = args.spool_root.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    spool_root.mkdir(parents=True, exist_ok=True)
    config = load_cir_config(args.config.expanduser().resolve())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    configure_canonical_fp32()
    disk_before = shutil.disk_usage(output.parent)
    soft_fd, hard_fd = resource.getrlimit(resource.RLIMIT_NOFILE)
    fd_before = _open_fd_count()
    gpu_before = _gpu_snapshot(device)
    rss_before = _rss_bytes()
    checkpoint_sha = _sha256_file(args.cir_checkpoint.expanduser().resolve())

    os.environ["MEDICAL_ROOT"] = str(args.medical_root.expanduser().resolve())
    os.environ["ACDCLIP_DATA_ROOT"] = str(args.medical_root.expanduser().resolve())
    dataset = _target_dataset("Brain", args.medical_root.expanduser().resolve())
    indices = _balanced_indices(dataset)
    sample = Subset(dataset, indices)
    checkpoint = torch.load(args.cir_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    model = build_phase2b_frozen(dict(checkpoint["parent_config"]), checkpoint, args.clip_asset.expanduser().resolve(), device)
    del checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    monitor = _RssMonitor()
    monitor.start()
    started = time.perf_counter()
    try:
        spools, seen, telemetry = _evaluate_model(
            model,
            sample,
            mode="cir",
            config=config,
            dataset_name="Brain",
            domain="Medical",
            device=device,
            spool_root=spool_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            monitor=monitor,
        )
        dataset_length = int(len(dataset))
        del model, sample, dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        monitor.set_phase("metric")
        metrics = {name: _metric(spool) for name, spool in spools.items()}
        for spool in spools.values():
            spool.close()
        spool_bytes = sum(path.stat().st_size for path in spool_root.rglob("*") if path.is_file())
        for spool in spools.values():
            spool.cleanup()
        shutil.rmtree(spool_root, ignore_errors=True)
        monitor.set_phase("final")
        monitor.stop()
        gpu_after = _gpu_snapshot(device)
        rss_after = _rss_bytes()
        children_after = _children()
        telemetry["elapsed_seconds"] = float(time.perf_counter() - started)
        telemetry["spool_bytes_before_cleanup"] = int(spool_bytes)
        telemetry["rss"] = monitor.report(rss_before)
        disk_after = shutil.disk_usage(output.parent)
        expected_full_cell_bytes = int(dataset_length * 518 * 518 * 4 * 4)
        bounded_vram = bool(
            device.type != "cuda"
            or int(telemetry["peak_reserved_vram_bytes"]) < int(gpu_before["total_bytes"] or 0) * 0.9
        )
        bounded_rss = bool(rss_after <= max(rss_before, int(telemetry["rss"].get("peak_inference_rss_mib") or 0) * 2**20) + 512 * 2**20)
        exact_metric = all(value.get("pixel_auroc") is not None and value.get("pixel_ap") is not None for value in metrics.values())
        preflight = {
            "status": "PASS" if bounded_vram and bounded_rss and bool(telemetry["spool_bytes_before_cleanup"] > 0) and not children_after and exact_metric else "FAIL",
            "admission_class": "SAFE" if bounded_vram and bounded_rss and bool(telemetry["spool_bytes_before_cleanup"] > 0) and not children_after and exact_metric else "UNSAFE",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": "medical",
            "target": "Brain",
            "sample": {"images": int(seen), "batches": 4, "indices": indices, "label_counts": {"normal": 12, "anomalous": 12}},
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "image_size": 518,
            "pixels_per_image": 518 * 518,
            "brain_images": dataset_length,
            "expected_full_cell_bytes_conservative": expected_full_cell_bytes,
            "disk": {"free_bytes_before": int(disk_before.free), "free_bytes_after": int(disk_after.free), "total_bytes": int(disk_before.total)},
            "file_descriptors": {"soft_limit": int(soft_fd), "hard_limit": int(hard_fd), "open_before": int(fd_before), "open_after": int(_open_fd_count())},
            "gpu_before": gpu_before,
            "gpu_after": gpu_after,
            "rss_before_bytes": int(rss_before),
            "rss_after_teardown_bytes": int(rss_after),
            "children_after_teardown": children_after,
            "bounded_vram_ok": bounded_vram,
            "bounded_rss_ok": bounded_rss,
            "worker_shutdown_ok": not children_after,
            "exact_full_resolution_spool": True,
            "metric_evaluation_after_teardown": True,
            "metric_status": "PASS" if exact_metric else "UNDEFINED_BOUNDED_SAMPLE",
            "metrics": metrics,
            "sample_forward": telemetry,
            "checkpoint": {"path": str(args.cir_checkpoint.expanduser().resolve()), "sha256": checkpoint_sha},
            "clip_asset": {"path": str(args.clip_asset.expanduser().resolve()), "sha256": _sha256_file(args.clip_asset.expanduser().resolve())},
            "config_sha256": config_sha256(config),
            "spool_root": str(spool_root),
            "engineering_adjustments": [],
        }
    except Exception as error:
        try:
            monitor.set_phase("failure")
            monitor.stop()
        except Exception:
            pass
        shutil.rmtree(spool_root, ignore_errors=True)
        preflight = {"status": "FAIL", "admission_class": "UNSAFE", "scope": "medical", "target": "Brain", "error": repr(error), "checkpoint_sha256": checkpoint_sha}
    output.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(preflight, indent=2, sort_keys=True))
    return 0 if preflight["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
