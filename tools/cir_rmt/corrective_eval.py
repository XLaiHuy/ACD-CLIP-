#!/usr/bin/env python3
"""Failure-resilient exact P/C0/C05 evaluation for the corrected run.

This is an evaluation harness only.  It reuses the frozen Phase2B/CIR
forward paths and the repository's exact disk-backed metric evaluator.  C0
and C05 are produced from the same CIR forward pass: C0 uses the native
deployed map and C05 uses the CIR transport map.  The journal is append-only
and every completed cell also has an atomically written JSON record, so a
resume skips only cells whose recorded hash still verifies.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from evaluation.evaluator import evaluate_spool, image_score
from evaluation.spool import EvaluationSpool
from model.phase2b_runtime import (
    build_phase2b_frozen,
    configure_canonical_fp32,
    deploy_native_logits,
    forward_phase2b,
)
from scripts.cir_rmt.eval_full import (
    ManifestDataset,
    _RssMonitor,
    _shutdown_loader,
    _target_dataset,
)
from tools.cir_rmt.identity import (
    config_sha256,
    load_cir_config,
    validate_checkpoint_identity,
)
from tools.cir_rmt.runtime import forward_cir


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = (10, 12, 14, 16, 18, 20)
MEDICAL_TARGETS = (
    "Brain",
    "Liver",
    "Retina",
    "Colon_clinicDB",
    "Colon_colonDB",
    "Colon_Kvasir",
)
SOURCE_TARGET = "VisA_SOURCE"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def _evaluator_sha() -> str:
    return _sha256_file(ROOT / "scripts/cir_rmt/eval_full.py")


def _cell_id(scope: str, method: str, epoch: int, target: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in target)
    return f"{scope}__{method}__E{int(epoch):02d}__{safe}"


def _spool_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _metrics(spool: EvaluationSpool) -> dict[str, float | None]:
    result = evaluate_spool(spool, allow_undefined_image_metrics=True)
    return {key: None if value is None else float(value) for key, value in result["macro"].items()}


def _loader(dataset: Any, batch_size: int, num_workers: int, prefetch_factor: int, device: torch.device) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(device.type == "cuda"),
    }
    if int(num_workers) > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": int(prefetch_factor)})
    return DataLoader(dataset, **kwargs)


def _append_scores(
    spools: Mapping[str, EvaluationSpool],
    output: Any,
    masks: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    map_kind: str,
    domain: str,
) -> int:
    if map_kind == "parent":
        pixel_maps = output.native_segmentation_probability
    elif map_kind == "cir":
        native_probability, _ = deploy_native_logits(output.native_logits, image_size=518, domain=domain)
        pixel_maps = native_probability[:, 1]
    else:
        raise ValueError(map_kind)
    for index, class_name in enumerate(class_names):
        pixel = pixel_maps[index].detach().float().cpu().numpy().reshape(-1)
        mask = masks[index].detach().float().cpu().numpy().reshape(-1)
        label = int(labels[index].detach().cpu())
        cls = float(output.classification_probability[index].detach().cpu())
        spools["native"].append(
            str(class_name),
            pixel,
            mask,
            float(image_score(cls, float(pixel.max()), domain)),
            label,
        )
    return len(class_names)


def _append_cir_scores(
    spools: Mapping[str, EvaluationSpool],
    output: Any,
    masks: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    domain: str,
) -> int:
    native_probability, _ = deploy_native_logits(output.native_logits, image_size=518, domain=domain)
    maps = {"C0": native_probability[:, 1], "C05": output.cir_segmentation_probability}
    for index, class_name in enumerate(class_names):
        mask = masks[index].detach().float().cpu().numpy().reshape(-1)
        label = int(labels[index].detach().cpu())
        cls = float(output.classification_probability[index].detach().cpu())
        for method, pixel_map in maps.items():
            pixel = pixel_map[index].detach().float().cpu().numpy().reshape(-1)
            spools[method].append(
                str(class_name),
                pixel,
                mask,
                float(image_score(cls, float(pixel.max()), domain)),
                label,
            )
    return len(class_names)


def _evaluate_model(
    model: Any,
    dataset: Any,
    *,
    mode: str,
    config: Mapping[str, Any],
    dataset_name: str,
    domain: str,
    device: torch.device,
    spool_root: Path,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    monitor: _RssMonitor,
) -> tuple[dict[str, EvaluationSpool], int, dict[str, Any]]:
    if mode == "parent":
        spools = {"native": EvaluationSpool.create(spool_root / "P")}
    elif mode == "cir":
        spools = {method: EvaluationSpool.create(spool_root / method) for method in ("C0", "C05")}
    else:
        raise ValueError(mode)
    loader = _loader(dataset, batch_size, num_workers, prefetch_factor, device)
    loader_iter: Any = None
    seen = 0
    started = time.perf_counter()
    monitor.set_phase("inference")
    try:
        loader_iter = iter(loader)
        for batch in loader_iter:
            image = batch["image"].to(device, non_blocking=device.type == "cuda").float()
            masks = batch["mask"].to(device, non_blocking=device.type == "cuda").float()
            labels = batch["label"].to(device, non_blocking=device.type == "cuda").long().reshape(-1)
            class_names = [str(value) for value in batch["class_name"]]
            if mode == "parent":
                with torch.inference_mode():
                    output = forward_phase2b(model, image, class_names, device, config, domain=domain, require_grad=False, dataset_name=dataset_name)
                seen += _append_scores(spools, output, masks, labels, class_names, "parent", domain)
            else:
                output = forward_cir(model, image, class_names, device, config, domain=domain, require_grad=False, dataset_name=dataset_name)
                seen += _append_cir_scores(spools, output, masks, labels, class_names, domain)
            del output, image, masks, labels
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        _shutdown_loader(loader, loader_iter)
        del loader, loader_iter
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        monitor.set_phase("after_teardown")
    elapsed = max(time.perf_counter() - started, 1e-9)
    telemetry = {
        "images": int(seen),
        "elapsed_seconds": float(elapsed),
        "images_per_second": float(seen / elapsed),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "spool_bytes_before_metric": _spool_bytes(spool_root),
    }
    return spools, seen, telemetry


def _evaluate_cell(
    *,
    method_group: str,
    epoch: int,
    target: str,
    scope: str,
    checkpoint_path: Path,
    config: Mapping[str, Any],
    parent_config: Mapping[str, Any],
    clip_asset: Path,
    target_root: Path,
    source_root: Path,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> list[dict[str, Any]]:
    configure_canonical_fp32()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if method_group == "P":
        mode = "parent"
        model_config = dict(parent_config)
        expected_protocol = str(parent_config["protocol_version"])
        if checkpoint.get("protocol_version") != expected_protocol:
            raise ValueError(f"parent protocol mismatch at E{epoch}: {checkpoint.get('protocol_version')}")
        model = build_phase2b_frozen(model_config, checkpoint, clip_asset, device)
        methods = ("P",)
    else:
        mode = "cir"
        model_config = dict(checkpoint["parent_config"])
        validate_checkpoint_identity(checkpoint, config, source_dataset="visa", expected_epoch=epoch)
        model = build_phase2b_frozen(model_config, checkpoint, clip_asset, device)
        methods = ("C0", "C05")
    del checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if scope == "source":
        dataset = ManifestDataset(source_root, ROOT / "dataset/hub/VisA.jsonl", int(model_config["img_size"]))
        dataset_name, domain = "VisA", "Industrial"
    elif scope == "medical":
        os.environ["MEDICAL_ROOT"] = str(target_root.expanduser().resolve())
        os.environ["ACDCLIP_DATA_ROOT"] = str(target_root.expanduser().resolve())
        dataset = _target_dataset(target, target_root)
        dataset_name, domain = target, "Medical"
    else:
        raise ValueError(scope)
    cell_id = _cell_id(scope, method_group, epoch, target)
    spool_root = output_root / "temporary_spools" / cell_id
    monitor = _RssMonitor()
    model_loaded_rss = 0
    monitor.start()
    try:
        monitor.set_phase("startup")
        spools, seen, telemetry = _evaluate_model(
            model,
            dataset,
            mode=mode,
            config=config if mode == "cir" else model_config,
            dataset_name=dataset_name,
            domain=domain,
            device=device,
            spool_root=spool_root,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            monitor=monitor,
        )
        del model, dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        monitor.set_phase("metric")
        metric_values: dict[str, dict[str, float | None]] = {}
        for method, spool in spools.items():
            metric_values[method] = _metrics(spool)
            spool.close()
        telemetry["spool_bytes_after_metric"] = _spool_bytes(spool_root)
        for spool in spools.values():
            spool.cleanup()
        shutil.rmtree(spool_root, ignore_errors=True)
        monitor.set_phase("final")
        monitor.stop()
        telemetry["rss"] = monitor.report(model_loaded_rss)
        rows = []
        checkpoint_sha = _sha256_file(checkpoint_path)
        for method in methods:
            metric_key = "native" if method == "P" else method
            row = {
                "status": "COMPLETE",
                "scope": scope,
                "method": method,
                "alpha": None if method == "P" or method == "C0" else 0.5,
                "epoch": int(epoch),
                "target": target,
                "n_images": int(seen),
                "pixel_auroc": metric_values[metric_key].get("pixel_auroc"),
                "pixel_ap": metric_values[metric_key].get("pixel_ap"),
                "image_auroc": metric_values[metric_key].get("image_auroc"),
                "image_ap": metric_values[metric_key].get("image_ap"),
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha256(config),
                "evaluator_git_sha": _git_sha(),
                "evaluator_sha256": _evaluator_sha(),
                "telemetry": telemetry,
            }
            rows.append(row)
        return rows
    except Exception:
        try:
            monitor.set_phase("failure")
            monitor.stop()
        except Exception:
            pass
        shutil.rmtree(spool_root, ignore_errors=True)
        raise


def _load_valid_journal(output_root: Path) -> dict[str, dict[str, Any]]:
    journal = output_root / "journal.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if not journal.is_file():
        return completed
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("status") != "COMPLETE":
            continue
        path = output_root / entry["cell_path"]
        if not path.is_file() or _sha256_file(path) != entry.get("cell_sha256"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "COMPLETE" and payload.get("cell_id") == entry.get("cell_id"):
            completed[str(entry["cell_id"])] = payload
    return completed


def _record_rows(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    journal = output_root / "journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a", encoding="utf-8") as handle:
        for row in rows:
            cell_id = _cell_id(str(row["scope"]), str(row["method"]), int(row["epoch"]), str(row["target"]))
            payload = dict(row)
            payload["cell_id"] = cell_id
            path = output_root / "cells" / f"{cell_id}.json"
            _atomic_json(path, payload)
            digest = _sha256_file(path)
            entry = {"cell_id": cell_id, "cell_path": str(path.relative_to(output_root)), "cell_sha256": digest, "status": "COMPLETE", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _incident(output_root: Path, cell: str, error: BaseException) -> None:
    path = output_root / "OOM_KILLED_INCIDENTS.csv"
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        fields = ["timestamp", "cell", "phase", "exit_code", "failure_class", "evidence", "fix", "parity_status", "resumed_from", "final_status"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if not exists:
            writer.writeheader()
        message = str(error)
        failure = "CUDA_OOM" if "out of memory" in message.lower() and "cuda" in message.lower() else "DATASET_ERROR" if "dataset" in message.lower() else "UNKNOWN"
        writer.writerow({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cell": cell, "phase": "evaluation", "exit_code": 1, "failure_class": failure, "evidence": message[:1000], "fix": "STOPPED_BEFORE_AUTOMATIC_RERUN", "parity_status": "NOT_RUN", "resumed_from": "", "final_status": "FAILED"})
        handle.flush()
        os.fsync(handle.fileno())


def _plan(scope: str) -> list[tuple[str, int, str]]:
    targets = (SOURCE_TARGET,) if scope == "source" else MEDICAL_TARGETS
    rows: list[tuple[str, int, str]] = []
    for epoch in EPOCHS:
        for target in targets:
            rows.extend([(method, epoch, target) for method in ("P", "CIR")])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("source", "medical"), required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/home/ai4/caohuy/data/VisA_20220922"))
    parser.add_argument("--medical-root", type=Path, default=Path("/home/ai4/caohuy/data"))
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--cir-run-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cir_dfg_rmt_v2.json"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        raise SystemExit("batch-size must be positive and num-workers must be non-negative")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_cir_config(args.config.expanduser().resolve())
    parent_config = json.loads((ROOT / str(config["parent_config_path"])).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    completed = _load_valid_journal(output_root) if args.resume else {}
    if not args.resume and (output_root / "journal.jsonl").exists():
        raise SystemExit(f"existing journal requires --resume: {output_root}")
    identity = {"scope": args.scope, "config_sha256": config_sha256(config), "source_root": str(args.source_root.resolve()), "medical_root": str(args.medical_root.resolve()), "parent_run_root": str(args.parent_run_root.resolve()), "cir_run_root": str(args.cir_run_root.resolve()), "clip_asset": str(args.clip_asset.resolve()), "evaluator_git_sha": _git_sha(), "evaluator_sha256": _evaluator_sha(), "epochs": list(EPOCHS), "targets": [SOURCE_TARGET] if args.scope == "source" else list(MEDICAL_TARGETS), "batch_size": int(args.batch_size), "num_workers": int(args.num_workers), "prefetch_factor": int(args.prefetch_factor)}
    _atomic_json(output_root / "identity.json", identity)
    for method_group, epoch, target in _plan(args.scope):
        method_names = ("P",) if method_group == "P" else ("C0", "C05")
        if all(_cell_id(args.scope, method, epoch, target) in completed for method in method_names):
            continue
        if method_group == "P":
            checkpoint = args.parent_run_root / "phase2b" / "checkpoints" / f"adapter_{epoch}.pth"
        else:
            checkpoint = args.cir_run_root / "visa" / "seed0" / "checkpoints" / f"epoch_{epoch:02d}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        cell_label = f"{args.scope}/{method_group}/E{epoch:02d}/{target}"
        try:
            rows = _evaluate_cell(method_group=method_group, epoch=epoch, target=target, scope=args.scope, checkpoint_path=checkpoint, config=config, parent_config=parent_config, clip_asset=args.clip_asset.expanduser().resolve(), target_root=args.medical_root, source_root=args.source_root, output_root=output_root, device=device, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor)
            _record_rows(output_root, rows)
            completed.update({_cell_id(args.scope, str(row["method"]), epoch, target): dict(row) for row in rows})
            _atomic_json(output_root / "progress.json", {"scope": args.scope, "completed_cells": len(completed), "planned_cell_groups": len(_plan(args.scope)), "last": cell_label, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        except Exception as error:
            _incident(output_root, cell_label, error)
            _atomic_json(output_root / "FAILED.json", {"status": "FAILED", "cell": cell_label, "error": str(error), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            raise
    planned = len(_plan(args.scope)) * 1 + (len(_plan(args.scope)) * 0)
    _atomic_json(output_root / "COMPLETE.json", {"status": "COMPLETED", "scope": args.scope, "completed_cells": len(completed), "planned_cells": len(_plan(args.scope)) * 3 // 2, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
