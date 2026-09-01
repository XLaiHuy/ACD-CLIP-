#!/usr/bin/env python3
"""Evaluate the single frozen H2-master winner on MVTec.

This script is intentionally one-shot: the pre-MVTec freeze names exactly
one checkpoint and native alpha=0 deployment.  It imports the recovered H2
model and dataset path, then uses the current disk-backed exact evaluator.
Per-class cells are independently hashed so an interruption can resume
without opening a second candidate or changing the scientific selection.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from evaluation.evaluator import evaluate_spool, image_score
from evaluation.spool import EvaluationSpool
from scripts.cir_rmt import train_h2_anchor_cir as runner
from tools.cir_rmt.h2_master_medical_eval import _configure_model
from tools.cir_rmt.h2_master_source_gate import _load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
H2_REPO = Path("/home/ai4/caohuy/ACD-CLIP-base-new-phase1-h2-anchor-cir-20260901")
TARGETS = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "transistor", "toothbrush", "wood", "zipper",
)
METRICS = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")


def _sha256(path: Path) -> str:
    return runner.sha256_file(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.8f}"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _cell_id(target: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in target)
    return f"mvtec__RA__E16__{safe}"


def _load_freeze(path: Path) -> dict[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("pre-MVTec freeze is not FROZEN")
    if freeze.get("final_candidate") != "RA" or int(freeze.get("final_epoch", -1)) != 16:
        raise RuntimeError("MVTec evaluator accepts only the frozen RA E16 winner")
    mvtec = freeze.get("mvtec", {})
    if mvtec.get("results_seen") is not False or mvtec.get("target_tuning") is not False:
        raise RuntimeError("MVTec results or tuning are already recorded in the freeze")
    if freeze.get("deployment_alpha") != 0.0:
        raise RuntimeError("final MVTec requires native deployment alpha=0")
    return freeze


def _load_completed(output: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for row in _read_csv(output / "MVTEC_LEDGER.csv"):
        if row.get("status") != "COMPLETE":
            continue
        cell_id = str(row.get("cell_id", ""))
        path = output / str(row.get("cell_path", ""))
        if not cell_id or not path.is_file() or _sha256(path) != row.get("cell_sha256"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "COMPLETE" and payload.get("cell_id") == cell_id:
            completed[cell_id] = payload
    return completed


def _write_ledger(output: Path, completed: Mapping[str, Mapping[str, Any]]) -> None:
    fields = ["cell_id", "target", "method", "epoch", "status", "cell_path", "cell_sha256", "checkpoint_sha256", "updated_at"]
    rows = []
    for cell_id, payload in sorted(completed.items()):
        path = output / "mvtec_cells" / f"{cell_id}.json"
        rows.append({
            "cell_id": cell_id,
            "target": payload["target"],
            "method": payload["method"],
            "epoch": payload["epoch"],
            "status": payload["status"],
            "cell_path": str(path.relative_to(output)),
            "cell_sha256": _sha256(path),
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "updated_at": payload["updated_at"],
        })
    _write_csv(output / "MVTEC_LEDGER.csv", fields, rows)


def _evaluate_target(
    *, model: Any, target: str, dataset_module: Any, h2_utils: Any,
    domains: Mapping[str, str], device: torch.device, img_size: int,
    batch_size: int, spool_root: Path,
) -> tuple[dict[str, float | None], int]:
    datasets = dataset_module.get_text_and_image_dataset("MVTec", img_size, "test")
    if set(datasets) != set(TARGETS):
        raise RuntimeError(f"unexpected H2 MVTec class mapping: {sorted(datasets)}")
    dataset = datasets[target]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    spool = EvaluationSpool.create(spool_root)
    seen = 0
    with torch.inference_mode():
        text_embeddings = h2_utils.get_multiple_adapted_text_embedding(model, "MVTec", device)
        class_text = text_embeddings[target]
        for batch in loader:
            image = batch["image"].to(device, non_blocking=device.type == "cuda")
            mask = batch["mask"].detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy().reshape(-1)
            names = [str(value) for value in batch["class_name"]]
            if set(names) != {target}:
                raise RuntimeError(f"mixed H2 MVTec class identities in {target}")
            seg_tokens, det_tokens = model(image)
            seg_features = torch.stack(seg_tokens, dim=0)
            det_features = torch.stack(det_tokens, dim=0)
            batch_text = class_text.unsqueeze(dim=1).repeat(1, image.shape[0], 1, 1)
            cls_preds = [
                torch.matmul(det_features[index].unsqueeze(dim=1), batch_text[index]).squeeze(1)
                for index in range(det_features.shape[0])
            ]
            cls_preds = torch.stack(cls_preds, dim=0).mean(dim=0)
            cls_probability = torch.softmax(cls_preds, dim=1)[:, 1]
            seg_probability = model.vision_text_fusion_gate_seg(
                seg_features, batch_text, test_mode=True, domain=domains["MVTec"]
            )
            pixel_max = seg_probability.reshape(seg_probability.shape[0], -1).max(dim=1).values
            image_probability = image_score(cls_probability, pixel_max, domains["MVTec"])
            for index, name in enumerate(names):
                spool.append(
                    name,
                    seg_probability[index].detach().float().cpu().numpy(),
                    mask[index].reshape(-1),
                    float(image_probability[index].detach().float().cpu()),
                    int(labels[index]),
                )
            seen += len(names)
            del image, seg_tokens, det_tokens, seg_features, det_features, batch_text
            del cls_preds, cls_probability, seg_probability, pixel_max, image_probability
            if device.type == "cuda":
                torch.cuda.empty_cache()
    evaluated = evaluate_spool(spool, allow_undefined_image_metrics=True)
    classes = evaluated["per_class"]
    if set(classes) != {target}:
        raise RuntimeError(f"MVTec spool class mismatch for {target}: {sorted(classes)}")
    values = {metric: classes[target].get(metric) for metric in METRICS}
    spool.cleanup()
    del loader, dataset, datasets, text_embeddings, class_text
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values, int(seen)


def _summarize(output: Path, completed: Mapping[str, Mapping[str, Any]], freeze: Mapping[str, Any]) -> None:
    rows = []
    for cell_id, payload in sorted(completed.items()):
        row = {key: payload.get(key) for key in ("method", "epoch", "target", "n_images", "checkpoint_sha256", "config_sha256", "evaluator_git_sha", "evaluator_sha256")}
        row.update({metric: payload.get(metric) for metric in METRICS})
        rows.append(row)
    fields = ["method", "epoch", "target", "n_images", "checkpoint_sha256", "config_sha256", "evaluator_git_sha", "evaluator_sha256", *METRICS]
    _write_csv(output / "FINAL_MVTEC_RESULTS.csv", fields, rows)
    macro = {metric: None for metric in METRICS}
    counts = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row[metric] is not None]
        macro[metric] = sum(values) / len(values) if values else None
        counts[f"n_defined_{metric}"] = len(values)
    _write_csv(output / "FINAL_MVTEC_MACRO.csv", ["method", "epoch", "n_targets", *METRICS, *counts], [{"method": "RA", "epoch": 16, "n_targets": len(rows), **macro, **counts}])
    lines = [
        "# Final MVTec confirmation",
        "",
        "Status: COMPLETE. Exactly one source/Medical-frozen winner was evaluated: RA E16 with native H2 alpha=0. No variants or tuning were run.",
        "",
        "| class | pixel AUROC | pixel AP | image AUROC | image AP |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['target']} | {_fmt(row['pixel_auroc'])} | {_fmt(row['pixel_ap'])} | {_fmt(row['image_auroc'])} | {_fmt(row['image_ap'])} |")
    lines.extend([
        "", "## Macro", "", "| candidate | pixel AUROC | pixel AP | image AUROC | image AP |", "|---|---:|---:|---:|---:|",
        f"| RA E16 | {_fmt(macro['pixel_auroc'])} | {_fmt(macro['pixel_ap'])} | {_fmt(macro['image_auroc'])} | {_fmt(macro['image_ap'])} |",
        "", f"Pre-MVTec freeze: `{freeze.get('final_checkpoint_sha256')}`.",
    ])
    (output / "FINAL_MVTEC_CONFIRMATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    os.chdir(ROOT)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = args.freeze.resolve()
    freeze = _load_freeze(freeze_path)
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha = _sha256(config_path)
    if config_sha != freeze.get("config_sha256"):
        raise RuntimeError("MVTec config SHA differs from the pre-MVTec freeze")
    clip_asset = Path(freeze["clip_asset"]).resolve()
    if _sha256(clip_asset) != freeze.get("clip_asset_sha256"):
        raise RuntimeError("MVTec CLIP asset SHA differs from the pre-MVTec freeze")
    h2_repo = Path(cfg.get("h2_repo_path", H2_REPO)).resolve()
    modules = dict(runner._load_h2_modules(h2_repo))
    h2_utils = __import__("utils", fromlist=["get_multiple_adapted_text_embedding"])
    dataset_module = __import__("dataset", fromlist=["get_text_and_image_dataset", "DOMAINS"])
    os.chdir(h2_repo)
    completed = _load_completed(output) if args.resume else {}
    if not args.resume and (output / "MVTEC_LEDGER.csv").exists():
        raise RuntimeError("existing MVTec ledger requires --resume")
    identity = {
        "status": "RUNNING", "scope": "h2_master_mvtec", "method": "RA", "epoch": 16,
        "checkpoint": freeze["final_checkpoint"], "checkpoint_sha256": freeze["final_checkpoint_sha256"],
        "config_sha256": config_sha, "architecture_freeze_sha256": freeze["architecture_freeze_sha256"],
        "clip_asset_sha256": freeze["clip_asset_sha256"], "pre_mvtec_freeze_sha256": _sha256(freeze_path),
        "source_dataset": "VisA", "targets": list(TARGETS), "deployment_alpha": 0.0,
        "target_tuning": False, "mvtec_tuning": False, "variant_search": False,
        "evaluator_git_sha": _git_sha(), "evaluator_sha256": _sha256(ROOT / "evaluation/evaluator.py"),
    }
    _atomic_json(output / "FINAL_MVTEC_IDENTITY.json", identity)
    device = torch.device(args.device)
    expected = {_cell_id(target) for target in TARGETS}
    checkpoint_path = ROOT / freeze["final_checkpoint"]
    try:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        payload = _load_checkpoint(checkpoint_path, cfg, config_sha, 16)
        model = runner.build_model(cfg, modules, device)
        _configure_model(model, payload)
        for target in TARGETS:
            cell_id = _cell_id(target)
            if cell_id in completed:
                continue
            started = time.perf_counter()
            values, seen = _evaluate_target(
                model=model, target=target, dataset_module=dataset_module, h2_utils=h2_utils,
                domains=dataset_module.DOMAINS, device=device, img_size=int(cfg["img_size"]),
                batch_size=int(args.batch_size), spool_root=output / "temporary_mvtec_spools" / cell_id,
            )
            updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            cell = {
                "status": "COMPLETE", "cell_id": cell_id, "scope": "mvtec", "method": "RA", "epoch": 16,
                "target": target, "n_images": seen, **values,
                "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": _sha256(checkpoint_path),
                "config_sha256": config_sha, "architecture_freeze_sha256": freeze["architecture_freeze_sha256"],
                "clip_asset_sha256": freeze["clip_asset_sha256"], "deployment": "native_h2_alpha0",
                "rmt_inference": "NOT_APPLIED; RCA CIR IS TRAINING_ONLY", "target_tuning": False,
                "mvtec_tuning": False, "evaluator_git_sha": identity["evaluator_git_sha"],
                "evaluator_sha256": identity["evaluator_sha256"], "elapsed_seconds": time.perf_counter() - started,
                "updated_at": updated,
            }
            path = output / "mvtec_cells" / f"{cell_id}.json"
            _atomic_json(path, cell)
            completed[cell_id] = cell
            _write_ledger(output, completed)
            _atomic_json(output / "FINAL_MVTEC_PROGRESS.json", {"status": "RUNNING", "completed_cells": len(completed), "planned_cells": len(expected), "last_cell": cell_id, "updated_at": updated})
            print(f"completed MVTec cell {cell_id}: {len(completed)}/{len(expected)}", flush=True)
        if set(completed) != expected:
            raise RuntimeError(f"MVTec ledger incomplete: missing={sorted(expected - set(completed))}")
        identity["status"] = "COMPLETED"
        identity["completed_cells"] = len(completed)
        _atomic_json(output / "FINAL_MVTEC_IDENTITY.json", identity)
        _atomic_json(output / "FINAL_MVTEC_COMPLETE.json", {"status": "COMPLETED", "completed_cells": len(completed), "planned_cells": len(expected), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        _summarize(output, completed, freeze)
        failed_marker = output / "FINAL_MVTEC_FAILED.json"
        if failed_marker.exists():
            failed_marker.unlink()
    except Exception as error:
        _atomic_json(output / "FINAL_MVTEC_FAILED.json", {"status": "FAILED", "error": repr(error), "completed_cells": len(completed), "planned_cells": len(expected), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/exact_h2_anchor_cir_master_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
