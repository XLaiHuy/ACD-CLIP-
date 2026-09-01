#!/usr/bin/env python3
"""Evaluate the frozen H2-master candidates on the six Medical targets.

The evaluator imports the recovered H2 model and dataset modules read-only,
uses native alpha=0 deployment for every arm, and computes exact metrics with
the repository's disk-backed evaluator.  The source-only candidate freeze is
required before this script can access Medical data.  Cells are written and
hashed independently so a technical interruption can resume without changing
the scientific candidate set.
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

from evaluation.evaluator import evaluate_spool
from evaluation.spool import EvaluationSpool
from scripts.cir_rmt import train_h2_anchor_cir as runner
from tools.cir_rmt.h2_master_source_gate import _load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
H2_REPO = Path("/home/ai4/caohuy/ACD-CLIP-base-new-phase1-h2-anchor-cir-20260901")
TARGETS = ("Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir")
METHODS = ("R", "RA", "RCA")
METRICS = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")


def _sha256(path: Path) -> str:
    digest = runner.sha256_file(path)
    return digest


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


def _fmt_metric(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.8f}"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _cell_id(method: str, epoch: int, target: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in target)
    return f"medical__{method}__E{int(epoch):02d}__{safe}"


def _load_freeze(path: Path) -> dict[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("pre-Medical candidate freeze is not FROZEN")
    if freeze.get("selection_rule", {}).get("medical_results_seen") is not False:
        raise RuntimeError("candidate freeze does not prove Medical-blind selection")
    if freeze.get("selection_rule", {}).get("mvtec_results_seen") is not False:
        raise RuntimeError("candidate freeze does not prove MVTec-blind selection")
    if freeze.get("deployment_alpha") != 0.0:
        raise RuntimeError("master Medical evaluation requires native alpha=0")
    return freeze


def _load_completed(output: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for row in _read_csv(output / "MEDICAL_LEDGER.csv"):
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
    fields = ["cell_id", "method", "epoch", "target", "status", "cell_path", "cell_sha256", "checkpoint_sha256", "updated_at"]
    rows = []
    for cell_id, payload in sorted(completed.items()):
        path = output / "cells" / f"{cell_id}.json"
        rows.append({
            "cell_id": cell_id,
            "method": payload["method"],
            "epoch": payload["epoch"],
            "target": payload["target"],
            "status": payload["status"],
            "cell_path": str(path.relative_to(output)),
            "cell_sha256": _sha256(path),
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "updated_at": payload["updated_at"],
        })
    _write_csv(output / "MEDICAL_LEDGER.csv", fields, rows)


def _configure_model(model: Any, payload: Mapping[str, Any]) -> None:
    model.image_adapter.load_state_dict(payload["image_adapter"])
    model.text_adapter.load_state_dict(payload["text_adapter"])
    model.soft_prompt.load_state_dict(payload["soft_prompt"])
    model.prompt_mode = "hybrid"
    model.use_hybrid_soft_prompt = True
    model.use_soft_prompt = True
    model.hybrid_alpha_current = 0.0
    model.dfg_beta_schedule = payload.get("dfg_beta_schedule", getattr(model, "dfg_beta_schedule", "fixed"))
    model.dfg_beta_target = float(payload.get("dfg_beta_target", getattr(model, "dfg_beta_target", 0.1)))
    model.dfg_weight_residual_fp32 = bool(payload.get("dfg_weight_residual_fp32", True))
    beta = float(payload.get("dfg_beta_current", payload.get("dfg_beta", 0.0)))
    if hasattr(model, "set_dfg_beta"):
        model.set_dfg_beta(beta)
    else:
        model.dfg_beta = beta
    model.eval()


def _evaluate_target(
    *,
    model: Any,
    target: str,
    dataset_module: Any,
    h2_utils: Any,
    domains: Mapping[str, str],
    device: torch.device,
    img_size: int,
    batch_size: int,
    spool_root: Path,
) -> tuple[dict[str, float | None], int, int]:
    datasets = dataset_module.get_text_and_image_dataset(target, img_size, "test")
    if set(datasets) != {target}:
        raise RuntimeError(f"unexpected H2 Medical class mapping for {target}: {sorted(datasets)}")
    dataset = datasets[target]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    spool = EvaluationSpool.create(spool_root)
    seen = 0
    with torch.inference_mode():
        text_embeddings = h2_utils.get_multiple_adapted_text_embedding(model, target, device)
        class_text = text_embeddings[target]
        for batch in loader:
            image = batch["image"].to(device, non_blocking=device.type == "cuda")
            mask = batch["mask"].detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy().reshape(-1)
            names = [str(value) for value in batch["class_name"]]
            if set(names) != {target}:
                raise RuntimeError(f"mixed H2 Medical class identities in {target}")
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
                seg_features,
                batch_text,
                test_mode=True,
                domain=domains[target],
            )
            pixel_max = seg_probability.reshape(seg_probability.shape[0], -1).max(dim=1).values
            image_probability = 0.5 * cls_probability + 0.5 * pixel_max
            for index, name in enumerate(names):
                spool.append(
                    name,
                    seg_probability[index].detach().float().cpu().numpy(),
                    mask[index].reshape(-1),
                    float(image_probability[index].detach().float().cpu()),
                    int(labels[index]),
                )
            seen += len(names)
            del image, seg_tokens, det_tokens, seg_features, det_features, batch_text, cls_preds, cls_probability, seg_probability, pixel_max, image_probability
            if device.type == "cuda":
                torch.cuda.empty_cache()
    evaluated = evaluate_spool(spool, allow_undefined_image_metrics=True)
    classes = evaluated["per_class"]
    if set(classes) != {target}:
        raise RuntimeError(f"Medical spool class mismatch for {target}: {sorted(classes)}")
    values = {metric: classes[target].get(metric) for metric in METRICS}
    spool.cleanup()
    del loader, dataset, datasets, text_embeddings, class_text
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values, int(seen), int(len(mask)) if "mask" in locals() else 0


def _summarize(output: Path, completed: Mapping[str, Mapping[str, Any]], freeze: Mapping[str, Any]) -> None:
    rows = []
    for cell_id, payload in sorted(completed.items()):
        row = {key: payload.get(key) for key in ("method", "epoch", "target", "n_images", "checkpoint_sha256", "config_sha256", "evaluator_git_sha", "evaluator_sha256")}
        row.update({metric: payload.get(metric) for metric in METRICS})
        rows.append(row)
    fields = ["method", "epoch", "target", "n_images", "checkpoint_sha256", "config_sha256", "evaluator_git_sha", "evaluator_sha256", *METRICS]
    _write_csv(output / "MEDICAL_RESULTS.csv", fields, rows)
    macro_rows = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        macro = {
            "method": method,
            "epoch": method_rows[0]["epoch"],
            "n_targets": len(method_rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in method_rows if row[metric] is not None]
            macro[metric] = sum(values) / len(values) if values else None
            macro[f"n_defined_{metric}"] = len(values)
        macro_rows.append(macro)
    _write_csv(
        output / "MEDICAL_MACRO.csv",
        ["method", "epoch", "n_targets", *METRICS, *(f"n_defined_{metric}" for metric in METRICS)],
        macro_rows,
    )
    lines = [
        "# H2 master Medical evaluation",
        "",
        "Status: COMPLETE. Native H2 deployment alpha=0 was used for R, RA, and RCA; RCA CIR was not applied at inference.",
        "",
        "The candidate set was frozen from VisA source evidence before Medical access. No target tuning or post-Medical checkpoint replacement is allowed.",
        "",
        "| method | epoch | pixel AUROC | pixel AP | image AUROC | image AP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['epoch']} | {_fmt_metric(row['pixel_auroc'])} | "
            f"{_fmt_metric(row['pixel_ap'])} | {_fmt_metric(row['image_auroc'])} | {_fmt_metric(row['image_ap'])} |"
        )
    lines.extend(["", "## Macro across six Medical targets", "", "| method | epoch | pixel AUROC | pixel AP | image AUROC | image AP |", "|---|---:|---:|---:|---:|---:|"])
    for row in macro_rows:
        lines.append(
            f"| {row['method']} | {row['epoch']} | {_fmt_metric(row['pixel_auroc'])} | "
            f"{_fmt_metric(row['pixel_ap'])} | {_fmt_metric(row['image_auroc'])} | {_fmt_metric(row['image_ap'])} |"
        )
    lines.extend(["", "MVTec remains NOT RUN until the final architecture/checkpoint decision and one-shot freeze.", "", f"Source freeze: `{freeze.get('experiment_id')}`."])
    (output / "MEDICAL_EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    os.chdir(ROOT)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze = _load_freeze(args.freeze.resolve())
    config_path = args.config.resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha = _sha256(config_path)
    if config_sha != freeze.get("config_sha256"):
        raise RuntimeError("Medical config SHA differs from frozen candidate selection")
    h2_repo = Path(cfg.get("h2_repo_path", H2_REPO)).resolve()
    modules = dict(runner._load_h2_modules(h2_repo))
    h2_utils = __import__("utils", fromlist=["get_multiple_adapted_text_embedding"])
    dataset_module = __import__("dataset", fromlist=["get_text_and_image_dataset", "DOMAINS"])
    domains = dataset_module.DOMAINS
    clip_asset = Path(freeze["clip_asset"]).resolve()
    if _sha256(clip_asset) != freeze.get("clip_asset_sha256"):
        raise RuntimeError("Medical CLIP asset SHA differs from frozen candidate selection")
    selected = freeze["source_selected_checkpoints"]
    completed = _load_completed(output) if args.resume else {}
    if not args.resume and (output / "MEDICAL_LEDGER.csv").exists():
        raise RuntimeError("existing Medical ledger requires --resume")
    identity = {
        "status": "RUNNING",
        "scope": "h2_master_medical",
        "config_sha256": config_sha,
        "architecture_freeze_sha256": freeze["architecture_freeze_sha256"],
        "clip_asset_sha256": freeze["clip_asset_sha256"],
        "source_freeze_sha256": _sha256(args.freeze.resolve()),
        "source_dataset": "VisA",
        "seed": 0,
        "methods": list(METHODS),
        "targets": list(TARGETS),
        "deployment_alpha": 0.0,
        "target_tuning": False,
        "mvtec": "NOT_RUN",
        "evaluator_git_sha": _git_sha(),
        "evaluator_sha256": _sha256(ROOT / "evaluation/evaluator.py"),
    }
    _atomic_json(output / "MEDICAL_IDENTITY.json", identity)
    device = torch.device(args.device)
    expected = {_cell_id(method, int(selected[method]["epoch"]), target) for method in METHODS for target in TARGETS}
    try:
        for method in METHODS:
            epoch = int(selected[method]["epoch"])
            checkpoint_path = ROOT / selected[method]["checkpoint"]
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            payload = _load_checkpoint(checkpoint_path, cfg, config_sha, epoch)
            model = runner.build_model(cfg, modules, device)
            _configure_model(model, payload)
            for target in TARGETS:
                cell_id = _cell_id(method, epoch, target)
                if cell_id in completed:
                    continue
                started = time.perf_counter()
                values, seen, _ = _evaluate_target(
                    model=model,
                    target=target,
                    dataset_module=dataset_module,
                    h2_utils=h2_utils,
                    domains=domains,
                    device=device,
                    img_size=int(cfg["img_size"]),
                    batch_size=int(args.batch_size),
                    spool_root=output / "temporary_spools" / cell_id,
                )
                cell = {
                    "status": "COMPLETE",
                    "cell_id": cell_id,
                    "scope": "medical",
                    "method": method,
                    "epoch": epoch,
                    "target": target,
                    "n_images": seen,
                    **values,
                    "checkpoint": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "config_sha256": config_sha,
                    "architecture_freeze_sha256": freeze["architecture_freeze_sha256"],
                    "clip_asset_sha256": freeze["clip_asset_sha256"],
                    "deployment": "native_h2_alpha0",
                    "rmt_inference": "NOT_APPLIED; RCA CIR IS TRAINING_ONLY",
                    "target_tuning": False,
                    "evaluator_git_sha": identity["evaluator_git_sha"],
                    "evaluator_sha256": identity["evaluator_sha256"],
                    "elapsed_seconds": time.perf_counter() - started,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                path = output / "cells" / f"{cell_id}.json"
                _atomic_json(path, cell)
                completed[cell_id] = cell
                _write_ledger(output, completed)
                _atomic_json(output / "MEDICAL_PROGRESS.json", {"status": "RUNNING", "completed_cells": len(completed), "planned_cells": len(expected), "last_cell": cell_id, "updated_at": cell["updated_at"]})
                print(f"completed Medical cell {cell_id}: {len(completed)}/{len(expected)}", flush=True)
            del model, payload
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if set(completed) != expected:
            raise RuntimeError(f"Medical ledger incomplete: missing={sorted(expected - set(completed))}")
        identity["status"] = "COMPLETED"
        identity["completed_cells"] = len(completed)
        _atomic_json(output / "MEDICAL_IDENTITY.json", identity)
        _atomic_json(output / "MEDICAL_COMPLETE.json", {"status": "COMPLETED", "completed_cells": len(completed), "planned_cells": len(expected), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        _summarize(output, completed, freeze)
        failed_marker = output / "MEDICAL_FAILED.json"
        if failed_marker.exists():
            failed_marker.unlink()
    except Exception as error:
        _atomic_json(output / "MEDICAL_FAILED.json", {"status": "FAILED", "error": repr(error), "completed_cells": len(completed), "planned_cells": len(expected), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
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
