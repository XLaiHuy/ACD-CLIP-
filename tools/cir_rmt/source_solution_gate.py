#!/usr/bin/env python3
"""Run the bounded source-only gate for the selected solution smoke checkpoint.

The selected smoke checkpoint is intentionally short.  This script reports its
metrics beside the frozen P/C0 E14 source baseline and refuses to call the
comparison a matched scientific result when the training horizons differ.
It never accesses Medical or MVTec data and never writes model checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model.phase2b_runtime import deploy_native_logits
from scripts.cir_rmt.eval_full import ManifestDataset
from tools.cir_rmt.identity import load_cir_config
from tools.cir_rmt.pre_full_run_diagnostics import (
    IMAGE_SIZE,
    _branch_rows,
    _capture,
    _concat,
    _deployment_rows,
    _heldout_rows,
    _load_payload,
    _make_model,
    _metric_rows,
    _sample_indices,
    _tail_rows,
)
from tools.cir_rmt.runtime import forward_cir


ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _baseline_rows(path: Path, epoch: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["epoch"]) == int(epoch) and row["method"] in {"P0", "C0", "P05", "C05"}:
                rows.append({
                    "scope": "frozen_baseline_e14",
                    "method": row["method"],
                    "epoch": int(row["epoch"]),
                    "training_horizon_comparable": True,
                    "n_images": int(row["n_images"]),
                    "pixel_auroc": float(row["pixel_auroc"]),
                    "pixel_ap": float(row["pixel_ap"]),
                    "image_auroc": float(row["image_auroc"]),
                    "image_ap": float(row["image_ap"]),
                })
    if len(rows) != 4:
        raise ValueError(f"expected four E{epoch} baseline rows in {path}, found {len(rows)}")
    return rows


def _selected_capture(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    archive = Path(args.archive_root)
    sample = json.loads((archive / "SOURCE_SAMPLE_IDENTITY.json").read_text(encoding="utf-8"))
    metadata = ROOT / "dataset" / "hub" / "VisA.jsonl"
    indices, rows = _sample_indices(metadata, int(sample["per_category"]), int(sample["sample_seed"]))
    if [int(row["manifest_index"]) for row in sample["selection"]] != indices:
        raise ValueError("frozen source sample selection does not reproduce from its recorded protocol")
    if [str(row["image_path"]) for row in sample["selection"]] != [str(row["image_path"]) for row in rows]:
        raise ValueError("frozen source sample image identity mismatch")
    dataset = ManifestDataset(Path(args.source_root), metadata, IMAGE_SIZE)
    loader = DataLoader(Subset(dataset, indices), batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=args.device.startswith("cuda"))
    cir_config = load_cir_config(Path(args.cir_config))
    parent_path = Path(cir_config.get("parent_config_path", "configs/phase2b_canonical_v1.json"))
    if not parent_path.is_absolute():
        parent_path = ROOT / parent_path
    parent_config = json.loads(parent_path.read_text(encoding="utf-8"))
    payload = _load_payload(Path(args.solution_checkpoint))
    device = torch.device(args.device)
    model = _make_model(parent_config, payload, Path(args.clip_asset), device)
    captures: list[dict[str, np.ndarray]] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    class_names: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device).float()
            names = [str(value) for value in batch["class_name"]]
            output = forward_cir(model, image, names, device, cir_config, domain="Industrial", dataset_name="VisA")
            native_prob, _ = deploy_native_logits(output.native_logits, image_size=IMAGE_SIZE, domain="Industrial")
            captures.append(_capture(output, native_prob, output.cir_segmentation_probability))
            labels.append(batch["label"].numpy().astype(np.int64))
            masks.append(batch["mask"].numpy().astype(np.float32)[:, 0])
            class_names.extend(names)
            del output, image
    data = _concat(captures)
    data["labels"] = np.concatenate(labels)
    data["masks"] = np.concatenate(masks)
    data["class_names"] = np.asarray(class_names, dtype=object)
    del model, loader, dataset, captures
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return data, {"sample": sample, "payload": payload, "parent_config": parent_config, "cir_config": cir_config}


def run(args: argparse.Namespace) -> None:
    archive = Path(args.archive_root)
    data, context = _selected_capture(args)
    baseline = _baseline_rows(archive / "SOURCE_BOUNDED_METRICS.csv", int(args.baseline_epoch))
    result_rows = list(baseline)
    for method in ("ANCHOR_SMOKE_C0", "ANCHOR_SMOKE_C05"):
        row = _metric_rows(int(context["payload"]["epoch"]), method, data)
        row.update({"scope": "selected_solution_smoke", "training_horizon_comparable": False})
        result_rows.append(row)
    _write_csv(archive / "SOURCE_SOLUTION_GATE_RESULTS.csv", result_rows, ["scope", "method", "epoch", "training_horizon_comparable", "n_images", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap"])

    holdout = set(context["sample"]["holdout_categories"])
    heldout_rows = []
    for method in ("ANCHOR_SMOKE_C0", "ANCHOR_SMOKE_C05"):
        for row in _heldout_rows(int(context["payload"]["epoch"]), method, data, holdout):
            row.update({"scope": "selected_solution_smoke", "training_horizon_comparable": False})
            heldout_rows.append(row)
    _write_csv(archive / "SOURCE_SOLUTION_GATE_HELDOUT.csv", heldout_rows, ["scope", "method", "epoch", "training_horizon_comparable", "split", "category", "n_images", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap"])

    tail_rows = []
    for row in _tail_rows(int(context["payload"]["epoch"]), "ANCHOR_SMOKE_C0", data["p0"], data["masks"]):
        row.update({"scope": "selected_solution_smoke", "training_horizon_comparable": False})
        tail_rows.append(row)
    for row in _tail_rows(int(context["payload"]["epoch"]), "ANCHOR_SMOKE_C05", data["p05"], data["masks"]):
        row.update({"scope": "selected_solution_smoke", "training_horizon_comparable": False})
        tail_rows.append(row)
    _write_csv(archive / "SOURCE_SOLUTION_GATE_AP_TAIL.csv", tail_rows, ["scope", "method", "epoch", "training_horizon_comparable", "cohort", "stat", "value", "n"])

    deployment_rows = []
    for row in _deployment_rows(int(context["payload"]["epoch"]), "ANCHOR_SMOKE_C0", data):
        row.update({"scope": "selected_solution_smoke", "training_horizon_comparable": False})
        deployment_rows.append(row)
    _write_csv(archive / "SOURCE_SOLUTION_GATE_DEPLOYMENT.csv", deployment_rows, ["scope", "method", "epoch", "training_horizon_comparable", "metric", "value"])

    branch_rows = []
    for row in _branch_rows(int(context["payload"]["epoch"]), "ANCHOR_SMOKE_C0", data):
        row.update({"scope": "selected_solution_smoke", "training_horizon_comparable": False})
        branch_rows.append(row)
    _write_csv(archive / "SOURCE_SOLUTION_GATE_BRANCH.csv", branch_rows, ["scope", "method", "epoch", "training_horizon_comparable", "branch", "image_auroc", "image_ap", "mean_score", "n_images"])

    status = {
        "implementation_smoke": "PASS",
        "scientific_gate": "INCONCLUSIVE",
        "source_only": True,
        "reason": "Selected solution has only the E02/5-step smoke horizon; frozen P/C0 comparison is E14. No matched-horizon scientific conclusion is allowed.",
        "baseline_epoch": int(args.baseline_epoch),
        "selected_solution_epoch": int(context["payload"]["epoch"]),
        "selected_solution_global_step": int(context["payload"].get("global_step", -1)),
        "representation_drift": "INCONCLUSIVE_HORIZON_MISMATCH",
        "culprit_image_drift": "INCONCLUSIVE_HORIZON_MISMATCH",
        "heldout_seen_gate": "INCONCLUSIVE_HORIZON_MISMATCH",
        "ap_tail_gate": "INCONCLUSIVE_HORIZON_MISMATCH",
        "deployment_gate": "INCONCLUSIVE_HORIZON_MISMATCH",
        "solution_hyperparameters": {"lambda_image_anchor": 0.001, "scope": "image_adapter_parameters_only", "reference_epoch": 14, "reference_checkpoint_sha256": context["payload"].get("image_anchor", {}).get("reference_checkpoint_sha256")},
        "medical_evaluation": "NOT_RUN",
        "mvtec_evaluation": "NOT_RUN",
        "full_run_authorized": False,
    }
    (archive / "SOURCE_SOLUTION_GATE_STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--solution-checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--cir-config", type=Path, required=True)
    parser.add_argument("--baseline-epoch", type=int, default=14)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
