#!/usr/bin/env python3
"""Post-hoc source-only evaluation for a matched-horizon E10/E12/E14 run.

Only the newly trained CIR checkpoints are forwarded.  Existing P/C0 rows are
copied from the frozen bounded archive and are never recomputed.  Features are
retained only long enough to calculate post-hoc representation drift between
the newly trained E10/E12/E14 checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
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
    _linear_cka,
    _load_payload,
    _make_model,
    _mean_cosine,
    _metric_rows,
    _norm_ratio,
    _pairwise_geometry_corr,
    _sample_indices,
    _tail_rows,
)
from tools.cir_rmt.runtime import forward_cir


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = (10, 12, 14)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _parent_config(cir_config: Mapping[str, Any]) -> dict[str, Any]:
    parent_path = Path(cir_config.get("parent_config_path", "configs/phase2b_canonical_v1.json"))
    if not parent_path.is_absolute():
        parent_path = ROOT / parent_path
    return json.loads(parent_path.read_text(encoding="utf-8"))


def _frozen_sample(archive: Path) -> tuple[list[int], list[dict[str, Any]], set[str]]:
    sample = json.loads((archive / "SOURCE_SAMPLE_IDENTITY.json").read_text(encoding="utf-8"))
    indices, rows = _sample_indices(ROOT / "dataset" / "hub" / "VisA.jsonl", int(sample["per_category"]), int(sample["sample_seed"]))
    if indices != [int(row["manifest_index"]) for row in sample["selection"]] or [str(row["image_path"]) for row in rows] != [str(row["image_path"]) for row in sample["selection"]]:
        raise ValueError("source sample no longer reproduces from the frozen protocol")
    return indices, rows, set(sample["holdout_categories"])


def _feature_drift(epoch: int, previous: Mapping[str, np.ndarray], current: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def add(signal: str, axis: str, before: np.ndarray, after: np.ndarray) -> None:
        before = np.asarray(before, dtype=np.float64)
        after = np.asarray(after, dtype=np.float64)
        rows.append({
            "epoch": epoch,
            "signal": signal,
            "axis": axis,
            "mean_cosine": _mean_cosine(before, after),
            "norm_ratio": _norm_ratio(before, after),
            "linear_cka": _linear_cka(before.reshape(-1, before.shape[-1]), after.reshape(-1, after.shape[-1]), seed=0),
            "pairwise_geometry_corr": _pairwise_geometry_corr(before.reshape(-1, before.shape[-1]), after.reshape(-1, after.shape[-1])),
        })
    for stage in range(previous["seg_pooled"].shape[0]):
        add(f"seg_stage{stage}", "feature", previous["seg_pooled"][stage], current["seg_pooled"][stage])
    previous_det = previous["det"].mean(axis=2) if previous["det"].ndim == 4 else previous["det"]
    current_det = current["det"].mean(axis=2) if current["det"].ndim == 4 else current["det"]
    for stage in range(previous_det.shape[0]):
        add(f"det_stage{stage}", "feature", previous_det[stage], current_det[stage])
    for key, axis in (("native_weights", "group_class"), ("group_margins", "patch_group"), ("native_margin", "patch"), ("cir_margin", "patch")):
        for stage in range(previous[key].shape[0]):
            before = previous[key][stage]
            after = current[key][stage]
            if before.ndim == 1:
                before = before[:, None]
                after = after[:, None]
            add(f"{key}_stage{stage}", axis, before, after)
    return rows


def _read_baseline(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["epoch"]) in EPOCHS and row["method"] in {"P0", "P05", "C0", "C05"}:
                rows.append({"epoch": int(row["epoch"]), "method": row["method"], "n_images": int(row["n_images"]), "pixel_auroc": float(row["pixel_auroc"]), "pixel_ap": float(row["pixel_ap"]), "image_auroc": float(row["image_auroc"]), "image_ap": float(row["image_ap"]), "recomputed": False, "source": "frozen_SOURCE_BOUNDED_METRICS.csv"})
    if len(rows) != 12:
        raise ValueError(f"expected 12 frozen P/C0 E10/E12/E14 rows, found {len(rows)}")
    return rows


def _evaluate_checkpoint(*, epoch: int, checkpoint: Path, parent_config: Mapping[str, Any], cir_config: Mapping[str, Any], dataset: Any, indices: list[int], clip_asset: Path, device: torch.device, batch_size: int, num_workers: int, holdout: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    payload = _load_payload(checkpoint)
    model = _make_model(parent_config, payload, clip_asset, device)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    captures: list[dict[str, np.ndarray]] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    class_names: list[str] = []
    inference_started = time.perf_counter()
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
    inference_seconds = time.perf_counter() - inference_started
    data = _concat(captures)
    data["labels"] = np.concatenate(labels)
    data["masks"] = np.concatenate(masks)
    data["class_names"] = np.asarray(class_names, dtype=object)
    metric_rows: list[dict[str, Any]] = []
    for method in ("C0", "C05"):
        row = _metric_rows(epoch, method, data)
        row.update({"recomputed": True, "source": "new_CIR_checkpoint", "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(), "evaluation_seconds": time.perf_counter() - started})
        metric_rows.append(row)
    tails: list[dict[str, Any]] = []
    for method, maps in (("C0", data["p0"]), ("C05", data["p05"])):
        for row in _tail_rows(epoch, method, maps, data["masks"]):
            row.update({"checkpoint_sha256": metric_rows[0]["checkpoint_sha256"], "source": "new_CIR_checkpoint"})
            tails.append(row)
    deployment = [dict(row, checkpoint_sha256=metric_rows[0]["checkpoint_sha256"], source="new_CIR_checkpoint") for row in _deployment_rows(epoch, "C0", data)]
    branches = [dict(row, checkpoint_sha256=metric_rows[0]["checkpoint_sha256"], source="new_CIR_checkpoint") for row in _branch_rows(epoch, "C0", data)]
    heldout = []
    for method in ("C0", "C05"):
        heldout.extend(dict(row, checkpoint_sha256=metric_rows[0]["checkpoint_sha256"], source="new_CIR_checkpoint") for row in _heldout_rows(epoch, method, data, holdout))
    features = {key: data[key] for key in ("seg_pooled", "det", "native_weights", "group_margins", "native_margin", "cir_margin")}
    timing = {"epoch": epoch, "checkpoint_sha256": metric_rows[0]["checkpoint_sha256"], "inference_seconds": inference_seconds, "total_evaluation_seconds": time.perf_counter() - started, "n_images": len(indices)}
    del model, loader, dataset, captures, data, payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metric_rows, tails, deployment, branches, heldout, features, timing


def run(args: argparse.Namespace) -> None:
    archive = Path(args.archive_root)
    indices, _, holdout = _frozen_sample(archive)
    cir_config = load_cir_config(Path(args.config))
    parent_config = _parent_config(cir_config)
    dataset = ManifestDataset(Path(args.source_root), ROOT / "dataset" / "hub" / "VisA.jsonl", IMAGE_SIZE)
    device = torch.device(args.device)
    baseline_archive = Path(args.baseline_archive_root) if args.baseline_archive_root else archive
    baseline = _read_baseline(baseline_archive / "SOURCE_BOUNDED_METRICS.csv")
    metrics: list[dict[str, Any]] = []
    tails: list[dict[str, Any]] = []
    deployments: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    heldouts: list[dict[str, Any]] = []
    representations: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []
    previous_features: dict[str, np.ndarray] | None = None
    for epoch in EPOCHS:
        checkpoint = Path(args.run_root) / "visa" / "seed0" / "checkpoints" / f"epoch_{epoch:02d}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result = _evaluate_checkpoint(epoch=epoch, checkpoint=checkpoint, parent_config=parent_config, cir_config=cir_config, dataset=dataset, indices=indices, clip_asset=Path(args.clip_asset), device=device, batch_size=int(args.batch_size), num_workers=int(args.num_workers), holdout=holdout)
        epoch_metrics, epoch_tails, epoch_deploy, epoch_branches, epoch_heldout, features, timing = result
        metrics.extend(epoch_metrics); tails.extend(epoch_tails); deployments.extend(epoch_deploy); branches.extend(epoch_branches); heldouts.extend(epoch_heldout); timings.append(timing)
        if previous_features is not None:
            for row in _feature_drift(epoch, previous_features, features):
                row.update({"method": "C_new_checkpoint", "comparison": f"C_E{epoch-2:02d}_to_C_E{epoch:02d}"})
                representations.append(row)
        previous_features = features
    _write_csv(archive / "MATCHED_HORIZON_SOURCE_RESULTS.csv", baseline + [dict(row, n_images=int(row["n_images"]), source=row.get("source", "new_CIR_checkpoint")) for row in metrics], ["epoch", "method", "n_images", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap", "recomputed", "source", "checkpoint_sha256", "evaluation_seconds"])
    _write_csv(archive / "MATCHED_HORIZON_EVAL_TIMES.csv", timings, ["epoch", "checkpoint_sha256", "n_images", "inference_seconds", "total_evaluation_seconds"])
    _write_csv(archive / "MATCHED_HORIZON_AP_TAIL.csv", tails, ["epoch", "method", "cohort", "stat", "value", "n", "checkpoint_sha256", "source"])
    _write_csv(archive / "MATCHED_HORIZON_DEPLOYMENT.csv", deployments, ["epoch", "method", "metric", "value", "checkpoint_sha256", "source"])
    _write_csv(archive / "MATCHED_HORIZON_BRANCH.csv", branches, ["epoch", "method", "branch", "image_auroc", "image_ap", "mean_score", "n_images", "checkpoint_sha256", "source"])
    _write_csv(archive / "MATCHED_HORIZON_HELDOUT.csv", heldouts, ["epoch", "method", "split", "category", "n_images", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap", "checkpoint_sha256", "source"])
    _write_csv(archive / "MATCHED_HORIZON_REPRESENTATION.csv", representations, ["epoch", "method", "comparison", "signal", "axis", "mean_cosine", "norm_ratio", "linear_cka", "pairwise_geometry_corr"])
    status = {"status": "PASS", "source_only": True, "epochs": list(EPOCHS), "new_checkpoints_only": True, "existing_p_c0_e10_e12_e14_reused": True, "p_c0_recomputed": False, "baseline_archive": str(baseline_archive), "medical": "NOT_RUN", "mvtec": "NOT_RUN", "n_images": len(indices), "timing_artifact": "MATCHED_HORIZON_EVAL_TIMES.csv"}
    (archive / "MATCHED_HORIZON_SOURCE_EVAL_STATUS.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--baseline-archive-root", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
