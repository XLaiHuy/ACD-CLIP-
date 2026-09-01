#!/usr/bin/env python3
"""Evaluate the PA control on the frozen VisA source sample.

This evaluator is deliberately native-only: PA has no CIR/RMT inference path.
The existing P/C_OLD/A source rows are consumed later by the factorial report;
only the new PA checkpoints are forwarded here.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from evaluation.evaluator import image_score
from model.phase2b_runtime import build_phase2b_frozen, configure_canonical_fp32, forward_phase2b
from scripts.cir_rmt.eval_full import ManifestDataset, _shutdown_loader
from tools.cir_rmt.identity import config_sha256
from tools.cir_rmt.pre_full_run_diagnostics import (
    IMAGE_SIZE,
    _branch_rows,
    _corr,
    _f,
    _linear_cka,
    _mean_cosine,
    _metrics_for,
    _norm_ratio,
    _pairwise_geometry_corr,
    _sample_indices,
    _stage_probability,
    _tail_rows,
    _training_probability,
    _deployment_rows,
)


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = (10, 12, 14, 16, 18, 20)
SAMPLE_ARCHIVE = ROOT / "research_artifacts/cir_rmt_v2/pre_full_run_root_cause_lock_20260831"
SAMPLE_IDENTITY = SAMPLE_ARCHIVE / "SOURCE_SAMPLE_IDENTITY.json"
METRIC_FIELDS = (
    "epoch",
    "method",
    "n_images",
    "pixel_auroc",
    "pixel_ap",
    "image_auroc",
    "image_ap",
    "checkpoint",
    "checkpoint_sha256",
    "config_sha256",
    "parent_config_sha256",
    "evaluator_git_sha",
    "evaluator_sha256",
    "source_sample_sha256",
    "evaluation_seconds",
    "status",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _concat(captures: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    axis_one = {"seg_pooled", "seg_patch", "det_pooled", "group_margins", "native_margin", "stage_probability"}
    result: dict[str, np.ndarray] = {}
    for key in captures[0]:
        axis = 1 if key in axis_one else 0
        result[key] = np.concatenate([capture[key] for capture in captures], axis=axis)
    return result


def _capture(output: Any) -> dict[str, np.ndarray]:
    seg = output.seg_features.detach().float().cpu()
    det = output.det_features.detach().float().cpu()
    text = output.text_features.detach().float().cpu()
    patch_indices = torch.linspace(0, seg.shape[2] - 1, 32).long()
    normalized_seg = torch.nn.functional.normalize(seg, dim=-1)
    # The canonical forward returns text as [n_groups, B, D, 2], while the
    # margin contraction uses the batch-first [B, n_groups, D, 2] view.
    normalized_text = torch.nn.functional.normalize(text, dim=-2).permute(1, 0, 2, 3)
    group_margins = torch.einsum("sbpd,bgdc->sbpgc", normalized_seg, normalized_text)
    group_margins = group_margins[..., 1] - group_margins[..., 0]
    deployed = output.deployed_segmentation_probability.detach().float().cpu().numpy()
    raw = _training_probability(output.native_logits.detach().float()).detach().cpu().numpy()[:, 1]
    return {
        "p0": deployed,
        "raw": raw,
        "classification_probability": output.classification_probability.detach().float().cpu().numpy(),
        "seg_pooled": seg.mean(dim=2).numpy(),
        "seg_patch": seg[:, :, patch_indices, :].numpy(),
        "det_pooled": det.mean(dim=2).numpy() if det.ndim == 4 else det.numpy(),
        "text": text.numpy(),
        "group_margins": group_margins.numpy(),
        "native_margin": output.native_margin.detach().float().cpu().numpy(),
        "stage_probability": _stage_probability(output.native_logits.detach().float()).cpu().numpy(),
    }


def _validate_sample() -> tuple[list[int], list[dict[str, Any]], set[str]]:
    sample = json.loads(SAMPLE_IDENTITY.read_text(encoding="utf-8"))
    metadata = ROOT / "dataset/hub/VisA.jsonl"
    indices, rows = _sample_indices(metadata, int(sample["per_category"]), int(sample["sample_seed"]))
    expected_indices = [int(row["manifest_index"]) for row in sample["selection"]]
    expected_paths = [str(row["image_path"]) for row in sample["selection"]]
    if indices != expected_indices or [str(row["image_path"]) for row in rows] != expected_paths:
        raise ValueError("frozen VisA source sample no longer reproduces")
    return indices, rows, set(str(value) for value in sample["holdout_categories"])


def _load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _model_for(path: Path, parent_config: Mapping[str, Any], clip_asset: Path, device: torch.device, *, pa: bool) -> Any:
    payload = _load_payload(path)
    if pa:
        identity = payload.get("control_identity", {})
        if identity.get("control_id") != "PA_PHASE2B_IMAGE_ANCHOR_V1" or identity.get("cir_training") is not False or identity.get("rmt_training") is not False:
            raise ValueError(f"PA checkpoint identity mismatch: {path}")
    elif payload.get("protocol_version") != parent_config.get("protocol_version"):
        raise ValueError(f"parent protocol mismatch: {path}")
    model = build_phase2b_frozen(parent_config, payload, clip_asset, device)
    model.eval()
    del payload
    gc.collect()
    return model


def _evaluate_checkpoint(
    *,
    checkpoint: Path,
    parent_config: Mapping[str, Any],
    clip_asset: Path,
    dataset: Any,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pa: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    started = time.perf_counter()
    model = _model_for(checkpoint, parent_config, clip_asset, device, pa=pa)
    kwargs: dict[str, Any] = {"batch_size": int(batch_size), "shuffle": False, "num_workers": int(num_workers), "pin_memory": device.type == "cuda"}
    if int(num_workers) > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(Subset(dataset, list(indices)), **kwargs)
    loader_iter: Any = None
    captures: list[dict[str, np.ndarray]] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    class_names: list[str] = []
    try:
        loader_iter = iter(loader)
        with torch.inference_mode():
            for batch in loader_iter:
                image = batch["image"].to(device, non_blocking=device.type == "cuda").float()
                names = [str(value) for value in batch["class_name"]]
                output = forward_phase2b(model, image, names, device, parent_config, domain="Industrial", require_grad=False, dataset_name="VisA")
                captures.append(_capture(output))
                labels.append(batch["label"].numpy().astype(np.int64))
                masks.append(batch["mask"].numpy().astype(np.float32)[:, 0])
                class_names.extend(names)
                del output, image
    finally:
        _shutdown_loader(loader, loader_iter)
        del loader, loader_iter, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    data = _concat(captures)
    data["labels"] = np.concatenate(labels)
    data["masks"] = np.concatenate(masks)
    data["class_names"] = np.asarray(class_names, dtype=object)
    maps = data["p0"]
    scores = np.asarray([image_score(float(c), float(m.max()), "Industrial") for c, m in zip(data["classification_probability"], maps)], dtype=np.float64)
    metrics = _metrics_for(maps, data["masks"], scores, data["labels"])
    checkpoint_sha = _sha256(checkpoint)
    row = {"epoch": int(_load_payload(checkpoint)["epoch"]), "method": "PA", "n_images": len(indices), **metrics, "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha, "evaluation_seconds": time.perf_counter() - started}
    tails = [dict(item, checkpoint_sha256=checkpoint_sha, source="new_PA_checkpoint") for item in _tail_rows(int(row["epoch"]), "PA", maps, data["masks"])]
    deployment = [dict(item, checkpoint_sha256=checkpoint_sha, source="new_PA_checkpoint") for item in _deployment_rows(int(row["epoch"]), "PA", data)]
    branches = [dict(item, checkpoint_sha256=checkpoint_sha, source="new_PA_checkpoint") for item in _branch_rows(int(row["epoch"]), "PA", data)]
    return row, data, tails, deployment, branches


def _feature_rows(epoch: int, parent: Mapping[str, np.ndarray], pa: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(signal: str, axis: str, before: np.ndarray, after: np.ndarray) -> None:
        before = np.asarray(before, dtype=np.float64)
        after = np.asarray(after, dtype=np.float64)
        before_flat = before.reshape(-1, before.shape[-1])
        after_flat = after.reshape(-1, after.shape[-1])
        rows.append({
            "epoch": int(epoch),
            "reference": f"P_E{int(epoch):02d}",
            "comparison": f"PA_E{int(epoch):02d}",
            "signal": signal,
            "axis": axis,
            "n_images": int(before.shape[0]),
            "mean_cosine": _mean_cosine(before_flat, after_flat),
            "norm_ratio": _norm_ratio(before_flat, after_flat),
            "linear_cka": _linear_cka(before_flat, after_flat, seed=int(epoch)),
            "pairwise_geometry_corr": _pairwise_geometry_corr(before.reshape(before.shape[0], -1), after.reshape(after.shape[0], -1)),
            "mean_abs_delta": _f(np.mean(np.abs(before - after))),
            "geometry_rows": int(before.shape[0]),
        })

    for stage in range(parent["seg_pooled"].shape[0]):
        add(f"seg_stage{stage}", "feature", parent["seg_pooled"][stage], pa["seg_pooled"][stage])
        add(f"seg_stage{stage}", "patch_subsample", parent["seg_patch"][stage], pa["seg_patch"][stage])
        add(f"det_stage{stage}", "feature", parent["det_pooled"][stage], pa["det_pooled"][stage])
        add(f"group_margin_stage{stage}", "patch_group", parent["group_margins"][stage], pa["group_margins"][stage])
        add(f"native_fused_margin_stage{stage}", "patch", parent["native_margin"][stage], pa["native_margin"][stage])
        add(f"stage_probability_stage{stage}", "patch", parent["stage_probability"][stage], pa["stage_probability"][stage])
    add("text_normal_abnormal_groups", "feature", parent["text"].transpose(0, 1, 3, 2), pa["text"].transpose(0, 1, 3, 2))
    add("native_deployed_map", "pixel", parent["p0"], pa["p0"])
    add("native_raw_map", "pixel", parent["raw"], pa["raw"])
    return rows


def run(args: argparse.Namespace) -> None:
    configure_canonical_fp32()
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    indices, sample_rows, holdout = _validate_sample()
    sample_sha = _sha256(SAMPLE_IDENTITY)
    parent_config = json.loads(args.parent_config.expanduser().resolve().read_text(encoding="utf-8"))
    parent_config_sha = config_sha256(parent_config)
    dataset = ManifestDataset(args.source_root.expanduser().resolve(), ROOT / "dataset/hub/VisA.jsonl", IMAGE_SIZE)
    device = torch.device(args.device)
    pa_rows: list[dict[str, Any]] = []
    tails: list[dict[str, Any]] = []
    deployments: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    evaluator_git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    evaluator_sha = _sha256(ROOT / "scripts/cir_rmt/eval_full.py")
    for epoch in EPOCHS:
        parent_path = args.parent_run_root.expanduser().resolve() / "phase2b" / "checkpoints" / f"adapter_{epoch}.pth"
        pa_path = args.pa_run_root.expanduser().resolve() / "visa" / "seed0" / "checkpoints" / f"adapter_{epoch}.pth"
        if not parent_path.is_file() or not pa_path.is_file():
            raise FileNotFoundError(f"missing paired source checkpoint at E{epoch}: {parent_path} / {pa_path}")
        p_row, p_data, _, _, _ = _evaluate_checkpoint(checkpoint=parent_path, parent_config=parent_config, clip_asset=args.clip_asset.expanduser().resolve(), dataset=dataset, indices=indices, device=device, batch_size=args.batch_size, num_workers=args.num_workers, pa=False)
        pa_row, pa_data, pa_tails, pa_deploy, pa_branch = _evaluate_checkpoint(checkpoint=pa_path, parent_config=parent_config, clip_asset=args.clip_asset.expanduser().resolve(), dataset=dataset, indices=indices, device=device, batch_size=args.batch_size, num_workers=args.num_workers, pa=True)
        pa_rows.append({**pa_row, "config_sha256": args.pa_config_sha256, "parent_config_sha256": parent_config_sha, "evaluator_git_sha": evaluator_git_sha, "evaluator_sha256": evaluator_sha, "source_sample_sha256": sample_sha, "status": "COMPLETE"})
        tails.extend(pa_tails)
        deployments.extend(pa_deploy)
        branches.extend(pa_branch)
        feature_rows.extend(_feature_rows(epoch, p_data, pa_data))
        for metric in ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap"):
            score_rows.append({"epoch": epoch, "metric": metric, "parent": p_row[metric], "pa": pa_row[metric], "pa_minus_parent": None if p_row[metric] is None or pa_row[metric] is None else float(pa_row[metric]) - float(p_row[metric])})
        print(f"completed PA/P native source evaluation E{epoch:02d} ({len(indices)} images)", flush=True)
    _write_csv(output / "PA_SOURCE_RESULTS.csv", pa_rows, METRIC_FIELDS)
    _write_csv(output / "PA_SOURCE_AP_TAIL.csv", tails, ["epoch", "method", "cohort", "stat", "value", "n", "checkpoint_sha256", "source"])
    _write_csv(output / "PA_SOURCE_DEPLOYMENT.csv", deployments, ["epoch", "method", "metric", "value", "checkpoint_sha256", "source"])
    _write_csv(output / "PA_SOURCE_IMAGE_BRANCH.csv", branches, ["epoch", "method", "branch", "image_auroc", "image_ap", "mean_score", "n_images", "checkpoint_sha256", "source"])
    _write_csv(output / "PA_SOURCE_FEATURE_DRIFT.csv", feature_rows, ["epoch", "reference", "comparison", "signal", "axis", "n_images", "mean_cosine", "norm_ratio", "linear_cka", "pairwise_geometry_corr", "mean_abs_delta", "geometry_rows"])
    _write_csv(output / "PA_SOURCE_METRIC_DELTAS.csv", score_rows, ["epoch", "metric", "parent", "pa", "pa_minus_parent"])
    (output / "PA_SOURCE_EVAL_STATUS.json").write_text(json.dumps({
        "status": "PASS",
        "source_only": True,
        "methods": ["PA"],
        "paired_reference": "P",
        "epochs": list(EPOCHS),
        "n_images": len(indices),
        "sample_identity": str(SAMPLE_IDENTITY),
        "sample_sha256": sample_sha,
        "sample_rows": len(sample_rows),
        "holdout_categories": sorted(holdout),
        "parent_config_sha256": parent_config_sha,
        "pa_config_sha256": args.pa_config_sha256,
        "medical": "NOT_RUN",
        "mvtec": "NOT_RUN",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pa-run-root", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, default=ROOT / "configs/phase2b_canonical_v1.json")
    parser.add_argument("--pa-config-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
