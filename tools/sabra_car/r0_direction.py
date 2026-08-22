"""Additive, cache-only SABRA-CAR R0 signed-direction oracle.

This module never imports a Medical dataset path and never updates Phase2B.
It validates the canonical VisA cache before opening source masks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.phase2b_runtime import deploy_with_delta
from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata, safe_data_path
from utils import calculate_seg_loss

ROOT = Path(__file__).resolve().parents[2]
BASE_SHA = "a986bcfee41c31f03d38e449efb8826d56c90525"
CHECKPOINT_SHA = "6643cd68eafabf9acdb724242ef5b2d1fbc4bf7e9d2ba7ad6c47776ea646da80"
CLIP_SHA = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
CONFIG_SHA = "edf5745686e3d3d0d3b4142341569da06ad5b54025a779b78d83f74303ce67fc"
METADATA_SHA = "468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842"
MARGIN_SCALE = 19.840438842773438
ALPHAS = (0.0, 0.125, 0.25, 0.5, 1.0)
EPSILON = 1e-8
IMAGE_SIZE = 518
PATCH_GRID = (37, 37)
PATCHES = 1369
STAGES = 3
CORE_HASHES = {
    "model/phase2b_runtime.py": "da65c64a07a9e22501ec5e363128ac97389f06f5a32d5ba6c6fff0c30f0f17d5",
    "model/phase2b_schedule.py": "adafe7236b73c678ef280ac9fd1c1afcf61062eda63ce2cee094dc7a31ceb30f",
    "tools/sabra/need.py": "c00923795a46d9849c0dfd31f7408bba2c971cc35f24ebbb55a5cf4b81aa832b",
    "tools/sabra/relational.py": "efb942a4b41a7d97104a1d661515167a5af5ffcb40f758c8276ce5ddb7dfb355",
    "tools/sabra/trust_v2/fast_geometry.py": "259ed74beeedbfef2980fe23d3f14bad31cd380e456d13fb606f2b5220c89ee8",
    "tools/sabra/trust_v2/numerical.py": "104e8d03fd6864a999c6ed14f345c784f9ad144a51d65482b1d7c025f5b1e4c0",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def configure_runtime() -> None:
    torch.set_grad_enabled(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def classify_actions(utility: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(utility, torch.Tensor):
        return torch.where(
            utility > EPSILON,
            torch.ones_like(utility, dtype=torch.int8),
            torch.where(utility < -EPSILON, -torch.ones_like(utility, dtype=torch.int8), torch.zeros_like(utility, dtype=torch.int8)),
        )
    utility = np.asarray(utility)
    return np.where(utility > EPSILON, 1, np.where(utility < -EPSILON, -1, 0)).astype(np.int8)


def intervention_delta(correction: torch.Tensor, native_logits: torch.Tensor) -> torch.Tensor:
    if native_logits.ndim != 4 or native_logits.shape[0] != STAGES or native_logits.shape[-1] != 2:
        raise ValueError("native logits must be [3,B,1369,2]")
    if correction.shape != native_logits.shape[1:3]:
        raise ValueError("correction must be [B,1369]")
    one_stage = torch.stack([torch.zeros_like(correction), correction], dim=-1)
    delta = one_stage.unsqueeze(0).expand_as(native_logits)
    if torch.count_nonzero(delta[..., 0]).item() != 0:
        raise AssertionError("normal-channel correction is nonzero")
    if not torch.equal(delta[0], delta[1]) or not torch.equal(delta[1], delta[2]):
        raise AssertionError("correction is not shared across stages")
    return delta


def deploy_correction(native_logits: torch.Tensor, correction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return deploy_with_delta(native_logits, intervention_delta(correction, native_logits), domain="Industrial")


def canonical_loss_per_image(probability: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if probability.ndim != 4 or probability.shape[1] != 2:
        raise ValueError("probability must be [B,2,H,W]")
    if mask.ndim != 4 or mask.shape[1] != 1 or probability.shape[0] != mask.shape[0]:
        raise ValueError("mask must be [B,1,H,W]")
    target = mask[:, 0].float()
    abnormal = probability[:, 1].float()
    normal = probability[:, 0].float()
    smooth = 1e-5
    pt = torch.where(
        target > 0.5,
        abnormal * (1.0 - smooth) + normal * smooth + smooth,
        normal * (1.0 - smooth) + abnormal * smooth + smooth,
    )
    focal = (-(1.0 - pt).square() * pt.log()).flatten(1).mean(1)
    normal_target = 1.0 - target
    normal_dice = 1.0 - (
        2.0 * (normal * normal_target).flatten(1).sum(1) + 1.0
    ) / (normal.flatten(1).sum(1) + normal_target.flatten(1).sum(1) + 1.0)
    abnormal_dice = 1.0 - (
        2.0 * (abnormal * target).flatten(1).sum(1) + 1.0
    ) / (abnormal.flatten(1).sum(1) + target.flatten(1).sum(1) + 1.0)
    return focal + normal_dice + abnormal_dice


def _descending_score_groups(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64, copy=False)
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    cumulative_positive = np.cumsum(sorted_labels, dtype=np.int64)
    positive_at_end = cumulative_positive[ends]
    group_positive = np.diff(np.r_[0, positive_at_end])
    group_size = np.diff(np.r_[0, ends + 1])
    return group_positive.astype(np.float64), group_size.astype(np.float64)


def exact_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    group_positive, group_size = _descending_score_groups(scores, labels)
    positives = float(group_positive.sum())
    if positives <= 0.0:
        raise ValueError("average precision requires a positive label")
    cumulative_positive = np.cumsum(group_positive)
    cumulative_count = np.cumsum(group_size)
    precision = cumulative_positive / cumulative_count
    return float(np.sum((group_positive / positives) * precision))


def exact_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if scores.shape != labels.shape or scores.size == 0:
        raise ValueError("score/label arrays must be non-empty and equal")
    if not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("invalid metric inputs")
    if labels.min() == labels.max():
        raise ValueError("both binary labels are required")
    group_positive, group_size = _descending_score_groups(scores, labels)
    group_negative = group_size - group_positive
    positives = float(group_positive.sum())
    negatives = float(group_negative.sum())
    cumulative_negative = np.cumsum(group_negative)
    negative_below = negatives - cumulative_negative
    auroc = float(np.sum(group_positive * (negative_below + 0.5 * group_negative)) / (positives * negatives))
    return {
        "pAUROC": auroc,
        "pAP": exact_average_precision(scores, labels),
    }


def metadata_and_root(data_root: Path) -> tuple[dict[str, dict[str, Any]], Path]:
    if (data_root / "VisA_20220922").is_dir():
        data_root = data_root / "VisA_20220922"
    rows = read_visa_metadata(ROOT / "dataset/hub/VisA.jsonl")
    return {str(row["image_path"]): row for row in rows}, data_root.resolve()


def load_masks(image_paths: Iterable[str], metadata: dict[str, dict[str, Any]], data_root: Path) -> np.ndarray:
    paths = list(image_paths)
    masks = np.zeros((len(paths), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for index, image_path in enumerate(paths):
        row = metadata[image_path]
        if int(row["label"]) == 0:
            continue
        mask_path = safe_data_path(data_root, str(row["mask_path"]))
        with Image.open(mask_path) as handle:
            masks[index] = np.asarray(
                handle.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST),
                dtype=np.uint8,
            ) > 0
    return masks


def validate_cache(cache_root: Path, checkpoint: Path, clip_asset: Path) -> dict[str, Any]:
    manifest_path = cache_root / "GT_FREE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("GT_FREE_CACHE_FINALIZED") is not True or manifest.get("immutable") is not True:
        raise RuntimeError("canonical GT-free cache is not finalized and immutable")
    if manifest.get("dataset") != "VisA" or manifest.get("classes") != list(EXPECTED_VISA_CLASSES):
        raise RuntimeError("VisA cache inventory mismatch")
    if int(manifest.get("record_count", -1)) != 2162:
        raise RuntimeError("VisA record count mismatch")
    source_hashes = manifest.get("source_hashes", {})
    expected = {
        "checkpoint": CHECKPOINT_SHA,
        "clip_asset": CLIP_SHA,
        "config": CONFIG_SHA,
        "metadata": METADATA_SHA,
        **CORE_HASHES,
    }
    observed = {
        "checkpoint": sha256_file(checkpoint),
        "clip_asset": sha256_file(clip_asset),
        "config": sha256_file(ROOT / "configs/phase2b_canonical_v1.json"),
        "metadata": sha256_file(ROOT / "dataset/hub/VisA.jsonl"),
        **{relative: sha256_file(ROOT / relative) for relative in CORE_HASHES},
    }
    if observed != expected or any(source_hashes.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"cache provenance mismatch: observed={observed}")
    shard_hashes: dict[str, str] = {}
    for class_name in EXPECTED_VISA_CLASSES:
        path = cache_root / "gt_free_cache" / f"{class_name}.npz"
        digest = sha256_file(path)
        if digest != manifest["shards"].get(class_name):
            raise RuntimeError(f"cache shard hash mismatch: {class_name}")
        shard_hashes[class_name] = digest
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=ROOT, check=True)
    return {
        "status": "PASS",
        "base_sha": BASE_SHA,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkpoint": str(checkpoint),
        "source_hashes": observed,
        "shard_hashes": shard_hashes,
        "record_count": 2162,
        "classes": list(EXPECTED_VISA_CLASSES),
        "medical_reads": 0,
        "phase2b_training_steps": 0,
    }


def load_shard(cache_root: Path, class_name: str) -> dict[str, np.ndarray]:
    path = cache_root / "gt_free_cache" / f"{class_name}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {
            "native_logits": np.asarray(data["native_logits"], dtype=np.float32),
            "native_pixel_probability": np.asarray(data["native_pixel_probability"], dtype=np.float32),
            "image_path": data["image_path"].astype(str),
        }


def utility_for_batch(native: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shared = torch.zeros(native.shape[1:3], device=native.device, dtype=torch.float32, requires_grad=True)
    probability, _ = deploy_correction(native, shared)
    loss = calculate_seg_loss(probability, mask) * native.shape[1]
    gradient = torch.autograd.grad(loss, shared, only_inputs=True, create_graph=False)[0]
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("non-finite signed utility")
    return -gradient.detach(), canonical_loss_per_image(probability.detach(), mask).detach()


def correction_from_actions(actions: np.ndarray, alpha: float, positive_only: bool) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.int8)
    selected = (actions > 0).astype(np.float32) if positive_only else actions.astype(np.float32)
    return selected * np.float32(alpha * MARGIN_SCALE)


def evaluate_correction(
    native_logits: np.ndarray,
    masks: np.ndarray,
    corrections: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = native_logits.shape[0]
    scores = np.empty((count, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    losses = np.empty(count, dtype=np.float64)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        native = torch.from_numpy(native_logits[start:stop]).permute(1, 0, 2, 3).to(device)
        correction = torch.from_numpy(corrections[start:stop]).to(device)
        mask = torch.from_numpy(masks[start:stop, None].astype(np.float32)).to(device)
        with torch.no_grad():
            probability, _ = deploy_correction(native, correction)
            loss = canonical_loss_per_image(probability, mask)
        scores[start:stop] = probability[:, 1].cpu().numpy()
        losses[start:stop] = loss.cpu().numpy().astype(np.float64)
    return scores, losses


def informative_coordinates(utility: np.ndarray, count: int = 3) -> list[tuple[int, int]]:
    utility = np.asarray(utility, dtype=np.float32)
    if utility.ndim != 2:
        raise ValueError("utility must be [images,patches]")
    order = np.argsort(-np.abs(utility), axis=None, kind="stable")[:count]
    return [tuple(int(value) for value in np.unravel_index(int(flat), utility.shape)) for flat in order]


def informative_utility_parity(
    args: argparse.Namespace,
    metadata: dict[str, dict[str, Any]],
    data_root: Path,
) -> list[dict[str, Any]]:
    class_name = EXPECTED_VISA_CLASSES[0]
    shard = load_shard(args.cache_root, class_name)
    masks = load_masks(shard["image_path"], metadata, data_root)
    with np.load(args.output / "utility" / f"{class_name}.npz", allow_pickle=False) as data:
        utility = np.asarray(data["utility"], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    device = torch.device("cuda")
    native = torch.from_numpy(shard["native_logits"][0:1]).permute(1, 0, 2, 3).to(device)
    with torch.no_grad():
        probability, _ = deploy_correction(native, torch.zeros((1, PATCHES), device=device))
    cache_error = float(np.max(np.abs(probability[0, 1].cpu().numpy() - shard["native_pixel_probability"][0])))
    rows.append({"test": "native_zero_delta_cache", "max_abs_error": cache_error, "pass": cache_error <= 2e-6})
    epsilon = 0.1
    for image_index, patch in informative_coordinates(utility, count=3):
        native = torch.from_numpy(shard["native_logits"][image_index:image_index + 1]).permute(1, 0, 2, 3).to(device)
        mask = torch.from_numpy(masks[image_index:image_index + 1, None].astype(np.float32)).to(device)
        values: list[float] = []
        for signed_epsilon in (epsilon, -epsilon):
            correction = torch.zeros((1, PATCHES), device=device)
            correction[0, patch] = signed_epsilon
            with torch.no_grad():
                candidate, _ = deploy_correction(native, correction)
                values.append(float(canonical_loss_per_image(candidate, mask)[0].item()))
        finite_difference = -(values[0] - values[1]) / (2.0 * epsilon)
        analytic = float(utility[image_index, patch])
        absolute_error = abs(analytic - finite_difference)
        tolerance = 2e-5 + 1e-2 * max(abs(analytic), abs(finite_difference))
        sign_match = bool(np.sign(analytic) == np.sign(finite_difference))
        rows.append({
            "test": "informative_utility_finite_difference",
            "class": class_name,
            "image_index": image_index,
            "image_path": str(shard["image_path"][image_index]),
            "patch": patch,
            "epsilon": epsilon,
            "analytic": analytic,
            "finite_difference": finite_difference,
            "absolute_error": absolute_error,
            "tolerance": tolerance,
            "sign_match": sign_match,
            "informative": abs(analytic) > EPSILON,
            "pass": absolute_error <= tolerance and sign_match and abs(analytic) > EPSILON,
        })
    return rows



def run_utilities(args: argparse.Namespace, provenance: dict[str, Any]) -> dict[str, Any]:
    output = args.output / "utility"
    output.mkdir(parents=True, exist_ok=True)
    metadata, data_root = metadata_and_root(args.data_root)
    device = torch.device("cuda")
    parity_rows: list[dict[str, Any]] = []
    total = 0
    started = time.perf_counter()
    for class_name in EXPECTED_VISA_CLASSES:
        destination = output / f"{class_name}.npz"
        shard = load_shard(args.cache_root, class_name)
        paths = shard["image_path"]
        if destination.exists():
            with np.load(destination, allow_pickle=False) as existing:
                if (
                    existing["utility"].shape == (len(paths), PATCHES)
                    and np.array_equal(existing["image_path"].astype(str), paths)
                    and np.isfinite(existing["utility"]).all()
                ):
                    total += len(paths)
                    continue
            raise RuntimeError(f"invalid partial utility artifact: {destination}")
        masks = load_masks(paths, metadata, data_root)
        utilities = np.empty((len(paths), PATCHES), dtype=np.float32)
        native_losses = np.empty(len(paths), dtype=np.float64)
        for start in range(0, len(paths), args.batch_size):
            stop = min(start + args.batch_size, len(paths))
            native = torch.from_numpy(shard["native_logits"][start:stop]).permute(1, 0, 2, 3).to(device)
            mask = torch.from_numpy(masks[start:stop, None].astype(np.float32)).to(device)
            utility, loss = utility_for_batch(native, mask)
            utilities[start:stop] = utility.cpu().numpy()
            native_losses[start:stop] = loss.cpu().numpy().astype(np.float64)
        if class_name == EXPECTED_VISA_CLASSES[0]:
            native = torch.from_numpy(shard["native_logits"][0:1]).permute(1, 0, 2, 3).to(device)
            zero = torch.zeros((1, PATCHES), device=device)
            with torch.no_grad():
                probability, _ = deploy_correction(native, zero)
            error = float(np.max(np.abs(probability[0, 1].cpu().numpy() - shard["native_pixel_probability"][0])))
            parity_rows.append({"test": "native_zero_delta_cache", "max_abs_error": error, "pass": error <= 2e-6})
            mask = torch.from_numpy(masks[0:1, None].astype(np.float32)).to(device)
            for patch in (0, 684, 1368):
                epsilon = 1e-4
                values = []
                for signed_epsilon in (epsilon, -epsilon):
                    correction = torch.zeros((1, PATCHES), device=device)
                    correction[0, patch] = signed_epsilon
                    with torch.no_grad():
                        prob, _ = deploy_correction(native, correction)
                        values.append(float(canonical_loss_per_image(prob, mask)[0].item()))
                finite_difference = -(values[0] - values[1]) / (2.0 * epsilon)
                analytic = float(utilities[0, patch])
                tolerance = 2e-3 + 2e-2 * max(abs(analytic), abs(finite_difference))
                parity_rows.append({
                    "test": "utility_finite_difference",
                    "patch": patch,
                    "analytic": analytic,
                    "finite_difference": finite_difference,
                    "absolute_error": abs(analytic - finite_difference),
                    "tolerance": tolerance,
                    "pass": abs(analytic - finite_difference) <= tolerance,
                })
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, utility=utilities, native_loss=native_losses, image_path=paths)
        temporary.replace(destination)
        total += len(paths)
    parity_rows = informative_utility_parity(args, metadata, data_root)
    summary = {
        "status": "PASS" if all(row["pass"] for row in parity_rows) else "FAIL",
        "records": total,
        "elapsed_seconds": time.perf_counter() - started,
        "parity": parity_rows,
        "provenance_head": provenance["head"],
        "medical_reads": 0,
        "phase2b_training_steps": 0,
    }
    write_json(args.output / "utility_summary.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError("R0 utility parity failed")
    return summary


def condition_rows(args: argparse.Namespace, include_radius: bool) -> list[dict[str, Any]]:
    metadata, data_root = metadata_and_root(args.data_root)
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    alpha_path = args.output / "alpha_per_class.csv"
    radius_only = include_radius and alpha_path.exists()
    if radius_only:
        with alpha_path.open(newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    started = time.perf_counter()
    for class_name in EXPECTED_VISA_CLASSES:
        shard = load_shard(args.cache_root, class_name)
        paths = shard["image_path"]
        masks = load_masks(paths, metadata, data_root)
        with np.load(args.output / "utility" / f"{class_name}.npz", allow_pickle=False) as data:
            utility = np.asarray(data["utility"], dtype=np.float32)
        actions = classify_actions(utility)
        if radius_only:
            with np.load(args.output / "radius" / f"{class_name}.npz", allow_pickle=False) as data:
                conditions = [("signed_radius", np.asarray(data["correction"], dtype=np.float32))]
        else:
            conditions = [("native", np.zeros_like(utility, dtype=np.float32))]
            for alpha in ALPHAS[1:]:
                conditions.append((f"positive_alpha_{alpha:g}", correction_from_actions(actions, alpha, True)))
                conditions.append((f"signed_alpha_{alpha:g}", correction_from_actions(actions, alpha, False)))
            if include_radius:
                with np.load(args.output / "radius" / f"{class_name}.npz", allow_pickle=False) as data:
                    conditions.append(("signed_radius", np.asarray(data["correction"], dtype=np.float32)))
        for condition, correction in conditions:
            scores, losses = evaluate_correction(shard["native_logits"], masks, correction, device, args.batch_size)
            if condition == "native":
                cache_error = float(np.max(np.abs(scores - shard["native_pixel_probability"])))
                if cache_error > 2e-6:
                    raise RuntimeError(f"native cache parity failed for {class_name}: {cache_error}")
            metric = exact_metrics(scores, masks)
            row = {
                "class": class_name,
                "condition": condition,
                "pAP": metric["pAP"],
                "pAUROC": metric["pAUROC"],
                "loss": float(losses.mean()),
                "images": len(paths),
                "pixels": int(masks.size),
            }
            rows.append(row)
            if condition == "native" and not include_radius:
                rows.append({**row, "condition": "positive_alpha_0"})
                rows.append({**row, "condition": "signed_alpha_0"})
            del scores, losses
    output_name = "per_class.csv" if include_radius else "alpha_per_class.csv"
    write_csv(args.output / output_name, rows)
    write_json(args.output / ("all_metrics_runtime.json" if include_radius else "alpha_metrics_runtime.json"), {
        "elapsed_seconds": time.perf_counter() - started,
        "classes": len(EXPECTED_VISA_CLASSES),
        "conditions": sorted({row["condition"] for row in rows}),
        "reused_alpha_rows": radius_only,
        "medical_reads": 0,
    })
    return rows


def macro_rows(per_class: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for condition in sorted({row["condition"] for row in per_class}):
        group = [row for row in per_class if row["condition"] == condition]
        output.append({
            "condition": condition,
            "macro_pAP": float(np.mean([float(row["pAP"]) for row in group])),
            "macro_pAUROC": float(np.mean([float(row["pAUROC"]) for row in group])),
            "mean_loss": float(np.mean([float(row["loss"]) for row in group])),
        })
    return output


def select_signed_alpha(macros: list[dict[str, Any]]) -> float:
    candidates = []
    native = next(row for row in macros if row["condition"] == "native")
    candidates.append((float(native["macro_pAP"]), 0.0))
    for alpha in ALPHAS[1:]:
        row = next(item for item in macros if item["condition"] == f"signed_alpha_{alpha:g}")
        candidates.append((float(row["macro_pAP"]), float(alpha)))
    return min(candidates, key=lambda item: (-item[0], item[1]))[1]


def build_sparse_basis(device: torch.device, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from model.adapter import gaussian_blur2d

    supports: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for start in range(0, PATCHES, batch_size):
        stop = min(start + batch_size, PATCHES)
        impulses = torch.zeros((stop - start, 1, *PATCH_GRID), device=device)
        local = torch.arange(stop - start, device=device)
        flat = impulses.view(stop - start, -1)
        flat[local, torch.arange(start, stop, device=device)] = 1.0
        blurred = gaussian_blur2d(impulses, (7, 7), (1, 1))
        deployed = F.interpolate(blurred, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True)
        arrays = deployed[:, 0].cpu().numpy().astype(np.float32)
        for array in arrays:
            indices = np.flatnonzero(array.reshape(-1) != 0.0).astype(np.int32)
            supports.append(indices)
            values.append(array.reshape(-1)[indices].astype(np.float32))
    maximum = max(len(item) for item in supports)
    index_array = np.zeros((PATCHES, maximum), dtype=np.int32)
    value_array = np.zeros((PATCHES, maximum), dtype=np.float32)
    valid_array = np.zeros((PATCHES, maximum), dtype=bool)
    for patch, (indices, patch_values) in enumerate(zip(supports, values)):
        length = len(indices)
        index_array[patch, :length] = indices
        value_array[patch, :length] = patch_values
        valid_array[patch, :length] = True
    return index_array, value_array, valid_array


def focal_terms(abnormal: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    smooth = 1e-5
    normal = 1.0 - abnormal
    pt = torch.where(
        target > 0.5,
        abnormal * (1.0 - smooth) + normal * smooth + smooth,
        normal * (1.0 - smooth) + abnormal * smooth + smooth,
    )
    return -(1.0 - pt).square() * pt.log()


def coordinate_radius_for_image(
    deployed_margin: torch.Tensor,
    mask: torch.Tensor,
    actions: torch.Tensor,
    basis_indices: torch.Tensor,
    basis_values: torch.Tensor,
    basis_valid: torch.Tensor,
    patch_batch: int,
) -> torch.Tensor:
    margin = deployed_margin.reshape(-1).float()
    target = mask.reshape(-1).float()
    base_probability = torch.sigmoid(margin)
    base_focal = focal_terms(base_probability, target)
    pixel_count = float(target.numel())
    sum_p = base_probability.sum()
    sum_y = target.sum()
    sum_py = (base_probability * target).sum()
    sum_n = (1.0 - base_probability).sum()
    sum_ny = ((1.0 - base_probability) * (1.0 - target)).sum()
    sum_normal_y = (1.0 - target).sum()
    base_loss = (
        base_focal.mean()
        + 1.0 - (2.0 * sum_ny + 1.0) / (sum_n + sum_normal_y + 1.0)
        + 1.0 - (2.0 * sum_py + 1.0) / (sum_p + sum_y + 1.0)
    )
    selected = torch.zeros(PATCHES, device=margin.device, dtype=torch.float32)
    magnitudes = torch.tensor(ALPHAS[1:], device=margin.device, dtype=torch.float32) * MARGIN_SCALE
    for start in range(0, PATCHES, patch_batch):
        stop = min(start + patch_batch, PATCHES)
        action = actions[start:stop].float()
        indices = basis_indices[start:stop]
        values = basis_values[start:stop]
        valid = basis_valid[start:stop]
        local_margin = margin[indices]
        local_target = target[indices]
        base_local_p = base_probability[indices]
        base_local_focal = base_focal[indices]
        shifts = action[:, None, None] * magnitudes[None, :, None] * values[:, None, :]
        candidate_p = torch.sigmoid(local_margin[:, None, :] + shifts)
        valid_f = valid[:, None, :].float()
        delta_p = ((candidate_p - base_local_p[:, None, :]) * valid_f)
        delta_focal = ((focal_terms(candidate_p, local_target[:, None, :]) - base_local_focal[:, None, :]) * valid_f).sum(2)
        delta_sum_p = delta_p.sum(2)
        delta_sum_py = (delta_p * local_target[:, None, :] * valid_f).sum(2)
        delta_sum_ny = (-delta_p * (1.0 - local_target[:, None, :]) * valid_f).sum(2)
        candidate_loss = (
            base_focal.sum() / pixel_count
            + delta_focal / pixel_count
            + 1.0 - (2.0 * (sum_ny + delta_sum_ny) + 1.0) / (sum_n - delta_sum_p + sum_normal_y + 1.0)
            + 1.0 - (2.0 * (sum_py + delta_sum_py) + 1.0) / (sum_p + delta_sum_p + sum_y + 1.0)
        )
        loss_with_zero = torch.cat([base_loss.expand(stop - start, 1), candidate_loss], dim=1)
        choice = torch.argmin(loss_with_zero, dim=1)
        selected[start:stop] = torch.tensor(ALPHAS, device=margin.device)[choice] * action
    return selected * MARGIN_SCALE


def load_or_build_basis(args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis_path = args.output / "sparse_deployment_basis.npz"
    if basis_path.exists():
        with np.load(basis_path, allow_pickle=False) as data:
            return (
                np.asarray(data["indices"], dtype=np.int32),
                np.asarray(data["values"], dtype=np.float32),
                np.asarray(data["valid"], dtype=bool),
            )
    indices, values, valid = build_sparse_basis(device)
    temporary = basis_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, indices=indices, values=values, valid=valid)
    temporary.replace(basis_path)
    return indices, values, valid


def direct_coordinate_choice(
    native: torch.Tensor,
    mask: torch.Tensor,
    patch: int,
    action: int,
) -> tuple[float, list[float]]:
    losses: list[float] = []
    with torch.no_grad():
        for alpha in ALPHAS:
            correction = torch.zeros((1, PATCHES), device=native.device)
            correction[0, patch] = float(action) * float(alpha) * MARGIN_SCALE
            probability, _ = deploy_correction(native, correction)
            losses.append(float(canonical_loss_per_image(probability, mask)[0].item()))
    choice = min(range(len(ALPHAS)), key=lambda index: (losses[index], ALPHAS[index]))
    return float(action) * float(ALPHAS[choice]) * MARGIN_SCALE, losses


def run_radius_probe(args: argparse.Namespace) -> dict[str, Any]:
    metadata, data_root = metadata_and_root(args.data_root)
    device = torch.device("cuda")
    basis_indices_np, basis_values_np, basis_valid_np = load_or_build_basis(args, device)
    basis_indices = torch.from_numpy(basis_indices_np).to(device=device, dtype=torch.long)
    basis_values = torch.from_numpy(basis_values_np).to(device)
    basis_valid = torch.from_numpy(basis_valid_np).to(device)
    class_name = EXPECTED_VISA_CLASSES[0]
    shard = load_shard(args.cache_root, class_name)
    masks = load_masks(shard["image_path"][0:1], metadata, data_root)
    with np.load(args.output / "utility" / f"{class_name}.npz", allow_pickle=False) as data:
        actions_np = classify_actions(np.asarray(data["utility"][0], dtype=np.float32))
    native = torch.from_numpy(shard["native_logits"][0:1]).permute(1, 0, 2, 3).to(device)
    mask = torch.from_numpy(masks[:, None].astype(np.float32)).to(device)
    started = time.perf_counter()
    with torch.no_grad():
        _, base_logits = deploy_correction(native, torch.zeros((1, PATCHES), device=device))
        deployed_margin = base_logits[0, 1] - base_logits[0, 0]
        selected = coordinate_radius_for_image(
            deployed_margin,
            mask[0, 0],
            torch.from_numpy(actions_np).to(device),
            basis_indices,
            basis_values,
            basis_valid,
            args.radius_patch_batch,
        )
    elapsed = time.perf_counter() - started
    nonkeep = np.flatnonzero(actions_np != 0)
    if nonkeep.size < 3:
        raise RuntimeError("radius parity requires at least three non-KEEP actions")
    probe_offsets = np.linspace(0, nonkeep.size - 1, 3, dtype=int)
    patches = [int(nonkeep[offset]) for offset in probe_offsets]
    rows: list[dict[str, Any]] = []
    for patch in patches:
        direct, losses = direct_coordinate_choice(native, mask, patch, int(actions_np[patch]))
        correction = torch.zeros((1, PATCHES), device=device)
        correction[0, patch] = 1.0
        with torch.no_grad():
            _, plus_logits = deploy_correction(native, correction)
        direct_basis = (plus_logits[0, 1] - base_logits[0, 1]).reshape(-1)
        sparse_basis = torch.zeros_like(direct_basis)
        valid = basis_valid[patch]
        sparse_basis[basis_indices[patch, valid]] = basis_values[patch, valid]
        basis_error = float((direct_basis - sparse_basis).abs().max().item())
        observed = float(selected[patch].item())
        rows.append({
            "patch": patch,
            "action": int(actions_np[patch]),
            "direct_correction": direct,
            "sparse_correction": observed,
            "correction_abs_error": abs(direct - observed),
            "basis_max_abs_error": basis_error,
            "direct_losses": losses,
            "pass": abs(direct - observed) <= 1e-6 and basis_error <= 2e-6,
        })
    summary = {
        "status": "PASS" if all(row["pass"] for row in rows) else "FAIL",
        "class": class_name,
        "image_path": str(shard["image_path"][0]),
        "coordinate_seconds_per_image": elapsed,
        "estimated_full_coordinate_minutes": elapsed * 2162.0 / 60.0,
        "basis_max_support": int(basis_indices_np.shape[1]),
        "rows": rows,
        "medical_reads": 0,
    }
    write_json(args.output / "radius_probe.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError("real sparse-radius parity failed")
    return summary



def run_radius(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output / "radius"
    output.mkdir(parents=True, exist_ok=True)
    metadata, data_root = metadata_and_root(args.data_root)
    device = torch.device("cuda")
    basis_path = args.output / "sparse_deployment_basis.npz"
    if basis_path.exists():
        with np.load(basis_path, allow_pickle=False) as data:
            basis_indices_np = np.asarray(data["indices"], dtype=np.int32)
            basis_values_np = np.asarray(data["values"], dtype=np.float32)
            basis_valid_np = np.asarray(data["valid"], dtype=bool)
    else:
        basis_indices_np, basis_values_np, basis_valid_np = build_sparse_basis(device)
        temporary = basis_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, indices=basis_indices_np, values=basis_values_np, valid=basis_valid_np)
        temporary.replace(basis_path)
    basis_indices = torch.from_numpy(basis_indices_np).to(device=device, dtype=torch.long)
    basis_values = torch.from_numpy(basis_values_np).to(device)
    basis_valid = torch.from_numpy(basis_valid_np).to(device)
    total = 0
    started = time.perf_counter()
    for class_name in EXPECTED_VISA_CLASSES:
        destination = output / f"{class_name}.npz"
        shard = load_shard(args.cache_root, class_name)
        paths = shard["image_path"]
        if destination.exists():
            with np.load(destination, allow_pickle=False) as existing:
                if existing["correction"].shape == (len(paths), PATCHES) and np.array_equal(existing["image_path"].astype(str), paths):
                    total += len(paths)
                    continue
            raise RuntimeError(f"invalid partial radius artifact: {destination}")
        masks = load_masks(paths, metadata, data_root)
        with np.load(args.output / "utility" / f"{class_name}.npz", allow_pickle=False) as data:
            actions = classify_actions(np.asarray(data["utility"], dtype=np.float32))
        corrections = np.zeros((len(paths), PATCHES), dtype=np.float32)
        for index in range(len(paths)):
            native = torch.from_numpy(shard["native_logits"][index:index + 1]).permute(1, 0, 2, 3).to(device)
            with torch.no_grad():
                _, deployed_logits = deploy_correction(native, torch.zeros((1, PATCHES), device=device))
                margin = deployed_logits[0, 1] - deployed_logits[0, 0]
                correction = coordinate_radius_for_image(
                    margin,
                    torch.from_numpy(masks[index]).to(device),
                    torch.from_numpy(actions[index]).to(device),
                    basis_indices,
                    basis_values,
                    basis_valid,
                    args.radius_patch_batch,
                )
            corrections[index] = correction.cpu().numpy()
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, correction=corrections, image_path=paths)
        temporary.replace(destination)
        total += len(paths)
    summary = {
        "status": "PASS",
        "records": total,
        "elapsed_seconds": time.perf_counter() - started,
        "basis_max_support": int(basis_indices_np.shape[1]),
        "medical_reads": 0,
    }
    write_json(args.output / "radius_summary.json", summary)
    return summary


def per_image_quadrants(args: argparse.Namespace, alpha: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    metadata, data_root = metadata_and_root(args.data_root)
    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    counts = {
        "loss_down_ap_up": 0,
        "loss_down_ap_down": 0,
        "loss_up_ap_up": 0,
        "loss_up_ap_down": 0,
        "loss_tie": 0,
        "ap_tie": 0,
    }
    for class_name in EXPECTED_VISA_CLASSES:
        shard = load_shard(args.cache_root, class_name)
        paths = shard["image_path"]
        masks = load_masks(paths, metadata, data_root)
        with np.load(args.output / "utility" / f"{class_name}.npz", allow_pickle=False) as data:
            actions = classify_actions(np.asarray(data["utility"], dtype=np.float32))
            native_loss = np.asarray(data["native_loss"], dtype=np.float64)
        correction = correction_from_actions(actions, alpha, False)
        scores, signed_loss = evaluate_correction(shard["native_logits"], masks, correction, device, args.batch_size)
        for index, path in enumerate(paths):
            labels = masks[index].reshape(-1)
            if labels.min() == labels.max():
                continue
            native_ap = exact_average_precision(shard["native_pixel_probability"][index].reshape(-1), labels)
            signed_ap = exact_average_precision(scores[index].reshape(-1), labels)
            loss_delta = float(signed_loss[index] - native_loss[index])
            ap_delta = float(signed_ap - native_ap)
            loss_direction = 0 if abs(loss_delta) <= 1e-12 else (-1 if loss_delta < 0 else 1)
            ap_direction = 0 if abs(ap_delta) <= 1e-12 else (1 if ap_delta > 0 else -1)
            if loss_direction == 0:
                counts["loss_tie"] += 1
            if ap_direction == 0:
                counts["ap_tie"] += 1
            if loss_direction and ap_direction:
                key = ("loss_down" if loss_direction < 0 else "loss_up") + ("_ap_up" if ap_direction > 0 else "_ap_down")
                counts[key] += 1
            rows.append({
                "class": class_name,
                "image_path": path,
                "native_loss": float(native_loss[index]),
                "signed_loss": float(signed_loss[index]),
                "loss_delta": loss_delta,
                "native_pAP": native_ap,
                "signed_pAP": signed_ap,
                "pAP_delta": ap_delta,
            })
    write_csv(args.output / "per_image_quadrants.csv", rows)
    return rows, counts


def decide(per_class: list[dict[str, Any]], macros: list[dict[str, Any]], alpha: float, utility_summary: dict[str, Any], quadrant_counts: dict[str, int]) -> dict[str, Any]:
    signed_name = "native" if alpha == 0.0 else f"signed_alpha_{alpha:g}"
    positive_name = "native" if alpha == 0.0 else f"positive_alpha_{alpha:g}"
    native_macro = next(row for row in macros if row["condition"] == "native")
    signed_macro = next(row for row in macros if row["condition"] == signed_name)
    positive_macro = next(row for row in macros if row["condition"] == positive_name)
    by_condition = {(row["class"], row["condition"]): row for row in per_class}
    breadth_positive = sum(
        float(by_condition[(class_name, signed_name)]["pAP"]) > float(by_condition[(class_name, positive_name)]["pAP"])
        for class_name in EXPECTED_VISA_CLASSES
    )
    breadth_native = sum(
        float(by_condition[(class_name, signed_name)]["pAP"]) > float(by_condition[(class_name, "native")]["pAP"])
        for class_name in EXPECTED_VISA_CLASSES
    )
    g0 = utility_summary["status"] == "PASS"
    g1_delta = 100.0 * (float(signed_macro["macro_pAP"]) - float(positive_macro["macro_pAP"]))
    g3_delta = 100.0 * (float(signed_macro["macro_pAUROC"]) - float(positive_macro["macro_pAUROC"]))
    gates = {
        "G0_correctness": {"threshold": "all PASS", "observed": utility_summary["status"], "pass": g0},
        "G1_direction_pp": {"threshold": ">=1.0", "observed": g1_delta, "pass": g1_delta >= 1.0},
        "G2_breadth_classes": {"threshold": ">=8", "observed": breadth_positive, "pass": breadth_positive >= 8},
        "G3_pAUROC_safety_pp": {"threshold": ">=-0.5", "observed": g3_delta, "pass": g3_delta >= -0.5},
    }
    r0_pass = all(item["pass"] for item in gates.values())
    eligible = sum(quadrant_counts[key] for key in ("loss_down_ap_up", "loss_down_ap_down", "loss_up_ap_up", "loss_up_ap_down"))
    discordant = quadrant_counts["loss_down_ap_down"] + quadrant_counts["loss_up_ap_up"]
    disagreement = float(discordant / eligible) if eligible else None
    signed_native_pap = 100.0 * (float(signed_macro["macro_pAP"]) - float(native_macro["macro_pAP"]))
    signed_native_pauroc = 100.0 * (float(signed_macro["macro_pAUROC"]) - float(native_macro["macro_pAUROC"]))
    r0b = (
        not r0_pass
        and signed_native_pap >= 1.0
        and breadth_native >= 8
        and signed_native_pauroc >= -0.5
        and disagreement is not None
        and disagreement >= 0.5
    )
    decision = "CONTINUE" if r0_pass else ("FALLBACK" if r0b else "STOP")
    return {
        "stage": "R0",
        "selected_alpha": alpha,
        "signed_condition": signed_name,
        "matched_positive_condition": positive_name,
        "gates": gates,
        "breadth_vs_native": breadth_native,
        "signed_vs_native_pAP_pp": signed_native_pap,
        "signed_vs_native_pAUROC_pp": signed_native_pauroc,
        "quadrant_counts": quadrant_counts,
        "loss_ap_disagreement_fraction": disagreement,
        "decision": decision,
        "next_stage": "R1" if r0_pass else ("R0B" if r0b else "FINAL_DECISION"),
        "medical_reads": 0,
        "phase2b_training_steps": 0,
    }


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    rows = condition_rows(args, include_radius=True)
    macros = macro_rows(rows)
    alpha = select_signed_alpha(macros)
    _, quadrant_counts = per_image_quadrants(args, alpha)
    utility_summary = json.loads((args.output / "utility_summary.json").read_text())
    actions = []
    for class_name in EXPECTED_VISA_CLASSES:
        with np.load(args.output / "utility" / f"{class_name}.npz", allow_pickle=False) as data:
            actions.append(classify_actions(np.asarray(data["utility"], dtype=np.float32)).reshape(-1))
    flat_actions = np.concatenate(actions)
    nonkeep = int(np.count_nonzero(flat_actions))
    action_rates = {
        "BOOST_rate": float(np.mean(flat_actions > 0)),
        "KEEP_rate": float(np.mean(flat_actions == 0)),
        "SUPPRESS_rate": float(np.mean(flat_actions < 0)),
        "sign_reversal_rate": float(np.count_nonzero(flat_actions < 0) / nonkeep) if nonkeep else None,
    }
    decision = decide(rows, macros, alpha, utility_summary, quadrant_counts)
    summary = {
        "status": "COMPLETE",
        "macro": macros,
        "action_rates": action_rates,
        "decision": decision,
        "medical_access": false,
        "training_steps": 0,
    }
    write_csv(args.output / "summary.csv", macros)
    write_json(args.output / "summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--phase", choices=("utilities", "alpha", "probe-radius", "radius", "finalize", "all"), default="all")
    result.add_argument("--cache-root", type=Path, default=Path("/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0/sabra_source"))
    result.add_argument("--checkpoint", type=Path, default=Path("/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0/phase2b/checkpoints/adapter_10.pth"))
    result.add_argument("--clip-asset", type=Path, default=Path("/home/ai4/.cache/clip/ViT-L-14-336px.pt"))
    result.add_argument("--data-root", type=Path, default=Path("/home/ai4/caohuy/data"))
    result.add_argument("--output", type=Path, default=ROOT / "results/sabra_car/r0")
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--radius-patch-batch", type=int, default=64)
    return result


def main() -> None:
    args = parser().parse_args()
    args.output = args.output.resolve()
    configure_runtime()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R0")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = validate_cache(args.cache_root, args.checkpoint, args.clip_asset)
    write_json(args.output / "provenance.json", provenance)
    if args.phase in ("utilities", "all"):
        run_utilities(args, provenance)
    if args.phase in ("alpha", "all"):
        rows = condition_rows(args, include_radius=False)
        macros = macro_rows(rows)
        write_csv(args.output / "alpha_summary.csv", macros)
        write_json(args.output / "alpha_selection.json", {"selected_alpha": select_signed_alpha(macros), "macro": macros})
    if args.phase == "probe-radius":
        print(json.dumps(run_radius_probe(args), indent=2, sort_keys=True))
    if args.phase in ("radius", "all"):
        run_radius(args)
    if args.phase in ("finalize", "all"):
        summary = run_finalize(args)
        print(json.dumps(summary["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
