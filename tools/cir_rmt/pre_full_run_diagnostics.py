#!/usr/bin/env python3
"""Bounded, source-only diagnostics for the corrected CIR-V2 checkpoints.

This module deliberately does not train, modify checkpoints, access Medical or
MVTec data, or change the production forward path.  It compares the frozen
Phase2B parent (P) and corrected CIR (C0/C05) checkpoints on a deterministic
small VisA sample and emits compact CSV evidence for the pre-full-run root
cause lock.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from evaluation.evaluator import image_score
from evaluation.metrics import binary_metrics
from model.phase2b_runtime import (
    build_phase2b_frozen,
    configure_canonical_fp32,
    deploy_native_logits,
    forward_phase2b,
)
from scripts.cir_rmt.eval_full import ManifestDataset
from tools.cir_rmt.identity import config_sha256, load_cir_config
from tools.cir_rmt.runtime import forward_cir


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = (10, 12, 14, 16, 18, 20)
IMAGE_SIZE = 518
PATCH_GRID = (37, 37)
DEFAULT_HOLDOUT = ("cashew", "macaroni2", "pcb3", "pipe_fryum")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _f(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _finite_mean(values: Iterable[float]) -> float | None:
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    return _f(a.mean()) if a.size else None


def _metric(scores: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
    auroc, ap = binary_metrics(
        np.asarray(scores, dtype=np.float64).reshape(-1),
        np.asarray(labels, dtype=np.int64).reshape(-1),
        allow_undefined=True,
    )
    return _f(auroc), _f(ap)


def _metrics_for(
    pixel_maps: np.ndarray,
    masks: np.ndarray,
    image_scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | None]:
    p_auc, p_ap = _metric(pixel_maps, masks)
    i_auc, i_ap = _metric(image_scores, labels)
    return {
        "pixel_auroc": p_auc,
        "pixel_ap": p_ap,
        "image_auroc": i_auc,
        "image_ap": i_ap,
    }


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return None
    x, y = x[keep], y[keep]
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return _f(np.corrcoef(x, y)[0, 1])


def _cosine_rows(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    numerator = np.sum(x * y, axis=-1)
    denominator = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return numerator / np.maximum(denominator, eps)


def _mean_cosine(x: np.ndarray, y: np.ndarray) -> float | None:
    return _finite_mean(_cosine_rows(x, y).reshape(-1))


def _norm_ratio(x: np.ndarray, y: np.ndarray) -> float | None:
    nx = np.linalg.norm(np.asarray(x, dtype=np.float64), axis=-1).reshape(-1)
    ny = np.linalg.norm(np.asarray(y, dtype=np.float64), axis=-1).reshape(-1)
    keep = np.isfinite(nx) & np.isfinite(ny) & (nx > 1e-12)
    return _f(np.mean(ny[keep] / nx[keep])) if keep.any() else None


def _linear_cka(x: np.ndarray, y: np.ndarray, max_rows: int = 2048, seed: int = 0) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)
    y = np.asarray(y, dtype=np.float64).reshape(y.shape[0], -1)
    n = min(x.shape[0], y.shape[0])
    if n < 3:
        return None
    if n > max_rows:
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(n, size=max_rows, replace=False))
        x, y = x[take], y[take]
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denom = np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro")
    return _f(float(np.sum(cross * cross) / denom)) if denom > 1e-18 else None


def _pairwise_geometry_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)
    y = np.asarray(y, dtype=np.float64).reshape(y.shape[0], -1)
    n = min(x.shape[0], y.shape[0])
    if n < 4:
        return None
    x = x[:n] / np.maximum(np.linalg.norm(x[:n], axis=1, keepdims=True), 1e-12)
    y = y[:n] / np.maximum(np.linalg.norm(y[:n], axis=1, keepdims=True), 1e-12)
    gx = x @ x.T
    gy = y @ y.T
    upper = np.triu_indices(n, k=1)
    return _corr(gx[upper], gy[upper])


def _level(cosine: float | None, cka: float | None) -> str:
    if cosine is None and cka is None:
        return "UNKNOWN"
    c = cosine if cosine is not None else cka
    if c >= 0.995:
        return "LOW"
    if c >= 0.95:
        return "MODERATE"
    return "HIGH"


def _training_probability(logits: torch.Tensor, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    stages, batch, patches, classes = logits.shape
    grid = int(round(patches ** 0.5))
    maps = logits.permute(0, 1, 3, 2).reshape(stages * batch, classes, grid, grid)
    maps = F.interpolate(maps, size=image_size, mode="bilinear", align_corners=True)
    maps = maps.reshape(stages, batch, classes, image_size, image_size).mean(dim=0)
    return F.softmax(maps, dim=1)


def _stage_probability(logits: torch.Tensor) -> torch.Tensor:
    stages, batch, patches, classes = logits.shape
    grid = int(round(patches ** 0.5))
    prob = F.softmax(logits.float(), dim=-1)[..., 1]
    prob = prob.reshape(stages * batch, 1, grid, grid)
    prob = F.interpolate(prob, size=IMAGE_SIZE, mode="bilinear", align_corners=True)
    return prob.reshape(stages, batch, IMAGE_SIZE, IMAGE_SIZE)


def _sample_indices(metadata: Path, per_category: int, seed: int) -> tuple[list[int], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_class: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for i, row in enumerate(rows):
        by_class.setdefault(str(row["class_name"]), []).append((i, row))
    selected: list[int] = []
    selected_rows: list[dict[str, Any]] = []
    for class_name in sorted(by_class):
        rng = random.Random(seed + sum(ord(c) for c in class_name))
        normal = [(i, r) for i, r in by_class[class_name] if int(r.get("label", 0)) == 0]
        anomaly = [(i, r) for i, r in by_class[class_name] if int(r.get("label", 0)) == 1]
        each = max(1, per_category // 2)
        rng.shuffle(normal)
        rng.shuffle(anomaly)
        chosen = sorted(normal[:each] + anomaly[:each], key=lambda item: str(item[1]["image_path"]))
        for i, row in chosen:
            selected.append(i)
            selected_rows.append(dict(row, manifest_index=i))
    order = np.argsort([str(r["image_path"]) for r in selected_rows], kind="stable")
    return [selected[int(i)] for i in order], [selected_rows[int(i)] for i in order]


def _checkpoint_paths(parent_root: Path, cir_root: Path, epoch: int) -> tuple[Path, Path]:
    return (
        parent_root / "phase2b" / "checkpoints" / f"adapter_{epoch}.pth",
        cir_root / "visa" / "seed0" / "checkpoints" / f"epoch_{epoch:02d}.pth",
    )


def _load_payload(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _make_model(
    parent_config: Mapping[str, Any],
    payload: Mapping[str, Any],
    clip_asset: Path,
    device: torch.device,
) -> Any:
    configure_canonical_fp32()
    model = build_phase2b_frozen(parent_config, payload, clip_asset, device)
    model.eval()
    return model


def _apply_modules(model: Any, payload: Mapping[str, Any], flags: str) -> None:
    if len(flags) != 3 or any(c not in "PC" for c in flags):
        raise ValueError(f"module flags must be PPP/CCC-style three-character value, got {flags}")
    source = payload
    model.image_adapter.load_state_dict(source["image_adapter"])
    model.text_adapter.load_state_dict(source["text_adapter"])
    model.soft_prompt.load_state_dict(source["soft_prompt"])


def _capture(output: Any, native_prob: torch.Tensor, transported_prob: torch.Tensor) -> dict[str, np.ndarray]:
    seg = output.seg_features.detach().float().cpu()
    # Fixed spatial subsample keeps the diagnostic compact while retaining the
    # actual feature space and identical coordinates for P/C comparisons.
    patch_indices = torch.linspace(0, seg.shape[2] - 1, 32).long()
    group_margins = torch.einsum(
        "sbpd,bgdc->sbpgc",
        F.normalize(seg, dim=-1),
        F.normalize(output.text_features.detach().float().cpu(), dim=-2),
    )
    group_margins = group_margins[..., 1] - group_margins[..., 0]
    return {
        "p0": native_prob[:, 1].detach().float().cpu().numpy(),
        "p05": transported_prob.detach().float().cpu().numpy(),
        "raw": _training_probability(output.native_logits.detach().float()).detach().cpu().numpy()[:, 1],
        "native_logits": output.native_logits.detach().float().cpu().numpy(),
        "cir_logits": output.cir_logits.detach().float().cpu().numpy(),
        "seg_pooled": seg.mean(dim=2).numpy(),
        "seg_patch": seg[:, :, patch_indices, :].numpy(),
        "det": output.det_features.detach().float().cpu().numpy(),
        "text": output.text_features.detach().float().cpu().numpy(),
        "native_weights": output.native_weights.detach().float().cpu().numpy(),
        "group_margins": group_margins.numpy(),
        "native_margin": output.native_margin.detach().float().cpu().numpy(),
        "cir_margin": output.cir_margin.detach().float().cpu().numpy(),
        "delta": output.delta.detach().float().cpu().numpy(),
        "classification_probability": output.classification_probability.detach().float().cpu().numpy(),
    }


def _concat(captures: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    batch_axis_keys = {"native_logits", "cir_logits", "seg_pooled", "seg_patch", "det", "native_weights", "group_margins", "native_margin", "cir_margin", "delta"}
    result: dict[str, np.ndarray] = {}
    for key in captures[0]:
        axis = 1 if key in batch_axis_keys else 0
        result[key] = np.concatenate([capture[key] for capture in captures], axis=axis)
    return result


def _paired_epoch(
    *,
    epoch: int,
    parent_config: Mapping[str, Any],
    cir_config: Mapping[str, Any],
    parent_path: Path,
    cir_path: Path,
    clip_asset: Path,
    dataset: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    p_payload = _load_payload(parent_path)
    c_payload = _load_payload(cir_path)
    model = _make_model(parent_config, p_payload, clip_asset, device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    p_caps: list[dict[str, np.ndarray]] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    class_names: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device).float()
            names = [str(v) for v in batch["class_name"]]
            output = forward_cir(model, image, names, device, cir_config, domain="Industrial", dataset_name="VisA")
            native_prob, _ = deploy_native_logits(output.native_logits, image_size=IMAGE_SIZE, domain="Industrial")
            p_caps.append(_capture(output, native_prob, output.cir_segmentation_probability))
            labels.append(batch["label"].numpy().astype(np.int64))
            masks.append(batch["mask"].numpy().astype(np.float32)[:, 0])
            class_names.extend(names)
            del output, image
    _apply_modules(model, c_payload, "CCC")
    c_caps: list[dict[str, np.ndarray]] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device).float()
            names = [str(v) for v in batch["class_name"]]
            output = forward_cir(model, image, names, device, cir_config, domain="Industrial", dataset_name="VisA")
            native_prob, _ = deploy_native_logits(output.native_logits, image_size=IMAGE_SIZE, domain="Industrial")
            c_caps.append(_capture(output, native_prob, output.cir_segmentation_probability))
            del output, image
    # The loader has no shuffle and the held-out sample is immutable; the two
    # capture sequences therefore correspond one-to-one by batch and image.
    p = _concat(p_caps)
    c = _concat(c_caps)
    p["labels"] = np.concatenate(labels)
    p["masks"] = np.concatenate(masks)
    p["class_names"] = np.asarray(class_names, dtype=object)
    c["labels"] = p["labels"]
    c["masks"] = p["masks"]
    c["class_names"] = p["class_names"]
    del model, loader, p_payload, c_payload, p_caps, c_caps
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return p, c, {"parent_checkpoint_sha256": _sha256(parent_path), "cir_checkpoint_sha256": _sha256(cir_path)}


def _method_image_scores(pixel_maps: np.ndarray, cls: np.ndarray) -> np.ndarray:
    return np.asarray([image_score(float(c), float(m.max()), "Industrial") for c, m in zip(cls, pixel_maps)], dtype=np.float64)


def _feature_rows(epoch: int, p: Mapping[str, np.ndarray], c: Mapping[str, np.ndarray], n_images: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    def add(signal: str, axis: str, x: np.ndarray, y: np.ndarray) -> None:
        x2 = np.asarray(x)
        y2 = np.asarray(y)
        cosine = _mean_cosine(x2.reshape(-1, x2.shape[-1]), y2.reshape(-1, y2.shape[-1]))
        norm = _norm_ratio(x2.reshape(-1, x2.shape[-1]), y2.reshape(-1, y2.shape[-1]))
        cka = _linear_cka(x2.reshape(-1, x2.shape[-1]), y2.reshape(-1, y2.shape[-1]), seed=epoch)
        geometry = _pairwise_geometry_corr(x2.reshape(x2.shape[0], -1), y2.reshape(y2.shape[0], -1))
        rows.append({
            "epoch": epoch, "signal": signal, "axis": axis,
            "mean_cosine": cosine, "norm_ratio_c": norm, "linear_cka": cka,
            "pairwise_geometry_corr": geometry,
            "mean_abs_delta": _f(np.mean(np.abs(x2.astype(np.float64) - y2.astype(np.float64)))),
            "drift_level": _level(cosine, cka), "n_images": n_images,
        })
    for stage in range(3):
        add(f"seg_stage{stage}", "feature", p["seg_pooled"][stage], c["seg_pooled"][stage])
        add(f"seg_stage{stage}", "patch_subsample", p["seg_patch"][stage], c["seg_patch"][stage])
        add(f"det_stage{stage}", "feature", p["det"][stage], c["det"][stage])
        add(f"native_dfg_weights_stage{stage}", "group_class", p["native_weights"][stage], c["native_weights"][stage])
        add(f"group_margin_stage{stage}", "patch_group", p["group_margins"][stage], c["group_margins"][stage])
        add(f"native_fused_margin_stage{stage}", "patch", p["native_margin"][stage], c["native_margin"][stage])
        add(f"transported_margin_stage{stage}", "patch", p["cir_margin"][stage], c["cir_margin"][stage])
    add("text_normal_abnormal_groups", "feature", p["text"].transpose(0, 1, 3, 2), c["text"].transpose(0, 1, 3, 2))
    return rows


def _metric_rows(epoch: int, method: str, data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    maps = data["p0"] if method.endswith("0") else data["p05"]
    cls = np.asarray(data.get("classification_probability", np.zeros(len(maps))), dtype=np.float64)
    scores = _method_image_scores(maps, cls)
    metrics = _metrics_for(maps, data["masks"], scores, data["labels"])
    return {"epoch": epoch, "method": method, "n_images": len(maps), **metrics}


def _append_method_metric_rows(rows: list[dict[str, Any]], epoch: int, method: str, maps: np.ndarray, masks: np.ndarray, cls: np.ndarray, labels: np.ndarray) -> None:
    scores = _method_image_scores(maps, cls)
    rows.append({"epoch": epoch, "method": method, "n_images": len(maps), **_metrics_for(maps, masks, scores, labels)})


def _tail_rows(epoch: int, method: str, maps: np.ndarray, masks: np.ndarray) -> list[dict[str, Any]]:
    normal = maps[masks.sum(axis=(1, 2)) == 0].reshape(-1).astype(np.float64)
    anomaly = maps[masks.sum(axis=(1, 2)) > 0].reshape(-1).astype(np.float64)
    rows: list[dict[str, Any]] = []
    for cohort, values in (("normal", normal), ("anomaly", anomaly)):
        for q in (0.10, 0.50, 0.90, 0.95, 0.99, 0.995, 0.999):
            if values.size:
                rows.append({"epoch": epoch, "method": method, "cohort": cohort, "stat": f"p{q*100:g}", "value": _f(np.quantile(values, q)), "n": int(values.size)})
    flat_scores = maps.reshape(-1)
    flat_labels = masks.reshape(-1).astype(np.int64)
    order = np.argsort(-flat_scores, kind="stable")
    for fraction in (0.001, 0.005, 0.01, 0.02, 0.05):
        k = max(1, int(round(flat_scores.size * fraction)))
        rows.append({"epoch": epoch, "method": method, "cohort": "all", "stat": f"top_precision_{fraction:g}", "value": _f(float(flat_labels[order[:k]].mean())), "n": k})
    return rows


def _deployment_rows(epoch: int, method: str, data: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    raw = data["raw"]
    deployed = data["p0"]
    masks = data["masks"]
    labels = data["labels"]
    cls = np.asarray(data.get("classification_probability", np.zeros(len(raw))), dtype=np.float64)
    raw_scores = _method_image_scores(raw, cls)
    dep_scores = _method_image_scores(deployed, cls)
    raw_metric = _metrics_for(raw, masks, raw_scores, labels)
    dep_metric = _metrics_for(deployed, masks, dep_scores, labels)
    raw_flat = raw.reshape(-1)
    dep_flat = deployed.reshape(-1)
    take = np.linspace(0, len(raw_flat) - 1, min(100000, len(raw_flat)), dtype=np.int64)
    rows = [
        {"epoch": epoch, "method": method, "metric": "raw_pixel_auroc", "value": raw_metric["pixel_auroc"]},
        {"epoch": epoch, "method": method, "metric": "raw_pixel_ap", "value": raw_metric["pixel_ap"]},
        {"epoch": epoch, "method": method, "metric": "deployed_pixel_auroc", "value": dep_metric["pixel_auroc"]},
        {"epoch": epoch, "method": method, "metric": "deployed_pixel_ap", "value": dep_metric["pixel_ap"]},
        {"epoch": epoch, "method": method, "metric": "pearson_sample", "value": _corr(raw_flat[take], dep_flat[take])},
        {"epoch": epoch, "method": method, "metric": "mae", "value": _f(np.mean(np.abs(raw_flat - dep_flat)))},
        {"epoch": epoch, "method": method, "metric": "max_abs_change", "value": _f(np.max(np.abs(raw_flat - dep_flat)))},
        {"epoch": epoch, "method": method, "metric": "deployed_minus_raw_pixel_ap", "value": None if dep_metric["pixel_ap"] is None or raw_metric["pixel_ap"] is None else dep_metric["pixel_ap"] - raw_metric["pixel_ap"]},
    ]
    for cohort, selector in (("normal", masks.sum(axis=(1, 2)) == 0), ("anomaly", masks.sum(axis=(1, 2)) > 0)):
        vals_raw = raw[selector].reshape(-1)
        vals_dep = deployed[selector].reshape(-1)
        for q in (0.99, 0.999) if cohort == "normal" else (0.50, 0.90):
            rows.append({"epoch": epoch, "method": method, "metric": f"{cohort}_raw_p{q*100:g}", "value": _f(np.quantile(vals_raw, q))})
            rows.append({"epoch": epoch, "method": method, "metric": f"{cohort}_deployed_p{q*100:g}", "value": _f(np.quantile(vals_dep, q))})
    return rows


def _branch_rows(epoch: int, method: str, data: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    maps = data["p0"]
    cls = np.asarray(data["classification_probability"], dtype=np.float64)
    labels = data["labels"]
    masks = data["masks"]
    pixel_max = maps.reshape(len(maps), -1).max(axis=1)
    branch_scores = {"classification_only": cls, "pixel_max_only": pixel_max, "fused": 0.9 * cls + 0.1 * pixel_max}
    rows = []
    for branch, scores in branch_scores.items():
        _, ap = _metric(scores, labels)
        auc, _ = _metric(scores, labels)
        rows.append({"epoch": epoch, "method": method, "branch": branch, "image_auroc": auc, "image_ap": ap, "mean_score": _f(scores.mean()), "n_images": len(scores)})
    return rows


def _compensation_rows(epoch: int, p: Mapping[str, np.ndarray], c: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, data, native, transported in (("P", p, p["p0"], p["p05"]), ("C", c, c["p0"], c["p05"])):
        d = transported.astype(np.float64) - native.astype(np.float64)
        flat_native, flat_trans = native.reshape(-1), transported.reshape(-1)
        order0 = np.argsort(-flat_native, kind="stable")
        order1 = np.argsort(-flat_trans, kind="stable")
        k = max(1, int(round(len(order0) * 0.01)))
        overlap = len(set(order0[:k].tolist()).intersection(order1[:k].tolist())) / k
        for metric, value in {
            "mean_abs_probability_delta": np.mean(np.abs(d)),
            "max_abs_probability_delta": np.max(np.abs(d)),
            "mean_probability_delta": np.mean(d),
            "pearson_probability": _corr(flat_native, flat_trans),
            "top1pct_overlap": overlap,
            "native_margin_mean": np.mean(data["native_margin"]),
            "transported_margin_mean": np.mean(data["cir_margin"]),
        }.items():
            rows.append({"epoch": epoch, "pair": f"{method}05_minus_{method}0", "metric": metric, "value": _f(value)})
        for cohort, selector in (("normal", data["masks"].sum(axis=(1, 2)) == 0), ("anomaly", data["masks"].sum(axis=(1, 2)) > 0)):
            vals = d[selector].reshape(-1)
            rows.append({"epoch": epoch, "pair": f"{method}05_minus_{method}0", "metric": f"{cohort}_mean_probability_delta", "value": _f(vals.mean())})
    return rows


def _heldout_rows(epoch: int, method: str, data: Mapping[str, np.ndarray], holdout: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    maps = data["p0"] if method.endswith("0") else data["p05"]
    cls = np.asarray(data["classification_probability"], dtype=np.float64)
    for split, selector in (("heldout_categories", np.asarray([x in holdout for x in data["class_names"]], dtype=bool)), ("seen_categories", np.asarray([x not in holdout for x in data["class_names"]], dtype=bool))):
        if not selector.any():
            continue
        scores = _method_image_scores(maps[selector], cls[selector])
        metrics = _metrics_for(maps[selector], data["masks"][selector], scores, data["labels"][selector])
        rows.append({"epoch": epoch, "method": method, "split": split, "category": "__macro__", "n_images": int(selector.sum()), **metrics})
        for category in sorted(set(data["class_names"][selector].tolist())):
            pick = selector & (data["class_names"] == category)
            if pick.sum() == 0:
                continue
            scores_c = _method_image_scores(maps[pick], cls[pick])
            metrics_c = _metrics_for(maps[pick], data["masks"][pick], scores_c, data["labels"][pick])
            rows.append({"epoch": epoch, "method": method, "split": split, "category": category, "n_images": int(pick.sum()), **metrics_c})
    return rows


def _module_swap(
    *,
    epoch: int,
    parent_config: Mapping[str, Any],
    p_payload: Mapping[str, Any],
    c_payload: Mapping[str, Any],
    clip_asset: Path,
    dataset: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    model = _make_model(parent_config, p_payload, clip_asset, device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    rows: list[dict[str, Any]] = []
    for combo in ("PPP", "CPP", "PCP", "PPC", "CCP", "CPC", "PCC", "CCC"):
        sources = {"P": p_payload, "C": c_payload}
        _apply_modules(model, sources[combo[0]], "PPP")
        model.text_adapter.load_state_dict(sources[combo[1]]["text_adapter"])
        model.soft_prompt.load_state_dict(sources[combo[2]]["soft_prompt"])
        maps: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        cls: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device).float()
                names = [str(v) for v in batch["class_name"]]
                out = forward_phase2b(model, image, names, device, parent_config, domain="Industrial", dataset_name="VisA")
                maps.append(out.deployed_segmentation_probability.detach().float().cpu().numpy())
                cls.append(out.classification_probability.detach().float().cpu().numpy())
                labels.append(batch["label"].numpy().astype(np.int64))
                masks.append(batch["mask"].numpy().astype(np.float32)[:, 0])
                del out, image
        map_a, cls_a = np.concatenate(maps), np.concatenate(cls)
        labels_a, masks_a = np.concatenate(labels), np.concatenate(masks)
        score_a = _method_image_scores(map_a, cls_a)
        rows.append({"epoch": epoch, "swap": combo, "n_images": len(map_a), **_metrics_for(map_a, masks_a, score_a, labels_a)})
    del model, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def _parameter_drift(parent_root: Path, cir_root: Path, output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        pp, cp = _checkpoint_paths(parent_root, cir_root, epoch)
        p, c = _load_payload(pp), _load_payload(cp)
        for component, key in (("image_adapter", "image_adapter"), ("text_adapter", "text_adapter"), ("soft_prompt", "soft_prompt")):
            names = sorted(set(p[key]) & set(c[key]))
            px = np.concatenate([p[key][n].detach().float().cpu().numpy().reshape(-1) for n in names])
            cx = np.concatenate([c[key][n].detach().float().cpu().numpy().reshape(-1) for n in names])
            diff = cx - px
            pnorm = float(np.linalg.norm(px))
            cnorm = float(np.linalg.norm(cx))
            dnorm = float(np.linalg.norm(diff))
            rows.append({
                "epoch": epoch, "component": component, "parameter_count": int(px.size),
                "l2_distance": dnorm, "normalized_l2": dnorm / max(pnorm, 1e-12),
                "cosine_flattened": float(np.dot(px, cx) / max(pnorm * cnorm, 1e-12)),
                "relative_update_magnitude": dnorm / max(cnorm, 1e-12),
                "max_abs_delta": float(np.max(np.abs(diff))),
                "common_initialization_reference": "UNAVAILABLE_NO_INIT_CHECKPOINT",
            })
            # Small submodule rows make it possible to identify a concentrated
            # drift without storing parameter tensors in the archive.
            for prefix in sorted({n.split(".")[0] for n in names}):
                sub = [n for n in names if n.split(".")[0] == prefix]
                sx = np.concatenate([p[key][n].detach().float().cpu().numpy().reshape(-1) for n in sub])
                sy = np.concatenate([c[key][n].detach().float().cpu().numpy().reshape(-1) for n in sub])
                sd = sy - sx
                rows.append({
                    "epoch": epoch, "component": f"{component}:{prefix}", "parameter_count": int(sx.size),
                    "l2_distance": float(np.linalg.norm(sd)),
                    "normalized_l2": float(np.linalg.norm(sd) / max(np.linalg.norm(sx), 1e-12)),
                    "cosine_flattened": float(np.dot(sx, sy) / max(np.linalg.norm(sx) * np.linalg.norm(sy), 1e-12)),
                    "relative_update_magnitude": float(np.linalg.norm(sd) / max(np.linalg.norm(sy), 1e-12)),
                    "max_abs_delta": float(np.max(np.abs(sd))),
                    "common_initialization_reference": "UNAVAILABLE_NO_INIT_CHECKPOINT",
                })
        del p, c
    _write_csv(output_root / "PARAMETER_DRIFT_BY_EPOCH.csv", rows, [
        "epoch", "component", "parameter_count", "l2_distance", "normalized_l2", "cosine_flattened", "relative_update_magnitude", "max_abs_delta", "common_initialization_reference"
    ])
    component_rows = [r for r in rows if ":" not in str(r["component"])]
    lines = [
        "# Parameter drift by epoch",
        "",
        "The CSV compares the frozen corrected parent checkpoint P with the corrected CIR checkpoint C0. Distances are descriptive; they are not by themselves evidence of harmful overspecialization.",
        "",
        "No pre-update/init checkpoint was recoverable from the frozen run roots, so the requested common-initialization reference is explicitly unavailable.",
        "",
        "| epoch | component | normalized L2 | flattened cosine | relative update |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in component_rows:
        lines.append(f"| {r['epoch']} | {r['component']} | {float(r['normalized_l2']):.6g} | {float(r['cosine_flattened']):.6g} | {float(r['relative_update_magnitude']):.6g} |")
    (output_root / "PARAMETER_DRIFT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root.expanduser().resolve()
    metadata = (ROOT / "dataset/hub/VisA.jsonl").resolve()
    parent_root = args.parent_run_root.expanduser().resolve()
    cir_root = args.cir_run_root.expanduser().resolve()
    clip_asset = args.clip_asset.expanduser().resolve()
    cir_config = load_cir_config(args.cir_config.expanduser().resolve())
    parent_config = json.loads((ROOT / str(cir_config["parent_config_path"])).read_text(encoding="utf-8"))
    indices, selected_rows = _sample_indices(metadata, args.per_category, args.seed)
    base_dataset = ManifestDataset(source_root, metadata, IMAGE_SIZE)
    dataset = Subset(base_dataset, indices)
    holdout = set(args.holdout_categories)
    identity = {
        "scope": "SOURCE_ONLY",
        "sample_seed": args.seed,
        "per_category": args.per_category,
        "n_images": len(indices),
        "categories": sorted({str(r["class_name"]) for r in selected_rows}),
        "holdout_categories": sorted(holdout),
        "source_root": str(source_root),
        "source_root_sha256": _sha256(source_root / "../VisA_20220922") if False else "see corrective manifest",
        "manifest": str(metadata),
        "manifest_sha256": _sha256(metadata),
        "cir_config_sha256": config_sha256(cir_config),
        "parent_config_sha256": _sha256(ROOT / "configs/phase2b_canonical_v1.json"),
        "clip_asset_sha256": _sha256(clip_asset),
        "selection": selected_rows,
        "protocol_note": "The model was trained on all VisA categories; category partitions are an assessment split, not a true unseen-training-category counterfactual.",
    }
    (output_root / "SOURCE_SAMPLE_IDENTITY.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _parameter_drift(parent_root, cir_root, output_root)
    device = torch.device(args.device)
    configure_canonical_fp32()
    all_rep_rows: list[dict[str, Any]] = []
    all_metric_rows: list[dict[str, Any]] = []
    all_comp_rows: list[dict[str, Any]] = []
    all_tail_rows: list[dict[str, Any]] = []
    all_branch_rows: list[dict[str, Any]] = []
    all_deploy_rows: list[dict[str, Any]] = []
    all_holdout_rows: list[dict[str, Any]] = []
    pair_cache: dict[int, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for epoch in EPOCHS:
        pp, cp = _checkpoint_paths(parent_root, cir_root, epoch)
        p, c, hashes = _paired_epoch(
            epoch=epoch, parent_config=parent_config, cir_config=cir_config,
            parent_path=pp, cir_path=cp, clip_asset=clip_asset, dataset=dataset,
            device=device, batch_size=args.batch_size, num_workers=args.num_workers,
        )
        pair_cache[epoch] = (p, c)
        all_rep_rows.extend(_feature_rows(epoch, p, c, len(indices)))
        for method, data in (("P0", p), ("P05", p), ("C0", c), ("C05", c)):
            all_metric_rows.append(_metric_rows(epoch, method, data))
            all_tail_rows.extend(_tail_rows(epoch, method, data["p0"] if method.endswith("0") else data["p05"], data["masks"]))
            all_holdout_rows.extend(_heldout_rows(epoch, method, data, holdout))
        all_comp_rows.extend(_compensation_rows(epoch, p, c))
        all_deploy_rows.extend(_deployment_rows(epoch, "P0", p))
        all_deploy_rows.extend(_deployment_rows(epoch, "C0", c))
        all_branch_rows.extend(_branch_rows(epoch, "P0", p))
        all_branch_rows.extend(_branch_rows(epoch, "C0", c))
        print(f"completed paired source diagnostic E{epoch:02d} ({len(indices)} images)", flush=True)
    _write_csv(output_root / "REPRESENTATION_DRIFT.csv", all_rep_rows, ["epoch", "signal", "axis", "mean_cosine", "norm_ratio_c", "linear_cka", "pairwise_geometry_corr", "mean_abs_delta", "drift_level", "n_images"])
    _write_csv(output_root / "SOURCE_BOUNDED_METRICS.csv", all_metric_rows, ["epoch", "method", "n_images", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap"])
    _write_csv(output_root / "RMT_COMPENSATION_AUDIT.csv", all_comp_rows, ["epoch", "pair", "metric", "value"])
    _write_csv(output_root / "AP_TAIL_DECOMPOSITION.csv", all_tail_rows, ["epoch", "method", "cohort", "stat", "value", "n"])
    _write_csv(output_root / "IMAGE_BRANCH_DECOMPOSITION.csv", all_branch_rows, ["epoch", "method", "branch", "image_auroc", "image_ap", "mean_score", "n_images"])
    _write_csv(output_root / "DEPLOYMENT_CAUSAL_DIAGNOSTIC.csv", all_deploy_rows, ["epoch", "method", "metric", "value"])
    _write_csv(output_root / "SOURCE_HELDOUT_RESULTS.csv", all_holdout_rows, ["epoch", "method", "split", "category", "n_images", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap"])
    # Module interventions are intentionally limited to E14, the preregistered
    # diagnostic checkpoint, and never touch a target-domain dataset.
    pp, cp = _checkpoint_paths(parent_root, cir_root, 14)
    p_payload, c_payload = _load_payload(pp), _load_payload(cp)
    swaps = _module_swap(epoch=14, parent_config=parent_config, p_payload=p_payload, c_payload=c_payload, clip_asset=clip_asset, dataset=dataset, device=device, batch_size=args.batch_size, num_workers=args.num_workers)
    _write_csv(output_root / "MODULE_SWAP_RESULTS.csv", swaps, ["epoch", "swap", "n_images", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap"])
    (output_root / "FORWARD_DIAGNOSTIC_STATUS.json").write_text(json.dumps({
        "status": "COMPLETE", "source_only": True, "epochs": list(EPOCHS), "module_swap_epoch": 14,
        "n_images": len(indices), "device": str(device), "holdout_categories": sorted(holdout),
        "checkpoint_hashes": {str(e): {"parent": _sha256(_checkpoint_paths(parent_root, cir_root, e)[0]), "cir": _sha256(_checkpoint_paths(parent_root, cir_root, e)[1])} for e in EPOCHS},
        "no_training": True, "no_medical": True, "no_mvtec": True,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run-root", type=Path, required=True)
    parser.add_argument("--cir-run-root", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--cir-config", type=Path, default=ROOT / "configs/cir_dfg_rmt_v2.json")
    parser.add_argument("--source-root", type=Path, default=Path("/home/ai4/caohuy/data/VisA_20220922"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9014)
    parser.add_argument("--holdout-categories", nargs="+", default=list(DEFAULT_HOLDOUT))
    args = parser.parse_args(argv)
    if args.per_category < 2 or args.batch_size < 1 or args.num_workers < 0:
        raise SystemExit("invalid bounded diagnostic sampling arguments")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
