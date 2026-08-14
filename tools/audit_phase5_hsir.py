#!/usr/bin/env python3
"""Phase5-A HSIR: inference-only hierarchical stage inconsistency audit.

The module deliberately keeps the predictor path delegated to the verified
Phase2B loader and uses one frozen implementation for both VisA and MVTec.
It writes only compact provenance, aggregate, and CSV artifacts; dense pixel
arrays live only in memory while one class is being reduced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from audit_p4_k1_oracle_utility import (  # noqa: E402
    DeterministicVisATrainDataset,
    _sha256,
)
from audit_p4v_phase2b_readiness import load_model as verified_load_model  # noqa: E402
from dataset import get_text_and_image_dataset  # noqa: E402
from model.adapter import gaussian_blur2d  # noqa: E402
from utils import configure_canonical_fp32, get_phase2b_global_text_features  # noqa: E402


OUTPUT_ROOT = ROOT / "runs" / "phase5" / "hsir"
CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
CONFIG = ROOT / "runs/phase4/k1/short64_seed0_attempt5/config.json"
VISA_MANIFEST = ROOT / "runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"
MVTEC_META = ROOT / "dataset/hub/MVTec.jsonl"

EPS = 1e-12
NUMERICAL_DYNAMIC_EPS = 1e-8
COVERAGE = 0.20
MARGIN_BINS = 10
BOOTSTRAP_REPS = 2000
AGGREGATE_SEEDS = {
    "stage_0_vs_stage_1": 10,
    "stage_0_vs_stage_2": 11,
    "stage_1_vs_stage_2": 12,
    "D_logit": 20,
    "D_rank": 21,
    "U_conf": 22,
}


class ProtocolAssumptionInvalid(RuntimeError):
    """The loaded runtime does not match the pinned Phase5 protocol."""


class AuditImplementationInvalid(RuntimeError):
    """A foundational implementation or parity invariant failed."""


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _finite(values):
    return np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)


def mean_defined(values):
    arr = _finite(values)
    return None if arr.size == 0 else float(arr.mean())


def median_defined(values):
    arr = _finite(values)
    return None if arr.size == 0 else float(np.median(arr))


def bootstrap_ci(values, seed: int = 0):
    arr = _finite(values)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, arr.size, size=(BOOTSTRAP_REPS, arr.size))
    means = arr[sample].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate_values(values, seed: int = 0):
    return {
        "mean": mean_defined(values),
        "median": median_defined(values),
        "bootstrap95_ci": bootstrap_ci(values, seed),
        "n_classes": int(_finite(values).size),
    }


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Average-tie percentile rank in [0, 1], ascending, vectorized."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    ranks_sorted = np.empty(n, dtype=np.float64)
    denom = max(n - 1, 1)
    for start, end in zip(starts, ends):
        ranks_sorted[start:end] = ((start + end - 1) / 2.0) / denom
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def population_std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.std(np.asarray(values, dtype=np.float64), axis=axis, ddof=0)


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or y.size != x.size:
        return None
    rx = percentile_rank(x)
    ry = percentile_rank(y)
    sx = float(rx.std())
    sy = float(ry.std())
    if sx <= EPS or sy <= EPS:
        return None
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def exact_auc_ap(scores, labels):
    """Exact class-pooled AUROC/AP with test.py's tie-group semantics."""
    scores = np.asarray(scores, dtype=np.float32).ravel()
    labels = np.asarray(labels, dtype=np.uint8).ravel()
    if scores.size != labels.size:
        raise ValueError("scores and labels must have equal length")
    total_pos = int(labels.sum())
    total_neg = int(labels.size - total_pos)
    if total_pos == 0 or total_neg == 0:
        return 0.0, 0.0
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64, copy=False)
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], sorted_scores.size]
    group_pos = np.add.reduceat(sorted_labels, starts)
    group_total = ends - starts
    group_neg = group_total - group_pos
    pos_before = np.r_[0, np.cumsum(group_pos[:-1])]
    pos_seen = np.cumsum(group_pos)
    total_seen = np.cumsum(group_total)
    auc_pairs = np.sum(group_neg * (pos_before + 0.5 * group_pos))
    ap = np.sum((pos_seen / total_seen) * (group_pos / total_pos))
    return float(auc_pairs / (total_pos * total_neg)), float(ap)


def project_exact_auc_ap(scores, labels):
    """Directly invoke the exact helper used by test.py on identical arrays."""
    from test import _exact_auc_ap_from_sorted_chunks

    with tempfile.TemporaryDirectory(prefix="phase5_hsir_metric_") as temp_dir:
        score_path = Path(temp_dir) / "scores.npy"
        label_path = Path(temp_dir) / "labels.npy"
        scores = np.asarray(scores, dtype=np.float32).ravel()
        labels = np.asarray(labels, dtype=np.uint8).ravel()
        order = np.argsort(-scores, kind="mergesort")
        np.save(score_path, scores[order])
        np.save(label_path, labels[order])
        auc_pct, ap_pct = _exact_auc_ap_from_sorted_chunks(
            [(str(score_path), str(label_path))], int(labels.sum()), int(labels.size - labels.sum())
        )
    return float(auc_pct / 100.0), float(ap_pct / 100.0)


def pairwise_risks(scores, labels):
    """Return AUROC-consistent positive and negative inversion risks."""
    scores = np.asarray(scores, dtype=np.float32).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    positive = scores[labels]
    negative = scores[~labels]
    if positive.size == 0 or negative.size == 0:
        return np.full(positive.size, np.nan), np.full(negative.size, np.nan)
    neg_sorted = np.sort(negative)
    pos_sorted = np.sort(positive)
    pos_left = np.searchsorted(neg_sorted, positive, side="left")
    pos_right = np.searchsorted(neg_sorted, positive, side="right")
    r_pos = (negative.size - pos_right + 0.5 * (pos_right - pos_left)) / negative.size
    neg_left = np.searchsorted(pos_sorted, negative, side="left")
    neg_right = np.searchsorted(pos_sorted, negative, side="right")
    r_neg = (neg_left + 0.5 * (neg_right - neg_left)) / positive.size
    return r_pos.astype(np.float64), r_neg.astype(np.float64)


def ap_contamination(scores, labels):
    """Exact positive AP-loss contribution after complete score groups."""
    scores = np.asarray(scores, dtype=np.float32).ravel()
    labels = np.asarray(labels, dtype=np.uint8).ravel()
    total_pos = int(labels.sum())
    out = np.full(scores.size, np.nan, dtype=np.float64)
    if total_pos == 0:
        return out
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64, copy=False)
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], sorted_scores.size]
    group_pos = np.add.reduceat(sorted_labels, starts)
    pos_seen = np.cumsum(group_pos)
    total_seen = np.cumsum(ends - starts)
    contamination = 1.0 - (pos_seen / total_seen)
    group_ids = np.repeat(np.arange(starts.size), ends - starts)
    sorted_out = contamination[group_ids]
    out[order] = sorted_out
    out[labels == 0] = np.nan
    return out


def upsample_patch_map(values, img_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    side = math.isqrt(values.size)
    if side * side != values.size:
        raise ValueError(f"patch count {values.size} is not a square")
    tensor = torch.from_numpy(values.reshape(1, 1, side, side))
    return (
        F.interpolate(tensor, size=img_size, mode="bilinear", align_corners=True)
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )


def deploy_from_native(native_group_logits: torch.Tensor, img_size: int, domain: str = "Industrial"):
    """Reconstruct deployment output from authoritative native group logits."""
    if native_group_logits.ndim != 4 or native_group_logits.shape[-1] != 2:
        raise ValueError(f"expected [G,B,P,2], got {tuple(native_group_logits.shape)}")
    _, batch, patch_count, _ = native_group_logits.shape
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise ValueError("native patch count must be square")
    sigma = 1 if domain == "Industrial" else 1.5
    kernel_size = 7 if domain == "Industrial" else 9
    group_logits = []
    for group in range(native_group_logits.shape[0]):
        logits = native_group_logits[group].permute(0, 2, 1).reshape(batch, 2, side, side)
        logits = gaussian_blur2d(logits, (kernel_size, kernel_size), (sigma, sigma))
        logits = F.interpolate(logits, size=img_size, mode="bilinear", align_corners=True)
        group_logits.append(logits)
    final_logits = torch.stack(group_logits, dim=0).mean(dim=0)
    return F.softmax(final_logits, dim=1), final_logits


def shifted_map(values: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.roll(values.reshape(height, width), shift=(height // 3, width // 3), axis=(0, 1)).reshape(-1)


def damage_capture(signal, damage, coverage=COVERAGE):
    signal = np.asarray(signal, dtype=np.float64).ravel()
    damage = np.asarray(damage, dtype=np.float64).ravel()
    valid = np.isfinite(signal) & np.isfinite(damage)
    signal = signal[valid]
    damage = damage[valid]
    if signal.size == 0:
        return {"capture_at_20": None, "auc": None, "random_capture_at_20": coverage, "random_auc": 0.5}
    order = np.argsort(-signal, kind="mergesort")
    cumulative = np.cumsum(np.maximum(damage[order], 0.0))
    total = float(cumulative[-1])
    if total <= EPS:
        normalized = np.zeros(signal.size, dtype=np.float64)
    else:
        normalized = cumulative / total
    x = np.arange(1, signal.size + 1, dtype=np.float64) / signal.size
    index = min(signal.size - 1, max(0, int(math.ceil(coverage * signal.size)) - 1))
    auc = float(np.trapezoid(np.r_[0.0, normalized], np.r_[0.0, x]))
    return {
        "capture_at_20": float(normalized[index]),
        "auc": auc,
        "random_capture_at_20": coverage,
        "random_auc": 0.5,
        "n_pixels": int(signal.size),
        "total_damage": total,
    }


def oracle_repair_ap_gain(scores, labels, signal, coverage=COVERAGE):
    scores = np.asarray(scores, dtype=np.float32).ravel()
    labels = np.asarray(labels, dtype=np.uint8).ravel()
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n = scores.size
    if n == 0 or labels.sum() == 0:
        return None
    selected_n = max(1, int(math.ceil(coverage * n)))
    selected_order = np.argsort(-signal, kind="mergesort")[:selected_n]
    selected = np.zeros(n, dtype=bool)
    selected[selected_order] = True
    base_order = np.argsort(-scores, kind="mergesort")
    selected_pos = base_order[selected[base_order] & (labels[base_order] == 1)]
    middle = base_order[~selected[base_order]]
    selected_neg = base_order[selected[base_order] & (labels[base_order] == 0)]
    repaired_order = np.concatenate([selected_pos, middle, selected_neg])
    repaired_labels = labels[repaired_order]
    pos = int(repaired_labels.sum())
    if pos == 0:
        return None
    cumulative_pos = np.cumsum(repaired_labels)
    precision = cumulative_pos / np.arange(1, n + 1)
    ap = float(np.sum(precision[repaired_labels == 1]) / pos)
    _, base_ap = exact_auc_ap(scores, labels)
    return {"base_ap": base_ap, "repaired_ap": ap, "gain": ap - base_ap, "selected_pixels": selected_n}


def summarize_distribution(values):
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def signal_associations(signal, r_pos, c_ap, r_neg, positive_mask, negative_mask):
    positive_signal = signal[positive_mask]
    negative_signal = signal[negative_mask]
    return {
        "positive_r_pos_spearman": spearman(positive_signal, r_pos),
        "positive_c_ap_spearman": spearman(positive_signal, c_ap[positive_mask]),
        "negative_r_neg_all_spearman": spearman(negative_signal, r_neg),
        "negative_r_neg_hard_spearman": spearman(
            negative_signal[r_neg > 0], r_neg[r_neg > 0]
        ) if np.any(r_neg > 0) else None,
    }


def score_matched_metrics(signals, margin, labels, r_pos, c_ap, r_neg):
    margin = np.asarray(margin, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    order = np.argsort(margin, kind="mergesort")
    bins = np.empty(margin.size, dtype=np.int64)
    bins[order] = np.minimum((np.arange(margin.size) * MARGIN_BINS) // max(margin.size, 1), MARGIN_BINS - 1)
    result = {name: {"positive_r_pos_spearman": [], "positive_c_ap_spearman": [], "negative_r_neg_hard_spearman": []} for name in signals}
    weights = {name: {key: [] for key in result[name]} for name in signals}
    for bucket in range(MARGIN_BINS):
        in_bin = bins == bucket
        pos = in_bin & labels
        neg = in_bin & ~labels
        hard = neg & (r_neg > 0)
        for name, signal in signals.items():
            signal = np.asarray(signal, dtype=np.float64)
            if np.sum(pos) >= 2:
                value = spearman(signal[pos], r_pos[pos])
                if value is not None:
                    result[name]["positive_r_pos_spearman"].append(value)
                    weights[name]["positive_r_pos_spearman"].append(int(np.sum(pos)))
                value = spearman(signal[pos], c_ap[pos])
                if value is not None:
                    result[name]["positive_c_ap_spearman"].append(value)
                    weights[name]["positive_c_ap_spearman"].append(int(np.sum(pos)))
            if np.sum(hard) >= 2:
                value = spearman(signal[hard], r_neg[hard])
                if value is not None:
                    result[name]["negative_r_neg_hard_spearman"].append(value)
                    weights[name]["negative_r_neg_hard_spearman"].append(int(np.sum(hard)))
    compact = {}
    for name in signals:
        compact[name] = {}
        for key, values in result[name].items():
            w = np.asarray(weights[name][key], dtype=np.float64)
            v = np.asarray(values, dtype=np.float64)
            compact[name][key] = None if v.size == 0 else float(np.average(v, weights=w))
        vals = [compact[name][key] for key in compact[name]]
        compact[name]["mean_signed_residual"] = mean_defined(vals)
    return compact


def analyze_class(class_name: str, items: list[dict[str, Any]], parity_errors: list[dict[str, float]]):
    scores = np.concatenate([item["score"].ravel() for item in items]).astype(np.float32, copy=False)
    margins = np.concatenate([item["final_margin"].ravel() for item in items]).astype(np.float32, copy=False)
    labels = np.concatenate([item["target"].ravel() for item in items]).astype(np.uint8, copy=False)
    signals = {
        "D_logit": np.concatenate([item["D_logit"].ravel() for item in items]),
        "D_rank": np.concatenate([item["D_rank"].ravel() for item in items]),
        "U_conf": np.concatenate([item["U_conf"].ravel() for item in items]),
    }
    d_jack = np.concatenate([item["D_jack"].ravel() for item in items])
    positive = labels.astype(bool)
    negative = ~positive
    auc, ap = exact_auc_ap(scores, labels)
    ref_auc, ref_ap = project_exact_auc_ap(scores, labels)
    r_pos, r_neg = pairwise_risks(scores, labels)
    c_ap = ap_contamination(scores, labels)
    r_pos_all = np.full(scores.size, np.nan, dtype=np.float64)
    r_neg_all = np.full(scores.size, np.nan, dtype=np.float64)
    r_pos_all[positive] = r_pos
    r_neg_all[negative] = r_neg

    signal_metrics = {}
    for name, signal in signals.items():
        assoc = signal_associations(signal, r_pos, c_ap, r_neg, positive, negative)
        capture = damage_capture(signal[positive], c_ap[positive])
        oracle = oracle_repair_ap_gain(scores, labels, signal)
        signal_metrics[name] = {
            **assoc,
            "damage_capture": capture,
            "oracle_repair_at_20": oracle,
        }

    signal_metrics["D_jack"] = {
        **signal_associations(d_jack, r_pos, c_ap, r_neg, positive, negative),
        "damage_capture": damage_capture(d_jack[positive], c_ap[positive]),
        "oracle_repair_at_20": oracle_repair_ap_gain(scores, labels, d_jack),
    }

    aligned_shifted = {}
    height = items[0]["score"].shape[0]
    width = items[0]["score"].shape[1]
    for name in ("D_logit", "D_rank"):
        shifted = np.concatenate([shifted_map(item[name], height, width) for item in items])
        shifted_metrics = signal_associations(shifted, r_pos, c_ap, r_neg, positive, negative)
        shifted_capture = damage_capture(shifted[positive], c_ap[positive])
        aligned_shifted[name] = {
            "aligned": signal_metrics[name],
            "shifted": {**shifted_metrics, "damage_capture": shifted_capture},
        }

    score_matched = score_matched_metrics(signals, margins, positive, r_pos_all, c_ap, r_neg_all)
    stage_margins = np.concatenate([item["native_margins"] for item in items], axis=1)
    pairwise_spearman = {
        f"stage_{i}_vs_stage_{j}": spearman(stage_margins[i], stage_margins[j])
        for i, j in ((0, 1), (0, 2), (1, 2))
    }

    stage_stats = {
        "pairwise_spearman": pairwise_spearman,
        "D_logit": summarize_distribution(signals["D_logit"]),
        "D_rank": summarize_distribution(signals["D_rank"]),
        "D_rank_effectively_zero_fraction": float(np.mean(signals["D_rank"] <= NUMERICAL_DYNAMIC_EPS)),
    }
    class_consistency = {}
    for name in ("D_logit", "D_rank"):
        values = [
            signal_metrics[name]["positive_r_pos_spearman"],
            signal_metrics[name]["positive_c_ap_spearman"],
            signal_metrics[name]["negative_r_neg_hard_spearman"],
        ]
        finite = _finite(values)
        positive_count = int(np.sum(finite > 0))
        negative_count = int(np.sum(finite < 0))
        if finite.size == 0 or (positive_count < 2 and negative_count < 2):
            status = "neutral"
        elif positive_count >= 2:
            status = "supported"
        else:
            status = "opposed"
        class_consistency[name] = {"status": status, "finite_association_count": int(finite.size)}

    per_image = []
    for item in items:
        item_scores = item["score"].ravel()
        item_labels = item["target"].ravel().astype(np.uint8)
        item_auc, item_ap = exact_auc_ap(item_scores, item_labels)
        per_image.append({
            "class_name": class_name,
            "file_name": item["file_name"],
            "label": int(item["label"]),
            "pixel_ap": item_ap if item_labels.sum() and (item_labels == 0).sum() else None,
            "pixel_auroc": item_auc if item_labels.sum() and (item_labels == 0).sum() else None,
            "positive_pixels": int(item_labels.sum()),
            "negative_pixels": int(item_labels.size - item_labels.sum()),
            "mean_D_logit": float(item["D_logit"].mean()),
            "p95_D_logit": float(np.quantile(item["D_logit"], 0.95)),
            "mean_D_rank": float(item["D_rank"].mean()),
            "p95_D_rank": float(np.quantile(item["D_rank"], 0.95)),
            "mean_U_conf": float(item["U_conf"].mean()),
        })

    return {
        "class_name": class_name,
        "n_images": len(items),
        "n_positive_pixels": int(positive.sum()),
        "n_negative_pixels": int(negative.sum()),
        "pixel_ap": ap,
        "pixel_auroc": auc,
        "mean_positive_C_AP": mean_defined(c_ap[positive]),
        "harmful_normal_fraction": float(np.mean(r_neg > 0)) if r_neg.size else None,
        "mean_positive_pairwise_risk": mean_defined(r_pos),
        "mean_negative_pairwise_risk": mean_defined(r_neg),
        "parity": {
            "predictor_max_abs_probability_error": max(x["predictor_max_abs_probability_error"] for x in parity_errors),
            "ap_reconstruction_error": abs(ap - ref_ap),
            "auroc_reconstruction_error": abs(auc - ref_auc),
        },
        "stage_diversity": stage_stats,
        "signals": signal_metrics,
        "negative_control": aligned_shifted,
        "score_matched": score_matched,
        "class_consistency": class_consistency,
        "per_image": per_image,
    }


def _flatten_class_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        key: row[key]
        for key in (
            "class_name", "n_images", "n_positive_pixels", "n_negative_pixels",
            "pixel_ap", "pixel_auroc", "mean_positive_C_AP", "harmful_normal_fraction",
            "mean_positive_pairwise_risk", "mean_negative_pairwise_risk",
        )
    }
    for name in ("D_logit", "D_rank", "U_conf"):
        metrics = row["signals"][name]
        for key in ("positive_r_pos_spearman", "positive_c_ap_spearman", "negative_r_neg_all_spearman", "negative_r_neg_hard_spearman"):
            flat[f"{name}_{key}"] = metrics[key]
        flat[f"{name}_damage_capture_auc"] = metrics["damage_capture"]["auc"]
        flat[f"{name}_capture_at_20"] = metrics["damage_capture"]["capture_at_20"]
        flat[f"{name}_oracle_ap_gain_at_20"] = None if metrics["oracle_repair_at_20"] is None else metrics["oracle_repair_at_20"]["gain"]
        flat[f"{name}_score_matched_residual"] = row["score_matched"][name]["mean_signed_residual"]
    for name in ("D_logit", "D_rank"):
        flat[f"{name}_shifted_damage_capture_auc"] = row["negative_control"][name]["shifted"]["damage_capture"]["auc"]
        flat[f"{name}_aligned_minus_shifted_capture_auc"] = (
            row["negative_control"][name]["aligned"]["damage_capture"]["auc"]
            - row["negative_control"][name]["shifted"]["damage_capture"]["auc"]
        )
    flat["D_logit_class_status"] = row["class_consistency"]["D_logit"]["status"]
    flat["D_rank_class_status"] = row["class_consistency"]["D_rank"]["status"]
    flat["stage_D_logit_p95"] = row["stage_diversity"]["D_logit"]["p95"]
    flat["stage_D_rank_p95"] = row["stage_diversity"]["D_rank"]["p95"]
    flat["stage_D_rank_zero_fraction"] = row["stage_diversity"]["D_rank_effectively_zero_fraction"]
    return flat


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def dataset_aggregate(class_rows: list[dict[str, Any]], provenance: dict[str, Any], architecture: dict[str, Any], parity: dict[str, Any]):
    stage_rows = [row["stage_diversity"] for row in class_rows]
    stage_aggregate = {
        "pairwise_spearman": {
            key: aggregate_values([row["pairwise_spearman"].get(key) for row in stage_rows], i)
            for i, key in enumerate(("stage_0_vs_stage_1", "stage_0_vs_stage_2", "stage_1_vs_stage_2"))
        },
        "D_logit": {key: aggregate_values([row["D_logit"].get(key) for row in stage_rows], i + 10) for i, key in enumerate(("mean", "p50", "p95", "max"))},
        "D_rank": {key: aggregate_values([row["D_rank"].get(key) for row in stage_rows], i + 20) for i, key in enumerate(("mean", "p50", "p95", "max"))},
        "D_rank_effectively_zero_fraction": aggregate_values([row["D_rank_effectively_zero_fraction"] for row in stage_rows], 30),
    }
    signal_aggregate = {}
    for name in ("D_logit", "D_rank", "U_conf"):
        signal_aggregate[name] = {}
        for key in (
            "positive_r_pos_spearman", "positive_c_ap_spearman", "negative_r_neg_all_spearman",
            "negative_r_neg_hard_spearman",
        ):
            signal_aggregate[name][key] = aggregate_values(
                [row["signals"][name][key] for row in class_rows],
                100 + AGGREGATE_SEEDS[name] + sum(ord(char) for char in key),
            )
        for key in ("auc", "capture_at_20"):
            aggregate_key = "damage_capture_auc" if key == "auc" else "damage_capture_at_20"
            signal_aggregate[name][aggregate_key] = aggregate_values(
                [row["signals"][name]["damage_capture"][key] for row in class_rows],
                200 + AGGREGATE_SEEDS[name] + (0 if key == "auc" else 1)
            )
        signal_aggregate[name]["oracle_ap_gain_at_20"] = aggregate_values(
            [None if row["signals"][name]["oracle_repair_at_20"] is None else row["signals"][name]["oracle_repair_at_20"]["gain"] for row in class_rows],
            300 + AGGREGATE_SEEDS[name],
        )
        signal_aggregate[name]["score_matched_residual"] = aggregate_values(
            [row["score_matched"][name]["mean_signed_residual"] for row in class_rows],
            400 + AGGREGATE_SEEDS[name]
        )

    negative_control = {}
    for name in ("D_logit", "D_rank"):
        negative_control[name] = {
            "aligned_damage_capture_auc": aggregate_values([row["negative_control"][name]["aligned"]["damage_capture"]["auc"] for row in class_rows], 500),
            "shifted_damage_capture_auc": aggregate_values([row["negative_control"][name]["shifted"]["damage_capture"]["auc"] for row in class_rows], 501),
            "aligned_positive_r_pos": aggregate_values([row["negative_control"][name]["aligned"]["positive_r_pos_spearman"] for row in class_rows], 502),
            "shifted_positive_r_pos": aggregate_values([row["negative_control"][name]["shifted"]["positive_r_pos_spearman"] for row in class_rows], 503),
        }

    base = {
        "pixel_ap": aggregate_values([row["pixel_ap"] for row in class_rows], 600),
        "pixel_auroc": aggregate_values([row["pixel_auroc"] for row in class_rows], 601),
        "mean_positive_C_AP": aggregate_values([row["mean_positive_C_AP"] for row in class_rows], 602),
        "harmful_normal_fraction": aggregate_values([row["harmful_normal_fraction"] for row in class_rows], 603),
        "mean_positive_pairwise_risk": aggregate_values([row["mean_positive_pairwise_risk"] for row in class_rows], 604),
    }
    dynamic = {
        "stage_dynamic_range_valid": any(
            (row["stage_diversity"]["D_logit"]["max"] or 0) > NUMERICAL_DYNAMIC_EPS
            or (row["stage_diversity"]["D_rank"]["max"] or 0) > NUMERICAL_DYNAMIC_EPS
            for row in class_rows
        ),
        "ranking_error_dynamic_range_valid": any(
            (row["mean_positive_C_AP"] or 0) > NUMERICAL_DYNAMIC_EPS
            or (row["harmful_normal_fraction"] or 0) > 0
            for row in class_rows
        ),
        "numerical_epsilon": NUMERICAL_DYNAMIC_EPS,
    }
    consistency = {
        name: {
            "supported_classes": sum(row["class_consistency"][name]["status"] == "supported" for row in class_rows),
            "neutral_classes": sum(row["class_consistency"][name]["status"] == "neutral" for row in class_rows),
            "opposed_classes": sum(row["class_consistency"][name]["status"] == "opposed" for row in class_rows),
            "total_classes": len(class_rows),
        }
        for name in ("D_logit", "D_rank")
    }
    return {
        "provenance": provenance,
        "architecture": architecture,
        "parity": parity,
        "stage_diversity": stage_aggregate,
        "base_error_range": base,
        "signals": signal_aggregate,
        "negative_control": negative_control,
        "score_matched": {name: signal_aggregate[name]["score_matched_residual"] for name in ("D_logit", "D_rank", "U_conf")},
        "class_consistency": consistency,
        "dynamic_range": dynamic,
        "per_class": class_rows,
    }


def decision_for_dataset(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary["dynamic_range"]["stage_dynamic_range_valid"] or not summary["dynamic_range"]["ranking_error_dynamic_range_valid"]:
        return {"status": "inconclusive", "reason": "insufficient stage or ranking-error dynamic range"}
    evidence = {}
    for name in ("D_logit", "D_rank"):
        agg = summary["signals"][name]
        consistency = summary["class_consistency"][name]
        assoc = mean_defined([
            agg["positive_r_pos_spearman"]["median"],
            agg["positive_c_ap_spearman"]["median"],
            agg["negative_r_neg_hard_spearman"]["median"],
        ])
        capture = (
            (agg["damage_capture_auc"]["median"] or 0) > 0.5
            and (agg["damage_capture_at_20"]["median"] or 0) > COVERAGE
        )
        aligned = summary["negative_control"][name]["aligned_damage_capture_auc"]["median"]
        shifted = summary["negative_control"][name]["shifted_damage_capture_auc"]["median"]
        oracle = agg["oracle_ap_gain_at_20"]["median"]
        conf_oracle = summary["signals"]["U_conf"]["oracle_ap_gain_at_20"]["median"]
        residual = agg["score_matched_residual"]["median"]
        conf_residual = summary["signals"]["U_conf"]["score_matched_residual"]["median"]
        evidence[name] = {
            "association_direction_positive": (assoc or 0) > 0,
            "damage_capture_above_random": capture,
            "multiple_class_direction": consistency["supported_classes"] > consistency["total_classes"] / 2,
            "aligned_beats_shifted": aligned is not None and shifted is not None and aligned > shifted,
            "score_matched_residual_positive": (residual or 0) > 0,
            "oracle_gain_beats_confidence": oracle is not None and conf_oracle is not None and oracle > conf_oracle,
            "coherent_support": all((assoc or 0) > 0 for _ in [0]) and capture and consistency["supported_classes"] > consistency["total_classes"] / 2,
            "score_matched_residual": residual,
            "confidence_score_matched_residual": conf_residual,
            "oracle_gain": oracle,
            "confidence_oracle_gain": conf_oracle,
        }
    strong = all(evidence[name]["coherent_support"] and evidence[name]["score_matched_residual_positive"] and evidence[name]["aligned_beats_shifted"] and evidence[name]["oracle_gain_beats_confidence"] for name in ("D_logit", "D_rank"))
    simple = any(evidence[name]["coherent_support"] for name in ("D_logit", "D_rank"))
    residuals = [evidence[name]["score_matched_residual"] for name in ("D_logit", "D_rank")]
    confidence_residual = summary["signals"]["U_conf"]["score_matched_residual"]["median"]
    confidence_redundant = all((value or 0) <= (confidence_residual or 0) for value in residuals)
    if strong:
        status = "supported"
    elif simple:
        status = "weak"
    else:
        status = "null"
    return {
        "status": status,
        "confidence_redundant": confidence_redundant,
        "evidence": evidence,
        "random_damage_capture_auc": 0.5,
        "random_capture_at_20": COVERAGE,
        "decision_rule": "co-primary signals are reported separately; strong requires both coherent support, aligned-vs-shifted control, score-matched residual, and oracle gain over confidence",
    }


def prepare_mvtec_probe(meta_path: Path, output_path: Path, img_size: int):
    datasets = get_text_and_image_dataset("MVTec", img_size, stage="test")
    samples = []
    classes = []
    for class_name in sorted(datasets):
        dataset = datasets[class_name]
        classes.append(class_name)
        for label in (0, 1):
            candidates = []
            for index, row in enumerate(dataset.meta):
                if int(row["label"]) != label:
                    continue
                file_name = str(row["image_path"])
                key = f"MVTec|{class_name}|{file_name}"
                candidates.append((hashlib.sha256(key.encode()).hexdigest(), index, row))
            candidates.sort(key=lambda value: value[0])
            for rank, (digest, index, row) in enumerate(candidates[:2]):
                samples.append({
                    "dataset": "MVTec",
                    "class_name": class_name,
                    "label": int(row["label"]),
                    "file_name": str(row["image_path"]),
                    "mask_file_name": row.get("mask_path"),
                    "source_index": int(index),
                    "selection_sha256": digest,
                    "selection_rank_within_label": rank,
                })
    payload = {
        "dataset": "MVTec",
        "metadata_path": str(meta_path),
        "metadata_sha256": _sha256(meta_path),
        "img_size": img_size,
        "selection": "up to two Normal and two anomaly images per class; SHA256 ordering of dataset|class|file_name",
        "classes": classes,
        "samples": samples,
        "counts_by_class": {
            class_name: {
                "normal": sum(x["class_name"] == class_name and x["label"] == 0 for x in samples),
                "anomaly": sum(x["class_name"] == class_name and x["label"] == 1 for x in samples),
            }
            for class_name in classes
        },
    }
    write_json(output_path, payload)
    missing = []
    for sample in samples:
        dataset = datasets[sample["class_name"]]
        image_path = Path(dataset.data_path) / sample["file_name"]
        if not image_path.is_file():
            missing.append(str(image_path))
    if missing:
        raise FileNotFoundError(
            "MVTec image tree is unavailable; first missing selected image: "
            + missing[0]
        )
    return payload, datasets


def predict_one(model, raw: dict[str, Any], dataset_name: str, class_name: str, img_size: int, text_cache: dict[str, torch.Tensor], device):
    image = raw["image"].unsqueeze(0).to(device).float()
    target = raw["mask"].to(device).float().squeeze(0).cpu().numpy().astype(np.uint8)
    visual = model(image, return_phase4_features=True)
    features = torch.stack(visual["seg_tokens"])
    if class_name not in text_cache:
        text_cache[class_name] = get_phase2b_global_text_features(
            model,
            dataset_name,
            [class_name],
            device,
            use_hybrid_soft_prompt=True,
            use_soft_prompt=False,
        ).float()
    text = text_cache[class_name]
    model_prob, native, native_margin = model.vision_text_fusion_gate_seg(
        features,
        text,
        img_size=img_size,
        test_mode=True,
        domain="Industrial",
        return_details=True,
    )
    reconstructed_prob, final_logits = deploy_from_native(native, img_size, "Industrial")
    parity = float((model_prob - reconstructed_prob[:, 1]).abs().max().detach().cpu())
    native_logits = native[:, 0].detach().float().cpu().numpy()
    native_margins = native_margin[:, 0].detach().float().cpu().numpy()
    d_logit = upsample_patch_map(population_std(native_margins, axis=0).astype(np.float32), img_size)
    rank_maps = np.stack([percentile_rank(stage) for stage in native_margins], axis=0)
    d_rank = upsample_patch_map(population_std(rank_maps, axis=0).astype(np.float32), img_size)
    if native_margins.shape[0] > 1:
        jackknife = np.stack([
            percentile_rank(np.mean(np.delete(native_margins, group, axis=0), axis=0))
            for group in range(native_margins.shape[0])
        ], axis=0)
        d_jack = upsample_patch_map(population_std(jackknife, axis=0).astype(np.float32), img_size)
    else:
        d_jack = np.zeros((img_size, img_size), dtype=np.float32)
    final_logits_np = final_logits[0].detach().float().cpu().numpy()
    score = reconstructed_prob[0, 1].detach().float().cpu().numpy()
    final_margin = final_logits_np[1] - final_logits_np[0]
    return {
        "score": score,
        "final_margin": final_margin,
        "target": target,
        "D_logit": d_logit.astype(np.float32),
        "D_rank": d_rank.astype(np.float32),
        "D_jack": d_jack.astype(np.float32),
        "U_conf": (-np.abs(final_margin)).astype(np.float32),
        "native_margins": native_margins.astype(np.float32),
        "native_logits": native_logits.astype(np.float32),
        "parity": {"predictor_max_abs_probability_error": parity},
        "file_name": str(raw["file_name"]),
        "label": int(raw["label"]),
    }


def build_architecture(model, config: dict[str, Any], checkpoint: dict[str, Any]):
    architecture = {
        "n_groups": int(model.n_groups),
        "image_levels": list(model.image_levels),
        "text_levels": list(model.text_levels),
        "dfg_mode": model.dfg_mode,
        "dfg_beta_runtime": float(model.dfg_beta),
        "dfg_beta_target_runtime": float(model.dfg_beta_target),
        "use_ss2d_dfg": bool(model.use_ss2d_dfg),
        "dfg_ss2d_fusion": model.dfg_ss2d_fusion,
        "dfg_attn_dim": int(model.dfg_attn_dim),
        "dfg_attn_tau": float(model.dfg_attn_tau),
        "hybrid_prompt_mode": checkpoint.get("prompt_mode", "unknown"),
        "hybrid_alpha_current": checkpoint.get("hybrid_alpha_current"),
        "image_size": int(config["img_size"]),
        "h6_enabled_but_not_supplied_to_phase2b_predictor": bool(getattr(model, "h6_enabled", False)),
    }
    expected = {
        "n_groups": 3,
        "image_levels": [8, 16, 24],
        "text_levels": [4, 8, 12],
        "dfg_mode": "attn",
        "use_ss2d_dfg": True,
        "dfg_ss2d_fusion": "weight_residual",
        "dfg_beta_target": 0.10,
    }
    mismatches = {}
    for key, value in expected.items():
        actual_key = "dfg_beta_target_runtime" if key == "dfg_beta_target" else key
        actual = architecture[actual_key]
        if actual != value:
            mismatches[key] = {"expected": value, "actual": actual}
    architecture["expected"] = expected
    architecture["mismatches"] = mismatches
    if mismatches:
        raise ProtocolAssumptionInvalid(json.dumps(mismatches, sort_keys=True))
    return architecture


def make_protocol(config: dict[str, Any], checkpoint: dict[str, Any], architecture: dict[str, Any], paths: dict[str, Path]):
    return {
        "protocol": "PHASE5-A v2 HSIR",
        "question": "Does hierarchical stage inconsistency predict ranking-critical error beyond final confidence?",
        "inference_only": True,
        "training_steps": 0,
        "repo_head": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)} for key, path in paths.items()
        },
        "checkpoint_epoch": checkpoint.get("epoch"),
        "architecture": architecture,
        "loader": {
            "source": "tools/audit_p4v_phase2b_readiness.py",
            "functions": ["build", "load_model", "get_phase2b_global_text_features"],
            "precision": "strict FP32; TF32 off; AMP off",
        },
        "predictor_semantics": {
            "features": "stack(model(image, return_phase4_features=True)['seg_tokens'])",
            "authoritative_call": "model.vision_text_fusion_gate_seg(features, text, img_size, test_mode=True, domain='Industrial', return_details=True)",
            "native_logits": "returned base_group_logits [G,B,P,2]",
            "deployment": "Industrial Gaussian blur kernel=7 sigma=1; bilinear align_corners=True; mean group logits; softmax class dimension",
            "native_margin": "z_abnormal - z_normal before deployment",
            "deployed_margin": "final_logits abnormal - normal after deployment",
        },
        "frozen_formulas": {
            "D_logit": "population std across groups of native margins",
            "D_rank": "population std across groups of within-image average-tie percentile ranks",
            "D_jack": "population std across leave-one-stage-out consensus percentile ranks; diagnostic only",
            "U_conf": "-abs(deployed final margin)",
            "R_pos": "fraction normal scores strictly greater plus half tied fraction",
            "R_neg": "fraction anomaly scores strictly lower plus half tied fraction",
            "C_AP": "1 - precision after complete equal-score group",
            "coverage": COVERAGE,
            "margin_bins": MARGIN_BINS,
            "spatial_shift": ["vertical=floor(H/3)", "horizontal=floor(W/3)", "wraparound"],
            "bootstrap": "classes, 2000 resamples, deterministic seeds",
        },
        "metric_reference": {
            "source": "test.py::_exact_auc_ap_from_sorted_chunks",
            "tie_semantics": "equal-score groups together; AUROC ties receive 0.5 pair credit",
        },
        "datasets": {
            "VisA": "existing deterministic 48-image audit manifest and deterministic test preprocessing",
            "MVTec": "get_text_and_image_dataset('MVTec', img_size, stage='test'); up to 2 normal + 2 anomaly per class by SHA256 ordering",
        },
        "decision_rules": {
            "dataset_status": "inconclusive if stage or ranking-error dynamic range is absent; supported requires both co-primary signals to show coherent association, above-random damage capture, multi-class direction, aligned-vs-shifted control, score-matched residual, and oracle gain over confidence; weak if only simple co-primary support; otherwise null",
            "final_terminals": [
                "PROTOCOL_ASSUMPTION_INVALID", "AUDIT_IMPLEMENTATION_INVALID", "STAGE_INCONSISTENCY_NO_DYNAMIC_RANGE",
                "STAGE_RANK_RISK_INCONCLUSIVE", "STAGE_RANK_RISK_NOT_SUPPORTED", "STAGE_INCONSISTENCY_CONFIDENCE_REDUNDANT",
                "STAGE_RANK_RISK_AUXILIARY_ONLY", "STAGE_RANK_RISK_DOMAIN_SHIFT_EMERGENT", "STAGE_RANK_RISK_ZERO_SHOT_SUPPORTED",
            ],
        },
    }


def run_unit_tests() -> dict[str, Any]:
    tests = {}
    margins = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float64)
    tests["identical_group_margins"] = {"D_logit": float(population_std(margins, 0).max()), "D_rank": float(population_std(np.stack([percentile_rank(x) for x in margins]), 0).max())}
    scaled = np.array([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=np.float64)
    tests["identical_order_scaled_values"] = {"D_logit": float(population_std(scaled, 0).max()), "D_rank": float(population_std(np.stack([percentile_rank(x) for x in scaled]), 0).max())}
    reversed_values = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.float64)
    tests["reversed_order"] = {"D_rank_max": float(population_std(np.stack([percentile_rank(x) for x in reversed_values]), 0).max())}
    native = torch.arange(3 * 1 * 16 * 2, dtype=torch.float32).reshape(3, 1, 16, 2) / 10
    actual_prob, actual_logits = deploy_from_native(native, 16)
    expected_groups = []
    for group in range(3):
        expected = native[group].permute(0, 2, 1).reshape(1, 2, 4, 4)
        expected = gaussian_blur2d(expected, (7, 7), (1, 1))
        expected_groups.append(F.interpolate(expected, size=16, mode="bilinear", align_corners=True))
    expected_logits = torch.stack(expected_groups).mean(0)
    tests["deployment_operator"] = {"max_abs_logit_error": float((actual_logits - expected_logits).abs().max()), "max_abs_probability_error": float((actual_prob - F.softmax(expected_logits, 1)).abs().max())}
    scores = np.array([0.9, 0.8, 0.8, 0.2, 0.1], dtype=np.float32)
    labels = np.array([1, 0, 1, 0, 0], dtype=np.uint8)
    auc, ap = exact_auc_ap(scores, labels)
    r_pos, r_neg = pairwise_risks(scores, labels)
    tests["pairwise_auc_identity"] = {"error_positive": abs(float(np.mean(r_pos)) - (1 - auc)), "error_negative": abs(float(np.mean(r_neg)) - (1 - auc))}
    c_ap = ap_contamination(scores, labels)
    tests["ap_identity"] = {"error": abs(float(np.mean(c_ap[labels == 1])) - (1 - ap))}
    shifted = shifted_map(np.arange(16, dtype=np.float32), 4, 4)
    tests["spatial_shift_control"] = {"distribution_error": float(np.max(np.sort(shifted) - np.sort(np.arange(16, dtype=np.float32)))), "alignment_changed": bool(not np.array_equal(shifted, np.arange(16, dtype=np.float32)))}
    checks = {
        "identical_group_margins": tests["identical_group_margins"]["D_logit"] <= EPS and tests["identical_group_margins"]["D_rank"] <= EPS,
        "identical_order_scaled_values": tests["identical_order_scaled_values"]["D_rank"] <= EPS and tests["identical_order_scaled_values"]["D_logit"] > 0,
        "reversed_order": tests["reversed_order"]["D_rank_max"] > 0,
        "deployment_operator": tests["deployment_operator"]["max_abs_logit_error"] <= EPS and tests["deployment_operator"]["max_abs_probability_error"] <= EPS,
        "pairwise_auc_identity": tests["pairwise_auc_identity"]["error_positive"] <= 1e-12 and tests["pairwise_auc_identity"]["error_negative"] <= 1e-12,
        "ap_identity": tests["ap_identity"]["error"] <= 1e-12,
        "spatial_shift_control": tests["spatial_shift_control"]["distribution_error"] <= EPS and tests["spatial_shift_control"]["alignment_changed"],
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "tests": tests, "checks": checks, "formulas": {"percentile_rank": "average ties, ascending, divide by n-1", "std": "unbiased=False"}}


def final_terminal(visa_decision: dict[str, Any], mvtec_decision: dict[str, Any], summaries: dict[str, Any]) -> str:
    visa = visa_decision["status"]
    mvtec = mvtec_decision["status"]
    visa_inconclusive = visa == "inconclusive"
    mvtec_inconclusive = mvtec == "inconclusive"
    if visa_inconclusive and mvtec_inconclusive:
        both_no_dynamic = not summaries["VisA"]["dynamic_range"]["stage_dynamic_range_valid"] and not summaries["MVTEC"]["dynamic_range"]["stage_dynamic_range_valid"]
        return "STAGE_INCONSISTENCY_NO_DYNAMIC_RANGE" if both_no_dynamic else "STAGE_RANK_RISK_INCONCLUSIVE"
    if visa_decision.get("confidence_redundant") and mvtec_decision.get("confidence_redundant"):
        return "STAGE_INCONSISTENCY_CONFIDENCE_REDUNDANT"
    if visa == "supported" and mvtec == "supported":
        return "STAGE_RANK_RISK_ZERO_SHOT_SUPPORTED"
    if visa == "supported" and mvtec != "supported":
        return "STAGE_RANK_RISK_AUXILIARY_ONLY"
    if visa != "supported" and mvtec == "supported":
        return "STAGE_RANK_RISK_DOMAIN_SHIFT_EMERGENT"
    if visa == "null" and mvtec == "null":
        return "STAGE_RANK_RISK_NOT_SUPPORTED"
    return "STAGE_RANK_RISK_INCONCLUSIVE"


def audit_dataset(model, dataset_name: str, records_by_class: dict[str, list[dict[str, Any]]], datasets: dict[str, Any], config: dict[str, Any], output_dir: Path, device):
    per_class = []
    per_image = []
    text_cache: dict[str, torch.Tensor] = {}
    parity_errors = []
    for class_name in sorted(records_by_class):
        items = []
        class_records = records_by_class[class_name]
        for record in class_records:
            if isinstance(datasets, dict):
                raw = datasets[class_name][record["source_index"]]
            else:
                raw = datasets[record["source_index"]]
            item = predict_one(model, raw, dataset_name, class_name, int(config["img_size"]), text_cache, device)
            parity_errors.append(item["parity"])
            if item["parity"]["predictor_max_abs_probability_error"] > 1e-5:
                raise AuditImplementationInvalid(f"predictor parity max_abs={item['parity']['predictor_max_abs_probability_error']}")
            items.append(item)
        row = analyze_class(class_name, items, parity_errors[-len(items):])
        per_class.append(row)
        per_image.extend(row.pop("per_image"))
        del items
    parity = {
        "predictor_max_abs_probability_error": max(x["predictor_max_abs_probability_error"] for x in parity_errors),
        "ap_reconstruction_error": max(row["parity"]["ap_reconstruction_error"] for row in per_class),
        "auroc_reconstruction_error": max(row["parity"]["auroc_reconstruction_error"] for row in per_class),
    }
    if parity["ap_reconstruction_error"] > 1e-10 or parity["auroc_reconstruction_error"] > 1e-10:
        raise AuditImplementationInvalid(json.dumps(parity, sort_keys=True))
    provenance = {"dataset": dataset_name, "number_classes": len(per_class), "number_images": len(per_image), "precision": "strict FP32"}
    architecture = {
        "n_groups": int(model.n_groups), "image_levels": list(model.image_levels), "text_levels": list(model.text_levels),
        "dfg_mode": model.dfg_mode, "dfg_beta_runtime": float(model.dfg_beta), "use_ss2d_dfg": bool(model.use_ss2d_dfg), "dfg_ss2d_fusion": model.dfg_ss2d_fusion,
    }
    summary = dataset_aggregate(per_class, provenance, architecture, parity)
    decision = decision_for_dataset(summary)
    write_json(output_dir / "SUMMARY.json", summary)
    write_json(output_dir / "DECISION.json", decision)
    write_csv(output_dir / "PER_CLASS.csv", [_flatten_class_row(row) for row in per_class])
    write_csv(output_dir / "PER_IMAGE.csv", per_image)
    return summary, decision


def write_final_report(root: Path, terminal: str, summaries: dict[str, Any], decisions: dict[str, Any], architecture: dict[str, Any], commit_sha: str | None = None, remote_head: str | None = None):
    visa = summaries.get("VisA", {})
    mvtec = summaries.get("MVTEC", {})
    payload = {
        "decision": terminal,
        "integrity": {name: summary.get("parity") for name, summary in summaries.items()},
        "architecture": architecture,
        "stage_diversity": {name: summary.get("stage_diversity") for name, summary in summaries.items()},
        "visa": {"decision": decisions.get("VisA"), "signals": visa.get("signals")},
        "mvtec": {"decision": decisions.get("MVTEC"), "signals": mvtec.get("signals")},
        "orthogonal_to_confidence": "yes" if all(not decisions[name].get("confidence_redundant") for name in decisions) else "no",
        "selective_potential": {name: summary.get("signals", {}).get("D_logit", {}).get("oracle_ap_gain_at_20") for name, summary in summaries.items()},
        "zero_shot_replication": "yes" if decisions.get("VisA", {}).get("status") == "supported" and decisions.get("MVTEC", {}).get("status") == "supported" else "partial",
        "q1_potential": "high" if terminal == "STAGE_RANK_RISK_ZERO_SHOT_SUPPORTED" else ("low" if terminal in {"STAGE_RANK_RISK_NOT_SUPPORTED", "STAGE_INCONSISTENCY_CONFIDENCE_REDUNDANT"} else "medium"),
        "q2_potential": "high" if terminal in {"STAGE_RANK_RISK_AUXILIARY_ONLY", "STAGE_RANK_RISK_DOMAIN_SHIFT_EMERGENT"} else "medium",
        "commit_sha": commit_sha,
        "remote_head": remote_head,
    }
    write_json(root / "FINAL_DECISION.json", payload)
    lines = [
        f"DECISION: {terminal}",
        f"INTEGRITY: predictor={payload['integrity'].get('VisA', {}).get('predictor_max_abs_probability_error')} AP={payload['integrity'].get('VisA', {}).get('ap_reconstruction_error')} AUROC={payload['integrity'].get('VisA', {}).get('auroc_reconstruction_error')}",
        f"VISA: {decisions.get('VisA', {}).get('status')}",
        f"MVTEC: {decisions.get('MVTEC', {}).get('status')}",
        f"ORTHOGONALITY: {payload['orthogonal_to_confidence']}",
        f"Q1/Q2 POTENTIAL: {payload['q1_potential']}/{payload['q2_potential']}",
        f"COMMIT: {commit_sha or 'pending'}",
        f"REMOTE HEAD: {remote_head or 'pending'}",
    ]
    (root / "FINAL_DECISION.md").write_text("\n".join(lines) + "\n")
    return payload


def prepare(args):
    config = json.loads(args.config.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    configure_canonical_fp32()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = verified_load_model(config, args.checkpoint, device)
    architecture = build_architecture(model, config, checkpoint)
    paths = {"checkpoint": args.checkpoint, "config": args.config, "visa_manifest": args.visa_manifest, "mvtec_metadata": args.mvtec_meta}
    protocol = make_protocol(config, checkpoint, architecture, paths)
    write_json(args.output_root / "AUDIT_PROTOCOL.json", protocol)
    print(json.dumps({"status": "PROTOCOL_WRITTEN", "architecture": architecture}, sort_keys=True))


def audit(args):
    root = args.output_root
    protocol = json.loads((root / "AUDIT_PROTOCOL.json").read_text())
    unit = json.loads((root / "UNIT_TESTS.json").read_text())
    if unit.get("status") != "PASS":
        raise AuditImplementationInvalid("UNIT_TESTS.json is not PASS")
    config = json.loads(args.config.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    configure_canonical_fp32()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = verified_load_model(config, args.checkpoint, device)
    architecture = build_architecture(model, config, checkpoint)
    if architecture != protocol["architecture"]:
        raise ProtocolAssumptionInvalid("runtime architecture differs from frozen AUDIT_PROTOCOL.json")
    visa_manifest = json.loads(args.visa_manifest.read_text())
    visa_dataset = DeterministicVisATrainDataset(visa_manifest, int(config["img_size"]))
    visa_records = defaultdict(list)
    for index, sample in enumerate(visa_manifest["samples"]):
        visa_records[str(sample["class_name"])].append({"source_index": index, "file_name": sample["image_path"], "label": int(sample["label"])})
    visa_summary_path = root / "VISA" / "SUMMARY.json"
    visa_decision_path = root / "VISA" / "DECISION.json"
    if args.reuse_visa and visa_summary_path.is_file() and visa_decision_path.is_file():
        visa_summary = json.loads(visa_summary_path.read_text())
        visa_decision = json.loads(visa_decision_path.read_text())
    else:
        visa_summary, visa_decision = audit_dataset(model, "VisA", visa_records, visa_dataset, config, root / "VISA", device)
    print(json.dumps({"STATUS": "VisA HSIR audit complete", "RESULT": {"predictor_parity": visa_summary["parity"]["predictor_max_abs_probability_error"], "D_logit_damage_capture_auc": visa_summary["signals"]["D_logit"]["damage_capture_auc"]["median"], "D_rank_damage_capture_auc": visa_summary["signals"]["D_rank"]["damage_capture_auc"]["median"], "confidence_damage_capture_auc": visa_summary["signals"]["U_conf"]["damage_capture_auc"]["median"]}, "DECISION": visa_decision["status"], "NEXT": "freeze VisA and run MVTec"}, sort_keys=True))
    probe, mvtec_datasets = prepare_mvtec_probe(args.mvtec_meta, root / "MVTEC" / "PROBE_MANIFEST.json", int(config["img_size"]))
    mvtec_records = defaultdict(list)
    for sample in probe["samples"]:
        mvtec_records[sample["class_name"]].append(sample)
    mvtec_summary, mvtec_decision = audit_dataset(model, "MVTec", mvtec_records, mvtec_datasets, config, root / "MVTEC", device)
    summaries = {"VisA": visa_summary, "MVTEC": mvtec_summary}
    decisions = {"VisA": visa_decision, "MVTEC": mvtec_decision}
    terminal = final_terminal(visa_decision, mvtec_decision, summaries)
    write_final_report(root, terminal, summaries, decisions, architecture)
    print(json.dumps({"STATUS": "MVTec HSIR audit complete", "RESULT": {"D_logit_damage_capture_auc": mvtec_summary["signals"]["D_logit"]["damage_capture_auc"]["median"], "D_rank_damage_capture_auc": mvtec_summary["signals"]["D_rank"]["damage_capture_auc"]["median"], "confidence_damage_capture_auc": mvtec_summary["signals"]["U_conf"]["damage_capture_auc"]["median"]}, "DECISION": terminal, "NEXT": "freeze final decision"}, sort_keys=True))


def write_failure(args, terminal: str, error: Exception):
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {"decision": terminal, "error": {"type": type(error).__name__, "message": str(error)}}
    write_json(args.output_root / "FINAL_DECISION.json", payload)
    (args.output_root / "FINAL_DECISION.md").write_text(f"DECISION: {terminal}\nINTEGRITY: not established\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "audit"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--visa-manifest", type=Path, default=VISA_MANIFEST)
    parser.add_argument("--mvtec-meta", type=Path, default=MVTEC_META)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--reuse-visa", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.mode == "prepare":
            prepare(args)
        else:
            audit(args)
    except FileNotFoundError as error:
        write_json(args.output_root / "MVTEC" / "INPUT_BLOCKER.json", {"status": "MVTec input missing", "error": str(error)})
        print(f"BLOCKED_INPUT: {error}")
        raise SystemExit(3)
    except ProtocolAssumptionInvalid as error:
        write_failure(args, "PROTOCOL_ASSUMPTION_INVALID", error)
        print(f"DECISION: PROTOCOL_ASSUMPTION_INVALID: {error}")
        raise SystemExit(2)
    except AuditImplementationInvalid as error:
        write_failure(args, "AUDIT_IMPLEMENTATION_INVALID", error)
        print(f"DECISION: AUDIT_IMPLEMENTATION_INVALID: {error}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
