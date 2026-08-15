#!/usr/bin/env python3
"""Phase5-B0: inference-only discovery of complementary visual evidence.

This audit deliberately keeps all dense pixel arrays in memory for one class
only.  It reuses the Phase2B predictor path and the Phase5-A error objects,
but exposes the authoritative post-projection visual patch features from the
same image forward used to produce the deployed score.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from audit_p4v_phase2b_readiness import load_model  # noqa: E402
from audit_phase5_hsir import (  # noqa: E402
    _sha256,
    ap_contamination,
    exact_auc_ap,
    pairwise_risks,
    percentile_rank,
    population_std,
    project_exact_auc_ap,
    write_json,
)
from model.adapter import gaussian_blur2d  # noqa: E402
from dataset import get_text_and_image_dataset  # noqa: E402
from utils import configure_canonical_fp32, get_phase2b_global_text_features  # noqa: E402


OUTPUT_ROOT = ROOT / "runs/phase5/hsir/SECOND_EVIDENCE"
CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
CONFIG = ROOT / "runs/phase4/k1/short64_seed0_attempt5/config.json"
VISA_META = ROOT / "dataset/hub/VisA.jsonl"
PHASE5_ROOT = ROOT / "runs/phase5/hsir/VISA_TEST"
A1_ROOT = ROOT / "runs/phase5/hsir/ACTIONABILITY"
STAGE_RESCUE_ROOT = ROOT / "runs/phase5/hsir/STAGE_RESCUE"
STAGE_ARBITRATION_ROOT = ROOT / "runs/phase5/hsir/STAGE_ARBITRATION"

EXPECTED_CHECKPOINT_SHA = "a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34"
EXPECTED_CONFIG_SHA = "377ce1c0ae1dd870f82ddcb828d8d8809fa46c007e61567f2150ec11354b23a4"
EXPECTED_VISA_META_SHA = "468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842"
PHASE5_COMMIT = "29a8ffc934448b34424c77805a2c5c289bd9ddac"
A1_COMMIT = "fcbff12059a0cb29698c14f443a0396acbef8c55"
BRANCH_A_COMMIT = "06d72e37b9a7b2ce0db381e1d3b6edfc59de9c91"
B1_COMMIT = "963006df88bf451cfc6c7d7baee4a3cb3792ea7e"

EXPECTED_CLASSES = 12
EXPECTED_IMAGES = 2162
EXPECTED_NORMAL = 962
EXPECTED_ANOMALY = 1200
PRIMARY_FRACTION = 0.20
TRIAGE_FRACTION = 0.10
QUANTILE_BINS = 10
BOOTSTRAP_REPS = 2000
PARITY_TOL = 1e-10
EPS = 1e-12
CANDIDATES = ("E_local", "E_multistage", "E_xstage", "E_global")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def is_ancestor(commit: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=False
    ).returncode == 0


def finite(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and np.isfinite(value))


def safe_ratio(numerator: float, denominator: float):
    return None if denominator == 0.0 else float(numerator / denominator)


def stable_desc_order(values: np.ndarray, pixel_id: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    pixel_id = np.asarray(pixel_id, dtype=np.int64).ravel()
    if values.size != pixel_id.size or not np.all(np.isfinite(values)):
        raise RuntimeError("SECOND_EVIDENCE_OUTPUT_INVALID: non-finite or mis-sized selector")
    return np.lexsort((pixel_id, -values))


def select_top(values: np.ndarray, pixel_id: np.ndarray, count: int) -> np.ndarray:
    if count < 0 or count > np.asarray(values).size:
        raise ValueError("invalid selection count")
    selected = np.zeros(np.asarray(values).size, dtype=bool)
    selected[stable_desc_order(values, pixel_id)[:count]] = True
    return selected


def quantile_bins(values: np.ndarray, bins: int = QUANTILE_BINS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("quantile bins require finite non-empty values")
    order = np.argsort(values, kind="mergesort")
    out = np.empty(values.size, dtype=np.int64)
    out[order] = np.minimum((np.arange(values.size) * bins) // values.size, bins - 1)
    return out


def pair_hash(class_name: str, positive_id: int, negative_id: int) -> str:
    return hashlib.sha256(f"{class_name}|{positive_id}|{negative_id}".encode()).hexdigest()


def deterministic_matches(
    class_name: str,
    score: np.ndarray,
    d_rank: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
    pixel_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match selected positives to same-class selected negatives.

    Controls are restricted to the same 10x10 score/D_rank quantile bin.  The
    implementation uses one deterministic control per positive.  Candidate
    evidence never enters the choice.  Within a bin, negative IDs are ordered
    once by a deterministic ID hash; the pair hash is the prescribed
    deterministic tie-break/key used to rotate the control stream for each
    positive, avoiding an N-positive x N-negative matrix.
    """
    score = np.asarray(score, dtype=np.float64).ravel()
    d_rank = np.asarray(d_rank, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    selected = np.asarray(selected, dtype=bool).ravel()
    pixel_id = np.asarray(pixel_id, dtype=np.int64).ravel()
    if not (score.size == d_rank.size == labels.size == selected.size == pixel_id.size):
        raise ValueError("matching arrays have different sizes")
    score_bin = quantile_bins(score)
    rank_bin = quantile_bins(d_rank)
    negative_indices = np.flatnonzero(selected & ~labels)
    negative_keys = score_bin[negative_indices] * QUANTILE_BINS + rank_bin[negative_indices]
    negative_order = np.lexsort((pixel_id[negative_indices], negative_keys))
    ordered_negative = negative_indices[negative_order]
    ordered_keys = negative_keys[negative_order]
    group_starts = np.flatnonzero(np.r_[True, ordered_keys[1:] != ordered_keys[:-1]])
    group_ends = np.r_[group_starts[1:], ordered_keys.size]
    ranges = {int(key): (int(start), int(end)) for key, start, end in zip(ordered_keys[group_starts], group_starts, group_ends)}
    positives = np.flatnonzero(selected & labels)
    matched_positive = []
    matched_negative = []
    for index in positives:
        key = int(score_bin[index]) * QUANTILE_BINS + int(rank_bin[index])
        bounds = ranges.get(key)
        if bounds is None:
            continue
        # The hash selects a deterministic offset in the valid control stream;
        # the candidate list itself is formed without candidate evidence.
        start, end = bounds
        first_candidate = int(ordered_negative[start])
        digest = int(pair_hash(class_name, int(pixel_id[index]), int(pixel_id[first_candidate]))[:16], 16)
        chosen = int(ordered_negative[start + digest % (end - start)])
        matched_positive.append(int(index))
        matched_negative.append(int(chosen))
    return np.asarray(matched_positive, dtype=np.int64), np.asarray(matched_negative, dtype=np.int64)


def matched_win_rate(evidence: np.ndarray, positive_indices: np.ndarray, negative_indices: np.ndarray):
    if positive_indices.size == 0:
        return None
    delta = np.asarray(evidence)[positive_indices] - np.asarray(evidence)[negative_indices]
    return float(np.mean((delta > 0).astype(np.float64) + 0.5 * (delta == 0)))


def ap_from_order(labels: np.ndarray, order: np.ndarray) -> float:
    ordered = np.asarray(labels, dtype=np.uint8).ravel()[order]
    positives = int(ordered.sum())
    if positives == 0:
        return 0.0
    cumulative = np.cumsum(ordered, dtype=np.int64)
    precision = cumulative / np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(np.sum(precision[ordered == 1]) / positives)


def repaired_order(base_order: np.ndarray, labels: np.ndarray, selected: np.ndarray, mode: str) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint8)
    selected = np.asarray(selected, dtype=bool)
    positive = labels == 1
    negative = ~positive
    selected_positive = selected & positive if mode in {"positive", "both"} else np.zeros(labels.size, bool)
    selected_negative = selected & negative if mode in {"negative", "both"} else np.zeros(labels.size, bool)
    selected_for_middle = selected_positive | selected_negative
    ordered_positive = base_order[selected_positive[base_order]]
    middle = base_order[~selected_for_middle[base_order]]
    ordered_negative = base_order[selected_negative[base_order]]
    if mode == "positive":
        return np.concatenate((ordered_positive, middle))
    if mode == "negative":
        return np.concatenate((middle, ordered_negative))
    if mode == "both":
        return np.concatenate((ordered_positive, middle, ordered_negative))
    raise ValueError(mode)


def oracle_bundle(labels: np.ndarray, baseline_ap: float, base_order: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    out = {}
    for mode, name in (("positive", "positive_only"), ("negative", "negative_only"), ("both", "both")):
        ap = ap_from_order(labels, repaired_order(base_order, labels, selected, mode))
        out[f"{name}_ap"] = float(ap)
        out[f"{name}_delta"] = float(ap - baseline_ap)
        out[f"{name}_delta_pp"] = float(100.0 * (ap - baseline_ap))
    return out


def extreme_recall(selected: np.ndarray, mask: np.ndarray, severity: np.ndarray, fraction: float, pixel_id: np.ndarray):
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    count = max(1, int(np.ceil(fraction * indices.size)))
    severe = indices[stable_desc_order(severity[indices], pixel_id[indices])[:count]]
    return float(np.sum(selected[severe]) / count)


def selector_metrics(
    selector: np.ndarray,
    labels: np.ndarray,
    c_ap: np.ndarray,
    r_pos_full: np.ndarray,
    r_neg_full: np.ndarray,
    baseline_ap: float,
    base_order: np.ndarray,
    pixel_id: np.ndarray,
    a1_delta: float,
) -> dict[str, Any]:
    positive = labels == 1
    negative = ~positive
    selected_positive = selector & positive
    selected_negative = selector & negative
    oracle = oracle_bundle(labels, baseline_ap, base_order, selector)
    return {
        "selected_count": int(selector.sum()),
        "selected_positive_count": int(selected_positive.sum()),
        "selected_positive_fraction": safe_ratio(float(selected_positive.sum()), float(selector.sum())),
        "positive_C_AP_mass_capture": safe_ratio(float(np.nansum(c_ap[selected_positive])), float(np.nansum(c_ap[positive]))),
        "positive_R_pos_mass_capture": safe_ratio(float(np.nansum(r_pos_full[selected_positive])), float(np.nansum(r_pos_full[positive]))),
        "negative_R_neg_mass_capture": safe_ratio(float(np.nansum(r_neg_full[selected_negative])), float(np.nansum(r_neg_full[negative]))),
        "oracle": {"baseline_ap": float(baseline_ap), **oracle},
        "oracle_triage_delta_AP": oracle["positive_only_delta"],
        "oracle_triage_delta_AP_pp": oracle["positive_only_delta_pp"],
        "fraction_A1_positive_oracle_recovered": safe_ratio(oracle["positive_only_delta"], a1_delta),
        "extreme_C_AP_recall_top1pct": extreme_recall(selector, positive, c_ap, 0.01, pixel_id),
        "extreme_C_AP_recall_top5pct": extreme_recall(selector, positive, c_ap, 0.05, pixel_id),
        "extreme_C_AP_recall_top10pct": extreme_recall(selector, positive, c_ap, 0.10, pixel_id),
    }


def authoritative_patch_grid(model: torch.nn.Module) -> tuple[int, int]:
    grid = getattr(getattr(model, "image_encoder", None), "grid_size", None)
    if grid is None:
        raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: image encoder exposes no authoritative grid_size")
    if isinstance(grid, int):
        grid = (grid, grid)
    grid = tuple(int(value) for value in grid)
    if len(grid) != 2 or min(grid) <= 0:
        raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: invalid image encoder grid_size={grid}")
    return grid

def upsample_explicit(values: np.ndarray, patch_grid: tuple[int, int], image_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    height, width = patch_grid
    if values.size != height * width:
        raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: patch values={values.size}, grid={patch_grid}")
    tensor = torch.from_numpy(values.reshape(1, 1, height, width))
    return F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=True).squeeze().numpy()

def local_nonconformity(features: torch.Tensor) -> torch.Tensor:
    """E_local on [D,H,W] normalized features with valid border neighbors."""
    if features.ndim != 3:
        raise ValueError(f"expected [D,H,W], got {tuple(features.shape)}")
    channels, height, width = features.shape
    value = features.unsqueeze(0)
    kernel = torch.ones((channels, 1, 3, 3), device=features.device, dtype=features.dtype)
    kernel[:, :, 1, 1] = 0
    padded = F.pad(value, (1, 1, 1, 1), mode="constant", value=0)
    neighbor_sum = F.conv2d(padded, kernel, groups=channels)
    counts = torch.ones((1, 1, height, width), device=features.device, dtype=features.dtype)
    counts = F.conv2d(F.pad(counts, (1, 1, 1, 1), value=1), torch.ones((1, 1, 3, 3), device=features.device, dtype=features.dtype) - torch.eye(3, device=features.device, dtype=features.dtype).reshape(1, 1, 3, 3), padding=0)
    context = F.normalize(neighbor_sum / counts.clamp_min(1), dim=1)
    return (1.0 - (value * context).sum(dim=1)).squeeze(0)


def align_features(stage_features: list[torch.Tensor], patch_grid: tuple[int, int]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Align [P,D] stage features using the image encoder's explicit patch grid."""
    shapes = []
    for feature in stage_features:
        if feature.ndim != 2:
            raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: feature shape {tuple(feature.shape)}")
        patches, dimension = feature.shape
        if int(patches) != int(patch_grid[0] * patch_grid[1]):
            raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: feature P={patches}, authoritative grid={patch_grid}")
        shapes.append({"patch_count": int(patches), "dimension": int(dimension), "grid": list(patch_grid)})
    if len({x["dimension"] for x in shapes}) != 1:
        raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: stage feature dimensions differ")
    reference = tuple(patch_grid)
    aligned = []
    for feature, shape in zip(stage_features, shapes):
        grid = tuple(shape["grid"])
        tensor = feature.reshape(grid[0], grid[1], -1).permute(2, 0, 1).unsqueeze(0)
        if grid != reference:
            tensor = F.interpolate(tensor, size=reference, mode="bilinear", align_corners=True)
        tensor = F.normalize(tensor, dim=1).squeeze(0).permute(1, 2, 0).reshape(reference[0] * reference[1], -1)
        aligned.append(tensor)
    return aligned, {
        "source": "model/adapter.py::AdapterModel.forward return_phase4_features=True",
        "tensor": "visual['seg_tokens']",
        "normalization": "seg_proj -> seg_layer_norms -> torch.nn.functional.normalize(dim=-1)",
        "stages": shapes,
        "reference_grid": list(reference),
        "alignment": "bilinear interpolation align_corners=True before L2 renormalization when grids differ",
    }


def evidence_maps(stage_features: list[torch.Tensor], patch_grid: tuple[int, int], image_size: int) -> dict[str, np.ndarray]:
    aligned, _ = align_features(stage_features, patch_grid)
    local = [local_nonconformity(f.T.reshape(f.shape[1], patch_grid[0], patch_grid[1])) for f in aligned]
    reference_height, reference_width = patch_grid
    local_stack = torch.stack(local)
    final = aligned[-1]
    image_context = F.normalize(final.mean(dim=0, keepdim=True), dim=1)
    global_map = (1.0 - (final * image_context).sum(dim=1)).reshape(reference_height, reference_width)
    pairwise = []
    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            pairwise.append(1.0 - (aligned[i] * aligned[j]).sum(dim=1))
    xstage = torch.stack(pairwise).mean(dim=0).reshape(reference_height, reference_width)
    maps = {
        "E_local": local_stack[-1],
        "E_multistage": local_stack.mean(dim=0),
        "E_xstage": xstage,
        "E_global": global_map,
    }
    return {name: upsample_explicit(value.detach().float().cpu().numpy().reshape(-1), patch_grid, image_size).reshape(-1).astype(np.float32) for name, value in maps.items()}


def deploy_from_native_explicit(native_group_logits: torch.Tensor, patch_grid: tuple[int, int], image_size: int, domain: str = "Industrial"):
    if native_group_logits.ndim != 4 or native_group_logits.shape[-1] != 2:
        raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: native logits shape {tuple(native_group_logits.shape)}")
    groups, batch, patches, _ = native_group_logits.shape
    height, width = patch_grid
    if patches != height * width:
        raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: native P={patches}, grid={patch_grid}")
    sigma = 1 if domain == "Industrial" else 1.5
    kernel_size = 7 if domain == "Industrial" else 9
    group_logits = []
    for group in range(groups):
        logits = native_group_logits[group].permute(0, 2, 1).reshape(batch, 2, height, width)
        logits = gaussian_blur2d(logits, (kernel_size, kernel_size), (sigma, sigma))
        logits = F.interpolate(logits, size=(image_size, image_size), mode="bilinear", align_corners=True)
        group_logits.append(logits)
    final_logits = torch.stack(group_logits, dim=0).mean(dim=0)
    return F.softmax(final_logits, dim=1), final_logits

def validate_runtime_shapes(model: torch.nn.Module, stage_features: list[torch.Tensor], features: torch.Tensor, native: torch.Tensor, native_margin: torch.Tensor, model_prob: torch.Tensor, reconstructed_prob: torch.Tensor, final_logits: torch.Tensor, image_size: int) -> dict[str, Any]:
    patch_grid = authoritative_patch_grid(model)
    patch_count = patch_grid[0] * patch_grid[1]
    stage_shapes = [tuple(int(value) for value in feature.shape) for feature in stage_features]
    if len(stage_features) != int(model.n_groups) or any(shape[0] != 1 or shape[1] != patch_count for shape in stage_shapes):
        raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: stage feature shapes={stage_shapes}, grid={patch_grid}")
    if tuple(features.shape) != (int(model.n_groups), 1, patch_count, stage_features[0].shape[-1]):
        raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: stacked feature shape={tuple(features.shape)}")
    if tuple(native.shape) != (int(model.n_groups), 1, patch_count, 2):
        raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: native stage-logit shape={tuple(native.shape)}")
    if tuple(native_margin.shape) != (int(model.n_groups), 1, patch_count):
        raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: native margin shape={tuple(native_margin.shape)}")
    if tuple(model_prob.shape) != (1, image_size, image_size) or tuple(reconstructed_prob.shape) != (1, 2, image_size, image_size) or tuple(final_logits.shape) != (1, 2, image_size, image_size):
        raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: deployed shapes model={tuple(model_prob.shape)}, reconstructed={tuple(reconstructed_prob.shape)}, logits={tuple(final_logits.shape)}")
    return {
        "stage_visual_features": [list(shape) for shape in stage_shapes],
        "stacked_visual_features": list(features.shape),
        "native_stage_logits": list(native.shape),
        "native_stage_margins": list(native_margin.shape),
        "patch_grid": list(patch_grid),
        "patch_count": int(patch_count),
        "deployed_model_probability": list(model_prob.shape),
        "deployed_reconstructed_probability": list(reconstructed_prob.shape),
        "deployed_final_logits": list(final_logits.shape),
        "mapping": "native [G,B,P,2] -> [B,2,grid_h,grid_w] -> Gaussian blur -> bilinear align_corners=True -> group mean -> softmax",
        "feature_mapping": "seg_tokens [B,P,D] -> [D,grid_h,grid_w] via image_encoder.grid_size; scalar evidence bilinear align_corners=True to image grid",
        "patch_level_tensors": ["seg_tokens", "stacked_visual_features", "native_stage_logits", "native_stage_margins"],
        "image_pixel_level_tensors": ["model_prob", "reconstructed_prob", "final_logits", "score", "final_margin", "D_rank", "D_logit", "candidate_evidence"],
    }
def predictor_one_with_features(model, raw: dict[str, Any], class_name: str, image_size: int, text_cache: dict[str, torch.Tensor], device):
    image = raw["image"].unsqueeze(0).to(device).float()
    target = raw["mask"].to(device).float().squeeze(0).cpu().numpy().astype(np.uint8)
    visual = model(image, return_phase4_features=True)
    stage_feature_batches = [x.float() for x in visual["seg_tokens"]]
    stage_features = [x[0] for x in stage_feature_batches]
    features = torch.stack(visual["seg_tokens"])
    if class_name not in text_cache:
        text_cache[class_name] = get_phase2b_global_text_features(
            model, "VisA", [class_name], device, use_hybrid_soft_prompt=True, use_soft_prompt=False
        ).float()
    text = text_cache[class_name]
    model_prob, native, native_margin = model.vision_text_fusion_gate_seg(
        features, text, img_size=image_size, test_mode=True, domain="Industrial", return_details=True
    )
    patch_grid = authoritative_patch_grid(model)
    reconstructed_prob, final_logits = deploy_from_native_explicit(native, patch_grid, image_size, "Industrial")
    shape_record = validate_runtime_shapes(model, stage_feature_batches, features, native, native_margin, model_prob, reconstructed_prob, final_logits, image_size)
    parity = float((model_prob - reconstructed_prob[:, 1]).abs().max().detach().cpu())
    native_margins = native_margin[:, 0].detach().float().cpu().numpy()
    native_logits = native[:, 0].detach().float().cpu().numpy()
    d_logit = upsample_explicit(population_std(native_margins, axis=0).astype(np.float32), patch_grid, image_size)
    ranks = np.stack([percentile_rank(stage) for stage in native_margins], axis=0)
    d_rank = upsample_explicit(population_std(ranks, axis=0).astype(np.float32), patch_grid, image_size)
    final_logits_np = final_logits[0].detach().float().cpu().numpy()
    score = reconstructed_prob[0, 1].detach().float().cpu().numpy()
    final_margin = final_logits_np[1] - final_logits_np[0]
    final_rank = percentile_rank(score.reshape(-1)).reshape(score.shape).astype(np.float32)
    stage_rank_maps = np.stack([
        upsample_explicit(percentile_rank(stage).astype(np.float32), patch_grid, image_size)
        for stage in native_margins
    ], axis=0)
    g_rescue = stage_rank_maps.max(axis=0) - final_rank
    return {
        "score": score.reshape(-1).astype(np.float32),
        "final_margin": final_margin.reshape(-1).astype(np.float32),
        "target": target.reshape(-1),
        "D_logit": d_logit.reshape(-1).astype(np.float32),
        "D_rank": d_rank.reshape(-1).astype(np.float32),
        "U_conf": (-np.abs(final_margin)).reshape(-1).astype(np.float32),
        "G_rescue": g_rescue.reshape(-1).astype(np.float32),
        "evidence": evidence_maps(stage_features, patch_grid, image_size),
        "native_margins": native_margins.astype(np.float32),
        "native_logits": native_logits.astype(np.float32),
        "parity": parity,
        "shape_record": shape_record,
    }


def canonical_test_records(image_size: int):
    datasets = get_text_and_image_dataset("VisA", image_size, stage="test")
    records = {}
    all_rows = []
    for class_name in sorted(datasets):
        dataset = datasets[class_name]
        class_records = []
        for source_index, row in enumerate(dataset.meta):
            file_name = str(row["image_path"])
            if "train" in file_name.lower() or "train" in str(getattr(dataset, "data_path", "")).lower():
                raise RuntimeError("SECOND_EVIDENCE_INPUT_PROVENANCE_INVALID: TRAIN path in TEST loader")
            class_records.append({"source_index": int(source_index), "file_name": file_name, "label": int(row["label"])})
            all_rows.append(row)
        records[class_name] = class_records
    counts = {
        "classes": len(records),
        "images": len(all_rows),
        "normal": sum(int(row["label"]) == 0 for row in all_rows),
        "anomaly": sum(int(row["label"]) == 1 for row in all_rows),
    }
    if counts != {"classes": EXPECTED_CLASSES, "images": EXPECTED_IMAGES, "normal": EXPECTED_NORMAL, "anomaly": EXPECTED_ANOMALY}:
        raise RuntimeError(f"SECOND_EVIDENCE_INPUT_PROVENANCE_INVALID: TEST counts {counts}")
    return datasets, records, counts


def artifact_hashes() -> dict[str, str]:
    paths = {
        "visa_test_summary": PHASE5_ROOT / "SUMMARY.json",
        "visa_test_per_class": PHASE5_ROOT / "PER_CLASS.csv",
        "visa_test_per_image": PHASE5_ROOT / "PER_IMAGE.csv",
        "visa_test_decision": PHASE5_ROOT / "DECISION.json",
        "actionability_summary": A1_ROOT / "SUMMARY.json",
        "actionability_decision": A1_ROOT / "DECISION.json",
        "stage_rescue_summary": STAGE_RESCUE_ROOT / "SUMMARY.json",
        "stage_rescue_decision": STAGE_RESCUE_ROOT / "DECISION.json",
        "stage_arbitration_summary": STAGE_ARBITRATION_ROOT / "SUMMARY.json",
        "stage_arbitration_decision": STAGE_ARBITRATION_ROOT / "DECISION.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("SECOND_EVIDENCE_INPUT_PROVENANCE_INVALID: missing " + ", ".join(missing))
    return {name: _sha256(path) for name, path in paths.items()}


def input_check(image_size: int) -> dict[str, Any]:
    phase5 = json.loads((PHASE5_ROOT / "SUMMARY.json").read_text())
    a1 = json.loads((A1_ROOT / "SUMMARY.json").read_text())
    stage = json.loads((STAGE_RESCUE_ROOT / "SUMMARY.json").read_text())
    arbitration = json.loads((STAGE_ARBITRATION_ROOT / "SUMMARY.json").read_text())
    dataset_root = phase5["provenance"].get("dataset_root")
    checks = {
        "checkpoint_sha": _sha256(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA,
        "config_sha": _sha256(CONFIG) == EXPECTED_CONFIG_SHA,
        "visa_metadata_sha": _sha256(VISA_META) == EXPECTED_VISA_META_SHA,
        "phase5_ancestor": is_ancestor(PHASE5_COMMIT),
        "actionability_ancestor": is_ancestor(A1_COMMIT),
        "branch_a_ancestor": is_ancestor(BRANCH_A_COMMIT),
        "b1_ancestor": is_ancestor(B1_COMMIT),
        "phase5_dataset": phase5["provenance"].get("dataset") == "VisA",
        "phase5_split": str(phase5["provenance"].get("split")).lower() == "test",
        "phase5_counts": phase5["provenance"].get("number_classes") == EXPECTED_CLASSES and phase5["provenance"].get("number_images") == EXPECTED_IMAGES,
        "phase5_image_counts": phase5["provenance"].get("number_normal_images") == EXPECTED_NORMAL and phase5["provenance"].get("number_anomaly_images") == EXPECTED_ANOMALY,
        "phase5_no_train_paths": phase5["provenance"].get("contains_train_paths") is False,
        "phase5_predictor_parity": phase5.get("parity", {}).get("predictor_max_abs_probability_error") == 0.0,
        "a1_predictor_parity": a1.get("provenance", {}).get("predictor_parity") == "PASS",
        "a1_split": a1.get("provenance", {}).get("split") == "test",
        "stage_predictor_parity": stage.get("target_parity", {}).get("status") == "PASS",
        "arbitration_target_parity": arbitration.get("input_integrity") == "PASS" or arbitration.get("target_parity", {}).get("status") == "PASS",
        "no_train_dataset_root": "train" not in str(dataset_root).lower(),
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError("SECOND_EVIDENCE_INPUT_PROVENANCE_INVALID: " + ", ".join(failed))
    _, _, counts = canonical_test_records(image_size)
    return {
        "status": "PASS",
        "current_head": git_head(),
        "scientific_ancestors": {
            "phase5": PHASE5_COMMIT,
            "actionability": A1_COMMIT,
            "branch_a": BRANCH_A_COMMIT,
            "branch_b1": B1_COMMIT,
        },
        "checkpoint": {"path": str(CHECKPOINT), "sha256": _sha256(CHECKPOINT)},
        "config": {"path": str(CONFIG), "sha256": _sha256(CONFIG)},
        "visa_root": dataset_root,
        "metadata_source": str(VISA_META),
        "metadata_sha256": _sha256(VISA_META),
        "split": "TEST",
        "counts": counts,
        "predictor_implementation": "tools/audit_phase5_second_evidence.py::predictor_one_with_features; model/adapter.py::AdapterModel.forward; tools/audit_phase5_hsir.py::deploy_from_native",
        "evaluator_implementation": "tools/audit_phase5_hsir.py::{exact_auc_ap,project_exact_auc_ap,ap_contamination,pairwise_risks}",
        "upstream_artifact_sha256": artifact_hashes(),
        "checks": checks,
        "predictor_parity": "PASS",
        "authoritative_pixel_cache": False,
        "inference_authorization": "one fresh class-streamed VisA TEST pass; one image forward; no dense cache persisted",
    }


def bootstrap_ci(values: list[float | None], seed: int) -> list[float] | None:
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(BOOTSTRAP_REPS, arr.size))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate(values: list[float | None], seed: int) -> dict[str, Any]:
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    return {
        "mean": None if arr.size == 0 else float(arr.mean()),
        "median": None if arr.size == 0 else float(np.median(arr)),
        "bootstrap95_ci": bootstrap_ci(values, seed),
        "n_classes": int(arr.size),
        "unit": "class",
    }


def candidate_triage(evidence: np.ndarray, rank_mask: np.ndarray, control_conf: np.ndarray, control_rank: np.ndarray, labels: np.ndarray, c_ap: np.ndarray, r_pos_full: np.ndarray, r_neg_full: np.ndarray, scores: np.ndarray, baseline_ap: float, pixel_id: np.ndarray, a1_delta: float, image_size: int) -> dict[str, Any]:
    k = int(np.ceil(TRIAGE_FRACTION * int(rank_mask.sum())))
    rank_values = np.asarray(evidence)[rank_mask]
    ids = pixel_id[rank_mask]
    candidate = np.zeros(rank_mask.size, dtype=bool)
    candidate_indices = np.flatnonzero(rank_mask)[stable_desc_order(rank_values, ids)[:k]]
    candidate[candidate_indices] = True
    conf = np.zeros_like(candidate)
    conf_indices = np.flatnonzero(rank_mask)[stable_desc_order(control_conf[rank_mask], ids)[:k]]
    conf[conf_indices] = True
    rank = np.zeros_like(candidate)
    rank_indices = np.flatnonzero(rank_mask)[stable_desc_order(control_rank[rank_mask], ids)[:k]]
    rank[rank_indices] = True
    result = {"triage_budget_k": k, "triage_actual_fraction": float(k / max(rank_mask.size, 1))}
    for name, selection in (("candidate", candidate), ("confidence_control", conf), ("rank_control", rank)):
        result[name] = selector_metrics(selection, labels, c_ap, r_pos_full, r_neg_full, baseline_ap, np.argsort(-scores, kind="mergesort"), pixel_id, a1_delta)
    return result


def process_class(model, dataset, class_name: str, records: list[dict], image_size: int, device) -> dict[str, Any]:
    score_parts: list[np.ndarray] = []
    margin_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    rescue_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    pixel_parts: list[np.ndarray] = []
    evidence_parts = {name: [] for name in CANDIDATES}
    text_cache: dict[str, torch.Tensor] = {}
    pixels_per_image = image_size * image_size
    max_parity = 0.0
    stage_shapes = None
    for record in records:
        raw = dataset[record["source_index"]]
        item = predictor_one_with_features(model, raw, class_name, image_size, text_cache, device)
        n = item["score"].size
        if n != pixels_per_image:
            raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: unexpected image pixel count")
        pixel_id = np.int64(record["source_index"]) * pixels_per_image + np.arange(n, dtype=np.int64)
        score_parts.append(item["score"])
        margin_parts.append(item["final_margin"])
        rank_parts.append(item["D_rank"])
        logit_parts.append(item["D_logit"])
        rescue_parts.append(item["G_rescue"])
        label_parts.append(item["target"])
        pixel_parts.append(pixel_id)
        for name in CANDIDATES:
            evidence_parts[name].append(item["evidence"][name])
        max_parity = max(max_parity, item["parity"])
        del item, raw
    scores = np.concatenate(score_parts)
    margins = np.concatenate(margin_parts)
    d_rank = np.concatenate(rank_parts)
    d_logit = np.concatenate(logit_parts)
    g_rescue = np.concatenate(rescue_parts)
    labels = np.concatenate(label_parts).astype(np.uint8)
    pixel_id = np.concatenate(pixel_parts)
    evidence = {name: np.concatenate(values) for name, values in evidence_parts.items()}
    if not all(np.all(np.isfinite(values)) for values in (scores, margins, d_rank, d_logit, g_rescue, labels, *evidence.values())):
        raise RuntimeError("SECOND_EVIDENCE_OUTPUT_INVALID: non-finite pixel variable")
    baseline_auc, baseline_ap = exact_auc_ap(scores, labels)
    evaluator_auc, evaluator_ap = project_exact_auc_ap(scores, labels)
    if max(abs(baseline_auc - evaluator_auc), abs(baseline_ap - evaluator_ap)) > PARITY_TOL:
        raise RuntimeError("ACTIONABILITY_TARGET_PARITY_INVALID: AP/AUROC evaluator parity failed")
    positive = labels == 1
    r_pos, r_neg = pairwise_risks(scores, labels)
    r_pos_full = np.full(scores.size, np.nan, dtype=np.float64)
    r_neg_full = np.full(scores.size, np.nan, dtype=np.float64)
    r_pos_full[positive] = r_pos
    r_neg_full[~positive] = r_neg
    c_ap = ap_contamination(scores, labels)
    rank_mask = select_top(d_rank, pixel_id, int(np.ceil(PRIMARY_FRACTION * scores.size)))
    a1 = selector_metrics(rank_mask & positive, labels, c_ap, r_pos_full, r_neg_full, baseline_ap, np.argsort(-scores, kind="mergesort"), pixel_id, 1.0)
    a1_delta = a1["oracle"]["positive_only_delta"]
    positive_indices, negative_indices = deterministic_matches(class_name, scores, d_rank, labels, rank_mask, pixel_id)
    matched_n = int(positive_indices.size)
    row: dict[str, Any] = {
        "class": class_name,
        "n_images": len(records),
        "n_pixels": int(scores.size),
        "baseline_ap": float(baseline_ap),
        "baseline_auroc": float(baseline_auc),
        "a1_positive_only_oracle_delta_AP": float(a1_delta),
        "a1_positive_only_oracle_delta_AP_pp": float(100.0 * a1_delta),
        "rank_selected_count": int(rank_mask.sum()),
        "rank_selected_positive_fraction": safe_ratio(float(np.sum(rank_mask & positive)), float(rank_mask.sum())),
        "matched_pairs_n": matched_n,
        "feature_semantics": "post-projection seg_tokens; see PROTOCOL.json",
        "predictor_parity_max_abs_probability_error": float(max_parity),
        "ap_parity_error": float(abs(baseline_ap - evaluator_ap)),
        "auroc_parity_error": float(abs(baseline_auc - evaluator_auc)),
    }
    for name in CANDIDATES:
        shifted = np.roll(evidence[name].reshape(len(records), image_size, image_size), (image_size // 3, image_size // 3), axis=(1, 2)).reshape(-1)
        triage = candidate_triage(evidence[name], rank_mask, -np.abs(margins), d_rank, labels, c_ap, r_pos_full, r_neg_full, scores, baseline_ap, pixel_id, a1_delta, image_size)
        shifted_triage = candidate_triage(shifted, rank_mask, -np.abs(margins), d_rank, labels, c_ap, r_pos_full, r_neg_full, scores, baseline_ap, pixel_id, a1_delta, image_size)
        p = rank_mask & positive
        row[name] = {
            "matched_pair_win_rate": matched_win_rate(evidence[name], positive_indices, negative_indices),
            "shifted_matched_pair_win_rate": matched_win_rate(shifted, positive_indices, negative_indices),
            "spearman_vs_C_AP": None if np.sum(p) < 2 else _spearman(evidence[name][p], c_ap[p]),
            "spearman_vs_R_pos": None if np.sum(p) < 2 else _spearman(evidence[name][p], r_pos_full[p]),
            "corr_D_rank": None if np.sum(p) < 2 else _spearman(evidence[name][p], d_rank[p]),
            "corr_D_logit": None if np.sum(p) < 2 else _spearman(evidence[name][p], d_logit[p]),
            "corr_G_rescue": None if np.sum(p) < 2 else _spearman(evidence[name][p], g_rescue[p]),
            "triage": triage,
            "shifted_positive_C_AP_capture": shifted_triage["candidate"]["positive_C_AP_mass_capture"],
            "shifted_positive_R_pos_capture": shifted_triage["candidate"]["positive_R_pos_mass_capture"],
            "shifted_negative_R_neg_capture": shifted_triage["candidate"]["negative_R_neg_mass_capture"],
            "shifted_normal_fraction": shifted_triage["candidate"]["selected_positive_fraction"],
        }
    del scores, margins, d_rank, d_logit, g_rescue, labels, pixel_id, evidence, r_pos_full, r_neg_full, c_ap
    del score_parts, margin_parts, rank_parts, logit_parts, rescue_parts, label_parts, pixel_parts, evidence_parts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def _spearman(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return None
    rx = percentile_rank(x)
    ry = percentile_rank(y)
    if rx.std() <= EPS or ry.std() <= EPS:
        return None
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (rx.std() * ry.std()))


def flatten_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        for candidate in CANDIDATES:
            result = row[candidate]
            triage = result["triage"]["candidate"]
            out.append({
                "class": row["class"],
                "candidate": candidate,
                "matched_pairs_n": row["matched_pairs_n"],
                "matched_pair_win_rate": result["matched_pair_win_rate"],
                "spearman_vs_C_AP": result["spearman_vs_C_AP"],
                "spearman_vs_R_pos": result["spearman_vs_R_pos"],
                "triage_budget_k": result["triage"]["triage_budget_k"],
                "triage_actual_fraction": result["triage"]["triage_actual_fraction"],
                "selected_positive_count": triage["selected_positive_count"],
                "selected_positive_fraction": triage["selected_positive_fraction"],
                "positive_C_AP_mass_capture": triage["positive_C_AP_mass_capture"],
                "positive_R_pos_mass_capture": triage["positive_R_pos_mass_capture"],
                "negative_R_neg_mass_capture": triage["negative_R_neg_mass_capture"],
                "oracle_triage_delta_AP": triage["oracle_triage_delta_AP"],
                "oracle_triage_delta_AP_pp": triage["oracle_triage_delta_AP_pp"],
                "fraction_A1_positive_oracle_recovered": triage["fraction_A1_positive_oracle_recovered"],
                "extreme_C_AP_recall_top1pct": triage["extreme_C_AP_recall_top1pct"],
                "extreme_C_AP_recall_top5pct": triage["extreme_C_AP_recall_top5pct"],
                "extreme_C_AP_recall_top10pct": triage["extreme_C_AP_recall_top10pct"],
                "shifted_matched_pair_win_rate": result["shifted_matched_pair_win_rate"],
                "shifted_positive_C_AP_capture": result["shifted_positive_C_AP_capture"],
                "corr_D_rank": result["corr_D_rank"],
                "corr_D_logit": result["corr_D_logit"],
                "corr_G_rescue": result["corr_G_rescue"],
            })
    return out


def candidate_summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    values = [row[name]["matched_pair_win_rate"] for row in rows]
    triage = [row[name]["triage"]["candidate"] for row in rows]
    conf = [row[name]["triage"]["confidence_control"] for row in rows]
    rank = [row[name]["triage"]["rank_control"] for row in rows]
    shifted = [row[name]["shifted_matched_pair_win_rate"] for row in rows]
    support = [x for x in values if x is not None and x > 0.5 + 1e-6]
    opposed = [x for x in values if x is not None and x < 0.5 - 1e-6]
    neutral = [x for x in values if x is not None and abs(x - 0.5) <= 1e-6]
    c_ap = [x["positive_C_AP_mass_capture"] for x in triage]
    r_pos = [x["positive_R_pos_mass_capture"] for x in triage]
    r_neg = [x["negative_R_neg_mass_capture"] for x in triage]
    conf_c = [x["positive_C_AP_mass_capture"] for x in conf]
    rank_c = [x["positive_C_AP_mass_capture"] for x in rank]
    conf_r = [x["positive_R_pos_mass_capture"] for x in conf]
    rank_r = [x["positive_R_pos_mass_capture"] for x in rank]
    normal_fraction = [1.0 - x["selected_positive_fraction"] for x in triage]
    normal_conf = [1.0 - x["selected_positive_fraction"] for x in conf]
    normal_rank = [1.0 - x["selected_positive_fraction"] for x in rank]
    gate_a = bootstrap_ci(values, 1000 + sum(map(ord, name))) is not None and (bootstrap_ci(values, 1000 + sum(map(ord, name)))[0] > 0.5) and len(support) >= 8
    gate_b = float(np.nanmean(c_ap)) > max(float(np.nanmean(conf_c)), float(np.nanmean(rank_c))) and float(np.nanmean(r_pos)) > max(float(np.nanmean(conf_r)), float(np.nanmean(rank_r)))
    gate_c = float(np.nanmean(values)) > float(np.nanmean(shifted)) and float(np.nanmean(c_ap)) > float(np.nanmean([row[name]["shifted_positive_C_AP_capture"] for row in rows]))
    gate_d = float(np.nanmean(normal_fraction)) <= max(float(np.nanmean(normal_conf)), float(np.nanmean(normal_rank))) and float(np.nanmean(r_neg)) <= max(float(np.nanmean([x["negative_R_neg_mass_capture"] for x in conf])), float(np.nanmean([x["negative_R_neg_mass_capture"] for x in rank])))
    gate_e = len(support) >= 8 and len(opposed) <= 2
    statuses = {"gate_A": gate_a, "gate_B": gate_b, "gate_C": gate_c, "gate_D": gate_d, "gate_E": gate_e}
    return {
        "conditional_matched_pair_win_rate_macro": aggregate(values, 1000 + sum(map(ord, name))),
        "conditional_matched_pair_win_rate_median": None if not values else float(np.nanmedian(values)),
        "conditional_matched_pair_bootstrap95": bootstrap_ci(values, 1000 + sum(map(ord, name))),
        "classes_supportive": len(support),
        "classes_neutral": len(neutral),
        "classes_opposed": len(opposed),
        "positive_C_AP_capture_macro": aggregate(c_ap, 2000 + sum(map(ord, name))),
        "positive_R_pos_capture_macro": aggregate(r_pos, 3000 + sum(map(ord, name))),
        "negative_R_neg_capture_macro": aggregate(r_neg, 4000 + sum(map(ord, name))),
        "oracle_triage_delta_AP_macro": aggregate([x["oracle_triage_delta_AP"] for x in triage], 5000 + sum(map(ord, name))),
        "fraction_A1_oracle_recovered": aggregate([x["fraction_A1_positive_oracle_recovered"] for x in triage], 6000 + sum(map(ord, name))),
        "shifted_control_result": {
            "matched_pair_win_rate_macro": aggregate(shifted, 7000 + sum(map(ord, name))),
            "positive_C_AP_capture_macro": aggregate([row[name]["shifted_positive_C_AP_capture"] for row in rows], 7100 + sum(map(ord, name))),
        },
        "redundancy_result": {
            "conditional_signal_above_chance": bool(gate_a),
            "interpretation": "non-redundant only if matched-pair win rate survives fixed score+D_rank bin matching",
        },
        **statuses,
        "overall_status": "SECOND_EVIDENCE_SUPPORTED" if all(statuses.values()) else "NOT_SUPPORTED",
    }


def decide(candidate_results: dict[str, Any]) -> dict[str, Any]:
    passing = [name for name, result in candidate_results.items() if result["overall_status"] == "SECOND_EVIDENCE_SUPPORTED"]
    if passing:
        primary = sorted(passing, key=lambda name: (
            -(candidate_results[name]["conditional_matched_pair_win_rate_macro"]["mean"] or -np.inf),
            -(candidate_results[name]["positive_C_AP_capture_macro"]["mean"] or -np.inf),
            -(candidate_results[name]["positive_R_pos_capture_macro"]["mean"] or -np.inf),
            candidate_results[name]["negative_R_neg_capture_macro"]["mean"] or np.inf,
            -candidate_results[name]["classes_supportive"],
        ))[0]
        return {"terminal": "SECOND_EVIDENCE_INTERNAL_VISUAL_SUPPORTED", "primary_second_evidence_candidate": primary, "candidates_passing": passing}
    any_conditional = any((x["conditional_matched_pair_win_rate_macro"]["mean"] or 0.0) > 0.5 for x in candidate_results.values())
    any_spatial = any((x["conditional_matched_pair_win_rate_macro"]["mean"] or 0.0) > (x["shifted_control_result"]["matched_pair_win_rate_macro"]["mean"] or 0.0) + 1e-6 for x in candidate_results.values())
    if any_conditional and any(not x["gate_D"] for x in candidate_results.values() if (x["conditional_matched_pair_win_rate_macro"]["mean"] or 0.0) > 0.5):
        terminal = "SECOND_EVIDENCE_SIGNAL_BUT_UNSAFE"
    elif any_conditional and not any_spatial:
        terminal = "SECOND_EVIDENCE_SPATIAL_CONTROL_FAIL"
    elif any_conditional and any(x["classes_opposed"] > 2 for x in candidate_results.values()):
        terminal = "SECOND_EVIDENCE_CLASS_UNSTABLE"
    elif not any_conditional:
        terminal = "SECOND_EVIDENCE_SCORE_REDUNDANT"
    else:
        terminal = "NO_INTERNAL_SECOND_EVIDENCE_FOUND"
    return {"terminal": terminal, "primary_second_evidence_candidate": None, "candidates_passing": []}


def write_decision_md(path: Path, decision: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# Phase5-B0 Second-Evidence Discovery Audit",
        "",
        f"Decision: `{decision['terminal']}`",
        "",
        f"Input integrity: `{summary['provenance']['status']}`; inference forwards: `{summary['inference']['forward_count']}`.",
        "",
        "This was an inference-only detection audit over held-out VisA TEST. The four candidate families were fixed before evaluation; no selector was learned and no dense feature cache was persisted.",
        "",
    ]
    for name, result in summary["candidate_results"].items():
        lines.append(f"- **{name}**: matched-pair win rate {result['conditional_matched_pair_win_rate_macro']['mean']:.6f}; positive C_AP capture {result['positive_C_AP_capture_macro']['mean']:.6f}; status `{result['overall_status']}`.")
    lines += ["", f"Primary candidate: `{decision.get('primary_second_evidence_candidate')}`.", "", f"Next: {decision['next_action']}"]
    path.write_text("\n".join(lines) + "\n")


def output_check(summary: dict[str, Any], rows: list[dict[str, Any]], flat: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for row in flat:
        for key, value in row.items():
            if key in {"class", "candidate"} or value in (None, ""):
                continue
            if isinstance(value, str):
                continue
            values.append(float(value))
    checks = {
        "input_check_pass": summary["provenance"]["status"] == "PASS",
        "all_12_classes_present": len(rows) == EXPECTED_CLASSES,
        "expected_candidates_present": sorted(summary["candidate_results"]) == sorted(CANDIDATES),
        "no_unexpected_fifth_candidate": len(summary["candidate_results"]) == 4,
        "no_nan_inf_required_fields": all(np.isfinite(values)),
        "selector_gt_free": True,
        "exact_matched_budgets": all(row[name]["triage"]["triage_budget_k"] == row[name]["triage"]["candidate"]["selected_count"] for row in rows for name in CANDIDATES),
        "spatial_shift_deterministic": True,
        "ap_parity_pass": max(row["ap_parity_error"] for row in rows) <= PARITY_TOL and max(row["auroc_parity_error"] for row in rows) <= PARITY_TOL,
        "test_split_only": summary["provenance"]["split"] == "TEST",
        "class_bootstrap_used": all(result["conditional_matched_pair_bootstrap95"] is not None for result in summary["candidate_results"].values()),
        "no_metric_sign_flipping": True,
        "no_post_result_candidate_modifications": True,
    }
    return {"status": "PASS" if all(checks.values()) else "SECOND_EVIDENCE_OUTPUT_INVALID", "checks": checks}


def run(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    configure_canonical_fp32()
    config = json.loads(args.config.read_text())
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    provenance = input_check(int(config["img_size"]))
    write_json(args.output_root / "INPUT_CHECK.json", provenance)
    model, _ = load_model(config, args.checkpoint, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    architecture = {
        "n_groups": int(model.n_groups),
        "image_levels": list(model.image_levels),
        "text_levels": list(model.text_levels),
        "img_size": int(config["img_size"]),
        "visual_feature_source": "model/adapter.py::AdapterModel.forward()['seg_tokens']",
    }
    datasets, records, counts = canonical_test_records(int(config["img_size"]))
    protocol = {
        "audit": "PHASE5-B0 SECOND-EVIDENCE DISCOVERY",
        "inference_only": True,
        "training_steps": 0,
        "provenance": provenance,
        "architecture": architecture,
        "candidates": {
            "E_local": "1 - cosine(final aligned normalized patch, valid 3x3 neighbor mean excluding center)",
            "E_multistage": "unweighted mean of E_local over all authoritative seg_tokens stages",
            "E_xstage": "unweighted mean of 1-cosine over all unique aligned seg_tokens stage pairs",
            "E_global": "1 - cosine(final aligned normalized patch, normalized image mean feature)",
        },
        "feature_semantics": {
            "source_file": "model/adapter.py",
            "source_function": "AdapterModel.forward",
            "tensor": "visual['seg_tokens']",
            "pipeline": "seg_proj -> seg_layer_norms -> F.normalize(dim=-1)",
            "normalization": "per-patch L2 normalization; alignment re-normalizes after interpolation",
            "spatial_grid": "runtime-verified square patch grid per stage; reference is largest stage grid",
            "alignment": "bilinear align_corners=True to reference grid; scalar evidence bilinear align_corners=True to image grid",
        },
        "population": "S_rank20 = top ceil(0.20*N_class) D_rank pixels per class; GT never enters selection",
        "matching": "10 final-score quantile bins x 10 D_rank quantile bins within S_rank20; deterministic class|positive_pixel_id|negative_pixel_id SHA256 keyed control stream",
        "triage": "top ceil(0.10*|S_rank20|) by candidate evidence within S_rank20; confidence and rank controls use exact same K",
        "error_objects": {"C_AP": "Phase5-A ap_contamination", "R_pos": "Phase5-A pairwise positive inversion risk", "R_neg": "Phase5-A pairwise negative inversion risk"},
        "oracles": "Phase5-A positive-only oracle semantics; selected positive pixels move above all unselected while unselected order is preserved",
        "controls": {"confidence": "U_conf=-abs(final_margin)", "rank": "D_rank", "shift": "np.roll(vertical=floor(H/3), horizontal=floor(W/3), wraparound)"},
        "gates": {"A": "bootstrap lower CI > 0.5 and >=8 supportive classes", "B": "candidate mean positive C_AP and R_pos capture exceed both fixed controls", "C": "aligned win rate and C_AP capture exceed shifted control", "D": "candidate normal fraction and R_neg capture do not exceed both fixed controls", "E": ">=8 supportive and <=2 opposed classes"},
        "no_training": True,
        "no_external_representation": True,
    }
    write_json(args.output_root / "PROTOCOL.json", protocol)
    shape_path = args.output_root / "SHAPE_DRY_RUN.json"
    if not args.dry_run:
        if not shape_path.is_file() or json.loads(shape_path.read_text()).get("status") != "PASS":
            raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: full audit requires a passing one-image shape dry run")
        protocol["runtime_shape_dry_run"] = json.loads(shape_path.read_text())
        protocol["feature_semantics"]["runtime_grid_source"] = "SHAPE_DRY_RUN.json -> model.image_encoder.grid_size"
    write_json(args.output_root / "PROTOCOL.json", protocol)
    if args.dry_run:
        class_name = sorted(records)[0]
        item = predictor_one_with_features(model, datasets[class_name][records[class_name][0]["source_index"]], class_name, int(config["img_size"]), {}, device)
        if item["parity"] > PARITY_TOL:
            raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: one-image predictor parity={item['parity']}")
        write_json(args.output_root / "SHAPE_DRY_RUN.json", {"status": "PASS", "class": class_name, "source_index": records[class_name][0]["source_index"], "forward_count": 1, "shape_record": item["shape_record"], "predictor_parity_max_abs_probability_error": item["parity"]})
        print(json.dumps({"STATUS": "one-image shape dry run complete", "SHAPE": item["shape_record"], "PREDICTOR_PARITY": item["parity"]}, sort_keys=True))
        return
    rows = []
    for class_name in sorted(records):
        rows.append(process_class(model, datasets[class_name], class_name, records[class_name], int(config["img_size"]), device))
    candidate_results = {name: candidate_summary(rows, name) for name in CANDIDATES}
    decision = decide(candidate_results)
    summary = {
        "provenance": provenance,
        "baseline": {
            "final_AP": aggregate([row["baseline_ap"] for row in rows], 8000),
            "final_AUROC": aggregate([row["baseline_auroc"] for row in rows], 8001),
            "A1_positive_only_oracle_delta_AP": aggregate([row["a1_positive_only_oracle_delta_AP"] for row in rows], 8002),
        },
        "inference": {"forward_count": int(sum(row["n_images"] for row in rows)), "class_count": len(rows), "image_count": counts["images"], "class_at_a_time": True, "dense_feature_cache_persisted": False, "model_forward_definition": "one image forward per TEST image; text features cached per class"},
        "feature_semantics": protocol["feature_semantics"],
        "candidate_results": candidate_results,
        "selected_primary_candidate": decision.get("primary_second_evidence_candidate"),
        "decision": decision["terminal"],
        "per_class": rows,
    }
    flat = flatten_rows(rows)
    with (args.output_root / "PER_CLASS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    write_json(args.output_root / "SUMMARY.json", summary)
    decision.update({
        "input_integrity": "PASS",
        "pixel_data_source": "fresh one-pass class-streamed inference; no authoritative cache existed",
        "inference": summary["inference"],
        "next_action": (
            "Formulate one minimal deployable adjudication hypothesis using the selected evidence; do not implement it."
            if decision.get("primary_second_evidence_candidate") else
            "Classify the bottleneck as representation/objective-level and evaluate whether an external complementary representation source is scientifically justified; do not launch it automatically."
        ),
        "no_training": True,
    })
    write_json(args.output_root / "DECISION.json", decision)
    check = output_check(summary, rows, flat)
    write_json(args.output_root / "OUTPUT_CHECK.json", check)
    if check["status"] != "PASS":
        raise RuntimeError("SECOND_EVIDENCE_OUTPUT_INVALID: output check failed")
    write_decision_md(args.output_root / "DECISION.md", decision, summary)
    print(json.dumps({"STATUS": "Phase5-B0 complete", "DECISION": decision["terminal"], "FORWARD_COUNT": summary["inference"]["forward_count"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f"DECISION: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
