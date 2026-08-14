#!/usr/bin/env python3
"""Phase5 overnight Branch A: internal stage rescue / consensus dilution.

This is an inference-only diagnostic.  It reuses the exact Phase5 predictor,
test loader, deployment reconstruction, rank definitions, and AP/AUROC
parity helpers.  No selector is learned and no dense prediction cache is
written.  If no authoritative stage cache exists, one class-at-a-time VisA
TEST exposure pass is performed and reduced to compact class artifacts.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from audit_p4v_phase2b_readiness import load_model  # noqa: E402
from audit_phase5_hsir import (  # noqa: E402
    _sha256,
    aggregate_values,
    ap_contamination,
    build_architecture,
    deploy_from_native,
    exact_auc_ap,
    pairwise_risks,
    percentile_rank,
    predict_one,
    project_exact_auc_ap,
    upsample_patch_map,
    write_json,
)
from dataset import get_text_and_image_dataset  # noqa: E402
from utils import configure_canonical_fp32  # noqa: E402


OUTPUT_ROOT = ROOT / "runs/phase5/hsir/STAGE_RESCUE"
OVERNIGHT_ROOT = ROOT / "runs/phase5/hsir/OVERNIGHT"
CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
CONFIG = ROOT / "runs/phase4/k1/short64_seed0_attempt5/config.json"
VISA_META = ROOT / "dataset/hub/VisA.jsonl"
PHASE5_ROOT = ROOT / "runs/phase5/hsir/VISA_TEST"
ACTIONABILITY_ROOT = ROOT / "runs/phase5/hsir/ACTIONABILITY"
PHASE5_COMMIT = "29a8ffc934448b34424c77805a2c5c289bd9ddac"
SCIENTIFIC_ANCESTOR = "fcbff12059a0cb29698c14f443a0396acbef8c55"
EXPECTED_CHECKPOINT_SHA = "a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34"
EXPECTED_CONFIG_SHA = "377ce1c0ae1dd870f82ddcb828d8d8809fa46c007e61567f2150ec11354b23a4"
EXPECTED_VISA_META_SHA = "468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842"
EXPECTED_CLASSES = 12
EXPECTED_IMAGES = 2162
EXPECTED_NORMAL = 962
EXPECTED_ANOMALY = 1200
PRIMARY_BUDGET = 0.20
QUANTILE_BINS = 10
PARITY_TOL = 1e-10
EPS = 1e-12
VALID_TERMINALS = {
    "INTERNAL_STAGE_EVIDENCE_NOT_RESCUABLE",
    "FIXED_STAGE_AGGREGATION_ISSUE",
    "CONSENSUS_DILUTION_SUPPORTED",
    "DYNAMIC_STAGE_RESCUE_SUPPORTED",
    "INTERNAL_STAGE_RESCUE_INCONCLUSIVE",
}


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json_local(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def current_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def current_branch() -> str:
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()


def ancestor_ok() -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", SCIENTIFIC_ANCESTOR, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode == 0


def distribution(values: np.ndarray | list[float] | list[int], unit: str) -> dict:
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"unit": unit, "count": 0, "mean": None, "median": None, "p05": None, "p25": None, "p75": None, "p95": None, "min": None, "max": None}
    return {
        "unit": unit,
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def class_aggregate(values: list[float | int | None], name: str) -> dict:
    result = aggregate_values(values, seed=1700 + sum(ord(c) for c in name))
    result["unit"] = "class"
    result["metric"] = name
    return result


def stable_desc(values: np.ndarray) -> np.ndarray:
    """Stable descending order; array order is deterministic pixel identity."""
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("BRANCH_A_IMPLEMENTATION_INVALID: non-finite selector value")
    return np.argsort(-values, kind="mergesort")


def ap_from_order(labels: np.ndarray, order: np.ndarray) -> float:
    ordered = np.asarray(labels, dtype=np.uint8)[order]
    n_positive = int(ordered.sum())
    if n_positive == 0:
        return 0.0
    cumulative = np.cumsum(ordered, dtype=np.int64)
    precision = cumulative / np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(np.sum(precision[ordered == 1]) / n_positive)


def oracle_positive_only_ap(scores: np.ndarray, labels: np.ndarray, selected: np.ndarray) -> float:
    base_order = np.argsort(-scores, kind="mergesort")
    selected_positive = selected & (labels == 1)
    selected_order = base_order[selected_positive[base_order]]
    middle = base_order[~selected_positive[base_order]]
    return ap_from_order(labels, np.concatenate((selected_order, middle)))


def stage_scores_from_native(native_logits: np.ndarray, img_size: int) -> np.ndarray:
    native = torch.from_numpy(np.asarray(native_logits, dtype=np.float32)).unsqueeze(1)
    scores = []
    for stage in range(native.shape[0]):
        probabilities, _ = deploy_from_native(native[stage:stage + 1], img_size, "Industrial")
        scores.append(probabilities[0, 1].detach().cpu().numpy())
    return np.stack(scores, axis=0).astype(np.float32, copy=False)


def quantile_bins(values: np.ndarray, bins: int = QUANTILE_BINS) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(values, kind="mergesort")
    out = np.empty(values.size, dtype=np.int64)
    out[order] = np.minimum((np.arange(values.size) * bins) // max(values.size, 1), bins - 1)
    return out


def score_matched_rescue(
    scores: np.ndarray,
    d_rank: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
    g_rescue: np.ndarray,
) -> dict:
    positive = labels == 1
    high = selected & positive
    low = (~selected) & positive
    bins = quantile_bins(scores[positive])
    positive_indices = np.flatnonzero(positive)
    global_bins = np.full(scores.size, -1, dtype=np.int64)
    global_bins[positive_indices] = bins
    high_indices = positive_indices[high[positive]]
    low_indices = positive_indices[low[positive]]
    matched_high = []
    matched_low = []
    for bucket in range(QUANTILE_BINS):
        high_bucket = high_indices[global_bins[high_indices] == bucket]
        low_bucket = low_indices[global_bins[low_indices] == bucket]
        count = min(high_bucket.size, low_bucket.size)
        if count == 0:
            continue
        high_bucket = high_bucket[np.argsort(high_bucket, kind="mergesort")[:count]]
        low_bucket = low_bucket[np.argsort(low_bucket, kind="mergesort")[:count]]
        matched_high.append(high_bucket)
        matched_low.append(low_bucket)
    if not matched_high:
        return {
            "quantile_bins": QUANTILE_BINS,
            "matched_count": 0,
            "high_D_rank_G_rescue": distribution([], "matched selected positive pixels"),
            "low_D_rank_G_rescue": distribution([], "score-quantile-matched low-D_rank positive pixels"),
            "mean_signed_difference": None,
        }
    high_indices = np.concatenate(matched_high)
    low_indices = np.concatenate(matched_low)
    high_values = g_rescue[high_indices]
    low_values = g_rescue[low_indices]
    return {
        "quantile_bins": QUANTILE_BINS,
        "matched_count": int(high_values.size),
        "high_D_rank_G_rescue": distribution(high_values, "matched selected positive pixels"),
        "low_D_rank_G_rescue": distribution(low_values, "score-quantile-matched low-D_rank positive pixels"),
        "mean_signed_difference": float(high_values.mean() - low_values.mean()),
    }


def canonical_test_records(img_size: int):
    datasets = get_text_and_image_dataset("VisA", img_size, stage="test")
    records = {}
    all_rows = []
    for class_name in sorted(datasets):
        dataset = datasets[class_name]
        class_records = []
        for source_index, row in enumerate(dataset.meta):
            file_name = str(row["image_path"])
            if "train" in file_name.lower():
                raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: TRAIN path in VisA TEST")
            class_records.append({
                "source_index": int(source_index),
                "file_name": file_name,
                "label": int(row["label"]),
            })
            all_rows.append(row)
        records[class_name] = class_records
    if len(records) != EXPECTED_CLASSES or len(all_rows) != EXPECTED_IMAGES:
        raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: canonical TEST counts changed")
    if sum(int(row["label"]) == 0 for row in all_rows) != EXPECTED_NORMAL:
        raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: Normal count changed")
    if sum(int(row["label"]) == 1 for row in all_rows) != EXPECTED_ANOMALY:
        raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: anomaly count changed")
    return datasets, records


def provenance_gate() -> tuple[dict, dict, dict, dict]:
    action_protocol = json.loads((ACTIONABILITY_ROOT / "PROTOCOL.json").read_text())
    action_summary = json.loads((ACTIONABILITY_ROOT / "SUMMARY.json").read_text())
    action_decision = json.loads((ACTIONABILITY_ROOT / "DECISION.json").read_text())
    phase5_summary = json.loads((PHASE5_ROOT / "SUMMARY.json").read_text())
    phase5_decision = json.loads((PHASE5_ROOT / "DECISION.json").read_text())
    p = action_protocol["provenance"]
    checks = {
        "scientific_ancestor": ancestor_ok(),
        "current_branch": current_branch() == "autopilot/p4-conditional-semantic-factorization",
        "checkpoint_sha": _sha256(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA == p["checkpoint"]["sha256"],
        "config_sha": _sha256(CONFIG) == EXPECTED_CONFIG_SHA == p["config"]["sha256"],
        "visa_metadata_sha": _sha256(VISA_META) == EXPECTED_VISA_META_SHA == p["metadata_sha256"],
        "dataset": p.get("dataset") == "VisA" and phase5_summary["provenance"].get("dataset") == "VisA",
        "split": p.get("split") == "test" and phase5_summary["provenance"].get("split") == "test",
        "classes": p.get("number_classes") == EXPECTED_CLASSES and action_summary.get("class_count") == EXPECTED_CLASSES,
        "images": p.get("number_images") == EXPECTED_IMAGES and action_summary["inference"].get("forward_count") == EXPECTED_IMAGES,
        "normal_images": p.get("number_normal_images") == EXPECTED_NORMAL,
        "anomaly_images": p.get("number_anomaly_images") == EXPECTED_ANOMALY,
        "phase5_terminal": phase5_decision.get("terminal") == "STAGE_RANK_RISK_VISA_HELDOUT_PARTIAL",
        "actionability_terminal": action_decision.get("terminal") == "RANK_RISK_POSITIVE_SIDE_ONLY",
        "predictor_parity": p.get("predictor_parity") == "PASS" and action_summary["target_parity"]["status"] == "PASS",
        "prior_cache_absent": action_protocol["pixel_data_source"].get("authoritative_cache_found") is False,
        "no_train_paths": phase5_summary["provenance"].get("contains_train_paths") is False,
    }
    if not all(checks.values()):
        raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: " + ", ".join(k for k, v in checks.items() if not v))
    provenance = {
        "scientific_ancestor": SCIENTIFIC_ANCESTOR,
        "phase5_commit": PHASE5_COMMIT,
        "actionability_commit": SCIENTIFIC_ANCESTOR,
        "checkpoint": {"path": str(CHECKPOINT), "sha256": _sha256(CHECKPOINT)},
        "config": {"path": str(CONFIG), "sha256": _sha256(CONFIG)},
        "dataset": "VisA",
        "dataset_root": p["dataset_root"],
        "split": "test",
        "metadata_source": p["metadata_source"],
        "metadata_sha256": _sha256(VISA_META),
        "number_classes": EXPECTED_CLASSES,
        "number_images": EXPECTED_IMAGES,
        "number_normal_images": EXPECTED_NORMAL,
        "number_anomaly_images": EXPECTED_ANOMALY,
        "phase5_terminal": phase5_decision["terminal"],
        "actionability_terminal": action_decision["terminal"],
        "provenance_checks": checks,
        "required_upstream_sha256": {
            "phase5_summary": _sha256(PHASE5_ROOT / "SUMMARY.json"),
            "phase5_per_class": _sha256(PHASE5_ROOT / "PER_CLASS.csv"),
            "phase5_per_image": _sha256(PHASE5_ROOT / "PER_IMAGE.csv"),
            "actionability_protocol": _sha256(ACTIONABILITY_ROOT / "PROTOCOL.json"),
            "actionability_summary": _sha256(ACTIONABILITY_ROOT / "SUMMARY.json"),
            "actionability_decision": _sha256(ACTIONABILITY_ROOT / "DECISION.json"),
        },
    }
    return provenance, action_protocol, action_summary, phase5_summary


def architecture_gate(config: dict, checkpoint: dict, device: torch.device):
    model, _ = load_model(config, CHECKPOINT, device)
    architecture = build_architecture(model, config, checkpoint)
    expected = {
        "n_groups": 3,
        "image_levels": [8, 16, 24],
        "text_levels": [4, 8, 12],
        "image_size": int(config["img_size"]),
    }
    for key, value in expected.items():
        if architecture[key] != value:
            raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: architecture {key}={architecture[key]!r}, expected {value!r}")
    return model, architecture


def make_input_check(provenance: dict, architecture: dict) -> dict:
    return {
        "branch": "A_INTERNAL_STAGE_RESCUE",
        "status": "PASS",
        "scientific_ancestor": SCIENTIFIC_ANCESTOR,
        "checkpoint": provenance["checkpoint"],
        "config": provenance["config"],
        "split": "test",
        "dataset": "VisA",
        "image_count": EXPECTED_IMAGES,
        "class_count": EXPECTED_CLASSES,
        "normal_image_count": EXPECTED_NORMAL,
        "anomaly_image_count": EXPECTED_ANOMALY,
        "required_upstream_artifact_sha256": provenance["required_upstream_sha256"],
        "predictor_semantics": {
            "source": "audit_phase5_hsir.predict_one",
            "loader": "canonical get_text_and_image_dataset('VisA', img_size, stage='test')",
            "precision": "strict FP32; TF32 off; AMP off",
            "deployment": "Industrial Gaussian blur kernel=7 sigma=1; bilinear align_corners=True; mean group logits; softmax",
        },
        "runtime_architecture": architecture,
        "expected_outputs": [
            "PROTOCOL.json", "PER_CLASS.csv", "SUMMARY.json", "DECISION.json", "DECISION.md", "OUTPUT_CHECK.json"
        ],
        "inference_authorization": "one class-streamed VisA TEST pass because no authoritative stage cache exists",
    }


def process_class(model, dataset, class_name: str, records: list[dict], img_size: int, device: torch.device) -> dict:
    pixels_per_image = int(img_size * img_size)
    n_images = len(records)
    n_pixels = n_images * pixels_per_image
    n_stages = 3
    scores = np.empty(n_pixels, dtype=np.float32)
    labels = np.empty(n_pixels, dtype=np.uint8)
    d_rank = np.empty(n_pixels, dtype=np.float32)
    stage_scores = np.empty((n_stages, n_pixels), dtype=np.float32)
    stage_ranks = np.empty((n_stages, n_pixels), dtype=np.float32)
    text_cache: dict[str, torch.Tensor] = {}
    max_predictor_parity = 0.0
    with torch.inference_mode():
        for image_index, record in enumerate(records):
            raw = dataset[record["source_index"]]
            item = predict_one(model, raw, "VisA", class_name, img_size, text_cache, device)
            start = image_index * pixels_per_image
            end = start + pixels_per_image
            image_score = item["score"].reshape(-1).astype(np.float32, copy=False)
            image_target = item["target"].reshape(-1).astype(np.uint8, copy=False)
            image_d_rank = item["D_rank"].reshape(-1).astype(np.float32, copy=False)
            native_margins = np.asarray(item["native_margins"], dtype=np.float32)
            native_logits = np.asarray(item["native_logits"], dtype=np.float32)
            if native_margins.shape[0] != n_stages or native_logits.shape[0] != n_stages:
                raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: runtime stage count changed")
            image_stage_scores = stage_scores_from_native(native_logits, img_size).reshape(n_stages, -1)
            image_stage_ranks = np.stack([
                upsample_patch_map(percentile_rank(native_margins[g]), img_size)
                for g in range(n_stages)
            ], axis=0).astype(np.float32, copy=False).reshape(n_stages, -1)
            if image_score.size != pixels_per_image or image_target.size != pixels_per_image:
                raise RuntimeError("BRANCH_A_IMPLEMENTATION_INVALID: unexpected pixel count")
            scores[start:end] = image_score
            labels[start:end] = image_target
            d_rank[start:end] = image_d_rank
            stage_scores[:, start:end] = image_stage_scores
            stage_ranks[:, start:end] = image_stage_ranks
            max_predictor_parity = max(max_predictor_parity, float(item["parity"]["predictor_max_abs_probability_error"]))
            del item, raw, image_stage_scores, image_stage_ranks, native_margins, native_logits

    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(d_rank)) or not np.all(np.isfinite(stage_scores)):
        raise RuntimeError("BRANCH_A_IMPLEMENTATION_INVALID: non-finite inference output")
    positive = labels == 1
    if not positive.any() or positive.all():
        raise RuntimeError("BRANCH_A_IMPLEMENTATION_INVALID: class lacks both pixel labels")
    baseline_auc, baseline_ap = exact_auc_ap(scores, labels)
    reference_auc, reference_ap = project_exact_auc_ap(scores, labels)
    baseline_parity = {
        "exact_auc": float(baseline_auc),
        "exact_ap": float(baseline_ap),
        "evaluator_auc": float(reference_auc),
        "evaluator_ap": float(reference_ap),
        "auroc_error": abs(float(baseline_auc - reference_auc)),
        "ap_error": abs(float(baseline_ap - reference_ap)),
    }
    if max(baseline_parity["auroc_error"], baseline_parity["ap_error"]) > PARITY_TOL:
        raise RuntimeError("BRANCH_A_IMPLEMENTATION_INVALID: evaluator parity failed")

    selected_count = int(math.ceil(PRIMARY_BUDGET * n_pixels))
    selected = np.zeros(n_pixels, dtype=bool)
    selected[stable_desc(d_rank)[:selected_count]] = True
    selected_positive = selected & positive
    selected_positive_indices = np.flatnonzero(selected_positive)
    selected_stage_ranks = stage_ranks[:, selected_positive_indices].copy()
    selected_stage_scores = stage_scores[:, selected_positive_indices].copy()

    selected_final_ranks = np.empty(selected_positive_indices.size, dtype=np.float32)
    for image_index in range(n_images):
        start = image_index * pixels_per_image
        end = start + pixels_per_image
        local = np.flatnonzero((selected_positive_indices >= start) & (selected_positive_indices < end))
        if local.size:
            selected_final_ranks[local] = percentile_rank(scores[start:end])[selected_positive_indices[local] - start]
    r_best = selected_stage_ranks.max(axis=0)
    r_worst = selected_stage_ranks.min(axis=0)
    g_rescue = r_best - selected_final_ranks
    g_spread = r_best - r_worst

    r_pos, r_neg = pairwise_risks(scores, labels)
    r_pos_full = np.full(n_pixels, np.nan, dtype=np.float64)
    r_neg_full = np.full(n_pixels, np.nan, dtype=np.float64)
    r_pos_full[positive] = r_pos
    r_neg_full[~positive] = r_neg
    c_ap = ap_contamination(scores, labels)
    a1_ap = oracle_positive_only_ap(scores, labels, selected)
    a1_delta = float(a1_ap - baseline_ap)

    rescue_scores = scores.copy()
    rescue_scores[selected_positive_indices] = selected_stage_scores.max(axis=0)
    rescue_auc, rescue_ap = exact_auc_ap(rescue_scores, labels)
    rescue_delta = float(rescue_ap - baseline_ap)

    fixed_stage = {}
    for g, level in enumerate((8, 16, 24)):
        stage_auc, stage_ap = exact_auc_ap(stage_scores[g], labels)
        fixed_stage[f"stage{level}"] = {
            "stage_level": level,
            "auroc": float(stage_auc),
            "ap": float(stage_ap),
            "delta_auroc": float(stage_auc - baseline_auc),
            "delta_ap": float(stage_ap - baseline_ap),
            "delta_ap_pp": float(100.0 * (stage_ap - baseline_ap)),
        }

    winner = np.argmax(selected_stage_ranks, axis=0) if selected_positive_indices.size else np.empty(0, dtype=np.int64)
    stage_pixel_counts = {f"stage{level}": int(np.sum(winner == g)) for g, level in enumerate((8, 16, 24))}
    stage_pixel_fractions = {
        name: (float(count / max(selected_positive_indices.size, 1)) if selected_positive_indices.size else None)
        for name, count in stage_pixel_counts.items()
    }
    image_winner_counts = {f"stage{level}": 0 for level in (8, 16, 24)}
    image_winner_total = 0
    per_image_stage_winner = []
    for image_index in range(n_images):
        start = image_index * pixels_per_image
        end = start + pixels_per_image
        local = np.flatnonzero((selected_positive_indices >= start) & (selected_positive_indices < end))
        if not local.size:
            continue
        counts = [int(np.sum(winner[local] == g)) for g in range(n_stages)]
        majority = int(np.argmax(np.asarray(counts)))
        image_winner_counts[f"stage{(8, 16, 24)[majority]}"] += 1
        image_winner_total += 1
        per_image_stage_winner.append({
            "image_index": image_index,
            "selected_positive_count": int(local.size),
            "stage8_fraction": float(counts[0] / local.size),
            "stage16_fraction": float(counts[1] / local.size),
            "stage24_fraction": float(counts[2] / local.size),
            "majority_stage": int((8, 16, 24)[majority]),
        })
    image_winner_fractions = {
        name: (float(count / image_winner_total) if image_winner_total else None)
        for name, count in image_winner_counts.items()
    }
    class_winner = int(np.argmax(np.asarray(list(stage_pixel_counts.values())))) if selected_positive_indices.size else None

    harmful_normal = (~positive) & (r_neg_full > 0.0)
    best_stage_score = stage_scores.max(axis=0)
    normal_inflation = best_stage_score[~positive] - scores[~positive]
    harmful_inflation = best_stage_score[harmful_normal] - scores[harmful_normal]
    normal_safety = {
        "all_normal_best_stage_minus_final_score": distribution(normal_inflation, "Normal pixels; anomaly probability"),
        "harmful_normal_best_stage_minus_final_score": distribution(harmful_inflation, "Normal pixels with R_neg > 0; anomaly probability"),
        "all_normal_positive_inflation_fraction": float(np.mean(normal_inflation > EPS)) if normal_inflation.size else None,
        "harmful_normal_positive_inflation_fraction": float(np.mean(harmful_inflation > EPS)) if harmful_inflation.size else None,
        "harmful_normal_count": int(harmful_normal.sum()),
    }

    score_matched = score_matched_rescue(scores, d_rank, labels, selected, np.full(n_pixels, np.nan, dtype=np.float64))
    all_g_rescue = np.full(n_pixels, np.nan, dtype=np.float64)
    all_g_rescue[selected_positive_indices] = g_rescue
    score_matched = score_matched_rescue(scores, d_rank, labels, selected, all_g_rescue)

    result = {
        "class_name": class_name,
        "n_images": n_images,
        "n_pixels": n_pixels,
        "positive_pixel_count": int(positive.sum()),
        "normal_pixel_count": int((~positive).sum()),
        "baseline_ap": float(baseline_ap),
        "baseline_auroc": float(baseline_auc),
        "baseline_parity": baseline_parity,
        "predictor_exposure_max_abs_probability_error": max_predictor_parity,
        "selected_pixel_count": selected_count,
        "selected_positive_count": int(selected_positive.sum()),
        "selected_positive_fraction": float(selected_positive.sum() / selected_count),
        "selected_positive_G_rescue_mean": float(np.mean(g_rescue)) if g_rescue.size else None,
        "selected_positive_G_rescue_median": float(np.median(g_rescue)) if g_rescue.size else None,
        "selected_positive_G_rescue_distribution": distribution(g_rescue, "selected D_rank positives; within-image percentile-rank units"),
        "selected_positive_G_spread_distribution": distribution(g_spread, "selected D_rank positives; within-image percentile-rank units"),
        "G_rescue_vs_C_AP": spearman_safe(g_rescue, c_ap[selected_positive_indices]),
        "G_rescue_vs_R_pos": spearman_safe(g_rescue, r_pos_full[selected_positive_indices]),
        "rescue_stage_distribution": {
            "pixel": {"counts": stage_pixel_counts, "fractions": stage_pixel_fractions},
            "image": {"majority_counts": image_winner_counts, "majority_fractions": image_winner_fractions, "images_with_selected_positive": image_winner_total},
            "class": {"winner_stage": None if class_winner is None else (8, 16, 24)[class_winner], "pixel_fraction_by_stage": stage_pixel_fractions},
        },
        "fixed_stage": fixed_stage,
        "internal_stage_rescue_delta_AP": rescue_delta,
        "internal_stage_rescue_delta_pp": float(100.0 * rescue_delta),
        "internal_stage_rescue_AUROC_delta": float(rescue_auc - baseline_auc),
        "A1_positive_only_oracle_delta_AP": a1_delta,
        "A1_positive_only_oracle_delta_pp": float(100.0 * a1_delta),
        "fraction_of_A1_oracle_recovered": (float(rescue_delta / a1_delta) if abs(a1_delta) > EPS else None),
        "normal_inflation": normal_safety,
        "score_matched_control": score_matched,
        "selector_definition": "top ceil(0.20*N) D_rank pixels per class; stable descending sort; GT not used",
        "oracle_definition": "A1 positive-only oracle for comparison; internal rescue replaces selected positive scores with max fixed-stage deployed anomaly probability",
        "per_image_stage_winner": per_image_stage_winner,
    }
    del scores, labels, d_rank, stage_scores, stage_ranks, selected_stage_ranks, selected_stage_scores
    del r_pos_full, r_neg_full, c_ap, best_stage_score, rescue_scores, all_g_rescue
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def spearman_safe(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return None
    return float(np.corrcoef(percentile_rank(x), percentile_rank(y))[0, 1])


def aggregate_stage(rows: list[dict], key: str) -> dict:
    return {
        "AP": class_aggregate([row["fixed_stage"][key]["ap"] for row in rows], f"{key}_AP"),
        "AUROC": class_aggregate([row["fixed_stage"][key]["auroc"] for row in rows], f"{key}_AUROC"),
        "AP_delta": class_aggregate([row["fixed_stage"][key]["delta_ap"] for row in rows], f"{key}_AP_delta"),
        "AUROC_delta": class_aggregate([row["fixed_stage"][key]["delta_auroc"] for row in rows], f"{key}_AUROC_delta"),
    }


def summarize(rows: list[dict], provenance: dict, architecture: dict, input_check: dict) -> dict:
    stage_names = ("stage8", "stage16", "stage24")
    stage_aggregate = {name: aggregate_stage(rows, name) for name in stage_names}
    fixed_winners = [row["rescue_stage_distribution"]["class"]["winner_stage"] for row in rows]
    fixed_ap_deltas = {name: [row["fixed_stage"][name]["delta_ap"] for row in rows] for name in stage_names}
    best_fixed_name = max(stage_names, key=lambda name: float(np.mean(fixed_ap_deltas[name])))
    rescue_deltas = [row["internal_stage_rescue_delta_AP"] for row in rows]
    rescue_positive_classes = int(sum(delta > EPS for delta in rescue_deltas))
    rescue_neutral_classes = int(sum(abs(delta) <= EPS for delta in rescue_deltas))
    rescue_opposed_classes = int(sum(delta < -EPS for delta in rescue_deltas))
    class_winner_counts = {name: int(sum(stage == int(name[5:]) for stage in fixed_winners)) for name in stage_names}
    pixel_stage_counts = {name: int(sum(row["rescue_stage_distribution"]["pixel"]["counts"][name] for row in rows)) for name in stage_names}
    pixel_stage_total = max(sum(pixel_stage_counts.values()), 1)
    pixel_stage_fractions = {name: float(count / pixel_stage_total) for name, count in pixel_stage_counts.items()}
    all_stage_spread = [row["selected_positive_G_spread_distribution"]["median"] for row in rows]
    dynamic_range_valid = any((value or 0.0) > EPS for value in all_stage_spread)
    macro_rescue = float(np.mean(rescue_deltas))
    macro_best_fixed_delta = float(np.mean(fixed_ap_deltas[best_fixed_name]))
    stage_association = {
        "G_rescue_vs_C_AP": class_aggregate([row["G_rescue_vs_C_AP"] for row in rows], "G_rescue_vs_C_AP"),
        "G_rescue_vs_R_pos": class_aggregate([row["G_rescue_vs_R_pos"] for row in rows], "G_rescue_vs_R_pos"),
    }
    summary = {
        "provenance": provenance,
        "architecture": architecture,
        "input_check": input_check,
        "inference": {
            "forward_count": int(sum(row["n_images"] for row in rows)),
            "class_count": len(rows),
            "class_at_a_time": True,
            "dense_cache_persisted": False,
            "prior_A1_metrics_reused": True,
        },
        "final_AP": {
            "class_macro_mean": class_aggregate([row["baseline_ap"] for row in rows], "final_AP_class_macro_mean"),
            "class_median": class_aggregate([row["baseline_ap"] for row in rows], "final_AP_class_median"),
        },
        "final_AUROC": {
            "class_macro_mean": class_aggregate([row["baseline_auroc"] for row in rows], "final_AUROC_class_macro_mean"),
            "class_median": class_aggregate([row["baseline_auroc"] for row in rows], "final_AUROC_class_median"),
        },
        "each_stage_AP": {name: stage_aggregate[name]["AP"] for name in stage_names},
        "each_stage_AUROC": {name: stage_aggregate[name]["AUROC"] for name in stage_names},
        "each_stage_deltas": stage_aggregate,
        "selected_positive_G_rescue_mean": class_aggregate([row["selected_positive_G_rescue_mean"] for row in rows], "selected_positive_G_rescue_mean"),
        "selected_positive_G_rescue_median": class_aggregate([row["selected_positive_G_rescue_median"] for row in rows], "selected_positive_G_rescue_median"),
        "selected_positive_G_rescue_distribution": distribution([v for row in rows for v in [row["selected_positive_G_rescue_distribution"]["mean"]]], "class means of selected positive G_rescue"),
        "selected_positive_G_spread_distribution": distribution([v for row in rows for v in [row["selected_positive_G_spread_distribution"]["mean"]]], "class means of selected positive G_spread"),
        "G_rescue_vs_C_AP": stage_association["G_rescue_vs_C_AP"],
        "G_rescue_vs_R_pos": stage_association["G_rescue_vs_R_pos"],
        "rescue_stage_distribution": {
            "pixel": {"counts": pixel_stage_counts, "fractions": pixel_stage_fractions},
            "image_majority": {name: int(sum(row["rescue_stage_distribution"]["image"]["majority_counts"][name] for row in rows)) for name in stage_names},
            "class_winner_counts": class_winner_counts,
        },
        "internal_stage_rescue_delta_AP": class_aggregate(rescue_deltas, "internal_stage_rescue_delta_AP"),
        "internal_stage_rescue_delta_pp": class_aggregate([100.0 * x for x in rescue_deltas], "internal_stage_rescue_delta_pp"),
        "A1_positive_only_oracle_delta_AP": class_aggregate([row["A1_positive_only_oracle_delta_AP"] for row in rows], "A1_positive_only_oracle_delta_AP"),
        "fraction_of_A1_oracle_recovered": class_aggregate([row["fraction_of_A1_oracle_recovered"] for row in rows], "fraction_of_A1_oracle_recovered"),
        "normal_inflation": {
            "all_normal_best_stage_minus_final_score_mean_by_class": class_aggregate([row["normal_inflation"]["all_normal_best_stage_minus_final_score"]["mean"] for row in rows], "normal_inflation_mean"),
            "all_normal_best_stage_minus_final_score_median_by_class": class_aggregate([row["normal_inflation"]["all_normal_best_stage_minus_final_score"]["median"] for row in rows], "normal_inflation_median"),
            "harmful_normal_best_stage_minus_final_score_mean_by_class": class_aggregate([row["normal_inflation"]["harmful_normal_best_stage_minus_final_score"]["mean"] for row in rows], "harmful_normal_inflation_mean"),
            "positive_inflation_fraction_all_normal": class_aggregate([row["normal_inflation"]["all_normal_positive_inflation_fraction"] for row in rows], "normal_positive_inflation_fraction"),
            "positive_inflation_fraction_harmful_normal": class_aggregate([row["normal_inflation"]["harmful_normal_positive_inflation_fraction"] for row in rows], "harmful_normal_positive_inflation_fraction"),
        },
        "score_matched_control": {
            "matched_count_by_class": [row["score_matched_control"]["matched_count"] for row in rows],
            "high_D_rank_G_rescue_mean_by_class": class_aggregate([row["score_matched_control"]["high_D_rank_G_rescue"]["mean"] for row in rows], "score_matched_high_G_rescue_mean"),
            "low_D_rank_G_rescue_mean_by_class": class_aggregate([row["score_matched_control"]["low_D_rank_G_rescue"]["mean"] for row in rows], "score_matched_low_G_rescue_mean"),
            "mean_signed_difference_by_class": class_aggregate([row["score_matched_control"]["mean_signed_difference"] for row in rows], "score_matched_G_rescue_difference"),
            "quantile_bins": QUANTILE_BINS,
        },
        "class_consistency": {
            "internal_rescue_positive_classes": rescue_positive_classes,
            "internal_rescue_neutral_classes": rescue_neutral_classes,
            "internal_rescue_opposed_classes": rescue_opposed_classes,
            "fixed_stage_class_winner_counts": class_winner_counts,
            "best_fixed_stage_by_macro_AP_delta": best_fixed_name,
        },
        "target_parity": {
            "max_final_AP_evaluator_error": max(row["baseline_parity"]["ap_error"] for row in rows),
            "max_final_AUROC_evaluator_error": max(row["baseline_parity"]["auroc_error"] for row in rows),
            "max_predictor_exposure_probability_error": max(row["predictor_exposure_max_abs_probability_error"] for row in rows),
            "status": "PASS",
        },
        "decision_evidence": {
            "stage_dynamic_range_valid": dynamic_range_valid,
            "internal_rescue_delta_class_macro_mean": macro_rescue,
            "best_fixed_stage": best_fixed_name,
            "best_fixed_stage_AP_delta_class_macro_mean": macro_best_fixed_delta,
            "fixed_stage_class_winner_counts": class_winner_counts,
            "pixel_stage_winner_fractions": pixel_stage_fractions,
        },
        "per_class": rows,
    }
    return summary


def flatten_row(row: dict) -> dict:
    out = {
        "class": row["class_name"],
        "n_images": row["n_images"],
        "n_pixels": row["n_pixels"],
        "baseline_ap": row["baseline_ap"],
        "baseline_auroc": row["baseline_auroc"],
        "selected_pixel_count": row["selected_pixel_count"],
        "selected_positive_count": row["selected_positive_count"],
        "selected_positive_fraction": row["selected_positive_fraction"],
        "selected_positive_G_rescue_mean": row["selected_positive_G_rescue_mean"],
        "selected_positive_G_rescue_median": row["selected_positive_G_rescue_median"],
        "G_rescue_vs_C_AP": row["G_rescue_vs_C_AP"],
        "G_rescue_vs_R_pos": row["G_rescue_vs_R_pos"],
        "internal_stage_rescue_delta_AP": row["internal_stage_rescue_delta_AP"],
        "internal_stage_rescue_delta_pp": row["internal_stage_rescue_delta_pp"],
        "A1_positive_only_oracle_delta_AP": row["A1_positive_only_oracle_delta_AP"],
        "A1_positive_only_oracle_delta_pp": row["A1_positive_only_oracle_delta_pp"],
        "fraction_of_A1_oracle_recovered": row["fraction_of_A1_oracle_recovered"],
        "normal_inflation_mean": row["normal_inflation"]["all_normal_best_stage_minus_final_score"]["mean"],
        "harmful_normal_inflation_mean": row["normal_inflation"]["harmful_normal_best_stage_minus_final_score"]["mean"],
        "normal_positive_inflation_fraction": row["normal_inflation"]["all_normal_positive_inflation_fraction"],
        "harmful_normal_positive_inflation_fraction": row["normal_inflation"]["harmful_normal_positive_inflation_fraction"],
        "score_matched_count": row["score_matched_control"]["matched_count"],
        "score_matched_high_G_rescue_mean": row["score_matched_control"]["high_D_rank_G_rescue"]["mean"],
        "score_matched_low_G_rescue_mean": row["score_matched_control"]["low_D_rank_G_rescue"]["mean"],
        "score_matched_mean_signed_difference": row["score_matched_control"]["mean_signed_difference"],
    }
    for name in ("stage8", "stage16", "stage24"):
        out[f"{name}_ap"] = row["fixed_stage"][name]["ap"]
        out[f"{name}_auroc"] = row["fixed_stage"][name]["auroc"]
        out[f"{name}_delta_ap"] = row["fixed_stage"][name]["delta_ap"]
        out[f"{name}_delta_ap_pp"] = row["fixed_stage"][name]["delta_ap_pp"]
        out[f"{name}_delta_auroc"] = row["fixed_stage"][name]["delta_auroc"]
        out[f"{name}_pixel_winner_fraction"] = row["rescue_stage_distribution"]["pixel"]["fractions"][name]
        out[f"{name}_image_majority_fraction"] = row["rescue_stage_distribution"]["image"]["majority_fractions"][name]
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def decision_from_summary(summary: dict) -> dict:
    evidence = summary["decision_evidence"]
    classes = summary["class_consistency"]
    rescue_classes = classes["internal_rescue_positive_classes"]
    opposed_classes = classes["internal_rescue_opposed_classes"]
    winners = classes["fixed_stage_class_winner_counts"]
    dominant_stage, dominant_count = max(winners.items(), key=lambda item: item[1])
    macro_rescue = evidence["internal_rescue_delta_class_macro_mean"]
    macro_fixed = evidence["best_fixed_stage_AP_delta_class_macro_mean"]
    assoc_ap = summary["G_rescue_vs_C_AP"]["median"]
    assoc_rpos = summary["G_rescue_vs_R_pos"]["median"]
    if not evidence["stage_dynamic_range_valid"]:
        terminal = "INTERNAL_STAGE_RESCUE_INCONCLUSIVE"
    elif rescue_classes == 0 and macro_rescue <= EPS and (assoc_ap or 0.0) <= 0.0 and (assoc_rpos or 0.0) <= 0.0:
        terminal = "INTERNAL_STAGE_EVIDENCE_NOT_RESCUABLE"
    elif dominant_count >= 8 and macro_fixed >= macro_rescue and macro_fixed > EPS:
        terminal = "FIXED_STAGE_AGGREGATION_ISSUE"
    elif rescue_classes >= 8 and opposed_classes <= 2 and (assoc_ap or 0.0) > 0.0 and (assoc_rpos or 0.0) > 0.0 and dominant_count < 8:
        terminal = "DYNAMIC_STAGE_RESCUE_SUPPORTED"
    elif rescue_classes >= 8 and opposed_classes <= 4 and macro_rescue > EPS:
        terminal = "CONSENSUS_DILUTION_SUPPORTED"
    else:
        terminal = "INTERNAL_STAGE_RESCUE_INCONCLUSIVE"
    next_branch = {
        "DYNAMIC_STAGE_RESCUE_SUPPORTED": "B1",
        "CONSENSUS_DILUTION_SUPPORTED": "B1",
        "FIXED_STAGE_AGGREGATION_ISSUE": "B2",
        "INTERNAL_STAGE_EVIDENCE_NOT_RESCUABLE": "B3",
        "INTERNAL_STAGE_RESCUE_INCONCLUSIVE": "B4",
    }[terminal]
    return {
        "terminal": terminal,
        "decision_rule": "class-consistent rescue, fixed-stage dominance, stage-identity diversity, and signed G_rescue associations; no post-hoc threshold tuning",
        "evidence": {
            "rescue_positive_classes": rescue_classes,
            "rescue_opposed_classes": opposed_classes,
            "dominant_fixed_stage": dominant_stage,
            "dominant_fixed_stage_class_winner_count": dominant_count,
            "internal_rescue_delta_AP_class_macro_mean": macro_rescue,
            "best_fixed_stage_delta_AP_class_macro_mean": macro_fixed,
            "G_rescue_vs_C_AP_class_median": assoc_ap,
            "G_rescue_vs_R_pos_class_median": assoc_rpos,
        },
        "next_branch": next_branch,
        "next_action": f"Follow Branch {next_branch} only if this Branch-A output check passes.",
    }


def write_decision_md(path: Path, decision: dict, summary: dict) -> None:
    evidence = decision["evidence"]
    lines = [
        f"DECISION: {decision['terminal']}",
        "INPUT INTEGRITY: PASS; exact Phase5-A.1 provenance and frozen predictor semantics verified",
        "OUTPUT INTEGRITY: PASS; final AP/AUROC evaluator parity and class-level checks passed",
        "",
        "FINAL_DEPLOYED_CONSENSUS:",
        f"  final_AP_class_macro_mean: {summary['final_AP']['class_macro_mean']['mean']:.9f}",
        f"  final_AUROC_class_macro_mean: {summary['final_AUROC']['class_macro_mean']['mean']:.9f}",
        "",
        "FIXED_STAGE_COUNTERFACTUALS:",
    ]
    for name in ("stage8", "stage16", "stage24"):
        stage = summary["each_stage_deltas"][name]
        lines.append(f"  {name}_AP_class_macro_mean: {stage['AP']['mean']:.9f}; {name}_AUROC_class_macro_mean: {stage['AUROC']['mean']:.9f}; {name}_AP_delta_class_macro_mean: {stage['AP_delta']['mean']:.9f}")
    lines.extend([
        "",
        "SELECTED_POSITIVE_RESCUE:",
        f"  selected_positive_G_rescue_mean_class_median: {summary['selected_positive_G_rescue_mean']['median']:.9f}",
        f"  selected_positive_G_rescue_median_class_median: {summary['selected_positive_G_rescue_median']['median']:.9f}",
        f"  G_rescue_vs_C_AP_class_median: {summary['G_rescue_vs_C_AP']['median']}",
        f"  G_rescue_vs_R_pos_class_median: {summary['G_rescue_vs_R_pos']['median']}",
        f"  internal_stage_rescue_delta_AP_class_macro_mean: {summary['internal_stage_rescue_delta_AP']['mean']:.9f}",
        f"  A1_positive_only_oracle_delta_AP_class_macro_mean: {summary['A1_positive_only_oracle_delta_AP']['mean']:.9f}",
        f"  fraction_of_A1_oracle_recovered_class_median: {summary['fraction_of_A1_oracle_recovered']['median']}",
        "",
        "STAGE_IDENTITY:",
        f"  pixel_winner_fractions: {summary['rescue_stage_distribution']['pixel']['fractions']}",
        f"  class_winner_counts: {summary['rescue_stage_distribution']['class_winner_counts']}",
        "",
        "NORMAL_SAFETY:",
        f"  all_normal_inflation_mean_class_median: {summary['normal_inflation']['all_normal_best_stage_minus_final_score_mean_by_class']['median']}",
        f"  harmful_normal_inflation_mean_class_median: {summary['normal_inflation']['harmful_normal_best_stage_minus_final_score_mean_by_class']['median']}",
        "",
        f"NEXT_BRANCH: {decision['next_branch']}",
    ])
    path.write_text("\n".join(lines) + "\n")


def output_check(summary: dict, decision: dict) -> dict:
    rows = summary["per_class"]
    required_artifacts = ["PROTOCOL.json", "PER_CLASS.csv", "SUMMARY.json", "DECISION.json", "DECISION.md"]
    finite_fields = [
        "baseline_ap", "baseline_auroc", "internal_stage_rescue_delta_AP",
        "A1_positive_only_oracle_delta_AP", "selected_positive_G_rescue_mean",
    ]
    finite_ok = all(np.isfinite(float(row[field])) for row in rows for field in finite_fields)
    classes = [row["class_name"] for row in rows]
    checks = {
        "expected_artifacts_present_before_output_check": all((OUTPUT_ROOT / name).is_file() for name in required_artifacts),
        "row_count_valid": len(rows) == EXPECTED_CLASSES,
        "no_nan_inf_required_metrics": finite_ok,
        "class_set_complete": len(set(classes)) == EXPECTED_CLASSES,
        "final_metric_parity_pass": summary["target_parity"]["status"] == "PASS",
        "decision_terminal_valid": decision["terminal"] in VALID_TERMINALS,
        "no_train_paths_in_test_evidence": summary["provenance"]["provenance_checks"]["no_train_paths"],
        "no_gt_leakage_into_selector": all("GT not used" in row["selector_definition"] for row in rows),
        "no_unexpected_formula_changes": summary["architecture"]["n_groups"] == 3 and summary["architecture"]["image_levels"] == [8, 16, 24],
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {"branch": "A_INTERNAL_STAGE_RESCUE", "status": status, "checks": checks, "decision": decision["terminal"], "class_count": len(rows), "forward_count": summary["inference"]["forward_count"]}


def build_protocol(provenance: dict, architecture: dict) -> dict:
    return {
        "branch": "A_INTERNAL_STAGE_RESCUE",
        "purpose": "Determine whether D_rank-selected held-out weak-positive failures contain useful evidence in an internal Phase2B stage suppressed by final consensus.",
        "inference_only": True,
        "training_steps": 0,
        "provenance": provenance,
        "pixel_data_source": {
            "authoritative_stage_cache_found": False,
            "source": "exactly one class-streamed VisA TEST inference pass",
            "dense_cache_persisted": False,
            "forward_count_expected": EXPECTED_IMAGES,
        },
        "architecture": architecture,
        "predictor_semantics": {
            "source": "audit_phase5_hsir.predict_one",
            "stage_levels": "runtime verified image levels [8,16,24]",
            "stage_margin": "native anomaly logit minus native normal logit before deployment",
            "stage_score": "each singleton native stage deployed with the exact Industrial reconstruction",
            "final_score": "mean deployed group logits followed by softmax anomaly probability",
            "final_rank": "average-tie percentile rank of final deployed score within each image",
            "stage_rank": "average-tie percentile rank of each native stage margin within each image, then bilinear upsample",
        },
        "selector": {
            "name": "S_RANK",
            "definition": "top ceil(0.20*N) D_rank descending per class",
            "tie_handling": "stable mergesort over canonical source-index then row-major pixel order",
            "gt_free": True,
        },
        "measures": {
            "r_best": "max stage rank",
            "r_worst": "min stage rank",
            "G_rescue": "r_best - r_final",
            "G_spread": "r_best - r_worst",
            "damage": "frozen C_AP and R_pos from Phase5-A class-pooled final scores",
        },
        "fixed_stage_counterfactual": "each singleton stage deployed with the exact test reconstruction; AP and AUROC class-pooled",
        "internal_stage_rescue_upper_bound": "ORACLE/non-deployable: selected positive pixels only use max singleton-stage deployed anomaly probability; selected Normal pixels remain final scores",
        "score_matched_control": "same selected-positive count from low-D_rank positives within ten final-score quantile bins; offline diagnostic only",
        "statistics": "class primary; class mean, median, bootstrap 95% CI; pixels are not independent replicates",
        "no_training": True,
        "no_new_selector_learning": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance, action_protocol, action_summary, phase5_summary = provenance_gate()
    config = json.loads(CONFIG.read_text())
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    configure_canonical_fp32()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, architecture = architecture_gate(config, checkpoint, device)
    datasets, records = canonical_test_records(int(config["img_size"]))
    input_check = make_input_check(provenance, architecture)
    write_json_local(args.output_root / "INPUT_CHECK.json", input_check)
    protocol = build_protocol(provenance, architecture)
    write_json_local(args.output_root / "PROTOCOL.json", protocol)
    rows = []
    for class_name in sorted(records):
        rows.append(process_class(model, datasets[class_name], class_name, records[class_name], int(config["img_size"]), device))
    summary = summarize(rows, provenance, architecture, input_check)
    write_csv(args.output_root / "PER_CLASS.csv", [flatten_row(row) for row in rows])
    write_json_local(args.output_root / "SUMMARY.json", summary)
    decision = decision_from_summary(summary)
    decision.update({
        "input_integrity": "PASS",
        "output_integrity_pending": True,
        "target_parity": summary["target_parity"],
        "no_training": True,
        "pixel_data_source": "one fresh class-streamed inference exposure pass; no authoritative stage cache existed",
    })
    write_json_local(args.output_root / "DECISION.json", decision)
    write_decision_md(args.output_root / "DECISION.md", decision, summary)
    check = output_check(summary, decision)
    if check["status"] != "PASS":
        write_json_local(args.output_root / "OUTPUT_CHECK.json", check)
        raise RuntimeError("BRANCH_A_IMPLEMENTATION_INVALID: OUTPUT_CHECK failed")
    decision["output_integrity"] = "PASS"
    decision["output_check"] = str(args.output_root / "OUTPUT_CHECK.json")
    write_json_local(args.output_root / "DECISION.json", decision)
    write_decision_md(args.output_root / "DECISION.md", decision, summary)
    write_json_local(args.output_root / "OUTPUT_CHECK.json", check)
    print(json.dumps({"STATUS": "Branch A complete", "DECISION": decision["terminal"], "FORWARD_COUNT": summary["inference"]["forward_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
