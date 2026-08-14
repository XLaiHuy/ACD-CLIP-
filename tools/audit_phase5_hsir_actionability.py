#!/usr/bin/env python3
"""Phase5-A.1 v2: inference-only actionability decomposition on VisA TEST.

The Phase5-A run did not persist authoritative per-pixel arrays.  This module
therefore performs exactly one predictor exposure pass, one class at a time,
and immediately reduces the retained pixel arrays to compact class records.
It never trains, learns a selector, or writes dense prediction caches.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

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
    exact_auc_ap,
    pairwise_risks,
    predict_one,
    write_json,
)
from dataset import get_text_and_image_dataset  # noqa: E402
from utils import configure_canonical_fp32  # noqa: E402


CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
CONFIG = ROOT / "runs/phase4/k1/short64_seed0_attempt5/config.json"
PROTOCOL = ROOT / "runs/phase5/hsir/AUDIT_PROTOCOL.json"
PHASE5_SUMMARY = ROOT / "runs/phase5/hsir/VISA_TEST/SUMMARY.json"
PHASE5_PER_CLASS = ROOT / "runs/phase5/hsir/VISA_TEST/PER_CLASS.csv"
PHASE5_PER_IMAGE = ROOT / "runs/phase5/hsir/VISA_TEST/PER_IMAGE.csv"
VISA_META = ROOT / "dataset/hub/VisA.jsonl"
OUTPUT_ROOT = ROOT / "runs/phase5/hsir/ACTIONABILITY"
PREVIOUS_COMMIT = "29a8ffc934448b34424c77805a2c5c289bd9ddac"
EXPECTED_CHECKPOINT_SHA = "a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34"
EXPECTED_CONFIG_SHA = "377ce1c0ae1dd870f82ddcb828d8d8809fa46c007e61567f2150ec11354b23a4"
EXPECTED_VISA_META_SHA = "468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842"
EXPECTED_CLASSES = 12
EXPECTED_IMAGES = 2162
EXPECTED_NORMAL = 962
EXPECTED_ANOMALY = 1200
PRIMARY_BUDGET = 0.20
EXTREME_FRACTIONS = (0.01, 0.05, 0.10)
EPS = 1e-12
PARITY_TOL = 1e-10


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def provenance_gate() -> dict:
    summary = json.loads(PHASE5_SUMMARY.read_text())
    decision = json.loads((PHASE5_SUMMARY.parent / "DECISION.json").read_text())
    prior_class_rows = list(csv.DictReader(PHASE5_PER_CLASS.open()))
    prior_image_rows = list(csv.DictReader(PHASE5_PER_IMAGE.open()))
    p = summary["provenance"]
    parity = summary["parity"]
    checks = {
        "checkpoint_sha": _sha256(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA,
        "config_sha": _sha256(CONFIG) == EXPECTED_CONFIG_SHA,
        "visa_metadata_sha": _sha256(VISA_META) == EXPECTED_VISA_META_SHA,
        "dataset": p.get("dataset") == "VisA",
        "split": p.get("split") == "test",
        "classes": p.get("number_classes") == EXPECTED_CLASSES and len(prior_class_rows) == EXPECTED_CLASSES,
        "images": p.get("number_images") == EXPECTED_IMAGES and len(prior_image_rows) == EXPECTED_IMAGES,
        "normal_images": p.get("number_normal_images") == EXPECTED_NORMAL,
        "anomaly_images": p.get("number_anomaly_images") == EXPECTED_ANOMALY,
        "no_train_paths": p.get("contains_train_paths") is False,
        "predictor_parity": parity.get("predictor_max_abs_probability_error") == 0.0,
        "ap_parity": parity.get("ap_reconstruction_error", float("inf")) <= PARITY_TOL,
        "auroc_parity": parity.get("auroc_reconstruction_error", float("inf")) <= PARITY_TOL,
        "phase5_commit": _git_head() == PREVIOUS_COMMIT,
        "phase5_terminal": decision.get("terminal") == "STAGE_RANK_RISK_VISA_HELDOUT_PARTIAL",
    }
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise RuntimeError("INPUT_PROVENANCE_MISMATCH: " + ", ".join(failed))
    return {
        "phase5_commit": PREVIOUS_COMMIT,
        "checkpoint": {"path": str(CHECKPOINT), "sha256": _sha256(CHECKPOINT)},
        "config": {"path": str(CONFIG), "sha256": _sha256(CONFIG)},
        "dataset": "VisA",
        "split": "test",
        "metadata_source": p["metadata_source"],
        "metadata_sha256": _sha256(VISA_META),
        "dataset_root": p["dataset_root"],
        "number_classes": EXPECTED_CLASSES,
        "number_images": EXPECTED_IMAGES,
        "number_normal_images": EXPECTED_NORMAL,
        "number_anomaly_images": EXPECTED_ANOMALY,
        "predictor_parity": "PASS",
        "phase5_summary": str(PHASE5_SUMMARY),
        "phase5_per_class": str(PHASE5_PER_CLASS),
        "phase5_per_image": str(PHASE5_PER_IMAGE),
        "provenance_checks": checks,
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
                raise RuntimeError("INPUT_PROVENANCE_MISMATCH: canonical VisA TEST contains TRAIN path")
            class_records.append({
                "source_index": int(source_index),
                "file_name": file_name,
                "label": int(row["label"]),
            })
            all_rows.append(row)
        records[class_name] = class_records
    if len(records) != EXPECTED_CLASSES or len(all_rows) != EXPECTED_IMAGES:
        raise RuntimeError("INPUT_PROVENANCE_MISMATCH: canonical TEST counts changed")
    if sum(int(row["label"]) == 0 for row in all_rows) != EXPECTED_NORMAL:
        raise RuntimeError("INPUT_PROVENANCE_MISMATCH: canonical TEST Normal count changed")
    if sum(int(row["label"]) == 1 for row in all_rows) != EXPECTED_ANOMALY:
        raise RuntimeError("INPUT_PROVENANCE_MISMATCH: canonical TEST anomaly count changed")
    return datasets, records


def stable_desc_order(values: np.ndarray, pixel_id: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("ACTIONABILITY_TARGET_PARITY_INVALID: non-finite selector value")
    return np.lexsort((pixel_id, -values))


def select_top(values: np.ndarray, pixel_id: np.ndarray, count: int) -> np.ndarray:
    selected = np.zeros(values.size, dtype=bool)
    selected[stable_desc_order(values, pixel_id)[:count]] = True
    return selected


def safe_ratio(numerator: float, denominator: float):
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def ap_from_order(labels: np.ndarray, order: np.ndarray) -> float:
    ordered = np.asarray(labels, dtype=np.uint8)[order]
    positive_count = int(ordered.sum())
    if positive_count == 0:
        return 0.0
    cumulative = np.cumsum(ordered, dtype=np.int64)
    precision = cumulative / np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(np.sum(precision[ordered == 1]) / positive_count)


def repaired_order(base_order: np.ndarray, labels: np.ndarray, selected: np.ndarray, mode: str) -> np.ndarray:
    positive = labels == 1
    negative = ~positive
    selected_positive = selected & positive if mode in {"positive", "both"} else np.zeros(labels.size, dtype=bool)
    selected_negative = selected & negative if mode in {"negative", "both"} else np.zeros(labels.size, dtype=bool)
    ordered_selected_positive = base_order[selected_positive[base_order]]
    ordered_selected_negative = base_order[selected_negative[base_order]]
    selected_for_middle = selected_positive | selected_negative
    middle = base_order[~selected_for_middle[base_order]]
    if mode == "positive":
        return np.concatenate((ordered_selected_positive, middle))
    if mode == "negative":
        return np.concatenate((middle, ordered_selected_negative))
    if mode == "both":
        return np.concatenate((ordered_selected_positive, middle, ordered_selected_negative))
    raise ValueError(mode)


def oracle_bundle(labels: np.ndarray, baseline_ap: float, base_order: np.ndarray, selected: np.ndarray) -> dict:
    result = {}
    for mode, name in (("positive", "positive_only"), ("negative", "negative_only"), ("both", "both")):
        ap = ap_from_order(labels, repaired_order(base_order, labels, selected, mode))
        delta = float(ap - baseline_ap)
        result[name] = {
            "ap": ap,
            "delta": delta,
            "delta_pp": float(100.0 * delta),
        }
    return result


def subset_mass(subset: np.ndarray, positive: np.ndarray, c_ap: np.ndarray, r_pos_full: np.ndarray, r_neg_full: np.ndarray) -> dict:
    subset_positive = subset & positive
    subset_negative = subset & ~positive
    return {
        "n_pixels": int(subset.sum()),
        "selected_positive_fraction": safe_ratio(float(subset_positive.sum()), float(subset.sum())),
        "positive_C_AP_mass": safe_ratio(float(np.nansum(c_ap[subset_positive])), float(np.nansum(c_ap[positive]))),
        "positive_R_pos_mass": safe_ratio(float(np.nansum(r_pos_full[subset_positive])), float(np.nansum(r_pos_full[positive]))),
        "negative_R_neg_mass": safe_ratio(float(np.nansum(r_neg_full[subset_negative])), float(np.nansum(r_neg_full[~positive]))),
    }


def selector_metrics(selected: np.ndarray, positive: np.ndarray, c_ap: np.ndarray, r_pos_full: np.ndarray, r_neg_full: np.ndarray, baseline_ap: float, labels: np.ndarray, base_order: np.ndarray, signal: np.ndarray, pixel_id: np.ndarray) -> dict:
    selected_positive = selected & positive
    selected_negative = selected & ~positive
    oracle = oracle_bundle(labels, baseline_ap, base_order, selected)
    extreme = {}
    for fraction in EXTREME_FRACTIONS:
        tag = f"top_{int(fraction * 100)}pct"
        n_pos = max(1, int(np.ceil(fraction * int(positive.sum()))))
        n_neg = max(1, int(np.ceil(fraction * int((~positive).sum()))))
        pos_indices = np.flatnonzero(positive)
        neg_indices = np.flatnonzero(~positive)
        pos_severe = pos_indices[stable_desc_order(c_ap[pos_indices], pixel_id[pos_indices])[:n_pos]]
        neg_severe = neg_indices[stable_desc_order(r_neg_full[neg_indices], pixel_id[neg_indices])[:n_neg]]
        extreme[tag] = {
            "positive_C_AP_selector_recall": float(np.sum(selected[pos_severe]) / n_pos),
            "negative_R_neg_selector_recall": float(np.sum(selected[neg_severe]) / n_neg),
            "positive_subset_size": n_pos,
            "negative_subset_size": n_neg,
        }
    return {
        "selected_count": int(selected.sum()),
        "selected_positive_fraction": safe_ratio(float(selected_positive.sum()), float(selected.sum())),
        "positive_AP_damage_capture_at_20pct": safe_ratio(float(np.nansum(c_ap[selected_positive])), float(np.nansum(c_ap[positive]))),
        "positive_pairwise_risk_capture_at_20pct": safe_ratio(float(np.nansum(r_pos_full[selected_positive])), float(np.nansum(r_pos_full[positive]))),
        "negative_pairwise_risk_capture_at_20pct": safe_ratio(float(np.nansum(r_neg_full[selected_negative])), float(np.nansum(r_neg_full[~positive]))),
        "oracle_AP": {
            "baseline_ap": float(baseline_ap),
            "positive_only_ap": oracle["positive_only"]["ap"],
            "positive_only_delta": oracle["positive_only"]["delta"],
            "positive_only_delta_pp": oracle["positive_only"]["delta_pp"],
            "negative_only_ap": oracle["negative_only"]["ap"],
            "negative_only_delta": oracle["negative_only"]["delta"],
            "negative_only_delta_pp": oracle["negative_only"]["delta_pp"],
            "both_ap": oracle["both"]["ap"],
            "both_delta": oracle["both"]["delta"],
            "both_delta_pp": oracle["both"]["delta_pp"],
        },
        "extreme_error_capture": extreme,
        "selection_definition": "descending selector with stable pixel_id tie-break",
        "selector_range": {"max": float(np.max(signal)), "min": float(np.min(signal))},
    }


def process_class(model, dataset, class_name: str, records: list[dict], img_size: int, device) -> dict:
    text_cache = {}
    score_parts = []
    margin_parts = []
    rank_parts = []
    logit_parts = []
    label_parts = []
    pixel_id_parts = []
    max_exposure_parity = 0.0
    pixels_per_image = int(img_size * img_size)
    for record in records:
        raw = dataset[record["source_index"]]
        item = predict_one(model, raw, "VisA", class_name, img_size, text_cache, device)
        score = item["score"].reshape(-1).astype(np.float32, copy=False)
        margin = item["final_margin"].reshape(-1).astype(np.float32, copy=False)
        d_rank = item["D_rank"].reshape(-1).astype(np.float32, copy=False)
        d_logit = item["D_logit"].reshape(-1).astype(np.float32, copy=False)
        labels = item["target"].reshape(-1).astype(np.uint8, copy=False)
        if score.size != pixels_per_image:
            raise RuntimeError("ACTIONABILITY_TARGET_PARITY_INVALID: unexpected image pixel count")
        pixel_id = np.int64(record["source_index"]) * pixels_per_image + np.arange(score.size, dtype=np.int64)
        score_parts.append(score)
        margin_parts.append(margin)
        rank_parts.append(d_rank)
        logit_parts.append(d_logit)
        label_parts.append(labels)
        pixel_id_parts.append(pixel_id)
        max_exposure_parity = max(max_exposure_parity, float(item["parity"]["predictor_max_abs_probability_error"]))
        del item, raw

    scores = np.concatenate(score_parts)
    margins = np.concatenate(margin_parts)
    d_rank = np.concatenate(rank_parts)
    d_logit = np.concatenate(logit_parts)
    labels = np.concatenate(label_parts)
    pixel_id = np.concatenate(pixel_id_parts)
    positive = labels == 1
    negative = ~positive
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(margins)):
        raise RuntimeError("ACTIONABILITY_TARGET_PARITY_INVALID: non-finite deployed pixel value")
    baseline_auc, baseline_ap = exact_auc_ap(scores, labels)
    r_pos, r_neg = pairwise_risks(scores, labels)
    c_ap = ap_contamination(scores, labels)
    r_pos_full = np.full(scores.size, np.nan, dtype=np.float64)
    r_neg_full = np.full(scores.size, np.nan, dtype=np.float64)
    r_pos_full[positive] = r_pos
    r_neg_full[negative] = r_neg
    parity = {
        "mean_positive_C_AP": float(np.nanmean(c_ap[positive])),
        "one_minus_baseline_ap": float(1.0 - baseline_ap),
        "mean_positive_C_AP_error": abs(float(np.nanmean(c_ap[positive])) - (1.0 - baseline_ap)),
        "mean_positive_R_pos": float(np.nanmean(r_pos)),
        "mean_negative_R_neg": float(np.nanmean(r_neg)),
        "one_minus_baseline_auroc": float(1.0 - baseline_auc),
        "positive_pairwise_identity_error": abs(float(np.nanmean(r_pos)) - (1.0 - baseline_auc)),
        "negative_pairwise_identity_error": abs(float(np.nanmean(r_neg)) - (1.0 - baseline_auc)),
        "predictor_exposure_max_abs_probability_error": max_exposure_parity,
    }
    if max(parity["mean_positive_C_AP_error"], parity["positive_pairwise_identity_error"], parity["negative_pairwise_identity_error"]) > PARITY_TOL:
        raise RuntimeError("ACTIONABILITY_TARGET_PARITY_INVALID: " + json.dumps(parity, sort_keys=True))

    signals = {
        "D_rank": d_rank,
        "U_conf": -np.abs(margins),
        "D_logit": d_logit,
    }
    n_pixels = scores.size
    k20 = int(np.ceil(PRIMARY_BUDGET * n_pixels))
    selections = {name: select_top(signal, pixel_id, k20) for name, signal in signals.items()}
    selector_rows = {
        name: selector_metrics(selected, positive, c_ap, r_pos_full, r_neg_full, baseline_ap, labels, np.argsort(-scores, kind="mergesort"), signal, pixel_id)
        for name, (selected, signal) in ((name, (selections[name], signals[name])) for name in signals)
    }
    overlap = {}
    for left, right, key in (("D_rank", "U_conf", "rank_conf"), ("D_rank", "D_logit", "rank_logit"), ("U_conf", "D_logit", "conf_logit")):
        left_set = selections[left]
        right_set = selections[right]
        intersection = left_set & right_set
        union = left_set | right_set
        overlap[key] = {
            "jaccard": safe_ratio(float(intersection.sum()), float(union.sum())),
            "rank_only" if key == "rank_conf" else "left_only": subset_mass(left_set & ~right_set, positive, c_ap, r_pos_full, r_neg_full),
            "conf_only" if key == "rank_conf" else "right_only": subset_mass(right_set & ~left_set, positive, c_ap, r_pos_full, r_neg_full),
            "intersection": subset_mass(intersection, positive, c_ap, r_pos_full, r_neg_full),
        }
    k10 = int(np.ceil(0.10 * n_pixels))
    union_selection = select_top(d_rank, pixel_id, k10) | select_top(signals["U_conf"], pixel_id, k10)
    k_union = int(union_selection.sum())
    matched_rank = select_top(d_rank, pixel_id, k_union)
    matched_conf = select_top(signals["U_conf"], pixel_id, k_union)
    base_order = np.argsort(-scores, kind="mergesort")
    union_oracle = oracle_bundle(labels, baseline_ap, base_order, union_selection)["both"]
    rank_matched_oracle = oracle_bundle(labels, baseline_ap, base_order, matched_rank)["both"]
    conf_matched_oracle = oracle_bundle(labels, baseline_ap, base_order, matched_conf)["both"]
    complementarity = {
        "union_selected_count": k_union,
        "union_actual_coverage": float(k_union / n_pixels),
        "union_oracle_AP": union_oracle["ap"],
        "union_oracle_delta": union_oracle["delta"],
        "union_oracle_delta_pp": union_oracle["delta_pp"],
        "rank_matched_union_budget_AP": rank_matched_oracle["ap"],
        "rank_matched_union_budget_delta": rank_matched_oracle["delta"],
        "rank_matched_union_budget_delta_pp": rank_matched_oracle["delta_pp"],
        "conf_matched_union_budget_AP": conf_matched_oracle["ap"],
        "conf_matched_union_budget_delta": conf_matched_oracle["delta"],
        "conf_matched_union_budget_delta_pp": conf_matched_oracle["delta_pp"],
    }
    result = {
        "class": class_name,
        "n_images": len(records),
        "n_pixels": int(n_pixels),
        "baseline_auroc": float(baseline_auc),
        "baseline_ap": float(baseline_ap),
        "target_parity": parity,
        "selectors": selector_rows,
        "overlap": overlap,
        "complementarity": complementarity,
    }
    del scores, margins, d_rank, d_logit, labels, pixel_id, r_pos_full, r_neg_full, c_ap
    del score_parts, margin_parts, rank_parts, logit_parts, label_parts, pixel_id_parts
    gc.collect()
    return result


def metric_summary(values, unit: str = "class") -> dict:
    result = aggregate_values(values, seed=9100 + sum(ord(c) for c in unit))
    result["unit"] = unit
    result["bootstrap_replicates"] = 2000
    return result


def summarize_rows(rows: list[dict]) -> dict:
    def values(path):
        out = []
        for row in rows:
            value = row
            for part in path.split("."):
                value = value[part]
            out.append(value)
        return out

    selector_summary = {}
    for name in ("D_rank", "U_conf", "D_logit"):
        selector_summary[name] = {
            "selected_positive_fraction_at_20pct": metric_summary(values(f"selectors.{name}.selected_positive_fraction")),
            "positive_AP_damage_capture_at_20pct": metric_summary(values(f"selectors.{name}.positive_AP_damage_capture_at_20pct")),
            "positive_pairwise_risk_capture_at_20pct": metric_summary(values(f"selectors.{name}.positive_pairwise_risk_capture_at_20pct")),
            "negative_pairwise_risk_capture_at_20pct": metric_summary(values(f"selectors.{name}.negative_pairwise_risk_capture_at_20pct")),
            "oracle_AP": {
                key: metric_summary(values(f"selectors.{name}.oracle_AP.{key}"))
                for key in ("baseline_ap", "positive_only_ap", "positive_only_delta", "positive_only_delta_pp", "negative_only_ap", "negative_only_delta", "negative_only_delta_pp", "both_ap", "both_delta", "both_delta_pp")
            },
            "extreme_error_capture": {
                tag: {
                    "positive_C_AP_selector_recall": metric_summary(values(f"selectors.{name}.extreme_error_capture.{tag}.positive_C_AP_selector_recall")),
                    "negative_R_neg_selector_recall": metric_summary(values(f"selectors.{name}.extreme_error_capture.{tag}.negative_R_neg_selector_recall")),
                }
                for tag in ("top_1pct", "top_5pct", "top_10pct")
            },
        }
    overlap_summary = {}
    for key in ("rank_conf", "rank_logit", "conf_logit"):
        overlap_summary[key] = {
            "jaccard": metric_summary(values(f"overlap.{key}.jaccard")),
        }
        subset_names = ("rank_only", "conf_only", "intersection") if key == "rank_conf" else ("left_only", "right_only", "intersection")
        for subset in subset_names:
            overlap_summary[key][subset] = {
                metric: metric_summary(values(f"overlap.{key}.{subset}.{metric}"))
                for metric in ("n_pixels", "selected_positive_fraction", "positive_C_AP_mass", "positive_R_pos_mass", "negative_R_neg_mass")
            }
    comp_summary = {
        key: metric_summary(values(f"complementarity.{key}"))
        for key in ("union_actual_coverage", "union_oracle_AP", "union_oracle_delta", "union_oracle_delta_pp", "rank_matched_union_budget_AP", "rank_matched_union_budget_delta", "rank_matched_union_budget_delta_pp", "conf_matched_union_budget_AP", "conf_matched_union_budget_delta", "conf_matched_union_budget_delta_pp")
    }
    consistency = {}
    for name in ("D_rank", "U_conf", "D_logit"):
        deltas = [row["selectors"][name]["oracle_AP"]["both_delta"] for row in rows]
        consistency[name] = {
            "positive_oracle_advantage_classes": int(sum(delta > EPS for delta in deltas)),
            "neutral_oracle_advantage_classes": int(sum(abs(delta) <= EPS for delta in deltas)),
            "opposed_oracle_advantage_classes": int(sum(delta < -EPS for delta in deltas)),
            "total_classes": len(deltas),
        }
    union_deltas = [row["complementarity"]["union_oracle_delta"] for row in rows]
    rank_matched = [row["complementarity"]["rank_matched_union_budget_delta"] for row in rows]
    conf_matched = [row["complementarity"]["conf_matched_union_budget_delta"] for row in rows]
    comp_summary["union_beats_both_matched_controls"] = {
        "classes": int(sum(u > max(r, c) + EPS for u, r, c in zip(union_deltas, rank_matched, conf_matched))),
        "total_classes": len(rows),
    }
    return {
        "class_count": len(rows),
        "pixel_unit": "pixel arrays are reduced within class; pixels are not statistical replicates",
        "selectors": selector_summary,
        "overlap": overlap_summary,
        "complementarity": comp_summary,
        "class_consistency": consistency,
        "target_parity": {
            "max_mean_positive_C_AP_error": max(row["target_parity"]["mean_positive_C_AP_error"] for row in rows),
            "max_positive_pairwise_identity_error": max(row["target_parity"]["positive_pairwise_identity_error"] for row in rows),
            "max_negative_pairwise_identity_error": max(row["target_parity"]["negative_pairwise_identity_error"] for row in rows),
            "max_predictor_exposure_probability_error": max(row["target_parity"]["predictor_exposure_max_abs_probability_error"] for row in rows),
            "status": "PASS",
        },
    }


def flatten_class(row: dict) -> dict:
    out = {
        "class": row["class"],
        "n_images": row["n_images"],
        "n_pixels": row["n_pixels"],
        "baseline_ap": row["baseline_ap"],
        "baseline_auroc": row["baseline_auroc"],
    }
    for prefix, name in (("rank", "D_rank"), ("conf", "U_conf"), ("logit", "D_logit")):
        metric = row["selectors"][name]
        out[f"{prefix}_selected_positive_fraction"] = metric["selected_positive_fraction"]
        out[f"{prefix}_pos_cap_capture20"] = metric["positive_AP_damage_capture_at_20pct"]
        out[f"{prefix}_rpos_capture20"] = metric["positive_pairwise_risk_capture_at_20pct"]
        out[f"{prefix}_rneg_capture20"] = metric["negative_pairwise_risk_capture_at_20pct"]
        out[f"{prefix}_oracle_positive_delta"] = metric["oracle_AP"]["positive_only_delta"]
        out[f"{prefix}_oracle_positive_delta_pp"] = metric["oracle_AP"]["positive_only_delta_pp"]
        out[f"{prefix}_oracle_negative_delta"] = metric["oracle_AP"]["negative_only_delta"]
        out[f"{prefix}_oracle_negative_delta_pp"] = metric["oracle_AP"]["negative_only_delta_pp"]
        out[f"{prefix}_oracle_both_delta"] = metric["oracle_AP"]["both_delta"]
        out[f"{prefix}_oracle_both_delta_pp"] = metric["oracle_AP"]["both_delta_pp"]
        for tag in ("top_1pct", "top_5pct", "top_10pct"):
            out[f"{prefix}_{tag}_positive_C_AP_recall"] = metric["extreme_error_capture"][tag]["positive_C_AP_selector_recall"]
            out[f"{prefix}_{tag}_negative_R_neg_recall"] = metric["extreme_error_capture"][tag]["negative_R_neg_selector_recall"]
    out["jaccard_rank_conf"] = row["overlap"]["rank_conf"]["jaccard"]
    out["jaccard_rank_logit"] = row["overlap"]["rank_logit"]["jaccard"]
    out["jaccard_conf_logit"] = row["overlap"]["conf_logit"]["jaccard"]
    for key, prefix in (("rank_only", "rank_conf_rank_only"), ("conf_only", "rank_conf_conf_only"), ("intersection", "rank_conf_intersection")):
        subset = row["overlap"]["rank_conf"][key]
        for metric in ("n_pixels", "selected_positive_fraction", "positive_C_AP_mass", "positive_R_pos_mass", "negative_R_neg_mass"):
            out[f"{prefix}_{metric}"] = subset[metric]
    for key in ("union_actual_coverage", "union_oracle_delta", "union_oracle_delta_pp", "rank_matched_union_budget_delta", "rank_matched_union_budget_delta_pp", "conf_matched_union_budget_delta", "conf_matched_union_budget_delta_pp"):
        out[key] = row["complementarity"][key]
    return out


def write_csv_lf(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def decision_from_evidence(summary: dict, rows: list[dict]) -> dict:
    rank = summary["selectors"]["D_rank"]
    conf = summary["selectors"]["U_conf"]
    rank_pos = rank["positive_AP_damage_capture_at_20pct"]["median"]
    rank_rpos = rank["positive_pairwise_risk_capture_at_20pct"]["median"]
    rank_rneg = rank["negative_pairwise_risk_capture_at_20pct"]["median"]
    rank_both = rank["oracle_AP"]["both_delta"]["median"]
    conf_both = conf["oracle_AP"]["both_delta"]["median"]
    union = summary["complementarity"]["union_beats_both_matched_controls"]
    overlap = summary["overlap"]["rank_conf"]["jaccard"]["median"]
    rank_positive_oracle = rank["oracle_AP"]["positive_only_delta"]["median"]
    rank_negative_oracle = rank["oracle_AP"]["negative_only_delta"]["median"]
    rank_positive_oracle_classes = sum(
        row["selectors"]["D_rank"]["oracle_AP"]["positive_only_delta"]
        > row["selectors"]["D_rank"]["oracle_AP"]["negative_only_delta"] + EPS
        for row in rows
    )
    rank_negative_oracle_classes = sum(
        row["selectors"]["D_rank"]["oracle_AP"]["negative_only_delta"]
        > row["selectors"]["D_rank"]["oracle_AP"]["positive_only_delta"] + EPS
        for row in rows
    )
    rank_selected_positive_fraction = rank["selected_positive_fraction_at_20pct"]["median"]
    rank_better = sum(row["selectors"]["D_rank"]["oracle_AP"]["both_delta"] > row["selectors"]["U_conf"]["oracle_AP"]["both_delta"] + EPS for row in rows)
    conf_better = sum(row["selectors"]["U_conf"]["oracle_AP"]["both_delta"] > row["selectors"]["D_rank"]["oracle_AP"]["both_delta"] + EPS for row in rows)
    if rank_both <= EPS and rank_pos > 0.20:
        terminal = "RANK_RISK_NOT_ACTIONABLE"
    elif union["classes"] >= 7 and overlap < 0.80:
        terminal = "RANK_RISK_COMPLEMENTARY_TO_CONFIDENCE"
    elif rank_positive_oracle_classes >= 8 and rank_positive_oracle > rank_negative_oracle + EPS:
        terminal = "RANK_RISK_POSITIVE_SIDE_ONLY"
    elif rank_negative_oracle_classes >= 8 and rank_negative_oracle > rank_positive_oracle + EPS:
        terminal = "RANK_RISK_NEGATIVE_SIDE_ONLY"
    elif rank_rpos > rank_rneg * 1.25 and rank_better >= conf_better:
        terminal = "RANK_RISK_POSITIVE_SIDE_ONLY"
    elif rank_rneg > rank_rpos * 1.25 and rank_better >= conf_better:
        terminal = "RANK_RISK_NEGATIVE_SIDE_ONLY"
    elif overlap >= 0.80 and union["classes"] < 7:
        terminal = "RANK_RISK_CONFIDENCE_REDUNDANT"
    else:
        terminal = "RANK_RISK_ACTIONABILITY_INCONCLUSIVE"
    if rank_positive_oracle_classes >= 8 and rank_positive_oracle > rank_negative_oracle + EPS:
        positive_side = "weak-anomaly-positive actionability"
    elif rank_negative_oracle_classes >= 8 and rank_negative_oracle > rank_positive_oracle + EPS:
        positive_side = "harmful-Normal actionability"
    else:
        positive_side = "both positive and negative sides"
    questions = {
        "Q1_why_D_rank_damage_capture_high": {
            "answer": f"D_rank selects a median positive fraction of {rank_selected_positive_fraction:.6f}, but its selected positive pixels capture {rank_rpos:.6f} of positive pairwise risk; negative pairwise-risk capture is {rank_rneg:.6f}. The concentration is therefore risk-specific rather than a large anomaly-pixel share.",
            "evidence_unit": "class",
        },
        "Q2_why_D_rank_both_oracle_lower_than_U_conf": {
            "answer": f"The D_rank BOTH-oracle AP delta median is {rank_both:.6f}, versus {conf_both:.6f} for U_conf; D_rank identifies ranking-risk mass that is not equivalent to the pixels with maximum score-repair leverage.",
            "evidence_unit": "class",
        },
        "Q3_D_rank_failure_mode": {
            "answer": f"D_rank is primarily {positive_side}: positive-only oracle delta exceeds negative-only delta in {rank_positive_oracle_classes} of {len(rows)} classes, with median deltas {rank_positive_oracle:.6f} versus {rank_negative_oracle:.6f}.",
            "evidence_unit": "class",
        },
        "Q4_U_conf_complementarity": {
            "answer": f"U_conf identifies a distinct harmful-Normal/high-repair mode: rank/confidence Jaccard is {overlap:.6f}, but the matched-budget union beats both single-selector controls in only {union['classes']} of {union['total_classes']} classes.",
            "evidence_unit": "class",
        },
        "Q5_matched_budget_union": {
            "answer": f"The fixed top-10% D_rank UNION U_conf beats both matched-budget single-selector controls in {union['classes']} of {union['total_classes']} classes.",
            "evidence_unit": "class",
        },
    }
    return {
        "terminal": terminal,
        "decision_rule": "pattern-based actionability decomposition; class is the statistical unit",
        "evidence": {
            "D_rank_both_oracle_delta_median": rank_both,
            "D_rank_positive_only_oracle_delta_median": rank_positive_oracle,
            "D_rank_negative_only_oracle_delta_median": rank_negative_oracle,
            "D_rank_positive_only_advantage_classes": rank_positive_oracle_classes,
            "D_rank_negative_only_advantage_classes": rank_negative_oracle_classes,
            "U_conf_both_oracle_delta_median": conf_both,
            "D_rank_vs_U_conf_both_oracle_advantage_classes": {"D_rank": rank_better, "U_conf": conf_better},
            "rank_conf_jaccard_median": overlap,
            "union_beats_both_matched_controls": union,
        },
        "root_questions": questions,
    }


def write_decision_md(path: Path, decision: dict, summary: dict) -> None:
    rank = summary["selectors"]["D_rank"]
    conf = summary["selectors"]["U_conf"]
    logit = summary["selectors"]["D_logit"]
    comp = summary["complementarity"]
    lines = [
        f"DECISION: {decision['terminal']}",
        "INPUT INTEGRITY: PASS; authoritative pixel cache absent; exactly one fresh inference exposure pass",
        "TARGET PARITY: PASS; mean C_AP and both pairwise inversion identities match frozen Phase5-A definitions",
        "",
        "D_RANK:",
        f"  selected_positive_fraction_at_20pct_median: {rank['selected_positive_fraction_at_20pct']['median']:.6f}",
        f"  positive_AP_damage_capture_at_20pct_median: {rank['positive_AP_damage_capture_at_20pct']['median']:.6f}",
        f"  positive_pairwise_risk_capture_at_20pct_median: {rank['positive_pairwise_risk_capture_at_20pct']['median']:.6f}",
        f"  negative_pairwise_risk_capture_at_20pct_median: {rank['negative_pairwise_risk_capture_at_20pct']['median']:.6f}",
        f"  both_oracle_AP_delta_median: {rank['oracle_AP']['both_delta']['median']:.6f}",
        "",
        "U_CONF:",
        f"  selected_positive_fraction_at_20pct_median: {conf['selected_positive_fraction_at_20pct']['median']:.6f}",
        f"  positive_AP_damage_capture_at_20pct_median: {conf['positive_AP_damage_capture_at_20pct']['median']:.6f}",
        f"  positive_pairwise_risk_capture_at_20pct_median: {conf['positive_pairwise_risk_capture_at_20pct']['median']:.6f}",
        f"  negative_pairwise_risk_capture_at_20pct_median: {conf['negative_pairwise_risk_capture_at_20pct']['median']:.6f}",
        f"  both_oracle_AP_delta_median: {conf['oracle_AP']['both_delta']['median']:.6f}",
        "",
        "D_LOGIT:",
        f"  selected_positive_fraction_at_20pct_median: {logit['selected_positive_fraction_at_20pct']['median']:.6f}",
        f"  positive_AP_damage_capture_at_20pct_median: {logit['positive_AP_damage_capture_at_20pct']['median']:.6f}",
        f"  positive_pairwise_risk_capture_at_20pct_median: {logit['positive_pairwise_risk_capture_at_20pct']['median']:.6f}",
        f"  negative_pairwise_risk_capture_at_20pct_median: {logit['negative_pairwise_risk_capture_at_20pct']['median']:.6f}",
        f"  both_oracle_AP_delta_median: {logit['oracle_AP']['both_delta']['median']:.6f}",
        "",
        "COMPLEMENTARITY:",
        f"  union_actual_coverage_median: {comp['union_actual_coverage']['median']:.6f}",
        f"  union_oracle_AP_delta_median: {comp['union_oracle_delta']['median']:.6f}",
        f"  matched_D_rank_oracle_AP_delta_median: {comp['rank_matched_union_budget_delta']['median']:.6f}",
        f"  matched_U_conf_oracle_AP_delta_median: {comp['conf_matched_union_budget_delta']['median']:.6f}",
        "",
    ]
    for key, value in decision["root_questions"].items():
        lines.append(f"{key}: {value['answer']}")
    path.write_text("\n".join(lines) + "\n")


def finalize_existing(output_root: Path) -> dict:
    provenance = provenance_gate()
    summary = json.loads((output_root / "SUMMARY.json").read_text())
    rows = summary["per_class"]
    decision = decision_from_evidence(summary, rows)
    decision.update({
        "provenance": provenance,
        "input_integrity": "PASS",
        "pixel_data_source": "one fresh inference exposure pass; no authoritative cache existed",
        "target_parity": summary["target_parity"],
        "no_training": True,
        "next": "Audit candidate SECOND-EVIDENCE sources on the identified held-out high-risk weak-positive pixels; do not start Phase5-B.",
    })
    write_json(output_root / "DECISION.json", decision)
    write_decision_md(output_root / "DECISION.md", decision, summary)
    return decision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    if args.finalize_existing:
        decision = finalize_existing(args.output_root)
        print(json.dumps({"STATUS": "Phase5-A.1 decision finalized from existing artifacts", "DECISION": decision["terminal"]}, sort_keys=True))
        return
    provenance = provenance_gate()
    config = json.loads(CONFIG.read_text())
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    configure_canonical_fp32()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(config, CHECKPOINT, device)
    architecture = build_architecture(model, config, checkpoint)
    frozen_protocol = json.loads(PROTOCOL.read_text())
    if architecture != frozen_protocol["architecture"]:
        raise RuntimeError("INPUT_PROVENANCE_MISMATCH: runtime architecture differs from frozen protocol")
    datasets, records = canonical_test_records(int(config["img_size"]))
    prior_runtime = json.loads((PHASE5_SUMMARY.parent / "RUNTIME_ESTIMATE.json").read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit_protocol = {
        "audit": "Phase5-A.1 v2 actionability decomposition of HSIR",
        "inference_only": True,
        "provenance": provenance,
        "pixel_data_source": {
            "authoritative_cache_found": False,
            "source": "exactly one fresh VisA TEST predictor exposure pass",
            "dense_cache_persisted": False,
            "forward_count_expected": EXPECTED_IMAGES,
            "prior_phase5_analysis_reused": True,
            "prior_runtime_estimate": prior_runtime,
        },
        "predictor_semantics": "reuse audit_phase5_hsir.predict_one with canonical VisA stage=test loader, exact Phase2B predictor, strict FP32",
        "selectors": {
            "D_rank": "D_rank descending",
            "D_logit": "D_logit descending",
            "U_conf": "-abs(final_margin) descending",
        },
        "selection_budget": {
            "primary_fraction": PRIMARY_BUDGET,
            "k20": "ceil(0.20 * class pixel count)",
            "union_fraction_per_selector": 0.10,
            "union_matched_budget": "top K_union by each single selector",
        },
        "tie_handling": "stable lexsort by descending selector then deterministic pixel_id=(canonical source_index * H*W + row-major y*W+x); GT is not used for selection",
        "error_objects": {
            "C_AP": "frozen Phase5-A positive AP contamination",
            "R_pos": "frozen Phase5-A positive pairwise inversion risk",
            "R_neg": "frozen Phase5-A negative pairwise inversion risk",
        },
        "oracles": "positive-only, negative-only, and both; selected pixels move across all unselected pixels while unselected relative score order is preserved",
        "no_training": True,
        "no_new_selector_learning": True,
    }
    write_json(args.output_root / "PROTOCOL.json", audit_protocol)
    rows = []
    for class_name in sorted(records):
        rows.append(process_class(model, datasets[class_name], class_name, records[class_name], int(config["img_size"]), device))
    summary = {
        "provenance": provenance,
        "architecture": architecture,
        "inference": {
            "forward_count": sum(row["n_images"] for row in rows),
            "class_count": len(rows),
            "dense_cache_persisted": False,
            "class_at_a_time": True,
        },
        "selection_budget": audit_protocol["selection_budget"],
        "tie_handling": audit_protocol["tie_handling"],
        "error_objects": audit_protocol["error_objects"],
        **summarize_rows(rows),
        "per_class": rows,
    }
    write_csv_lf(args.output_root / "PER_CLASS.csv", [flatten_class(row) for row in rows])
    write_json(args.output_root / "SUMMARY.json", summary)
    decision = decision_from_evidence(summary, rows)
    decision.update({
        "provenance": provenance,
        "input_integrity": "PASS",
        "pixel_data_source": "one fresh inference exposure pass; no authoritative cache existed",
        "target_parity": summary["target_parity"],
        "no_training": True,
        "next": "Audit candidate SECOND-EVIDENCE sources on the identified held-out high-risk weak-positive pixels; do not start Phase5-B.",
    })
    write_json(args.output_root / "DECISION.json", decision)
    write_decision_md(args.output_root / "DECISION.md", decision, summary)
    print(json.dumps({"STATUS": "Phase5-A.1 actionability audit complete", "DECISION": decision["terminal"], "FORWARD_COUNT": summary["inference"]["forward_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
