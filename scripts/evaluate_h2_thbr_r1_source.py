#!/usr/bin/env python3
"""Evaluate THBR-R1 endpoints on the frozen source-only VisA cohort."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
os.sys.path.insert(0, str(REPO))

import scripts.run_h2_thbr_r1_audit as audit
from h2_clean.precision import PrecisionPolicy
from scripts.evaluate_h2_fixed_fusion_r2_source import (
    ENDPOINT_SUBSET,
    SPLIT_IDENTITY,
    binary_metrics,
    load_endpoint,
    load_endpoint_selection,
    morphology,
    parameter_drift,
    stage_logits_from_features,
)
from h2_clean.stage_fusion import H2_EQUAL_STAGE_FUSION_WEIGHTS, fuse_stage_logits
from dataset import CLASS_NAMES, get_text_and_image_dataset
from utils import get_hybrid_soft_prompt_single_class_text_embedding


PROTOCOL_ID = "H2_THBR_R1"
ARM_E10 = audit.ARM_E10
ARM_CONTROL = "A_THBR_R1_CONTROL"
ARM_CANDIDATE = "A_THBR_R1_CANDIDATE"
ARM_ORDER = (ARM_E10, ARM_CONTROL, ARM_CANDIDATE)
E10 = audit.E10
CONTROL = REPO / "runs/h2_thbr_r1/A_THBR_R1_CONTROL/final.pth"
CANDIDATE = REPO / "runs/h2_thbr_r1/A_THBR_R1_CANDIDATE/final.pth"
CONTROL_CSV = REPO / "audit/H2_THBR_R1_CONTROL.csv"
CANDIDATE_CSV = REPO / "audit/H2_THBR_R1_CANDIDATE.csv"
OUTPUT_JSON = REPO / "audit/H2_THBR_R1_ENDPOINT_EVAL.json"
OUTPUT_CSV = REPO / "audit/H2_THBR_R1_ENDPOINT_EVAL.csv"
PAIR_SEED = 1729
PAIR_SAMPLE_SIZE = 200_000


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def stats(values: np.ndarray, quantiles=(.95, .99)) -> dict:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    result = {
        "count": int(x.size),
        "mean": float(x.mean()) if x.size else float("nan"),
        "median": float(np.median(x)) if x.size else float("nan"),
        "max": float(x.max()) if x.size else float("nan"),
    }
    for q in quantiles:
        percentage = q * 100
        label = str(int(percentage)) if float(percentage).is_integer() else str(percentage)
        result[f"p{label}"] = float(np.quantile(x, q)) if x.size else float("nan")
    return result


def region_values(scores: np.ndarray, masks: np.ndarray) -> dict[str, np.ndarray]:
    chunks = {key: [] for key in ("positive", "negative", "boundary", "interior", "near_background", "far_background")}
    for score, mask in zip(scores, masks):
        boundary, interior, near, far = morphology(mask.astype(bool))
        for key, region in (
            ("positive", mask.astype(bool)), ("negative", ~mask.astype(bool)),
            ("boundary", boundary), ("interior", interior),
            ("near_background", near), ("far_background", far),
        ):
            if region.any():
                chunks[key].append(score[region].astype(np.float32))
    return {key: np.concatenate(value) if value else np.empty(0, dtype=np.float32) for key, value in chunks.items()}


def pair_indices(regions: dict[str, np.ndarray]) -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(PAIR_SEED)
    n = PAIR_SAMPLE_SIZE
    return {
        "seed": PAIR_SEED,
        "sample_size": n,
        "near_positive_near": rng.integers(regions["near_background"].size, size=n, dtype=np.int64),
        "near_positive_positive": rng.integers(regions["positive"].size, size=n, dtype=np.int64),
        "near_interior_near": rng.integers(regions["near_background"].size, size=n, dtype=np.int64),
        "near_interior_interior": rng.integers(regions["interior"].size, size=n, dtype=np.int64),
    }


def inversion_rates(regions: dict[str, np.ndarray], pairs: dict) -> dict:
    near = regions["near_background"].astype(np.float64)
    positive = regions["positive"].astype(np.float64)
    interior = regions["interior"].astype(np.float64)
    near_pos = near[pairs["near_positive_near"]]
    pos = positive[pairs["near_positive_positive"]]
    near_int = near[pairs["near_interior_near"]]
    interior_values = interior[pairs["near_interior_interior"]]
    return {
        "near_background_above_positive_median_fraction": float((near > np.median(positive)).mean()),
        "near_background_above_interior_median_fraction": float((near > np.median(interior)).mean()),
        "near_background_positive_pairwise_inversion_rate": float((near_pos > pos).mean()),
        "near_background_interior_pairwise_inversion_rate": float((near_int > interior_values).mean()),
        "pairwise_sampling": {"seed": PAIR_SEED, "sample_size": PAIR_SAMPLE_SIZE, "same_pairs_for_all_arms": True},
    }


def top_p_composition(score: np.ndarray, mask: np.ndarray) -> dict:
    boundary, _, near, far = morphology(mask.astype(bool))
    p = int(mask.astype(bool).sum())
    if p <= 0:
        return {"P": 0, "anomaly_fraction": float("nan"), "anomaly_boundary_fraction": float("nan"), "near_background_fraction": float("nan"), "far_background_fraction": float("nan")}
    order = np.argsort(-score.reshape(-1), kind="stable")[:p]
    flat = lambda region: float(region.reshape(-1)[order].mean())
    return {"P": p, "anomaly_fraction": flat(mask.astype(bool)), "anomaly_boundary_fraction": flat(boundary), "near_background_fraction": flat(near), "far_background_fraction": flat(far)}


def matched_violation(score: np.ndarray, mask: np.ndarray) -> dict:
    anomaly = mask.astype(bool)
    background = ~anomaly
    if not anomaly.any() or not background.any():
        return {"K": 0, "violation_fraction": float("nan")}
    k = min(int(anomaly.sum()), int(background.sum()))
    hard = np.sort(score[background])[-k:][::-1]
    weak = np.sort(score[anomaly])[:k]
    return {"K": k, "violation_fraction": float((hard > weak).mean()), "hard_background_mean": float(hard.mean()), "weak_anomaly_mean": float(weak.mean())}


def geometry(reference: np.ndarray, live: np.ndarray) -> list[dict]:
    values = []
    for stage in range(3):
        left = reference[stage].astype(np.float64)
        right = live[stage].astype(np.float64)
        left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
        right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
        cosines = (left_norm * right_norm).sum(axis=1)
        centered_left = left - left.mean(axis=0, keepdims=True)
        centered_right = right - right.mean(axis=0, keepdims=True)
        cross = centered_left.T @ centered_right
        denom = np.linalg.norm(centered_left.T @ centered_left, "fro") * np.linalg.norm(centered_right.T @ centered_right, "fro")
        cka = float(np.linalg.norm(cross, "fro") ** 2 / denom) if denom else float("nan")
        values.append({
            "stage": stage + 1,
            "pooled_feature_cosine_to_e10_mean": float(cosines.mean()),
            "pooled_feature_cosine_to_e10_median": float(np.median(cosines)),
            "pooled_feature_cosine_to_e10_p01": float(np.quantile(cosines, .01)),
            "linear_cka_to_e10": cka,
        })
    return values


def evaluate_arm(model, checkpoint: Path, selected: list[dict], by_category: dict[str, list[dict]], device, policy, e10_pooled, e10_state) -> tuple[dict, np.ndarray]:
    payload = load_endpoint(model, checkpoint)
    datasets = get_text_and_image_dataset("VisA", audit.IMG, "test")
    expected_names = [row["file_name"] for row in selected]
    names, masks, finals, stages, pooled_chunks, late_ratios = [], [], [], [], [[] for _ in range(3)], []
    with torch.no_grad():
        for category in CLASS_NAMES["VisA"]:
            dataset = datasets[category]
            index_by_name = {meta["image_path"]: index for index, meta in enumerate(dataset.meta)}
            indices = []
            for row in by_category[category]:
                index = index_by_name[row["file_name"]]
                expected_label = 1 if row["label_type"] == "anomaly" else 0
                if int(dataset.meta[index]["label"]) != expected_label:
                    raise RuntimeError(f"endpoint label mismatch: {row['file_name']}")
                indices.append(index)
            loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False, num_workers=0)
            text, _, _ = get_hybrid_soft_prompt_single_class_text_embedding(model, "VisA", category, device, return_kg=False)
            for batch in loader:
                image = batch["image"].to(device)
                mask = batch["mask"][:, 0].numpy().astype(np.uint8)
                names.extend(list(batch["file_name"]))
                with policy.autocast(device):
                    seg_tokens, _ = model(image)
                    vision = torch.stack(seg_tokens)
                    logits = stage_logits_from_features(model, vision, text)
                    fused = fuse_stage_logits(logits, H2_EQUAL_STAGE_FUSION_WEIGHTS)
                    final = F.softmax(fused, dim=1)[:, 1]
                    stage_probs = F.softmax(logits, dim=2)[:, :, 1]
                    pooled = F.normalize(vision.float(), dim=-1).mean(dim=2)
                stage_abs = logits.float().abs().mean(dim=(2, 3, 4))
                weight = torch.as_tensor(H2_EQUAL_STAGE_FUSION_WEIGHTS, device=device, dtype=stage_abs.dtype)
                denominator = (weight[:, None] * stage_abs).sum(dim=0).clamp_min(1e-12)
                late = (weight[1] * stage_abs[1] + weight[2] * stage_abs[2]) / denominator
                masks.append(mask)
                finals.append(final.float().cpu().numpy())
                stages.append(stage_probs.float().cpu().numpy().transpose(1, 0, 2, 3))
                late_ratios.append(late.float().cpu().numpy())
                for stage in range(3):
                    pooled_chunks[stage].append(pooled[stage].cpu().numpy())
    if names != expected_names:
        raise RuntimeError("endpoint ordering mismatch")
    masks_array = np.concatenate(masks, axis=0)
    finals_array = np.concatenate(finals, axis=0)
    stages_array = np.concatenate(stages, axis=0)
    pooled_array = np.stack([np.concatenate(chunk, axis=0) for chunk in pooled_chunks], axis=0)
    late_array = np.concatenate(late_ratios, axis=0)
    regions = region_values(finals_array, masks_array)
    pairs = pair_indices(regions) if e10_pooled is None else pair_indices(regions)
    anomalous = [top_p_composition(score, mask) for score, mask, row in zip(finals_array, masks_array, selected) if row["label_type"] == "anomaly"]
    violations = [matched_violation(score, mask) for score, mask, row in zip(finals_array, masks_array, selected) if row["label_type"] == "anomaly"]
    result = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": audit.sha256_file(checkpoint),
        "sample_count": len(names), "anomalous_image_count": len(anomalous),
        "ranking": {"final": binary_metrics(finals_array, masks_array), **{f"stage_{stage + 1}": binary_metrics(stages_array[:, stage], masks_array) for stage in range(3)}},
        "coverage": {key: stats(value, quantiles=(.95, .99, .995) if key in ("negative", "near_background", "far_background") else (.95, .99)) for key, value in regions.items()},
        "background_inversions": inversion_rates(regions, pairs),
        "matched_cardinality": {
            "definition": "For each anomalous image, K=min(GT anomaly pixels, GT background pixels); compare top-K background scores with bottom-K anomaly scores.",
            "violation_fraction": stats(np.array([x["violation_fraction"] for x in violations], dtype=np.float64), quantiles=(.95, .99)),
            "K": stats(np.array([x["K"] for x in violations], dtype=np.float64), quantiles=(.95, .99)),
            "hard_background_mean": stats(np.array([x["hard_background_mean"] for x in violations], dtype=np.float64), quantiles=(.95, .99)),
            "weak_anomaly_mean": stats(np.array([x["weak_anomaly_mean"] for x in violations], dtype=np.float64), quantiles=(.95, .99)),
        },
        "top_P_composition": {
            "definition": "P equals each anomalous image's GT anomaly-pixel count; report composition of the top-P predicted anomaly-score pixels.",
            "mean": {key: float(np.nanmean([row[key] for row in anomalous])) for key in ("anomaly_fraction", "anomaly_boundary_fraction", "near_background_fraction", "far_background_fraction")},
            "median": {key: float(np.nanmedian([row[key] for row in anomalous])) for key in ("anomaly_fraction", "anomaly_boundary_fraction", "near_background_fraction", "far_background_fraction")},
            "per_image": anomalous,
        },
        "R_late": {"mean": float(late_array.mean()), "median": float(np.median(late_array)), "p99": float(np.quantile(late_array, .99))},
        "parameter_drift_from_e10": parameter_drift(e10_state, payload.get("model_state", payload)["image_adapter"]),
        "feature_geometry_drift_from_e10": geometry(e10_pooled, pooled_array) if e10_pooled is not None else None,
        "endpoint_weights": list(H2_EQUAL_STAGE_FUSION_WEIGHTS),
        "endpoint_finite": bool(np.isfinite(finals_array).all() and np.isfinite(stages_array).all() and np.isfinite(pooled_array).all()),
    }
    return result, pooled_array


def delta(candidate: dict, control: dict) -> dict:
    result = {}
    for key in ("final", "stage_1", "stage_2", "stage_3"):
        result[f"{key}_auroc"] = candidate["ranking"][key]["auroc"] - control["ranking"][key]["auroc"]
        result[f"{key}_ap"] = candidate["ranking"][key]["ap"] - control["ranking"][key]["ap"]
    for region in ("positive", "interior", "boundary", "negative", "near_background", "far_background"):
        for stat_name in ("mean", "median", "p95", "p99", "p995", "max"):
            left = candidate["coverage"][region].get(stat_name)
            right = control["coverage"][region].get(stat_name)
            if left is not None and right is not None:
                result[f"{region}_{stat_name}"] = left - right
    for metric in ("near_background_positive_pairwise_inversion_rate", "near_background_interior_pairwise_inversion_rate", "near_background_above_positive_median_fraction", "near_background_above_interior_median_fraction"):
        result[metric] = candidate["background_inversions"][metric] - control["background_inversions"][metric]
    for metric in ("mean", "median", "p95", "p99"):
        result[f"matched_violation_{metric}"] = candidate["matched_cardinality"]["violation_fraction"][metric] - control["matched_cardinality"]["violation_fraction"][metric]
    for metric in ("anomaly_fraction", "anomaly_boundary_fraction", "near_background_fraction", "far_background_fraction"):
        result[f"top_P_{metric}"] = candidate["top_P_composition"]["mean"][metric] - control["top_P_composition"]["mean"][metric]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO / "runs/h2_thbr_r1"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    for path in (E10, CONTROL, CANDIDATE):
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    policy = PrecisionPolicy("fp16")
    selected, by_category = load_endpoint_selection()
    split = json.loads(SPLIT_IDENTITY.read_text())
    if split["endpoint_eval_count"] != 96 or split["intersection_count"] != 0:
        raise RuntimeError("endpoint split identity mismatch")
    paths = {ARM_E10: E10, ARM_CONTROL: CONTROL, ARM_CANDIDATE: CANDIDATE}
    e10_payload = torch.load(E10, map_location="cpu", weights_only=False)
    e10_state = e10_payload.get("model_state", e10_payload)["image_adapter"]
    model = audit.make_model(e10_payload, device, training_graph=False)
    results = {}
    e10_pooled = None
    for arm in ARM_ORDER:
        result, pooled = evaluate_arm(model, paths[arm], selected, by_category, device, policy, e10_pooled, e10_state)
        if arm == ARM_E10:
            e10_pooled = pooled
            result["feature_geometry_drift_from_e10"] = geometry(e10_pooled, e10_pooled)
        results[arm] = result
        del pooled
        torch.cuda.empty_cache()
    control = results[ARM_CONTROL]
    candidate = results[ARM_CANDIDATE]
    control_summary = json.loads((CONTROL.parent / "summary.json").read_text())
    candidate_summary = json.loads((CANDIDATE.parent / "summary.json").read_text())
    control_rows = list(csv.DictReader(CONTROL_CSV.open(newline=""))) if CONTROL_CSV.is_file() else []
    candidate_rows = list(csv.DictReader(CANDIDATE_CSV.open(newline=""))) if CANDIDATE_CSV.is_file() else []
    identity_match = len(control_rows) == len(candidate_rows) == 500 and all(
        all(left.get(key) == right.get(key) for key in audit.IDENTITY_KEYS)
        for left, right in zip(control_rows, candidate_rows)
    )
    pair_status_match = identity_match and all(
        (left.get("status", "").endswith("skip") == right.get("status", "").endswith("skip"))
        for left, right in zip(control_rows, candidate_rows)
    )
    numerical_validity = bool(
        identity_match and pair_status_match and control_summary["attempted_steps"] == candidate_summary["attempted_steps"] == 500
        and control_summary["successful_steps"] == candidate_summary["successful_steps"]
        and control_summary["extra_natural_skips"] == [] and candidate_summary["extra_natural_skips"] == []
        and control_summary["numerical_failure"] is None and candidate_summary["numerical_failure"] is None
        and all(result["endpoint_finite"] for result in results.values())
    )
    output = {
        "protocol_id": PROTOCOL_ID,
        "scope": "equal-fusion inference on the frozen 96-image VisA endpoint cohort; no Medical, MVTec, or target inference",
        "endpoint_subset_sha256": audit.sha256_file(ENDPOINT_SUBSET),
        "endpoint_sample_count": 96,
        "split_identity": split,
        "arms": results,
        "train_fusion_weights": list(H2_EQUAL_STAGE_FUSION_WEIGHTS),
        "exact_batch_identity_match": identity_match,
        "paired_skip_status_match": pair_status_match,
        "numerical_validity": numerical_validity,
        "training_summaries": {ARM_CONTROL: control_summary, ARM_CANDIDATE: candidate_summary},
        "candidate_minus_control": delta(candidate, control),
        "checkpoint_identities": {arm: {"path": str(path), "sha256": audit.sha256_file(path)} for arm, path in paths.items()},
    }
    json_dump(OUTPUT_JSON, output)
    rows = []
    for arm in ARM_ORDER:
        result = results[arm]
        for metric_name, metric in result["ranking"].items():
            rows.append({"arm": arm, "record": "ranking", "metric": metric_name, **metric})
        for region, metric in result["coverage"].items():
            rows.append({"arm": arm, "record": "coverage", "metric": region, **metric})
        for metric_name, metric in result["background_inversions"].items():
            if isinstance(metric, (int, float)):
                rows.append({"arm": arm, "record": "background_inversions", "metric": metric_name, "value": metric})
        for metric_name, metric in result["matched_cardinality"].items():
            if isinstance(metric, dict):
                for stat_name, value in metric.items():
                    rows.append({"arm": arm, "record": "matched_cardinality", "metric": f"{metric_name}.{stat_name}", "value": value})
        for metric_name, metric in result["top_P_composition"]["mean"].items():
            rows.append({"arm": arm, "record": "top_P_composition", "metric": metric_name, "value": metric})
        for metric_name, metric in result["parameter_drift_from_e10"]["requested_families"].items():
            rows.append({"arm": arm, "record": "parameter_drift", "metric": metric_name, **metric})
        for item in result["feature_geometry_drift_from_e10"]:
            rows.append({"arm": arm, "record": "feature_geometry", "metric": f"stage_{item['stage']}", **item})
    fields = sorted({key for row in rows for key in row})
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS" if numerical_validity else "INVALID_NUMERICAL", "output": str(OUTPUT_JSON), "csv": str(OUTPUT_CSV), "sample_count": 96, "exact_batch_identity_match": identity_match}, sort_keys=True))
    if not numerical_validity:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
