#!/usr/bin/env python3
"""Diagnostic-only frozen R2 2x2 cross-evaluation.

This script performs inference only.  It never constructs an optimizer, calls
backward, or updates a checkpoint.  The four cells differ only in which
already-finished endpoint state is loaded and which frozen stage-fusion tuple
is used for inference.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset import CLASS_NAMES, get_text_and_image_dataset
from h2_clean.precision import PrecisionPolicy
from h2_clean.stage_fusion import (
    H2_EQUAL_STAGE_FUSION_WEIGHTS,
    H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS,
    fuse_stage_logits,
)
from utils import get_hybrid_soft_prompt_single_class_text_embedding
from scripts.evaluate_h2_fixed_fusion_r2_source import (
    ENDPOINT_SUBSET,
    IMG,
    START_CHECKPOINT,
    binary_metrics,
    load_endpoint,
    load_endpoint_selection,
    make_model,
    morphology,
    stage_logits_from_features,
)


PROTOCOL_ID = "H2_POST_R2_CAUSAL_DECOMPOSITION_2X2_FROZEN_CROSS_EVALUATION"
R2_ENDPOINT_JSON = REPO / "audit/H2_FUSION_R2_ENDPOINT_EVAL.json"
R2_CONTROL_CSV = REPO / "audit/H2_FUSION_R2_CONTROL.csv"
R2_CANDIDATE_CSV = REPO / "audit/H2_FUSION_R2_CANDIDATE.csv"
SPLIT_IDENTITY = REPO / "audit/H2_FUSION_SPLIT_IDENTITY.json"
RUN_ROOT = Path("/workspace/h2_fixed_fusion_bounded_r2")
PAIR_SEED = 1729
PAIR_SAMPLE_SIZE = 200_000
REPRO_TOLERANCE = 1.0e-6


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
    }


def region_values(scores: np.ndarray, masks: np.ndarray) -> dict[str, np.ndarray]:
    chunks = {key: [] for key in (
        "positive", "negative", "boundary", "interior", "near_background", "far_background",
    )}
    for score, mask in zip(scores, masks):
        boundary, interior, near, far = morphology(mask.astype(bool))
        for key, region in (
            ("positive", mask.astype(bool)), ("negative", ~mask.astype(bool)),
            ("boundary", boundary), ("interior", interior),
            ("near_background", near), ("far_background", far),
        ):
            if region.any():
                chunks[key].append(score[region].astype(np.float32))
    return {
        key: np.concatenate(value) if value else np.empty(0, dtype=np.float32)
        for key, value in chunks.items()
    }


def precision_recall_diagnostic(scores: np.ndarray, masks: np.ndarray) -> dict:
    x = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(masks, dtype=np.uint8).reshape(-1)
    order = np.argsort(-x, kind="stable")
    sy = y[order]
    tp = np.cumsum(sy, dtype=np.float64)
    fp = np.cumsum(1 - sy, dtype=np.float64)
    recall = tp / float(tp[-1])
    precision = tp / np.maximum(tp + fp, 1.0)
    result = {
        "ap": float((precision * sy).sum() / float(tp[-1])),
        "precision_at_recall": {},
        "recall_at_precision": {},
        "definition": {
            "precision_at_recall": "precision at the first sorted operating point whose recall reaches the requested value",
            "recall_at_precision": "maximum sorted recall among operating points whose precision reaches the requested value; null if undefined",
        },
    }
    for target in (0.25, 0.50, 0.75):
        key = f"{target:.2f}"
        indices = np.flatnonzero(recall >= target)
        result["precision_at_recall"][key] = float(precision[indices[0]]) if indices.size else None
        eligible = np.flatnonzero(precision >= target)
        result["recall_at_precision"][key] = float(recall[eligible].max()) if eligible.size else None
    return result


def make_pair_indices(regions: dict[str, np.ndarray]) -> dict[str, np.ndarray | int]:
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


def background_diagnostic(regions: dict[str, np.ndarray], pair_indices: dict) -> dict:
    positive = regions["positive"].astype(np.float64)
    interior = regions["interior"].astype(np.float64)
    near = regions["near_background"].astype(np.float64)
    far = regions["far_background"].astype(np.float64)
    negative = regions["negative"].astype(np.float64)
    boundary = regions["boundary"].astype(np.float64)
    near_pos = near[pair_indices["near_positive_near"]]
    pos_pairs = positive[pair_indices["near_positive_positive"]]
    near_int = near[pair_indices["near_interior_near"]]
    int_pairs = interior[pair_indices["near_interior_interior"]]
    return {
        "positive_vs_negative_mean_gap": float(positive.mean() - negative.mean()),
        "positive_vs_near_background_mean_gap": float(positive.mean() - near.mean()),
        "interior_vs_near_background_mean_gap": float(interior.mean() - near.mean()),
        "positive_median_minus_near_bg_p95": float(np.median(positive) - np.quantile(near, .95)),
        "interior_median_minus_near_bg_p95": float(np.median(interior) - np.quantile(near, .95)),
        "near_bg_p95": float(np.quantile(near, .95)),
        "near_bg_p99": float(np.quantile(near, .99)),
        "far_bg_p95": float(np.quantile(far, .95)),
        "far_bg_p99": float(np.quantile(far, .99)),
        "negative_p95": float(np.quantile(negative, .95)),
        "negative_p99": float(np.quantile(negative, .99)),
        "boundary_p95": float(np.quantile(boundary, .95)),
        "boundary_p99": float(np.quantile(boundary, .99)),
        "fraction_near_background_above_positive_median": float((near > np.median(positive)).mean()),
        "fraction_near_background_above_interior_median": float((near > np.median(interior)).mean()),
        "near_background_positive_pairwise_rank_inversion_rate": float((near_pos > pos_pairs).mean()),
        "near_background_interior_pairwise_rank_inversion_rate": float((near_int > int_pairs).mean()),
        "pairwise_sampling": {
            "seed": int(pair_indices["seed"]),
            "sample_size": int(pair_indices["sample_size"]),
            "same_pairs_for_all_cells": True,
        },
    }


def evaluate_cell(
    model,
    checkpoint: Path,
    weights: tuple[float, float, float],
    selected_rows: list[dict],
    selected_by_category: dict[str, list[dict]],
    device: torch.device,
    policy: PrecisionPolicy,
    pair_indices: dict | None,
) -> tuple[dict, dict[str, np.ndarray], dict | None]:
    model.stage_fusion_weights = tuple(weights)
    load_endpoint(model, checkpoint)
    model.eval()
    datasets = get_text_and_image_dataset("VisA", IMG, "test")
    expected_names = [row["file_name"] for row in selected_rows]
    names = []
    masks = []
    finals = []
    stages = []
    late_ratios = []
    with torch.no_grad():
        for category in CLASS_NAMES["VisA"]:
            rows = selected_by_category[category]
            dataset = datasets[category]
            index_by_name = {meta["image_path"]: index for index, meta in enumerate(dataset.meta)}
            indices = []
            for row in rows:
                index = index_by_name.get(row["file_name"])
                if index is None:
                    raise RuntimeError(f"endpoint file missing: {row['file_name']}")
                expected_label = 1 if row["label_type"] == "anomaly" else 0
                if int(dataset.meta[index]["label"]) != expected_label:
                    raise RuntimeError(f"endpoint label mismatch: {row['file_name']}")
                indices.append(index)
            loader = DataLoader(Subset(dataset, indices), batch_size=8, shuffle=False, num_workers=0)
            text, _, _ = get_hybrid_soft_prompt_single_class_text_embedding(
                model, "VisA", category, device, return_kg=False,
            )
            for batch in loader:
                image = batch["image"].to(device)
                mask = batch["mask"][:, 0].numpy().astype(np.uint8)
                names.extend(list(batch["file_name"]))
                with policy.autocast(device):
                    seg_tokens, _ = model(image)
                    vision = torch.stack(seg_tokens)
                    logits = stage_logits_from_features(model, vision, text)
                    fused = fuse_stage_logits(logits, weights)
                    final = F.softmax(fused, dim=1)[:, 1]
                    stage_probs = F.softmax(logits, dim=2)[:, :, 1]
                stage_abs = logits.float().abs().mean(dim=(2, 3, 4))
                weight_tensor = torch.as_tensor(weights, device=device, dtype=stage_abs.dtype)
                denominator = (weight_tensor[:, None] * stage_abs).sum(dim=0).clamp_min(1e-12)
                late = (weight_tensor[1] * stage_abs[1] + weight_tensor[2] * stage_abs[2]) / denominator
                masks.append(mask)
                finals.append(final.float().cpu().numpy())
                stages.append(stage_probs.float().cpu().numpy().transpose(1, 0, 2, 3))
                late_ratios.append(late.float().cpu().numpy())
    if names != expected_names:
        raise RuntimeError("endpoint order mismatch")
    masks_array = np.concatenate(masks, axis=0)
    finals_array = np.concatenate(finals, axis=0)
    stages_array = np.concatenate(stages, axis=0)
    late_array = np.concatenate(late_ratios, axis=0)
    regions = region_values(finals_array, masks_array)
    if pair_indices is None:
        pair_indices = make_pair_indices(regions)
    ranking = {"final": binary_metrics(finals_array, masks_array)}
    for stage in range(3):
        ranking[f"stage_{stage + 1}"] = binary_metrics(stages_array[:, stage], masks_array)
    coverage = {key: summarize(value) for key, value in regions.items()}
    result = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "weights": list(weights),
        "sample_count": len(names),
        "ranking": ranking,
        "coverage": coverage,
        "R_late": {
            "mean": float(late_array.mean()),
            "median": float(np.median(late_array)),
            "p95": float(np.quantile(late_array, .95)),
            "p99": float(np.quantile(late_array, .99)),
        },
        "background_tail": background_diagnostic(regions, pair_indices),
        "pr": precision_recall_diagnostic(finals_array, masks_array),
    }
    arrays = {
        "final": finals_array,
        "masks": masks_array,
        "stage_scores": stages_array,
        "regions": regions,
    }
    return result, arrays, pair_indices


def metric_values(cell: dict) -> dict[str, float]:
    values = {
        "final_auroc": cell["ranking"]["final"]["auroc"],
        "final_ap": cell["ranking"]["final"]["ap"],
        "stage1_auroc": cell["ranking"]["stage_1"]["auroc"],
        "stage1_ap": cell["ranking"]["stage_1"]["ap"],
        "stage2_auroc": cell["ranking"]["stage_2"]["auroc"],
        "stage2_ap": cell["ranking"]["stage_2"]["ap"],
        "stage3_auroc": cell["ranking"]["stage_3"]["auroc"],
        "stage3_ap": cell["ranking"]["stage_3"]["ap"],
        "R_late_mean": cell["R_late"]["mean"],
    }
    for region in ("positive", "interior", "boundary", "negative", "near_background", "far_background"):
        for stat in ("mean", "median", "p95", "p99"):
            values[f"{region}_{stat}"] = cell["coverage"][region][stat]
    return values


def contrasts(cells: dict[str, dict]) -> dict:
    names = ("C_EQ", "C_W", "T_EQ", "T_W")
    metrics = sorted(metric_values(cells["C_EQ"]))
    values = {name: metric_values(cells[name]) for name in names}
    output = {}
    for metric in metrics:
        output[metric] = {
            "inference_effect_control": values["C_W"][metric] - values["C_EQ"][metric],
            "training_effect_equal": values["T_EQ"][metric] - values["C_EQ"][metric],
            "training_effect_weighted": values["T_W"][metric] - values["C_W"][metric],
            "interaction": (values["T_W"][metric] - values["T_EQ"][metric]) - (values["C_W"][metric] - values["C_EQ"][metric]),
        }
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_identity() -> dict:
    endpoint = json.loads(R2_ENDPOINT_JSON.read_text())
    cells = {
        "CONTROL": Path(endpoint["arms"]["A_FUSE_SHORT_R2_CONTROL"]["checkpoint"]),
        "CANDIDATE": Path(endpoint["arms"]["A_FUSE_SHORT_R2_CANDIDATE"]["checkpoint"]),
    }
    expected = {
        "CONTROL": {"attempted_steps": 500, "successful_steps": 499, "nonfinite_grad_skips": 1},
        "CANDIDATE": {"attempted_steps": 500, "successful_steps": 499, "nonfinite_grad_skips": 1},
    }
    verified = {}
    for name, path in cells.items():
        summary = json.loads((path.parent / "summary.json").read_text())
        verified[name] = {
            "checkpoint_path": str(path),
            "checkpoint_sha256": sha256_file(path),
            "summary": {key: summary[key] for key in expected[name]},
            "expected_summary": expected[name],
            "identity_ok": path.is_file() and all(summary[key] == value for key, value in expected[name].items()),
        }
    return verified


def reproduction_check(cells: dict[str, dict]) -> dict:
    r2 = json.loads(R2_ENDPOINT_JSON.read_text())["arms"]
    checks = {}
    for cell, arm in (("C_EQ", "A_FUSE_SHORT_R2_CONTROL"), ("T_W", "A_FUSE_SHORT_R2_CANDIDATE")):
        current = cells[cell]
        prior = r2[arm]
        values = []
        for key in ("final", "stage_1", "stage_2", "stage_3"):
            for metric in ("auroc", "ap"):
                values.append((f"ranking.{key}.{metric}", current["ranking"][key][metric], prior["ranking"][key][metric]))
        for region in ("positive", "interior", "boundary", "negative", "near_background", "far_background"):
            for stat in ("mean", "median", "p99"):
                values.append((f"coverage.{region}.{stat}", current["coverage"][region][stat], prior["coverage"][region][stat]))
        for stat in ("mean", "median", "p99"):
            values.append((f"R_late.{stat}", current["R_late"][stat], prior["R_late"][stat]))
        max_diff = max(abs(float(left) - float(right)) for _, left, right in values)
        checks[cell] = {"max_absolute_difference": max_diff, "tolerance": REPRO_TOLERANCE, "pass": max_diff <= REPRO_TOLERANCE}
    return {"cells": checks, "CROSS_EVAL_REPRODUCTION": all(value["pass"] for value in checks.values())}


def main() -> None:
    identity = checkpoint_identity()
    endpoint = json.loads(R2_ENDPOINT_JSON.read_text())
    split = json.loads(SPLIT_IDENTITY.read_text())
    selected_rows, selected_by_category = load_endpoint_selection()
    endpoint_identity = (
        all(value["identity_ok"] for value in identity.values())
        and len(selected_rows) == 96
        and split["calibration_count"] == 96
        and split["endpoint_eval_count"] == 96
        and split["intersection_count"] == 0
    )
    if not endpoint_identity:
        raise RuntimeError("ENDPOINT_IDENTITY=FAIL")
    cells_spec = {
        "C_EQ": ("CONTROL", H2_EQUAL_STAGE_FUSION_WEIGHTS),
        "C_W": ("CONTROL", H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS),
        "T_EQ": ("CANDIDATE", H2_EQUAL_STAGE_FUSION_WEIGHTS),
        "T_W": ("CANDIDATE", H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS),
    }
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = PrecisionPolicy("fp16")
    model = make_model(device, H2_EQUAL_STAGE_FUSION_WEIGHTS)
    cells = {}
    arrays = {}
    pair_indices = None
    for cell, (endpoint_name, weights) in cells_spec.items():
        checkpoint = Path(identity[endpoint_name]["checkpoint_path"])
        result, cell_arrays, pair_indices = evaluate_cell(
            model, checkpoint, tuple(weights), selected_rows, selected_by_category,
            device, policy, pair_indices,
        )
        cells[cell] = result
        arrays[cell] = cell_arrays
        torch.cuda.empty_cache()
    reproduction = reproduction_check(cells)
    if not reproduction["CROSS_EVAL_REPRODUCTION"]:
        json_dump(REPO / "audit/H2_POST_R2_CROSS_EVAL.json", {
            "protocol_id": PROTOCOL_ID,
            "ENDPOINT_IDENTITY": "PASS",
            "CROSS_EVAL_REPRODUCTION": "FAIL",
            "identity": identity,
            "reproduction": reproduction,
        })
        raise RuntimeError("CROSS_EVAL_REPRODUCTION=FAIL")

    contrast_values = contrasts(cells)
    metrics_rows = []
    for cell, result in cells.items():
        for metric, value in metric_values(result).items():
            metrics_rows.append({"cell": cell, "metric": metric, "value": value})
    write_csv(REPO / "audit/H2_POST_R2_2X2_METRICS.csv", metrics_rows)

    background_rows = []
    for cell, result in cells.items():
        for metric, value in result["background_tail"].items():
            if metric != "pairwise_sampling":
                background_rows.append({"cell": cell, "metric": metric, "value": value})
    write_csv(REPO / "audit/H2_POST_R2_BACKGROUND_TAIL.csv", background_rows)

    pr_rows = []
    for cell, result in cells.items():
        pr = result["pr"]
        for grid, values in pr["precision_at_recall"].items():
            pr_rows.append({"cell": cell, "curve_metric": "precision_at_recall", "grid": grid, "value": values})
        for grid, values in pr["recall_at_precision"].items():
            pr_rows.append({"cell": cell, "curve_metric": "recall_at_precision", "grid": grid, "value": values})
        pr_rows.append({"cell": cell, "curve_metric": "AP", "grid": "all", "value": pr["ap"]})
    write_csv(REPO / "audit/H2_POST_R2_PR_DIAGNOSTIC.csv", pr_rows)

    training_stage_deltas = {
        f"stage{stage}_{metric}_delta": cells["T_EQ"]["ranking"][f"stage_{stage}"][metric] - cells["C_EQ"]["ranking"][f"stage_{stage}"][metric]
        for stage in (1, 2, 3) for metric in ("ap", "auroc")
    }
    result = {
        "protocol_id": PROTOCOL_ID,
        "diagnostic_only": True,
        "training_run": False,
        "optimizer_step_used": False,
        "backward_used": False,
        "scaler_step_used": False,
        "target_data_used": False,
        "medical_or_mvtec_used": False,
        "full_e15_started": False,
        "endpoint_identity": identity,
        "ENDPOINT_IDENTITY": "PASS",
        "CROSS_EVAL_REPRODUCTION": "PASS",
        "reproduction": reproduction,
        "split": {
            "calibration_count": split["calibration_count"],
            "endpoint_eval_count": split["endpoint_eval_count"],
            "intersection_count": split["intersection_count"],
            "endpoint_subset_sha256": sha256_file(ENDPOINT_SUBSET),
        },
        "cells": cells,
        "contrasts": contrast_values,
        "training_stage_deltas": training_stage_deltas,
        "pairwise_sampling": {
            "seed": PAIR_SEED,
            "sample_size": PAIR_SAMPLE_SIZE,
            "same_pairs_for_all_cells": True,
        },
        "prior_r2_decision": {
            "BOUNDED_SCREEN": endpoint.get("scope", "").split(";")[0],
            "authoritative_decision": json.loads((REPO / "results/H2_FUSION_R2_BOUNDED_DECISION.json").read_text())["BOUNDED_SCREEN"],
        },
    }
    json_dump(REPO / "audit/H2_POST_R2_CROSS_EVAL.json", result)
    report = [
        "# H2 Post-R2 Frozen 2x2 Cross-Evaluation",
        "",
        f"`PROTOCOL_ID={PROTOCOL_ID}`",
        "",
        "`DIAGNOSTIC_ONLY=YES`",
        "",
        "`ENDPOINT_IDENTITY=PASS`",
        "",
        "`CROSS_EVAL_REPRODUCTION=PASS`",
        "",
        "No training, optimizer/scaler step, backward pass, target inference, Medical/MVTec inference, tuning, or recalibration was performed.",
        "",
        "## Frozen cells",
        "",
        "| Cell | Checkpoint | Inference fusion | Final AUROC | Final AP | R_late |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = {"C_EQ": "CONTROL / equal", "C_W": "CONTROL / weighted", "T_EQ": "WEIGHTED-TRAINED / equal", "T_W": "WEIGHTED-TRAINED / weighted"}
    for cell in cells:
        result_cell = cells[cell]
        report.append(
            f"| {cell} | {labels[cell]} | `{result_cell['weights']}` | "
            f"{result_cell['ranking']['final']['auroc']:.12f} | {result_cell['ranking']['final']['ap']:.12f} | {result_cell['R_late']['mean']:.12f} |"
        )
    report += [
        "",
        "## Interpretation inputs",
        "",
        "Training-stage deltas are T_EQ minus C_EQ, so equal inference isolates the training-trajectory contrast.",
        "",
        f"- Stage-1 AP delta: `{training_stage_deltas['stage1_ap_delta']:.12f}`",
        f"- Stage-2 AP delta: `{training_stage_deltas['stage2_ap_delta']:.12f}`",
        f"- Stage-3 AP delta: `{training_stage_deltas['stage3_ap_delta']:.12f}`",
        "",
        "The complete compact metrics, background-tail diagnostics, and PR diagnostics are in the required CSV/JSON artifacts.",
    ]
    (REPO / "audit/H2_POST_R2_CROSS_EVAL.md").write_text("\n".join(report) + "\n")
    print(json.dumps({
        "status": "PASS",
        "ENDPOINT_IDENTITY": "PASS",
        "CROSS_EVAL_REPRODUCTION": "PASS",
        "cells": list(cells),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
