#!/usr/bin/env python3
"""Phase5 overnight Branch B1: GT-free stage-arbitration feasibility.

This branch evaluates only the preregistered P0--P3 counterfactuals from the
overnight controller.  It is inference-only and does not learn a selector,
gate, weight, temperature, or new model component.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
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
    build_architecture,
    exact_auc_ap,
    predict_one,
    project_exact_auc_ap,
    write_json,
)
from audit_phase5_hsir_stage_rescue import (  # noqa: E402
    ACTIONABILITY_ROOT,
    CHECKPOINT,
    CONFIG,
    EXPECTED_ANOMALY,
    EXPECTED_CHECKPOINT_SHA,
    EXPECTED_CLASSES,
    EXPECTED_CONFIG_SHA,
    EXPECTED_IMAGES,
    EXPECTED_NORMAL,
    EXPECTED_VISA_META_SHA,
    PHASE5_COMMIT,
    SCIENTIFIC_ANCESTOR,
    OUTPUT_ROOT as STAGE_RESCUE_ROOT,
    canonical_test_records,
    current_branch,
    current_head,
    stage_scores_from_native,
    stable_desc,
    write_json_local,
    architecture_gate,
    distribution,
    class_aggregate,
)
from dataset import get_text_and_image_dataset  # noqa: E402
from utils import configure_canonical_fp32  # noqa: E402


OUTPUT_ROOT = ROOT / "runs/phase5/hsir/STAGE_ARBITRATION"
VISA_META = ROOT / "dataset/hub/VisA.jsonl"
PHASE5_ROOT = ROOT / "runs/phase5/hsir/VISA_TEST"
BRANCH_A_COMMIT = "06d72e37b9a7b2ce0db381e1d3b6edfc59de9c91"
VALID_TERMINALS = {
    "GT_FREE_STAGE_RESCUE_FEASIBLE",
    "STAGE_RESCUE_ORACLE_ONLY",
    "FIXED_STAGE_SUFFICIENT",
    "STAGE_ARBITRATION_NOT_SUPPORTED",
    "STAGE_ARBITRATION_PARTIAL",
}
PARITY_TOL = 1e-10
EPS = 1e-12
PRIMARY_BUDGET = 0.20


def current_branch_commit() -> str:
    return current_head()


def process_class(model, dataset, class_name: str, records: list[dict], img_size: int, device: torch.device) -> dict:
    pixels_per_image = int(img_size * img_size)
    n_images = len(records)
    n_pixels = n_images * pixels_per_image
    n_stages = 3
    final_scores = np.empty(n_pixels, dtype=np.float32)
    labels = np.empty(n_pixels, dtype=np.uint8)
    d_rank = np.empty(n_pixels, dtype=np.float32)
    stage_scores = np.empty((n_stages, n_pixels), dtype=np.float32)
    text_cache: dict[str, torch.Tensor] = {}
    max_predictor_parity = 0.0
    with torch.inference_mode():
        for image_index, record in enumerate(records):
            raw = dataset[record["source_index"]]
            item = predict_one(model, raw, "VisA", class_name, img_size, text_cache, device)
            start = image_index * pixels_per_image
            end = start + pixels_per_image
            final_scores[start:end] = item["score"].reshape(-1).astype(np.float32, copy=False)
            labels[start:end] = item["target"].reshape(-1).astype(np.uint8, copy=False)
            d_rank[start:end] = item["D_rank"].reshape(-1).astype(np.float32, copy=False)
            native_logits = np.asarray(item["native_logits"], dtype=np.float32)
            image_stage_scores = stage_scores_from_native(native_logits, img_size).reshape(n_stages, -1)
            if image_stage_scores.shape != (n_stages, pixels_per_image):
                raise RuntimeError("BRANCH_B1_IMPLEMENTATION_INVALID: stage score shape changed")
            stage_scores[:, start:end] = image_stage_scores
            max_predictor_parity = max(max_predictor_parity, float(item["parity"]["predictor_max_abs_probability_error"]))
            del item, raw, native_logits, image_stage_scores
    if not np.all(np.isfinite(final_scores)) or not np.all(np.isfinite(stage_scores)):
        raise RuntimeError("BRANCH_B1_IMPLEMENTATION_INVALID: non-finite score")
    positive = labels == 1
    baseline_auc, baseline_ap = exact_auc_ap(final_scores, labels)
    evaluator_auc, evaluator_ap = project_exact_auc_ap(final_scores, labels)
    parity = {
        "final_ap_error": abs(float(baseline_ap - evaluator_ap)),
        "final_auroc_error": abs(float(baseline_auc - evaluator_auc)),
        "predictor_exposure_max_abs_probability_error": max_predictor_parity,
    }
    if max(parity["final_ap_error"], parity["final_auroc_error"]) > PARITY_TOL:
        raise RuntimeError("BRANCH_B1_IMPLEMENTATION_INVALID: final evaluator parity failed")

    selected_count = int(math.ceil(PRIMARY_BUDGET * n_pixels))
    selected = np.zeros(n_pixels, dtype=bool)
    selected[stable_desc(d_rank)[:selected_count]] = True
    max_stage_scores = stage_scores.max(axis=0)
    p3_scores = final_scores.copy()
    p3_scores[selected] = max_stage_scores[selected]

    def policy(name: str, values: np.ndarray) -> dict:
        auc, ap = exact_auc_ap(values, labels)
        shifts = values.astype(np.float64) - final_scores.astype(np.float64)
        return {
            "name": name,
            "ap": float(ap),
            "auroc": float(auc),
            "delta_ap": float(ap - baseline_ap),
            "delta_ap_pp": float(100.0 * (ap - baseline_ap)),
            "delta_auroc": float(auc - baseline_auc),
            "positive_score_shift": distribution(shifts[positive], "anomaly pixels; anomaly probability"),
            "normal_score_shift": distribution(shifts[~positive], "Normal pixels; anomaly probability"),
            "selected_score_shift": distribution(shifts[selected], "D_rank-selected pixels; anomaly probability"),
        }

    policies = {
        "P0_original_final_consensus": policy("P0_original_final_consensus", final_scores),
        "P1_stage8": policy("P1_stage8", stage_scores[0]),
        "P1_stage16": policy("P1_stage16", stage_scores[1]),
        "P1_stage24": policy("P1_stage24", stage_scores[2]),
        "P2_global_max_stage": policy("P2_global_max_stage", max_stage_scores),
        "P3_D_rank_selective_max_stage": policy("P3_D_rank_selective_max_stage", p3_scores),
    }
    result = {
        "class_name": class_name,
        "n_images": n_images,
        "n_pixels": n_pixels,
        "positive_pixel_count": int(positive.sum()),
        "normal_pixel_count": int((~positive).sum()),
        "selected_pixel_count": selected_count,
        "selected_positive_count": int((selected & positive).sum()),
        "selected_positive_fraction": float((selected & positive).sum() / selected_count),
        "parity": parity,
        "policies": policies,
        "normal_inflation": {
            "P2_global_max_stage": policies["P2_global_max_stage"]["normal_score_shift"],
            "P3_D_rank_selective_max_stage": policies["P3_D_rank_selective_max_stage"]["normal_score_shift"],
        },
        "positive_rescue": {
            "P2_global_max_stage": policies["P2_global_max_stage"]["positive_score_shift"],
            "P3_D_rank_selective_max_stage": policies["P3_D_rank_selective_max_stage"]["positive_score_shift"],
        },
        "selector_definition": "top ceil(0.20*N) D_rank pixels per class; stable descending order; GT not used",
        "counterfactual_definition": "P3 changes only selected pixels to their strongest existing singleton-stage deployed anomaly score; all other pixels retain final consensus",
    }
    del final_scores, labels, d_rank, stage_scores, max_stage_scores, p3_scores
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def flatten_row(row: dict) -> dict:
    out = {
        "class": row["class_name"],
        "n_images": row["n_images"],
        "n_pixels": row["n_pixels"],
        "selected_pixel_count": row["selected_pixel_count"],
        "selected_positive_count": row["selected_positive_count"],
        "selected_positive_fraction": row["selected_positive_fraction"],
    }
    for name, metric in row["policies"].items():
        prefix = name.lower()
        out[f"{prefix}_ap"] = metric["ap"]
        out[f"{prefix}_auroc"] = metric["auroc"]
        out[f"{prefix}_delta_ap"] = metric["delta_ap"]
        out[f"{prefix}_delta_ap_pp"] = metric["delta_ap_pp"]
        out[f"{prefix}_delta_auroc"] = metric["delta_auroc"]
        out[f"{prefix}_positive_score_shift_mean"] = metric["positive_score_shift"]["mean"]
        out[f"{prefix}_normal_score_shift_mean"] = metric["normal_score_shift"]["mean"]
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def provenance_gate() -> dict:
    branch_a_summary = json.loads((STAGE_RESCUE_ROOT / "SUMMARY.json").read_text())
    branch_a_decision = json.loads((STAGE_RESCUE_ROOT / "DECISION.json").read_text())
    branch_a_input = json.loads((STAGE_RESCUE_ROOT / "INPUT_CHECK.json").read_text())
    branch_a_output = json.loads((STAGE_RESCUE_ROOT / "OUTPUT_CHECK.json").read_text())
    action_protocol = json.loads((ACTIONABILITY_ROOT / "PROTOCOL.json").read_text())
    phase5_summary = json.loads((PHASE5_ROOT / "SUMMARY.json").read_text())
    checks = {
        "scientific_ancestor": subprocess.run(["git", "merge-base", "--is-ancestor", SCIENTIFIC_ANCESTOR, "HEAD"], cwd=ROOT, check=False).returncode == 0,
        "branch": current_branch() == "autopilot/p4-conditional-semantic-factorization",
        "branch_a_commit_ancestor": subprocess.run(["git", "merge-base", "--is-ancestor", "06d72e3", "HEAD"], cwd=ROOT, check=False).returncode == 0,
        "checkpoint_sha": _sha256(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA,
        "config_sha": _sha256(CONFIG) == EXPECTED_CONFIG_SHA,
        "visa_metadata_sha": _sha256(VISA_META) == EXPECTED_VISA_META_SHA,
        "dataset_split": phase5_summary["provenance"].get("dataset") == "VisA" and phase5_summary["provenance"].get("split") == "test",
        "counts": branch_a_summary["inference"].get("forward_count") == EXPECTED_IMAGES and branch_a_summary["inference"].get("class_count") == EXPECTED_CLASSES,
        "phase5_parity": action_protocol["provenance"].get("predictor_parity") == "PASS",
        "branch_a_input_pass": branch_a_input.get("status") == "PASS",
        "branch_a_output_pass": branch_a_output.get("status") == "PASS",
        "branch_a_terminal": branch_a_decision.get("terminal") == "CONSENSUS_DILUTION_SUPPORTED",
        "no_train_paths": phase5_summary["provenance"].get("contains_train_paths") is False,
        "no_stage_cache": branch_a_summary["inference"].get("dense_cache_persisted") is False,
    }
    if not all(checks.values()):
        raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: " + ", ".join(k for k, v in checks.items() if not v))
    return {
        "branch_a_commit": BRANCH_A_COMMIT,
        "scientific_ancestor": SCIENTIFIC_ANCESTOR,
        "checkpoint": {"path": str(CHECKPOINT), "sha256": _sha256(CHECKPOINT)},
        "config": {"path": str(CONFIG), "sha256": _sha256(CONFIG)},
        "dataset": "VisA",
        "split": "test",
        "image_count": EXPECTED_IMAGES,
        "class_count": EXPECTED_CLASSES,
        "normal_image_count": EXPECTED_NORMAL,
        "anomaly_image_count": EXPECTED_ANOMALY,
        "required_upstream_sha256": {
            "stage_rescue_summary": _sha256(STAGE_RESCUE_ROOT / "SUMMARY.json"),
            "stage_rescue_decision": _sha256(STAGE_RESCUE_ROOT / "DECISION.json"),
            "stage_rescue_output_check": _sha256(STAGE_RESCUE_ROOT / "OUTPUT_CHECK.json"),
            "actionability_protocol": _sha256(ACTIONABILITY_ROOT / "PROTOCOL.json"),
            "phase5_summary": _sha256(PHASE5_ROOT / "SUMMARY.json"),
        },
        "checks": checks,
        "pixel_data_source": "one new class-streamed Branch-B1 inference pass; Branch-A dense stage outputs were not persisted",
    }


def make_input_check(provenance: dict, architecture: dict) -> dict:
    return {
        "branch": "B1_GT_FREE_STAGE_ARBITRATION",
        "status": "PASS",
        "scientific_ancestor": SCIENTIFIC_ANCESTOR,
        "checkpoint": provenance["checkpoint"],
        "config": provenance["config"],
        "split": "test",
        "image_count": EXPECTED_IMAGES,
        "class_count": EXPECTED_CLASSES,
        "normal_image_count": EXPECTED_NORMAL,
        "anomaly_image_count": EXPECTED_ANOMALY,
        "required_upstream_artifact_sha256": provenance["required_upstream_sha256"],
        "runtime_architecture": architecture,
        "counterfactuals": [
            "P0 original final consensus",
            "P1 each fixed stage",
            "P2 global max-stage evidence",
            "P3 D_rank-selective max-stage evidence",
        ],
        "inference_authorization": "one class-streamed VisA TEST pass because Branch-A stage outputs were compacted and not persisted",
    }


def summarize(rows: list[dict], provenance: dict, architecture: dict, input_check: dict) -> dict:
    policy_names = tuple(rows[0]["policies"])
    policy_summary = {}
    for name in policy_names:
        policy_summary[name] = {
            "AP": class_aggregate([row["policies"][name]["ap"] for row in rows], f"{name}_AP"),
            "AUROC": class_aggregate([row["policies"][name]["auroc"] for row in rows], f"{name}_AUROC"),
            "AP_delta": class_aggregate([row["policies"][name]["delta_ap"] for row in rows], f"{name}_AP_delta"),
            "AUROC_delta": class_aggregate([row["policies"][name]["delta_auroc"] for row in rows], f"{name}_AUROC_delta"),
            "positive_score_shift_mean": class_aggregate([row["policies"][name]["positive_score_shift"]["mean"] for row in rows], f"{name}_positive_score_shift_mean"),
            "normal_score_shift_mean": class_aggregate([row["policies"][name]["normal_score_shift"]["mean"] for row in rows], f"{name}_normal_score_shift_mean"),
        }
    consistency = {}
    for name in policy_names:
        ap_deltas = [row["policies"][name]["delta_ap"] for row in rows]
        auc_deltas = [row["policies"][name]["delta_auroc"] for row in rows]
        consistency[name] = {
            "AP_improved_classes": int(sum(value > EPS for value in ap_deltas)),
            "AP_neutral_classes": int(sum(abs(value) <= EPS for value in ap_deltas)),
            "AP_harmed_classes": int(sum(value < -EPS for value in ap_deltas)),
            "AUROC_improved_classes": int(sum(value > EPS for value in auc_deltas)),
            "AUROC_neutral_classes": int(sum(abs(value) <= EPS for value in auc_deltas)),
            "AUROC_harmed_classes": int(sum(value < -EPS for value in auc_deltas)),
            "total_classes": len(rows),
        }
    fixed_names = ("P1_stage8", "P1_stage16", "P1_stage24")
    best_fixed = max(fixed_names, key=lambda name: policy_summary[name]["AP_delta"]["mean"])
    p3 = policy_summary["P3_D_rank_selective_max_stage"]
    p2 = policy_summary["P2_global_max_stage"]
    summary = {
        "provenance": provenance,
        "architecture": architecture,
        "input_check": input_check,
        "inference": {
            "forward_count": int(sum(row["n_images"] for row in rows)),
            "class_count": len(rows),
            "class_at_a_time": True,
            "dense_cache_persisted": False,
        },
        "P0_AP_AUROC": {"AP": policy_summary["P0_original_final_consensus"]["AP"], "AUROC": policy_summary["P0_original_final_consensus"]["AUROC"]},
        "P1_fixed_stage_AP_AUROC": {name: {"AP": policy_summary[name]["AP"], "AUROC": policy_summary[name]["AUROC"]} for name in fixed_names},
        "P2_AP_AUROC": {"AP": p2["AP"], "AUROC": p2["AUROC"]},
        "P3_AP_AUROC": {"AP": p3["AP"], "AUROC": p3["AUROC"]},
        "AP_deltas": {name: policy_summary[name]["AP_delta"] for name in policy_names},
        "AUROC_deltas": {name: policy_summary[name]["AUROC_delta"] for name in policy_names},
        "positive_score_shifts": {name: policy_summary[name]["positive_score_shift_mean"] for name in policy_names},
        "normal_score_shifts": {name: policy_summary[name]["normal_score_shift_mean"] for name in policy_names},
        "normal_inflation": {
            "P2_global_max_stage": class_aggregate([row["normal_inflation"]["P2_global_max_stage"]["mean"] for row in rows], "P2_normal_inflation_mean"),
            "P3_D_rank_selective_max_stage": class_aggregate([row["normal_inflation"]["P3_D_rank_selective_max_stage"]["mean"] for row in rows], "P3_normal_inflation_mean"),
        },
        "positive_rescue": {
            "P2_global_max_stage": class_aggregate([row["positive_rescue"]["P2_global_max_stage"]["mean"] for row in rows], "P2_positive_score_shift_mean"),
            "P3_D_rank_selective_max_stage": class_aggregate([row["positive_rescue"]["P3_D_rank_selective_max_stage"]["mean"] for row in rows], "P3_positive_score_shift_mean"),
        },
        "class_consistency": consistency,
        "best_fixed_stage": best_fixed,
        "target_parity": {
            "max_final_AP_evaluator_error": max(row["parity"]["final_ap_error"] for row in rows),
            "max_final_AUROC_evaluator_error": max(row["parity"]["final_auroc_error"] for row in rows),
            "max_predictor_exposure_probability_error": max(row["parity"]["predictor_exposure_max_abs_probability_error"] for row in rows),
            "status": "PASS",
        },
        "decision_evidence": {
            "P3_AP_improved_classes": consistency["P3_D_rank_selective_max_stage"]["AP_improved_classes"],
            "P3_AP_harmed_classes": consistency["P3_D_rank_selective_max_stage"]["AP_harmed_classes"],
            "P3_AUROC_harmed_classes": consistency["P3_D_rank_selective_max_stage"]["AUROC_harmed_classes"],
            "P3_normal_shift_positive_classes": int(sum(row["policies"]["P3_D_rank_selective_max_stage"]["normal_score_shift"]["mean"] > EPS for row in rows)),
            "P2_AP_improved_classes": consistency["P2_global_max_stage"]["AP_improved_classes"],
            "fixed_best_stage": best_fixed,
            "fixed_best_stage_AP_delta_class_macro_mean": policy_summary[best_fixed]["AP_delta"]["mean"],
            "P3_AP_delta_class_macro_mean": p3["AP_delta"]["mean"],
            "branch_a_internal_rescue_delta_AP_class_macro_mean": json.loads((STAGE_RESCUE_ROOT / "SUMMARY.json").read_text())["internal_stage_rescue_delta_AP"]["mean"],
        },
        "per_class": rows,
    }
    return summary


def decision_from_summary(summary: dict) -> dict:
    e = summary["decision_evidence"]
    c = summary["class_consistency"]
    p3 = c["P3_D_rank_selective_max_stage"]
    p2 = c["P2_global_max_stage"]
    fixed = c[summary["best_fixed_stage"]]
    if p3["AP_improved_classes"] >= 8 and p3["AUROC_harmed_classes"] <= 2 and e["P3_normal_shift_positive_classes"] <= 4:
        if fixed["AP_improved_classes"] >= 8 and e["fixed_best_stage_AP_delta_class_macro_mean"] >= e["P3_AP_delta_class_macro_mean"]:
            terminal = "FIXED_STAGE_SUFFICIENT"
        else:
            terminal = "GT_FREE_STAGE_RESCUE_FEASIBLE"
    elif (p2["AP_improved_classes"] >= 6 or p3["AP_improved_classes"] >= 6) and (p3["AP_harmed_classes"] <= 5):
        terminal = "STAGE_ARBITRATION_PARTIAL"
    elif e["branch_a_internal_rescue_delta_AP_class_macro_mean"] > EPS and p3["AP_improved_classes"] < 6:
        terminal = "STAGE_RESCUE_ORACLE_ONLY"
    else:
        terminal = "STAGE_ARBITRATION_NOT_SUPPORTED"
    return {
        "terminal": terminal,
        "decision_rule": "pre-registered P0-P3 counterfactual pattern; class consistency and broad Normal-shift guard; no tuned threshold",
        "evidence": e,
        "class_consistency": c,
        "next_branch": "C_NOT_RUN_MVTec_UNAVAILABLE" if terminal == "GT_FREE_STAGE_RESCUE_FEASIBLE" else "STOP",
        "next_action": "External replication is unavailable because MVTec is not installed; stop autonomous branching." if terminal == "GT_FREE_STAGE_RESCUE_FEASIBLE" else "Stop autonomous branching after B1.",
    }


def output_check(summary: dict, decision: dict) -> dict:
    rows = summary["per_class"]
    required = ["INPUT_CHECK.json", "PROTOCOL.json", "PER_CLASS.csv", "SUMMARY.json", "DECISION.json", "DECISION.md"]
    finite = all(np.isfinite(float(row["policies"][name]["ap"])) and np.isfinite(float(row["policies"][name]["auroc"])) for row in rows for name in row["policies"])
    checks = {
        "expected_artifacts_present": all((OUTPUT_ROOT / name).is_file() for name in required),
        "row_count_valid": len(rows) == EXPECTED_CLASSES,
        "class_set_complete": len({row["class_name"] for row in rows}) == EXPECTED_CLASSES,
        "no_nan_inf_required_metrics": finite,
        "target_parity_pass": summary["target_parity"]["status"] == "PASS",
        "decision_terminal_valid": decision["terminal"] in VALID_TERMINALS,
        "no_train_paths": summary["provenance"]["checks"]["no_train_paths"],
        "selector_gt_free": all("GT not used" in row["selector_definition"] for row in rows),
        "counterfactuals_preregistered": all(name in {"P0_original_final_consensus", "P1_stage8", "P1_stage16", "P1_stage24", "P2_global_max_stage", "P3_D_rank_selective_max_stage"} for row in rows for name in row["policies"]),
    }
    return {"branch": "B1_GT_FREE_STAGE_ARBITRATION", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "decision": decision["terminal"], "forward_count": summary["inference"]["forward_count"]}


def write_csv_lf(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    provenance = provenance_gate()
    config = json.loads(CONFIG.read_text())
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    configure_canonical_fp32()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, architecture = architecture_gate(config, checkpoint, device)
    datasets, records = canonical_test_records(int(config["img_size"]))
    input_check = make_input_check(provenance, architecture)
    write_json_local(args.output_root / "INPUT_CHECK.json", input_check)
    protocol = {
        "branch": "B1_GT_FREE_STAGE_ARBITRATION",
        "purpose": "Test whether simple GT-free internal-stage evidence use preserves Branch-A rescue without broad Normal harm.",
        "inference_only": True,
        "training_steps": 0,
        "provenance": provenance,
        "pixel_data_source": provenance["pixel_data_source"],
        "predictor_semantics": "exact Phase5 predictor, canonical VisA TEST loader, strict FP32, exact Industrial deployment reconstruction",
        "counterfactuals": {
            "P0": "original final consensus",
            "P1": "each fixed singleton stage",
            "P2": "global max-stage deployed anomaly probability for every pixel",
            "P3": "top ceil(0.20*N) D_rank pixels use max-stage evidence; unselected pixels retain final consensus",
        },
        "selection": "P3 uses frozen GT-free D_rank descending selector; diagnostic budget is not a deployable threshold",
        "metrics": "exact class-pooled AP/AUROC; class primary; score shifts reported separately for anomaly and Normal pixels",
        "no_training": True,
        "no_new_selector_learning": True,
    }
    write_json_local(args.output_root / "PROTOCOL.json", protocol)
    rows = []
    for class_name in sorted(records):
        rows.append(process_class(model, datasets[class_name], class_name, records[class_name], int(config["img_size"]), device))
    summary = summarize(rows, provenance, architecture, input_check)
    write_csv_lf(args.output_root / "PER_CLASS.csv", [flatten_row(row) for row in rows])
    write_json_local(args.output_root / "SUMMARY.json", summary)
    decision = decision_from_summary(summary)
    decision.update({"input_integrity": "PASS", "output_integrity_pending": True, "target_parity": summary["target_parity"], "no_training": True})
    write_json_local(args.output_root / "DECISION.json", decision)
    (args.output_root / "DECISION.md").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    check = output_check(summary, decision)
    write_json_local(args.output_root / "OUTPUT_CHECK.json", check)
    if check["status"] != "PASS":
        raise RuntimeError("BRANCH_B1_IMPLEMENTATION_INVALID: OUTPUT_CHECK failed")
    decision["output_integrity"] = "PASS"
    decision["output_check"] = str(args.output_root / "OUTPUT_CHECK.json")
    write_json_local(args.output_root / "DECISION.json", decision)
    (args.output_root / "DECISION.md").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"STATUS": "Branch B1 complete", "DECISION": decision["terminal"], "FORWARD_COUNT": summary["inference"]["forward_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
