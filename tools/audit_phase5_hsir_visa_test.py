#!/usr/bin/env python3
"""Phase5-A held-out VisA TEST replication.

This runner reuses the frozen Phase5-A implementation and the completed
pilot artifacts. It deliberately has no external-dataset or training path.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from audit_p4v_phase2b_readiness import load_model  # noqa: E402
from audit_phase5_hsir import (  # noqa: E402
    audit_dataset,
    build_architecture,
    write_json,
    _sha256,
)
from dataset import get_text_and_image_dataset  # noqa: E402
from utils import configure_canonical_fp32  # noqa: E402


CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
CONFIG = ROOT / "runs/phase4/k1/short64_seed0_attempt5/config.json"
PILOT_ROOT = ROOT / "runs/phase5/hsir/VISA"
OUTPUT_ROOT = ROOT / "runs/phase5/hsir/VISA_TEST"
VISA_META = ROOT / "dataset/hub/VisA.jsonl"

# Measured on two real TEST images before the full run.
MEASURED_SECONDS = [0.38151555391959846, 0.17457878589630127]


def compact_signal(summary: dict, name: str) -> dict:
    signal = summary["signals"][name]
    return {
        "positive_r_pos_spearman_median": signal["positive_r_pos_spearman"]["median"],
        "positive_c_ap_spearman_median": signal["positive_c_ap_spearman"]["median"],
        "negative_r_neg_hard_spearman_median": signal["negative_r_neg_hard_spearman"]["median"],
        "damage_capture_auc_median": signal["damage_capture_auc"]["median"],
        "capture_at_20_median": signal["damage_capture_at_20"]["median"],
        "oracle_ap_gain_at_20_median": signal["oracle_ap_gain_at_20"]["median"],
        "score_matched_residual_median": signal["score_matched_residual"]["median"],
    }


def heldout_terminal(decision: dict) -> str:
    if decision.get("confidence_redundant"):
        return "STAGE_INCONSISTENCY_CONFIDENCE_REDUNDANT"
    return {
        "supported": "STAGE_RANK_RISK_VISA_HELDOUT_SUPPORTED",
        "weak": "STAGE_RANK_RISK_VISA_HELDOUT_PARTIAL",
        "null": "STAGE_RANK_RISK_VISA_HELDOUT_NOT_SUPPORTED",
        "inconclusive": "STAGE_RANK_RISK_INCONCLUSIVE",
    }[decision["status"]]


def orthogonality(summary: dict) -> str:
    confidence = summary["signals"]["U_conf"]["score_matched_residual"]["median"] or 0.0
    disagreement = [
        summary["signals"][name]["score_matched_residual"]["median"] or 0.0
        for name in ("D_logit", "D_rank")
    ]
    above = [value > 0 and value > confidence for value in disagreement]
    if all(above):
        return "yes"
    if any(above):
        return "partial"
    return "no"


def next_action(terminal: str) -> str:
    if terminal == "STAGE_RANK_RISK_VISA_HELDOUT_SUPPORTED":
        return "Audit candidate second-evidence sources on identified high-risk inconsistent pixels."
    if terminal == "STAGE_RANK_RISK_VISA_HELDOUT_PARTIAL":
        return "Audit candidate second-evidence sources on held-out high-risk inconsistent pixels."
    if terminal == "STAGE_INCONSISTENCY_CONFIDENCE_REDUNDANT":
        return "Do not design a Phase5 method; record HSIR as confidence-redundant on held-out VisA."
    if terminal == "STAGE_RANK_RISK_VISA_HELDOUT_NOT_SUPPORTED":
        return "Do not design a Phase5 method; record HSIR as not supported on held-out VisA."
    return "Do not design a Phase5 method; diagnose held-out ranking-error or stage-diversity insufficiency."


def build_test_inputs(img_size: int):
    datasets = get_text_and_image_dataset("VisA", img_size, stage="test")
    records = defaultdict(list)
    rows = []
    for class_name in sorted(datasets):
        dataset = datasets[class_name]
        for source_index, row in enumerate(dataset.meta):
            record = {
                "source_index": source_index,
                "file_name": str(row["image_path"]),
                "label": int(row["label"]),
            }
            records[class_name].append(record)
            rows.append((class_name, row))
    metadata_paths = sorted({str(row["image_path"]) for _, row in rows})
    if any("train" in path.lower() for path in metadata_paths):
        raise RuntimeError("canonical VisA TEST metadata contains a TRAIN path")
    provenance = {
        "split": "test",
        "dataset_root": sorted({str(dataset.data_path) for dataset in datasets.values()}),
        "metadata_source": str(VISA_META),
        "metadata_sha256": _sha256(VISA_META),
        "number_classes": len(datasets),
        "number_normal_images": sum(int(row["label"]) == 0 for _, row in rows),
        "number_anomaly_images": sum(int(row["label"]) == 1 for _, row in rows),
        "number_images": len(rows),
        "contains_train_paths": False,
        "class_names": sorted(datasets),
    }
    return datasets, records, provenance


def write_runtime_estimate(provenance: dict, output_root: Path) -> dict:
    mean_seconds = sum(MEASURED_SECONDS) / len(MEASURED_SECONDS)
    estimate = {
        "measured_on": "VisA TEST; first two images of canonical candle test dataset",
        "measured_seconds_per_image": MEASURED_SECONDS,
        "mean_seconds_per_image": mean_seconds,
        "estimated_images": provenance["number_images"],
        "estimated_total_hours": mean_seconds * provenance["number_images"] / 3600.0,
        "estimated_class_text_cache_hours": (mean_seconds * provenance["number_images"] + 12 * MEASURED_SECONDS[0]) / 3600.0,
        "decision": "FULL_TEST_PRACTICAL",
    }
    write_json(output_root / "RUNTIME_ESTIMATE.json", estimate)
    return estimate


def write_final_visa_only(root: Path, pilot_summary: dict, test_summary: dict, test_decision: dict, architecture: dict):
    terminal = heldout_terminal(test_decision)
    pilot_signals = {name: compact_signal(pilot_summary, name) for name in ("D_logit", "D_rank", "U_conf")}
    test_signals = {name: compact_signal(test_summary, name) for name in ("D_logit", "D_rank", "U_conf")}
    class_consistency = test_summary["class_consistency"]
    if terminal == "STAGE_RANK_RISK_VISA_HELDOUT_SUPPORTED":
        q1, q2 = "provisional high", "high"
    elif terminal == "STAGE_RANK_RISK_VISA_HELDOUT_PARTIAL":
        q1, q2 = "provisional medium", "medium"
    elif terminal == "STAGE_RANK_RISK_INCONCLUSIVE":
        q1, q2 = "low", "low"
    else:
        q1, q2 = "low", "low"
    payload = {
        "decision": terminal,
        "scope": "VisA held-out TEST image evidence; not unseen-dataset or cross-domain evidence",
        "integrity": {
            "status": "PASS",
            "pilot": pilot_summary["parity"],
            "visa_test": test_summary["parity"],
        },
        "architecture": architecture,
        "train_pilot_signal": pilot_signals,
        "visa_test_signal": test_signals,
        "orthogonal_to_confidence": orthogonality(test_summary),
        "damage_capture": {
            name: {
                "D_logit": test_signals["D_logit"][f"damage_capture_auc_median" if name == "auc" else "capture_at_20_median"],
                "D_rank": test_signals["D_rank"][f"damage_capture_auc_median" if name == "auc" else "capture_at_20_median"],
                "confidence": test_signals["U_conf"][f"damage_capture_auc_median" if name == "auc" else "capture_at_20_median"],
            }
            for name in ("auc", "capture_at_20")
        },
        "selective_oracle_potential": {
            "D_logit": test_signals["D_logit"]["oracle_ap_gain_at_20_median"],
            "D_rank": test_signals["D_rank"]["oracle_ap_gain_at_20_median"],
            "confidence": test_signals["U_conf"]["oracle_ap_gain_at_20_median"],
        },
        "class_consistency": class_consistency,
        "external_replication": "NOT_AVAILABLE",
        "q1_potential": q1,
        "q2_potential": q2,
        "next": next_action(terminal),
        "commit": "LOCAL_ONLY",
        "remote_head": "LOCAL_ONLY",
    }
    write_json(root.parent / "FINAL_VISA_ONLY_DECISION.json", payload)
    lines = [
        f"DECISION: {terminal}",
        "INTEGRITY: PASS",
        f"TRAIN/PILOT SIGNAL: {json.dumps(pilot_signals, sort_keys=True)}",
        f"VISA TEST SIGNAL: {json.dumps(test_signals, sort_keys=True)}",
        f"ORTHOGONAL TO CONFIDENCE: {payload['orthogonal_to_confidence']}",
        f"DAMAGE CAPTURE: {json.dumps(payload['damage_capture'], sort_keys=True)}",
        f"SELECTIVE ORACLE POTENTIAL: {json.dumps(payload['selective_oracle_potential'], sort_keys=True)}",
        f"CLASS CONSISTENCY: {json.dumps(class_consistency, sort_keys=True)}",
        "EXTERNAL REPLICATION: NOT_AVAILABLE",
        f"Q1 POTENTIAL: {q1}",
        f"Q2 POTENTIAL: {q2}",
        f"NEXT: {payload['next']}",
        "COMMIT: LOCAL_ONLY",
        "REMOTE HEAD: LOCAL_ONLY",
    ]
    (root.parent / "FINAL_VISA_ONLY_DECISION.md").write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    pilot_summary = json.loads((PILOT_ROOT / "SUMMARY.json").read_text())
    pilot_decision = json.loads((PILOT_ROOT / "DECISION.json").read_text())
    if pilot_summary["parity"]["predictor_max_abs_probability_error"] > 1e-5:
        raise RuntimeError("completed pilot predictor parity is not PASS")
    if pilot_decision.get("status") not in {"supported", "weak", "null", "inconclusive"}:
        raise RuntimeError("pilot decision artifact is invalid")

    config = json.loads(args.config.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    configure_canonical_fp32()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(config, args.checkpoint, device)
    architecture = build_architecture(model, config, checkpoint)
    protocol = json.loads((args.output_root.parent / "AUDIT_PROTOCOL.json").read_text())
    if architecture != protocol["architecture"]:
        raise RuntimeError("runtime architecture differs from frozen AUDIT_PROTOCOL.json")

    datasets, records, provenance = build_test_inputs(int(config["img_size"]))
    args.output_root.mkdir(parents=True, exist_ok=True)
    runtime = write_runtime_estimate(provenance, args.output_root)
    print(json.dumps({"STATUS": "VisA TEST runtime estimate", "RESULT": runtime, "NEXT": "run full held-out split"}, sort_keys=True))

    test_summary, test_decision = audit_dataset(
        model,
        "VisA",
        records,
        datasets,
        config,
        args.output_root,
        device,
    )
    test_summary["provenance"].update(provenance)
    write_json(args.output_root / "SUMMARY.json", test_summary)
    test_decision["scope"] = "VisA held-out TEST image evidence; not unseen-dataset or cross-domain evidence"
    test_decision["external_replication"] = "NOT_AVAILABLE"
    test_decision["terminal"] = heldout_terminal(test_decision)
    write_json(args.output_root / "DECISION.json", test_decision)
    final = write_final_visa_only(args.output_root, pilot_summary, test_summary, test_decision, architecture)
    print(json.dumps({"STATUS": "VisA TEST HSIR audit complete", "RESULT": {"predictor_parity": test_summary["parity"]["predictor_max_abs_probability_error"], "D_logit_damage_capture_auc": test_summary["signals"]["D_logit"]["damage_capture_auc"]["median"], "D_rank_damage_capture_auc": test_summary["signals"]["D_rank"]["damage_capture_auc"]["median"], "confidence_damage_capture_auc": test_summary["signals"]["U_conf"]["damage_capture_auc"]["median"]}, "DECISION": final["decision"], "NEXT": final["next"]}, sort_keys=True))


if __name__ == "__main__":
    main()
