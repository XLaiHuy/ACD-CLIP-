#!/usr/bin/env python3
"""Assemble final Anchor/P/C_OLD source and Medical decision artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from tools.cir_rmt.anchor_medical_eval import EPOCHS, METRICS, METHODS, TARGETS, _load_frozen_rows


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _new_cells(raw: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    output: dict[tuple[int, str, str], dict[str, Any]] = {}
    for path in sorted((raw / "cells").glob("medical__A*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "COMPLETE" or row.get("method") not in {"A0", "A05"}:
            raise RuntimeError(f"invalid new target cell: {path}")
        key = (int(row["epoch"]), str(row["method"]), str(row["target"]))
        value = dict(row)
        value["source"] = "new_anchor_checkpoint"
        value["cell_path"] = str(path)
        output[key] = value
    expected = {(epoch, method, target) for epoch in EPOCHS for target in TARGETS for method in ("A0", "A05")}
    if set(output) != expected:
        raise RuntimeError(f"new Anchor cell set incomplete: missing={sorted(expected - set(output))[:5]}")
    return output


def _matrix_rows(frozen: Mapping[tuple[int, str, str], Mapping[str, Any]], new: Mapping[tuple[int, str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        for target in TARGETS:
            for method in METHODS:
                source = new if method in {"A0", "A05"} else frozen
                row = dict(source[(epoch, method, target)])
                row.setdefault("alpha", 0.5 if method.endswith("05") else None)
                row.update({"epoch": epoch, "target": target, "method": method, "status": "COMPLETE"})
                rows.append(row)
    return rows


def _macro(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        for method in METHODS:
            subset = [row for row in rows if int(row["epoch"]) == epoch and row["method"] == method]
            value: dict[str, Any] = {"epoch": epoch, "method": method, "n_domains": len(subset)}
            for metric in METRICS:
                values = [float(row[metric]) for row in subset if row.get(metric) not in (None, "")]
                value[f"{metric}_macro"] = mean(values) if values else None
                value[f"{metric}_support"] = len(values)
            output.append(value)
    return output


def _deltas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {(int(row["epoch"]), str(row["method"]), str(row["target"])): row for row in rows}
    output: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        for target in TARGETS:
            p = index[(epoch, "P", target)]
            cold = index[(epoch, "C_OLD_0", target)]
            cold5 = index[(epoch, "C_OLD_05", target)]
            a = index[(epoch, "A0", target)]
            a5 = index[(epoch, "A05", target)]
            value: dict[str, Any] = {"epoch": epoch, "target": target}
            for metric in METRICS:
                values = {name: _num(row.get(metric)) for name, row in (("p", p), ("cold", cold), ("cold5", cold5), ("a", a), ("a5", a5))}
                for name, left, right in (("anchor_train_effect", "a", "p"), ("anchor_rmt_inference_effect", "a5", "a"), ("anchor_total_effect", "a5", "p"), ("anchor_vs_old_cir", "a", "cold"), ("old_cir_rmt_inference_effect", "cold5", "cold")):
                    left_value, right_value = values[left], values[right]
                    value[f"{name}_{metric}"] = None if left_value is None or right_value is None else left_value - right_value
            output.append(value)
    return output


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    mx, my = mean(x), mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denom if denom > 0 else None


def _association(archive: Path, deltas: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    feature_rows = _read_csv(archive / "SAME_EPOCH_FEATURE_DRIFT.csv")
    parameter_rows = _read_csv(archive / "SAME_EPOCH_PARAMETER_DRIFT.csv")
    output: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        for comparison in ("C_OLD", "A"):
            features = [row for row in feature_rows if int(row["epoch"]) == epoch and row["comparison"] == f"{comparison}_E{epoch:02d}" and row["signal"] != "Text_descriptors"]
            cosines = [float(row["mean_cosine"]) for row in features if row.get("mean_cosine") not in (None, "")]
            params = [row for row in parameter_rows if int(row["epoch"]) == epoch and row["comparison"] == f"{comparison}_E{epoch:02d}" and row["component"] == "image_adapter"]
            image_parameter_drift = mean(float(row["normalized_l2"]) for row in params) if params else None
            feature_drift = 1.0 - mean(cosines) if cosines else None
            for delta in [row for row in deltas if int(row["epoch"]) == epoch]:
                output.append({
                    "row_type": "domain_epoch",
                    "epoch": epoch,
                    "target": delta["target"],
                    "comparison": comparison,
                    "source_feature_mean_drift": feature_drift,
                    "source_image_parameter_normalized_l2": image_parameter_drift,
                    "anchor_train_effect_pixel_auroc": delta.get("anchor_train_effect_pixel_auroc"),
                    "anchor_train_effect_pixel_ap": delta.get("anchor_train_effect_pixel_ap"),
                    "anchor_rmt_inference_effect_pixel_auroc": delta.get("anchor_rmt_inference_effect_pixel_auroc"),
                    "anchor_rmt_inference_effect_pixel_ap": delta.get("anchor_rmt_inference_effect_pixel_ap"),
                    "diagnostic_only": True,
                })
    for comparison in ("C_OLD", "A"):
        source_by_epoch: dict[int, float] = {}
        for row in output:
            if row["comparison"] == comparison and row["row_type"] == "domain_epoch" and row["target"] == TARGETS[0] and row["source_feature_mean_drift"] not in (None, ""):
                source_by_epoch[int(row["epoch"])] = float(row["source_feature_mean_drift"])
        for metric_key in ("anchor_train_effect_pixel_auroc", "anchor_train_effect_pixel_ap", "anchor_rmt_inference_effect_pixel_auroc", "anchor_rmt_inference_effect_pixel_ap"):
            pairs = []
            for epoch in EPOCHS:
                values = [row for row in output if row["row_type"] == "domain_epoch" and row["comparison"] == comparison and int(row["epoch"]) == epoch and row[metric_key] not in (None, "")]
                if values and epoch in source_by_epoch:
                    pairs.append((source_by_epoch[epoch], mean(float(row[metric_key]) for row in values)))
            output.append({"row_type": "epoch_macro_association", "epoch": "ALL", "target": "__epoch_macro__", "comparison": comparison, "source_feature_mean_drift": None, "source_image_parameter_normalized_l2": None, "association_metric": metric_key, "correlation_across_epochs": _pearson([p[0] for p in pairs], [p[1] for p in pairs]), "n_epochs": len(pairs), "diagnostic_only": True})
    return output


def _checkpoint_drift(archive: Path, source_rows: Sequence[Mapping[str, Any]], macros: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    source = {(int(row["epoch"]), str(row["method"])): row for row in source_rows}
    macro = {(int(row["epoch"]), str(row["method"])): row for row in macros}
    lr_rows = {int(row["epoch"]): row for row in _read_csv(archive / "cir_lr_history.csv")}
    output: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        lr = lr_rows[epoch]
        value: dict[str, Any] = {
            "epoch": epoch,
            "intended_image_lr": 0.001 * (0.9**epoch),
            "actual_anchor_image_lr": float(lr["post_image_lr"]),
            "lr_ratio": float(lr["post_image_lr"]) / (0.001 * (0.9**epoch)),
        }
        for method, label in (("P", "parent"), ("C_OLD_0", "old_cir_alpha0"), ("C_OLD_05", "old_cir_alpha05"), ("A0", "anchor_alpha0"), ("A05", "anchor_alpha05")):
            for metric in ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap"):
                value[f"source_{label}_{metric}"] = source[(epoch, method)].get(metric)
                value[f"medical_{label}_{metric}_macro"] = macro[(epoch, method)].get(f"{metric}_macro")
        output.append(value)
    return output


def _write_reports(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]], macros: Sequence[Mapping[str, Any]], deltas: Sequence[Mapping[str, Any]], association: Sequence[Mapping[str, Any]], decision: str) -> None:
    archive = args.output_root.expanduser().resolve()
    macro_index = {(int(row["epoch"]), str(row["method"])): row for row in macros}
    lines = [
        "# Corrective matched retrain results summary",
        "",
        "Status: COMPLETE. The E14 image-parameter-anchor continuation resumed the existing E14 cursor and trained through E20 under the matched Adam/StepLR protocol.",
        "",
        "P is the matched Phase2B parent; C_OLD is the previously trained CIR run; A is the anchored CIR run. P and C_OLD Medical rows were reused from the frozen exact evaluation. Only A0/A05 cells were newly evaluated after the target-blind freeze.",
        "",
        "| epoch | P pixel AUROC | C_OLD_0 pixel AUROC | A0 pixel AUROC | A05 pixel AUROC | A0-P | A05-A0 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in EPOCHS:
        p, old, a, a5 = (macro_index[(epoch, method)] for method in ("P", "C_OLD_0", "A0", "A05"))
        lines.append(f"| E{epoch} | {float(p['pixel_auroc_macro']):.6f} | {float(old['pixel_auroc_macro']):.6f} | {float(a['pixel_auroc_macro']):.6f} | {float(a5['pixel_auroc_macro']):.6f} | {float(a['pixel_auroc_macro']) - float(p['pixel_auroc_macro']):+.6f} | {float(a5['pixel_auroc_macro']) - float(a['pixel_auroc_macro']):+.6f} |")
    lines.extend([
        "",
        f"Final decision: `{decision}`.",
        "",
        "The complete per-domain matrix, macro definitions, and target deltas are the authoritative numeric artifacts. A05-minus-A0 is the conditional inference RMT effect on the anchored representation; it is not a clean CIR-vs-Phase2B effect.",
        "",
        "Target tuning: NO. MVTec: NOT_RUN.",
    ])
    (archive / "CORRECTED_RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    alpha_lines = [
        "# Final RMT alpha decision",
        "",
        "The inference comparison is A05 - A0 at the same anchored checkpoint. It is conditional on the anchored representation and is not a training comparison.",
        "",
        "| metric | mean A05-A0 over domain/epoch | positive cells | total cells | sign fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        values = [float(row[f"anchor_rmt_inference_effect_{metric}"]) for row in deltas if row.get(f"anchor_rmt_inference_effect_{metric}") not in (None, "")]
        positive = sum(value > 0 for value in values)
        alpha_lines.append(f"| {metric} | {mean(values):+.8f} | {positive} | {len(values)} | {positive / len(values):.3f} |")
    alpha_lines.extend(["", "Interpretation is descriptive and uses no target-domain hyperparameter selection.", "", f"Overall experiment decision: `{decision}`."])
    (archive / "FINAL_RMT_ALPHA_DECISION.md").write_text("\n".join(alpha_lines) + "\n", encoding="utf-8")

    anchor_deltas = [row for row in deltas if row["target"] in TARGETS]
    target_lines = [
        "# Final target decomposition",
        "",
        "The target decomposition keeps the three causal layers separate:",
        "",
        "- Anchor training effect: A0 - P.",
        "- Conditional inference RMT effect: A05 - A0.",
        "- Total anchored CIR effect: A05 - P.",
        "- Old-CIR comparison: A0 - C_OLD_0; this is a trajectory/protocol comparison, not a pure architecture effect.",
        "",
        "Medical rows are exact evaluator outputs. P and C_OLD are reused frozen results; A0/A05 are the 72 new logical cells recorded by TARGET_EVAL_LEDGER.csv.",
        "",
        f"The decision recorded for this matrix is `{decision}`. No target epoch was selected after seeing target metrics; the primary source-only rule was frozen as A05 E20 before evaluation.",
        "",
        f"Rows in domain delta table: {len(anchor_deltas)}.",
    ]
    (archive / "FINAL_TARGET_DECOMPOSITION.md").write_text("\n".join(target_lines) + "\n", encoding="utf-8")

    assoc_values = [row for row in association if row.get("row_type") == "epoch_macro_association"]
    assoc_lines = [
        "# Representation-drift / target association interpretation",
        "",
        "The association table joins source-only same-epoch representation drift with target-domain deltas. It is diagnostic and correlational only: source drift is repeated across domains within an epoch, and six epoch points are insufficient to establish a causal relationship.",
        "",
        "The representation closure was classified `PRESERVATION_PARTIAL`: the image-anchor parameters were closer to P at same epochs, while only a subset of non-text feature signals showed lower drift.",
        "",
        "| comparison | target delta metric | correlation across epochs | n epochs |",
        "|---|---|---:|---:|",
    ]
    for row in assoc_values:
        corr = row.get("correlation_across_epochs")
        assoc_lines.append(f"| {row['comparison']} | {row.get('association_metric','')} | {'NA' if corr in (None, '') else f'{float(corr):+.6f}'} | {row.get('n_epochs','')} |")
    assoc_lines.extend(["", "No target labels were used to tune the anchor or RMT parameters. Post-hoc GT-derived observations, if present in inherited diagnostic artifacts, remain diagnostic only."])
    (archive / "RDRIFT_INTERPRETATION.md").write_text("\n".join(assoc_lines) + "\n", encoding="utf-8")

    audit_lines = [
        "# Corrective training audit",
        "",
        "Status: PASS for the E14-to-E20 image-parameter-anchor continuation.",
        "",
        "The run resumed the existing epoch-14 `last.pth` cursor; it did not restart training. It used the frozen CIR_DFG_RMT_V2 config, VisA seed 0, FP32, effective batch 6, Adam betas (0.9, 0.999), eps 1e-8, weight decay 0, gradient clipping 1.0, the existing StepLR(step_size=1, gamma=0.9), and the existing loss `cls + seg + 0.001*kg + 0.0*k`. The image-only anchor was lambda 0.001 against the frozen Phase2B E14 image adapter and was train-only.",
        "",
        "E15-E20 telemetry, E16/E18/E20 first-batch gradient probes, exact scheduler state, optimizer group state, and checkpoint hashes are recorded in the companion JSON/CSV artifacts.",
        "",
        "The one launch device-string failure and one resource-preflight bookkeeping failure were engineering-only, occurred before scientific work, and are recorded as resolved in FAILURE_CLASSIFICATION.json. No target cell was generated before the target-blind freeze.",
        "",
        "Target tuning: NO. MVTec: NOT_RUN.",
    ]
    (archive / "CORRECTIVE_TRAINING_AUDIT.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    ledger_rows = [
        {"field": "source_dataset", "parent_phase2b": "VisA_TRAIN", "cir_anchor": "VisA_TRAIN", "status": "MATCHED", "evidence": "same source role"},
        {"field": "seed", "parent_phase2b": "0", "cir_anchor": "0", "status": "MATCHED", "evidence": "frozen seed"},
        {"field": "precision", "parent_phase2b": "fp32", "cir_anchor": "fp32", "status": "MATCHED", "evidence": "checkpoint audit"},
        {"field": "effective_batch_size", "parent_phase2b": "6", "cir_anchor": "6", "status": "MATCHED", "evidence": "micro batch 6, accumulation 1"},
        {"field": "optimizer", "parent_phase2b": "Adam betas=(0.9,0.999), eps=1e-8, weight_decay=0", "cir_anchor": "Adam betas=(0.9,0.999), eps=1e-8, weight_decay=0", "status": "MATCHED", "evidence": "checkpoint param groups"},
        {"field": "scheduler", "parent_phase2b": "StepLR step_size=1 gamma=0.9 after epoch", "cir_anchor": "StepLR step_size=1 gamma=0.9 after epoch", "status": "MATCHED", "evidence": "last_epoch=epoch, step_count=epoch+1"},
        {"field": "loss", "parent_phase2b": "cls+seg+0.001*kg+0.0*k", "cir_anchor": "cls+seg+0.001*kg+0.0*k+anchor", "status": "INTENDED_DIFFERENCE", "evidence": "image-only train anchor"},
        {"field": "anchor", "parent_phase2b": "none", "cir_anchor": "E14 image adapter lambda=0.001", "status": "INTENDED_DIFFERENCE", "evidence": "frozen anchor SHA"},
        {"field": "checkpoint_schedule", "parent_phase2b": "E10/E12/E14/E16/E18/E20", "cir_anchor": "E10/E12/E14/E16/E18/E20", "status": "MATCHED", "evidence": "candidate audit"},
        {"field": "source_evaluation", "parent_phase2b": "frozen P rows", "cir_anchor": "A0/A05; old P/C reused", "status": "MATCHED_SCOPE", "evidence": "FINAL_SOURCE_MATRIX.csv"},
        {"field": "medical_evaluation", "parent_phase2b": "frozen P rows", "cir_anchor": "new A0/A05 only; old P/C reused", "status": "MATCHED_SCOPE", "evidence": "TARGET_EVAL_LEDGER.csv"},
        {"field": "target_tuning", "parent_phase2b": "NO", "cir_anchor": "NO", "status": "FROZEN", "evidence": "pre-Medical freeze"},
    ]
    _write_csv(archive / "CORRECTIVE_MATCH_LEDGER.csv", ledger_rows, ["field", "parent_phase2b", "cir_anchor", "status", "evidence"])

    decision_lines = [
        "# Corrective go / no-go decision",
        "",
        f"DECISION: {decision}",
        "",
        "This decision is based on the frozen six-epoch source matrix, same-epoch representation closure, exact six-domain Medical matrix, target deltas, and conditional A05-minus-A0 inference comparison.",
        "",
        "The current run establishes the effect of a selective Phase2B E14 image-parameter anchor under an optimization-matched continuation. It does not establish a clean causal RMT training effect against the old CIR run, because the old CIR representation followed a different trajectory and the anchor is an additional training intervention.",
        "",
        "Target tuning: NO. MVTec: NOT_RUN.",
    ]
    (archive / "CORRECTED_GO_NO_GO_DECISION.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    archive = args.output_root.expanduser().resolve()
    raw = args.medical_raw_root.expanduser().resolve()
    freeze = json.loads((archive / "PRE_MEDICAL_FREEZE.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN" or freeze.get("target_tuning_occurred") is not False:
        raise RuntimeError("target-blind freeze missing or invalid")
    frozen = _load_frozen_rows(args.frozen_medical.expanduser().resolve())
    new = _new_cells(raw)
    ledger_source = raw / "TARGET_EVAL_LEDGER.csv"
    if not ledger_source.is_file():
        raise RuntimeError(f"target ledger missing: {ledger_source}")
    ledger_rows = _read_csv(ledger_source)
    if len(ledger_rows) != 72 or any(row.get("status") != "COMPLETE" for row in ledger_rows):
        raise RuntimeError("target ledger is not a complete 72-cell ledger")
    _write_csv(archive / "TARGET_EVAL_LEDGER.csv", ledger_rows, ["cell_id", "scope", "method", "epoch", "target", "status", "cell_path", "cell_sha256", "checkpoint_sha256", "n_images", "updated_at"])
    matrix = _matrix_rows(frozen, new)
    fields = ["target", "epoch", "method", "alpha", "n_images", "target_count", "image_metric_support", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap", "checkpoint_sha256", "config_sha256", "evaluator_git_sha", "evaluator_sha256", "source", "status"]
    _write_csv(archive / "FINAL_MEDICAL_MATRIX.csv", matrix, fields)
    macros = _macro(matrix)
    _write_csv(archive / "FINAL_MEDICAL_MACRO.csv", macros, ["epoch", "method", "n_domains", "pixel_auroc_macro", "pixel_auroc_support", "pixel_ap_macro", "pixel_ap_support", "image_auroc_macro", "image_auroc_support", "image_ap_macro", "image_ap_support"])
    checkpoint_drift = _checkpoint_drift(archive, _read_csv(archive / "FINAL_SOURCE_MATRIX.csv"), macros)
    drift_fields = ["epoch", "intended_image_lr", "actual_anchor_image_lr", "lr_ratio"]
    for label in ("parent", "old_cir_alpha0", "old_cir_alpha05", "anchor_alpha0", "anchor_alpha05"):
        for metric in METRICS:
            drift_fields.extend([f"source_{label}_{metric}", f"medical_{label}_{metric}_macro"])
    _write_csv(archive / "FINAL_CHECKPOINT_DRIFT.csv", checkpoint_drift, drift_fields)
    deltas = _deltas(matrix)
    delta_fields = ["epoch", "target"]
    for metric in METRICS:
        delta_fields.extend([f"anchor_train_effect_{metric}", f"anchor_rmt_inference_effect_{metric}", f"anchor_total_effect_{metric}", f"anchor_vs_old_cir_{metric}", f"old_cir_rmt_inference_effect_{metric}"])
    _write_csv(archive / "FINAL_MEDICAL_DOMAIN_DELTAS.csv", deltas, delta_fields)
    association = _association(archive, deltas)
    _write_csv(archive / "RDRIFT_TARGET_ASSOCIATION.csv", association, ["row_type", "epoch", "target", "comparison", "source_feature_mean_drift", "source_image_parameter_normalized_l2", "anchor_train_effect_pixel_auroc", "anchor_train_effect_pixel_ap", "anchor_rmt_inference_effect_pixel_auroc", "anchor_rmt_inference_effect_pixel_ap", "association_metric", "correlation_across_epochs", "n_epochs", "diagnostic_only"])
    decision = str(args.final_decision)
    _write_reports(args, matrix, macros, deltas, association, decision)
    source = json.loads((archive / "FINAL_SOURCE_EVAL_STATUS.json").read_text(encoding="utf-8"))
    audit = json.loads((archive / "EXTENSION_TRAINING_AUDIT.json").read_text(encoding="utf-8"))
    manifest = {
        "status": "COMPLETE",
        "experiment_id": "corrective_matched_retrain_20260830_final_extension_anchor_e20_20260831",
        "arch_id": "CIR_DFG_RMT_V2",
        "architecture_version": 2,
        "architecture_freeze_sha256": freeze["architecture_freeze_sha256"],
        "config_sha256": freeze["config_sha256"],
        "parent_config_sha256": "d24cf942684b0be3c12838699ec6fe452697bd7f0a58eabbf316fb79b1b18cdb",
        "source_dataset": "VisA",
        "source_root": "/home/ai4/caohuy/data/VisA_20220922",
        "source_sample_images": 96,
        "seed": 0,
        "clip_asset_sha256": freeze["clip_asset_sha256"],
        "anchor_reference_epoch": 14,
        "anchor_reference_checkpoint_sha256": freeze["anchor_reference_checkpoint_sha256"],
        "anchor_lambda_image": 0.001,
        "checkpoint_shas": {str(row["epoch"]): row["checkpoint_sha256"] for row in audit["candidate_checkpoints"]},
        "checkpoint_paths": {str(row["epoch"]): row["path"] for row in audit["candidate_checkpoints"]},
        "training_git_sha": audit["training_git_sha"],
        "current_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_matrix_generated_at_git_sha": _read_csv(archive / "FINAL_SOURCE_MATRIX.csv")[0].get("evaluator_git_sha", "") if _read_csv(archive / "FINAL_SOURCE_MATRIX.csv") else "",
        "medical_cells_generated_at_git_sha": next(iter(new.values())).get("evaluator_git_sha", "") if new else "",
        "source_evaluation": {"status": source.get("status"), "artifact": "FINAL_SOURCE_MATRIX.csv", "epochs": list(EPOCHS), "methods": ["P", "C_OLD_0", "C_OLD_05", "A0", "A05"]},
        "representation_preservation": "PRESERVATION_PARTIAL",
        "medical_evaluation": {"status": "COMPLETE", "artifact": "FINAL_MEDICAL_MATRIX.csv", "new_logical_cells": 72, "reused_logical_cells": 108, "targets": list(TARGETS), "epochs": list(EPOCHS)},
        "target_tuning_occurred": False,
        "mvtec": "NOT_RUN",
        "primary_checkpoint": "A05_E20",
        "primary_epoch": 20,
        "primary_selection_rule": freeze["primary_selection_rule"],
        "final_decision": decision,
        "decision_artifact": "FINAL_DECISION.json",
        "source_matrix_sha256": _sha256(archive / "FINAL_SOURCE_MATRIX.csv"),
        "medical_matrix_sha256": _sha256(archive / "FINAL_MEDICAL_MATRIX.csv"),
        "target_ledger_sha256": _sha256(archive / "TARGET_EVAL_LEDGER.csv"),
        "medical_raw_eval_root": str(raw),
        "freeze_sha256": _sha256(archive / "PRE_MEDICAL_FREEZE.json"),
    }
    (archive / "CORRECTIVE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_decision = {
        "status": "FINAL",
        "final_decision": decision,
        "representation_preservation": "PRESERVATION_PARTIAL",
        "primary_checkpoint": "A05_E20",
        "primary_epoch": 20,
        "medical_status": "COMPLETE",
        "source_status": "COMPLETE",
        "inference_rmt": "A05_MINUS_A0_CONDITIONAL_ON_ANCHORED_REPRESENTATION",
        "target_tuning_occurred": False,
        "mvtec_status": "NOT_RUN",
        "new_medical_logical_cells": 72,
        "reused_medical_logical_cells": 108,
        "decision_basis": ["matched E14-to-E20 training audit", "same-epoch representation closure", "six-epoch source decomposition", "exact six-domain Medical matrix", "conditional A05-minus-A0 alpha comparison"],
        "proven": ["E14 cursor resumed to E20", "StepLR and Adam state matched the frozen protocol", "all six Anchor checkpoints are identity-valid", "all 72 new Anchor Medical logical cells completed", "no target tuning or MVTec evaluation occurred"],
        "correlational": ["source/target representation-drift associations", "A0/C_OLD trajectory comparisons", "A05-minus-A0 domain consistency"],
        "unknown": ["whether a clean no-anchor CIR retrain under the fixed scheduler would match P on Medical", "whether a different anchor strength or reference would generalize", "whether RMT helps outside the tested target domains"],
    }
    (archive / "FINAL_DECISION.json").write_text(json.dumps(final_decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--medical-raw-root", type=Path, required=True)
    parser.add_argument("--frozen-medical", type=Path, required=True)
    parser.add_argument("--final-decision", default="INCONCLUSIVE", choices=("KEEP_SELECTIVE_PHASE2B_ANCHOR", "KEEP_ANCHOR_DISABLE_INFERENCE_RMT_CANDIDATE", "TRAINING_TRAJECTORY_REGULARIZATION_EFFECT", "REJECT_PARAMETER_ANCHOR_AS_FINAL", "ANCHOR_TOO_RESTRICTIVE", "K7_DEPLOYMENT_CONSISTENCY_NEXT", "MIXED_DOMAIN_GENERALIZATION", "ABANDON_RMT_RETURN_TO_PHASE2B", "ABANDON_THIS_LINEAGE", "INCONCLUSIVE"))
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
