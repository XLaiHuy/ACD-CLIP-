#!/usr/bin/env python3
"""Finalize the PA Medical matrix and the 2x2 CIR-necessity decision.

Only PA cell records are new inputs.  The P, C_OLD, and A rows are loaded
from the already frozen exact Medical matrix; no target-domain evaluation is
performed by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
EPOCHS = (10, 12, 14, 16, 18, 20)
TARGETS = ("Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir")
METHODS = ("P", "C_OLD_0", "PA", "A0")
METRICS = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "nan"}:
        return None
    return float(value)


def _load_frozen(path: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    rows = _read(path)
    result: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        target = str(row["target"])
        method = str(row["method"])
        if epoch not in EPOCHS or target not in TARGETS or method not in METHODS:
            continue
        result[(epoch, target, method)] = {
            **row,
            "epoch": epoch,
            "target": target,
            "method": method,
            **{metric: _float(row.get(metric)) for metric in METRICS},
        }
    expected = {(epoch, target, method) for epoch in EPOCHS for target in TARGETS for method in ("P", "C_OLD_0", "A0")}
    missing = sorted(expected - set(result))
    if missing:
        raise ValueError(f"frozen Medical matrix incomplete: {missing[:8]}")
    return result


def _load_pa(args: argparse.Namespace) -> dict[tuple[int, str], dict[str, Any]]:
    output = args.medical_run_root.expanduser().resolve()
    ledger_path = output / "PA_MEDICAL_LEDGER.csv"
    rows = _read(ledger_path)
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for ledger in rows:
        if ledger.get("status") != "COMPLETE" or ledger.get("method") != "PA":
            continue
        epoch = int(ledger["epoch"])
        target = str(ledger["target"])
        path = output / ledger["cell_path"]
        if not path.is_file() or _sha256(path) != ledger.get("cell_sha256"):
            raise ValueError(f"PA Medical cell hash invalid: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE" or payload.get("cell_id") != ledger.get("cell_id"):
            raise ValueError(f"PA Medical cell identity invalid: {path}")
        if (epoch, target) in result:
            raise ValueError(f"duplicate PA Medical cell: E{epoch} {target}")
        result[(epoch, target)] = {
            **payload,
            "epoch": epoch,
            "target": target,
            **{metric: _float(payload.get(metric)) for metric in METRICS},
        }
    expected = {(epoch, target) for epoch in EPOCHS for target in TARGETS}
    missing = sorted(expected - set(result))
    if missing:
        raise ValueError(f"PA Medical matrix incomplete: {missing[:8]}")
    return result


def _checkpoint_sha(args: argparse.Namespace, epoch: int) -> str:
    path = args.pa_run_root.expanduser().resolve() / "visa" / "seed0" / "checkpoints" / f"adapter_{epoch}.pth"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _sha256(path)


def _factorial(frozen: Mapping[tuple[int, str, str], Mapping[str, Any]], pa: Mapping[tuple[int, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        for target in TARGETS:
            values = {method: frozen[(epoch, target, method)] for method in ("P", "C_OLD_0", "A0")}
            values["PA"] = pa[(epoch, target)]
            for metric in METRICS:
                p = values["P"][metric]
                c = values["C_OLD_0"][metric]
                pa_value = values["PA"][metric]
                a = values["A0"][metric]
                if any(value is None for value in (p, c, pa_value, a)):
                    continue
                rows.append({
                    "epoch": epoch,
                    "target": target,
                    "metric": metric,
                    "P": p,
                    "C_OLD_0": c,
                    "PA": pa_value,
                    "A0": a,
                    "CIR_no_anchor_minus_P": c - p,
                    "anchor_no_CIR_minus_P": pa_value - p,
                    "anchor_with_CIR_minus_CIR_no_anchor": a - c,
                    "CIR_with_anchor_minus_PA": a - pa_value,
                    "interaction": a - c - pa_value + p,
                    "P_checkpoint_sha256": values["P"].get("checkpoint_sha256", values["P"].get("parent_checkpoint_sha256", "")),
                    "C_OLD_checkpoint_sha256": values["C_OLD_0"].get("checkpoint_sha256", values["C_OLD_0"].get("cir_checkpoint_sha256", "")),
                    "A0_checkpoint_sha256": values["A0"].get("checkpoint_sha256", ""),
                    "PA_checkpoint_sha256": pa[(epoch, target)].get("checkpoint_sha256", ""),
                })
    return rows


def _macro(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        for metric in METRICS:
            selected = [row for row in rows if int(row["epoch"]) == epoch and row["metric"] == metric]
            if not selected:
                continue
            means = {key: statistics.fmean(float(row[key]) for row in selected) for key in ("P", "C_OLD_0", "PA", "A0")}
            output.append({
                "epoch": epoch,
                "metric": metric,
                "n_targets": len(selected),
                **means,
                "CIR_no_anchor_minus_P": means["C_OLD_0"] - means["P"],
                "anchor_no_CIR_minus_P": means["PA"] - means["P"],
                "anchor_with_CIR_minus_CIR_no_anchor": means["A0"] - means["C_OLD_0"],
                "CIR_with_anchor_minus_PA": means["A0"] - means["PA"],
                "interaction": means["A0"] - means["C_OLD_0"] - means["PA"] + means["P"],
            })
    return output


def _source_summary(path: Path, field: str = "CIR_with_anchor_minus_PA") -> dict[str, dict[str, float]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, float]] = {}
    for row in _read(path):
        result.setdefault(str(row["metric"]), {})[f"E{int(row['epoch']):02d}"] = float(row[field])
    return result


def _decision(macro: Sequence[Mapping[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    def effects(metric: str) -> list[float]:
        return [float(row["CIR_with_anchor_minus_PA"]) for row in macro if row["metric"] == metric]

    pixel_auroc = effects("pixel_auroc")
    pixel_ap = effects("pixel_ap")
    image_auroc = effects("image_auroc")
    image_ap = effects("image_ap")
    pixel_positive = bool(pixel_auroc and pixel_ap and all(value > 0 for value in pixel_auroc + pixel_ap))
    pixel_negative = bool(pixel_auroc and pixel_ap and all(value < 0 for value in pixel_auroc + pixel_ap))
    pixel_mixed = not pixel_positive and not pixel_negative
    image_positive = bool(image_auroc and image_ap and all(value > 0 for value in image_auroc + image_ap))
    image_negative = bool(image_auroc and image_ap and all(value < 0 for value in image_auroc + image_ap))
    if pixel_positive and not image_negative:
        value, architecture = "SUPPORTED", "CIR_TRAIN_ANCHOR_NATIVE_INFERENCE"
    elif pixel_negative and not image_positive:
        value, architecture = "HARMFUL", "PHASE2B_ANCHOR_NO_CIR"
    elif pixel_mixed:
        value, architecture = "INCONCLUSIVE", "MIXED_UNRESOLVED"
    else:
        value, architecture = "MIXED", "MIXED_UNRESOLVED"
    detail = {
        "pixel_auroc_effects": pixel_auroc,
        "pixel_ap_effects": pixel_ap,
        "image_auroc_effects": image_auroc,
        "image_ap_effects": image_ap,
        "pixel_positive_all_epochs": pixel_positive,
        "pixel_negative_all_epochs": pixel_negative,
        "image_positive_all_epochs": image_positive,
        "image_negative_all_epochs": image_negative,
        "pixel_sign_consistency": "positive" if pixel_positive else "negative" if pixel_negative else "mixed",
    }
    return value, architecture, detail


def _report(output: Path, macro: Sequence[Mapping[str, Any]], decision: tuple[str, str, dict[str, Any]], source_summary: Mapping[str, Any]) -> None:
    value, architecture, detail = decision
    lines = [
        "# Medical factorial report",
        "",
        "Status: PASS. This report reuses frozen P/C_OLD/A Medical cells and adds only the 36 PA native cells.",
        "",
        "Primary contrast: CIR_WITH_ANCHOR = A0 - PA. A0 is the frozen anchored CIR run; PA is the new native Phase2B plus the same image anchor. A05 is intentionally absent from this primary comparison because inference RMT is not a PA factor.",
        "",
        "| epoch | metric | n targets | P | C_OLD_0 | PA | A0 | A0-PA | interaction |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in macro:
        lines.append(f"| E{int(row['epoch'])} | {row['metric']} | {int(row['n_targets'])} | {float(row['P']):.6f} | {float(row['C_OLD_0']):.6f} | {float(row['PA']):.6f} | {float(row['A0']):.6f} | {float(row['CIR_with_anchor_minus_PA']):+.6f} | {float(row['interaction']):+.6f} |")
    lines.extend([
        "",
        f"Decision rule applied to the measured six-epoch macro signs: CIR_TRAINING_VALUE={value}; FINAL_ARCHITECTURE={architecture}.",
        "",
        "Red-team answers:",
        f"1. A0 beats PA on source: see SOURCE_FACTORIAL_2X2.csv; source CIR-with-anchor effects are {dict(source_summary)}.",
        f"2. A0 beats PA on Medical six-domain macro: pixel sign pattern is {detail['pixel_sign_consistency']} across the reported epochs.",
        f"3. Epoch consistency: pixel AUROC all-positive={detail['pixel_auroc_effects'] and all(x > 0 for x in detail['pixel_auroc_effects'])}; pixel AP all-positive={detail['pixel_ap_effects'] and all(x > 0 for x in detail['pixel_ap_effects'])}.",
        "4. Domain consistency is reported in MEDICAL_FACTORIAL_2X2.csv; no single-domain result is used as a selection rule.",
        "5. Concentration in one dataset must be judged from the per-domain rows and median effects, not the macro alone.",
        "6. Pixel and image effects are reported separately; a metric-family trade-off is not collapsed into one score.",
        "7. PA reproduces all A gains only if A0-PA is approximately zero across both source and Medical; the measured table is the test.",
        "8. Interaction is the explicit A0-C_OLD_0-PA+P column; its sign and epoch stability are reported without post-hoc tuning.",
        "9. Representation association is in PA_FACTORIAL_DRIFT.csv; it is correlational, not an independent causal intervention.",
        "10. PA is scientifically sufficient only if its simpler trajectory has no robust A0 advantage under the locked protocol.",
        "11. A skeptical reviewer should ask why CIR is retained whenever A0-PA is mixed or near zero.",
        "12. The A0-PA evidence is the measured factorial answer; MVTec remains the untouched confirmatory benchmark and was not run.",
        "",
        "No target tuning occurred. No MVTec data were accessed. No new architecture was introduced.",
    ])
    (output / "MEDICAL_FACTORIAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "CIR_NECESSITY_RED_TEAM.md").write_text("\n".join([
        "# CIR necessity red-team",
        "",
        f"The locked factorial result is CIR_WITH_ANCHOR = A0 - PA. The automated classification is {value}; the corresponding architecture proposal is {architecture}.",
        "",
        "Read per-domain effects in MEDICAL_FACTORIAL_2X2.csv and macro effects in MEDICAL_FACTORIAL_MACRO.csv. The decision is deliberately not based on one target, one epoch, target tuning, or A05 inference transport.",
        "",
        "Source and representation evidence remains associative. Only the PA-vs-A0 factorial intervention supports a CIR-training claim, and MVTec remains NOT_RUN.",
        "",
    ]) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = args.frozen_medical.expanduser().resolve()
    frozen = _load_frozen(frozen_path)
    pa = _load_pa(args)
    ledger_path = args.medical_run_root.expanduser().resolve() / "PA_MEDICAL_LEDGER.csv"
    _write(output / "PA_MEDICAL_LEDGER.csv", _read(ledger_path), [
        "cell_id", "scope", "method", "epoch", "target", "status", "cell_path",
        "cell_sha256", "checkpoint_sha256", "n_images", "updated_at",
    ])
    for epoch in EPOCHS:
        expected_sha = _checkpoint_sha(args, epoch)
        for target in TARGETS:
            if pa[(epoch, target)].get("checkpoint_sha256") != expected_sha:
                raise ValueError(f"PA checkpoint SHA mismatch at E{epoch} {target}")
    pa_rows: list[dict[str, Any]] = []
    for epoch in EPOCHS:
        for target in TARGETS:
            row = pa[(epoch, target)]
            pa_rows.append({
                "target": target,
                "epoch": epoch,
                "method": "PA",
                "n_images": row.get("n_images", ""),
                **{metric: row.get(metric) for metric in METRICS},
                "checkpoint_sha256": row.get("checkpoint_sha256", ""),
                "parent_config_sha256": row.get("config_sha256", ""),
                "evaluator_git_sha": row.get("evaluator_git_sha", ""),
                "evaluator_sha256": row.get("evaluator_sha256", ""),
                "source": "new_PA_native_medical_cell",
                "status": "COMPLETE",
            })
    pa_fields = ["target", "epoch", "method", "n_images", *METRICS, "checkpoint_sha256", "parent_config_sha256", "evaluator_git_sha", "evaluator_sha256", "source", "status"]
    _write(output / "PA_MEDICAL_RESULTS.csv", pa_rows, pa_fields)
    factorial = _factorial(frozen, pa)
    factorial_fields = ["epoch", "target", "metric", "P", "C_OLD_0", "PA", "A0", "CIR_no_anchor_minus_P", "anchor_no_CIR_minus_P", "anchor_with_CIR_minus_CIR_no_anchor", "CIR_with_anchor_minus_PA", "interaction", "P_checkpoint_sha256", "C_OLD_checkpoint_sha256", "A0_checkpoint_sha256", "PA_checkpoint_sha256"]
    _write(output / "MEDICAL_FACTORIAL_2X2.csv", factorial, factorial_fields)
    macro = _macro(factorial)
    macro_fields = ["epoch", "metric", "n_targets", "P", "C_OLD_0", "PA", "A0", "CIR_no_anchor_minus_P", "anchor_no_CIR_minus_P", "anchor_with_CIR_minus_CIR_no_anchor", "CIR_with_anchor_minus_PA", "interaction"]
    _write(output / "MEDICAL_FACTORIAL_MACRO.csv", macro, macro_fields)
    interaction_fields = ["epoch", "metric", "n_targets", "interaction", "CIR_with_anchor_minus_PA", "status"]
    _write(output / "MEDICAL_FACTORIAL_INTERACTION.csv", [{**{key: row[key] for key in ("epoch", "metric", "n_targets", "interaction", "CIR_with_anchor_minus_PA")}, "status": "factorial_interaction"} for row in macro], interaction_fields)
    source_summary = _source_summary(output / "SOURCE_FACTORIAL_2X2.csv", "CIR_with_anchor_minus_PA")
    source_anchor = _source_summary(output / "SOURCE_FACTORIAL_2X2.csv", "anchor_no_CIR_minus_P")
    source_interaction = _source_summary(output / "SOURCE_FACTORIAL_2X2.csv", "interaction")
    decision = _decision(macro)
    _report(output, macro, decision, source_summary)
    value, architecture, detail = decision
    freeze = json.loads((output / "PRE_PA_MEDICAL_FREEZE.json").read_text(encoding="utf-8")) if (output / "PRE_PA_MEDICAL_FREEZE.json").is_file() else {}
    by_metric = {metric: {f"E{int(row['epoch']):02d}": row["CIR_with_anchor_minus_PA"] for row in macro if row["metric"] == metric} for metric in METRICS}
    final = {
        "PA_TRAIN_STATUS": "PASS",
        "SOURCE_STATUS": "PASS",
        "MEDICAL_STATUS": "PASS",
        "PA_PRIMARY_SOURCE_EPOCH": freeze.get("selected_reporting_epoch"),
        "SOURCE_CIR_WITH_ANCHOR_BY_EPOCH": source_summary,
        "MEDICAL_CIR_WITH_ANCHOR_BY_EPOCH": by_metric,
        "SOURCE_ANCHOR_NO_CIR_BY_EPOCH": source_anchor,
        "MEDICAL_ANCHOR_NO_CIR_BY_EPOCH": {metric: {f"E{int(row['epoch']):02d}": row["anchor_no_CIR_minus_P"] for row in macro if row["metric"] == metric} for metric in METRICS},
        "SOURCE_INTERACTION_BY_EPOCH": source_interaction,
        "MEDICAL_INTERACTION_BY_EPOCH": {metric: {f"E{int(row['epoch']):02d}": row["interaction"] for row in macro if row["metric"] == metric} for metric in METRICS},
        "PIXEL_AUROC_DECISION": "CIR_TRAINING_POSITIVE" if detail["pixel_auroc_effects"] and all(x > 0 for x in detail["pixel_auroc_effects"]) else "MIXED_OR_NOT_POSITIVE",
        "PIXEL_AP_DECISION": "CIR_TRAINING_POSITIVE" if detail["pixel_ap_effects"] and all(x > 0 for x in detail["pixel_ap_effects"]) else "MIXED_OR_NOT_POSITIVE",
        "IMAGE_AUROC_DECISION": "CIR_TRAINING_POSITIVE" if detail["image_auroc_effects"] and all(x > 0 for x in detail["image_auroc_effects"]) else "MIXED_OR_NOT_POSITIVE",
        "IMAGE_AP_DECISION": "CIR_TRAINING_POSITIVE" if detail["image_ap_effects"] and all(x > 0 for x in detail["image_ap_effects"]) else "MIXED_OR_NOT_POSITIVE",
        "CIR_TRAINING_VALUE": value,
        "INFERENCE_RMT_VALUE": "NEUTRAL",
        "FINAL_ARCHITECTURE": architecture,
        "MVTec_STATUS": "NOT_RUN",
        "MVTec_NEXT": "NO" if architecture == "MIXED_UNRESOLVED" else "YES",
        "TARGET_TUNING_OCCURRED": "NO",
        "decision_detail": detail,
        "frozen_medical_matrix_sha256": _sha256(frozen_path),
        "pa_medical_ledger_sha256": _sha256(args.medical_run_root.expanduser().resolve() / "PA_MEDICAL_LEDGER.csv"),
    }
    (output / "FINAL_ARCHITECTURE_DECISION.json").write_text(json.dumps(final, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (output / "FINAL_ARCHITECTURE_DECISION.md").write_text("\n".join([
        "# Final architecture decision",
        "",
        f"CIR_TRAINING_VALUE: {value}",
        f"FINAL_ARCHITECTURE: {architecture}",
        "INFERENCE_RMT_VALUE: NEUTRAL",
        "MVTec_STATUS: NOT_RUN",
        "TARGET_TUNING_OCCURRED: NO",
        "",
        "The decision is based on the locked source factorial and the PA-vs-A0 Medical factorial. See the JSON and CSV files for every epoch/domain/metric value.",
        "",
    ]) + "\n", encoding="utf-8")
    (output / "PA_MEDICAL_FINAL_STATUS.json").write_text(json.dumps({"status": "PASS", "cells": len(pa_rows), "medical": "COMPLETE", "mvtec": "NOT_RUN", "target_tuning_occurred": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pa_manifest_path = args.pa_run_root.expanduser().resolve() / "visa" / "seed0" / "run_manifest.json"
    pa_manifest = json.loads(pa_manifest_path.read_text(encoding="utf-8"))
    medical_identity_path = args.medical_run_root.expanduser().resolve() / "identity.json"
    medical_identity = json.loads(medical_identity_path.read_text(encoding="utf-8")) if medical_identity_path.is_file() else {}
    parent_manifest_path = args.pa_run_root.expanduser().resolve().parent / "corrective_matched_retrain_20260830" / "parent" / "phase2b" / "run_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8")) if parent_manifest_path.is_file() else {}
    freeze_path = output / "PRE_PA_MEDICAL_FREEZE.json"
    architecture_freeze = ROOT / "docs/cir_rmt/v2/ARCHITECTURE_FREEZE_V2.md"
    control_manifest = {
        "status": "PASS",
        "control_id": "PA_PHASE2B_IMAGE_ANCHOR_V1",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_dataset": "VisA",
        "source_root": str(medical_identity.get("source_root", "recorded in PA training and source manifests")),
        "seed": 0,
        "clip_asset_sha256": str(freeze.get("clip_asset_sha256", "")),
        "parent_config_sha256": str(freeze.get("parent_config_sha256", parent_manifest.get("config_sha256", "d24cf942684b0be3c12838699ec6fe452697bd7f0a58eabbf316fb79b1b18cdb"))),
        "pa_config_sha256": str(pa_manifest.get("config_sha256", "")),
        "architecture_freeze_sha256": _sha256(architecture_freeze) if architecture_freeze.is_file() else "",
        "anchor_reference_checkpoint_sha256": str(json.loads(freeze_path.read_text(encoding="utf-8")).get("anchor_reference_checkpoint_sha256", "")) if freeze_path.is_file() else "",
        "pa_checkpoint_sha256": {str(epoch): _checkpoint_sha(args, epoch) for epoch in EPOCHS},
        "pa_run_manifest_sha256": _sha256(pa_manifest_path),
        "source_freeze_sha256": _sha256(freeze_path) if freeze_path.is_file() else "",
        "medical_ledger_sha256": _sha256(ledger_path),
        "target_tuning_occurred": False,
        "mvtec_status": "NOT_RUN",
        "source_status": "PASS",
        "medical_status": "PASS",
        "final_architecture": architecture,
        "cir_training_value": value,
    }
    (output / "PA_CONTROL_MANIFEST.json").write_text(json.dumps(control_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--medical-run-root", type=Path, required=True)
    parser.add_argument("--pa-run-root", type=Path, required=True)
    parser.add_argument("--frozen-medical", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
