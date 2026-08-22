"""Evaluate preregistered SABRA-CAR R1 OOF actions with canonical deployment."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.sabra_car.r0_direction import (
    MARGIN_SCALE,
    evaluate_correction,
    exact_metrics,
    load_masks,
    load_shard,
    metadata_and_root,
)
from tools.sabra_car.r1_common import (
    EXPECTED_CLASSES,
    load_r1_shards,
    stable_argmax_predictions,
    threshold_actions,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[2]
ALPHA = 0.25


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def native_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["condition"] == "native"
        ]
    result = {row["class"]: row for row in rows}
    if tuple(result) != EXPECTED_CLASSES:
        raise RuntimeError("R0 native row inventory/order failed")
    return result


def _selected_risk_row(selection: dict[str, Any], threshold: float) -> dict[str, Any]:
    matches = [
        row for row in selection["threshold_rows"]
        if row["threshold"] != "unfiltered" and float(row["threshold"]) == threshold
    ]
    if len(matches) != 1 or matches[0].get("risk_gate_pass") is not True:
        raise RuntimeError("selected threshold does not have exactly one passing risk row")
    return matches[0]


def stop_without_evaluation(args: argparse.Namespace, selection: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "status": "COMPLETE",
        "stage": "R1",
        "decision": "STOP",
        "next_stage": "FINAL_DECISION",
        "reason": "NO_RISK_QUALIFIED_THRESHOLD",
        "selected_threshold": None,
        "threshold_rows": selection["threshold_rows"],
        "per_class": [],
        "macro": None,
        "gates": {
            "correctness": {"threshold": "all PASS", "observed": "PASS", "pass": True},
            "risk_threshold_exists": {"threshold": "yes", "observed": "no", "pass": False},
        },
        "medical_access": False,
        "mvtec_access": False,
        "training_steps": 0,
    }
    write_json(args.output / "summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    selection = json.loads((args.fit_root / "selection.json").read_text())
    selected_value = selection.get("selected_threshold")
    if selected_value is None:
        return stop_without_evaluation(args, selection)
    threshold = float(selected_value)
    risk = _selected_risk_row(selection, threshold)
    shards, common_provenance = load_r1_shards(
        args.source_root, args.trust_root, args.utility_root, verify_hashes=not args.skip_hashes
    )
    metadata, data_root = metadata_and_root(args.data_root)
    native = native_rows(args.r0_per_class)
    device = torch.device("cuda")
    per_class: list[dict[str, Any]] = []
    for class_name in EXPECTED_CLASSES:
        shard = shards[class_name]
        with np.load(args.fit_root / "folds" / f"{class_name}.npz", allow_pickle=False) as fold:
            if not np.array_equal(fold["image_path"].astype(str), shard.image_path.astype(str)):
                raise RuntimeError(f"R1 fold path mismatch: {class_name}")
            probability = np.asarray(fold["probability"], dtype=np.float32)
            classes = np.asarray(fold["classes"], dtype=np.int8)
            prediction, confidence = stable_argmax_predictions(
                probability.reshape(-1, 3), classes
            )
        action = threshold_actions(prediction, confidence, threshold).reshape(
            shard.oracle_action.shape
        )
        save_npz(
            args.output / "actions" / f"{class_name}.npz",
            action=action,
            confidence=confidence.reshape(shard.oracle_action.shape).astype(np.float32),
            image_path=shard.image_path,
        )
        cache = load_shard(args.source_root, class_name)
        masks = load_masks(cache["image_path"], metadata, data_root)
        correction = action.astype(np.float32) * np.float32(ALPHA * MARGIN_SCALE)
        scores, losses = evaluate_correction(
            cache["native_logits"], masks, correction, device, args.batch_size
        )
        metric = exact_metrics(scores, masks)
        native_row = native[class_name]
        row = {
            "class": class_name,
            "native_pAP": float(native_row["pAP"]),
            "r1_pAP": metric["pAP"],
            "pAP_delta_pp": 100.0 * (metric["pAP"] - float(native_row["pAP"])),
            "native_pAUROC": float(native_row["pAUROC"]),
            "r1_pAUROC": metric["pAUROC"],
            "pAUROC_delta_pp": 100.0 * (metric["pAUROC"] - float(native_row["pAUROC"])),
            "native_loss": float(native_row["loss"]),
            "r1_loss": float(losses.mean()),
            "images": len(shard.image_path),
            "acted_patches": int(np.count_nonzero(action)),
            "coverage": float(np.mean(action != 0)),
        }
        if not all(np.isfinite(float(row[key])) for key in (
            "native_pAP", "r1_pAP", "pAP_delta_pp", "native_pAUROC",
            "r1_pAUROC", "pAUROC_delta_pp", "native_loss", "r1_loss", "coverage"
        )):
            raise RuntimeError(f"non-finite R1 class metric: {class_name}")
        per_class.append(row)
    write_csv(args.output / "per_class.csv", per_class)
    macro = {
        "native_pAP": float(np.mean([row["native_pAP"] for row in per_class])),
        "r1_pAP": float(np.mean([row["r1_pAP"] for row in per_class])),
        "pAP_delta_pp": float(np.mean([row["pAP_delta_pp"] for row in per_class])),
        "native_pAUROC": float(np.mean([row["native_pAUROC"] for row in per_class])),
        "r1_pAUROC": float(np.mean([row["r1_pAUROC"] for row in per_class])),
        "pAUROC_delta_pp": float(np.mean([row["pAUROC_delta_pp"] for row in per_class])),
        "native_loss": float(np.mean([row["native_loss"] for row in per_class])),
        "r1_loss": float(np.mean([row["r1_loss"] for row in per_class])),
        "nonnegative_pAP_classes": int(sum(row["pAP_delta_pp"] >= 0.0 for row in per_class)),
    }
    baseline = next(
        row for row in selection["threshold_rows"] if row["threshold"] == "unfiltered"
    )
    zero_exception = (
        baseline["opposite_sign_rate"] == 0.0 and risk["opposite_sign_rate"] == 0.0
    )
    gates = {
        "G0_correctness": {"threshold": "all PASS", "observed": "PASS", "pass": True},
        "G1_coverage": {
            "threshold": ">=0.10", "observed": risk["coverage"],
            "pass": risk["coverage"] >= 0.10,
        },
        "G2_opposite_sign_rate": {
            "threshold": "<=0.05", "observed": risk["opposite_sign_rate"],
            "pass": risk["opposite_sign_rate"] is not None and risk["opposite_sign_rate"] <= 0.05,
        },
        "G3_relative_risk_reduction": {
            "threshold": ">=0.25", "observed": risk["relative_opposite_sign_reduction"],
            "pass": zero_exception or (
                risk["relative_opposite_sign_reduction"] is not None
                and risk["relative_opposite_sign_reduction"] >= 0.25
            ),
        },
        "G4_macro_pAP_delta_pp": {
            "threshold": ">=0.50", "observed": macro["pAP_delta_pp"],
            "pass": macro["pAP_delta_pp"] >= 0.50,
        },
        "G5_macro_pAUROC_delta_pp": {
            "threshold": ">=-0.50", "observed": macro["pAUROC_delta_pp"],
            "pass": macro["pAUROC_delta_pp"] >= -0.50,
        },
        "G6_nonnegative_breadth": {
            "threshold": ">=7", "observed": macro["nonnegative_pAP_classes"],
            "pass": macro["nonnegative_pAP_classes"] >= 7,
        },
    }
    decision = "CONTINUE" if all(item["pass"] for item in gates.values()) else "STOP"
    summary = {
        "status": "COMPLETE",
        "stage": "R1",
        "selected_threshold": threshold,
        "alpha": ALPHA,
        "margin_scale": MARGIN_SCALE,
        "risk": risk,
        "unfiltered_risk": baseline,
        "macro": macro,
        "per_class": per_class,
        "gates": gates,
        "decision": decision,
        "next_stage": "R2" if decision == "CONTINUE" else "FINAL_DECISION",
        "elapsed_seconds": time.perf_counter() - started,
        "git_head": _git_head(),
        "provenance": common_provenance,
        "medical_access": False,
        "mvtec_access": False,
        "training_steps": 0,
    }
    write_json(args.output / "summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0/sabra_source"),
    )
    result.add_argument(
        "--trust-root",
        type=Path,
        default=ROOT / "runs/phase5/sabra/TRUST_V2_DEVELOPMENT",
    )
    result.add_argument(
        "--utility-root",
        type=Path,
        default=ROOT / "results/sabra_car/r0/utility",
    )
    result.add_argument("--fit-root", type=Path, default=ROOT / "results/sabra_car/r1")
    result.add_argument("--output", type=Path, default=ROOT / "results/sabra_car/r1")
    result.add_argument("--r0-per-class", type=Path, default=ROOT / "results/sabra_car/r0/per_class.csv")
    result.add_argument("--data-root", type=Path, default=Path("/home/ai4/caohuy/data"))
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--skip-hashes", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    args.output = args.output.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R1 evaluation")
    summary = run(args)
    print(json.dumps(
        {
            "decision": summary["decision"],
            "next_stage": summary["next_stage"],
            "selected_threshold": summary["selected_threshold"],
            "gates": summary["gates"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
