#!/usr/bin/env python3
"""Scalar-only selection-semantic sensitivity audit for P5FR1CE1A.

This script deliberately does not import the historical evaluator and never
opens images, masks, checkpoints, common c/G caches, or GT.  It compares the
historical min-max interpretation with the plausible ordinal-rank
interpretation using only committed per-class/config scalar rows.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION"
FORENSIC = ROOT / "runs/phase5/hsir/P5FR1CE1A_FINAL_FORENSIC"
FAMILIES = ("PCRR", "CSRC", "ASR", "PGM")
SEEDS = {
    "matched_win": 5101,
    "b1_matched_win": 5102,
    "delta_vs_B1": 5103,
    "aligned_minus_shifted": 5104,
    "C_AP_delta": 5105,
    "R_pos_delta": 5106,
    "R_neg_delta": 5107,
}
METRICS = ("delta_vs_b1", "C_AP_delta", "R_pos_delta", "R_neg_delta")


def bootstrap(values: list[float], seed: int) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = arr[rng.integers(0, arr.size, size=(2000, arr.size))]
    means = sample.mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "per_class": [float(x) for x in arr],
        "n": int(arr.size),
        "seed": seed,
        "reps": 2000,
    }


def exact_sign_flip(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    observed = float(arr.mean())
    count = 0
    for bits in range(1 << arr.size):
        signs = np.asarray([1.0 if (bits >> i) & 1 else -1.0 for i in range(arr.size)])
        if float(np.mean(arr * signs)) >= observed - 1e-15:
            count += 1
    return float(count / (1 << arr.size))


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda pair: (pair[1], pair[0]))
    running = 0.0
    out: dict[str, float] = {}
    for index, (name, value) in enumerate(ordered):
        running = max(running, (len(ordered) - index) * value)
        out[name] = min(running, 1.0)
    return out


def load() -> tuple[dict, dict, dict, dict]:
    configs = json.loads((HISTORICAL / "CANONICAL_CONFIGS.json").read_text())
    rows = json.loads((HISTORICAL / "CONFIG_METRICS.json").read_text())
    folds = json.loads((HISTORICAL / "FOLD_ASSIGNMENT.json").read_text())["folds"]
    historical = json.loads((HISTORICAL / "FOLD_SELECTIONS.json").read_text())
    return configs, rows, folds, historical


def margin_values(rows: dict[str, dict], dev: list[str]) -> tuple[list[float], int, int]:
    metrics = {
        "delta_vs_B1": bootstrap([rows[c]["delta_vs_b1"] for c in dev], SEEDS["delta_vs_B1"]),
        "C_AP_delta": bootstrap([rows[c]["C_AP_delta"] for c in dev], SEEDS["C_AP_delta"]),
        "R_pos_delta": bootstrap([rows[c]["R_pos_delta"] for c in dev], SEEDS["R_pos_delta"]),
        "R_neg_delta": bootstrap([rows[c]["R_neg_delta"] for c in dev], SEEDS["R_neg_delta"]),
    }
    margins = [
        metrics["delta_vs_B1"]["ci95"][0],
        metrics["C_AP_delta"]["ci95"][0],
        metrics["R_pos_delta"]["ci95"][0],
        -metrics["R_neg_delta"]["ci95"][1],
    ]
    gate_count = sum(x > 0 for x in margins[:3]) + int(margins[3] >= 0)
    direction_count = sum(rows[c]["delta_vs_b1"] > 0 for c in dev)
    direction_count += sum(rows[c]["C_AP_delta"] > 0 for c in dev)
    direction_count += sum(rows[c]["R_pos_delta"] > 0 for c in dev)
    direction_count += sum(rows[c]["R_neg_delta"] <= 0 for c in dev)
    return [float(x) for x in margins], int(gate_count), int(direction_count)


def normalize_minmax(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=np.float64)
    for column in range(matrix.shape[1]):
        low, high = matrix[:, column].min(), matrix[:, column].max()
        out[:, column] = 1.0 if high == low else (matrix[:, column] - low) / (high - low)
    return out


def normalize_ordinal(matrix: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    """Best margin gets 1 and worst gets 0; ties receive average rank."""
    out = np.zeros_like(matrix, dtype=np.float64)
    tie_groups: list[list[float]] = []
    n = matrix.shape[0]
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        unique = sorted(set(float(x) for x in values), reverse=True)
        tie_groups.append([float(x) for x in unique if int(np.sum(values == x)) > 1])
        for value in unique:
            indexes = np.flatnonzero(values == value)
            zero_based_positions = np.asarray([int(np.flatnonzero(values == v)[0]) for v in []])
            del zero_based_positions
            descending_positions = np.flatnonzero(np.isin(values, [value]))
            # Position is based on the sorted value list, independent of config order.
            better = sum(int(np.sum(values > value)) for _ in [0])
            average_position = better + (indexes.size - 1) / 2.0
            out[indexes, column] = 1.0 if n == 1 else 1.0 - average_position / (n - 1)
    return out, tie_groups


def score_family(family: str, configs: list[dict], rows: dict[str, dict], folds: dict, method: str) -> dict:
    selections: dict[str, dict] = {}
    diff_rows: list[dict] = []
    for fold, holdout in folds.items():
        dev = sorted(c for c in rows[next(iter(rows))] if c not in holdout)
        scored = []
        for config in configs:
            cid = config["config_id"]
            margins, gate_count, direction_count = margin_values(rows[cid], dev)
            scored.append({
                "config": config,
                "margins": margins,
                "gate_count": gate_count,
                "direction_count": direction_count,
            })
        matrix = np.asarray([item["margins"] for item in scored], dtype=np.float64)
        normalized = normalize_minmax(matrix) if method == "minmax" else normalize_ordinal(matrix)[0]
        for index, item in enumerate(scored):
            item["worst_rank"] = float(normalized[index].min())
            item["mean_rank"] = float(normalized[index].mean())
        scored.sort(key=lambda item: (
            -item["gate_count"], -item["direction_count"], -item["worst_rank"],
            -item["mean_rank"], item["config"]["complexity_rank"], item["config"]["config_id"],
        ))
        selected = scored[0]
        selections[fold] = {
            "selected_config_id": selected["config"]["config_id"],
            "selected_config": selected["config"],
            "dev_classes": dev,
            "holdout_classes": holdout,
            "gate_count": selected["gate_count"],
            "direction_count": selected["direction_count"],
            "worst_rank": selected["worst_rank"],
            "mean_rank": selected["mean_rank"],
        }
        historical_id = json.loads((HISTORICAL / "FOLD_SELECTIONS.json").read_text())[family][fold]["selected_config_id"]
        diff_rows.append({
            "family": family,
            "fold": fold,
            "historical_config": historical_id,
            "reconciled_config": selected["config"]["config_id"],
            "changed": historical_id != selected["config"]["config_id"],
            "reason": "ordinal_rank_counterfactual" if method == "ordinal" else "historical_minmax_reproduction",
            "historical_gate_count": None,
            "reconciled_gate_count": selected["gate_count"],
            "historical_direction_count": None,
            "reconciled_direction_count": selected["direction_count"],
            "historical_worst_rank": None,
            "reconciled_worst_rank": selected["worst_rank"],
            "historical_mean_rank": None,
            "reconciled_mean_rank": selected["mean_rank"],
        })
    return {"family": family, "method": method, "selections": selections, "diff_rows": diff_rows}


def reconstruct(method_results: dict[str, dict], rows: dict[str, dict], folds: dict) -> dict:
    standard: dict[str, dict] = {}
    for family, result in method_results.items():
        oof = []
        for fold, selection in result["selections"].items():
            selected = selection["selected_config_id"]
            for class_name in selection["holdout_classes"]:
                row = dict(rows[family][selected][class_name])
                row["outer_fold"] = fold
                row["family"] = family
                row["config_id"] = selected
                oof.append(row)
        metrics = {}
        fields = {
            "matched_win": "matched_win",
            "b1_matched_win": "b1_matched_win",
            "delta_vs_B1": "delta_vs_b1",
            "aligned_minus_shifted": "aligned_minus_shifted",
            "C_AP_delta": "C_AP_delta",
            "R_pos_delta": "R_pos_delta",
            "R_neg_delta": "R_neg_delta",
        }
        for output, field in fields.items():
            metrics[output] = bootstrap([r[field] for r in oof], SEEDS[output])
        supportive = sum(x > 0.5 for x in metrics["matched_win"]["per_class"])
        positive = sum(x > 0 for x in metrics["delta_vs_B1"]["per_class"])
        aligned = sum(x > 0 for x in metrics["aligned_minus_shifted"]["per_class"])
        gates = {
            "G0": True,
            "G1": metrics["matched_win"]["ci95"][0] > 0.5 and supportive >= 10,
            "G2": metrics["delta_vs_B1"]["ci95"][0] > 0 and positive >= 10,
            "G3": metrics["aligned_minus_shifted"]["ci95"][0] > 0 and aligned >= 10,
            "G4": metrics["C_AP_delta"]["ci95"][0] > 0 and metrics["R_pos_delta"]["ci95"]["0"] if False else (
                metrics["C_AP_delta"]["ci95"][0] > 0
                and metrics["R_pos_delta"]["ci95"][0] > 0
                and metrics["R_neg_delta"]["ci95"][1] <= 0
            ),
            "supportive_classes": supportive,
            "positive_direction_classes": positive,
            "aligned_better_classes": aligned,
        }
        standard[family] = {"metrics": metrics, "gates": gates, "oof": oof}
    raw = {family: exact_sign_flip(standard[family]["metrics"]["delta_vs_B1"]["per_class"]) for family in FAMILIES}
    adjusted = holm(raw)
    eligible = [
        family for family in FAMILIES
        if all(standard[family]["gates"][gate] for gate in ("G0", "G1", "G2", "G3", "G4"))
        and adjusted[family] < 0.05
    ]
    ranking = sorted(FAMILIES, key=lambda family: (-standard[family]["metrics"]["delta_vs_B1"]["mean"], family))
    return {
        "standard": standard,
        "raw_one_sided_p": raw,
        "holm_adjusted_p": adjusted,
        "empirical_ranking": ranking,
        "fully_eligible_families": eligible,
        "provisional_winner": eligible[0] if len(eligible) == 1 else "NONE",
    }


def main() -> None:
    configs, config_rows, folds, historical = load()
    method_results: dict[str, dict] = {}
    for method in ("minmax", "ordinal"):
        method_results[method] = {}
        for family in FAMILIES:
            method_results[method][family] = score_family(family, configs["families"][family], config_rows[family], folds, method)
    reconstructed = {
        method: reconstruct(method_results[method], config_rows, folds)
        for method in method_results
    }
    ordinal_diff = [row for family in FAMILIES for row in method_results["ordinal"][family]["diff_rows"]]
    FORENSIC.mkdir(parents=True, exist_ok=True)
    with (FORENSIC / "SELECTION_DIFF.csv").open("w", newline="") as handle:
        fields = list(ordinal_diff[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordinal_diff)
    historical_ids = {family: {fold: historical[family][fold]["selected_config_id"] for fold in folds} for family in FAMILIES}
    ordinal_ids = {family: {fold: method_results["ordinal"][family]["selections"][fold]["selected_config_id"] for fold in folds} for family in FAMILIES}
    changes = {
        family: sum(historical_ids[family][fold] != ordinal_ids[family][fold] for fold in folds)
        for family in FAMILIES
    }
    payload = {
        "schema_version": "P5FR1CE1A_SELECTION_SEMANTIC_SENSITIVITY_V1",
        "source_artifacts": [
            "runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION/CANONICAL_CONFIGS.json",
            "runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION/CONFIG_METRICS.json",
            "runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION/FOLD_ASSIGNMENT.json",
            "runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION/FOLD_SELECTIONS.json",
        ],
        "no_new_gt_image_mask_or_model_access": True,
        "interpretations": {
            "historical_minmax": "per-metric raw margin min-max normalization across configs",
            "ordinal_counterfactual": "per-metric descending ordinal rank normalized to [0,1], average ties",
        },
        "ties": {
            family: {
                "fold": {
                    "method": "ordinal",
                    "selected_config_id": method_results["ordinal"][family]["selections"][fold]["selected_config_id"],
                }
                for fold in folds
            }
            for family in FAMILIES
        },
        "selection_diff": {
            "SELECTIONS_CHANGED_TOTAL": int(sum(changes.values())),
            "PCRR_CHANGED_FOLDS": changes["PCRR"],
            "CSRC_CHANGED_FOLDS": changes["CSRC"],
            "ASR_CHANGED_FOLDS": changes["ASR"],
            "PGM_CHANGED_FOLDS": changes["PGM"],
        },
        "methods": reconstructed,
    }
    (FORENSIC / "SELECTION_SEMANTIC_SENSITIVITY.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["selection_diff"], sort_keys=True))
    print(json.dumps({method: reconstructed[method]["empirical_ranking"] for method in reconstructed}, sort_keys=True))


if __name__ == "__main__":
    main()
