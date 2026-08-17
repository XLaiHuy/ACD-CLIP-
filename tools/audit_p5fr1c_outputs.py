#!/usr/bin/env python3
"""Independent scalar/CSV output checker for P5FR1C."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

FAMILIES = ("PCRR", "CSRC", "ASR", "PGM")
REPS = 2000
TOL = 1e-12
SIGN_TOL = 1e-15


def one_sided(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    observed = float(arr.mean())
    count = 0
    for bits in range(1 << arr.size):
        signs = np.asarray([1.0 if (bits >> i) & 1 else -1.0 for i in range(arr.size)])
        if float(np.mean(signs * arr)) >= observed - SIGN_TOL:
            count += 1
    return count / (1 << arr.size)


def holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: (item[1], item[0]))
    out: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, (len(ordered) - index) * float(value))
        out[name] = min(running, 1.0)
    return out


def bootstrap(values: list[float], seed: int) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = arr[rng.integers(0, arr.size, size=(REPS, arr.size))]
    means = sample.mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def close(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=TOL)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key]) for key in left)
    return left == right


def load(root: Path, name: str) -> Any:
    return json.loads((root / name).read_text())


def frozen_configs(root: Path) -> dict[str, list[dict[str, Any]]]:
    return load(root, "CANONICAL_CONFIGS.json")["families"]


def fold_map(root: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    folds = load(root, "FOLD_ASSIGNMENT.json")["folds"]
    return folds, {class_name: fold for fold, classes in folds.items() for class_name in classes}


def recompute_gate(standard: dict[str, Any], gate_doc: dict[str, Any]) -> dict[str, Any]:
    metrics = standard["metrics"]
    supportive = sum(value > 0.5 for value in metrics["matched_win"]["per_class"])
    positive = sum(value > 0 for value in metrics["delta_vs_B1"]["per_class"])
    aligned = sum(value > 0 for value in metrics["aligned_minus_shifted"]["per_class"])
    subchecks = gate_doc.get("G0_subchecks")
    if not isinstance(subchecks, dict) or not subchecks:
        raise ValueError("missing G0 subchecks")
    return {
        "G0": bool(all(bool(value) for value in subchecks.values())),
        "G0_subchecks": subchecks,
        "G1": metrics["matched_win"]["ci95"][0] > 0.5 and supportive >= 10,
        "G2": metrics["delta_vs_B1"]["ci95"][0] > 0 and positive >= 10,
        "G3": metrics["aligned_minus_shifted"]["ci95"][0] > 0 and aligned >= 10,
        "G4": metrics["C_AP_delta"]["ci95"][0] > 0 and metrics["R_pos_delta"]["ci95"][0] > 0 and metrics["R_neg_delta"]["ci95"][1] <= 0,
        "supportive_classes": supportive,
        "positive_direction_classes": positive,
        "aligned_better_classes": aligned,
    }


def validate_gate_document(standard: dict[str, Any], gates: dict[str, Any]) -> bool:
    try:
        return all(recompute_gate(standard[family], gates[family]) == gates[family] for family in FAMILIES)
    except (KeyError, TypeError, ValueError):
        return False


def validate_winner_document(ranking: dict[str, Any], decision: dict[str, Any]) -> bool:
    return (
        decision.get("empirical_scientific_ranking") == ranking.get("ranking")
        and decision.get("empirical_provisional_winner") == ranking.get("provisional_winner")
        and decision.get("runner_up") == ranking.get("runner_up")
    )


def recompute_selection(configs: list[dict[str, Any]], metrics_by_config: dict[str, dict[str, dict[str, Any]]], dev_classes: list[str]) -> str:
    scored = []
    for config in configs:
        config_id = config["config_id"]
        rows = [metrics_by_config[config_id][class_name] for class_name in dev_classes]
        margins = [
            bootstrap([float(row["delta_vs_b1"]) for row in rows], 5103)["ci95"][0],
            bootstrap([float(row["C_AP_delta"]) for row in rows], 5105)["ci95"][0],
            bootstrap([float(row["R_pos_delta"]) for row in rows], 5106)["ci95"][0],
            -bootstrap([float(row["R_neg_delta"]) for row in rows], 5107)["ci95"][1],
        ]
        gate_count = sum(value > 0 for value in margins[:3]) + int(margins[3] >= 0)
        direction_count = sum(float(row[key]) > 0 for row in rows for key in ("delta_vs_b1", "C_AP_delta", "R_pos_delta")) + sum(float(row["R_neg_delta"]) <= 0 for row in rows)
        scored.append({"config_id": config_id, "complexity_rank": config["complexity_rank"], "gate_count": gate_count, "direction_count": direction_count, "margins": margins})
    matrix = np.asarray([item["margins"] for item in scored], dtype=np.float64)
    ranks = np.zeros_like(matrix)
    for column in range(4):
        low, high = float(matrix[:, column].min()), float(matrix[:, column].max())
        ranks[:, column] = 1.0 if high == low else (matrix[:, column] - low) / (high - low)
    for index, item in enumerate(scored):
        item["worst_margin_rank"] = float(ranks[index].min())
        item["mean_margin_rank"] = float(ranks[index].mean())
    scored.sort(key=lambda item: (-item["gate_count"], -item["direction_count"], -item["worst_margin_rank"], -item["mean_margin_rank"], item["complexity_rank"], item["config_id"]))
    return scored[0]["config_id"]


def validate_head_to_head(root: Path, oof: list[dict[str, str]], gates: dict[str, Any], holm_p: dict[str, float], ranking: list[str], decision: dict[str, Any], head: dict[str, Any]) -> bool:
    eligible = [family for family in FAMILIES if all(bool(gates[family][key]) for key in ("G0", "G1", "G2", "G3", "G4")) and holm_p[family] < 0.05]
    eligible_ranked = [family for family in ranking if family in eligible]
    raw: dict[str, float] = {}
    comparisons: dict[str, Any] = {}
    if len(eligible_ranked) > 1:
        best = eligible_ranked[0]
        by_family = {family: {row["class"]: row for row in oof if row["family"] == family} for family in FAMILIES}
        for competitor in eligible_ranked[1:]:
            key = f"{best}_vs_{competitor}"
            values = [float(by_family[best][class_name]["matched_win"]) - float(by_family[competitor][class_name]["matched_win"]) for class_name in sorted(by_family[best])]
            raw[key] = one_sided(values)
            comparisons[key] = {"best": best, "competitor": competitor, "per_class": values, "raw_p": raw[key]}
    adjusted = holm(raw)
    expected_status = "NONE" if not eligible_ranked else "DECISIVE_PROVISIONAL_WINNER" if len(eligible_ranked) == 1 or all(value < 0.05 for value in adjusted.values()) else "BEST_OBSERVED_NOT_SEPARATED"
    expected_winner = "NONE" if not eligible_ranked else eligible_ranked[0]
    expected_runner = "NONE" if len(eligible_ranked) <= 1 else eligible_ranked[1]
    return (head.get("raw_one_sided_p") == raw and close(head.get("holm_adjusted_p"), adjusted) and close(head.get("comparisons"), comparisons) and head.get("winner_status") == expected_status and decision.get("empirical_provisional_winner") == expected_winner and decision.get("runner_up") == expected_runner)


def check(root: Path, write: bool = False) -> dict[str, bool]:
    configs = frozen_configs(root)
    folds, class_to_fold = fold_map(root)
    standard = load(root, "STANDARD_METRICS.json")
    gates = load(root, "SCIENTIFIC_GATES.json")
    multiplicity = load(root, "MULTIPLICITY_TESTS.json")
    ranking_doc = load(root, "EMPIRICAL_RANKING.json")
    decision = load(root, "DECISION.json")
    head = load(root, "HEAD_TO_HEAD.json")
    zero = load(root, "ZERO_TUNE_RESULT.json")
    selections = load(root, "FOLD_SELECTIONS.json")
    config_metrics = load(root, "CONFIG_METRICS.json")
    checks: dict[str, bool] = {}
    expected_ids = {family: [config["config_id"] for config in configs[family]] for family in FAMILIES}
    checks["families_exact"] = set(configs) == set(FAMILIES) and sum(len(ids) for ids in expected_ids.values()) == 26
    with (root / "OOF_PER_CLASS.csv").open(newline="") as handle:
        oof = list(csv.DictReader(handle))
    checks["oof_rows_exact"] = len(oof) == 60
    for family in FAMILIES:
        family_rows = [row for row in oof if row["family"] == family]
        classes = [row["class"] for row in family_rows]
        checks[f"{family}_classes_once"] = len(family_rows) == 15 and len(set(classes)) == 15 and all(class_to_fold.get(row["class"]) == row["outer_fold"] for row in family_rows)
        checks[f"{family}_configs_frozen"] = all(row["config_id"] in expected_ids[family] for row in family_rows)
        checks[f"{family}_config_metrics_complete"] = set(config_metrics[family]) == set(expected_ids[family]) and all(set(config_metrics[family][config_id]) == set(class_to_fold) for config_id in expected_ids[family])
        for fold, selection in selections[family].items():
            expected_holdout = folds[fold]
            expected_dev = sorted(class_to_fold.keys() - set(expected_holdout))
            checks[f"{family}_{fold}_selection_isolated"] = (selection["holdout_classes"] == expected_holdout and selection["dev_classes"] == expected_dev and selection["selected_config_id"] in expected_ids[family] and recompute_selection(configs[family], config_metrics[family], expected_dev) == selection["selected_config_id"])
    checks["gates_reconstruct"] = all(recompute_gate(standard[family], gates[family]) == gates[family] for family in FAMILIES)
    raw = {family: one_sided(standard[family]["metrics"]["delta_vs_B1"]["per_class"]) for family in FAMILIES}
    checks["signflip_reconstruct"] = all(close(raw[family], multiplicity["raw_one_sided_p"][family]) for family in FAMILIES)
    adjusted = holm(raw)
    checks["holm_reconstruct"] = close(adjusted, multiplicity["holm_adjusted_p"])
    ranking = sorted(FAMILIES, key=lambda family: (-standard[family]["metrics"]["delta_vs_B1"]["mean"], family))
    checks["ranking_reconstruct"] = ranking_doc["ranking"] == ranking
    eligible = [family for family in FAMILIES if all(bool(gates[family][key]) for key in ("G0", "G1", "G2", "G3", "G4")) and adjusted[family] < 0.05]
    eligible_ranked = [family for family in ranking if family in eligible]
    expected_winner = eligible_ranked[0] if eligible_ranked else "NONE"
    expected_runner = eligible_ranked[1] if len(eligible_ranked) > 1 else "NONE"
    expected_status = "NONE" if not eligible_ranked else "DECISIVE_PROVISIONAL_WINNER" if len(eligible_ranked) == 1 else "BEST_OBSERVED_NOT_SEPARATED"
    checks["eligibility_reconstruct"] = decision["fully_eligible_families"] == eligible
    checks["winner_reconstruct"] = (ranking_doc["provisional_winner"] == expected_winner and ranking_doc["runner_up"] == expected_runner and decision["empirical_scientific_ranking"] == ranking and decision["empirical_provisional_winner"] == expected_winner and decision["runner_up"] == expected_runner)
    checks["head_to_head_reconstruct"] = validate_head_to_head(root, oof, gates, adjusted, ranking, decision, head)
    canonical = load(root, "CANONICAL_CONFIGS.json")["canonical_zero_tune"]
    checks["zero_tune_ids_exact"] = all(zero[family]["config_id"] == canonical[family] and zero[family]["result"]["n_classes"] == 15 for family in FAMILIES)
    try:
        checks["values_finite"] = all(np.isfinite(float(row[key])) for row in oof for key in ("matched_win", "b1_matched_win", "delta_vs_b1", "aligned_minus_shifted", "C_AP_delta", "R_pos_delta", "R_neg_delta", "pixel_auroc", "pixel_ap"))
    except (TypeError, ValueError):
        checks["values_finite"] = False
    checks["model_forwards_zero"] = decision.get("model_forwards") == 0
    checks["training_zero"] = decision.get("training_steps") == 0
    checks["medical_false"] = decision.get("medical") is False
    checks["candidate_none"] = decision.get("candidate") == "NONE"
    checks["final_external_winner_false"] = decision.get("final_external_winner") is False
    if expected_winner == "NONE":
        checks["final_config_not_selected"] = decision.get("final_mvtec_selected_config") is None
    else:
        final_config = decision.get("final_mvtec_selected_config") or {}
        checks["final_config_selected_from_full_dev"] = final_config.get("config_id") == recompute_selection(configs[expected_winner], config_metrics[expected_winner], sorted(class_to_fold))
    checks["all_pass"] = all(checks.values())
    if write:
        payload = {"schema_version": "P5FR1C_OUTPUT_CHECK_V1", "status": "PASS" if checks["all_pass"] else "FAIL", "checks": checks, "model_forwards": 0, "training_steps": 0, "medical": False, "GT_access_after_freeze": True}
        (root / "OUTPUT_CHECK.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    checks = check(args.root, args.write)
    print(json.dumps({"status": "PASS" if checks["all_pass"] else "FAIL", "checks": checks}, indent=2, sort_keys=True))
    raise SystemExit(0 if checks["all_pass"] else 1)


if __name__ == "__main__":
    main()
