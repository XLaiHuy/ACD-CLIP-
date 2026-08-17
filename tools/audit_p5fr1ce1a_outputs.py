#!/usr/bin/env python3
"""Independent scalar-only checker for the P5FR1CE1A forensic audit.

The checker intentionally does not import the historical evaluator and never
opens images, masks, checkpoints, common feature caches, or GT.  It validates
only the committed scalar outputs, fold isolation, frozen configuration
shape, bootstrap/sign-flip/Holm reconstruction, and process counters.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION"
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
FIELDS = ("matched_win", "b1_matched_win", "delta_vs_b1", "aligned_minus_shifted", "C_AP_delta", "R_pos_delta", "R_neg_delta")


def read_json(name: str) -> dict:
    return json.loads((BASE / name).read_text())


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


def sign_flip(values: list[float]) -> float:
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


def close(a: object, b: object, tol: float = 1e-12) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return bool(np.isclose(float(a), float(b), rtol=0.0, atol=tol))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return a == b


def main() -> None:
    checks: dict[str, bool] = {}
    configs = read_json("CANONICAL_CONFIGS.json")
    rows = read_json("CONFIG_METRICS.json")
    folds = read_json("FOLD_ASSIGNMENT.json")["folds"]
    selections = read_json("FOLD_SELECTIONS.json")
    standard = read_json("STANDARD_METRICS.json")
    gates = read_json("SCIENTIFIC_GATES.json")
    multiplicity = read_json("MULTIPLICITY_TESTS.json")
    ranking = read_json("EMPIRICAL_RANKING.json")
    zero_tune = read_json("ZERO_TUNE_RESULT.json")
    input_lock = read_json("INPUT_LOCK.json")
    derived = read_json("GT_FREE_DERIVED_MANIFEST.json")

    family_ids = {
        family: [item["config_id"] for item in configs["families"][family]]
        for family in FAMILIES
    }
    all_ids = [cid for family in FAMILIES for cid in family_ids[family]]
    checks["families_exact"] = tuple(configs["families"]) == FAMILIES
    checks["config_count_exact"] = configs["total_configs"] == 26 and len(all_ids) == 26
    checks["config_ids_unique"] = len(set(all_ids)) == 26
    checks["zero_tune_ids_exact"] = all(
        zero_tune[family]["config_id"] == configs["canonical_zero_tune"][family]
        for family in FAMILIES
    )

    for family in FAMILIES:
        ids = family_ids[family]
        checks[f"{family}_config_metrics_complete"] = set(rows[family]) == set(ids)
        checks[f"{family}_classes_once"] = all(
            len(rows[family][cid]) == 15 and len(set(rows[family][cid])) == 15
            for cid in ids
        )
        for cid in ids:
            values = rows[family][cid]
            checks[f"{family}_{cid}_finite"] = all(
                np.isfinite(float(values[class_name][field]))
                for class_name in values
                for field in FIELDS
            )

    oof_rows = list(csv.DictReader((BASE / "OOF_PER_CLASS.csv").open()))
    checks["oof_rows_exact"] = len(oof_rows) == 60
    for family in FAMILIES:
        family_oof = [row for row in oof_rows if row["family"] == family]
        checks[f"{family}_oof_rows_exact"] = len(family_oof) == 15
        checks[f"{family}_oof_classes_once"] = len({row["class"] for row in family_oof}) == 15
        for fold, holdout in folds.items():
            item = selections[family][fold]
            selected = item["selected_config_id"]
            dev = set(item["dev_classes"])
            held = set(item["holdout_classes"])
            checks[f"{family}_{fold}_selection_isolated"] = (
                dev.isdisjoint(held) and dev | held == set(rows[family][selected])
                and all(row["config_id"] == selected for row in family_oof if row["outer_fold"] == fold)
                and set(row["class"] for row in family_oof if row["outer_fold"] == fold) == held
            )

    rebuilt: dict[str, dict] = {}
    for family in FAMILIES:
        by_class = {row["class"]: row for row in oof_rows if row["family"] == family}
        class_order = standard[family]["classes"]
        rebuilt[family] = {
            output: bootstrap([float(by_class[c][field]) for c in class_order], seed)
            for output, field, seed in (
                ("matched_win", "matched_win", SEEDS["matched_win"]),
                ("b1_matched_win", "b1_matched_win", SEEDS["b1_matched_win"]),
                ("delta_vs_B1", "delta_vs_b1", SEEDS["delta_vs_B1"]),
                ("aligned_minus_shifted", "aligned_minus_shifted", SEEDS["aligned_minus_shifted"]),
                ("C_AP_delta", "C_AP_delta", SEEDS["C_AP_delta"]),
                ("R_pos_delta", "R_pos_delta", SEEDS["R_pos_delta"]),
                ("R_neg_delta", "R_neg_delta", SEEDS["R_neg_delta"]),
            )
        }
        checks[f"{family}_metrics_reconstruct"] = all(
            close(rebuilt[family][name]["mean"], standard[family]["metrics"][name]["mean"])
            and close(rebuilt[family][name]["ci95"], standard[family]["metrics"][name]["ci95"])
            and close(rebuilt[family][name]["per_class"], standard[family]["metrics"][name]["per_class"])
            and rebuilt[family][name]["n"] == standard[family]["metrics"][name]["n"]
            and rebuilt[family][name]["seed"] == standard[family]["metrics"][name]["seed"]
            and rebuilt[family][name]["reps"] == standard[family]["metrics"][name]["reps"]
            for name in rebuilt[family]
        )
        m = rebuilt[family]
        supportive = sum(x > 0.5 for x in m["matched_win"]["per_class"])
        positive = sum(x > 0 for x in m["delta_vs_B1"]["per_class"])
        aligned = sum(x > 0 for x in m["aligned_minus_shifted"]["per_class"])
        reconstructed_gates = {
            "G0": True,
            "G1": m["matched_win"]["ci95"][0] > 0.5 and supportive >= 10,
            "G2": m["delta_vs_B1"]["ci95"][0] > 0 and positive >= 10,
            "G3": m["aligned_minus_shifted"]["ci95"][0] > 0 and aligned >= 10,
            "G4": (
                m["C_AP_delta"]["ci95"][0] > 0
                and m["R_pos_delta"]["ci95"][0] > 0
                and m["R_neg_delta"]["ci95"][1] <= 0
            ),
            "supportive_classes": supportive,
            "positive_direction_classes": positive,
            "aligned_better_classes": aligned,
        }
        checks[f"{family}_gates_reconstruct"] = all(
            reconstructed_gates[name] == gates[family][name]
            for name in reconstructed_gates
        )

    raw = {family: sign_flip(rebuilt[family]["delta_vs_B1"]["per_class"]) for family in FAMILIES}
    adjusted = holm(raw)
    checks["signflip_reconstruct"] = all(close(raw[f], multiplicity["raw_one_sided_p"][f]) for f in FAMILIES)
    checks["holm_reconstruct"] = all(close(adjusted[f], multiplicity["holm_adjusted_p"][f]) for f in FAMILIES)
    rebuilt_ranking = sorted(FAMILIES, key=lambda family: (-rebuilt[family]["delta_vs_B1"]["mean"], family))
    checks["ranking_reconstruct"] = rebuilt_ranking == ranking["ranking"]
    eligible = [
        family for family in FAMILIES
        if all(gates[family][gate] for gate in ("G0", "G1", "G2", "G3", "G4"))
        and adjusted[family] < 0.05
    ]
    checks["eligibility_reconstruct"] = eligible == []
    checks["winner_reconstruct"] = ranking["provisional_winner"] == "NONE"

    checks["historical_counters_zero"] = (
        input_lock["model_forwards_in_p5fr1c"] == 0
        and derived["model_forwards"] == 0
        and input_lock["model_rerun"] is False
        and input_lock["model_resume"] is False
        and derived["images_opened"] == 0
        and derived["masks_read"] == 0
        and derived["GT_metrics_read"] is False
        and input_lock["scientific_metrics_read"] is False
        and derived["medical"] is False
        and derived["training_steps"] == 0
    )
    checks["no_external_winner"] = read_json("DECISION.json")["final_external_winner"] is False
    checks["no_head_to_head_winner"] = read_json("HEAD_TO_HEAD.json")["winner_status"] == "NONE"

    payload = {
        "schema_version": "P5FR1CE1A_INDEPENDENT_OUTPUT_CHECK_V1",
        "checker": "tools/audit_p5fr1ce1a_outputs.py",
        "scope": "committed scalar outputs only; no GT/image/mask/model access",
        "source_commit": "2ef784ff91b91e3b2c2c880dfaa74c02e94445d2",
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model_forwards": 0,
        "new_image_reads": 0,
        "new_mask_pixel_reads": 0,
        "training_steps": 0,
        "medical": False,
    }
    FORENSIC.mkdir(parents=True, exist_ok=True)
    (FORENSIC / "INDEPENDENT_OUTPUT_CHECK.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"OUTPUT_CHECK={payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
