"""Aggregate fixed P30 fold scores against the frozen native and P29 baselines."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from tools.sabra.data import EXPECTED_VISA_CLASSES
from tools.sabra_v2.region_cache import atomic_write_json


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", choices=EXPECTED_VISA_CLASSES, default=list(EXPECTED_VISA_CLASSES))
    parser.add_argument("--p29-class-table", type=Path, required=True)
    return parser


def _p29_baseline(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["class"]: {
                "p29_pAP": float(row["p29_pAP"]),
                "p29_pAUROC": float(row["p29_pAUROC"]),
            }
            for row in csv.DictReader(handle)
        }


def run(args: argparse.Namespace) -> dict[str, object]:
    p29 = _p29_baseline(args.p29_class_table)
    rows: list[dict[str, object]] = []
    for name in args.classes:
        path = args.run_root / name / "metrics" / "p30_held_metrics.json"
        if not path.is_file():
            raise RuntimeError(f"all requested P30 folds must be scored before aggregation: {path}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("held_class") != name or metrics.get("fit_or_teacher_steps") != 0 or name not in p29:
            raise RuntimeError(f"P30 metric provenance failed: {name}")
        native = metrics["native_metrics"]
        student = metrics["p30_metrics"]
        row = {
            "class": name,
            "native_pAP": float(native["pAP"]),
            "p30_pAP": float(student["pAP"]),
            "delta_pAP": float(student["pAP"] - native["pAP"]),
            "p29_pAP": p29[name]["p29_pAP"],
            "p30_minus_p29_pAP": float(student["pAP"] - p29[name]["p29_pAP"]),
            "native_pAUROC": float(native["pAUROC"]),
            "p30_pAUROC": float(student["pAUROC"]),
            "delta_pAUROC": float(student["pAUROC"] - native["pAUROC"]),
            "p29_pAUROC": p29[name]["p29_pAUROC"],
            "p30_minus_p29_pAUROC": float(student["pAUROC"] - p29[name]["p29_pAUROC"]),
        }
        rows.append(row)
    p_ap = [float(row["delta_pAP"]) for row in rows]
    p_auc = [float(row["delta_pAUROC"]) for row in rows]
    result: dict[str, object] = {
        "schema_version": "P30_RESULTS_V1",
        "fold_count": len(rows),
        "classes": rows,
        "native_macro_pAP": statistics.fmean(float(row["native_pAP"]) for row in rows),
        "p30_macro_pAP": statistics.fmean(float(row["p30_pAP"]) for row in rows),
        "delta_macro_pAP": statistics.fmean(p_ap),
        "p29_macro_pAP": statistics.fmean(float(row["p29_pAP"]) for row in rows),
        "p30_minus_p29_macro_pAP": statistics.fmean(float(row["p30_minus_p29_pAP"]) for row in rows),
        "native_macro_pAUROC": statistics.fmean(float(row["native_pAUROC"]) for row in rows),
        "p30_macro_pAUROC": statistics.fmean(float(row["p30_pAUROC"]) for row in rows),
        "delta_macro_pAUROC": statistics.fmean(p_auc),
        "p29_macro_pAUROC": statistics.fmean(float(row["p29_pAUROC"]) for row in rows),
        "p30_minus_p29_macro_pAUROC": statistics.fmean(float(row["p30_minus_p29_pAUROC"]) for row in rows),
        "p30_improving_vs_native_pAP_count": sum(value > 0 for value in p_ap),
        "p30_non_regressing_vs_native_pAP_count": sum(value >= 0 for value in p_ap),
        "p30_regressing_vs_native_pAP_count": sum(value < 0 for value in p_ap),
        "p30_improving_vs_native_pAUROC_count": sum(value > 0 for value in p_auc),
        "p30_regressing_vs_native_pAUROC_count": sum(value < 0 for value in p_auc),
        "median_delta_pAP": statistics.median(p_ap),
        "best_delta_pAP": max(p_ap),
        "worst_delta_pAP": min(p_ap),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "P30_RESULTS.json", result)
    with (args.output / "P30_CLASS_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
