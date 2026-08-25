"""Aggregate all twelve frozen P27 LOCO scores without partial-fold interpretation."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from tools.sabra.data import EXPECTED_VISA_CLASSES
from tools.sabra_v2.region_cache import atomic_write_json


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for class_name in EXPECTED_VISA_CLASSES:
        path = args.run_root / class_name / "metrics" / "p27_held_metrics.json"
        if not path.is_file():
            raise RuntimeError(f"all 12 folds must be scored before aggregation; missing {path}")
        payload = json.loads(path.read_text())
        if payload.get("held_class") != class_name or payload.get("fit_or_teacher_steps") != 0:
            raise RuntimeError(f"invalid held metric provenance for {class_name}")
        native = payload["native_metrics"]
        p27 = payload["p27_metrics"]
        rows.append(
            {
                "class": class_name,
                "native_pAP": float(native["pAP"]),
                "p27_pAP": float(p27["pAP"]),
                "delta_pAP": float(p27["pAP"] - native["pAP"]),
                "native_pAUROC": float(native["pAUROC"]),
                "p27_pAUROC": float(p27["pAUROC"]),
                "delta_pAUROC": float(p27["pAUROC"] - native["pAUROC"]),
            }
        )
    deltas = [row["delta_pAP"] for row in rows]
    positive = sorted((delta for delta in deltas if delta > 0.0), reverse=True)
    positive_total = sum(positive)
    result = {
        "schema_version": "P27_AGGREGATE_V1",
        "fold_count": len(rows),
        "native_macro_pAP": statistics.fmean(row["native_pAP"] for row in rows),
        "p27_macro_pAP": statistics.fmean(row["p27_pAP"] for row in rows),
        "delta_macro_pAP": statistics.fmean(row["delta_pAP"] for row in rows),
        "native_macro_pAUROC": statistics.fmean(row["native_pAUROC"] for row in rows),
        "p27_macro_pAUROC": statistics.fmean(row["p27_pAUROC"] for row in rows),
        "delta_macro_pAUROC": statistics.fmean(row["delta_pAUROC"] for row in rows),
        "improving_class_count": sum(delta > 0.0 for delta in deltas),
        "non_regressing_class_count": sum(delta >= 0.0 for delta in deltas),
        "regressing_class_count": sum(delta < 0.0 for delta in deltas),
        "median_delta_pAP": statistics.median(deltas),
        "best_delta_pAP": max(deltas),
        "worst_delta_pAP": min(deltas),
        "top_1_positive_gain_concentration": positive[0] / positive_total if positive_total else 0.0,
        "top_2_positive_gain_concentration": sum(positive[:2]) / positive_total if positive_total else 0.0,
        "classes": rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "P27_AGGREGATE.json", result)
    csv_path = args.output / "P27_CLASS_METRICS.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
