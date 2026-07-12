#!/usr/bin/env python3
"""Create a reproducible A-prime/B diagnostic report from Phase2C artifacts."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("pixel_auc", "pixel_ap", "image_auc", "image_ap")
FOCUS_EPOCHS = (4, 5, 6, 10, 11, 12, 13)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Phase2C A-prime/B diagnostics")
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--b-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conflict-rate-threshold", type=float, default=0.50,
        help="Flag a shared path when its negative-cosine fraction exceeds this threshold.",
    )
    parser.add_argument(
        "--norm-ratio-threshold", type=float, default=10.0,
        help="Flag a group when median max(cls, seg)/min(cls, seg) exceeds this threshold.",
    )
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(value):
    return None if value in ("", "NA", None) else float(value)


def median(values):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def selected_epoch(run_dir):
    with (run_dir / "selection.json").open() as handle:
        return json.load(handle)["selected_epoch"]


def macro_at_epoch(rows, epoch):
    matches = [row for row in rows if row["scope"] == "macro" and int(row["epoch"]) == epoch]
    if len(matches) != 1:
        raise ValueError(f"Expected one macro row at epoch {epoch}, found {len(matches)}")
    return matches[0]


def category_at_epoch(rows, epoch):
    return {
        row["class_name"]: row
        for row in rows
        if row["scope"] == "category" and int(row["epoch"]) == epoch
    }


def metric_delta(a_row, b_row):
    return {metric: float(b_row[metric]) - float(a_row[metric]) for metric in METRICS}


def gradient_summary(rows, conflict_rate_threshold, norm_ratio_threshold):
    grouped = defaultdict(list)
    for row in rows:
        epoch = int(row["epoch"])
        if epoch not in FOCUS_EPOCHS:
            continue
        cosine = numeric(row["cosine"])
        cls_norm = numeric(row["cls_grad_norm"])
        seg_norm = numeric(row["seg_grad_norm"])
        if cosine is None or cls_norm is None or seg_norm is None:
            continue
        ratio = max(cls_norm, seg_norm) / max(min(cls_norm, seg_norm), 1e-12)
        grouped[(epoch, row["parameter_group"])].append((cosine, ratio))

    summary = []
    for (epoch, group), values in sorted(grouped.items()):
        cosines = [value[0] for value in values]
        ratios = [value[1] for value in values]
        negative_rate = sum(value < 0 for value in cosines) / len(cosines)
        median_ratio = median(ratios)
        summary.append({
            "epoch": epoch,
            "parameter_group": group,
            "n": len(values),
            "median_cosine": median(cosines),
            "negative_cosine_rate": negative_rate,
            "median_norm_ratio": median_ratio,
            "conflict_flag": negative_rate > conflict_rate_threshold,
            "norm_imbalance_flag": median_ratio > norm_ratio_threshold,
        })
    return summary


def aggregate_gradient(rows):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["parameter_group"]].append(row)
    output = []
    for group, values in sorted(by_group.items()):
        output.append({
            "parameter_group": group,
            "n_epoch_group_rows": len(values),
            "median_cosine": median([value["median_cosine"] for value in values]),
            "max_negative_cosine_rate": max(value["negative_cosine_rate"] for value in values),
            "max_median_norm_ratio": max(value["median_norm_ratio"] for value in values),
            "any_conflict_flag": any(value["conflict_flag"] for value in values),
            "any_norm_imbalance_flag": any(value["norm_imbalance_flag"] for value in values),
        })
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_number(value):
    return "NA" if value is None or not math.isfinite(value) else f"{value:.4f}"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    a_metrics = read_csv(args.a_dir / "visa_val_metrics.csv")
    b_metrics = read_csv(args.b_dir / "visa_val_metrics.csv")
    a_epoch, b_epoch = selected_epoch(args.a_dir), selected_epoch(args.b_dir)
    a_macro, b_macro = macro_at_epoch(a_metrics, a_epoch), macro_at_epoch(b_metrics, b_epoch)
    macro_delta = metric_delta(a_macro, b_macro)

    a_categories, b_categories = category_at_epoch(a_metrics, a_epoch), category_at_epoch(b_metrics, b_epoch)
    if set(a_categories) != set(b_categories):
        raise ValueError("A-prime and B category sets differ")
    category_rows = []
    for category in sorted(a_categories):
        row = {"class_name": category, "n": int(a_categories[category]["n"])}
        for metric, value in metric_delta(a_categories[category], b_categories[category]).items():
            row[f"b_minus_a_{metric}"] = value
        category_rows.append(row)

    a_gradient = gradient_summary(
        read_csv(args.a_dir / "gradient_diagnostics.csv"),
        args.conflict_rate_threshold, args.norm_ratio_threshold,
    )
    b_gradient = gradient_summary(
        read_csv(args.b_dir / "gradient_diagnostics.csv"),
        args.conflict_rate_threshold, args.norm_ratio_threshold,
    )
    a_gradient_aggregate, b_gradient_aggregate = aggregate_gradient(a_gradient), aggregate_gradient(b_gradient)

    conflict = any(row["any_conflict_flag"] for row in a_gradient_aggregate + b_gradient_aggregate)
    imbalance = any(row["any_norm_imbalance_flag"] for row in a_gradient_aggregate + b_gradient_aggregate)
    branch = "C_D_E" if not conflict and not imbalance else "DIAGNOSIS_REQUIRED"

    report = {
        "inputs": {"a_dir": str(args.a_dir), "b_dir": str(args.b_dir)},
        "selected_epochs": {"A_prime": a_epoch, "B": b_epoch},
        "macro_metrics": {
            "A_prime": {metric: float(a_macro[metric]) for metric in METRICS},
            "B": {metric: float(b_macro[metric]) for metric in METRICS},
            "B_minus_A_prime": macro_delta,
        },
        "decision_thresholds": {
            "negative_cosine_rate_gt": args.conflict_rate_threshold,
            "median_cls_seg_norm_ratio_gt": args.norm_ratio_threshold,
            "focus_epochs": list(FOCUS_EPOCHS),
        },
        "gate": {
            "shared_path_conflict_flag": conflict,
            "norm_imbalance_flag": imbalance,
            "recommended_branch": branch,
            "note": "Thresholds are configurable screening criteria. Pre-register them before using a branch choice as confirmatory.",
        },
        "gradient_by_epoch_group": {"A_prime": a_gradient, "B": b_gradient},
        "gradient_by_group": {"A_prime": a_gradient_aggregate, "B": b_gradient_aggregate},
    }
    with (args.output_dir / "diagnostic_summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    write_csv(args.output_dir / "per_category_b_minus_a.csv", category_rows)
    write_csv(args.output_dir / "gradient_a_prime.csv", a_gradient)
    write_csv(args.output_dir / "gradient_b.csv", b_gradient)

    markdown = [
        "# Phase2C BF16 A-prime/B diagnostic",
        "",
        f"Selected epochs: A-prime e{a_epoch}; B e{b_epoch}.",
        "",
        "## Macro delta (B - A-prime)",
        "",
        "| Pixel AUC | Pixel AP | Image AUC | Image AP |",
        "|---:|---:|---:|---:|",
        "| " + " | ".join(format_number(macro_delta[metric]) for metric in METRICS) + " |",
        "",
        "## Decision gate",
        "",
        f"- Shared-path conflict flag: `{conflict}` (negative cosine rate > {args.conflict_rate_threshold:.2f}).",
        f"- Norm-imbalance flag: `{imbalance}` (median CLS/SEG norm ratio > {args.norm_ratio_threshold:.1f}).",
        f"- Recommended branch: `{branch}`.",
        "",
        "Threshold-based flags are screening signals. Pre-register thresholds before treating the branch choice as confirmatory.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(markdown) + "\n")


if __name__ == "__main__":
    main()
