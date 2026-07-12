#!/usr/bin/env python3
"""Compare Phase2C A-prime with delayed-activation condition C.

The comparison uses activation-relative epochs: C is delayed by two epochs, so
A-prime epoch N is aligned with C epoch N + 2.  It is intentionally an
exploratory diagnostic: its thresholds are not a confirmation test.
"""

import argparse
import json
from pathlib import Path

from phase2c_analyze_ab import (
    METRICS,
    aggregate_gradient,
    category_at_epoch,
    format_number,
    gradient_summary,
    macro_at_epoch,
    metric_delta,
    read_csv,
    selected_epoch,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Phase2C A-prime/C delayed-activation diagnostics")
    parser.add_argument("--a-dir", type=Path, required=True)
    parser.add_argument("--c-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activation-delay-epochs", type=int, default=2)
    parser.add_argument("--conflict-rate-threshold", type=float, default=0.50)
    parser.add_argument("--norm-ratio-threshold", type=float, default=10.0)
    return parser.parse_args()


def gradient_summary_for_epochs(rows, epochs, conflict_rate_threshold, norm_ratio_threshold):
    """Use the common summarizer with a caller-specified diagnostic window."""
    import phase2c_analyze_ab

    original = phase2c_analyze_ab.FOCUS_EPOCHS
    try:
        phase2c_analyze_ab.FOCUS_EPOCHS = frozenset(epochs)
        return gradient_summary(rows, conflict_rate_threshold, norm_ratio_threshold)
    finally:
        phase2c_analyze_ab.FOCUS_EPOCHS = original


def index_by_epoch_group(rows):
    return {(row["epoch"], row["parameter_group"]): row for row in rows}


def activation_window_summary(a_rows, c_rows, activation_delay_epochs):
    """Compare the three epochs beginning with A-prime's first alpha ramp."""
    a_by_key, c_by_key = index_by_epoch_group(a_rows), index_by_epoch_group(c_rows)
    rows = []
    for (a_epoch, group), a_row in sorted(a_by_key.items()):
        if a_epoch not in (4, 5, 6):
            continue
        c_row = c_by_key.get((a_epoch + activation_delay_epochs, group))
        if c_row is None:
            continue
        rows.append({
            "a_epoch": a_epoch,
            "c_epoch": c_row["epoch"],
            "parameter_group": group,
            "a_negative_cosine_rate": a_row["negative_cosine_rate"],
            "c_negative_cosine_rate": c_row["negative_cosine_rate"],
            "a_median_norm_ratio": a_row["median_norm_ratio"],
            "c_median_norm_ratio": c_row["median_norm_ratio"],
            "negative_cosine_rate_delta": c_row["negative_cosine_rate"] - a_row["negative_cosine_rate"],
        })
    return rows


def decision_gate(a_gradient, c_gradient, activation_rows):
    c_aggregate = aggregate_gradient(c_gradient)
    c_conflict = any(row["any_conflict_flag"] for row in c_aggregate)
    c_imbalance = any(row["any_norm_imbalance_flag"] for row in c_aggregate)
    shared_rows = [row for row in activation_rows if row["parameter_group"] == "shared_image_lora"]
    shared_conflict_reduced = bool(shared_rows) and max(
        row["c_negative_cosine_rate"] for row in shared_rows
    ) < max(row["a_negative_cosine_rate"] for row in shared_rows)

    if c_conflict:
        branch = "TARGETED_CONFLICT_INTERVENTION"
    elif c_imbalance:
        branch = "LOSS_BALANCING_REVIEW"
    elif shared_conflict_reduced:
        branch = "D_RESTART_CANDIDATE"
    else:
        branch = "MANUAL_REVIEW"
    return {
        "shared_image_lora_activation_conflict_reduced": shared_conflict_reduced,
        "c_any_conflict_flag": c_conflict,
        "c_any_norm_imbalance_flag": c_imbalance,
        "recommended_branch": branch,
        "note": "Exploratory screening only. Lock D's restart epoch, optimizer-state policy, learning rates, and scheduler before training.",
    }


def main():
    args = parse_args()
    if args.activation_delay_epochs < 0:
        raise ValueError("activation-delay-epochs must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    a_metrics = read_csv(args.a_dir / "visa_val_metrics.csv")
    c_metrics = read_csv(args.c_dir / "visa_val_metrics.csv")
    a_epoch, c_epoch = selected_epoch(args.a_dir), selected_epoch(args.c_dir)
    a_macro, c_macro = macro_at_epoch(a_metrics, a_epoch), macro_at_epoch(c_metrics, c_epoch)
    macro_delta = metric_delta(a_macro, c_macro)

    a_categories, c_categories = category_at_epoch(a_metrics, a_epoch), category_at_epoch(c_metrics, c_epoch)
    if set(a_categories) != set(c_categories):
        raise ValueError("A-prime and C category sets differ")
    category_rows = []
    for category in sorted(a_categories):
        row = {"class_name": category, "n": int(a_categories[category]["n"])}
        for metric, value in metric_delta(a_categories[category], c_categories[category]).items():
            row[f"c_minus_a_{metric}"] = value
        category_rows.append(row)

    a_epochs = range(4, 14)
    c_epochs = range(4 + args.activation_delay_epochs, 14 + args.activation_delay_epochs)
    a_gradient = gradient_summary_for_epochs(read_csv(args.a_dir / "gradient_diagnostics.csv"), a_epochs, args.conflict_rate_threshold, args.norm_ratio_threshold)
    c_gradient = gradient_summary_for_epochs(read_csv(args.c_dir / "gradient_diagnostics.csv"), c_epochs, args.conflict_rate_threshold, args.norm_ratio_threshold)
    activation_rows = activation_window_summary(a_gradient, c_gradient, args.activation_delay_epochs)
    gate = decision_gate(a_gradient, c_gradient, activation_rows)

    report = {
        "inputs": {"a_dir": str(args.a_dir), "c_dir": str(args.c_dir)},
        "selected_epochs": {"A_prime": a_epoch, "C": c_epoch},
        "macro_metrics": {
            "A_prime": {metric: float(a_macro[metric]) for metric in METRICS},
            "C": {metric: float(c_macro[metric]) for metric in METRICS},
            "C_minus_A_prime": macro_delta,
        },
        "alignment": {"activation_delay_epochs": args.activation_delay_epochs, "A_prime_epochs": list(a_epochs), "C_epochs": list(c_epochs)},
        "decision_thresholds": {"negative_cosine_rate_gt": args.conflict_rate_threshold, "median_cls_seg_norm_ratio_gt": args.norm_ratio_threshold},
        "gate": gate,
        "gradient_by_epoch_group": {"A_prime": a_gradient, "C": c_gradient},
        "gradient_by_group": {"A_prime": aggregate_gradient(a_gradient), "C": aggregate_gradient(c_gradient)},
        "activation_window": activation_rows,
    }
    with (args.output_dir / "diagnostic_summary.json").open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    write_csv(args.output_dir / "per_category_c_minus_a.csv", category_rows)
    write_csv(args.output_dir / "gradient_a_prime.csv", a_gradient)
    write_csv(args.output_dir / "gradient_c.csv", c_gradient)
    write_csv(args.output_dir / "activation_window.csv", activation_rows)

    markdown = [
        "# Phase2C BF16 A-prime/C delayed-activation diagnostic",
        "",
        f"Selected epochs: A-prime e{a_epoch}; C e{c_epoch}.",
        "",
        "## Macro delta (C - A-prime)",
        "",
        "| Pixel AUC | Pixel AP | Image AUC | Image AP |",
        "|---:|---:|---:|---:|",
        "| " + " | ".join(format_number(macro_delta[metric]) for metric in METRICS) + " |",
        "",
        "## Activation-relative diagnostic gate",
        "",
        f"- Shared-image-LoRA activation conflict reduced: `{gate['shared_image_lora_activation_conflict_reduced']}`.",
        f"- C has any conflict flag: `{gate['c_any_conflict_flag']}`.",
        f"- C has any norm-imbalance flag: `{gate['c_any_norm_imbalance_flag']}`.",
        f"- Recommended branch: `{gate['recommended_branch']}`.",
        "",
        gate["note"],
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(markdown) + "\n")


if __name__ == "__main__":
    main()
