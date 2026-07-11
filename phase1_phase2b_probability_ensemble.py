import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torchmetrics.functional import auroc, average_precision


IMAGE_DATASETS = ["Brain", "Liver", "Retina"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline probability ensemble for Phase1 and Phase2B cls-only image scores."
    )
    parser.add_argument("--phase1_raw", required=True, help="Path to Phase1 raw predictions CSV")
    parser.add_argument("--phase2b_raw", required=True, help="Path to Phase2B raw predictions CSV")
    parser.add_argument("--phase1_summary", default=None, help="Path to Phase1 summary CSV (fixed_config_epoch_sweep.csv)")
    parser.add_argument("--phase2b_summary", default=None, help="Path to Phase2B summary CSV (fixed_config_epoch_sweep.csv)")
    parser.add_argument("--output_dir", required=True, help="Directory to save the ensemble results")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--image_datasets", nargs="+", default=IMAGE_DATASETS)
    return parser.parse_args()


def read_raw(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=1):
            dataset = row["dataset"]
            file_name = row["file_name"]
            key = (dataset, file_name)
            if key in rows:
                raise ValueError(
                    f"Duplicate key {key} in {path} at line {row_idx}. "
                    "Ensure the CSV only contains predictions for a single epoch and single prompt config."
                )
            rows[key] = {
                "dataset": dataset,
                "file_name": file_name,
                "label": int(row["label"]),
                "cls_score": float(row["cls_score"]),
            }
    return rows


def read_summary_metrics(summary_path):
    with open(summary_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
        return (
            float(row["image_auc_3"]),
            float(row["image_ap_3"]),
            float(row["pixel_auc_6"]),
            float(row["pixel_ap_6"]),
        )


def metric_or_none(scores, labels):
    label_tensor = torch.tensor(labels, dtype=torch.int32)
    if label_tensor.max() == label_tensor.min():
        return None, None
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    return (
        round(auroc(score_tensor, label_tensor, task="binary").item(), 4) * 100,
        round(average_precision(score_tensor, label_tensor, task="binary").item(), 4) * 100,
    )


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    print(f"Reading Phase1 raw predictions from {args.phase1_raw}")
    phase1 = read_raw(args.phase1_raw)
    print(f"Reading Phase2B raw predictions from {args.phase2b_raw}")
    phase2b = read_raw(args.phase2b_raw)

    phase1_keys = set(phase1.keys())
    phase2b_keys = set(phase2b.keys())

    # Strict alignment assertions
    if phase1_keys != phase2b_keys:
        missing_in_p2b = phase1_keys - phase2b_keys
        missing_in_p1 = phase2b_keys - phase1_keys
        error_msg = (
            f"Mismatched set of samples.\n"
            f"Phase1 contains {len(phase1_keys)} samples.\n"
            f"Phase2B contains {len(phase2b_keys)} samples.\n"
        )
        if missing_in_p2b:
            error_msg += f"Keys in Phase1 but missing in Phase2B (showing up to 5): {list(missing_in_p2b)[:5]}\n"
        if missing_in_p1:
            error_msg += f"Keys in Phase2B but missing in Phase1 (showing up to 5): {list(missing_in_p1)[:5]}\n"
        raise ValueError(error_msg)

    print(f"Successfully joined {len(phase1)} samples between Phase1 and Phase2B.")

    # Validate all labels match exactly
    for key in phase1:
        if phase1[key]["label"] != phase2b[key]["label"]:
            raise ValueError(
                f"Mismatched label for sample {key}: "
                f"Phase1 label = {phase1[key]['label']}, Phase2B label = {phase2b[key]['label']}"
            )

    rows_by_dataset = defaultdict(list)
    for key in sorted(phase1.keys()):
        row1 = phase1[key]
        row2 = phase2b[key]
        if row1["dataset"] in args.image_datasets:
            rows_by_dataset[row1["dataset"]].append((row1, row2))

    detail_rows = []
    aggregate_rows = []

    for beta in sorted(args.betas):
        image_auc_vals = []
        image_ap_vals = []
        beta_str = f"{beta:.2f}"

        for dataset_name in args.image_datasets:
            pairs = rows_by_dataset.get(dataset_name, [])
            if not pairs:
                continue
            scores = [
                beta * row1["cls_score"] + (1.0 - beta) * row2["cls_score"]
                for row1, row2 in pairs
            ]
            labels = [row1["label"] for row1, _ in pairs]
            image_auc, image_ap = metric_or_none(scores, labels)
            
            detail_rows.append({
                "beta": beta_str,
                "dataset": dataset_name,
                "image_auc": "" if image_auc is None else f"{image_auc:.2f}",
                "image_ap": "" if image_ap is None else f"{image_ap:.2f}",
                "n": len(pairs),
            })
            if image_auc is not None and image_ap is not None:
                image_auc_vals.append(image_auc)
                image_ap_vals.append(image_ap)

        aggregate_rows.append({
            "beta": beta_str,
            "image_auc_3": "" if not image_auc_vals else f"{sum(image_auc_vals) / len(image_auc_vals):.2f}",
            "image_ap_3": "" if not image_ap_vals else f"{sum(image_ap_vals) / len(image_ap_vals):.2f}",
            "image_n": len(image_ap_vals),
        })

    output_dir = Path(args.output_dir)
    write_csv(
        output_dir / "probability_ensemble_by_dataset.csv",
        detail_rows,
        ["beta", "dataset", "image_auc", "image_ap", "n"],
    )
    write_csv(
        output_dir / "probability_ensemble_summary.csv",
        aggregate_rows,
        ["beta", "image_auc_3", "image_ap_3", "image_n"],
    )
    print(f"\nEnsemble results saved to {output_dir}")
    
    # Run Sanity Checks / assertions against original summaries if provided
    if args.phase1_summary:
        p1_auc, p1_ap, _, _ = read_summary_metrics(args.phase1_summary)
        beta1_row = next(r for r in aggregate_rows if r["beta"] == "1.00")
        assert abs(float(beta1_row["image_auc_3"]) - p1_auc) < 1e-2, f"Ensemble beta=1.0 AUC ({beta1_row['image_auc_3']}) != Phase1 rescore AUC ({p1_auc})"
        assert abs(float(beta1_row["image_ap_3"]) - p1_ap) < 1e-2, f"Ensemble beta=1.0 AP ({beta1_row['image_ap_3']}) != Phase1 rescore AP ({p1_ap})"
        print(f"Sanity check passed: beta=1.0 matches Phase1 rescore AUC/AP ({p1_auc:.2f}/{p1_ap:.2f})")

    if args.phase2b_summary:
        p2b_auc, p2b_ap, p2b_p_auc, p2b_p_ap = read_summary_metrics(args.phase2b_summary)
        beta0_row = next(r for r in aggregate_rows if r["beta"] == "0.00")
        assert abs(float(beta0_row["image_auc_3"]) - p2b_auc) < 1e-2, f"Ensemble beta=0.0 AUC ({beta0_row['image_auc_3']}) != Phase2B rescore AUC ({p2b_auc})"
        assert abs(float(beta0_row["image_ap_3"]) - p2b_ap) < 1e-2, f"Ensemble beta=0.0 AP ({beta0_row['image_ap_3']}) != Phase2B rescore AP ({p2b_ap})"
        print(f"Sanity check passed: beta=0.0 matches Phase2B rescore AUC/AP ({p2b_auc:.2f}/{p2b_ap:.2f})")
        
        # Verify Phase 2B reproduces historical metrics (90.98 / 40.35 / 73.77 / 74.24) within 0.05% tolerance
        assert abs(p2b_p_ap - 40.35) <= 0.05, f"Phase2B Pixel AP ({p2b_p_ap}) deviates from baseline 40.35"
        assert abs(p2b_p_auc - 90.98) <= 0.05, f"Phase2B Pixel AUC ({p2b_p_auc}) deviates from baseline 90.98"
        assert abs(p2b_ap - 74.24) <= 0.05, f"Phase2B Image AP ({p2b_ap}) deviates from baseline 74.24"
        assert abs(p2b_auc - 73.77) <= 0.05, f"Phase2B Image AUC ({p2b_auc}) deviates from baseline 73.77"
        print("Sanity check passed: Phase2B metrics successfully reproduced reference checkpoints.")

    print("\nProbability ensemble summary:")
    for row in aggregate_rows:
        print(f"  beta={row['beta']} | image_auc_3={row['image_auc_3']} | image_ap_3={row['image_ap_3']}")


if __name__ == "__main__":
    main()
