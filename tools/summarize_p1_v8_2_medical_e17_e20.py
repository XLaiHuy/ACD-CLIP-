#!/usr/bin/env python3
"""Summarize P1-v8.2 medical evaluation results for epochs 17 and 20 across 6 medical datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.support_aware_aggregation import harmonic_mean_safe, DATASETS

VALID_IMAGE_DATASETS = ["Brain", "Liver", "Retina"]
ALL_DATASETS = ["Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"]


def compute_epoch_summary(output_dir: Path, epoch: int) -> dict[str, Any]:
    epoch_dir = output_dir / f"epoch_{epoch}"
    dataset_results: dict[str, dict[str, float]] = {}

    run_dir = Path("runs/phase4/p1_v8_2_full20_seed0")
    for ds in ALL_DATASETS:
        ds_json = epoch_dir / ds / "metrics.json"
        ds_csv = epoch_dir / f"exact_results_{ds}_test_epoch_{epoch}.csv"
        run_csv = run_dir / f"exact_results_{ds}_test_epoch_{epoch}.csv"

        if ds_json.is_file():
            with ds_json.open("r") as f:
                dataset_results[ds] = json.load(f)
        elif ds_csv.is_file():
            with ds_csv.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dataset_results[ds] = {
                        "pixel AUC": float(row["pixel AUC"]),
                        "pixel AP": float(row["pixel AP"]),
                        "image AUC": float(row["image AUC"]),
                        "image AP": float(row["image AP"]),
                    }
        elif run_csv.is_file():
            with run_csv.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dataset_results[ds] = {
                        "pixel AUC": float(row["pixel AUC"]),
                        "pixel AP": float(row["pixel AP"]),
                        "image AUC": float(row["image AUC"]),
                        "image AP": float(row["image AP"]),
                    }
        else:
            raise FileNotFoundError(f"Missing metric file for epoch {epoch}, dataset {ds}: tried {ds_json}, {ds_csv}, {run_csv}")

    pix_aucs = [dataset_results[ds]["pixel AUC"] for ds in ALL_DATASETS]
    pix_aps = [dataset_results[ds]["pixel AP"] for ds in ALL_DATASETS]
    
    img_aucs = [dataset_results[ds]["image AUC"] for ds in VALID_IMAGE_DATASETS]
    img_aps = [dataset_results[ds]["image AP"] for ds in VALID_IMAGE_DATASETS]

    pix_auc_macro = sum(pix_aucs) / len(pix_aucs)
    pix_ap_macro = sum(pix_aps) / len(pix_aps)

    img_auc_macro = sum(img_aucs) / len(img_aucs)
    img_ap_macro = sum(img_aps) / len(img_aps)

    image_score = (img_auc_macro + img_ap_macro) / 2.0
    pixel_score = (pix_auc_macro + pix_ap_macro) / 2.0
    combined_score = harmonic_mean_safe(image_score, pixel_score)

    return {
        "epoch": epoch,
        "pixel_auroc_macro": round(pix_auc_macro, 4),
        "pixel_ap_macro": round(pix_ap_macro, 4),
        "image_auroc_macro": round(img_auc_macro, 4),
        "image_ap_macro": round(img_ap_macro, 4),
        "support_aware_combined_score": round(combined_score, 4),
        "dataset_results": dataset_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize P1-v8.2 medical evaluation epochs 17 and 20")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/phase4/p1_v8_2_medical_e17_e20"))
    args = parser.parse_args()

    output_dir = args.output_dir
    epochs = [17, 20]
    summaries = []

    for ep in epochs:
        summary = compute_epoch_summary(output_dir, ep)
        summaries.append(summary)

    csv_path = output_dir / "medical_e17_e20_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "pixel_auroc_macro", "pixel_ap_macro", "image_auroc_macro", "image_ap_macro",
            "support_aware_combined_score", "status"
        ])
        for s in summaries:
            status = "CANONICAL_FINAL" if s["epoch"] == 20 else "DIAGNOSTIC_TRAJECTORY"
            writer.writerow([
                s["epoch"], s["pixel_auroc_macro"], s["pixel_ap_macro"],
                s["image_auroc_macro"], s["image_ap_macro"],
                s["support_aware_combined_score"], status
            ])

    json_path = output_dir / "medical_e17_e20_summary.json"
    with json_path.open("w") as f:
        json.dump(summaries, f, indent=2)

    report_path = output_dir / "FINAL_MEDICAL_E17_E20_REPORT.md"
    with report_path.open("w") as f:
        f.write("# P1-v8.2 Six-Medical Evaluation Final Report (Epochs 17 & 20)\n\n")
        f.write("**Repository**: `/home/ai4/caohuy/ACD-CLIP-phase4`  \n")
        f.write("**Branch**: `phase4-progress1-cops-dynamic-prompt`  \n")
        f.write("**Commit HEAD**: `96c5b9c6ad8ec2b3b2eaec11a5b0deab58d41b2c`  \n")
        f.write("**Evaluation Split**: Official 6 Medical Test Split (`split=test`)  \n")
        f.write("**Protocol Status**: `MEDICAL_E17_E20_COMPLETED`  \n\n")

        f.write("> [!IMPORTANT]\n")
        f.write("> **Epoch 20** is the canonical final checkpoint.\n")
        f.write("> **Epoch 17** is POST-HOC DIAGNOSTIC TRAJECTORY reporting only (NOT model selection).\n\n")

        f.write("## 1. Two-Epoch Medical Summary Table\n\n")
        f.write("| Epoch | Pixel AUROC Macro | Pixel AP Macro | Image AUROC Macro | Image AP Macro | Support-Aware Combined Score | Status |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for s in summaries:
            status_tag = "**CANONICAL FINAL**" if s["epoch"] == 20 else "DIAGNOSTIC TRAJECTORY"
            f.write(f"| {s['epoch']} | {s['pixel_auroc_macro']:.2f}% | {s['pixel_ap_macro']:.2f}% | {s['image_auroc_macro']:.2f}% | {s['image_ap_macro']:.2f}% | {s['support_aware_combined_score']:.2f}% | {status_tag} |\n")

        f.write("\n## 2. Per-Dataset Breakdown (Epochs 17 & 20)\n\n")
        for s in summaries:
            f.write(f"### Epoch {s['epoch']} ({'Canonical Final' if s['epoch'] == 20 else 'Diagnostic'})\n\n")
            f.write("| Dataset | Pixel AUROC | Pixel AP | Image AUROC | Image AP | Image Metric Support |\n")
            f.write("|---|---|---|---|---|---|\n")
            for ds in ALL_DATASETS:
                res = s["dataset_results"][ds]
                supp = "Supported (Valid)" if ds in VALID_IMAGE_DATASETS else "Unsupported (Excluded)"
                img_auc_str = f"{res['image AUC']:.2f}%" if ds in VALID_IMAGE_DATASETS else "N/A"
                img_ap_str = f"{res['image AP']:.2f}%" if ds in VALID_IMAGE_DATASETS else "N/A"
                f.write(f"| {ds} | {res['pixel AUC']:.2f}% | {res['pixel AP']:.2f}% | {img_auc_str} | {img_ap_str} | {supp} |\n")
            f.write("\n")

    print(f"[OK] Summary generated: {csv_path}, {json_path}, {report_path}")

if __name__ == "__main__":
    main()
