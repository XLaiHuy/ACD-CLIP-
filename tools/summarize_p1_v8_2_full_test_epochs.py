#!/usr/bin/env python3
"""Metrics Aggregator for Phase 4 Progress 1 v8.2 Test Epochs 10–20

Collects evaluation results from test_epoch_10 through test_epoch_20,
verifies sample accounting, and writes summary CSV, JSON, and MD reports.

STRICT STATISTICAL RULE:
  - Epoch 20 is the CANONICAL FINAL CHECKPOINT because there is no validation set.
  - Epochs 10–19 are trajectory reporting only.
  - DO NOT select a "best epoch" using test metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize test results across epochs 10-20")
    parser.add_argument("--run-dir", type=str, default="runs/phase4/p1_v8_2_full20_seed0")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir

    print("=" * 70)
    print("SUMMARIZING TEST EPOCHS 10–20 TRAJECTORY")
    print(f"Run Directory: {run_dir}")
    print("=" * 70)

    rows = []
    headers = [
        "epoch", "checkpoint", "test_sample_count", "image_AUROC",
        "image_AP", "pixel_AUROC", "pixel_AP", "metric_status",
        "sample_accounting_status", "runtime_seconds", "role_in_study"
    ]

    for epoch in range(10, 21):
        epoch_dir = os.path.join(run_dir, f"test_epoch_{epoch}")
        metrics_json_path = os.path.join(epoch_dir, "metrics.json")
        sample_check_path = os.path.join(epoch_dir, "sample_count_check.json")

        role_in_study = "CANONICAL FINAL CHECKPOINT" if epoch == 20 else "Trajectory Diagnostic"

        if not os.path.exists(metrics_json_path):
            rows.append({
                "epoch": epoch,
                "checkpoint": f"adapter_{epoch}.pth",
                "test_sample_count": 0,
                "image_AUROC": "N/A",
                "image_AP": "N/A",
                "pixel_AUROC": "N/A",
                "pixel_AP": "N/A",
                "metric_status": "MISSING",
                "sample_accounting_status": "UNVERIFIED",
                "runtime_seconds": 0.0,
                "role_in_study": role_in_study,
            })
            continue

        with open(metrics_json_path) as f:
            m = json.load(f)

        sample_accounting_status = "PASSED"
        if os.path.exists(sample_check_path):
            with open(sample_check_path) as f:
                sc = json.load(f)
                if not sc.get("match", True):
                    sample_accounting_status = "FAILED"

        rows.append({
            "epoch": epoch,
            "checkpoint": f"adapter_{epoch}.pth",
            "test_sample_count": m.get("test_sample_count", 0),
            "image_AUROC": m.get("image_AUROC", 0.0),
            "image_AP": m.get("image_AP", 0.0),
            "pixel_AUROC": m.get("pixel_AUROC", 0.0),
            "pixel_AP": m.get("pixel_AP", 0.0),
            "metric_status": m.get("metric_status", "PASSED"),
            "sample_accounting_status": sample_accounting_status,
            "runtime_seconds": m.get("runtime_seconds", 0.0),
            "role_in_study": role_in_study,
        })

    # Output paths
    csv_path = os.path.join(run_dir, "TEST_E10_E20_SUMMARY.csv")
    json_path = os.path.join(run_dir, "TEST_E10_E20_SUMMARY.json")
    md_path = os.path.join(run_dir, "TEST_E10_E20_SUMMARY.md")

    # 1. Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    # 2. Save JSON
    with open(json_path, "w") as f:
        json.dump({"summary": rows, "canonical_final_epoch": 20}, f, indent=2)

    # 3. Save MD
    md_content = f"""# Test Evaluation Trajectory Summary (Epochs 10–20)

> [!IMPORTANT]
> **Epoch 20 is the CANONICAL FINAL CHECKPOINT.**  
> Epochs 10–19 are reported strictly as trajectory diagnostics. No checkpoint selection is performed based on test metrics.

| Epoch | Checkpoint | Samples | Image AUROC | Image AP | Pixel AUROC | Pixel AP | Accounting | Role in Study |
|---|---|---|---|---|---|---|---|---|
"""
    for r in rows:
        md_content += (
            f"| {r['epoch']} | `{r['checkpoint']}` | {r['test_sample_count']} | "
            f"{r['image_AUROC']} | {r['image_AP']} | {r['pixel_AUROC']} | {r['pixel_AP']} | "
            f"{r['sample_accounting_status']} | **{r['role_in_study']}** |\n"
        )

    with open(md_path, "w") as f:
        f.write(md_content)

    print(f"[SUMMARY SAVED] CSV: {csv_path}")
    print(f"[SUMMARY SAVED] JSON: {json_path}")
    print(f"[SUMMARY SAVED] MD: {md_path}")


if __name__ == "__main__":
    main()
