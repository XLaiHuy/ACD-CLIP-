#!/usr/bin/env python3
"""Select the Medical epoch for the fresh rental protocol."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DATASETS = (
    "Brain",
    "Liver",
    "Retina",
    "Colon_clinicDB",
    "Colon_colonDB",
    "Colon_Kvasir",
)
EPOCH_RE = re.compile(r"load model from epoch\s+(\d+)")
AVERAGE_RE = re.compile(
    r"\bAverage\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)"
)


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    current_epoch = None
    values: dict[int, dict[str, float]] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    if "evaluator_mode': 'benchmark_exact'" not in text or "'pixel_stride': 1" not in text:
        raise SystemExit(f"selection log is not benchmark_exact pixel_stride=1: {path}")
    for line in text.splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
        average_match = AVERAGE_RE.search(line)
        if average_match and current_epoch is not None:
            pixel_auroc, pixel_ap, image_auroc, image_ap = (
                float(value) for value in average_match.groups()
            )
            values[current_epoch] = {
                "pixel_auroc": pixel_auroc,
                "pixel_ap": pixel_ap,
                "image_auroc": image_auroc,
                "image_ap": image_ap,
            }
    return values


def select_epoch(results_root: Path) -> tuple[list[dict], dict]:
    per_dataset = {}
    for dataset in DATASETS:
        log_path = results_root / dataset / "test.log"
        if not log_path.is_file():
            raise SystemExit(f"missing Medical log: {log_path}")
        per_dataset[dataset] = parse_log(log_path)

    rows = []
    for epoch in range(1, 21):
        missing = [dataset for dataset in DATASETS if epoch not in per_dataset[dataset]]
        if missing:
            raise SystemExit(f"missing epoch {epoch} results for: {','.join(missing)}")
        rows.append(
            {
                "epoch": epoch,
                "medical_macro_pixel_auroc": sum(
                    per_dataset[dataset][epoch]["pixel_auroc"] for dataset in DATASETS
                ) / len(DATASETS),
                "medical_macro_pixel_ap": sum(
                    per_dataset[dataset][epoch]["pixel_ap"] for dataset in DATASETS
                ) / len(DATASETS),
                "per_dataset": {
                    dataset: per_dataset[dataset][epoch] for dataset in DATASETS
                },
            }
        )

    winner = min(
        rows,
        key=lambda row: (
            -row["medical_macro_pixel_ap"],
            -row["medical_macro_pixel_auroc"],
            row["epoch"],
        ),
    )
    return rows, winner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows, winner = select_epoch(args.results_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "medical_macro_pixel_ap", "medical_macro_pixel_auroc"])
        for row in rows:
            writer.writerow([
                row["epoch"],
                row["medical_macro_pixel_ap"],
                row["medical_macro_pixel_auroc"],
            ])
    args.output_json.write_text(
        json.dumps(
            {
                "selection_label": "TARGET-SELECTED EXPLORATORY TRANSFER EXPERIMENT",
                "rule": "highest Medical macro Pixel AP; tie by higher Medical macro Pixel AUROC; then earlier epoch",
                "tta": "disabled",
                "evaluator_mode": "benchmark_exact",
                "pixel_stride": 1,
                "datasets": list(DATASETS),
                "winner": winner,
                "all_epochs": rows,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"winner_epoch={winner['epoch']}")
    print(f"winner_medical_macro_pixel_ap={winner['medical_macro_pixel_ap']:.10f}")
    print(f"winner_medical_macro_pixel_auroc={winner['medical_macro_pixel_auroc']:.10f}")


if __name__ == "__main__":
    main()
