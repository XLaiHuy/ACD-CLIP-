#!/usr/bin/env python3
"""Summarize exact six-medical Phase4 CSV outputs without inventing metrics."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DATASETS = ["Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"]
PATTERN = re.compile(r"^exact_results_(.+)_(val|test)_epoch_(\d+)\.csv$")


def harmonic_mean(left: float, right: float) -> float:
    return 0.0 if left + right == 0 else 2.0 * left * right / (left + right)


def read_exact_rows(save_path: Path, split: str, selected_epochs: set[int] | None) -> pd.DataFrame:
    rows = []
    for path in sorted(save_path.glob(f"exact_results_*_{split}_epoch_*.csv")):
        match = PATTERN.match(path.name)
        if match is None:
            continue
        dataset, result_split, epoch_text = match.groups()
        epoch = int(epoch_text)
        if result_split != split or dataset not in DATASETS or (selected_epochs is not None and epoch not in selected_epochs):
            continue
        frame = pd.read_csv(path)
        average = frame.loc[frame["class name"] == "Average"]
        if len(average) != 1:
            raise ValueError(f"{path} must contain exactly one Average row")
        row = average.iloc[0]
        image_score = (float(row["image AUC"]) + float(row["image AP"])) / 2.0
        pixel_score = (float(row["pixel AUC"]) + float(row["pixel AP"])) / 2.0
        rows.append({
            "epoch": epoch,
            "dataset": dataset,
            "image_AUROC": float(row["image AUC"]),
            "image_AP": float(row["image AP"]),
            "pixel_AUROC": float(row["pixel AUC"]),
            "pixel_AP": float(row["pixel AP"]),
            "image_score": image_score,
            "pixel_score": pixel_score,
            "combined_score": harmonic_mean(image_score, pixel_score),
            "source": path.name,
        })
    if not rows:
        raise FileNotFoundError(f"no exact_results_<dataset>_{split}_epoch_<epoch>.csv files found")
    return pd.DataFrame(rows).sort_values(["epoch", "dataset"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Summarize six-dataset exact Phase4 medical results")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--epochs", type=int, nargs="+", default=None)
    args = parser.parse_args()
    save_path = Path(args.save_path)
    rows = read_exact_rows(save_path, args.split, None if args.epochs is None else set(args.epochs))
    rows.to_csv(save_path / f"medical_{args.split}_results_by_dataset.csv", index=False)
    by_epoch = rows.groupby("epoch", as_index=False)[
        ["image_AUROC", "image_AP", "pixel_AUROC", "pixel_AP", "image_score", "pixel_score", "combined_score"]
    ].mean()
    counts = rows.groupby("epoch")["dataset"].nunique()
    incomplete = {int(epoch): int(count) for epoch, count in counts.items() if count != len(DATASETS)}
    if incomplete:
        raise ValueError(f"exact result set is incomplete; expected all six datasets: {incomplete}")
    by_epoch.to_csv(save_path / f"medical_{args.split}_results_by_epoch.csv", index=False)
    summary = {
        "split": args.split,
        "datasets": DATASETS,
        "epochs": [int(value) for value in by_epoch["epoch"].tolist()],
        "macro_by_epoch": by_epoch.to_dict(orient="records"),
    }
    if args.split == "val":
        best = by_epoch.sort_values(["combined_score", "epoch"], ascending=[False, True]).iloc[0].to_dict()
        summary["selection_rule"] = "maximize the six-dataset validation macro average of per-dataset harmonic_mean(image_score,pixel_score)"
        summary["best_epoch"] = best
        (save_path / "medical_validation_selection.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(best, indent=2, sort_keys=True))
    else:
        summary["selection_rule"] = "none; test metrics are reported for the validation-selected epoch only"
        (save_path / "medical_test_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
