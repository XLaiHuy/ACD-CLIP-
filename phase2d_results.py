"""Artifact writing and preregistered selection for Phase2D results."""
import argparse
import csv
import json
from pathlib import Path


METRICS = ("pixel_auc", "pixel_ap", "image_auc", "image_ap")
PARENTS = {
    "A_prime": {"pixel_auc": 94.8038, "pixel_ap": 55.5341, "image_auc": 97.9028, "image_ap": 98.4225},
    "B": {"pixel_auc": 96.2236, "pixel_ap": 55.1342, "image_auc": 97.8750, "image_ap": 98.4287},
}


def macro_rows(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["scope"] == "macro"]


def write_config(run_dir, args):
    config = {
        "experiment": "Phase2D_AB_interpolation",
        "dataset": "VisA",
        "seed": 42,
        "train_manifest": "splits/visa_train_seed42.csv",
        "val_manifest": "splits/visa_val_seed42.csv",
        "split_metadata": "splits/visa_split_seed42_metadata.json",
        "score_rule": "cls_only",
        "training_performed": False,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "parents": {"A_prime": {"path": args.a_path, "sha256": args.a_sha}, "B": {"path": args.b_path, "sha256": args.b_sha}},
        "locked_candidates": {"AB25": 0.25, "AB50": 0.50, "AB75": 0.75},
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parent_gate(run_dir):
    rows = {row["checkpoint_name"]: row for row in macro_rows(run_dir / "parent_reproduction.csv")}
    if set(rows) != set(PARENTS):
        raise ValueError("parent reproduction macro rows are incomplete")
    result = {"tolerance_percentage_points": 0.05, "parents": {}, "passed": True}
    for name, reference in PARENTS.items():
        measured = {metric: float(rows[name][metric]) for metric in METRICS}
        deltas = {metric: measured[metric] - reference[metric] for metric in METRICS}
        passed = all(abs(delta) <= 0.05 for delta in deltas.values())
        result["parents"][name] = {"registered": reference, "reproduced": measured, "deltas": deltas, "passed": passed}
        result["passed"] = result["passed"] and passed
    (run_dir / "parent_reproduction.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError("parent reproduction gate failed")


def select(run_dir, results_markdown):
    rows = {row["checkpoint_name"]: row for row in macro_rows(run_dir / "visa_val_metrics.csv")}
    candidates = []
    for name in ("AB25", "AB50", "AB75"):
        row = rows[name]
        values = {metric: float(row[metric]) for metric in METRICS}
        primary = values["image_ap"] >= 97.4225 and values["pixel_ap"] > 55.5341
        secondary = values["pixel_auc"] > 94.8038 and values["pixel_ap"] >= 55.0341 and values["image_ap"] >= 97.4225
        candidates.append({"candidate": name, "lambda_b": float(row["lambda_b"]), "metrics": values, "delta_vs_a_prime": {metric: values[metric] - PARENTS["A_prime"][metric] for metric in METRICS}, "delta_vs_b": {metric: values[metric] - PARENTS["B"][metric] for metric in METRICS}, "primary_eligible": primary, "secondary_pareto": secondary, "checkpoint_path": row["checkpoint_path"]})
    winners = [candidate for candidate in candidates if candidate["primary_eligible"]]
    winner = sorted(winners, key=lambda item: (-item["metrics"]["pixel_ap"], -item["metrics"]["image_ap"], -item["metrics"]["pixel_auc"], item["lambda_b"]))[0] if winners else None
    selection = {"primary_winner": winner, "primary_success": winner is not None, "secondary_pareto_candidates": [candidate["candidate"] for candidate in candidates if candidate["secondary_pareto"]], "candidates": candidates, "decision": "primary_winner" if winner else "keep_A_prime_primary_winner"}
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Phase2D A-prime/B Interpolation Results", "", "## Parent reproduction", "", "The parent reproduction gate passed within 0.05 percentage points for every registered macro metric.", "", "## Candidate macro metrics", "", "| Candidate | Pixel AUC | Pixel AP | Image AUC | Image AP |", "| --- | ---: | ---: | ---: | ---: |"]
    for candidate in candidates:
        metric = candidate["metrics"]
        lines.append(f"| {candidate['candidate']} | {metric['pixel_auc']:.4f} | {metric['pixel_ap']:.4f} | {metric['image_auc']:.4f} | {metric['image_ap']:.4f} |")
    lines.extend(["", "## Decision", ""])
    if winner:
        lines.append(f"{winner['candidate']} is the preregistered primary winner. Multi-seed confirmation is the next step; LB_0p1 is not run.")
    elif selection["secondary_pareto_candidates"]:
        lines.append("A-prime remains the primary winner. The secondary Pareto candidates are exploratory; the locked three-point interpolation test is closed. A later LB_0p1 preregistration may be considered.")
    else:
        lines.append("A-prime remains the primary winner. No locked candidate met the primary or secondary criterion, so interpolation is closed. A later LB_0p1 preregistration may be considered.")
    lines.extend(["", "This is a single seed-42 comparison and does not establish statistical robustness.", "", "Per-category metrics are retained in `runs/phase2d_ab_interpolation_seed42/visa_val_metrics.csv`.", ""])
    Path(results_markdown).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--parent-gate", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--results-markdown", default="PHASE2D_AB_RESULTS.md")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--a-path")
    parser.add_argument("--a-sha")
    parser.add_argument("--b-path")
    parser.add_argument("--b-sha")
    return parser.parse_args()


def main():
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.write_config:
        write_config(args.run_dir, args)
    if args.parent_gate:
        parent_gate(args.run_dir)
    if args.select:
        select(args.run_dir, args.results_markdown)


if __name__ == "__main__":
    main()
