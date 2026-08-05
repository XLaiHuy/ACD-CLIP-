#!/usr/bin/env python3
"""Re-aggregate P1-v7 validation results with support-aware logic.

Reads existing exact_results_*.csv artifacts, applies support-aware
aggregation, and writes:

  selection/selection_support_aware_v2.json
  selection/selection_support_aware_v2.csv

Does NOT overwrite medical_validation_selection.json.
Does NOT rerun any test or validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

try:
    from .support_aware_aggregation import (
        DATASETS,
        SELECTION_RULE_VERSION,
        _METRIC_IMPL_VERSION,
        build_fingerprint_config,
        compute_config_fingerprint,
        reaggregate_all_epochs,
        select_best_epoch,
    )
except ImportError:
    from support_aware_aggregation import (
        DATASETS,
        SELECTION_RULE_VERSION,
        _METRIC_IMPL_VERSION,
        build_fingerprint_config,
        compute_config_fingerprint,
        reaggregate_all_epochs,
        select_best_epoch,
    )


DEFAULT_SAVE_PATH = "runs/phase4/progress1_v7_full_seed0_ready3/train"
DEFAULT_MANIFEST_ROOT = "runs/phase4/progress1_v7_full_seed0_ready3/medical_manifests"
DEFAULT_EPOCHS = list(range(8, 21))


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-aggregate P1-v7 validation results with support-aware aggregation."
    )
    parser.add_argument("--save_path", default=DEFAULT_SAVE_PATH,
                        help="Path to training directory with exact_results_*.csv files")
    parser.add_argument("--manifest_root", default=DEFAULT_MANIFEST_ROOT,
                        help="Path to directory with dataset JSONL manifests")
    parser.add_argument("--epochs", type=int, nargs="+", default=DEFAULT_EPOCHS,
                        help="Epochs to include (default: 8–20)")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Which split to re-aggregate (default: val)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for artifacts (default: <save_path>/../selection)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pixel_stride", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    save_path = Path(args.save_path)
    manifest_root = Path(args.manifest_root)

    if args.output_dir is None:
        output_dir = save_path.parent / "selection"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Protect: do NOT overwrite original selection artifact
    original = save_path / "medical_validation_selection.json"
    if not original.exists():
        raise FileNotFoundError(
            f"Original selection artifact not found: {original}\n"
            "Run validation summarization first."
        )

    print(f"[support-aware-v2] save_path={save_path}")
    print(f"[support-aware-v2] manifest_root={manifest_root}")
    print(f"[support-aware-v2] epochs={args.epochs}")
    print(f"[support-aware-v2] split={args.split}")
    print(f"[support-aware-v2] output_dir={output_dir}")

    # Re-aggregate
    epoch_summaries = reaggregate_all_epochs(
        save_path=save_path,
        split=args.split,
        epochs=args.epochs,
        manifest_root=manifest_root,
        datasets=DATASETS,
    )

    if not epoch_summaries:
        raise RuntimeError(
            f"No data found in {save_path} for split={args.split}, epochs={args.epochs}"
        )

    # Select best epoch
    best = select_best_epoch(epoch_summaries)
    best_epoch = int(best["epoch"])
    best_checkpoint = save_path / f"adapter_{best_epoch}.pth"

    # Build fingerprint
    fingerprint_config = build_fingerprint_config(
        validation_epochs=args.epochs,
        manifest_root=manifest_root,
        datasets=DATASETS,
        pixel_stride=args.pixel_stride,
        seed=args.seed,
        temperature=args.temperature,
    )
    fingerprint = compute_config_fingerprint(fingerprint_config)

    # Also load original selection for legacy comparison
    with open(original) as fh:
        original_selection = json.load(fh)
    original_best_epoch = int(original_selection.get("best_epoch", {}).get("epoch", -1))
    original_combined = original_selection.get("best_epoch", {}).get("combined_score", None)

    # Build output artifact
    artifact = {
        "selection_rule": (
            "maximize support_aware_combined_score = "
            "harmonic_mean(image_macro_over_valid_datasets, pixel_macro_over_valid_datasets); "
            "tie-break: lower epoch wins on exact numerical ties"
        ),
        "selection_rule_version": SELECTION_RULE_VERSION,
        "metric_implementation_version": _METRIC_IMPL_VERSION,
        "split": args.split,
        "datasets": list(DATASETS),
        "epochs": sorted(args.epochs),
        "valid_image_datasets": best.get("valid_image_datasets", []),
        "excluded_image_datasets": best.get("excluded_image_datasets", []),
        "valid_pixel_datasets": best.get("valid_pixel_datasets", []),
        "exclusion_reasons": best.get("exclusion_reasons", {}),
        "selected_epoch": best_epoch,
        "selected_checkpoint": str(best_checkpoint),
        "support_aware_combined_score": best["support_aware_combined_score"],
        "support_aware_image_macro": best["support_aware_image_macro"],
        "support_aware_pixel_macro": best["support_aware_pixel_macro"],
        "legacy_combined_score_for_selected_epoch": best["legacy_combined_score"],
        "legacy_selected_epoch": original_best_epoch,
        "legacy_combined_score_for_legacy_selected_epoch": original_combined,
        "epoch_changed_from_legacy": best_epoch != original_best_epoch,
        "command_config_fingerprint": fingerprint,
        "fingerprint_config": fingerprint_config,
        "deterministic_tie_break_reason": (
            "lower epoch wins after sorting support_aware_combined_score desc, epoch asc"
        ),
        "macro_by_epoch": [
            {
                "epoch": s["epoch"],
                "support_aware_combined_score": s["support_aware_combined_score"],
                "support_aware_image_macro": s["support_aware_image_macro"],
                "support_aware_pixel_macro": s["support_aware_pixel_macro"],
                "legacy_combined_score": s["legacy_combined_score"],
                "valid_image_datasets": s["valid_image_datasets"],
                "excluded_image_datasets": s["excluded_image_datasets"],
                "valid_pixel_datasets": s["valid_pixel_datasets"],
                "excluded_pixel_datasets": s.get("excluded_pixel_datasets", []),
                "exclusion_reasons": s.get("exclusion_reasons", {}),
                "per_dataset": s["per_dataset"],
            }
            for s in epoch_summaries
        ],
    }

    suffix = "test_summary_support_aware_v2" if args.split == "test" else "selection_support_aware_v2"
    json_out = output_dir / f"{suffix}.json"
    write_json_atomic(json_out, artifact)
    print(f"[support-aware-v2] wrote JSON: {json_out}")

    # CSV: one row per epoch
    csv_out = output_dir / f"{suffix}.csv"
    csv_fields = [
        "epoch",
        "support_aware_combined_score",
        "support_aware_image_macro",
        "support_aware_pixel_macro",
        "legacy_combined_score",
        "valid_image_datasets",
        "excluded_image_datasets",
        "valid_pixel_datasets",
        "is_selected",
    ]
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for s in epoch_summaries:
            writer.writerow({
                "epoch": s["epoch"],
                "support_aware_combined_score": round(s["support_aware_combined_score"], 6),
                "support_aware_image_macro": (
                    round(s["support_aware_image_macro"], 6)
                    if s["support_aware_image_macro"] is not None else ""
                ),
                "support_aware_pixel_macro": (
                    round(s["support_aware_pixel_macro"], 6)
                    if s["support_aware_pixel_macro"] is not None else ""
                ),
                "legacy_combined_score": round(s["legacy_combined_score"], 6),
                "valid_image_datasets": "|".join(s["valid_image_datasets"]),
                "excluded_image_datasets": "|".join(s["excluded_image_datasets"]),
                "valid_pixel_datasets": "|".join(s["valid_pixel_datasets"]),
                "is_selected": "1" if s["epoch"] == best_epoch else "0",
            })
    print(f"[support-aware-v2] wrote CSV:  {csv_out}")

    # Print summary table
    print("\n--- Epoch table (support-aware-v2 vs legacy) ---")
    print(f"{'epoch':>6} {'sa_combined':>14} {'sa_img_macro':>14} {'sa_pix_macro':>14} {'legacy':>10} {'selected':>8}")
    for s in epoch_summaries:
        sel = " <--- SELECTED" if s["epoch"] == best_epoch else ""
        img = s["support_aware_image_macro"]
        pix = s["support_aware_pixel_macro"]
        print(
            f"{s['epoch']:>6} "
            f"{s['support_aware_combined_score']:>14.4f} "
            f"{img if img is None else f'{img:>14.4f}':>14} "
            f"{pix if pix is None else f'{pix:>14.4f}':>14} "
            f"{s['legacy_combined_score']:>10.4f}"
            f"{sel}"
        )

    print(f"\n[support-aware-v2] selected_epoch={best_epoch}")
    print(f"[support-aware-v2] selected_checkpoint={best_checkpoint}")
    print(f"[support-aware-v2] support_aware_combined_score={best['support_aware_combined_score']:.6f}")
    print(f"[support-aware-v2] support_aware_image_macro={best['support_aware_image_macro']}")
    print(f"[support-aware-v2] support_aware_pixel_macro={best['support_aware_pixel_macro']}")
    print(f"[support-aware-v2] legacy_selected_epoch={original_best_epoch}")
    print(f"[support-aware-v2] epoch_changed={best_epoch != original_best_epoch}")
    print(f"[support-aware-v2] fingerprint={fingerprint}")
    print(f"[support-aware-v2] artifacts: {json_out}")
    print(f"[support-aware-v2] artifacts: {csv_out}")

    if best_epoch != original_best_epoch:
        print(
            f"\n[support-aware-v2] EPOCH CHANGED: {original_best_epoch} -> {best_epoch}\n"
            f"  The running test (epoch {original_best_epoch}) remains valid but is now labeled:\n"
            f"  pre_fix_selection_epoch{original_best_epoch} / exploratory / not_official_under_{SELECTION_RULE_VERSION}\n"
            f"\n"
            f"  DRY-RUN command for the corrected official test:\n"
            f"  bash scripts/phase4/test_6medical_exact.sh --split test {best_epoch}\n"
            f"  (Set SAVE_PATH, MEDICAL_MANIFEST_ROOT, etc. as per protocol; use a NEW result root.)"
        )
    else:
        print(
            f"\n[support-aware-v2] Epoch unchanged ({best_epoch}). "
            f"Running test remains consistent with corrected selection."
        )
        print(
            f"\n  DRY-RUN command (for reference, already running):\n"
            f"  bash scripts/phase4/test_6medical_exact.sh --split test {best_epoch}"
        )


if __name__ == "__main__":
    main()
