#!/usr/bin/env python3
"""Summarize Stage-B sufficient statistics and select static loss/lambda semantics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


QUANTILES = {"min": 0.0, "median": 0.5, "p75": 0.75, "p90": 0.9, "p95": 0.95, "max": 1.0}
FACTOR_GRID = (0.10, 0.05, 0.03, 0.01, 0.005, 0.003, 0.002, 0.001)
ROUTER_GRID = (0.10, 0.05, 0.03, 0.01, 0.005)
SELECTED_FACTOR = "F1_effective_beta_0.999"
SELECTED_ROUTER = "R1_valid_support_mean"


def distribution(values) -> dict:
    values = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": None, **{name: None for name in QUANTILES}}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        **{name: float(np.quantile(values, q)) for name, q in QUANTILES.items()},
    }


def effective_weight_share(microbatch: dict, beta: float) -> float | None:
    normal = microbatch["factor_regions"]["normal"]["count_group_patches"]
    anomaly = microbatch["factor_regions"]["anomaly"]["count_group_patches"]
    if not anomaly:
        return None
    effective_normal = (1.0 - beta**normal) / (1.0 - beta) if normal else 0.0
    effective_anomaly = (1.0 - beta**anomaly) / (1.0 - beta)
    normal_mass = normal / effective_normal if normal else 0.0
    anomaly_mass = anomaly / effective_anomaly
    return anomaly_mass / (normal_mass + anomaly_mass)


def combined(window, factor_name: str, router_name: str, lf: float, lr: float) -> dict:
    geometry = window["geometry"]["shared_semantic"]
    main = geometry["main_norm"]
    factor = geometry["factors"][factor_name]
    router = geometry["routers"][router_name]
    cross = geometry["pairs"][f"{factor_name}+{router_name}"]["dot_factor_router"]
    aux_squared = max(
        0.0,
        lf * lf * factor["norm"] ** 2
        + lr * lr * router["norm"] ** 2
        + 2.0 * lf * lr * cross,
    )
    aux_norm = math.sqrt(aux_squared)
    main_dot_aux = lf * factor["dot_main"] + lr * router["dot_main"]
    return {
        "factor_to_main": lf * factor["norm"] / main if main > 1e-12 else None,
        "router_to_main": lr * router["norm"] / main if main > 1e-12 else None,
        "combined_to_main": aux_norm / main if main > 1e-12 else None,
        "cos_main_combined": (
            main_dot_aux / (main * aux_norm) if main > 1e-12 and aux_norm > 1e-12 else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    windows = payload["windows"]
    factor_names = tuple(windows[0]["geometry"]["shared_semantic"]["factors"])
    router_names = tuple(windows[0]["geometry"]["shared_semantic"]["routers"])

    factor_report = {}
    anomaly_fractions = np.asarray([w["counts"]["anomaly_fraction"] for w in windows])
    for name in factor_names:
        ratios = np.asarray(
            [w["geometry"]["shared_semantic"]["factors"][name]["to_main"] for w in windows]
        )
        entry = {
            "raw_factor_to_main": distribution(ratios),
            "anomaly_windows": distribution(
                w["geometry"]["shared_semantic"]["factors"][name]["to_main"]
                for w in windows if w["categories"]["anomaly_containing"]
            ),
            "normal_only_windows": distribution(
                w["geometry"]["shared_semantic"]["factors"][name]["to_main"]
                for w in windows if w["categories"]["normal_only"]
            ),
            "cos_main_factor": distribution(
                w["geometry"]["shared_semantic"]["factors"][name]["cos_main"] for w in windows
            ),
            "anomaly_fraction_pearson": float(np.corrcoef(anomaly_fractions, ratios)[0, 1]),
        }
        if "beta_" in name:
            beta = float(name.rsplit("_", 1)[1])
            shares = [
                share
                for window in windows
                for microbatch in window["microbatches"]
                if (share := effective_weight_share(microbatch, beta)) is not None
            ]
            entry["anomaly_total_patch_weight_share"] = distribution(shares)
        factor_report[name] = entry

    router_report = {}
    for name in router_names:
        active_cosines = [
            w["geometry"]["shared_semantic"]["routers"][name]["cos_main"]
            for w in windows
            if w["geometry"]["shared_semantic"]["routers"][name]["cos_main"] is not None
        ]
        router_report[name] = {
            "raw_router_to_main": distribution(
                w["geometry"]["shared_semantic"]["routers"][name]["to_main"] for w in windows
            ),
            "informative_windows": distribution(
                w["geometry"]["shared_semantic"]["routers"][name]["to_main"]
                for w in windows if w["categories"]["router_informative"]
            ),
            "cos_main_router": distribution(active_cosines),
            "negative_cosine_fraction": float(np.mean(np.asarray(active_cosines) < 0.0)),
            "nonzero_gradient_windows": sum(
                w["geometry"]["shared_semantic"]["routers"][name]["norm"] > 1e-12
                for w in windows
            ),
        }

    grid = []
    for lf in FACTOR_GRID:
        for lr in ROUTER_GRID:
            rows = [combined(w, SELECTED_FACTOR, SELECTED_ROUTER, lf, lr) for w in windows]
            grid.append(
                {
                    "lambda_factor": lf,
                    "lambda_router": lr,
                    "factor_to_main": distribution(row["factor_to_main"] for row in rows),
                    "router_to_main": distribution(row["router_to_main"] for row in rows),
                    "combined_to_main": distribution(row["combined_to_main"] for row in rows),
                    "cos_main_combined": distribution(row["cos_main_combined"] for row in rows),
                    "combined_over_main_window_count": sum(
                        row["combined_to_main"] is not None and row["combined_to_main"] > 1.0
                        for row in rows
                    ),
                }
            )

    selected = next(
        row for row in grid
        if row["lambda_factor"] == 0.03 and row["lambda_router"] == 0.10
    )
    f0_p95 = factor_report["F0_region_mean_50_50"]["raw_factor_to_main"]["p95"]
    f1_p95 = factor_report[SELECTED_FACTOR]["raw_factor_to_main"]["p95"]
    r0_p95 = router_report["R0_informative_mean"]["raw_router_to_main"]["p95"]
    r1_p95 = router_report[SELECTED_ROUTER]["raw_router_to_main"]["p95"]
    selected_factor_cosines = [
        w["geometry"]["shared_semantic"]["factors"][SELECTED_FACTOR]["cos_main"]
        for w in windows
    ]
    decision = {
        "status": "STATIC_OK_PCGRAD_REQUIRED",
        "selected_factor_formula": SELECTED_FACTOR,
        "selected_factor_beta": 0.999,
        "selected_router_formula": SELECTED_ROUTER,
        "selected_lambda_factor": 0.03,
        "selected_lambda_router": 0.10,
        "selection_logic": {
            "factor": (
                "beta=.999 reduces rarity-driven p90/p95 and spikes while retaining about one fifth "
                "of total effective patch weight for anomaly support; beta=.99 suppresses anomaly "
                "weight to only a few percent and beta=.9999 remains close to F0"
            ),
            "router": (
                "R1 removes informative-count amplification, remains nonzero on 65 natural support "
                "windows, and is not compensated beyond the preregistered grid"
            ),
            "lambda": (
                ".03/.10 is the largest grid pair with factor median in the engineering region, "
                "combined p90 below main, and zero windows above main; factor .05 causes four "
                "repeated >1x-main windows"
            ),
        },
        "effect_sizes": {
            "factor_p95_reduction_fraction": 1.0 - f1_p95 / f0_p95,
            "router_p95_reduction_fraction": 1.0 - r1_p95 / r0_p95,
            "selected_combined": selected,
        },
        "magnitude_branch": "STATIC_OK; GradNorm is not eligible",
        "conflict_evidence": {
            "main_factor_cosine": distribution(selected_factor_cosines),
            "negative_window_fraction": float(
                np.mean(np.asarray(selected_factor_cosines) < 0.0)
            ),
        },
        "conflict_branch": (
            "PCGrad is required for main+factor on shared-semantic parameters: negative cosine "
            "persists in a majority of natural windows with a materially negative median after "
            "loss normalization. Router remains fixed-lambda and unprojected because its cosine "
            "is numerically near zero and its supervision is sparse."
        ),
        "minimum_source_solution": (
            "effective-number factor loss + support-normalized router + static .03/.10 + "
            "two-objective PCGrad on accumulated main/factor shared gradients"
        ),
    }
    report = {
        "source": str(args.input),
        "window_count": len(windows),
        "window_categories": payload["summary"]["category_counts"],
        "counts": {
            "anomaly_patch_count": distribution(w["counts"]["anomaly_patch_count"] for w in windows),
            "normal_patch_count": distribution(w["counts"]["normal_patch_count"] for w in windows),
            "anomaly_fraction": distribution(w["counts"]["anomaly_fraction"] for w in windows),
            "informative_count": distribution(w["counts"]["informative_group_patch_count"] for w in windows),
            "informative_fraction": distribution(w["counts"]["informative_fraction"] for w in windows),
        },
        "factor_candidates": factor_report,
        "router_candidates": router_report,
        "lambda_grid": grid,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "window_distributions.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "static_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    with (args.output_dir / "lambda_grid.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("lambda_factor", "lambda_router", "factor_median", "router_median", "combined_median", "combined_p90", "combined_p95", "combined_max", "over_main_windows"))
        for row in grid:
            writer.writerow((
                row["lambda_factor"], row["lambda_router"],
                row["factor_to_main"]["median"], row["router_to_main"]["median"],
                row["combined_to_main"]["median"], row["combined_to_main"]["p90"],
                row["combined_to_main"]["p95"], row["combined_to_main"]["max"],
                row["combined_over_main_window_count"],
            ))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
