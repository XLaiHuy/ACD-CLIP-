#!/usr/bin/env python3
"""Explicit two-phase SABRA source calibration and MVTec lambda selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evaluation.datasets import resolve_mvtec_root
from evaluation.evaluator import evaluate_records
from evaluation.metrics import selection_score
from tools.sabra.artifacts import (
    build_freeze_payload,
    validate_source_calibration,
    write_json,
)
from tools.sabra.correction import margin_scale_p90
from tools.sabra.need import fit_need
from tools.sabra.relational import FEATURE_ORDER, NEED_ORDER
from tools.sabra.trust import fit_trust

PROTOCOL_VERSION = "SABRA_CANONICAL_V1"
MEDICAL_DATASETS = {"Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"}
COARSE_LAMBDAS = tuple(float(value) for value in np.round(np.arange(0.0, 1.0001, 0.025), 6))
REFINEMENT_RULE = "center +/- 0.05 clamped to [0,1], step 0.005; no duplicate coarse points"


def fit_source_payload(
    records: Iterable[Mapping[str, Any]],
    selected_epoch: int,
    checkpoint_sha256: str,
    margin_values: np.ndarray,
    git_sha: str,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("VisA source calibration requires non-empty records")
    if any(str(row.get("class_name")) in MEDICAL_DATASETS for row in rows):
        raise ValueError("Medical records cannot enter VisA source calibration")
    trust = fit_trust(rows)
    need = fit_need(rows)
    provenance: dict[str, Any] = {"git_sha": str(git_sha)}
    if source_hashes is not None:
        provenance["critical_source_hashes"] = dict(source_hashes)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "SOURCE_FITTED",
        "phase2b": {"selected_epoch": int(selected_epoch), "checkpoint_sha256": str(checkpoint_sha256)},
        "relational": {
            "implementation": "tools.sabra.relational.build_relational_record",
            "peer_count": 8,
            "feature_contract": list(FEATURE_ORDER),
        },
        "trust": {**trust, "feature_order": list(FEATURE_ORDER)},
        "need": {**need, "feature_order": list(NEED_ORDER)},
        "margin_scale": margin_scale_p90(margin_values),
        "provenance": provenance,
    }
    validate_source_calibration(payload)
    return payload


def lambda_grid() -> np.ndarray:
    return np.asarray(COARSE_LAMBDAS, dtype=np.float64)


def refined_lambda_grid(center: float, exclude: Sequence[float] = ()) -> np.ndarray:
    lo = max(0.0, float(center) - 0.05)
    hi = min(1.0, float(center) + 0.05)
    grid = np.round(np.arange(lo, hi + 0.0001, 0.005), 6)
    excluded = {round(float(value), 6) for value in exclude}
    return np.asarray([value for value in grid if round(float(value), 6) not in excluded], dtype=np.float64)


def select_lambda(curve: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    required = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")
    for row in curve:
        if "lambda" not in row or any(name not in row for name in required):
            raise ValueError("lambda curve rows must include lambda and four exact metrics")
        metrics = {name: float(row[name]) for name in required}
        if not all(0.0 <= value <= 1.0 for value in metrics.values()):
            raise ValueError("lambda curve metrics must be in [0,1]")
        rows.append(dict(row) | {"score": selection_score(metrics), "lambda": float(row["lambda"])})
    if not rows:
        raise ValueError("empty lambda curve")
    return min(rows, key=lambda row: (-float(row["score"]), float(row["lambda"])))


def _reject_medical(dataset: str) -> None:
    if str(dataset) in MEDICAL_DATASETS or str(dataset).lower() == "medical":
        raise SystemExit("Medical is final zero-shot test data and cannot be a calibration input")


def _read_curve(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("curve", payload.get("rows"))
    if not isinstance(payload, list):
        raise ValueError("curve JSON must be a list or an object containing curve/rows")
    return [dict(row) for row in payload]


def _write_lambda_outputs(
    output_dir: Path,
    source: Mapping[str, Any],
    rows: list[dict[str, Any]],
    selected: Mapping[str, Any],
    git_sha: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["lambda", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap", "score"]
    extra = [name for name in rows[0] if name not in fieldnames]
    fieldnames.extend(extra)
    with (output_dir / "lambda_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    freeze = build_freeze_payload(
        source,
        selected_lambda=float(selected["lambda"]),
        selected_score=float(selected["score"]),
        git_sha=git_sha,
        coarse_grid=COARSE_LAMBDAS,
        refinement_rule=REFINEMENT_RULE,
    )
    write_json(output_dir / "SABRA_FREEZE.json", freeze)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit-source", help="fit Trust/Need and margin scale on VisA only")
    fit.add_argument("--phase2b-selection", type=Path, required=True)
    fit.add_argument("--visa-root", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--dataset", default="VisA")
    fit.add_argument("--records-json", type=Path)
    fit.add_argument("--git-sha", default="WORKTREE_SHA")

    select = subparsers.add_parser("select-lambda", help="select frozen correction scale on MVTec development data")
    select.add_argument("--source-calibration", type=Path, required=True)
    select.add_argument("--mvtec-root", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--curve-json", type=Path)
    select.add_argument("--git-sha", default="WORKTREE_SHA")

    args = parser.parse_args(argv)
    if args.command == "fit-source":
        _reject_medical(args.dataset)
        selection = json.loads(args.phase2b_selection.read_text(encoding="utf-8"))
        if selection.get("status") != "FROZEN":
            raise SystemExit("phase2b selection must be FROZEN")
        if args.records_json is None:
            raise SystemExit("setup does not run real VisA calibration; provide future frozen records via --records-json")
        rows = json.loads(args.records_json.read_text(encoding="utf-8"))
        payload = fit_source_payload(
            rows,
            int(selection["selected_epoch"]),
            str(selection["selected_checkpoint_sha256"]),
            np.asarray([value for row in rows for value in row["native_margins"]], dtype=np.float64),
            args.git_sha,
        )
        write_json(args.output_dir / "sabra_source_calibration.json", payload)
        return 0

    source = json.loads(args.source_calibration.read_text(encoding="utf-8"))
    validate_source_calibration(source)
    root = resolve_mvtec_root(args.mvtec_root)
    if root is None or not root.exists():
        raise SystemExit(f"MVTec root does not exist: {args.mvtec_root}")
    if args.curve_json is None:
        raise SystemExit("setup does not run a real lambda sweep; provide future frozen curve via --curve-json")
    rows = _read_curve(args.curve_json)
    scored = []
    for row in rows:
        selected_row = select_lambda([row])
        scored.append(selected_row)
    selected = min(scored, key=lambda row: (-float(row["score"]), float(row["lambda"])))
    _write_lambda_outputs(args.output_dir, source, scored, selected, args.git_sha)
    return 0


def evaluate_lambda_records(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Apply the same exact metric evaluator used by final test.py."""
    return evaluate_records(records, method="sabra")["macro"]


if __name__ == "__main__":
    raise SystemExit(main())
