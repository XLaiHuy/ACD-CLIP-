#!/usr/bin/env python3
"""Select one frozen Phase2B checkpoint using development MVTec only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from evaluation.datasets import MVTecDatasetAdapter, resolve_mvtec_root
from evaluation.evaluator import evaluate_records
from evaluation.metrics import selection_score

PROTOCOL_VERSION = "PHASE2B_CANONICAL_V1"
DEFAULT_EPOCHS = (10, 12, 14, 16, 18, 20)
METRIC_NAMES = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_candidate(candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Select by the preregistered score; exact ties choose the earlier epoch."""
    rows = []
    required = {"epoch", "path", *METRIC_NAMES}
    for candidate in candidates:
        missing = required - set(candidate)
        if missing:
            raise ValueError(f"candidate missing fields: {sorted(missing)}")
        metrics = {name: float(candidate[name]) for name in METRIC_NAMES}
        if not all(0.0 <= value <= 1.0 for value in metrics.values()):
            raise ValueError("candidate metrics must be in [0,1]")
        score = selection_score(metrics)
        rows.append(dict(candidate) | {"score": score})
    if not rows:
        raise ValueError("no Phase2B candidates supplied")
    return min(rows, key=lambda row: (-float(row["score"]), int(row["epoch"])))


def _metric_row(path: Path, epoch: int, result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result.get("macro", result)
    if not isinstance(metrics, Mapping) or any(name not in metrics for name in METRIC_NAMES):
        raise ValueError("candidate evaluation must return four exact macro metrics")
    row = {"epoch": int(epoch), "path": str(path)}
    row.update({name: float(metrics[name]) for name in METRIC_NAMES})
    row["sha256"] = sha256_file(path)
    return row


def evaluate_checkpoint_candidates(
    checkpoint_paths: Sequence[Path],
    candidate_epochs: Sequence[int],
    evaluate_one: Callable[[Path, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate candidates through a caller-supplied native Phase2B evaluator.

    The selector owns no model or metric implementation.  This seam lets the
    bounded setup tests use precomputed rows while the later experiment wires
    the same frozen runtime and shared evaluator into ``evaluate_one``.
    """
    if len(checkpoint_paths) != len(candidate_epochs):
        raise ValueError("checkpoint count must match candidate epoch count")
    rows = []
    for path, epoch in zip(checkpoint_paths, candidate_epochs):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(_metric_row(path, int(epoch), evaluate_one(path, int(epoch))))
    return rows


def _output_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "epoch": int(row["epoch"]),
            "path": str(row["path"]),
            "sha256": str(row.get("sha256", "")),
            "pAUROC": float(row["pixel_auroc"]),
            "pAP": float(row["pixel_ap"]),
            "iAUROC": float(row["image_auroc"]),
            "iAP": float(row["image_ap"]),
            "score": float(row["score"]),
        }
        for row in candidates
    ]


def _write_selection(output_dir: Path, candidates: list[dict[str, Any]], selected: dict[str, Any], code_sha: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = _output_rows(candidates)
    selected_output = next(row for row in outputs if int(row["epoch"]) == int(selected["epoch"]))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "FROZEN",
        "source_training_dataset": "VisA",
        "development_dataset": "MVTecAD",
        "score": {
            "unit": "[0,1]",
            "formula": ".35*pAUROC+.35*pAP+.15*iAUROC+.15*iAP",
            "weights": {"pixel_auroc": 0.35, "pixel_ap": 0.35, "image_auroc": 0.15, "image_ap": 0.15},
        },
        "candidates": outputs,
        "selected_epoch": int(selected["epoch"]),
        "selected_checkpoint": str(selected["path"]),
        "selected_checkpoint_sha256": selected_output["sha256"],
        "evaluator": {"exact": True, "pixel_stride": 1, "image_score_contract": "domain-specific frozen contract", "code_sha": code_sha},
    }
    (output_dir / "phase2b_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "phase2b_selection_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "path", "sha256", "pAUROC", "pAP", "iAUROC", "iAP", "score"])
        writer.writeheader()
        writer.writerows(outputs)


def _candidate_paths(args: argparse.Namespace, epochs: Sequence[int]) -> list[Path]:
    paths = list(args.checkpoint)
    if args.checkpoint_dir is not None:
        paths.extend(args.checkpoint_dir / f"adapter_{epoch}.pth" for epoch in epochs)
    if len(paths) != len(epochs):
        raise SystemExit("provide exactly one checkpoint per candidate epoch")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--candidate-epochs", type=str, default=",".join(map(str, DEFAULT_EPOCHS)))
    parser.add_argument("--mvtec-root", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2b_canonical_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase2b_selection"))
    parser.add_argument("--metrics-json", type=Path, help="bounded/precomputed candidate macro metrics; no model inference")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    if args.preflight_only:
        root = resolve_mvtec_root(args.mvtec_root)
        if root is None:
            print("MVTEC_PREFLIGHT = NOT_RUN_NO_ROOT")
            print("--mvtec-root <PATH>")
            return 0
        report = MVTecDatasetAdapter(root).preflight()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    root = resolve_mvtec_root(args.mvtec_root)
    if root is None:
        raise SystemExit("MVTec development root is required; use --mvtec-root <PATH>")
    epochs = tuple(int(value) for value in args.candidate_epochs.split(",") if value.strip())
    if not epochs or epochs != tuple(sorted(set(epochs))):
        raise SystemExit("--candidate-epochs must be unique and ascending")
    paths = _candidate_paths(args, epochs)
    if args.metrics_json is None:
        raise SystemExit("setup does not run a real MVTec sweep; provide future precomputed --metrics-json")
    raw = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("--metrics-json must contain a list of candidate macro metrics")
    by_epoch = {int(row["epoch"]): row for row in raw}
    rows = []
    for path, epoch in zip(paths, epochs):
        if epoch not in by_epoch:
            raise SystemExit(f"metrics JSON has no candidate epoch {epoch}")
        rows.append(_metric_row(path, epoch, by_epoch[epoch]))
    selected = select_candidate(rows)
    _write_selection(args.output_dir, rows, selected, code_sha=sha256_file(Path(__file__).resolve()))
    return 0


def evaluate_candidate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Apply the same exact metric evaluator used by final test.py."""
    return evaluate_records(records, method="phase2b")["macro"]


if __name__ == "__main__":
    raise SystemExit(main())
