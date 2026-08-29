#!/usr/bin/env python3
"""CIR/REPORT: consolidate exact result rows without selecting a best epoch."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import Any

EPOCHS = (12, 14, 16, 18, 20)
METRIC_FILES = {
    "pixel_ap": "pixel_ap",
    "pixel_auroc": "pixel_auroc",
    "image_ap": "image_ap",
    "image_auroc": "image_auroc",
}


def collect(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("results.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    if rows:
        return rows
    for path in sorted(root.rglob("metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta = {key: payload[key] for key in ("arch_id", "source", "target", "epoch", "checkpoint", "config_sha256", "git_sha", "evaluator_protocol") if key in payload}
        macro = payload.get("macro", payload.get("metrics", {}))
        if isinstance(macro, dict):
            rows.append({**meta, **{f"macro_{key}": value for key, value in macro.items()}, "metrics_path": str(path)})
    return rows


def _merge_long(path: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    keys = {(str(row.get("source")), str(row.get("target")), str(row.get("epoch")), str(row.get("checkpoint_sha256", row.get("checkpoint")))) for row in rows}
    kept = [row for row in existing if (str(row.get("source")), str(row.get("target")), str(row.get("epoch")), str(row.get("checkpoint_sha256", row.get("checkpoint")))) not in keys]
    return kept + rows


def _write_matrix(global_root: Path, source: str, metric: str, rows: list[dict[str, Any]]) -> None:
    path = global_root / f"{source}_source_{metric}.csv"
    by_target: dict[str, dict[str, str]] = {}
    for row in rows:
        if str(row.get("source")) != source:
            continue
        target = str(row.get("target", ""))
        if not target:
            continue
        by_target.setdefault(target, {})
        epoch = str(row.get("epoch", ""))
        field = f"macro_{metric}"
        if epoch in {str(value) for value in EPOCHS}:
            by_target[target][f"e{epoch}"] = str(row.get(field, ""))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", *[f"e{epoch}" for epoch in EPOCHS]])
        writer.writeheader()
        for target in sorted(by_target):
            writer.writerow({"dataset": target, **{f"e{epoch}": by_target[target].get(f"e{epoch}", "") for epoch in EPOCHS}})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/cir_rmt/CIR_DFG_RMT_V1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rows = collect(args.run_root)
    destination = args.output or (args.run_root / "summary.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["arch_id", "source", "target", "epoch"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    global_root = args.run_root
    if args.run_root.name.startswith("seed") and args.run_root.parent.name in {"visa", "mvtec"}:
        global_root = args.run_root.parent.parent
    long_path = global_root / "results_long.csv"
    merged = _merge_long(long_path, rows)
    long_fields = sorted({key for row in merged for key in row}) or ["arch_id", "source", "target", "epoch"]
    with long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    for source in ("visa", "mvtec"):
        for metric in METRIC_FILES:
            _write_matrix(global_root, source, metric, merged)
    (destination.with_suffix(".json")).write_text(json.dumps({"stage": "CIR/REPORT", "status": "PASS", "row_count": len(rows), "rows": rows}, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "CIR/REPORT", "status": "PASS", "row_count": len(rows), "output": str(destination), "results_long": str(long_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
