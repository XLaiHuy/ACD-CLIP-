#!/usr/bin/env python3
"""Create an explicit Single-View versus locked Four-View TTA delta."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METRIC_NAMES = ("pixel_auc", "pixel_ap", "image_auc", "image_ap")
ROW_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s+([-+0-9.eE]+|nan)\s+([-+0-9.eE]+|nan)\s+([-+0-9.eE]+|nan)\s+([-+0-9.eE]+|nan)\s*$"
)


def parse_single_view_log(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        rows[name] = {
            key: float(value) for key, value in zip(METRIC_NAMES, match.groups()[1:])
        }
    if "Average" not in rows:
        raise SystemExit(f"missing Average row in {path}")
    return rows


def subtract(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: float(left[key] - right[key]) for key in METRIC_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-root", type=Path, required=True)
    parser.add_argument("--tta-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    tta = json.loads(args.tta_json.read_text(encoding="utf-8"))
    results = {}
    for dataset_name in tta["datasets"]:
        single_rows = parse_single_view_log(args.single_root / dataset_name / "test.log")
        four = tta["results"][dataset_name]
        four_rows = four["per_class"] | {"Average": four["macro"]}
        common = sorted(set(single_rows).intersection(four_rows))
        if "Average" not in common:
            raise SystemExit(f"single/four-view Average rows do not match for {dataset_name}")
        results[dataset_name] = {
            "reported_image_metrics": dataset_name in {"Brain", "Liver", "Retina"},
            "single_view": {name: single_rows[name] for name in common},
            "four_view": {name: four_rows[name] for name in common},
            "delta_tta": {
                name: subtract(four_rows[name], single_rows[name]) for name in common
            },
        }
    payload = {
        "protocol": "RENTAL_DELTA_TTA_V1",
        "checkpoint": tta["checkpoint"],
        "checkpoint_sha256": tta["checkpoint_sha256"],
        "policy": tta["policy"],
        "datasets": tta["datasets"],
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "datasets": list(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
