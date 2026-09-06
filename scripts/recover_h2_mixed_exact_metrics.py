#!/usr/bin/env python3
"""Resume only missing exact-metric postprocessing from durable source arrays.

This recovery entry point deliberately cannot run model inference.  It validates
the inference completion marker and array shapes, imports the newest valid
superset checkpoint, and computes only absent distribution/morphology cohorts.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "audit"
LARGE = Path("/workspace/h2_mixed_generalization_audit_v1")
PROGRESS = AUDIT / "H2_MIXED_EXACT_METRIC_PROGRESS.json"
TMP_PROGRESS = AUDIT / "H2_MIXED_EXACT_METRIC_PROGRESS.json.tmp"
N = 2162
IMG = 518
KINDS = ("positive", "negative", "boundary", "interior", "near_background", "far_background")


class Moments:
    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.ss = 0.0
        self.sample: list[float] = []

    def add(self, values):
        a = np.asarray(values, np.float32).reshape(-1)
        if not a.size:
            return
        start = self.n
        self.n += a.size
        self.s += float(a.sum(dtype=np.float64))
        self.ss += float(np.square(a, dtype=np.float64).sum())
        self.sample.extend(a[(-start) % 257::257].tolist())

    def result(self):
        q = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        values = np.quantile(np.asarray(self.sample), q)
        mean = self.s / self.n
        return {
            "count": int(self.n), "mean": mean,
            "std": max(0.0, self.ss / self.n - mean * mean) ** 0.5,
            **{f"p{int(x * 100):02d}": float(v) for x, v in zip(q, values)},
            "quantile_method": "deterministic_stride_257",
        }


def read_valid(path: Path):
    try:
        data = json.loads(path.read_text())
        rows = data["completed"]
        if not isinstance(rows, list):
            raise TypeError("completed is not a list")
        keys = [(r["arm"], r["cohort"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate completed cohort")
        return rows
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def durable_json(path: Path, payload):
    staging = path.with_name(path.name + ".recovery")
    with staging.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(staging, path)


def morphology(mask):
    structure = np.ones((7, 7), bool)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    dilated = ndimage.binary_dilation(mask, structure=structure)
    return mask & ~eroded, eroded, dilated & ~mask, ~dilated


def main():
    marker = json.loads((AUDIT / "H2_MIXED_INFERENCE_COMPLETE.json").read_text())
    if marker.get("status") != "PASS" or marker.get("source_count") != N:
        raise RuntimeError("source inference completion marker is not valid")
    main_rows, tmp_rows = read_valid(PROGRESS), read_valid(TMP_PROGRESS)
    valid = [x for x in (main_rows, tmp_rows) if x is not None]
    if not valid:
        raise RuntimeError("no valid exact-metric checkpoint")
    rows = max(valid, key=len)
    completed = {(r["arm"], r["cohort"]) for r in rows}
    expected_shape = (N, IMG, IMG)
    masks = np.load(LARGE / "masks.npy", mmap_mode="r")
    if masks.shape != expected_shape:
        raise RuntimeError(f"mask shape mismatch: {masks.shape}")

    for arm in ("H", "A"):
        missing = [k for k in KINDS if (arm, f"distribution:{k}") not in completed]
        if not missing:
            print(f"arm={arm} distributions=reused", flush=True)
            continue
        scores = np.load(LARGE / f"{arm}_final.npy", mmap_mode="r")
        if scores.shape != expected_shape:
            raise RuntimeError(f"{arm} final-map shape mismatch: {scores.shape}")
        moments = {kind: Moments() for kind in missing}
        for i in range(N):
            mask = np.asarray(masks[i], bool)
            score = np.asarray(scores[i], np.float32)
            boundary, interior, near, far = morphology(mask)
            selections = {
                "positive": mask, "negative": ~mask, "boundary": boundary,
                "interior": interior, "near_background": near, "far_background": far,
            }
            for kind in missing:
                moments[kind].add(score[selections[kind]])
            if (i + 1) % 120 == 0:
                print(f"arm={arm} morphology={i + 1}/{N}", flush=True)
        for kind in missing:
            row = {"arm": arm, "cohort": f"distribution:{kind}", **moments[kind].result()}
            rows.append(row)
            completed.add((arm, row["cohort"]))
            durable_json(PROGRESS, {"completed": rows})
            print(f"completed={arm}/{row['cohort']}", flush=True)

    expected = {(arm, cohort) for arm in ("H", "A") for cohort in (
        "raw", "final", "image", *[f"category:{x}" for x in (
            "candle", "pcb3", "capsules", "pipe_fryum", "pcb4", "macaroni2",
            "pcb2", "chewinggum", "macaroni1", "cashew", "fryum", "pcb1")],
        "stage:1", "stage:2", "stage:3", *[f"distribution:{x}" for x in KINDS])}
    missing = sorted(expected - completed)
    if missing:
        raise RuntimeError(f"exact metric recovery incomplete: {missing}")

    with (AUDIT / "H2_MIXED_SOURCE_PER_IMAGE.csv").open(newline="") as f:
        image_rows = list(csv.DictReader(f))
    if len(image_rows) != 2 * N:
        raise RuntimeError(f"per-image row count mismatch: {len(image_rows)}")
    fields = list(dict.fromkeys(k for r in rows + image_rows for k in r))
    with (AUDIT / "H2_MIXED_SOURCE_PIXEL_RANKING.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows + image_rows)
    durable_json(AUDIT / "H2_MIXED_EXACT_METRIC_COMPLETE.json", {
        "status": "PASS", "source_inference_reused": True,
        "completed_metric_cohorts": len(rows),
        "pixel_ranking_rows": len(rows) + len(image_rows),
    })
    print(json.dumps({"status": "PASS", "completed_metric_cohorts": len(rows)}))


if __name__ == "__main__":
    main()
