"""Source-only P35 actionability-map comparison.

This module reads only the locked Tier-B source teacher-region cache.  It does
not read held masks/labels, run CLIP/Phase2B/teacher models, rebuild a cache,
or train an adapter.  The candidate maps are deliberately kept as pure
functions so the comparison cannot change the full functional target.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from tools.sabra_v2.p29_contract import CORRECTION_SCALE
from tools.sabra_v2.p32_objective import deployed_margin_effect


DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_CLASS = "candle"
IMAGE_SIZE = 518
RECORD_CHUNK = 8
SAMPLE_TARGET = 300_000
MAP_NAMES = ("clamp", "tanh", "rational")


def actionability_map(x: np.ndarray, name: str) -> np.ndarray:
    """Apply one of the preregistered candidate maps to nonnegative ``x``."""
    if name == "clamp":
        return np.clip(x, 0.0, 1.0)
    if name == "tanh":
        return np.tanh(x)
    if name == "rational":
        return x / (1.0 + x)
    raise ValueError(f"unknown actionability map: {name}")


def actionability_derivative(x: float, name: str) -> float | None:
    """Return the analytic derivative where the candidate is differentiable."""
    if name == "clamp":
        if x < 1.0:
            return 1.0
        if x > 1.0:
            return 0.0
        return None
    if name == "tanh":
        value = math.tanh(x)
        return 1.0 - value * value
    if name == "rational":
        return 1.0 / ((1.0 + x) ** 2)
    raise ValueError(f"unknown actionability map: {name}")


def _gini(values: np.ndarray) -> float:
    values = np.abs(np.asarray(values, dtype=np.float64).reshape(-1))
    total = float(values.sum())
    if values.size == 0 or total == 0.0:
        return 0.0
    ordered = np.sort(values)
    positions = np.arange(1, ordered.size + 1, dtype=np.float64)
    return float(np.sum((2.0 * positions - ordered.size - 1.0) * ordered) / (ordered.size * total))


def _sample_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    total = float(values.sum())
    sq_total = float(np.square(values).sum())
    n = int(values.size)
    ordered = np.sort(values)

    def top_mass(fraction: float) -> float:
        if total == 0.0:
            return 0.0
        count = max(1, math.ceil(fraction * n))
        return float(ordered[-count:].sum() / total)

    return {
        "n": n,
        "mean": float(values.mean()),
        "median": float(np.quantile(values, 0.50)),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "min": float(values.min()),
        "max": float(values.max()),
        "exact_zero_fraction": float(np.mean(values == 0.0)),
        "gt_025_fraction": float(np.mean(values > 0.25)),
        "gt_050_fraction": float(np.mean(values > 0.50)),
        "gt_075_fraction": float(np.mean(values > 0.75)),
        "gt_090_fraction": float(np.mean(values > 0.90)),
        "exact_one_fraction": float(np.mean(values == 1.0)),
        "effective_fraction": float(total * total / (n * sq_total)) if sq_total else 0.0,
        "gini": _gini(values),
        "top_1_percent_mass": top_mass(0.01),
        "top_5_percent_mass": top_mass(0.05),
        "top_10_percent_mass": top_mass(0.10),
        "finite": bool(np.isfinite(values).all()),
    }


class _Accumulator:
    def __init__(self) -> None:
        self.n = 0
        self.total = 0.0
        self.sq_total = 0.0
        self.minimum = float("inf")
        self.maximum = -float("inf")
        self.counts: dict[str, int] = {}
        self.sample: list[np.ndarray] = []

    def add(self, values: np.ndarray, offset: int, stride: int, predicates: dict[str, np.ndarray] | None = None) -> None:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        self.n += int(values.size)
        self.total += float(values.sum())
        self.sq_total += float(np.square(values).sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        if predicates:
            for key, mask in predicates.items():
                self.counts[key] = self.counts.get(key, 0) + int(np.count_nonzero(mask))
        first = (-offset) % stride
        self.sample.append(values[first::stride].copy())

    def summary(self) -> dict[str, Any]:
        sample = np.concatenate(self.sample) if self.sample else np.zeros(0, dtype=np.float64)
        result = _sample_summary(sample)
        result.update(
            {
                "full_n": self.n,
                "full_mean": self.total / self.n,
                "full_rms": math.sqrt(self.sq_total / self.n),
                "full_min": self.minimum,
                "full_max": self.maximum,
                "full_counts": self.counts,
                "systematic_sample_size": int(sample.size),
            }
        )
        return result


def _fixed_points() -> dict[str, dict[str, float | None]]:
    points = (0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    result: dict[str, dict[str, float | None]] = {}
    for name in MAP_NAMES:
        result[name] = {}
        for point in points:
            result[name][f"x={point:g}"] = float(actionability_map(np.asarray(point), name))
            result[name][f"dw_dx@x={point:g}"] = actionability_derivative(point, name)
    return result


def analyze_source(cache_root: Path = DEFAULT_CACHE_ROOT, held_class: str = DEFAULT_CLASS) -> dict[str, Any]:
    shard = cache_root / "tier_b" / held_class
    manifest_path = shard / "manifest.json"
    teacher_path = shard / "teacher_region.npy"
    if not manifest_path.is_file() or not teacher_path.is_file():
        raise RuntimeError("locked Tier-B source teacher cache is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_ids = list(manifest["sample_ids"])
    teacher = np.load(teacher_path, mmap_mode="r", allow_pickle=False)
    if tuple(teacher.shape) != (len(sample_ids), 9, 9):
        raise RuntimeError("source teacher-region shape changed")
    if any(str(sample_id).split(":", 1)[0] == held_class for sample_id in sample_ids):
        raise RuntimeError("held-class sample reached source-only P35 analysis")

    total_pixels = len(sample_ids) * IMAGE_SIZE * IMAGE_SIZE
    stride = max(1, math.ceil(total_pixels / SAMPLE_TARGET))
    raw_x = _Accumulator()
    weights = {name: _Accumulator() for name in MAP_NAMES}
    optimization_mass = {name: _Accumulator() for name in MAP_NAMES}
    category_rows: dict[str, list[list[float]]] = {}
    seen = 0

    with torch.no_grad():
        for start in range(0, len(sample_ids), RECORD_CHUNK):
            values = torch.from_numpy(np.array(teacher[start : start + RECORD_CHUNK], copy=True)).float()
            effect = deployed_margin_effect(values).cpu().numpy().astype(np.float64, copy=False)
            flat_effect = effect.reshape(-1)
            x = np.abs(flat_effect) / CORRECTION_SCALE
            raw_x.add(
                x,
                seen,
                stride,
                {
                    "x_eq_0": x == 0.0,
                    "x_lt_01": x < 0.1,
                    "x_lt_025": x < 0.25,
                    "x_lt_05": x < 0.5,
                    "x_lt_1": x < 1.0,
                    "x_ge_1": x >= 1.0,
                    "x_ge_2": x >= 2.0,
                    "x_ge_5": x >= 5.0,
                },
            )
            slope_at_zero_student = np.minimum(np.abs(flat_effect), 1.0)
            for name in MAP_NAMES:
                weight = actionability_map(x, name)
                weights[name].add(
                    weight,
                    seen,
                    stride,
                    {
                        "w_eq_0": weight == 0.0,
                        "w_gt_025": weight > 0.25,
                        "w_gt_05": weight > 0.5,
                        "w_gt_075": weight > 0.75,
                        "w_gt_09": weight > 0.9,
                        "w_eq_1": weight == 1.0,
                    },
                )
                optimization_mass[name].add(weight * slope_at_zero_student, seen, stride)
            for local, sample_id in enumerate(sample_ids[start : start + RECORD_CHUNK]):
                category = str(sample_id).split(":", 1)[0]
                per_pixel = x[local * IMAGE_SIZE * IMAGE_SIZE : (local + 1) * IMAGE_SIZE * IMAGE_SIZE]
                category_rows.setdefault(category, []).append(
                    [
                        float(per_pixel.mean()),
                        float(np.tanh(per_pixel).mean()),
                        float((per_pixel / (1.0 + per_pixel)).mean()),
                        float(np.mean(per_pixel >= 1.0)),
                    ]
                )
            seen += flat_effect.size

    raw_summary = raw_x.summary()
    candidate_summary: dict[str, Any] = {}
    for name in MAP_NAMES:
        candidate_summary[name] = {
            "weight": weights[name].summary(),
            "initial_student_zero_effect_gradient_mass_proxy": optimization_mass[name].summary(),
            "full_target_preserved": True,
            "target_definition": "E_t (unchanged for every candidate; only source-example importance changes)",
        }

    category_summary: dict[str, Any] = {}
    for category, rows in sorted(category_rows.items()):
        values = np.asarray(rows, dtype=np.float64)
        category_summary[category] = {
            "records": int(values.shape[0]),
            "mean_x": float(values[:, 0].mean()),
            "mean_tanh_weight": float(values[:, 1].mean()),
            "mean_rational_weight": float(values[:, 2].mean()),
            "mean_fraction_x_ge_1": float(values[:, 3].mean()),
        }

    return {
        "schema_version": "P35_SOURCE_ONLY_ACTIONABILITY_V1",
        "protocol_id": "P35",
        "status": "SOURCE_ONLY_COMPLETE",
        "cache_root": str(cache_root.resolve()),
        "source_shard": str(shard.resolve()),
        "held_class": held_class,
        "records": len(sample_ids),
        "pixels": total_pixels,
        "correction_scale_C": CORRECTION_SCALE,
        "sample": {"method": "deterministic_systematic_flat_index", "target": SAMPLE_TARGET, "stride": stride},
        "raw_normalized_effect_x": raw_summary,
        "fixed_points": _fixed_points(),
        "candidates": candidate_summary,
        "category_summary": category_summary,
        "source_only_safety": {
            "held_reads": 0,
            "new_clip_forwards": 0,
            "new_phase2b_forwards": 0,
            "new_teacher_forwards": 0,
            "cache_rebuilds": 0,
            "teacher_target_preserved": True,
            "category_specific_rule": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--held-class", default=DEFAULT_CLASS)
    args = parser.parse_args()
    print(json.dumps(analyze_source(args.cache_root, args.held_class), sort_keys=True))


if __name__ == "__main__":
    main()
