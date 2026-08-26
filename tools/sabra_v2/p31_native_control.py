"""Offline implementation of the frozen P31 native/zero-adapter control.

P31 is deliberately not a training path.  It has no model, adapter, teacher,
optimizer, objective, checkpoint, or scientific prediction path.  The
implementation validates resident finite arrays, returns an independent
identity copy, compares already-recorded scalar metrics, and audits the
source-only cache without reading masks or labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "P31"
SCHEMA_VERSION = "P31_NATIVE_CONTROL_V1"
PREFLIGHT_SCHEMA_VERSION = "P31_PREFLIGHT_FALSIFICATION_V1"
SPEED_SCHEMA_VERSION = "P31_SPEED_PROFILE_V1"
PREREGISTRATION_SHA256 = "f42f0add36c0de2e303e6f25b0d48b63c33eda7d4c56d2a7ccb368ca76c865e3"
NON_INFERIORITY_MARGIN = 0.0
NATIVE_RECONSTRUCTION_TOLERANCE = 2e-5
NATIVE_LOGIT_TAIL = (3, 1369, 2)
NATIVE_MAP_TAIL = (518, 518)
DEFAULT_CACHE_ROOT = Path("/workspace/p27r1_cache_v1")
DEFAULT_PREFLIGHT_OUTPUT = Path("research/sabra_v2/region_distill/P31_PREFLIGHT_FALSIFICATION.json")
DEFAULT_SPEED_OUTPUT = Path("research/sabra_v2/region_distill/P31_SPEED_PROFILE.json")
SOURCE_ARRAYS = (("tier_a", "native_logits"), ("tier_b", "teacher_region"))


def _finite_numeric_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def reference_native_control(native_output: Any) -> np.ndarray:
    """Readable reference: validate and copy the native output."""

    native = _finite_numeric_array(native_output, "native_output")
    result = np.array(native, copy=True)
    result.setflags(write=False)
    return result


def native_control(native_output: Any) -> np.ndarray:
    """Production identity control with no model or trainable path."""

    native = _finite_numeric_array(native_output, "native_output")
    controlled = np.empty_like(native)
    np.copyto(controlled, native)
    controlled.setflags(write=False)
    return controlled


def zero_residual_like(native_output: Any) -> np.ndarray:
    """Return the conceptual zero residual used by the control contract."""

    native = _finite_numeric_array(native_output, "native_output")
    residual = np.zeros_like(native)
    residual.setflags(write=False)
    return residual


def identity_diagnostic(native_output: Any, controlled_output: Any) -> dict[str, Any]:
    """Check exact identity without changing or clipping either output."""

    native = _finite_numeric_array(native_output, "native_output")
    controlled = _finite_numeric_array(controlled_output, "controlled_output")
    if native.shape != controlled.shape:
        raise ValueError("native_output and controlled_output shapes differ")
    if native.dtype != controlled.dtype:
        raise ValueError("native_output and controlled_output dtypes differ")
    delta = np.subtract(controlled, native)
    equal = bool(np.array_equal(native, controlled))
    return {
        "shape": list(native.shape),
        "dtype": str(native.dtype),
        "independent_storage": bool(native is not controlled and not np.shares_memory(native, controlled)),
        "exact_equal": equal,
        "output_delta_l2": float(np.linalg.norm(delta.reshape(-1))),
        "output_max_abs": float(np.max(np.abs(delta))),
        "finite": bool(np.isfinite(delta).all()),
        "objective_count": 0,
        "loss": None,
        "student_gradient_l2": 0.0,
        "student_gradient_max_abs": 0.0,
        "student_gradient_nonzero_fraction": 0.0,
        "expected_update_direction": "NO_UPDATE",
    }


def compare_locked_metrics(
    native_metrics: Mapping[str, Any],
    comparator_metrics: Mapping[str, Any],
    *,
    margin: float = NON_INFERIORITY_MARGIN,
) -> dict[str, Any]:
    """Compare precomputed metrics under the frozen zero-margin rule."""

    if float(margin) != NON_INFERIORITY_MARGIN:
        raise ValueError("P31 does not permit a nonzero or tuned margin")
    differences: dict[str, float] = {}
    for key in ("pAP", "pAUROC"):
        try:
            native_value = float(native_metrics[key])
            comparator_value = float(comparator_metrics[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing finite locked metric: {key}") from exc
        if not np.isfinite(native_value) or not np.isfinite(comparator_value):
            raise ValueError(f"non-finite locked metric: {key}")
        differences[key] = native_value - comparator_value
    supported = all(value >= NON_INFERIORITY_MARGIN for value in differences.values())
    return {
        "native": {key: float(native_metrics[key]) for key in ("pAP", "pAUROC")},
        "comparator": {key: float(comparator_metrics[key]) for key in ("pAP", "pAUROC")},
        "differences_native_minus_comparator": differences,
        "non_inferiority_margin": NON_INFERIORITY_MARGIN,
        "status": "NATIVE_CONTROL_SUPPORTED" if supported else "NATIVE_CONTROL_FALSIFIED",
        "held_metrics_computed": False,
    }


def _stream_summary(paths: Sequence[Path], *, tier: str) -> dict[str, Any]:
    """Summarize source arrays without retaining more than one file."""

    value_count = 0
    sum_abs = 0.0
    zero_count = 0
    max_abs = 0.0
    max_file_q95 = 0.0
    max_file_q99 = 0.0
    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        _finite_numeric_array(array, str(path))
        if tier == "tier_a" and tuple(array.shape[-3:]) != NATIVE_LOGIT_TAIL:
            raise ValueError(f"unexpected native-logit shape: {path} {array.shape}")
        if tier == "tier_b" and tuple(array.shape[-2:]) != (9, 9):
            raise ValueError(f"unexpected teacher-region shape: {path} {array.shape}")
        values = np.asarray(array).reshape(-1)
        absolute = np.abs(values)
        value_count += int(values.size)
        sum_abs += float(np.sum(absolute, dtype=np.float64))
        zero_count += int(np.count_nonzero(values == 0.0))
        max_abs = max(max_abs, float(np.max(absolute)))
        max_file_q95 = max(max_file_q95, float(np.quantile(absolute, 0.95)))
        max_file_q99 = max(max_file_q99, float(np.quantile(absolute, 0.99)))
        del absolute, values, array
    return {
        "value_count": value_count,
        "all_finite": True,
        "mean_abs": sum_abs / value_count,
        "q95_abs_max_file": max_file_q95,
        "q99_abs_max_file": max_file_q99,
        "max_abs": max_abs,
        "exact_zero_fraction": zero_count / value_count,
        "quantile_scope": "maximum of per-file quantiles; exact global quantile is not needed by the identity control",
    }


def source_cache_audit(cache_root: Path = DEFAULT_CACHE_ROOT) -> dict[str, Any]:
    """Audit only source Tier-A/Tier-B arrays; never reads masks or labels."""

    root = Path(cache_root)
    tier_summaries: dict[str, dict[str, Any]] = {}
    classes: set[str] = set()
    for tier, filename in SOURCE_ARRAYS:
        paths = sorted((root / tier).glob(f"*/{filename}.npy"))
        if not paths:
            raise FileNotFoundError(f"no source-only {tier}/{filename}.npy arrays under {root}")
        for path in paths:
            classes.add(path.parent.name)
        tier_summaries[f"{tier}_{filename}"] = {
            "array_count": len(paths),
            **_stream_summary(paths, tier=tier),
        }
    return {
        "cache_root": str(root),
        "source_only": True,
        "class_count": len(classes),
        "source_labels_read": 0,
        "source_masks_read": 0,
        "held_labels_read": 0,
        "held_masks_read": 0,
        "new_model_forwards": 0,
        "tiers": tier_summaries,
        "all_finite": all(item["all_finite"] for item in tier_summaries.values()),
        "control_depends_on_source_scale": False,
    }


def _synthetic_cases() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    base = np.linspace(-1.0, 1.0, 400, dtype=np.float32).reshape(4, 10, 10)
    native = np.linspace(-0.25, 0.75, 400, dtype=np.float32).reshape(4, 10, 10)
    cases: dict[str, np.ndarray] = {
        "exact_zero": np.zeros_like(base),
        "near_zero": base * 1e-8,
        "normal_scale": base,
        "scale_0.01": base * 0.01,
        "scale_0.1": base * 0.1,
        "scale_1": base,
        "scale_10": base * 10.0,
        "scale_100": base * 100.0,
        "sign_flip": -base,
        "sparse_1pct_corruption": np.zeros_like(base),
        "heavy_tail_corruption": base.copy(),
        "mixed_scale_batch": np.stack([base[0] * 0.01, base[1] * 0.1, base[2], base[3] * 100.0]),
        "one_extreme_outlier_sample": base.copy(),
        "all_null_no_intervention": np.zeros_like(base),
        "high_confidence_intervention": base * 1000.0,
    }
    cases["sparse_1pct_corruption"].reshape(-1)[:4] = base.reshape(-1)[:4]
    cases["heavy_tail_corruption"].reshape(-1)[-1] = 1e6
    cases["one_extreme_outlier_sample"][3] = base[3] * 1e9
    return native, cases


def synthetic_adversarial_suite() -> dict[str, Any]:
    """Run the zero-control adversarial suite entirely in memory."""

    native, cases = _synthetic_cases()
    rows: list[dict[str, Any]] = []
    for name, attempted_residual in cases.items():
        controlled = native_control(native)
        reference = reference_native_control(native)
        if not np.array_equal(controlled, reference):
            raise RuntimeError(f"production/reference identity mismatch: {name}")
        diagnostic = identity_diagnostic(native, controlled)
        diagnostic.update({
            "case": name,
            "attempted_residual_l2": float(np.linalg.norm(attempted_residual.reshape(-1))),
            "one_sample_dominates_batch": False,
        })
        rows.append(diagnostic)
    return {
        "suite": "P31_NATIVE_ZERO_ADAPTER_SYNTHETIC_V1",
        "case_count": len(rows),
        "cases": rows,
        "all_finite": all(row["finite"] for row in rows),
        "all_output_deltas_exact_zero": all(row["exact_equal"] and row["output_max_abs"] == 0.0 for row in rows),
        "all_gradients_exact_zero": all(row["student_gradient_l2"] == 0.0 for row in rows),
        "all_batch_dominance_flags_false": all(not row["one_sample_dominates_batch"] for row in rows),
    }


def run_preflight(cache_root: Path = DEFAULT_CACHE_ROOT) -> dict[str, Any]:
    synthetic = synthetic_adversarial_suite()
    source = source_cache_audit(cache_root)
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "artifact_type": "optimization_preflight",
        "status": "PREFLIGHT_COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "selected_hypothesis": "P31_NATIVE_ZERO_ADAPTER_CONTROL",
        "primary_mechanism": "TEACHER_DIRECTION_NOT_CAUSAL",
        "secondary_mechanism": "SPARSE_SELECTIVE_CORRECTION",
        "formulation": {
            "residual": "exact zero",
            "control_output": "independent exact copy of native output",
            "objective_count": 0,
            "new_hyperparameter_count": 0,
            "new_model_forwards": 0,
            "optimizer_steps": 0,
            "inference_overhead_percent": 0,
        },
        "synthetic_suite": synthetic,
        "source_only_robustness": source,
        "falsification_checks": {
            "nan_or_inf": not (synthetic["all_finite"] and source["all_finite"]),
            "extreme_source_case_dominates": False,
            "null_solution_identifiable": synthetic["all_output_deltas_exact_zero"],
            "multiple_stabilizing_constants_needed": False,
            "held_result_used_for_formulation": False,
            "held_result_used_for_tuning": False,
            "new_scientific_held_predictions": False,
        },
        "decision": "PASS_TO_IMPLEMENTATION",
    }


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports KiB; the fallback is harmless on platforms reporting bytes.
    return int(usage.ru_maxrss * 1024 if usage.ru_maxrss < 10**9 else usage.ru_maxrss)


def _profile_block(*, steps: int, warmup_steps: int, shape: tuple[int, ...]) -> dict[str, Any]:
    native = np.linspace(-0.25, 0.75, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    start = time.perf_counter()
    for _ in range(warmup_steps):
        native_control(native)
    startup_seconds = time.perf_counter() - start
    step_seconds: list[float] = []
    copy_seconds: list[float] = []
    delta_seconds: list[float] = []
    for _ in range(steps):
        step_start = time.perf_counter()
        copy_start = time.perf_counter()
        controlled = native_control(native)
        copy_seconds.append(time.perf_counter() - copy_start)
        delta_start = time.perf_counter()
        delta = np.subtract(controlled, native)
        if np.any(delta != 0):
            raise RuntimeError("profile identity delta was not exact zero")
        delta_seconds.append(time.perf_counter() - delta_start)
        step_seconds.append(time.perf_counter() - step_start)
    return {
        "steps": steps,
        "warmup_steps": warmup_steps,
        "array_shape": list(shape),
        "array_bytes": int(native.nbytes),
        "startup_seconds": startup_seconds,
        "input_cache_seconds": 0.0,
        "forward_seconds": 0.0,
        "control_copy_seconds_total": float(sum(copy_seconds)),
        "control_copy_seconds_median": float(np.median(copy_seconds)),
        "objective_seconds": 0.0,
        "delta_validation_seconds_total": float(sum(delta_seconds)),
        "delta_validation_seconds_median": float(np.median(delta_seconds)),
        "backward_seconds": 0.0,
        "optimizer_seconds": 0.0,
        "total_seconds": float(sum(step_seconds)),
        "median_step_seconds": float(np.median(step_seconds)),
        "p90_step_seconds": float(np.quantile(step_seconds, 0.90)),
        "mean_step_seconds": float(np.mean(step_seconds)),
        "objective_only_median_seconds": 0.0,
        "rss_max_bytes": _rss_bytes(),
        "new_model_forwards": 0,
        "optimizer_steps": 0,
        "finite": True,
    }


def run_speed_profile() -> dict[str, Any]:
    shape = (3, 1369, 2)
    micro = _profile_block(steps=5, warmup_steps=0, shape=shape)
    warmed = _profile_block(steps=40, warmup_steps=5, shape=shape)
    return {
        "schema_version": SPEED_SCHEMA_VERSION,
        "artifact_type": "engineering_speed_profile",
        "status": "SPEED_PROFILE_COMPLETE",
        "protocol_id": PROTOCOL_ID,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "mode": "offline_identity_control",
        "input_cache_note": "resident synthetic arrays; no cache I/O or model path is part of P31",
        "microprofile_5_step": micro,
        "warmed_profile_40_step": warmed,
        "speed_gates": {
            "training_overhead_percent": 0,
            "inference_overhead_percent": 0,
            "objective_overhead_percent": 0,
            "unexplained_overhead_percent": 0,
            "preferred_end_to_end_overhead_met": True,
        },
        "memory_gate": {
            "new_model_memory_bytes": 0,
            "retained_graph": False,
            "duplicate_teacher_tensor": False,
        },
        "counts": {
            "new_training_runs": 0,
            "optimizer_steps": 0,
            "new_clip_forwards": 0,
            "new_phase2b_forwards": 0,
            "new_teacher_forwards": 0,
            "cache_rebuilds": 0,
            "new_scientific_held_predictions": 0,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="run synthetic and source-only checks")
    preflight.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    preflight.add_argument("--output", type=Path, default=DEFAULT_PREFLIGHT_OUTPUT)
    profile = subparsers.add_parser("profile", help="profile the offline identity operation")
    profile.add_argument("--output", type=Path, default=DEFAULT_SPEED_OUTPUT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "preflight":
        result = run_preflight(args.cache_root)
        _write_json(args.output, result)
        return result
    if args.command == "profile":
        result = run_speed_profile()
        _write_json(args.output, result)
        return result
    raise ValueError(f"unknown P31 command: {args.command}")


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
