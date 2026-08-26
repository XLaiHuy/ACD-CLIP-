"""P29R1 zero-training forensic helpers.

The helpers in this module are deliberately small and cache-only.  The one
historical operation that needs autograd is the R0 utility computation; its
caller is wrapped in a local ``enable_grad`` scope and all returned values are
detached before they leave the helper.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from tools.sabra_car.r0_direction import utility_for_batch
from tools.sabra_v2.p28_mechanism_diagnostic import alignment_metrics
from tools.sabra_v2.p29_objective import source_pure_normal_regions

CLASS_NAMES = ("candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")


def _as_finite_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array

def prediction_hash(path: Path, expected: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    observed = digest.hexdigest()
    if observed != expected:
        raise RuntimeError(f"frozen artifact hash mismatch: {path}")
    return observed


def forensic_utility_for_batch(native: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        utility, loss = utility_for_batch(native.detach(), mask.detach())
    return utility.detach(), loss.detach()


def sign_alignment(teacher: np.ndarray, student: np.ndarray) -> dict[str, float | None]:
    return alignment_metrics(_as_finite_array(teacher, "teacher"), _as_finite_array(student, "student"))

_PROBE_POSITIONS = (0, 3, 6, 9)

def _quantile(values: np.ndarray, q: float) -> float | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return None
    return float(np.quantile(values, q, method="linear"))


def residual_magnitude_summary(residual: np.ndarray) -> dict[str, float | None]:
    values = np.abs(_as_finite_array(np.asarray(residual, dtype=np.float64), "residual").reshape(-1))
    return {
        "mean_abs": float(values.mean()) if values.size else None,
        "median_abs": _quantile(values, 0.5),
        "q90_abs": _quantile(values, 0.9),
        "q99_abs": _quantile(values, 0.99),
    }


def _flatten_gradients(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    values = [value.detach().to(dtype=torch.float64, device="cpu").reshape(-1) for value in tensors]
    return torch.cat(values) if values else torch.zeros(0, dtype=torch.float64)


def gradient_summary(groups: Mapping[str, Iterable[torch.Tensor]]) -> dict[str, dict[str, float | None]]:
    vectors = {name: _flatten_gradients(values) for name, values in groups.items()}
    norms = {name: float(torch.linalg.vector_norm(value).item()) for name, value in vectors.items()}
    cosines: dict[str, float | None] = {}
    for left, left_value in vectors.items():
        for right, right_value in vectors.items():
            if left >= right:
                continue
            denominator = norms[left] * norms[right]
            cosines[f"{left}__{right}"] = None if denominator == 0.0 else float(torch.dot(left_value, right_value) / denominator)
    return {"norms": norms, "cosines": cosines}


def normal_guard_conflict(source_mask: torch.Tensor, teacher_region: torch.Tensor) -> dict[str, float | int | None]:
    pure = source_pure_normal_regions(source_mask).squeeze(1)
    teacher = teacher_region.detach().to(dtype=torch.float32)
    if not torch.isfinite(teacher).all():
        raise ValueError("teacher_region contains non-finite values")
    if teacher.ndim != 3 or teacher.shape != pure.shape:
        raise ValueError("teacher_region must be [B,9,9] aligned with source masks")
    values = teacher[pure]
    count = int(values.numel())
    if not count:
        return {"pure_normal_region_count": 0, "pure_normal_region_fraction": 0.0, "teacher_positive_fraction": None, "teacher_zero_fraction": None, "teacher_negative_fraction": None, "positive_strength_mass": None}
    absolute = values.abs().sum()
    return {"pure_normal_region_count": count, "pure_normal_region_fraction": float(pure.float().mean()), "teacher_positive_fraction": float((values > 0).float().mean()), "teacher_zero_fraction": float((values == 0).float().mean()), "teacher_negative_fraction": float((values < 0).float().mean()), "positive_strength_mass": float(values.clamp_min(0).sum() / absolute) if absolute.item() else 0.0}


def vectorized_pixel_shifts(native: np.ndarray, state: np.ndarray, mask: np.ndarray) -> dict[str, dict[str, float | None]]:
    baseline = _as_finite_array(np.asarray(native, dtype=np.float32), "native").reshape(-1)
    candidate = _as_finite_array(np.asarray(state, dtype=np.float32), "state").reshape(-1)
    labels = np.asarray(mask, dtype=np.uint8).reshape(-1).astype(bool)
    if baseline.shape != candidate.shape or baseline.shape != labels.shape:
        raise ValueError("native, state, and mask must have matching pixel counts")
    shifts = candidate - baseline
    normal, anomaly = shifts[~labels], shifts[labels]
    return {"normal": {"mean": float(normal.mean()) if normal.size else None, "median": _quantile(normal, 0.5), "q95": _quantile(normal, 0.95), "q99": _quantile(normal, 0.99)}, "anomaly": {"mean": float(anomaly.mean()) if anomaly.size else None, "median": _quantile(anomaly, 0.5)}}


def select_probe_source_classes(held_class: str) -> tuple[str, str, str, str]:
    if held_class not in CLASS_NAMES:
        raise ValueError(f"unknown held class: {held_class}")
    sources = tuple(name for name in CLASS_NAMES if name != held_class)
    return tuple(sources[index] for index in _PROBE_POSITIONS)  # type: ignore[return-value]


def estimate_forensic_runtime(*, seconds_per_class: float, classes: int, fixed_seconds: float = 0.0) -> dict[str, float | str]:
    if seconds_per_class < 0 or classes <= 0 or fixed_seconds < 0:
        raise ValueError("runtime inputs must be non-negative with a positive class count")
    projected_seconds = float(seconds_per_class) * int(classes) + float(fixed_seconds)
    projected_minutes = projected_seconds / 60.0
    if projected_minutes <= 45.0:
        decision = "PROCEED"
    elif projected_minutes <= 90.0:
        decision = "OPTIMIZE_ONCE"
    else:
        decision = "PERFORMANCE_STOP"
    return {"seconds_per_class": float(seconds_per_class), "classes": int(classes), "fixed_seconds": float(fixed_seconds), "projected_seconds": projected_seconds, "projected_minutes": projected_minutes, "decision": decision}


def validate_forensic_path_source() -> None:
    source = inspect.getsource(inspect.getmodule(validate_forensic_path_source))
    forbidden = ("torch" + ".optim", "optimizer" + ".step", "clip" + ".load", "Phase" + "2B(", "MV" + "Tec", "Med" + "ical")
    found = [token for token in forbidden if token in source]
    if found:
        raise RuntimeError(f"forbidden P29R1 execution path: {found}")
