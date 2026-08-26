"""Fail-closed P30 preregistration and cache bindings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tools.sabra.data import EXPECTED_VISA_CLASSES
from tools.sabra_v2.p29_contract import p29_cache_provenance
from tools.sabra_v2.region_cache import sha256_file
from tools.sabra_v2.train_region_distill import ROOT


P30_PREREGISTRATION_PATH = ROOT / "research/sabra_v2/region_distill/P30_PREREGISTRATION.json"
P30_PREREGISTRATION_SHA_PATH = ROOT / "research/sabra_v2/region_distill/P30_PREREGISTRATION_SHA256.txt"
P30_OUTPUT_ROOT = ROOT / "research/sabra_v2/region_distill/P30"
P30_UUID = "71a16efe-2388-458a-9106-bd87f882805a"
P30_CLASS_ORDER = tuple(EXPECTED_VISA_CLASSES)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def p30_preregistration_hash(path: Path = P30_PREREGISTRATION_PATH) -> str:
    return sha256_file(path)


def load_and_audit_p30_preregistration(
    path: Path = P30_PREREGISTRATION_PATH,
    expected_hash: str | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != "P30_SPEED_PERFORMANCE_DIRECTIONAL_DISTILLATION_V1":
        raise RuntimeError("P30 preregistration schema drift")
    if payload.get("status") != "P30_PREREGISTERED":
        raise RuntimeError("P30 preregistration is not frozen")
    experiment = payload.get("experiment", {})
    if experiment.get("uuid") != P30_UUID:
        raise RuntimeError("P30 experiment UUID drift")
    if experiment.get("branch") != "research/p29r1-fast-objective-forensic-v1":
        raise RuntimeError("P30 branch identity drift")
    if tuple(payload.get("frozen_components", {}).get("class_order", ())) != P30_CLASS_ORDER:
        raise RuntimeError("P30 class order drift")
    objective = payload.get("objective", {})
    if objective.get("name") != "P30_DIRECTIONAL_COSINE_V1":
        raise RuntimeError("P30 objective name drift")
    if objective.get("one_primary_objective_only") is not True or objective.get("additional_losses") != []:
        raise RuntimeError("P30 objective stacking drift")
    if float(objective.get("correction_scale_C", -1.0)) != 4.960109710693359:
        raise RuntimeError("P30 correction scale drift")
    if float(objective.get("normalization_epsilon", -1.0)) != 0.01:
        raise RuntimeError("P30 normalization epsilon drift")
    training = payload.get("training", {})
    if (
        training.get("epochs"),
        training.get("batch_size"),
        training.get("learning_rate"),
        training.get("seed"),
    ) != (20, 1, 0.001, 0):
        raise RuntimeError("P30 training schedule drift")
    if payload.get("trainable_components") != ["RegionResidualAdapter"]:
        raise RuntimeError("P30 trainable ownership drift")
    observed_hash = p30_preregistration_hash(path)
    hash_path = path.with_name("P30_PREREGISTRATION_SHA256.txt")
    hash_fields = hash_path.read_text(encoding="utf-8").split()
    if not hash_fields or hash_fields[0] != observed_hash:
        raise RuntimeError("P30 preregistration external hash mismatch")
    if expected_hash is not None and observed_hash != expected_hash:
        raise RuntimeError("P30 preregistration hash does not match execution input")
    return payload


def p30_cache_provenance(metadata: Path):
    """Reuse the exact P27 cache provenance contract; do not rebuild caches."""
    return p29_cache_provenance(metadata)


def p30_json(path: Path) -> dict[str, Any]:
    return _read_json(path)
