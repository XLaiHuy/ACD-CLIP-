"""Fail-closed bindings for the frozen P30R1 preregistration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tools.sabra.data import EXPECTED_VISA_CLASSES
from tools.sabra_v2.p29_contract import p29_cache_provenance
from tools.sabra_v2.p29_objective import CORRECTION_SCALE
from tools.sabra_v2.p30r1_objective import (
    P30R1_FORMULATION_HASH,
    P30R1_NORMALIZATION_EPSILON,
    P30R1_OBJECTIVE_NAME,
    P30R1_SMOOTH_L1_BETA,
)
from tools.sabra_v2.region_cache import sha256_file


ROOT = Path(__file__).resolve().parents[2]
P30R1_PREREGISTRATION_PATH = ROOT / "research/sabra_v2/region_distill/P30R1_PREREGISTRATION.json"
P30R1_PREREGISTRATION_SHA_PATH = ROOT / "research/sabra_v2/region_distill/P30R1_PREREGISTRATION_SHA256.txt"
P30R1_UUID = "7374c95c-2ada-41f3-89e7-24d7e48338af"
P30R1_BRANCH = "research/p29r1-fast-objective-forensic-v1"
P30R1_CLASS_ORDER = tuple(EXPECTED_VISA_CLASSES)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def p30r1_preregistration_hash(path: Path = P30R1_PREREGISTRATION_PATH) -> str:
    return sha256_file(path)


def _first_hash_from_manifest(path: Path) -> str:
    fields = path.read_text(encoding="utf-8").split()
    if not fields:
        raise RuntimeError(f"empty P30R1 hash manifest: {path}")
    return fields[0]


def load_and_audit_p30r1_preregistration(
    path: Path = P30R1_PREREGISTRATION_PATH,
    expected_hash: str | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != "P30R1_TEACHER_RELATIVE_RADIAL_STABILIZATION_V1":
        raise RuntimeError("P30R1 preregistration schema drift")
    if payload.get("status") != "P30R1_PREREGISTERED_DESIGN_ONLY":
        raise RuntimeError("P30R1 preregistration status drift")
    experiment = payload.get("experiment", {})
    if experiment.get("uuid") != P30R1_UUID or experiment.get("branch") != P30R1_BRANCH:
        raise RuntimeError("P30R1 experiment identity drift")
    objective = payload.get("objective", {})
    if (
        objective.get("name") != P30R1_OBJECTIVE_NAME
        or objective.get("objective_count") != 1
        or objective.get("same_teacher_denominator") is not True
        or objective.get("student_self_normalization") is not False
        or objective.get("exact_zero_teacher_active") is not True
        or objective.get("additional_losses") != []
        or objective.get("formulation_hash") != P30R1_FORMULATION_HASH
    ):
        raise RuntimeError("P30R1 objective contract drift")
    if tuple(payload.get("class_order", ())) != P30R1_CLASS_ORDER:
        raise RuntimeError("P30R1 class order drift")
    scalars = payload.get("frozen_scalars", {})
    if (
        float(scalars.get("correction_scale_C", {}).get("value", -1.0)) != CORRECTION_SCALE
        or float(scalars.get("normalization_epsilon", {}).get("value", -1.0)) != P30R1_NORMALIZATION_EPSILON
        or float(scalars.get("smooth_l1_beta", {}).get("value", -1.0)) != P30R1_SMOOTH_L1_BETA
    ):
        raise RuntimeError("P30R1 scalar contract drift")
    training = payload.get("optimizer_and_training", {})
    if (
        training.get("epochs"),
        training.get("batch_size"),
        training.get("learning_rate"),
        training.get("seed"),
    ) != (20, 1, 0.001, 0):
        raise RuntimeError("P30R1 training schedule drift")
    stage2 = payload.get("stage_protocol", {}).get("stage_2_one_class", {})
    if stage2.get("class") != "candle" or stage2.get("fit_records") != 1962 or stage2.get("held_records") != 200:
        raise RuntimeError("P30R1 Stage 2 identity drift")
    observed_hash = p30r1_preregistration_hash(path)
    if _first_hash_from_manifest(P30R1_PREREGISTRATION_SHA_PATH) != observed_hash:
        raise RuntimeError("P30R1 preregistration external hash mismatch")
    if expected_hash is not None and observed_hash != expected_hash:
        raise RuntimeError("P30R1 preregistration hash does not match execution input")
    return payload


def p30r1_cache_provenance(metadata: Path):
    """Reuse the exact frozen P27 cache provenance contract."""
    return p29_cache_provenance(metadata)


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected mapping for {name}")
    return value
