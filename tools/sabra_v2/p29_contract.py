"""Fail-closed frozen P29 protocol and cache bindings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from tools.sabra_v2.region_cache import CacheProvenance, sha256_file
from tools.sabra_v2.train_region_distill import ROOT

P29_PROTOCOL_PATH = ROOT / "research/sabra_v2/region_distill/P29_PROTOCOL.json"
P27_CACHE_EXECUTION_BASE = "de41b380449dcbc0b124f71f4f8fbb789e1a96f0"
CORRECTION_SCALE = 4.960109710693359


def p29_cache_provenance(metadata: Path) -> CacheProvenance:
    return CacheProvenance(P27_CACHE_EXECUTION_BASE, sha256_file(metadata))


def audit_p29_protocol(protocol: Mapping[str, Any]) -> dict[str, object]:
    if protocol.get("schema_version") != "P29_SABRA_SIGN_GUARDED_NORMALIZED_REGION_DISTILL_V1":
        raise RuntimeError("P29 schema drift")
    method = protocol.get("frozen_method", {})
    teacher = method.get("teacher", {})
    if (teacher.get("alpha"), teacher.get("margin_scale"), teacher.get("correction_scale_C")) != (0.25, 19.840438842773438, CORRECTION_SCALE):
        raise RuntimeError("P29 teacher or correction scale drift")
    geometry = method.get("geometry", {})
    if (geometry.get("patch_grid"), geometry.get("region_grid"), geometry.get("stages")) != ([37, 37], [9, 9], 3):
        raise RuntimeError("P29 geometry drift")
    loss = method.get("loss", {})
    if loss.get("total") != "L_value + L_sign + L_normal" or not loss.get("calculate_seg_loss_forbidden"):
        raise RuntimeError("P29 loss contract drift")
    training = protocol.get("frozen_training", {})
    if (training.get("epochs"), training.get("batch_size"), training.get("learning_rate"), training.get("seed")) != (20, 1, 0.001, 0):
        raise RuntimeError("P29 schedule drift")
    if training.get("only_trainable") != ["RegionResidualAdapter"]:
        raise RuntimeError("P29 trainable ownership drift")
    return {"status": "PASS", "protocol_sha256": sha256_file(P29_PROTOCOL_PATH), "cache_execution_base": P27_CACHE_EXECUTION_BASE}


def load_and_audit_p29_protocol() -> dict[str, object]:
    return audit_p29_protocol(json.loads(P29_PROTOCOL_PATH.read_text()))
