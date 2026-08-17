#!/usr/bin/env python3
"""P5FR1C GT-free derivation from the immutable P5FR1 common snapshot.

This tool never imports a model loader, opens an image, reads GT, or runs a
forward. It applies only the 26 frozen pure geometry transforms to c/G/valid
arrays from the preserved common records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys_tools = ROOT / "tools"
import sys
sys.path[:0] = [str(ROOT), str(sys_tools)]
from p5f_geometry import asr, csrc, pcrr, pgm  # noqa: E402

NAMESPACE = ROOT / "runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION"
SNAPSHOT_ROOT = Path("/workspace/P5FR1_LATE_COMPLETION_SNAPSHOT")
RUNTIME_ROOT = Path("/tmp/p5fr1c_all_config_evidence")
STATUS_PATH = RUNTIME_ROOT / "DERIVE_STATUS.json"
EXPECTED_RECORDS = 1725
PATCH_COUNT = 1369
CONFIG_COUNT = 26
FAMILIES = ("PCRR", "CSRC", "ASR", "PGM")
MODULES = {"PCRR": pcrr, "CSRC": csrc, "ASR": asr, "PGM": pgm}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        f.write(json_bytes(value)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def record_name(identity: dict[str, Any]) -> str:
    key = f"{identity['canonical_index']}|{identity['class_name']}|{identity['image_path']}".encode()
    return f"{int(identity['canonical_index']):05d}_{hashlib.sha256(key).hexdigest()[:16]}.npz"


def load_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    lock = json.loads((NAMESPACE / "INPUT_LOCK.json").read_text())
    if lock.get("late_completion_validated") is not True or lock.get("model_forwards_in_p5fr1c") != 0:
        raise RuntimeError("P5FR1C_INPUT_LOCK_INVALID")
    identities = json.loads(Path("/workspace/P5F_MVTEC_CANONICAL_IDENTITIES.json").read_text())["identities"]
    if len(identities) != EXPECTED_RECORDS:
        raise RuntimeError("P5FR1C_INPUT_INVALID: identity count")
    configs_doc = json.loads((NAMESPACE / "CANONICAL_CONFIGS.json").read_text())
    rows = [(family, config) for family in FAMILIES for config in configs_doc["families"][family]]
    ids = [config["config_id"] for _, config in rows]
    if len(rows) != CONFIG_COUNT or len(set(ids)) != CONFIG_COUNT:
        raise RuntimeError("P5FR1C_CONFIG_INVALID")
    if ids != [x["config_id"] for family in FAMILIES for x in configs_doc["families"][family]]:
        raise RuntimeError("P5FR1C_CONFIG_ORDER_INVALID")
    return lock, identities, rows


def validate_output(path: Path, identity: dict[str, Any], config_ids: list[str], source_sha: str) -> None:
    with np.load(path, allow_pickle=False) as d:
        required = {"canonical_index", "class_name", "image_path", "config_ids", "evidence", "valid_reference", "source_record_sha256"}
        if set(d.files) != required:
            raise RuntimeError(f"P5FR1C_DERIVED_SCHEMA_INVALID: {path.name}")
        if int(d["canonical_index"]) != int(identity["canonical_index"]):
            raise RuntimeError(f"P5FR1C_IDENTITY_INVALID: {path.name}")
        if str(d["class_name"]) != identity["class_name"] or str(d["image_path"]) != identity["image_path"]:
            raise RuntimeError(f"P5FR1C_IDENTITY_INVALID: {path.name}")
        if list(d["config_ids"].astype(str)) != config_ids or d["evidence"].shape != (CONFIG_COUNT, PATCH_COUNT):
            raise RuntimeError(f"P5FR1C_CONFIG_SCHEMA_INVALID: {path.name}")
        if d["valid_reference"].shape != (PATCH_COUNT,) or d["valid_reference"].dtype != np.bool_:
            raise RuntimeError(f"P5FR1C_VALID_SCHEMA_INVALID: {path.name}")
        if not np.all(np.isfinite(d["evidence"])) or str(d["source_record_sha256"]) != source_sha:
            raise RuntimeError(f"P5FR1C_DERIVED_VALUE_INVALID: {path.name}")
        if np.any(d["evidence"][:, ~d["valid_reference"]] != 0.0):
            raise RuntimeError(f"P5FR1C_INVALID_REFERENCE_NONZERO: {path.name}")


def derive() -> dict[str, Any]:
    lock, identities, config_rows = load_inputs()
    if RUNTIME_ROOT.exists():
        raise RuntimeError("P5FR1C_DERIVED_ROOT_EXISTS: refusing overwrite")
    RUNTIME_ROOT.mkdir(parents=True)
    record_root = RUNTIME_ROOT / "records"
    record_root.mkdir()
    config_ids = [config["config_id"] for _, config in config_rows]
    source_hashes = lock["common_record_sha256"]
    start = time.time(); start_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    status = {"schema_version":"P5FR1C_DERIVE_STATUS_V1","state":"RUNNING","start_time_iso":start_iso,"start_unix_seconds":start,"completed_records":0,"model_forwards":0,"gt_accessed":False,"mask_accessed":False,"training_steps":0,"medical":False}
    atomic_json(STATUS_PATH, status)
    evidence_hashes = {}
    for index, identity in enumerate(identities):
        name = record_name(identity)
        source = SNAPSHOT_ROOT / "records" / name
        if not source.is_file() or sha256_file(source) != source_hashes.get(name):
            raise RuntimeError(f"P5FR1C_SOURCE_HASH_INVALID: {name}")
        with np.load(source, allow_pickle=False) as d:
            c = np.asarray(d["query_peer_cos"])
            G = np.asarray(d["peer_gram_upper"])
            valid = np.asarray(d["valid_reference"])
        evidence = np.zeros((CONFIG_COUNT, PATCH_COUNT), dtype=np.float32)
        for j, (family, config) in enumerate(config_rows):
            result = MODULES[family].transform(c, G, valid, config)
            evidence[j] = np.asarray(result["final"], dtype=np.float32)
        if not np.all(np.isfinite(evidence)):
            raise RuntimeError(f"P5FR1C_DERIVED_NONFINITE: {name}")
        target = record_root / name
        atomic_npz(target, {"canonical_index":np.asarray(identity["canonical_index"],dtype=np.int64),"class_name":np.asarray(identity["class_name"]),"image_path":np.asarray(identity["image_path"]),"config_ids":np.asarray(config_ids),"evidence":evidence,"valid_reference":valid.astype(bool),"source_record_sha256":np.asarray(source_hashes[name])})
        validate_output(target, identity, config_ids, source_hashes[name])
        evidence_hashes[name] = sha256_file(target)
        status["completed_records"] = index + 1
        # This is durable derivation accounting, not a model-forward state.
        atomic_json(STATUS_PATH, status)
    end = time.time(); end_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    aggregate = hashlib.sha256(json_bytes({"source_record_hashes":source_hashes,"all_config_evidence_hashes":evidence_hashes})).hexdigest()
    status.update({"state":"FINISHED","end_time_iso":end_iso,"end_unix_seconds":end,"elapsed_seconds":end-start,"completed_records":EXPECTED_RECORDS,"evidence_hashes":evidence_hashes,"aggregate_hash":aggregate})
    atomic_json(STATUS_PATH, status)
    return {"status":"PASS","records":EXPECTED_RECORDS,"config_count":CONFIG_COUNT,"model_forwards":0,"gt_accessed":False,"mask_accessed":False,"aggregate_hash":aggregate,"elapsed_seconds":end-start}


def freeze_manifest() -> dict[str, Any]:
    lock, identities, config_rows = load_inputs()
    status = json.loads(STATUS_PATH.read_text())
    if status.get("state") != "FINISHED" or status.get("completed_records") != EXPECTED_RECORDS or status.get("model_forwards") != 0:
        raise RuntimeError("P5FR1C_DERIVED_INCOMPLETE")
    config_ids = [config["config_id"] for _, config in config_rows]
    hashes = {}
    for identity in identities:
        path=RUNTIME_ROOT/"records"/record_name(identity)
        validate_output(path, identity, config_ids, lock["common_record_sha256"][path.name])
        hashes[path.name]=sha256_file(path)
    aggregate=hashlib.sha256(json_bytes({"source_record_hashes":lock["common_record_sha256"],"all_config_evidence_hashes":hashes})).hexdigest()
    manifest={"schema_version":"P5FR1C_GT_FREE_DERIVED_MANIFEST_V1","source_snapshot":"/workspace/P5FR1_LATE_COMPLETION_SNAPSHOT","original_process_implementation_sha":lock["frozen_implementation_sha"],"common_record_aggregate_hash":lock["common_record_aggregate_sha256"],"all_config_record_hashes":hashes,"all_config_aggregate_hash":aggregate,"config_ids":config_ids,"config_count":CONFIG_COUNT,"image_count":EXPECTED_RECORDS,"model_forwards":0,"images_opened":0,"labels_read":0,"masks_read":0,"GT_metrics_read":False,"training_steps":0,"medical":False,"finalized":True}
    atomic_json(NAMESPACE/"GT_FREE_DERIVED_MANIFEST.json",manifest)
    return {"status":"PASS","aggregate_hash":aggregate,"records":EXPECTED_RECORDS,"configs":CONFIG_COUNT,"model_forwards":0,"gt_metrics_read":False}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("derive","freeze"),required=True); args=parser.parse_args()
    print(json.dumps(derive() if args.mode=="derive" else freeze_manifest(),indent=2,sort_keys=True))

if __name__ == "__main__": main()
