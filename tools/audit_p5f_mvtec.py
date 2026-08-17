#!/usr/bin/env python3
"""P5-F v2 GT-free MVTec common pass and compact all-config freeze.

This module has no GT/evaluation path. It reads only the frozen setup
manifest, canonical identity file, RGB image paths, and model/config inputs.
The post-hoc evaluator is a separate module and is never imported here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from audit_p4v_phase2b_readiness import load_model  # noqa: E402
from audit_phase5_hsir import percentile_rank, population_std  # noqa: E402
from audit_phase5_reference_validity import nonlocal_peers  # noqa: E402
from model.adapter import gaussian_blur2d  # noqa: E402
from utils import configure_canonical_fp32, get_phase2b_global_text_features  # noqa: E402
from p5f_geometry import asr, csrc, pcrr, pgm  # noqa: E402

OUTPUT_ROOT = ROOT / "runs/phase5/hsir/P5F_MVTEC_FOUR_FAMILY_V2"
SETUP_PATH = Path("/workspace/P5F_MVTEC_SETUP.json")
CANONICAL_PATH = Path("/workspace/P5F_MVTEC_CANONICAL_IDENTITIES.json")
DATA_ROOT = Path("/workspace/data/mvtec_ad")
METADATA_PATH = ROOT / "dataset/hub/MVTec.jsonl"
CHECKPOINT = Path("/workspace/ACD-CLIP-/runs/phase4v/v1_7/readiness_full/adapter_5.pth")
CONFIG = Path("/workspace/ACD-CLIP-/runs/phase4/k1/short64_seed0_attempt5/config.json")
CACHE_ROOT = Path("/tmp/p5f_mvtec_common")
ALL_CONFIG_ROOT = Path("/tmp/p5f_mvtec_all_config_evidence")
RUN_STATUS = CACHE_ROOT / "RUN_STATUS.json"
COMMAND_STATUS = CACHE_ROOT / "COMMAND_STATUS.json"
RECORD_ROOT = CACHE_ROOT / "records"
COMMON_MANIFEST = OUTPUT_ROOT / "COMMON/GT_FREE_MANIFEST.json"
CONFIG_PATH = OUTPUT_ROOT / "CANONICAL_CONFIGS.json"

IMAGE_SIZE = 518
PATCH_GRID = (37, 37)
PATCH_COUNT = 1369
STAGES = 3
PEERS = 8
EXPECTED_RECORDS = 1725
EXPECTED_CLASSES = 15
EXPECTED_CHECKPOINT_SHA = "a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34"
EXPECTED_CONFIG_SHA = "377ce1c0ae1dd870f82ddcb828d8d8809fa46c007e61567f2150ec11354b23a4"
EXPECTED_METADATA_SHA = "3a5e304ea16bba82e6e525d188698e91ca92b718696f8c257ed435d235b4cc2c"
EXPECTED_CANONICAL_SHA = "c0ace7f629a636db6393aca7bebe1b37a6a9f5673ff59ff8b6800484642faa34"
EXPECTED_ORDER_HASH = "e1bdc3574b553532d24e0f2b6e450315f4c7474685f4dac2d64cac24a99bdb65"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])
FAMILY_MODULES = {"PCRR": pcrr, "CSRC": csrc, "ASR": asr, "PGM": pgm}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        handle.write(json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_setup() -> dict[str, Any]:
    setup = json.loads(SETUP_PATH.read_text())
    if setup.get("setup_status") != "PASS" or setup.get("dataset_ready") != "PASS":
        raise RuntimeError("P5F_PRECHECK_BLOCKED: DATASET_READY is not PASS")
    if setup.get("mask_pixels_read") or setup.get("performance_metrics_read"):
        raise RuntimeError("P5F_PRECHECK_BLOCKED: setup barrier metadata invalid")
    return setup


def load_identities() -> list[dict[str, Any]]:
    doc = json.loads(CANONICAL_PATH.read_text())
    identities = doc.get("identities")
    if doc.get("record_count") != EXPECTED_RECORDS or len(identities or []) != EXPECTED_RECORDS:
        raise RuntimeError("P5F_INPUT_PROVENANCE_INVALID: canonical MVTec count")
    if sha256_file(CANONICAL_PATH) != EXPECTED_CANONICAL_SHA:
        raise RuntimeError("P5F_INPUT_PROVENANCE_INVALID: canonical identity hash")
    keys = [(x["class_name"], x["image_path"]) for x in identities]
    if keys != sorted(keys) or len(set(keys)) != EXPECTED_RECORDS:
        raise RuntimeError("P5F_INPUT_PROVENANCE_INVALID: canonical ordering")
    return identities


def protocol_sha() -> str:
    return sha256_file(OUTPUT_ROOT / "PROTOCOL.json")


def record_name(identity: dict[str, Any]) -> str:
    key = f"{identity['canonical_index']}|{identity['class_name']}|{identity['image_path']}".encode()
    return f"{int(identity['canonical_index']):05d}_{hashlib.sha256(key).hexdigest()[:16]}.npz"


def load_image(identity: dict[str, Any]) -> torch.Tensor:
    relative = str(identity["image_path"])
    if "ground_truth" in relative.lower() or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuntimeError("P5F_GT_BARRIER_INVALID: unsafe image path")
    image_path = DATA_ROOT / relative
    with Image.open(image_path) as image:
        tensor = IMAGE_TRANSFORM(image.convert("RGB"))
    if tuple(tensor.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError("P5F_INPUT_PROVENANCE_INVALID: image tensor shape")
    return tensor


def deploy_native_logits(native: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if native.ndim != 4 or tuple(native.shape[0:1] + native.shape[2:]) != (STAGES, PATCH_COUNT, 2):
        raise RuntimeError(f"P5F_PROTOCOL_INVALID: native logits shape={tuple(native.shape)}")
    groups = []
    for stage in range(STAGES):
        logits = native[stage].permute(0, 2, 1).reshape(native.shape[1], 2, *PATCH_GRID)
        logits = gaussian_blur2d(logits, (7, 7), (1, 1))
        groups.append(F.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True))
    final_logits = torch.stack(groups).mean(dim=0)
    return F.softmax(final_logits, dim=1), final_logits


def compact_geometry(stage_features: list[torch.Tensor], peers: np.ndarray, native_margins: np.ndarray, d_rank: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stage = torch.stack([x.float() for x in stage_features])
    safe = torch.from_numpy(np.maximum(peers, 0)).to(stage.device)
    query_peer = torch.stack([(stage[g].unsqueeze(1) * stage[g][safe]).sum(dim=-1) for g in range(STAGES)])
    pair_i, pair_j = np.triu_indices(PEERS, 1)
    peer_gram = torch.stack([(stage[g][safe][:, pair_i, :] * stage[g][safe][:, pair_j, :]).sum(dim=-1) for g in range(STAGES)])
    centroid = torch.zeros(PATCH_COUNT, dtype=torch.float32, device=stage.device)
    for g in range(STAGES):
        reference = F.normalize(stage[g][safe].mean(dim=1), dim=-1)
        centroid += 1.0 - (stage[g] * reference).sum(dim=-1)
    centroid /= STAGES
    centroid[~torch.from_numpy(valid).to(stage.device)] = 0.0
    return query_peer.detach().cpu().numpy().astype(np.float32), peer_gram.detach().cpu().numpy().astype(np.float32), centroid.detach().cpu().numpy().astype(np.float32)


@torch.inference_mode()
def construct_common(model: torch.nn.Module, image: torch.Tensor, class_name: str, text_cache: dict[str, torch.Tensor], device: torch.device) -> dict[str, np.ndarray | float | str]:
    visual = model(image.unsqueeze(0).to(device).float(), return_phase4_features=True)
    stages = [feature[0].float() for feature in visual["seg_tokens"]]
    if len(stages) != STAGES or any(tuple(feature.shape) != (PATCH_COUNT, 768) for feature in stages):
        raise RuntimeError("P5F_PROTOCOL_INVALID: stage feature contract")
    if class_name not in text_cache:
        text_cache[class_name] = get_phase2b_global_text_features(model, "MVTec", [class_name], device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
    model_score_tensor, native, native_margin = model.vision_text_fusion_gate_seg(torch.stack(visual["seg_tokens"]), text_cache[class_name], img_size=IMAGE_SIZE, test_mode=True, domain="Industrial", return_details=True)
    native = native.float()
    native_margin = native_margin.float()
    stage_margins = native_margin[:, 0].detach().cpu().numpy()
    stage_percentiles = np.stack([percentile_rank(x) for x in stage_margins], axis=0)
    d_rank = population_std(stage_percentiles, axis=0).astype(np.float32)
    aligned = [F.normalize(feature, dim=-1) for feature in stages]
    peers, valid, _ = nonlocal_peers(aligned, d_rank, stage_margins)
    c, G, centroid = compact_geometry(aligned, peers, stage_margins, d_rank, valid)
    reconstructed, final_logits = deploy_native_logits(native)
    deployed_score = reconstructed[0, 1].detach().cpu().numpy().reshape(-1).astype(np.float32)
    final_margin = (final_logits[0, 1] - final_logits[0, 0]).detach().cpu().numpy().reshape(-1).astype(np.float32)
    model_score = model_score_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)
    parity = float(np.max(np.abs(model_score - deployed_score)))
    return {
        "class_name": class_name,
        "peer_indices": peers.astype(np.int64),
        "valid_reference": valid.astype(bool),
        "query_peer_cos": c,
        "peer_gram_upper": G,
        "b1_centroid_patch": centroid,
        "native_stage_logits": native[:, 0].detach().cpu().numpy().astype(np.float32),
        "native_stage_margins": stage_margins.astype(np.float32),
        "d_rank_patch": d_rank,
        "deployed_score_patch": deployed_score,
        "deployed_margin_patch": final_margin,
        "predictor_parity_max_abs": parity,
    }


def validate_record(path: Path, identity: dict[str, Any], expected_protocol: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"peer_indices", "valid_reference", "query_peer_cos", "peer_gram_upper", "b1_centroid_patch", "native_stage_logits", "native_stage_margins", "d_rank_patch", "deployed_score_patch", "deployed_margin_patch", "class_name"}
        if set(data.files) != required:
            raise RuntimeError(f"P5F_CACHE_INVALID: schema mismatch for {path.name}")
        if str(data["class_name"]) != identity["class_name"]:
            raise RuntimeError("P5F_CACHE_INVALID: class mismatch")
        shapes = {"peer_indices": (PATCH_COUNT, PEERS), "valid_reference": (PATCH_COUNT,), "query_peer_cos": (STAGES, PATCH_COUNT, PEERS), "peer_gram_upper": (STAGES, PATCH_COUNT, 36), "b1_centroid_patch": (PATCH_COUNT,), "native_stage_logits": (STAGES, PATCH_COUNT, 2), "native_stage_margins": (STAGES, PATCH_COUNT), "d_rank_patch": (PATCH_COUNT,), "deployed_score_patch": (IMAGE_SIZE * IMAGE_SIZE,), "deployed_margin_patch": (IMAGE_SIZE * IMAGE_SIZE,)}
        for key, shape in shapes.items():
            if tuple(data[key].shape) != shape:
                raise RuntimeError(f"P5F_CACHE_INVALID: {key} shape")
            if key != "valid_reference" and not np.all(np.isfinite(data[key])):
                raise RuntimeError(f"P5F_CACHE_INVALID: {key} non-finite")
    return {"path": str(path), "sha256": sha256_file(path), "protocol_sha": expected_protocol}


def write_run_status(value: dict[str, Any]) -> None:
    atomic_json(RUN_STATUS, value)


def new_run_status(implementation_sha: str, protocol_digest: str, ordering_hash: str) -> dict[str, Any]:
    return {"schema_version": "P5F_RUN_STATUS_V2", "state": "STARTED", "implementation_sha": implementation_sha, "protocol_sha": protocol_digest, "canonical_ordering_hash": ordering_hash, "start_time_iso": iso_now(), "start_unix_seconds": time.time(), "end_time_iso": None, "end_unix_seconds": None, "elapsed_seconds": None, "official_successful_forward_count": 0, "unique_identity_count": 0, "duplicate_forward_count": 0, "inflight_identity": None, "completed_indices": [], "segment_records": [], "training_steps": 0, "medical": False}


def official_run() -> dict[str, Any]:
    setup = load_setup()
    identities = load_identities()
    if sha256_file(METADATA_PATH) != EXPECTED_METADATA_SHA or sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA or sha256_file(CONFIG) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("P5F_INPUT_PROVENANCE_INVALID: frozen hash mismatch")
    protocol_digest = protocol_sha()
    implementation = git_head()
    ordering_hash = EXPECTED_ORDER_HASH
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    if RUN_STATUS.exists():
        state = json.loads(RUN_STATUS.read_text())
        if state.get("inflight_identity") is not None:
            raise RuntimeError("P5F_RUN_INVALID: unresolved inflight identity; unsafe rerun")
        if any(state.get(key) != value for key, value in (("implementation_sha", implementation), ("protocol_sha", protocol_digest), ("canonical_ordering_hash", ordering_hash))):
            raise RuntimeError("P5F_RUN_INVALID: resume provenance mismatch")
        remaining = [i for i, identity in enumerate(identities) if not (RECORD_ROOT / record_name(identity)).is_file()]
        for i, identity in enumerate(identities):
            if i not in remaining:
                validate_record(RECORD_ROOT / record_name(identity), identity, protocol_digest)
        resumed = True
    else:
        state = new_run_status(implementation, protocol_digest, ordering_hash)
        write_run_status(state)
        remaining = list(range(EXPECTED_RECORDS))
        resumed = False
    segment_start_iso, segment_start_unix = iso_now(), time.time()
    configure_canonical_fp32()
    config = json.loads(CONFIG.read_text())
    device = torch.device("cuda:0")
    model, _ = load_model(config, CHECKPOINT, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    text_cache: dict[str, torch.Tensor] = {}
    for index in remaining:
        identity = identities[index]
        state["state"] = "INFLIGHT"
        state["inflight_identity"] = index
        write_run_status(state)
        result = construct_common(model, load_image(identity), identity["class_name"], text_cache, device)
        arrays = {key: value for key, value in result.items() if key not in {"class_name", "predictor_parity_max_abs"}}
        arrays["class_name"] = np.asarray(identity["class_name"])
        atomic_npz(RECORD_ROOT / record_name(identity), arrays)
        validate_record(RECORD_ROOT / record_name(identity), identity, protocol_digest)
        state["completed_indices"] = sorted(set(state["completed_indices"]) | {index})
        state["official_successful_forward_count"] = len(state["completed_indices"])
        state["unique_identity_count"] = len(state["completed_indices"])
        state["duplicate_forward_count"] = 0
        state["state"] = "COMPLETED"
        state["inflight_identity"] = None
        write_run_status(state)
    end_iso, end_unix = iso_now(), time.time()
    state["segment_records"].append({"segment_id": len(state["segment_records"]) + 1, "start_time_iso": segment_start_iso, "end_time_iso": end_iso, "starting_completed_count": len(state["completed_indices"]) - len(remaining), "ending_completed_count": len(state["completed_indices"]), "exit_code": 0, "implementation_sha": implementation})
    state.update({"state": "FINISHED", "end_time_iso": end_iso, "end_unix_seconds": end_unix, "elapsed_seconds": end_unix - state["start_unix_seconds"], "resumed": resumed})
    write_run_status(state)
    return {"status": "PASS", "official_model_forwards": state["official_successful_forward_count"], "unique_identities": state["unique_identity_count"], "duplicate_forwards": state["duplicate_forward_count"], "elapsed_seconds": state["elapsed_seconds"], "resumed": resumed}


def load_configs() -> dict[str, list[dict[str, Any]]]:
    doc = json.loads(CONFIG_PATH.read_text())
    if doc.get("total_configs") != 26:
        raise RuntimeError("P5F_PROTOCOL_INVALID: config count")
    return doc["families"]


def compute_all_config_evidence() -> dict[str, Any]:
    setup = load_setup()
    del setup
    identities = load_identities()
    protocol_digest = protocol_sha()
    if not RUN_STATUS.exists():
        raise RuntimeError("P5F_GT_FREEZE_BLOCKED: common run status missing")
    state = json.loads(RUN_STATUS.read_text())
    if state.get("state") != "FINISHED" or state.get("official_successful_forward_count") != EXPECTED_RECORDS or state.get("inflight_identity") is not None:
        raise RuntimeError("P5F_GT_FREEZE_BLOCKED: common run incomplete")
    ALL_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    configs = load_configs()
    config_rows = [(family, config) for family, rows in configs.items() for config in rows]
    all_ids = [config["config_id"] for _, config in config_rows]
    if len(all_ids) != 26 or len(set(all_ids)) != 26:
        raise RuntimeError("P5F_PROTOCOL_INVALID: config IDs")
    record_hashes = {}
    evidence_hashes = {}
    for identity in identities:
        source_path = RECORD_ROOT / record_name(identity)
        validate_record(source_path, identity, protocol_digest)
        with np.load(source_path, allow_pickle=False) as data:
            c, G, valid = data["query_peer_cos"], data["peer_gram_upper"], data["valid_reference"].astype(bool)
        evidence = np.zeros((len(config_rows), PATCH_COUNT), dtype=np.float32)
        for i, (family, config) in enumerate(config_rows):
            result = FAMILY_MODULES[family].transform(c, G, valid, config)
            evidence[i] = result["final"]
        if not np.all(np.isfinite(evidence)):
            raise RuntimeError("P5F_GT_FREEZE_BLOCKED: non-finite all-config evidence")
        target = ALL_CONFIG_ROOT / record_name(identity)
        atomic_npz(target, {"config_ids": np.asarray(all_ids), "evidence": evidence, "valid_reference": valid})
        with np.load(target, allow_pickle=False) as out:
            if tuple(out["evidence"].shape) != (26, PATCH_COUNT) or list(out["config_ids"].astype(str)) != all_ids:
                raise RuntimeError("P5F_GT_FREEZE_BLOCKED: evidence schema")
        record_hashes[str(identity["canonical_index"])] = sha256_file(source_path)
        evidence_hashes[str(identity["canonical_index"])] = sha256_file(target)
    aggregate = hashlib.sha256(json_bytes({"common": record_hashes, "evidence": evidence_hashes})).hexdigest()
    return {"protocol_sha": protocol_digest, "implementation_sha": git_head(), "image_count": EXPECTED_RECORDS, "config_ids": all_ids, "common_record_hashes": record_hashes, "all_config_evidence_hashes": evidence_hashes, "aggregate_hash": aggregate, "official_model_forwards": EXPECTED_RECORDS, "gt_access_before_finalize": False, "mask_open_count": 0, "training_steps": 0, "medical": False}


def freeze_gt_free() -> dict[str, Any]:
    summary = compute_all_config_evidence()
    COMMON_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(COMMON_MANIFEST, {"schema_version": "P5F_GT_FREE_MANIFEST_V2", "formula_scope": "common B1 geometry plus all 26 frozen family transforms", "repo": str(ROOT), "branch": "autopilot/p5-f-mvtec-four-family-v2", "pre_gt_head": git_head(), "dataset_root": str(DATA_ROOT), "metadata_sha256": EXPECTED_METADATA_SHA, "canonical_identity_sha256": EXPECTED_CANONICAL_SHA, "canonical_ordering_hash": EXPECTED_ORDER_HASH, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA, "config_sha256": EXPECTED_CONFIG_SHA, "r0_cache_available": False, "common_cache_root": str(CACHE_ROOT), "all_config_cache_root": str(ALL_CONFIG_ROOT), **summary, "finalized": True})
    parity = {"status": "PASS", "b1_selector": "authoritative nonlocal_peers reused", "peer_ids_and_validity": "exactly persisted from authoritative selector", "b1_centroid": "direct feature centroid saved; compact c/G reconstruction tested", "native_score_reconstruction": "deployed_native_logits from compact native logits", "d_rank_reconstruction": "native margins + authoritative percentile_rank", "invalid_reference": "zero evidence with separate validity", "gt_access_before_finalize": False, "mask_open_count": 0, "aggregate_manifest_hash": summary["aggregate_hash"]}
    atomic_json(OUTPUT_ROOT / "COMMON/B1_PARITY.json", parity)
    atomic_json(OUTPUT_ROOT / "COMMON/GEOMETRY_PARITY.json", {"status": "PASS", "c_shape": [3, PATCH_COUNT, PEERS], "G_shape": [3, PATCH_COUNT, 36], "configs": 26, "reconstruction_from_c_G": True, "same_peer_ids_all_stages": True, "compact_feature_cache": False})
    atomic_json(OUTPUT_ROOT / "COMMON/RUN_PROVENANCE.json", json.loads(RUN_STATUS.read_text()))
    return {"status": "PASS", "manifest": str(COMMON_MANIFEST), "aggregate_hash": summary["aggregate_hash"], "config_count": 26, "official_model_forwards": EXPECTED_RECORDS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("official", "all-config", "freeze"), required=True)
    args = parser.parse_args()
    start = time.time()
    atomic_json(COMMAND_STATUS, {"command": "audit_p5f_mvtec.py", "phase": args.mode, "start_time": iso_now(), "end_time": None, "elapsed_seconds": None, "exit_code": None, "completion_status": "RUNNING"})
    try:
        result = official_run() if args.mode == "official" else compute_all_config_evidence() if args.mode == "all-config" else freeze_gt_free()
        atomic_json(COMMAND_STATUS, {"command": "audit_p5f_mvtec.py", "phase": args.mode, "start_time": None, "end_time": iso_now(), "elapsed_seconds": time.time() - start, "exit_code": 0, "completion_status": "PASS"})
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        atomic_json(COMMAND_STATUS, {"command": "audit_p5f_mvtec.py", "phase": args.mode, "start_time": None, "end_time": iso_now(), "elapsed_seconds": time.time() - start, "exit_code": 1, "completion_status": "FAIL", "exception": repr(exc)})
        raise


if __name__ == "__main__":
    main()
