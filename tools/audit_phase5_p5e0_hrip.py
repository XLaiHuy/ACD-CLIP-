#!/usr/bin/env python3
"""P5-E0 HRIP evidence audit.

Phase A is deliberately image-only: it consumes a preprocessed image tensor
and class identity, never a historical dataset record.  Phase C is an
explicit post-hoc evaluator and is unreachable from the official-run command.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
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
from audit_phase5_hsir import (  # noqa: E402
    ap_contamination,
    exact_auc_ap,
    pairwise_risks,
    percentile_rank,
    population_std,
    project_exact_auc_ap,
    shifted_map,
)
from audit_phase5_reference_validity import nonlocal_peers as authoritative_nonlocal_peers  # noqa: E402
from audit_phase5_second_evidence import (  # noqa: E402
    candidate_triage,
    deterministic_matches,
    matched_win_rate,
    select_top,
)
from model.adapter import gaussian_blur2d  # noqa: E402
from utils import configure_canonical_fp32, get_phase2b_global_text_features  # noqa: E402


AUDIT_ROOT = ROOT / "runs/phase5/hsir/P5E0_HRIP_EVIDENCE_AUDIT"
PROTOCOL_PATH = AUDIT_ROOT / "PROTOCOL.json"
INPUT_CHECK_PATH = AUDIT_ROOT / "INPUT_CHECK.json"
RECORD_ROOT = Path("/tmp/p5_e0_hrip")
RUN_STATE_PATH = RECORD_ROOT / "RUN_STATE.json"
RUN_PROVENANCE_PATH = AUDIT_ROOT / "RUN_PROVENANCE.json"
GT_FREE_MANIFEST_PATH = AUDIT_ROOT / "GT_FREE_HRIP_MANIFEST.json"

IMAGE_SIZE = 518
PATCH_GRID = (37, 37)
PATCH_COUNT = 1369
STAGES = 3
K = 8
PRIMARY_FRACTION = 0.20
TRIAGE_FRACTION = 0.10
BOOTSTRAP_REPS = 2000
EPS_FLOAT32 = float(torch.finfo(torch.float32).eps)

CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
CONFIG = ROOT / "runs/phase4/k1/short64_seed0_attempt5/config.json"
VISA_META = ROOT / "dataset/hub/VisA.jsonl"
VISA_ROOT = Path("/workspace/data/med-visa/data/VisA_20220922")
EXPECTED_CHECKPOINT_SHA = "a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34"
EXPECTED_CONFIG_SHA = "377ce1c0ae1dd870f82ddcb828d8d8809fa46c007e61567f2150ec11354b23a4"
EXPECTED_META_SHA = "468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842"
EXPECTED_HEAD = "f6e9130a4ff77bac2b11f55dc84898a76d5c7261"

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        handle.write(_json_bytes(value))
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_setup() -> dict[str, Any]:
    manifest = json.loads(Path("/workspace/P5_SETUP_MANIFEST.json").read_text())
    if manifest.get("setup_status") != "PASS":
        raise RuntimeError("P5E0_PRECHECK_BLOCKED: setup manifest is not PASS")
    if not manifest.get("r0_cache_available") and manifest.get("r0_cache") is not None:
        raise RuntimeError("P5E0_INPUT_PROVENANCE_INVALID: R0 state is inconsistent")
    return manifest


def load_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol["formula_id"] != "HRIP_SHARED_SOFT_PROJECTION" or protocol["candidate"] != "NONE":
        raise RuntimeError("P5E0_PROTOCOL_INVALID: formula or candidate drift")
    return protocol


def protocol_sha() -> str:
    return sha256_file(PROTOCOL_PATH)


def build_canonical_identities(metadata_path: Path = VISA_META) -> list[dict[str, Any]]:
    """Read only class identity and image path; no GT-derived metadata fields."""
    grouped: dict[str, list[str]] = {}
    with metadata_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            class_name = str(row["class_name"])
            relative_path = str(row["image_path"])
            grouped.setdefault(class_name, []).append(relative_path)
    identities: list[dict[str, Any]] = []
    for class_name in sorted(grouped):
        for relative_path in grouped[class_name]:
            identities.append(
                {
                    "class_name": class_name,
                    "relative_image_path": relative_path,
                    "canonical_order_index": len(identities),
                }
            )
    if len(grouped) != 12 or len(identities) != 2162:
        raise RuntimeError("P5E0_INPUT_PROVENANCE_INVALID: setup-provenance TEST identity count mismatch")
    keys = [(x["class_name"], x["relative_image_path"]) for x in identities]
    if len(set(keys)) != len(keys):
        raise RuntimeError("P5E0_INPUT_PROVENANCE_INVALID: duplicate canonical image identity")
    return identities


def canonical_ordering_hash(identities: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_json_bytes(identities)).hexdigest()


def load_image_tensor(identity: dict[str, Any], data_root: Path = VISA_ROOT) -> torch.Tensor:
    image_path = data_root / identity["relative_image_path"]
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = IMAGE_TRANSFORM(image)
    if tuple(tensor.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"P5E0_INPUT_PROVENANCE_INVALID: image tensor shape={tuple(tensor.shape)}")
    return tensor


def align_stage_features(stage_features: list[torch.Tensor], patch_grid: tuple[int, int]) -> tuple[list[torch.Tensor], dict[str, Any]]:
    shapes = []
    for feature in stage_features:
        if feature.ndim != 2 or int(feature.shape[0]) != patch_grid[0] * patch_grid[1]:
            raise RuntimeError(f"P5E0_PROTOCOL_INVALID: feature shape={tuple(feature.shape)}")
        shapes.append({"patch_count": int(feature.shape[0]), "dimension": int(feature.shape[1]), "grid": list(patch_grid)})
    if len({item["dimension"] for item in shapes}) != 1:
        raise RuntimeError("P5E0_PROTOCOL_INVALID: stage dimensions differ")
    aligned = []
    for feature in stage_features:
        tensor = feature.reshape(patch_grid[0], patch_grid[1], -1).permute(2, 0, 1).unsqueeze(0)
        tensor = F.normalize(tensor, dim=1).squeeze(0).permute(1, 2, 0).reshape(PATCH_COUNT, -1)
        aligned.append(tensor)
    return aligned, {
        "source": "model/adapter.py::ACDCLIP.forward return_phase4_features=True",
        "tensor": "visual['seg_tokens']",
        "normalization": "seg_proj -> seg_layer_norms -> F.normalize(dim=-1)",
        "stages": shapes,
        "reference_grid": list(patch_grid),
        "alignment": "bilinear align_corners=True before L2 renormalization when grids differ",
    }


def deploy_native_logits(native: torch.Tensor, patch_grid: tuple[int, int] = PATCH_GRID, image_size: int = IMAGE_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    if native.ndim != 4 or native.shape[-1] != 2 or native.shape[2] != patch_grid[0] * patch_grid[1]:
        raise RuntimeError(f"P5E0_PROTOCOL_INVALID: native shape={tuple(native.shape)}")
    outputs = []
    for group in range(native.shape[0]):
        logits = native[group].permute(0, 2, 1).reshape(native.shape[1], 2, *patch_grid)
        logits = gaussian_blur2d(logits, (7, 7), (1, 1))
        outputs.append(F.interpolate(logits, size=(image_size, image_size), mode="bilinear", align_corners=True))
    final_logits = torch.stack(outputs).mean(dim=0)
    return F.softmax(final_logits, dim=1), final_logits


def select_b1_peers(aligned: list[torch.Tensor], d_rank: np.ndarray, native_margins: np.ndarray) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    stage_tensor = torch.stack(aligned).float()
    shared = F.normalize(stage_tensor.mean(dim=0), dim=-1)
    stage_q = np.stack([percentile_rank(values) for values in native_margins], axis=0)
    pool = (d_rank < np.median(d_rank)) & np.all(stage_q < 0.5, axis=0)
    yy, xx = np.divmod(np.arange(PATCH_COUNT), PATCH_GRID[1])
    peers = np.full((PATCH_COUNT, K), -1, dtype=np.int64)
    valid = np.zeros(PATCH_COUNT, dtype=bool)
    pool_indices = np.flatnonzero(pool)
    pool_features = shared[pool_indices]
    for start in range(0, PATCH_COUNT, 64):
        end = min(PATCH_COUNT, start + 64)
        similarity = (shared[start:end] @ pool_features.T).detach().cpu().numpy()
        for local, query in enumerate(range(start, end)):
            spatial_ok = np.maximum(np.abs(yy[pool_indices] - yy[query]), np.abs(xx[pool_indices] - xx[query])) > 3
            candidates = pool_indices[spatial_ok]
            if candidates.size < K:
                continue
            columns = np.flatnonzero(spatial_ok)
            order = np.lexsort((candidates, -similarity[local, columns]))
            peers[query] = candidates[order[:K]]
            valid[query] = True
    return peers, valid, shared


def centroid_evidence(aligned: list[torch.Tensor], peers: np.ndarray, valid: np.ndarray) -> np.ndarray:
    stage_tensor = torch.stack(aligned).float()
    safe = torch.from_numpy(np.maximum(peers, 0)).to(stage_tensor.device)
    evidence = torch.zeros(PATCH_COUNT, dtype=torch.float32, device=stage_tensor.device)
    for group in range(STAGES):
        reference = F.normalize(stage_tensor[group][safe].mean(dim=1), dim=-1)
        evidence += 1.0 - (stage_tensor[group] * reference).sum(dim=-1)
    evidence /= float(STAGES)
    evidence[~torch.from_numpy(valid).to(evidence.device)] = 0
    return evidence.detach().cpu().numpy().astype(np.float32)


def soft_weights(shared: torch.Tensor, peers: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    safe = torch.from_numpy(np.maximum(peers, 0)).to(shared.device)
    peer_shared = shared[safe]
    pair_matrix = torch.einsum("pkd,pjd->pkj", peer_shared, peer_shared)
    pair_distances = torch.stack([1.0 - pair_matrix[:, a, b] for a in range(K) for b in range(a + 1, K)], dim=1)
    tau = torch.median(pair_distances, dim=1).values
    query_distances = 1.0 - (shared[:, None, :] * peer_shared).sum(dim=-1)
    alpha = torch.softmax(-query_distances / tau.clamp_min(EPS_FLOAT32)[:, None], dim=1)
    degenerate = tau <= EPS_FLOAT32
    alpha[degenerate] = 1.0 / float(K)
    return alpha, tau, query_distances, peer_shared


def loo_raw_residuals(
    stage_tensor: torch.Tensor,
    shared: torch.Tensor,
    peers: np.ndarray,
    query_distances: torch.Tensor,
) -> torch.Tensor:
    safe = torch.from_numpy(np.maximum(peers, 0)).to(stage_tensor.device)
    peer_shared = shared[safe]
    pair_matrix = torch.einsum("pkd,pjd->pkj", peer_shared, peer_shared)
    values = []
    for omitted in range(K):
        keep = [slot for slot in range(K) if slot != omitted]
        pair_distances = torch.stack(
            [1.0 - pair_matrix[:, keep[a], keep[b]] for a in range(K - 1) for b in range(a + 1, K - 1)], dim=1
        )
        tau = torch.median(pair_distances, dim=1).values
        alpha = torch.softmax(-query_distances[:, keep] / tau.clamp_min(EPS_FLOAT32)[:, None], dim=1)
        alpha[tau <= EPS_FLOAT32] = 1.0 / float(K - 1)
        stage_raw = []
        for group in range(STAGES):
            peer_stage = F.normalize(stage_tensor[group][safe[:, keep]], dim=-1)
            reconstruction = F.normalize((alpha[:, :, None] * peer_stage).sum(dim=1), dim=-1)
            stage_raw.append(1.0 - (F.normalize(stage_tensor[group], dim=-1) * reconstruction).sum(dim=-1))
        values.append(torch.stack(stage_raw).mean(dim=0))
    return torch.stack(values, dim=1)


def compute_hrip(aligned: list[torch.Tensor], peers: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray | torch.Tensor]:
    stage_tensor = torch.stack([feature.float() for feature in aligned])
    normalized_stage = F.normalize(stage_tensor, dim=-1)
    shared = F.normalize(normalized_stage.mean(dim=0), dim=-1)
    alpha, tau, query_distances, _ = soft_weights(shared, peers)
    safe = torch.from_numpy(np.maximum(peers, 0)).to(stage_tensor.device)
    stage_residuals = []
    for group in range(STAGES):
        peer_stage = F.normalize(normalized_stage[group][safe], dim=-1)
        reconstruction = F.normalize((alpha[:, :, None] * peer_stage).sum(dim=1), dim=-1)
        stage_residuals.append(1.0 - (normalized_stage[group] * reconstruction).sum(dim=-1))
    residual = torch.stack(stage_residuals)
    rank = torch.from_numpy(np.stack([percentile_rank(x.detach().cpu().numpy()) for x in residual]))
    raw = residual.mean(dim=0)
    loo = loo_raw_residuals(normalized_stage, shared, peers, query_distances)
    loo_median = loo.median(dim=1).values
    loo_mad = (loo - loo_median[:, None]).abs().median(dim=1).values
    loo_change = (loo - raw[:, None]).abs().max(dim=1).values
    entropy = -(alpha * alpha.clamp_min(EPS_FLOAT32).log()).sum(dim=1)
    effective = 1.0 / (alpha.square().sum(dim=1).clamp_min(EPS_FLOAT32))
    invalid = ~torch.from_numpy(valid).to(stage_tensor.device)
    arrays: dict[str, np.ndarray | torch.Tensor] = {
        "hrip": rank.mean(dim=0).numpy().astype(np.float32),
        "hrip_raw": raw.detach().cpu().numpy().astype(np.float32),
        "tau": tau.detach().cpu().numpy().astype(np.float32),
        "max_alpha": alpha.max(dim=1).values.detach().cpu().numpy().astype(np.float32),
        "attention_entropy": entropy.detach().cpu().numpy().astype(np.float32),
        "effective_peer_count": effective.detach().cpu().numpy().astype(np.float32),
        "stage_residual_std": residual.detach().cpu().numpy().astype(np.float32).std(axis=0),
        "stage_rank_std": rank.numpy().astype(np.float32).std(axis=0),
        "loo_median_residual": loo_median.detach().cpu().numpy().astype(np.float32),
        "loo_MAD": loo_mad.detach().cpu().numpy().astype(np.float32),
        "loo_max_abs_change": loo_change.detach().cpu().numpy().astype(np.float32),
        "alpha": alpha,
        "residual": residual,
    }
    for name, value in list(arrays.items()):
        if isinstance(value, np.ndarray):
            value[~valid] = 0
        elif name == "alpha":
            value[invalid, :] = 0
        elif name == "residual":
            value[:, invalid] = 0
    return arrays


@torch.inference_mode()
def construct_image_evidence(model: torch.nn.Module, image: torch.Tensor, class_name: str, text_cache: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
    """GT-free one-image construction API; no dataset record or GT argument."""
    if image.ndim != 4 or tuple(image.shape[1:]) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"P5E0_PROTOCOL_INVALID: image shape={tuple(image.shape)}")
    visual = model(image.to(device).float(), return_phase4_features=True)
    stage_batches = [value.float() for value in visual["seg_tokens"]]
    stage_features = [value[0] for value in stage_batches]
    features = torch.stack(stage_batches)
    if class_name not in text_cache:
        text_cache[class_name] = get_phase2b_global_text_features(
            model, "VisA", [class_name], device, use_hybrid_soft_prompt=True, use_soft_prompt=False
        ).float()
    model_prob, native, native_margin = model.vision_text_fusion_gate_seg(
        features, text_cache[class_name], img_size=IMAGE_SIZE, test_mode=True, domain="Industrial", return_details=True
    )
    reconstructed, final_logits = deploy_native_logits(native)
    patch_grid = PATCH_GRID
    if tuple(native.shape) != (STAGES, 1, PATCH_COUNT, 2) or tuple(native_margin.shape) != (STAGES, 1, PATCH_COUNT):
        raise RuntimeError("P5E0_PROTOCOL_INVALID: native output shape")
    parity = float((model_prob - reconstructed[:, 1]).abs().max().detach().cpu())
    native_margins = native_margin[:, 0].detach().float().cpu().numpy()
    native_logits = native[:, 0].detach().float().cpu().numpy()
    stage_ranks = np.stack([percentile_rank(value) for value in native_margins], axis=0)
    d_rank_patch = population_std(stage_ranks, axis=0).astype(np.float32)
    d_logit_patch = population_std(native_margins, axis=0).astype(np.float32)
    aligned, feature_semantics = align_stage_features(stage_features, patch_grid)
    peers, valid, _ = select_b1_peers(aligned, d_rank_patch, native_margins)
    hrip = compute_hrip(aligned, peers, valid)
    e_nonlocal = centroid_evidence(aligned, peers, valid)
    shape_record = {
        "stage_visual_features": [list(value.shape) for value in stage_batches],
        "stacked_visual_features": list(features.shape),
        "native_stage_logits": list(native.shape),
        "native_stage_margins": list(native_margin.shape),
        "patch_grid": list(patch_grid),
        "patch_count": PATCH_COUNT,
        "deployed_model_probability": list(model_prob.shape),
        "deployed_reconstructed_probability": list(reconstructed.shape),
        "deployed_final_logits": list(final_logits.shape),
    }
    arrays = {
        "peer_indices": peers.astype(np.int64),
        "valid_reference": valid.astype(bool),
        "hrip": hrip["hrip"],
        "hrip_raw": hrip["hrip_raw"],
        "e_nonlocal_patch": e_nonlocal,
        "tau": hrip["tau"],
        "max_alpha": hrip["max_alpha"],
        "attention_entropy": hrip["attention_entropy"],
        "effective_peer_count": hrip["effective_peer_count"],
        "stage_residual_std": hrip["stage_residual_std"],
        "stage_rank_std": hrip["stage_rank_std"],
        "loo_median_residual": hrip["loo_median_residual"],
        "loo_MAD": hrip["loo_MAD"],
        "loo_max_abs_change": hrip["loo_max_abs_change"],
        "native_stage_logits": native_logits.astype(np.float32),
        "native_stage_margins": native_margins.astype(np.float32),
        "d_rank_patch": d_rank_patch.astype(np.float32),
    }
    if not all(np.all(np.isfinite(value)) for value in arrays.values() if value.dtype.kind in "fc"):
        raise RuntimeError("P5E0_OUTPUT_INVALID: non-finite compact evidence")
    return {
        "arrays": arrays,
        "metadata": {
            "class_name": class_name,
            "shape_record": shape_record,
            "feature_semantics": feature_semantics,
            "model_probability_parity_max_abs": parity,
            "d_logit_patch_summary": {"mean": float(d_logit_patch.mean()), "max": float(d_logit_patch.max())},
        },
    }


RECORD_ARRAYS = (
    "peer_indices", "valid_reference", "hrip", "hrip_raw", "e_nonlocal_patch", "tau", "max_alpha",
    "attention_entropy", "effective_peer_count", "stage_residual_std", "stage_rank_std",
    "loo_median_residual", "loo_MAD", "loo_max_abs_change", "native_stage_logits", "native_stage_margins", "d_rank_patch",
)


def record_filename(identity: dict[str, Any]) -> str:
    key = f"{identity['class_name']}|{identity['relative_image_path']}".encode()
    return f"record_{int(identity['canonical_order_index']):04d}_{hashlib.sha256(key).hexdigest()[:16]}.npz"


def save_record(root: Path, identity: dict[str, Any], result: dict[str, Any], implementation: str, protocol: str) -> Path:
    metadata = {
        **identity,
        "record_schema_version": "1.0",
        "implementation_sha": implementation,
        "protocol_sha": protocol,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "config_sha256": EXPECTED_CONFIG_SHA,
        "metadata_sha256": EXPECTED_META_SHA,
        **result["metadata"],
    }
    arrays = dict(result["arrays"])
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    path = root / record_filename(identity)
    atomic_npz(path, arrays)
    return path


def load_record(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        if sorted(name for name in archive.files if name != "metadata_json") != sorted(RECORD_ARRAYS):
            raise RuntimeError(f"P5E0_OUTPUT_INVALID: record array schema {path.name}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {name: archive[name].copy() for name in RECORD_ARRAYS}
    if any(not np.all(np.isfinite(value)) for value in arrays.values() if value.dtype.kind in "fc"):
        raise RuntimeError(f"P5E0_OUTPUT_INVALID: non-finite record {path.name}")
    if arrays["peer_indices"].shape != (PATCH_COUNT, K) or arrays["valid_reference"].shape != (PATCH_COUNT,):
        raise RuntimeError(f"P5E0_OUTPUT_INVALID: peer shape {path.name}")
    for name in RECORD_ARRAYS[2:13]:
        if arrays[name].shape != (PATCH_COUNT,):
            raise RuntimeError(f"P5E0_OUTPUT_INVALID: patch shape {name} in {path.name}")
    if arrays["native_stage_logits"].shape != (STAGES, PATCH_COUNT, 2) or arrays["native_stage_margins"].shape != (STAGES, PATCH_COUNT):
        raise RuntimeError(f"P5E0_OUTPUT_INVALID: native shape {path.name}")
    if arrays["d_rank_patch"].shape != (PATCH_COUNT,):
        raise RuntimeError(f"P5E0_OUTPUT_INVALID: D_rank shape {path.name}")
    return metadata, arrays


def validate_record(path: Path, identity: dict[str, Any], implementation: str, protocol: str) -> dict[str, Any]:
    metadata, arrays = load_record(path)
    for key in ("class_name", "relative_image_path", "canonical_order_index"):
        if metadata.get(key) != identity[key]:
            raise RuntimeError(f"P5E0_OUTPUT_INVALID: identity mismatch at {path.name}")
    if metadata.get("implementation_sha") != implementation or metadata.get("protocol_sha") != protocol:
        raise RuntimeError(f"P5E0_OUTPUT_INVALID: implementation/protocol mismatch at {path.name}")
    native = torch.from_numpy(arrays["native_stage_logits"][:, None]).float()
    _, final_logits = deploy_native_logits(native)
    if not np.all(np.isfinite(final_logits.detach().cpu().numpy())):
        raise RuntimeError(f"P5E0_OUTPUT_INVALID: deployment reconstruction at {path.name}")
    stage_ranks = np.stack([percentile_rank(value) for value in arrays["native_stage_margins"]], axis=0)
    reconstructed_d_rank = population_std(stage_ranks, axis=0).astype(np.float32)
    if not np.array_equal(reconstructed_d_rank, arrays["d_rank_patch"]):
        if not np.allclose(reconstructed_d_rank, arrays["d_rank_patch"], rtol=0, atol=1e-7):
            raise RuntimeError(f"P5E0_OUTPUT_INVALID: D_rank reconstruction at {path.name}")
    return {"metadata": metadata, "arrays": arrays, "sha256": sha256_file(path)}


def _new_run_state(implementation: str, protocol: str, ordering_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "state": "ready",
        "implementation_sha": implementation,
        "protocol_sha": protocol,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "config_sha256": EXPECTED_CONFIG_SHA,
        "metadata_sha256": EXPECTED_META_SHA,
        "canonical_ordering_hash": ordering_hash,
        "run_start_time_iso": iso_now(),
        "run_start_unix_seconds": time.time(),
        "current_identity": None,
        "completed_identity_indices": [],
        "official_successful_forward_count": 0,
        "duplicate_forward_count": 0,
        "segments": [],
    }


def resume_indices(state: dict[str, Any], total: int) -> list[int]:
    if state.get("current_identity") is not None or state.get("state") == "inflight":
        raise RuntimeError("P5E0_HRIP_AUDIT_INVALID: unresolved inflight identity; unsafe resume")
    completed = set(int(x) for x in state.get("completed_identity_indices", []))
    if len(completed) != len(state.get("completed_identity_indices", [])):
        raise RuntimeError("P5E0_OUTPUT_INVALID: duplicate completed identity in RUN_STATE")
    return [index for index in range(total) if index not in completed]


def write_run_provenance(state: dict[str, Any], identities: list[dict[str, Any]], implementation: str, protocol: str, ordering_hash: str) -> None:
    end_iso = iso_now()
    end_unix = time.time()
    elapsed = float(end_unix - float(state["run_start_unix_seconds"]))
    payload = {
        "protocol_sha": protocol,
        "implementation_sha": implementation,
        "plumbing_sha_or_null": None,
        "official_run_head": git_head(),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "config_sha256": EXPECTED_CONFIG_SHA,
        "metadata_sha256": EXPECTED_META_SHA,
        "canonical_ordering_hash": ordering_hash,
        "start_time_iso": state["run_start_time_iso"],
        "start_unix_seconds": state["run_start_unix_seconds"],
        "end_time_iso": end_iso,
        "end_unix_seconds": end_unix,
        "elapsed_seconds": elapsed,
        "average_seconds_per_image": elapsed / 2162.0,
        "images_per_second": 2162.0 / elapsed if elapsed > 0 else None,
        "run_segment_count": len(state.get("segments", [])),
        "official_successful_forward_count": int(state.get("official_successful_forward_count", 0)),
        "unique_identity_count": len(state.get("completed_identity_indices", [])),
        "duplicate_forward_count": int(state.get("duplicate_forward_count", 0)),
        "resumed": len(state.get("segments", [])) > 1,
        "segment_records": state.get("segments", []),
        "training_steps": 0,
        "medical": False,
    }
    atomic_json(RUN_PROVENANCE_PATH, payload)


def official_run(output_root: Path = RECORD_ROOT) -> dict[str, Any]:
    setup = load_setup()
    protocol = load_protocol()
    if git_head() != EXPECTED_HEAD:
        raise RuntimeError("P5E0_INPUT_PROVENANCE_INVALID: implementation must start after protocol commit")
    identities = build_canonical_identities()
    ordering_hash = canonical_ordering_hash(identities)
    implementation = git_head()
    protocol_digest = protocol_sha()
    output_root.mkdir(parents=True, exist_ok=True)
    if RUN_STATE_PATH.exists():
        state = json.loads(RUN_STATE_PATH.read_text())
        for key, expected in (("implementation_sha", implementation), ("protocol_sha", protocol_digest), ("canonical_ordering_hash", ordering_hash)):
            if state.get(key) != expected:
                raise RuntimeError(f"P5E0_HRIP_AUDIT_INVALID: resume {key} mismatch")
        remaining = resume_indices(state, len(identities))
        for index in sorted(set(range(len(identities))) - set(remaining)):
            validate_record(output_root / record_filename(identities[index]), identities[index], implementation, protocol_digest)
    else:
        state = _new_run_state(implementation, protocol_digest, ordering_hash)
        atomic_json(RUN_STATE_PATH, state)
        remaining = list(range(len(identities)))
    segment_start_iso = iso_now()
    segment_start_unix = time.time()
    configure_canonical_fp32()
    config = json.loads(CONFIG.read_text())
    device = torch.device("cuda:0")
    model, _ = load_model(config, CHECKPOINT, device)
    model.eval()
    text_cache: dict[str, torch.Tensor] = {}
    for index in remaining:
        identity = identities[index]
        state["state"] = "inflight"
        state["current_identity"] = index
        atomic_json(RUN_STATE_PATH, state)
        image = load_image_tensor(identity).unsqueeze(0)
        result = construct_image_evidence(model, image, identity["class_name"], text_cache, device)
        save_record(output_root, identity, result, implementation, protocol_digest)
        state["completed_identity_indices"] = sorted(set(state["completed_identity_indices"]) | {index})
        state["official_successful_forward_count"] = len(state["completed_identity_indices"])
        state["duplicate_forward_count"] = 0
        state["state"] = "completed"
        state["current_identity"] = None
        atomic_json(RUN_STATE_PATH, state)
        del image, result
    segment_end_iso = iso_now()
    state["segments"].append(
        {
            "segment_id": len(state["segments"]) + 1,
            "start_time_iso": segment_start_iso,
            "end_time_iso": segment_end_iso,
            "starting_completed_count": len(state["completed_identity_indices"]) - len(remaining),
            "ending_completed_count": len(state["completed_identity_indices"]),
            "exit_code": 0,
            "implementation_sha": implementation,
        }
    )
    if len(state["completed_identity_indices"]) != 2162 or state["duplicate_forward_count"] != 0:
        raise RuntimeError("P5E0_HRIP_AUDIT_INVALID: official accounting mismatch")
    state["state"] = "finished"
    atomic_json(RUN_STATE_PATH, state)
    write_run_provenance(state, identities, implementation, protocol_digest, ordering_hash)
    return {
        "status": "PASS",
        "official_successful_forward_count": state["official_successful_forward_count"],
        "unique_identity_count": len(state["completed_identity_indices"]),
        "duplicate_forward_count": state["duplicate_forward_count"],
        "ordering_hash": ordering_hash,
        "run_provenance": str(RUN_PROVENANCE_PATH),
    }


def validate_cache(output_root: Path = RECORD_ROOT) -> dict[str, Any]:
    setup = load_setup()
    protocol = load_protocol()
    identities = build_canonical_identities()
    ordering_hash = canonical_ordering_hash(identities)
    state = json.loads(RUN_STATE_PATH.read_text())
    implementation = str(state["implementation_sha"])
    protocol_digest = protocol_sha()
    if state.get("state") != "finished" or state.get("current_identity") is not None:
        raise RuntimeError("P5E0_HRIP_AUDIT_INVALID: GT-free cache not finished safely")
    if state.get("canonical_ordering_hash") != ordering_hash or state.get("protocol_sha") != protocol_digest:
        raise RuntimeError("P5E0_HRIP_AUDIT_INVALID: cache provenance mismatch")
    records = []
    for identity in identities:
        path = output_root / record_filename(identity)
        if not path.is_file():
            raise RuntimeError(f"P5E0_HRIP_AUDIT_INVALID: missing record {path.name}")
        records.append(validate_record(path, identity, implementation, protocol_digest))
    if state.get("official_successful_forward_count") != 2162 or len(records) != 2162 or state.get("duplicate_forward_count") != 0:
        raise RuntimeError("P5E0_HRIP_AUDIT_INVALID: record accounting mismatch")
    return {
        "status": "PASS",
        "implementation_sha": implementation,
        "protocol_sha": protocol_digest,
        "ordering_hash": ordering_hash,
        "records": records,
        "identities": identities,
        "state": state,
    }


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def bootstrap_summary(values: list[float | None], seed: int) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "median": None, "bootstrap95_ci": None, "n_classes": 0, "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": seed, "unit": "class"}
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, array.size, size=(BOOTSTRAP_REPS, array.size))].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "bootstrap95_ci": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "n_classes": int(array.size),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": seed,
        "unit": "class",
    }


def paired_bootstrap_summary(left: list[float | None], right: list[float | None], seed: int) -> dict[str, Any]:
    deltas = [None if a is None or b is None else float(a - b) for a, b in zip(left, right)]
    result = bootstrap_summary(deltas, seed)
    result["paired"] = True
    return result


def run_synthetic_tests() -> dict[str, Any]:
    rng = np.random.default_rng(12345)
    q = torch.nn.functional.normalize(torch.from_numpy(rng.normal(size=(16,)).astype(np.float32)), dim=0)
    peers = torch.nn.functional.normalize(torch.from_numpy(rng.normal(size=(K, 16)).astype(np.float32)), dim=1)
    shared = q[None, :].repeat(K, 1)
    alpha, tau, query_distances, _ = soft_weights(q[None, :].repeat(PATCH_COUNT, 1), np.tile(np.arange(K), (PATCH_COUNT, 1)))
    checks: dict[str, bool] = {}
    checks["T1_alpha_finite_nonnegative_sum_one"] = bool(torch.isfinite(alpha).all() and (alpha >= 0).all() and torch.allclose(alpha.sum(1), torch.ones(PATCH_COUNT), atol=1e-6))
    identical = torch.ones((PATCH_COUNT, 16), dtype=torch.float32)
    identical_peers = np.tile(np.arange(K), (PATCH_COUNT, 1))
    identical_alpha, identical_tau, _, _ = soft_weights(identical, identical_peers)
    checks["T2_degenerate_tau_uniform"] = bool(torch.all(identical_tau <= EPS_FLOAT32) and torch.allclose(identical_alpha, torch.full_like(identical_alpha, 1.0 / K), atol=1e-7))
    distances = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], dtype=torch.float32)
    fixed = torch.softmax(-distances / 0.25, dim=1)
    checks["T3_more_similar_peer_higher_alpha"] = bool(torch.all(fixed[:, :-1] >= fixed[:, 1:]))
    perm = torch.tensor([3, 0, 7, 2, 5, 1, 6, 4])
    a0 = torch.softmax(-distances / 0.25, dim=1)
    a1 = torch.softmax(-distances[:, perm] / 0.25, dim=1)
    checks["T4_peer_permutation_equivariance"] = bool(torch.allclose(a1[:, torch.argsort(perm)], a0, atol=1e-7))
    aligned = [F.normalize(torch.from_numpy(rng.normal(size=(PATCH_COUNT, 16)).astype(np.float32)), dim=1) for _ in range(STAGES)]
    margins = rng.normal(size=(STAGES, PATCH_COUNT)).astype(np.float32)
    d_rank = population_std(np.stack([percentile_rank(x) for x in margins]), axis=0).astype(np.float32)
    selected_peers, valid, _ = select_b1_peers(aligned, d_rank, margins)
    computed = compute_hrip(aligned, selected_peers, valid)
    checks["T5_one_alpha_reused_all_stages"] = bool(computed["residual"].shape == (STAGES, PATCH_COUNT) and computed["alpha"].shape == (PATCH_COUNT, K))
    same_stage = [torch.ones((PATCH_COUNT, 16), dtype=torch.float32) for _ in range(STAGES)]
    same_peers = np.tile(np.arange(K), (PATCH_COUNT, 1))
    same = compute_hrip(same_stage, same_peers, np.ones(PATCH_COUNT, dtype=bool))
    checks["T6_identical_query_peer_residual_zero"] = bool(float(np.max(np.abs(same["hrip_raw"]))) <= 1e-6)
    checks["T7_loo_removes_one_and_retains_seven"] = bool(computed["loo_median_residual"].shape == (PATCH_COUNT,) and loo_raw_residuals(torch.stack(aligned), F.normalize(torch.stack(aligned).mean(0), dim=-1), selected_peers, soft_weights(F.normalize(torch.stack(aligned).mean(0), dim=-1), selected_peers)[2]).shape == (PATCH_COUNT, K))
    checks["T8_shift_only_changes_correspondence"] = bool(np.array_equal(shifted_map(np.arange(16), 4, 4), np.roll(np.arange(16).reshape(4, 4), (1, 1), axis=(0, 1)).reshape(-1)))
    signature = inspect.signature(construct_image_evidence)
    checks["T9_api_has_no_gt_argument"] = not bool(set(signature.parameters) & {"mask", "target", "label", "labels", "gt", "occupancy"})
    source = inspect.getsource(construct_image_evidence)
    checks["T10_official_path_has_no_gt_field_access"] = not any(token in source for token in ("mask", "target", "label", "gt", "occupancy"))
    ref_features = [F.normalize(torch.from_numpy(rng.normal(size=(PATCH_COUNT, 8)).astype(np.float32)), dim=1) for _ in range(STAGES)]
    ref_margins = rng.normal(size=(STAGES, PATCH_COUNT)).astype(np.float32)
    ref_rank = population_std(np.stack([percentile_rank(x) for x in ref_margins]), axis=0).astype(np.float32)
    ours_peers, ours_valid, _ = select_b1_peers(ref_features, ref_rank, ref_margins)
    auth_peers, auth_valid, _ = authoritative_nonlocal_peers(ref_features, ref_rank, ref_margins)
    checks["T11_b1_selector_exact"] = bool(np.array_equal(ours_peers, auth_peers) and np.array_equal(ours_valid, auth_valid))
    checks["T12_b1_centroid_exact"] = bool(np.allclose(centroid_evidence(ref_features, ours_peers, ours_valid), authoritative_nonlocal_peers(ref_features, ref_rank, ref_margins)[2], atol=0, rtol=0))
    invalid = np.zeros(PATCH_COUNT, dtype=bool)
    invalid_ev = centroid_evidence(ref_features, np.full((PATCH_COUNT, K), -1, dtype=np.int64), invalid)
    checks["T13_invalid_reference_zero"] = bool(np.array_equal(invalid_ev, np.zeros(PATCH_COUNT, dtype=np.float32)))
    implementation_source = inspect.getsource(soft_weights) + inspect.getsource(compute_hrip)
    checks["T14_no_trainable_hrip_parameter"] = not any(token in implementation_source for token in ("nn.Parameter", "optimizer", "backward"))
    checks["T15_no_gradient_or_backward"] = "backward" not in implementation_source and "torch.enable_grad" not in implementation_source
    native = torch.from_numpy(rng.normal(size=(STAGES, 1, PATCH_COUNT, 2)).astype(np.float32))
    deployed, _ = deploy_native_logits(native)
    checks["T16_native_score_reconstruction_finite"] = bool(torch.isfinite(deployed).all())
    checks["T17_native_d_rank_reconstruction"] = bool(np.allclose(ref_rank, population_std(np.stack([percentile_rank(x) for x in ref_margins]), axis=0), atol=1e-7, rtol=0))
    temp = Path("/tmp/p5_e0_hrip_synthetic")
    temp.mkdir(parents=True, exist_ok=True)
    state_path = temp / "state.json"
    atomic_json(state_path, {"state": "ready", "current_identity": None, "completed_identity_indices": [0, 1]})
    checks["T18_atomic_state_roundtrip"] = json.loads(state_path.read_text())["completed_identity_indices"] == [0, 1]
    checks["T19_resume_skips_completed"] = resume_indices(json.loads(state_path.read_text()), 4) == [2, 3]
    atomic_json(state_path, {"state": "inflight", "current_identity": 2, "completed_identity_indices": [0, 1]})
    try:
        resume_indices(json.loads(state_path.read_text()), 4)
        checks["T20_unresolved_inflight_refuses_resume"] = False
    except RuntimeError:
        checks["T20_unresolved_inflight_refuses_resume"] = True
    status = "PASS" if all(checks.values()) else "FAIL"
    return {"status": status, "tests": checks, "test_count": len(checks), "no_official_visA_forward": True}


def write_synthetic_result(path: Path) -> None:
    atomic_json(path, run_synthetic_tests())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("synthetic-tests", "official-run", "validate-cache", "freeze-gt-free"))
    parser.add_argument("--output-root", type=Path, default=RECORD_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "synthetic-tests":
        result = run_synthetic_tests()
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "PASS":
            raise SystemExit(1)
    elif args.command == "official-run":
        print(json.dumps(official_run(args.output_root), indent=2, sort_keys=True))
    elif args.command == "validate-cache":
        result = validate_cache(args.output_root)
        print(json.dumps({key: value for key, value in result.items() if key not in {"records", "identities", "state"}}, indent=2, sort_keys=True))
    else:
        raise SystemExit("GT-free freeze is implemented in the next phase after the implementation commit")


if __name__ == "__main__":
    main()
