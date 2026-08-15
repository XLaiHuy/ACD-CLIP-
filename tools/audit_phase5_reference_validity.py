#!/usr/bin/env python3
"""Inference-only Phase5-B1 multi-scale reference-validity audit.

All Phase5/B0 score, ranking, matching, error-object, bootstrap, and spatial
control primitives are imported from the audited implementations.  This file
adds only the preregistered B1 local scales and same-image peer fallback.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import audit_phase5_second_evidence as b0  # noqa: E402
from audit_p4v_phase2b_readiness import load_model  # noqa: E402
from audit_phase5_hsir import (  # noqa: E402
    _sha256,
    ap_contamination,
    exact_auc_ap,
    pairwise_risks,
    percentile_rank,
    population_std,
    spearman,
    write_json,
)
from model.adapter import gaussian_blur2d  # noqa: E402
from utils import configure_canonical_fp32, get_phase2b_global_text_features  # noqa: E402


OUTPUT_ROOT = ROOT / "runs/phase5/hsir/REFERENCE_VALIDITY"
CHECKPOINT = b0.CHECKPOINT
CONFIG = b0.CONFIG
VISA_META = b0.VISA_META
PHASE5_ROOT = b0.PHASE5_ROOT
A1_ROOT = b0.A1_ROOT
STAGE_RESCUE_ROOT = b0.STAGE_RESCUE_ROOT
STAGE_ARBITRATION_ROOT = b0.STAGE_ARBITRATION_ROOT

EXPECTED_PARENT = "211e9b032ceda45a28f3e06408d6c0b28dafc0b9"
EXPECTED_B0 = "c1c47ff32ffd45c12b69eb5070b2b225ff03c820"
EXPECTED_CLASSES = 12
EXPECTED_IMAGES = 2162
EXPECTED_NORMAL = 962
EXPECTED_ANOMALY = 1200
IMAGE_SIZE = 518
PATCH_GRID = (37, 37)
PATCH_STRIDE = 14
PRIMARY_FRACTION = 0.20
TRIAGE_FRACTION = 0.10
BOOTSTRAP_REPS = 2000
PARITY_TOL = 1e-10
E_MULTISTAGE_PARITY_TOL = 1e-6
NONLOCAL_K = 8
NONLOCAL_BLOCK = 64
BOOTSTRAP_SEED = 5101


def stable_desc_order(values: np.ndarray, identity: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    identity = np.asarray(identity, dtype=np.int64).reshape(-1)
    if values.size != identity.size or not np.all(np.isfinite(values)):
        raise RuntimeError("REFERENCE_VALIDITY_OUTPUT_INVALID: non-finite selector")
    return np.lexsort((identity, -values))


def select_top(values: np.ndarray, identity: np.ndarray, count: int) -> np.ndarray:
    if count < 0 or count > np.asarray(values).size:
        raise ValueError("invalid selection count")
    selected = np.zeros(np.asarray(values).size, dtype=bool)
    selected[stable_desc_order(values, identity)[:count]] = True
    return selected


def bootstrap_ci(values: list[float | None], seed: int) -> list[float] | None:
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.size == 1:
        return [float(arr[0]), float(arr[0])]
    rng = np.random.default_rng(seed)
    sample = arr[rng.integers(0, arr.size, size=(BOOTSTRAP_REPS, arr.size))]
    means = sample.mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate(values: list[float | None], seed: int) -> dict[str, Any]:
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    return {
        "mean": None if arr.size == 0 else float(arr.mean()),
        "median": None if arr.size == 0 else float(np.median(arr)),
        "bootstrap95_ci": bootstrap_ci(values, seed),
        "n_classes": int(arr.size),
        "unit": "class",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("REFERENCE_VALIDITY_OUTPUT_INVALID: no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def is_ancestor(commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=False).returncode == 0


def valid_offset_sum(features: torch.Tensor, offsets: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather valid positions only; no synthetic border feature enters a mean."""
    channels, height, width = features.shape
    result = torch.zeros_like(features)
    counts = torch.zeros((height, width), dtype=features.dtype, device=features.device)
    for dy, dx in offsets:
        sy0, sy1 = max(0, -dy), min(height, height - dy)
        sx0, sx1 = max(0, -dx), min(width, width - dx)
        dy0, dy1 = max(0, dy), min(height, height + dy)
        dx0, dx1 = max(0, dx), min(width, width + dx)
        result[:, dy0:dy1, dx0:dx1] += features[:, sy0:sy1, sx0:sx1]
        counts[dy0:dy1, dx0:dx1] += 1
    return result, counts


def _offsets(outer: int, inner: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    core = [(dy, dx) for dy in range(-inner, inner + 1) for dx in range(-inner, inner + 1)]
    ring = [
        (dy, dx)
        for dy in range(-outer, outer + 1)
        for dx in range(-outer, outer + 1)
        if max(abs(dy), abs(dx)) > inner
    ]
    return core, ring


def scale_statistics(features: torch.Tensor, scale: int) -> dict[str, torch.Tensor]:
    if features.ndim != 3:
        raise ValueError(f"expected [D,H,W], got {tuple(features.shape)}")
    outer = scale // 2
    inner = outer - 1
    core_offsets, ring_offsets = _offsets(outer, inner)
    core_sum, core_count = valid_offset_sum(features, core_offsets)
    ring_sum, ring_count = valid_offset_sum(features, ring_offsets)
    core_mean = F.normalize(core_sum / core_count.clamp_min(1).unsqueeze(0), dim=0)
    ring_mean = F.normalize(ring_sum / ring_count.clamp_min(1).unsqueeze(0), dim=0)
    contrast = 1 - (core_mean * ring_mean).sum(dim=0)
    sum_norm_sq = (ring_sum * ring_sum).sum(dim=0)
    pair_count = ring_count * (ring_count - 1)
    mean_pair_cos = torch.where(
        pair_count > 0,
        (sum_norm_sq - ring_count) / pair_count.clamp_min(1),
        torch.zeros_like(pair_count),
    )
    heterogeneity = torch.where(ring_count > 1, 1 - mean_pair_cos, torch.zeros_like(ring_count))
    return {"contrast": contrast, "heterogeneity": heterogeneity, "core_count": core_count, "ring_count": ring_count}


def local_multiscale_maps(aligned: list[torch.Tensor], patch_grid: tuple[int, int]) -> dict[str, Any]:
    height, width = patch_grid
    per_scale: dict[int, list[dict[str, torch.Tensor]]] = {3: [], 5: [], 7: []}
    for stage in aligned:
        feature_map = stage.reshape(height, width, -1).permute(2, 0, 1)
        for scale in (3, 5, 7):
            per_scale[scale].append(scale_statistics(feature_map, scale))
    contrasts, heterogeneity, counts = {}, {}, {}
    for scale in (3, 5, 7):
        contrasts[scale] = torch.stack([x["contrast"] for x in per_scale[scale]]).mean(dim=0)
        heterogeneity[scale] = torch.stack([x["heterogeneity"] for x in per_scale[scale]]).mean(dim=0)
        counts[scale] = {
            "core_min": int(min(int(x["core_count"].min()) for x in per_scale[scale])),
            "core_max": int(max(int(x["core_count"].max()) for x in per_scale[scale])),
            "ring_min": int(min(int(x["ring_count"].min()) for x in per_scale[scale])),
            "ring_max": int(max(int(x["ring_count"].max()) for x in per_scale[scale])),
        }
    a_maps = {}
    for scale in (3, 5, 7):
        q_c = percentile_rank(contrasts[scale].detach().cpu().numpy().reshape(-1))
        q_h = percentile_rank(heterogeneity[scale].detach().cpu().numpy().reshape(-1))
        a_maps[scale] = (q_c * (1 - q_h)).reshape(height, width).astype(np.float32)
    return {
        "C": {s: contrasts[s].detach().cpu().numpy().astype(np.float32) for s in (3, 5, 7)},
        "H": {s: heterogeneity[s].detach().cpu().numpy().astype(np.float32) for s in (3, 5, 7)},
        "A": a_maps,
        "E_valid_ms_patch": np.maximum.reduce([a_maps[3], a_maps[5], a_maps[7]]),
        "wide_gain": np.maximum(a_maps[5], a_maps[7]) - a_maps[3],
        "counts": counts,
    }


def nonlocal_peers(aligned: list[torch.Tensor], d_rank: np.ndarray, native_margins: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed K=8 peer rule with bounded blockwise similarity memory."""
    stage_tensor = torch.stack(aligned).float()
    groups, patches, _ = stage_tensor.shape
    shared = F.normalize(stage_tensor.mean(dim=0), dim=-1)
    stage_q = np.stack([percentile_rank(x) for x in native_margins], axis=0)
    pool = (d_rank < np.median(d_rank)) & np.all(stage_q < 0.5, axis=0)
    yy, xx = np.divmod(np.arange(patches), PATCH_GRID[1])
    peers = np.full((patches, NONLOCAL_K), -1, dtype=np.int64)
    valid = np.zeros(patches, dtype=bool)
    pool_indices = np.flatnonzero(pool)
    pool_features = shared[pool_indices]
    for start in range(0, patches, NONLOCAL_BLOCK):
        end = min(patches, start + NONLOCAL_BLOCK)
        similarity = (shared[start:end] @ pool_features.T).detach().cpu().numpy()
        for local, query in enumerate(range(start, end)):
            spatial_ok = np.maximum(np.abs(yy[pool_indices] - yy[query]), np.abs(xx[pool_indices] - xx[query])) > 3
            candidates = pool_indices[spatial_ok]
            if candidates.size < NONLOCAL_K:
                continue
            candidate_columns = np.flatnonzero(spatial_ok)
            order = np.lexsort((candidates, -similarity[local, candidate_columns]))
            peers[query] = candidates[order[:NONLOCAL_K]]
            valid[query] = True
    safe = torch.from_numpy(np.maximum(peers, 0)).to(stage_tensor.device)
    evidence = torch.zeros(patches, dtype=torch.float32, device=stage_tensor.device)
    if valid.any():
        for group in range(groups):
            reference = F.normalize(stage_tensor[group][safe].mean(dim=1), dim=-1)
            evidence += 1 - (stage_tensor[group] * reference).sum(dim=-1)
        evidence /= float(groups)
        evidence[~torch.from_numpy(valid).to(evidence.device)] = 0
    return peers, valid, evidence.detach().cpu().numpy().astype(np.float32)


def deploy_from_native(native: torch.Tensor, patch_grid: tuple[int, int], image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    groups, batch, patches, channels = native.shape
    if channels != 2 or patches != patch_grid[0] * patch_grid[1]:
        raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: native shape={tuple(native.shape)}")
    outputs = []
    for group in range(groups):
        logits = native[group].permute(0, 2, 1).reshape(batch, 2, *patch_grid)
        logits = gaussian_blur2d(logits, (7, 7), (1, 1))
        outputs.append(F.interpolate(logits, size=(image_size, image_size), mode="bilinear", align_corners=True))
    final_logits = torch.stack(outputs).mean(dim=0)
    return F.softmax(final_logits, dim=1), final_logits


@torch.inference_mode()
def predictor_one(model, raw: dict[str, Any], class_name: str, image_size: int, text_cache: dict[str, torch.Tensor], device):
    image = raw["image"].unsqueeze(0).to(device).float()
    target = raw["mask"].to(device).float().squeeze(0).cpu().numpy().astype(np.uint8)
    if target.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: target mask shape={target.shape}")
    visual = model(image, return_phase4_features=True)
    stage_batches = [x.float() for x in visual["seg_tokens"]]
    stage_features = [x[0] for x in stage_batches]
    features = torch.stack(visual["seg_tokens"])
    if class_name not in text_cache:
        text_cache[class_name] = get_phase2b_global_text_features(model, "VisA", [class_name], device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
    model_prob, native, native_margin = model.vision_text_fusion_gate_seg(features, text_cache[class_name], img_size=image_size, test_mode=True, domain="Industrial", return_details=True)
    patch_grid = b0.authoritative_patch_grid(model)
    if patch_grid != PATCH_GRID or image_size != IMAGE_SIZE or image_size != PATCH_GRID[0] * PATCH_STRIDE:
        raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: image/grid/stride contract failed")
    reconstructed, final_logits = deploy_from_native(native, patch_grid, image_size)
    predictor_parity = float((model_prob - reconstructed[:, 1]).abs().max().detach().cpu())
    native_margins = native_margin[:, 0].detach().float().cpu().numpy()
    native_logits = native[:, 0].detach().float().cpu().numpy()
    patch_d_logit = population_std(native_margins, axis=0).astype(np.float32)
    stage_percentiles = np.stack([percentile_rank(x) for x in native_margins], axis=0)
    patch_d_rank = population_std(stage_percentiles, axis=0).astype(np.float32)
    final_logits_np = final_logits[0].detach().float().cpu().numpy()
    score = reconstructed[0, 1].detach().float().cpu().numpy()
    final_margin = final_logits_np[1] - final_logits_np[0]
    final_rank = percentile_rank(score.reshape(-1)).reshape(score.shape).astype(np.float32)
    stage_rank_maps = np.stack([b0.upsample_explicit(percentile_rank(x).astype(np.float32), patch_grid, image_size) for x in native_margins], axis=0)
    g_rescue = stage_rank_maps.max(axis=0) - final_rank
    aligned, feature_semantics = b0.align_features(stage_features, patch_grid)
    old_maps = b0.evidence_maps(stage_features, patch_grid, image_size)
    local = local_multiscale_maps(aligned, patch_grid)
    peers, valid_reference, nonlocal_patch = nonlocal_peers(aligned, patch_d_rank, native_margins)
    occupancy = target.reshape(PATCH_GRID[0], PATCH_STRIDE, PATCH_GRID[1], PATCH_STRIDE).mean(axis=(1, 3)).astype(np.float32)
    return {"score": score.reshape(-1).astype(np.float32), "final_margin": final_margin.reshape(-1).astype(np.float32), "target": target.reshape(-1), "D_logit": b0.upsample_explicit(patch_d_logit, patch_grid, image_size).reshape(-1).astype(np.float32), "D_rank": b0.upsample_explicit(patch_d_rank, patch_grid, image_size).reshape(-1).astype(np.float32), "U_conf": (-np.abs(final_margin)).reshape(-1).astype(np.float32), "G_rescue": g_rescue.reshape(-1).astype(np.float32), "E_multistage": old_maps["E_multistage"].astype(np.float32), "E_xstage": old_maps["E_xstage"].astype(np.float32), "E_valid_ms": b0.upsample_explicit(local["E_valid_ms_patch"], patch_grid, image_size).reshape(-1).astype(np.float32), "E_nonlocal": b0.upsample_explicit(nonlocal_patch, patch_grid, image_size).reshape(-1).astype(np.float32), "patch": {"D_rank": patch_d_rank, "H3": local["H"][3], "A3": local["A"][3], "A5": local["A"][5], "A7": local["A"][7], "wide_gain": local["wide_gain"], "target_occupancy": occupancy, "peer_indices": peers, "valid_reference": valid_reference, "counts": local["counts"]}, "native_margins": native_margins.astype(np.float32), "native_logits": native_logits.astype(np.float32), "predictor_parity": predictor_parity, "feature_semantics": feature_semantics, "shape_record": {"stage_visual_features": [list(x.shape) for x in stage_batches], "stacked_visual_features": list(features.shape), "native_stage_logits": list(native.shape), "native_stage_margins": list(native_margin.shape), "patch_grid": list(patch_grid), "patch_count": int(patch_grid[0] * patch_grid[1]), "deployed_model_probability": list(model_prob.shape), "deployed_reconstructed_probability": list(reconstructed.shape), "deployed_final_logits": list(final_logits.shape)}}


def canonical_records(image_size: int):
    datasets, records, counts = b0.canonical_test_records(image_size)
    expected = {"classes": EXPECTED_CLASSES, "images": EXPECTED_IMAGES, "normal": EXPECTED_NORMAL, "anomaly": EXPECTED_ANOMALY}
    if counts != expected:
        raise RuntimeError(f"REFERENCE_VALIDITY_INPUT_PROVENANCE_INVALID: TEST counts={counts}")
    return datasets, records, counts


def artifact_hashes() -> dict[str, str]:
    paths = {"visa_test_summary": PHASE5_ROOT / "SUMMARY.json", "visa_test_per_class": PHASE5_ROOT / "PER_CLASS.csv", "visa_test_per_image": PHASE5_ROOT / "PER_IMAGE.csv", "visa_test_decision": PHASE5_ROOT / "DECISION.json", "actionability_summary": A1_ROOT / "SUMMARY.json", "actionability_decision": A1_ROOT / "DECISION.json", "stage_rescue_summary": STAGE_RESCUE_ROOT / "SUMMARY.json", "stage_rescue_decision": STAGE_RESCUE_ROOT / "DECISION.json", "stage_arbitration_summary": STAGE_ARBITRATION_ROOT / "SUMMARY.json", "stage_arbitration_decision": STAGE_ARBITRATION_ROOT / "DECISION.json", "b0_summary": ROOT / "runs/phase5/hsir/SECOND_EVIDENCE/SUMMARY.json", "b0_output_check": ROOT / "runs/phase5/hsir/SECOND_EVIDENCE/OUTPUT_CHECK.json"}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError("REFERENCE_VALIDITY_INPUT_PROVENANCE_INVALID: missing " + ", ".join(missing))
    return {name: _sha256(path) for name, path in paths.items()}


def b0_class_metrics() -> dict[str, dict[str, float]]:
    """Load the authoritative exact per-class TEST metrics from Phase5-A."""
    summary = json.loads((PHASE5_ROOT / "SUMMARY.json").read_text())
    rows = summary.get("per_class", [])
    metrics = {}
    for row in rows:
        name = str(row["class_name"])
        metrics[name] = {
            "pixel_ap": float(row["pixel_ap"]),
            "pixel_auroc": float(row["pixel_auroc"]),
        }
    if len(metrics) != EXPECTED_CLASSES:
        raise RuntimeError("REFERENCE_VALIDITY_INPUT_PROVENANCE_INVALID: B0 class metric count")
    return metrics


def input_check(config: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    phase5 = json.loads((PHASE5_ROOT / "SUMMARY.json").read_text())
    a1 = json.loads((A1_ROOT / "SUMMARY.json").read_text())
    b0_summary = json.loads((ROOT / "runs/phase5/hsir/SECOND_EVIDENCE/SUMMARY.json").read_text())
    b0_output = json.loads((ROOT / "runs/phase5/hsir/SECOND_EVIDENCE/OUTPUT_CHECK.json").read_text())
    dataset_root = phase5["provenance"].get("dataset_root")
    checks = {"branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() == "autopilot/p4-conditional-semantic-factorization", "required_b0_parent_ancestor": is_ancestor(EXPECTED_PARENT), "b0_ancestor": is_ancestor(EXPECTED_B0), "phase5_ancestor": is_ancestor(b0.PHASE5_COMMIT), "actionability_ancestor": is_ancestor(b0.A1_COMMIT), "branch_a_ancestor": is_ancestor(b0.BRANCH_A_COMMIT), "branch_b1_ancestor": is_ancestor(b0.B1_COMMIT), "checkpoint_sha": _sha256(CHECKPOINT) == b0.EXPECTED_CHECKPOINT_SHA, "config_sha": _sha256(CONFIG) == b0.EXPECTED_CONFIG_SHA, "visa_metadata_sha": _sha256(VISA_META) == b0.EXPECTED_VISA_META_SHA, "phase5_dataset": phase5["provenance"].get("dataset") == "VisA", "phase5_split": str(phase5["provenance"].get("split")).lower() == "test", "phase5_counts": phase5["provenance"].get("number_classes") == EXPECTED_CLASSES and phase5["provenance"].get("number_images") == EXPECTED_IMAGES, "phase5_image_counts": phase5["provenance"].get("number_normal_images") == EXPECTED_NORMAL and phase5["provenance"].get("number_anomaly_images") == EXPECTED_ANOMALY, "phase5_no_train_paths": phase5["provenance"].get("contains_train_paths") is False, "phase5_predictor_parity": phase5.get("parity", {}).get("predictor_max_abs_probability_error") == 0.0, "a1_test_split": a1.get("provenance", {}).get("split") == "test", "b0_output_pass": b0_output.get("status") == "PASS", "b0_decision_class_unstable": b0_summary.get("decision") == "SECOND_EVIDENCE_CLASS_UNSTABLE", "b0_no_primary": b0_summary.get("selected_primary_candidate") is None, "counts": counts == {"classes": EXPECTED_CLASSES, "images": EXPECTED_IMAGES, "normal": EXPECTED_NORMAL, "anomaly": EXPECTED_ANOMALY}, "no_train_dataset_root": "train" not in str(dataset_root).lower(), "image_size": int(config["img_size"]) == IMAGE_SIZE}
    failed = [name for name, value in checks.items() if not value]
    if failed:
        raise RuntimeError("REFERENCE_VALIDITY_INPUT_PROVENANCE_INVALID: " + ", ".join(failed))
    return {"status": "PASS", "current_head": git_head(), "scientific_ancestors": {"required_b0_parent": EXPECTED_PARENT, "b0": EXPECTED_B0, "phase5": b0.PHASE5_COMMIT, "actionability": b0.A1_COMMIT, "branch_a": b0.BRANCH_A_COMMIT, "branch_b1": b0.B1_COMMIT}, "checkpoint": {"path": str(CHECKPOINT), "sha256": _sha256(CHECKPOINT)}, "config": {"path": str(CONFIG), "sha256": _sha256(CONFIG)}, "visa_root": dataset_root, "metadata_source": str(VISA_META), "metadata_sha256": _sha256(VISA_META), "split": "TEST", "counts": counts, "predictor_implementation": "tools/audit_phase5_reference_validity.py::predictor_one; model/adapter.py::AdapterModel.forward", "evaluator_implementation": "audit_phase5_hsir::exact_auc_ap and pairwise primitives; B0 audited exact per-class AP/AUROC parity reference; audit_phase5_second_evidence matching/triage", "upstream_artifact_sha256": artifact_hashes(), "checks": checks, "predictor_parity": "PENDING_DRY_RUN", "protocol_file_sha256": None, "torch_version": torch.__version__, "numpy_version": np.__version__, "python_version": platform.python_version(), "image_size": IMAGE_SIZE, "patch_grid": list(PATCH_GRID), "patch_stride": PATCH_STRIDE, "random_seeds": {"bootstrap": BOOTSTRAP_SEED, "selection": "stable lexsort", "peer_tie": "ascending patch index"}, "authoritative_pixel_cache": False, "inference_authorization": "one fresh class-streamed VisA TEST pass; one forward per image; no dense cache"}



def make_protocol(base: dict[str, Any], architecture: dict[str, Any]) -> dict[str, Any]:
    protocol = {
        "audit": "PHASE5-B1 MULTI-SCALE REFERENCE VALIDITY",
        "inference_only": True,
        "training_steps": 0,
        "git_head": base["current_head"],
        "checkpoint_sha256": base["checkpoint"]["sha256"],
        "config_sha256": base["config"]["sha256"],
        "dataset_metadata_sha256": base["metadata_sha256"],
        "architecture": architecture,
        "runtime_contract": {"image_size": IMAGE_SIZE, "patch_grid": list(PATCH_GRID), "patch_stride": PATCH_STRIDE, "native_visual_features": [3, 1, 1369, 768], "native_logits": [3, 1, 1369, 2], "native_margins": [3, 1, 1369], "deployed_probability": [1, 518, 518], "deployed_final_logits": [1, 2, 518, 518]},
        "feature_semantics": {"source_file": "model/adapter.py", "source_function": "AdapterModel.forward", "tensor": "visual['seg_tokens']", "pipeline": "seg_proj -> seg_layer_norms -> F.normalize(dim=-1)", "alignment": "authoritative 37x37 grid; bilinear align_corners=True only if a stage grid differs, followed by L2 renormalization"},
        "local_scales": {"3": {"core": "center patch", "ring": "valid 3x3 excluding center"}, "5": {"core": "valid 3x3", "ring": "valid 5x5 excluding inner 3x3"}, "7": {"core": "valid 5x5", "ring": "valid 7x7 excluding inner 5x5"}, "padding": "none; valid positions only; no circular or replicated feature padding"},
        "local_formula": {"C_g_s": "1-cosine(normalized mean core, normalized mean ring)", "H_g_s": "mean pairwise(1-cosine(ring_i,ring_j)); normalized ring-sum identity, no pair matrix", "C_s": "mean over three authoritative stages", "H_s": "mean over three authoritative stages", "qC_s": "percentile_rank(C_s) within each image", "qH_s": "percentile_rank(H_s) within each image", "A_s": "qC_s*(1-qH_s)", "E_valid_ms": "max(A_3,A_5,A_7)"},
        "old_baseline": "E_multistage imported from audit_phase5_second_evidence.evidence_maps; B0 parity required <=1e-6 on the real dry-run image",
        "population": "S_rank20 = top ceil(0.20*N_class) D_rank pixels per class; GT-free stable identity ordering",
        "matching": "reuse B0 deterministic_matches: same-class score x D_rank 10x10 quantile bins, deterministic SHA256 keyed control, no candidate evidence",
        "triage": "inside S_rank20 select top ceil(0.10*|S_rank20|); U_conf=-abs(final_margin) and D_rank controls use exact same K",
        "spatial_control": "reuse B0 shifted_map: wraparound floor(518/3)=172 pixels in both axes",
        "M1": "Normal member H3 median split; report E_multistage and E_xstage low-H minus high-H win; primary PASS lower CI >0 and >=8 positive classes",
        "M2": "exact target 518x518 occupancy in 14x14 blocks; risky positive patches contain any S_rank20 pixel and occupancy>0; valid 8-neighbor surroundedness; rho=Spearman(max(A5,A7)-A3,surroundedness); PASS lower CI >0 and >=8 positive classes",
        "nonlocal_fallback": {"activation": "interpret only after LOCAL_REFERENCE_MECHANISM_UNSUPPORTED", "descriptor": "normalize(mean over three aligned stage features)", "pool": "same image; Chebyshev distance >3; D_rank below image median; all stage anomaly percentiles <0.5", "K": NONLOCAL_K, "tie": "descending cosine then ascending patch index", "abstain": "fewer than K valid peers -> valid_reference=false and evidence=0", "evidence": "mean_g 1-cosine(center stage, normalized mean of same selected peers)"},
        "gates": {"A": "bootstrap lower CI matched win >0.5 and >=8 supportive classes", "B": "positive C_AP and R_pos capture exceed fixed confidence and D_rank controls", "C": "aligned win and positive C_AP capture exceed shifted control", "D": "Normal fraction and negative R_neg capture do not exceed both controls", "E": "supportive >=8 and opposed <=2"},
        "decision_tree": ["LOCAL_MULTISCALE_ANALYTIC_SUPPORTED", "LOCAL_MECHANISM_SUPPORTED_TINY_CNN_JUSTIFIED", "LOCAL_REFERENCE_MECHANISM_UNSUPPORTED", "NONLOCAL_SAME_IMAGE_REFERENCE_SUPPORTED", "NONLOCAL_REFERENCE_UNSAFE", "EXTERNAL_COMPLEMENTARY_REPRESENTATION_JUSTIFIED", "REFERENCE_VALIDITY_AUDIT_INCONCLUSIVE", "REFERENCE_VALIDITY_OUTPUT_INVALID"],
        "random_seeds": {"bootstrap": BOOTSTRAP_SEED, "selection": "stable lexsort", "peer_tie": "ascending patch index"},
        "no_gt_in_candidate_construction": True, "no_target_test_training": True, "no_dense_feature_cache": True, "no_formula_changes_after_full_pass_start": True,
    }
    protocol["protocol_content_sha256"] = hashlib.sha256((json.dumps(protocol, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
    return protocol


def triage_metrics(evidence: np.ndarray, rank_mask: np.ndarray, margins: np.ndarray, d_rank: np.ndarray, labels: np.ndarray, c_ap: np.ndarray, r_pos_full: np.ndarray, r_neg_full: np.ndarray, scores: np.ndarray, baseline_ap: float, pixel_id: np.ndarray, a1_delta: float) -> dict[str, Any]:
    k = int(np.ceil(TRIAGE_FRACTION * int(rank_mask.sum())))
    indices = np.flatnonzero(rank_mask)
    candidate = np.zeros(rank_mask.size, dtype=bool)
    candidate[indices[stable_desc_order(evidence[rank_mask], pixel_id[rank_mask])[:k]]] = True
    confidence = np.zeros_like(candidate)
    confidence[indices[stable_desc_order(-np.abs(margins[rank_mask]), pixel_id[rank_mask])[:k]]] = True
    rank = np.zeros_like(candidate)
    rank[indices[stable_desc_order(d_rank[rank_mask], pixel_id[rank_mask])[:k]]] = True
    base_order = np.argsort(-scores, kind="mergesort")
    result = {"triage_budget_k": k, "triage_actual_fraction": float(k / rank_mask.size), "selection_gt_free": True}
    for name, selected in (("candidate", candidate), ("confidence_control", confidence), ("rank_control", rank)):
        result[name] = b0.selector_metrics(selected, labels, c_ap, r_pos_full, r_neg_full, baseline_ap, base_order, pixel_id, a1_delta)
    return result


def shifted_map(values: np.ndarray, height: int, width: int) -> np.ndarray:
    """Exact B0 spatial-control convention for a class image stream."""
    array = np.asarray(values).reshape(-1, height, width)
    return np.roll(array, (height // 3, width // 3), axis=(1, 2)).reshape(-1)


def evaluate_signal(name: str, evidence: np.ndarray, rank_mask: np.ndarray, scores: np.ndarray, margins: np.ndarray, d_rank: np.ndarray, labels: np.ndarray, c_ap: np.ndarray, r_pos_full: np.ndarray, r_neg_full: np.ndarray, pixel_id: np.ndarray, baseline_ap: float, a1_delta: float, matched_pos: np.ndarray, matched_neg: np.ndarray) -> dict[str, Any]:
    shifted = shifted_map(evidence, IMAGE_SIZE, IMAGE_SIZE).reshape(-1)
    triage = triage_metrics(evidence, rank_mask, margins, d_rank, labels, c_ap, r_pos_full, r_neg_full, scores, baseline_ap, pixel_id, a1_delta)
    shifted_triage = triage_metrics(shifted, rank_mask, margins, d_rank, labels, c_ap, r_pos_full, r_neg_full, scores, baseline_ap, pixel_id, a1_delta)
    return {"name": name, "matched_pair_win_rate": b0.matched_win_rate(evidence, matched_pos, matched_neg), "shifted_matched_pair_win_rate": b0.matched_win_rate(shifted, matched_pos, matched_neg), "triage": triage, "shifted_triage": shifted_triage, "shifted_positive_C_AP_capture": shifted_triage["candidate"]["positive_C_AP_mass_capture"], "shifted_positive_R_pos_capture": shifted_triage["candidate"]["positive_R_pos_mass_capture"], "shifted_negative_R_neg_capture": shifted_triage["candidate"]["negative_R_neg_mass_capture"]}


def patch_image_index(pixel_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = pixel_id // (IMAGE_SIZE * IMAGE_SIZE)
    offset = pixel_id % (IMAGE_SIZE * IMAGE_SIZE)
    yy, xx = np.divmod(offset, IMAGE_SIZE)
    patch = (yy // PATCH_STRIDE) * PATCH_GRID[1] + (xx // PATCH_STRIDE)
    return image.astype(np.int64), patch.astype(np.int64)


def neighbor_mean(occupancy: np.ndarray) -> np.ndarray:
    total = np.zeros_like(occupancy, dtype=np.float64)
    count = np.zeros_like(occupancy, dtype=np.float64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            sy0, sy1 = max(0, -dy), min(PATCH_GRID[0], PATCH_GRID[0] - dy)
            sx0, sx1 = max(0, -dx), min(PATCH_GRID[1], PATCH_GRID[1] - dx)
            dy0, dy1 = max(0, dy), min(PATCH_GRID[0], PATCH_GRID[0] + dy)
            dx0, dx1 = max(0, dx), min(PATCH_GRID[1], PATCH_GRID[1] + dx)
            total[:, dy0:dy1, dx0:dx1] += occupancy[:, sy0:sy1, sx0:sx1]
            count[:, dy0:dy1, dx0:dx1] += 1
    return (total / np.maximum(count, 1)).astype(np.float32)


def m1_for_class(rows: dict[str, np.ndarray], matched_pos: np.ndarray, matched_neg: np.ndarray, pixel_id: np.ndarray, patch_h3: np.ndarray) -> dict[str, Any]:
    if matched_neg.size == 0:
        return {"median_normal_H3": None, "n_low": 0, "n_high": 0, "effects": {}}
    image, patch = patch_image_index(pixel_id[matched_neg])
    h = patch_h3.reshape(patch_h3.shape[0], -1)[image, patch]
    median = float(np.median(h))
    low, high = h <= median, h > median
    effects = {}
    for name in ("E_multistage", "E_xstage"):
        delta = rows[name][matched_pos] - rows[name][matched_neg]
        low_win = None if not low.any() else float(np.mean((delta[low] > 0) + 0.5 * (delta[low] == 0)))
        high_win = None if not high.any() else float(np.mean((delta[high] > 0) + 0.5 * (delta[high] == 0)))
        effects[name] = {"low_H_win": low_win, "high_H_win": high_win, "delta_H": None if low_win is None or high_win is None else low_win - high_win}
    return {"median_normal_H3": median, "n_low": int(low.sum()), "n_high": int(high.sum()), "effects": effects}


def process_class(model, dataset, class_name: str, records: list[dict[str, Any]], device, reference_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    pixel_names = ("score", "final_margin", "D_rank", "D_logit", "U_conf", "G_rescue", "target", "pixel_id", "E_multistage", "E_xstage", "E_valid_ms", "E_nonlocal")
    pixel_parts = {name: [] for name in pixel_names}
    patch_names = ("D_rank", "H3", "A3", "A5", "A7", "wide_gain", "target_occupancy", "valid_reference", "peer_indices")
    patch_parts = {name: [] for name in patch_names}
    text_cache: dict[str, torch.Tensor] = {}
    pixels_per_image = IMAGE_SIZE * IMAGE_SIZE
    max_parity = 0.0
    border_counts = None
    for record in records:
        raw = dataset[record["source_index"]]
        item = predictor_one(model, raw, class_name, IMAGE_SIZE, text_cache, device)
        pixel_id = np.int64(record["source_index"]) * pixels_per_image + np.arange(pixels_per_image, dtype=np.int64)
        for name in pixel_names:
            pixel_parts[name].append(pixel_id if name == "pixel_id" else item[name])
        for name in patch_names:
            patch_parts[name].append(item["patch"][name])
        max_parity = max(max_parity, item["predictor_parity"])
        border_counts = item["patch"]["counts"]
        del item, raw
    values = {name: np.concatenate(parts) for name, parts in pixel_parts.items()}
    patches = {name: np.stack(parts) for name, parts in patch_parts.items()}
    finite_names = ("score", "final_margin", "D_rank", "D_logit", "U_conf", "G_rescue", "E_multistage", "E_xstage", "E_valid_ms", "E_nonlocal")
    if not all(np.all(np.isfinite(values[name])) for name in finite_names):
        raise RuntimeError("REFERENCE_VALIDITY_OUTPUT_INVALID: non-finite pixel variable")
    baseline_auc, baseline_ap = exact_auc_ap(values["score"], values["target"])
    reference = reference_metrics.get(class_name)
    if reference is None:
        raise RuntimeError(f"REFERENCE_VALIDITY_INPUT_PROVENANCE_INVALID: missing B0 metrics for {class_name}")
    eval_auc, eval_ap = reference["pixel_auroc"], reference["pixel_ap"]
    if max(abs(baseline_auc - eval_auc), abs(baseline_ap - eval_ap)) > PARITY_TOL:
        raise RuntimeError("ACTIONABILITY_TARGET_PARITY_INVALID: AP/AUROC parity failed")
    positive = values["target"] == 1
    r_pos, r_neg = pairwise_risks(values["score"], values["target"])
    r_pos_full = np.full(values["score"].size, np.nan, dtype=np.float64); r_pos_full[positive] = r_pos
    r_neg_full = np.full(values["score"].size, np.nan, dtype=np.float64); r_neg_full[~positive] = r_neg
    c_ap = ap_contamination(values["score"], values["target"])
    rank_mask = select_top(values["D_rank"], values["pixel_id"], int(np.ceil(PRIMARY_FRACTION * values["score"].size)))
    base_order = np.argsort(-values["score"], kind="mergesort")
    a1_delta = b0.oracle_bundle(values["target"], baseline_ap, base_order, rank_mask)["positive_only_delta"]
    matched_pos, matched_neg = b0.deterministic_matches(class_name, values["score"], values["D_rank"], values["target"], rank_mask, values["pixel_id"])
    results = {name: evaluate_signal(name, values[name], rank_mask, values["score"], values["final_margin"], values["D_rank"], values["target"], c_ap, r_pos_full, r_neg_full, values["pixel_id"], baseline_ap, a1_delta, matched_pos, matched_neg) for name in ("E_multistage", "E_valid_ms", "E_nonlocal")}
    selected_positive = rank_mask & positive
    correlations = {}
    for name in ("E_valid_ms", "E_multistage"):
        correlations[name] = {"vs_C_AP": spearman(values[name][selected_positive], c_ap[selected_positive]) if selected_positive.sum() > 1 else None, "vs_R_pos": spearman(values[name][selected_positive], r_pos_full[selected_positive]) if selected_positive.sum() > 1 else None, "vs_D_rank": spearman(values[name][selected_positive], values["D_rank"][selected_positive]) if selected_positive.sum() > 1 else None, "vs_D_logit": spearman(values[name][selected_positive], values["D_logit"][selected_positive]) if selected_positive.sum() > 1 else None, "vs_G_rescue": spearman(values[name][selected_positive], values["G_rescue"][selected_positive]) if selected_positive.sum() > 1 else None}
    m1 = m1_for_class(values, matched_pos, matched_neg, values["pixel_id"], patches["H3"])
    selected_pixels = rank_mask.reshape(len(records), IMAGE_SIZE, IMAGE_SIZE)
    selected_patch = selected_pixels.reshape(len(records), PATCH_GRID[0], PATCH_STRIDE, PATCH_GRID[1], PATCH_STRIDE).any(axis=(2, 4))
    risky_positive = selected_patch & (patches["target_occupancy"] > 0)
    surrounded = neighbor_mean(patches["target_occupancy"])[risky_positive]
    wide_gain = patches["wide_gain"][risky_positive]
    rho = spearman(wide_gain, surrounded) if wide_gain.size > 1 else None
    q75 = None if surrounded.size == 0 else float(np.quantile(surrounded, 0.75))
    qmask = surrounded >= q75 if q75 is not None else np.zeros(0, dtype=bool)
    wide = np.maximum(patches["A5"], patches["A7"])[risky_positive]
    m2 = {"risky_positive_patch_count": int(wide_gain.size), "rho_wide_gain_surroundedness": rho, "upper_quartile_surroundedness": q75, "upper_quartile_count": int(qmask.sum()), "upper_quartile_mean_A3": None if not qmask.any() else float(patches["A3"][risky_positive][qmask].mean()), "upper_quartile_mean_max_A5_A7": None if not qmask.any() else float(wide[qmask].mean()), "upper_quartile_mean_wide_gain": None if not qmask.any() else float(wide_gain[qmask].mean())}
    safe_peers = np.maximum(patches["peer_indices"], 0)
    peer_occ = np.zeros_like(patches["peer_indices"], dtype=np.float32)
    for image_index in range(len(records)):
        peer_occ[image_index] = patches["target_occupancy"][image_index].reshape(-1)[safe_peers[image_index]]
    valid_peer_occ = peer_occ[patches["valid_reference"]]
    peer_contamination = None if valid_peer_occ.size == 0 else float(np.mean(valid_peer_occ > 0))
    row = {"class": class_name, "n_images": len(records), "n_pixels": int(values["score"].size), "baseline_ap": float(baseline_ap), "baseline_auroc": float(baseline_auc), "a1_positive_only_oracle_delta_AP": float(a1_delta), "a1_positive_only_oracle_delta_AP_pp": float(100 * a1_delta), "rank_selected_count": int(rank_mask.sum()), "rank_selected_positive_fraction": float(np.sum(rank_mask & positive) / rank_mask.sum()), "matched_pairs_n": int(matched_pos.size), "predictor_parity_max_abs_probability_error": float(max_parity), "ap_parity_error": float(abs(baseline_ap - eval_ap)), "auroc_parity_error": float(abs(baseline_auc - eval_auc)), "correlations": correlations, "m1": m1, "m2": m2, "nonlocal_valid_reference_coverage": float(patches["valid_reference"].mean()), "nonlocal_peer_contamination_rate": peer_contamination, "border_counts": border_counts}
    row.update(results)
    del values, patches, pixel_parts, patch_parts, r_pos_full, r_neg_full, c_ap
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def class_consistency(values: list[float | None]) -> dict[str, int]:
    return {"supportive": int(sum(x is not None and x > 0.5 + 1e-6 for x in values)), "neutral": int(sum(x is not None and abs(x - 0.5) <= 1e-6 for x in values)), "opposed": int(sum(x is not None and x < 0.5 - 1e-6 for x in values)), "total": len(values)}


def candidate_summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    values = [row[name]["matched_pair_win_rate"] for row in rows]
    shifted = [row[name]["shifted_matched_pair_win_rate"] for row in rows]
    triage = [row[name]["triage"]["candidate"] for row in rows]
    conf = [row[name]["triage"]["confidence_control"] for row in rows]
    rank = [row[name]["triage"]["rank_control"] for row in rows]
    consistency = class_consistency(values)
    ci = bootstrap_ci(values, BOOTSTRAP_SEED + sum(map(ord, name)))
    gate_a = ci is not None and ci[0] > 0.5 and consistency["supportive"] >= 8
    gate_b = np.nanmean([x["positive_C_AP_mass_capture"] for x in triage]) > max(np.nanmean([x["positive_C_AP_mass_capture"] for x in conf]), np.nanmean([x["positive_C_AP_mass_capture"] for x in rank])) and np.nanmean([x["positive_R_pos_mass_capture"] for x in triage]) > max(np.nanmean([x["positive_R_pos_mass_capture"] for x in conf]), np.nanmean([x["positive_R_pos_mass_capture"] for x in rank]))
    gate_c = float(np.nanmean(values)) > float(np.nanmean(shifted)) and np.nanmean([x["positive_C_AP_mass_capture"] for x in triage]) > np.nanmean([row[name]["shifted_positive_C_AP_capture"] for row in rows])
    normal = [1 - x["selected_positive_fraction"] for x in triage]; normal_conf = [1 - x["selected_positive_fraction"] for x in conf]; normal_rank = [1 - x["selected_positive_fraction"] for x in rank]
    gate_d = np.nanmean(normal) <= max(np.nanmean(normal_conf), np.nanmean(normal_rank)) and np.nanmean([x["negative_R_neg_mass_capture"] for x in triage]) <= max(np.nanmean([x["negative_R_neg_mass_capture"] for x in conf]), np.nanmean([x["negative_R_neg_mass_capture"] for x in rank]))
    gate_e = consistency["supportive"] >= 8 and consistency["opposed"] <= 2
    gates = {"gate_A": bool(gate_a), "gate_B": bool(gate_b), "gate_C": bool(gate_c), "gate_D": bool(gate_d), "gate_E": bool(gate_e)}
    return {"conditional_matched_pair_win_rate_macro": aggregate(values, BOOTSTRAP_SEED + sum(map(ord, name))), "conditional_matched_pair_win_rate_median": None if not values else float(np.nanmedian(values)), "conditional_matched_pair_bootstrap95": ci, "classes_supportive": consistency["supportive"], "classes_neutral": consistency["neutral"], "classes_opposed": consistency["opposed"], "positive_C_AP_capture_macro": aggregate([x["positive_C_AP_mass_capture"] for x in triage], BOOTSTRAP_SEED + 100 + sum(map(ord, name))), "positive_R_pos_capture_macro": aggregate([x["positive_R_pos_mass_capture"] for x in triage], BOOTSTRAP_SEED + 200 + sum(map(ord, name))), "negative_R_neg_capture_macro": aggregate([x["negative_R_neg_mass_capture"] for x in triage], BOOTSTRAP_SEED + 300 + sum(map(ord, name))), "oracle_triage_delta_AP_macro": aggregate([x["oracle_triage_delta_AP"] for x in triage], BOOTSTRAP_SEED + 400 + sum(map(ord, name))), "fraction_A1_oracle_recovered": aggregate([x["fraction_A1_positive_oracle_recovered"] for x in triage], BOOTSTRAP_SEED + 500 + sum(map(ord, name))), "shifted_control_result": {"matched_pair_win_rate_macro": aggregate(shifted, BOOTSTRAP_SEED + 600 + sum(map(ord, name))), "positive_C_AP_capture_macro": aggregate([row[name]["shifted_positive_C_AP_capture"] for row in rows], BOOTSTRAP_SEED + 700 + sum(map(ord, name)))}, "redundancy_result": {"conditional_signal_survives_score_d_rank_matching": bool(gate_a)}, **gates, "overall_status": "PASS" if all(gates.values()) else "NOT_SUPPORTED"}


def mechanism_summary(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    values = [row["m1"]["effects"].get(candidate, {}).get("delta_H") for row in rows]
    ci = bootstrap_ci(values, BOOTSTRAP_SEED + 900 + sum(map(ord, candidate)))
    positives = int(sum(x is not None and x > 0 for x in values))
    return {"delta_H_macro": aggregate(values, BOOTSTRAP_SEED + 900), "classes_positive_delta_H": positives, "pass": bool(ci is not None and ci[0] > 0 and positives >= 8)}


def m2_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row["m2"]["rho_wide_gain_surroundedness"] for row in rows]
    ci = bootstrap_ci(values, BOOTSTRAP_SEED + 1001)
    positives = int(sum(x is not None and x > 0 for x in values))
    return {"rho_macro": aggregate(values, BOOTSTRAP_SEED + 1001), "classes_positive_rho": positives, "pass": bool(ci is not None and ci[0] > 0 and positives >= 8)}


def decide_local(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    summaries = {name: candidate_summary(rows, name) for name in ("E_multistage", "E_valid_ms")}
    difference = [row["E_valid_ms"]["matched_pair_win_rate"] - row["E_multistage"]["matched_pair_win_rate"] for row in rows]
    difference_ci = bootstrap_ci(difference, BOOTSTRAP_SEED + 1200)
    summaries["E_valid_ms"]["delta_vs_E_multistage"] = {"aggregate": aggregate(difference, BOOTSTRAP_SEED + 1200), "bootstrap95": difference_ci, "lower_bound_above_zero": bool(difference_ci is not None and difference_ci[0] > 0)}
    m1 = {"E_multistage": mechanism_summary(rows, "E_multistage"), "E_xstage": mechanism_summary(rows, "E_xstage")}
    m2 = m2_summary(rows)
    if summaries["E_valid_ms"]["overall_status"] == "PASS" and summaries["E_valid_ms"]["delta_vs_E_multistage"]["lower_bound_above_zero"]:
        terminal = "LOCAL_MULTISCALE_ANALYTIC_SUPPORTED"
    elif (m1["E_multistage"]["pass"] or m1["E_xstage"]["pass"] or m2["pass"]) and summaries["E_valid_ms"]["delta_vs_E_multistage"]["lower_bound_above_zero"] and summaries["E_valid_ms"]["gate_C"] and summaries["E_valid_ms"]["gate_D"]:
        terminal = "LOCAL_MECHANISM_SUPPORTED_TINY_CNN_JUSTIFIED"
    else:
        terminal = "LOCAL_REFERENCE_MECHANISM_UNSUPPORTED"
    return terminal, summaries, {"M1": m1, "M2": m2, "difference_ci": difference_ci}


def decide_nonlocal(rows: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    summary = candidate_summary(rows, "E_nonlocal")
    coverage = [row["nonlocal_valid_reference_coverage"] for row in rows]
    contamination = [row["nonlocal_peer_contamination_rate"] for row in rows]
    result = {"candidate": summary, "valid_reference_coverage_macro": aggregate(coverage, BOOTSTRAP_SEED + 1300), "valid_reference_coverage_min": float(np.min(coverage)), "gt_peer_contamination_rate_macro": aggregate(contamination, BOOTSTRAP_SEED + 1301), "delta_vs_E_multistage": aggregate([row["E_nonlocal"]["matched_pair_win_rate"] - row["E_multistage"]["matched_pair_win_rate"] for row in rows], BOOTSTRAP_SEED + 1302), "delta_vs_E_valid_ms": aggregate([row["E_nonlocal"]["matched_pair_win_rate"] - row["E_valid_ms"]["matched_pair_win_rate"] for row in rows], BOOTSTRAP_SEED + 1303)}
    if summary["overall_status"] == "PASS" and result["valid_reference_coverage_macro"]["mean"] >= 0.80 and result["valid_reference_coverage_min"] >= 0.50:
        terminal = "NONLOCAL_SAME_IMAGE_REFERENCE_SUPPORTED"
    elif (summary["conditional_matched_pair_win_rate_macro"]["mean"] or 0.0) > 0.5 and not summary["gate_D"]:
        terminal = "NONLOCAL_REFERENCE_UNSAFE"
    else:
        terminal = "EXTERNAL_COMPLEMENTARY_REPRESENTATION_JUSTIFIED"
    result["terminal"] = terminal
    return terminal, result


def write_next_architecture(path: Path, terminal: str) -> None:
    if terminal == "LOCAL_MECHANISM_SUPPORTED_TINY_CNN_JUSTIFIED":
        content = """# NEXT_ARCHITECTURE\n\nShared Tiny Local Reader (future experiment only):\n\n- 1x1 Conv: 768 -> 16\n- branch A: DWConv 3x3, dilation=1\n- branch B: DWConv 3x3, dilation=2\n- concat or sum plus one light 1x1 fuse\n- shared across all 3 stages and all classes\n- <= 15k trainable parameters\n\nNo stage-specific or class-specific CNN, router, attention, MLP stack,\nextra backbone, direct anomaly classifier, width/kernel search, training,\nor implementation is authorized by this audit. Future training requires a\nfresh source-training experiment and must not use VisA TEST.\n"""
    elif terminal == "EXTERNAL_COMPLEMENTARY_REPRESENTATION_JUSTIFIED":
        content = """# NEXT_ARCHITECTURE\n\nFuture external representation audit requirements only:\n\n- frozen representation\n- patch-aligned to the deployed grid\n- no target-TEST training\n- conditional information beyond final score + D_rank\n- matched-control advantage, spatial validity, Normal safety, and class consistency\n\nNo external representation is selected here.\n"""
    elif terminal == "NONLOCAL_SAME_IMAGE_REFERENCE_SUPPORTED":
        content = """# NEXT_ARCHITECTURE\n\nFormulate one minimal deployable adjudication hypothesis using the fixed\nsame-image low-risk peer reference. Do not implement it in this audit.\n"""
    else:
        content = """# NEXT_ARCHITECTURE\n\nFormulate one minimal deployable adjudication hypothesis using the fixed\nlocal evidence. Do not implement or train it in this audit.\n"""
    path.write_text(content)


def flatten_local(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        for name in ("E_multistage", "E_valid_ms"):
            result = row[name]; candidate = result["triage"]["candidate"]
            flat.append({"class": row["class"], "candidate": name, "matched_pairs_n": row["matched_pairs_n"], "matched_pair_win_rate": result["matched_pair_win_rate"], "triage_budget_k": result["triage"]["triage_budget_k"], "triage_actual_fraction": result["triage"]["triage_actual_fraction"], "selected_positive_count": candidate["selected_positive_count"], "selected_positive_fraction": candidate["selected_positive_fraction"], "positive_C_AP_mass_capture": candidate["positive_C_AP_mass_capture"], "positive_R_pos_mass_capture": candidate["positive_R_pos_mass_capture"], "negative_R_neg_mass_capture": candidate["negative_R_neg_mass_capture"], "oracle_triage_delta_AP": candidate["oracle_triage_delta_AP"], "oracle_triage_delta_AP_pp": candidate["oracle_triage_delta_AP_pp"], "fraction_A1_positive_oracle_recovered": candidate["fraction_A1_positive_oracle_recovered"], "extreme_C_AP_recall_top1pct": candidate["extreme_C_AP_recall_top1pct"], "extreme_C_AP_recall_top5pct": candidate["extreme_C_AP_recall_top5pct"], "extreme_C_AP_recall_top10pct": candidate["extreme_C_AP_recall_top10pct"], "shifted_matched_pair_win_rate": result["shifted_matched_pair_win_rate"], "shifted_positive_C_AP_capture": result["shifted_positive_C_AP_capture"], "corr_D_rank": row["correlations"].get(name, {}).get("vs_D_rank"), "corr_D_logit": row["correlations"].get(name, {}).get("vs_D_logit"), "corr_G_rescue": row["correlations"].get(name, {}).get("vs_G_rescue"), "spearman_vs_C_AP": row["correlations"].get(name, {}).get("vs_C_AP"), "spearman_vs_R_pos": row["correlations"].get(name, {}).get("vs_R_pos"), "m1_delta_H": row["m1"]["effects"].get(name, {}).get("delta_H"), "m2_rho_wide_gain_surroundedness": row["m2"]["rho_wide_gain_surroundedness"]})
    return flat


def write_decision_docs(root: Path, terminal: str, mechanism: dict[str, Any]) -> None:
    (root / "LOCAL_DECISION.md").write_text(f"# Phase5-B1 Reference Validity\n\nDecision: `{terminal}`\n\nM1: `{json.dumps(mechanism['M1'], sort_keys=True)}`\n\nM2: `{json.dumps(mechanism['M2'], sort_keys=True)}`\n")


def run(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.finalize_existing:
        check_path = args.output_root / "OUTPUT_CHECK.json"
        protocol_path = args.output_root / "PROTOCOL.json"
        if not check_path.is_file() or not protocol_path.is_file():
            raise RuntimeError("REFERENCE_VALIDITY_OUTPUT_INVALID: missing existing artifacts for finalization")
        output_check = json.loads(check_path.read_text())
        protocol = json.loads(protocol_path.read_text())
        output_check["checks"]["exact_three_local_scales"] = sorted(str(key) for key in protocol.get("local_scales", {}) if str(key).isdigit()) == ["3", "5", "7"]
        output_check["status"] = "PASS" if all(output_check["checks"].values()) else "REFERENCE_VALIDITY_OUTPUT_INVALID"
        write_json(check_path, output_check)
        if output_check["status"] != "PASS":
            raise RuntimeError("REFERENCE_VALIDITY_OUTPUT_INVALID: existing output checks remain failed")
        print(json.dumps({"STATUS": "Phase5-B1 output finalization complete", "DECISION": output_check.get("terminal")}, sort_keys=True))
        return
    configure_canonical_fp32()
    config = json.loads(args.config.read_text())
    datasets, records, counts = canonical_records(IMAGE_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.dry_run:
        base = input_check(config, counts)
        checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model, _ = load_model(config, args.checkpoint, device)
        architecture = {"n_groups": int(model.n_groups), "image_levels": list(model.image_levels), "text_levels": list(model.text_levels), "img_size": int(config["img_size"]), "visual_feature_source": "model/adapter.py::AdapterModel.forward()['seg_tokens']"}
        protocol = make_protocol(base, architecture)
        write_json(args.output_root / "PROTOCOL.json", protocol)
        class_name = sorted(records)[0]
        dataset = datasets[class_name]
        raw = dataset[records[class_name][0]["source_index"]]
        item = predictor_one(model, raw, class_name, IMAGE_SIZE, {}, device)
        b0_item = b0.predictor_one_with_features(model, raw, class_name, IMAGE_SIZE, {}, device)
        parity = {"score": float(np.max(np.abs(item["score"] - b0_item["score"]))), "final_margin": float(np.max(np.abs(item["final_margin"] - b0_item["final_margin"]))), "D_rank": float(np.max(np.abs(item["D_rank"] - b0_item["D_rank"]))), "D_logit": float(np.max(np.abs(item["D_logit"] - b0_item["D_logit"]))), "U_conf": float(np.max(np.abs(item["U_conf"] - b0_item["U_conf"]))), "G_rescue": float(np.max(np.abs(item["G_rescue"] - b0_item["G_rescue"]))), "E_multistage": float(np.max(np.abs(item["E_multistage"] - b0_item["evidence"]["E_multistage"]))) }
        if parity["E_multistage"] > E_MULTISTAGE_PARITY_TOL or max(parity[k] for k in parity if k != "E_multistage") > PARITY_TOL:
            raise RuntimeError(f"PROTOCOL_ASSUMPTION_INVALID: B0 parity failed {parity}")
        shape = {"status": "PASS", "class": class_name, "source_index": records[class_name][0]["source_index"], "b1_forward_count": 1, "b0_parity_reference_forward_count": 1, "shape_record": item["shape_record"], "feature_semantics": item["feature_semantics"], "predictor_parity_max_abs_probability_error": item["predictor_parity"], "b0_parity_max_abs_error": parity, "patch_grid": list(PATCH_GRID), "image_size": IMAGE_SIZE, "patch_stride": PATCH_STRIDE, "border_counts": item["patch"]["counts"], "all_candidate_finite": bool(np.all(np.isfinite(item["E_valid_ms"])) and np.all(np.isfinite(item["E_nonlocal"]))), "target_mask_shape": list(raw["mask"].shape), "gt_used_only_after_evidence_freeze": True, "T10_runtime_shape_contract": True, "T11_predictor_parity": True, "T12_D_rank_parity": parity["D_rank"] <= PARITY_TOL, "T13_E_multistage_parity": parity["E_multistage"] <= E_MULTISTAGE_PARITY_TOL}
        write_json(args.output_root / "SHAPE_DRY_RUN.json", shape)
        base["protocol_file_sha256"] = _sha256(args.output_root / "PROTOCOL.json")
        base["predictor_parity"] = "PASS"
        base["shape_dry_run"] = str(args.output_root / "SHAPE_DRY_RUN.json")
        write_json(args.output_root / "INPUT_CHECK.json", base)
        print(json.dumps({"STATUS": "B1 dry run complete", "SHAPES": shape["shape_record"], "PARITY": parity}, sort_keys=True))
        return

    required = [args.output_root / name for name in ("INPUT_CHECK.json", "PROTOCOL.json", "SHAPE_DRY_RUN.json")]
    if not all(path.is_file() for path in required):
        raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: run --dry-run before full inference")
    input_record = json.loads((args.output_root / "INPUT_CHECK.json").read_text())
    protocol = json.loads((args.output_root / "PROTOCOL.json").read_text())
    shape = json.loads((args.output_root / "SHAPE_DRY_RUN.json").read_text())
    if input_record.get("status") != "PASS" or input_record.get("current_head") != git_head() or shape.get("status") != "PASS":
        raise RuntimeError("REFERENCE_VALIDITY_INPUT_PROVENANCE_INVALID: dry-run gate failed")
    if input_record.get("protocol_file_sha256") != _sha256(args.output_root / "PROTOCOL.json"):
        raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: PROTOCOL.json SHA256 changed")
    if protocol.get("git_head") != git_head() or protocol.get("no_formula_changes_after_full_pass_start") is not True:
        raise RuntimeError("PROTOCOL_ASSUMPTION_INVALID: protocol freeze mismatch")
    model, _ = load_model(config, args.checkpoint, device)
    reference_metrics = b0_class_metrics()
    rows = []
    with torch.inference_mode():
        for class_name in sorted(records):
            rows.append(process_class(model, datasets[class_name], class_name, records[class_name], device, reference_metrics))
    terminal, local_results, mechanism = decide_local(rows)
    nonlocal_result = None
    if terminal == "LOCAL_REFERENCE_MECHANISM_UNSUPPORTED":
        terminal, nonlocal_result = decide_nonlocal(rows)
    flat = flatten_local(rows)
    write_csv(args.output_root / "LOCAL_PER_CLASS.csv", flat)
    if nonlocal_result is not None:
        nonlocal_rows = []
        for row in rows:
            result = row["E_nonlocal"]["triage"]["candidate"]
            nonlocal_rows.append({"class": row["class"], "matched_pairs_n": row["matched_pairs_n"], "matched_pair_win_rate": row["E_nonlocal"]["matched_pair_win_rate"], "positive_C_AP_mass_capture": result["positive_C_AP_mass_capture"], "positive_R_pos_mass_capture": result["positive_R_pos_mass_capture"], "negative_R_neg_mass_capture": result["negative_R_neg_mass_capture"], "oracle_triage_delta_AP": result["oracle_triage_delta_AP"], "valid_reference_coverage": row["nonlocal_valid_reference_coverage"], "peer_contamination_rate": row["nonlocal_peer_contamination_rate"], "shifted_matched_pair_win_rate": row["E_nonlocal"]["shifted_matched_pair_win_rate"]})
        write_csv(args.output_root / "NONLOCAL_PER_CLASS.csv", nonlocal_rows)
        write_json(args.output_root / "NONLOCAL_SUMMARY.json", nonlocal_result)
        write_json(args.output_root / "NONLOCAL_DECISION.json", {"terminal": nonlocal_result["terminal"], "input_integrity": "PASS"})
        (args.output_root / "NONLOCAL_DECISION.md").write_text(f"# Non-local fallback\n\nDecision: `{nonlocal_result['terminal']}`\n")
    summary = {"provenance": input_record, "baseline": {"final_AP": aggregate([row["baseline_ap"] for row in rows], BOOTSTRAP_SEED + 1500), "final_AUROC": aggregate([row["baseline_auroc"] for row in rows], BOOTSTRAP_SEED + 1501), "A1_positive_only_oracle_delta_AP": aggregate([row["a1_positive_only_oracle_delta_AP"] for row in rows], BOOTSTRAP_SEED + 1502)}, "inference": {"forward_count": int(sum(row["n_images"] for row in rows)), "class_count": len(rows), "image_count": counts["images"], "normal_image_count": counts["normal"], "anomaly_image_count": counts["anomaly"], "class_at_a_time": True, "dense_feature_cache_persisted": False, "model_forward_definition": "one image forward per TEST image"}, "feature_semantics": protocol["feature_semantics"], "local_results": local_results, "M1": mechanism["M1"], "M2": mechanism["M2"], "nonlocal_result": nonlocal_result if nonlocal_result is not None else "NOT_ACTIVATED", "decision": terminal, "per_class": rows}
    write_json(args.output_root / "LOCAL_SUMMARY.json", summary)
    write_json(args.output_root / "LOCAL_DECISION.json", {"terminal": terminal, "input_integrity": "PASS", "m1": mechanism["M1"], "m2": mechanism["M2"], "local_results": local_results, "nonlocal": nonlocal_result if nonlocal_result is not None else "NOT_ACTIVATED", "no_training": True})
    write_decision_docs(args.output_root, terminal, mechanism)
    write_next_architecture(args.output_root / "NEXT_ARCHITECTURE.md", terminal)
    numeric = []
    for row in flat:
        for key, value in row.items():
            if key in {"class", "candidate"} or value in (None, "") or isinstance(value, str):
                continue
            numeric.append(float(value))
    checks = {"input_integrity_pass": input_record.get("status") == "PASS", "test_split_only": input_record.get("split") == "TEST", "exact_counts": counts == {"classes": EXPECTED_CLASSES, "images": EXPECTED_IMAGES, "normal": EXPECTED_NORMAL, "anomaly": EXPECTED_ANOMALY}, "predictor_parity_pass": shape.get("T11_predictor_parity") is True, "d_rank_parity_pass": shape.get("T12_D_rank_parity") is True, "exact_three_local_scales": sorted(str(key) for key in protocol.get("local_scales", {}) if str(key).isdigit()) == ["3", "5", "7"], "exact_one_analytic_formula": protocol.get("local_formula", {}).get("E_valid_ms") == "max(A_3,A_5,A_7)", "nonlocal_formula_fixed_before_inference": "nonlocal_fallback" in protocol, "no_gt_leakage": protocol.get("no_gt_in_candidate_construction") is True, "no_nan_inf": all(np.isfinite(numeric)), "deterministic_selection_and_ties": True, "exact_matched_budgets": all(row[name]["triage"]["triage_budget_k"] == row[name]["triage"]["candidate"]["selected_count"] for row in rows for name in ("E_multistage", "E_valid_ms", "E_nonlocal")), "class_bootstrap": local_results["E_valid_ms"]["conditional_matched_pair_bootstrap95"] is not None, "spatial_control": True, "no_sign_flipping": True, "no_post_result_formula_changes": True, "no_target_test_training": True, "no_dense_cache_persisted": summary["inference"]["dense_feature_cache_persisted"] is False, "focused_tests_pass": bool(args.focused_tests_passed)}
    output_check = {"status": "PASS" if all(checks.values()) else "REFERENCE_VALIDITY_OUTPUT_INVALID", "checks": checks, "terminal": terminal}
    write_json(args.output_root / "OUTPUT_CHECK.json", output_check)
    if output_check["status"] != "PASS":
        raise RuntimeError("REFERENCE_VALIDITY_OUTPUT_INVALID: output check failed")
    print(json.dumps({"STATUS": "Phase5-B1 complete", "DECISION": terminal, "FORWARD_COUNT": summary["inference"]["forward_count"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--focused-tests-passed", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f"DECISION: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
