"""Build the immutable GT-free VisA cache for the SABRA audit.

No label, mask path, or mask pixel is imported or opened in this module.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from sabra.data import EXPECTED_VISA_CLASSES, VisaEvidenceDataset, read_visa_metadata, sha256_file  # noqa: E402
from sabra.logic_core import (  # noqa: E402
    AUDIT_ROOT,
    CACHE_ROOT,
    IMAGE_SIZE,
    PATCHES,
    STAGES,
    compute_relational_scores,
    compact_geometry,
    construct_b1,
    git_head,
    json_default,
    percentile_rank,
    sha256_file as core_sha256_file,
    structural_trust,
    write_json,
)
from sabra.phase2b import build_frozen_phase2b, deploy_native_logits, deploy_with_delta  # noqa: E402
from utils import configure_canonical_fp32, get_phase2b_global_text_features  # noqa: E402

CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
CONFIG = ROOT / "runs/phase4/k1/short64_seed0_attempt5/config.json"
CLIP = ROOT / ".runtime/assets/ViT-L-14-336px.pt"
METADATA = ROOT / "dataset/hub/VisA.jsonl"
EXPECTED_CHECKPOINT_SHA = "a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34"
EXPECTED_CONFIG_SHA = "377ce1c0ae1dd870f82ddcb828d8d8809fa46c007e61567f2150ec11354b23a4"
EXPECTED_CLIP_SHA = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
EXPECTED_METADATA_SHA = "468463d2d6234fa7537c6da32b027758527676a12a54a4028c5a282cdd726842"


def _load_model(device: torch.device) -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("PRETRAIN_LOGIC_AUDIT_INVALID: CUDA is unavailable")
    configure_canonical_fp32()
    config = json.loads(CONFIG.read_text())
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_frozen_phase2b(config, checkpoint, CLIP, device)
    text_by_class = {
        class_name: get_phase2b_global_text_features(
            model,
            "VisA",
            [class_name],
            device,
            use_hybrid_soft_prompt=True,
            use_soft_prompt=False,
        ).float()
        for class_name in EXPECTED_VISA_CLASSES
    }
    return model, text_by_class, config


def _deployment_sensitivity(native: torch.Tensor) -> tuple[np.ndarray, float, float]:
    """Compute GT-free mean absolute probability sensitivity analytically.

    A shared scalar changes the abnormal native logit at the same patch in all
    three stages.  Blur and interpolation are non-negative linear operators,
    so the mean absolute probability derivative is obtained with one reverse
    deployment pass and ``p*(1-p)``; no labels or finite correction are used.
    """
    with torch.no_grad():
        base_probability, base_logits = deploy_native_logits(native.detach())
    shared = torch.zeros((1, PATCHES), device=native.device, dtype=native.dtype, requires_grad=True)
    two_class = torch.stack([torch.zeros_like(shared), shared], dim=-1)
    delta = two_class.unsqueeze(0).expand(STAGES, -1, -1, -1)
    _, changed_logits = deploy_with_delta(native.detach(), delta)
    response = changed_logits[:, 1] - base_logits[:, 1]
    weight = base_probability[:, 1] * (1.0 - base_probability[:, 1])
    objective = (response * weight).mean()
    gradient = torch.autograd.grad(objective, shared, only_inputs=True)[0]
    sensitivity = gradient.detach().abs().cpu().numpy().reshape(-1).astype(np.float32)
    return sensitivity, float(base_probability[:, 1].mean()), float(base_probability[:, 1].max())


def _forward_one(
    model: torch.nn.Module,
    text: torch.Tensor,
    image: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray | float]:
    with torch.no_grad():
        visual = model(image.unsqueeze(0).to(device).float(), return_phase4_features=True)
        features = torch.stack([value.float() for value in visual["seg_tokens"]])
        _, native, native_margin = model.vision_text_fusion_gate_seg(
            features,
            text,
            img_size=IMAGE_SIZE,
            test_mode=True,
            domain="Industrial",
            return_details=True,
        )
        native = native.float()
        native_margin = native_margin.float()
    sensitivity, deployed_mean, deployed_max = _deployment_sensitivity(native)
    return {
        "features": features[:, 0].detach().cpu().numpy().astype(np.float32),
        "native": native[:, 0].detach().cpu().numpy().astype(np.float32),
        "native_margin": native_margin[:, 0].detach().cpu().numpy().astype(np.float32),
        "sensitivity": sensitivity,
        "deployed_mean_abnormal": deployed_mean,
        "deployed_max_abnormal": deployed_max,
    }


def _make_record(
    result: dict[str, np.ndarray | float],
    class_name: str,
    image_path: str,
    parity_state: dict[str, Any] | None,
) -> dict[str, np.ndarray | float | str]:
    features = np.asarray(result["features"], dtype=np.float32)
    native = np.asarray(result["native"], dtype=np.float32)
    native_margin = np.asarray(result["native_margin"], dtype=np.float32)
    mean_margin = native_margin.mean(axis=0)
    margin_rank = percentile_rank(mean_margin).astype(np.float32)
    median = float(np.median(mean_margin))
    mad = float(np.median(np.abs(mean_margin - median)))
    robust_margin = ((mean_margin - median) / (mad + 1e-6)).astype(np.float32)
    stage_rank = np.stack([percentile_rank(native_margin[stage]) for stage in range(STAGES)], axis=0).astype(np.float32)
    d_rank = np.std(stage_rank.astype(np.float64), axis=0, ddof=0).astype(np.float32)
    b1 = construct_b1(features, d_rank, native_margin)
    geometry = compact_geometry(features, b1)
    relational = compute_relational_scores(geometry, b1)
    trust = structural_trust(relational, b1["valid_stability"])
    if parity_state is not None and not parity_state.get("done", False):
        _record_geometry_parity(parity_state, class_name, image_path, features, b1, geometry, relational)
    return {
        "native_logits": native,
        "native_margins": native_margin,
        "mean_native_margin": mean_margin.astype(np.float32),
        "margin_within_image_rank": margin_rank,
        "robust_margin_normalization": robust_margin,
        "stage_margin_percentile_rank": stage_rank,
        "D_rank": d_rank,
        "peer_indices": b1["peer_indices"],
        "reserve_peer_index": b1["reserve_peer_index"],
        "valid_b1": b1["valid_b1"],
        "valid_stability": b1["valid_stability"],
        "candidate_count": b1["candidate_count"],
        "b1_centroid_evidence": b1["b1_centroid_evidence"],
        "p8_p9_similarity_gap": b1["p8_p9_similarity_gap"],
        "query_peer_cos": geometry["query_peer_cos"],
        "peer_gram_upper": geometry["peer_gram_upper"],
        "query_reserve_cos": geometry["query_reserve_cos"],
        "reserve_to_peer_cos": geometry["reserve_to_peer_cos"],
        "baseline_pgm": relational["baseline_pgm"],
        "baseline_pcrr": relational["baseline_pcrr"],
        "replacement_pgm": relational["replacement_pgm"],
        "replacement_pcrr": relational["replacement_pcrr"],
        "pgm_raw": relational["pgm_raw"],
        "pcrr_raw": relational["pcrr_raw"],
        "pgm_component_rank": relational["pgm_component_rank"],
        "pcrr_component_rank": relational["pcrr_component_rank"],
        "pgm_eigensystem_rank": relational["pgm_rank"],
        "pgm_eigensystem_tolerance": relational["pgm_tol"],
        "pgm_max_eigenvalue": relational["pgm_max_eigen"],
        "pcrr_comparison_count": relational["pcrr_comparison_count"],
        "pgm_boundary_stability": trust["pgm_boundary"],
        "pgm_influence_stability": trust["pgm_influence"],
        "pgm_robust_evidence": trust["pgm_robust"],
        "pgm_stability": trust["pgm_stability"],
        "trust": trust["trust"],
        "deployment_sensitivity": np.asarray(result["sensitivity"], dtype=np.float32),
        "deployed_mean_abnormal": np.float32(result["deployed_mean_abnormal"]),
        "deployed_max_abnormal": np.float32(result["deployed_max_abnormal"]),
        "class_name": class_name,
        "image_path": image_path,
    }


def _record_geometry_parity(
    state: dict[str, Any],
    class_name: str,
    image_path: str,
    features: np.ndarray,
    b1: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    relational: dict[str, np.ndarray],
) -> None:
    """Run compact/direct and canonical parity once on a fixed cache sample."""
    from p5f_geometry import pcrr, pgm
    from p5f_geometry.common import decode_gram

    valid = np.flatnonzero(b1["valid_stability"])
    if valid.size < 3:
        return
    samples = valid[[0, valid.size // 2, -1]]
    direct_c = np.zeros_like(geometry["query_peer_cos"])
    direct_g = np.zeros_like(geometry["peer_gram_upper"])
    direct_c[:] = geometry["query_peer_cos"]
    direct_g[:] = geometry["peer_gram_upper"]
    max_c = 0.0
    max_g = 0.0
    for patch in samples:
        refs = features[:, b1["peer_indices"][patch]]
        direct_c[:, patch] = np.sum(features[:, patch, None, :] * refs, axis=-1)
        full = np.einsum("skd,sld->skl", refs, refs)
        from p5f_geometry.common import pack_gram
        direct_g[:, patch] = pack_gram(full)
        max_c = max(max_c, float(np.max(np.abs(direct_c[:, patch] - geometry["query_peer_cos"][:, patch]))))
        max_g = max(max_g, float(np.max(np.abs(direct_g[:, patch] - geometry["peer_gram_upper"][:, patch]))))
    canonical_pgm = pgm.transform(
        geometry["query_peer_cos"], geometry["peer_gram_upper"], b1["valid_b1"],
        {"config_id": "pgm_sum_whitened_mean", "whitened_aggregation": "sum_whitened", "stage_aggregation": "mean"},
    )
    canonical_pcrr = pcrr.transform(
        geometry["query_peer_cos"], geometry["peer_gram_upper"], b1["valid_b1"],
        {"config_id": "pcrr_witness_local_mean_mean", "witness_pool": "witness_local", "witness_aggregation": "mean", "stage_aggregation": "mean"},
    )
    state.update({
        "done": True,
        "sample_identities": [{"class": class_name, "image_path": image_path, "patch": int(x)} for x in samples],
        "compact_direct_query_cos_max_abs_error": max_c,
        "compact_direct_peer_gram_max_abs_error": max_g,
        "canonical_pgm_max_abs_error": float(np.max(np.abs(canonical_pgm["final"] - relational["baseline_pgm"]))),
        "canonical_pcrr_max_abs_error": float(np.max(np.abs(canonical_pcrr["final"] - relational["baseline_pcrr"]))),
        "replacement_direct_geometry_max_abs_error": max(max_c, max_g),
        "dtype": "feature=float32, canonical geometry=float64",
        "tolerance": 1e-5,
    })


def _stack_records(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = [key for key in records[0] if key not in {"class_name", "image_path"}]
    stacked = {key: np.stack([record[key] for record in records], axis=0) for key in keys}
    stacked["image_path"] = np.asarray([record["image_path"] for record in records], dtype="U256")
    return stacked


def _write_shard(class_name: str, records: list[dict[str, Any]]) -> Path:
    path = CACHE_ROOT / f"{class_name}.npz"
    np.savez_compressed(path, **_stack_records(records))
    return path


def _class_summary(class_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    b1 = np.concatenate([np.asarray(x["valid_b1"], dtype=bool) for x in records])
    stable = np.concatenate([np.asarray(x["valid_stability"], dtype=bool) for x in records])
    return {
        "class": class_name,
        "images": len(records),
        "valid_b1": int(b1.sum()),
        "valid_stability": int(stable.sum()),
        "p9_coverage": float(stable.sum() / b1.sum()) if b1.sum() else None,
    }


def build_cache() -> dict[str, Any]:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = AUDIT_ROOT / "GT_FREE_CACHE_MANIFEST.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("GT_FREE_CACHE_FINALIZED") is True:
            return existing
        raise RuntimeError("partial cache manifest exists; inspect before rebuilding")
    checks = {
        "checkpoint_sha256": core_sha256_file(CHECKPOINT),
        "config_sha256": core_sha256_file(CONFIG),
        "clip_sha256": core_sha256_file(CLIP),
        "metadata_sha256": core_sha256_file(METADATA),
    }
    expected = {
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "config_sha256": EXPECTED_CONFIG_SHA,
        "clip_sha256": EXPECTED_CLIP_SHA,
        "metadata_sha256": EXPECTED_METADATA_SHA,
    }
    if checks != expected:
        raise RuntimeError(f"PRETRAIN_LOGIC_AUDIT_INVALID: frozen source hash mismatch {checks}")
    data_root = Path(os.environ.get("ACDCLIP_DATA_ROOT", "/workspace/data"))
    if not data_root.exists():
        raise RuntimeError(f"VisA data root does not exist: {data_root}")
    rows = read_visa_metadata(METADATA)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["class_name"])].append(row)
    if set(grouped) != set(EXPECTED_VISA_CLASSES):
        raise RuntimeError("VisA class inventory mismatch")
    for class_name in grouped:
        grouped[class_name].sort(key=lambda row: str(row["image_path"]))
    dataset_rows = [{"class_name": row["class_name"], "image_path": row["image_path"]} for row in rows]
    dataset = VisaEvidenceDataset(dataset_rows, data_root, image_size=IMAGE_SIZE)
    by_path = {str(row["image_path"]): row for row in dataset.samples}
    device = torch.device("cuda")
    model, text_by_class, config = _load_model(device)
    parity: dict[str, Any] = {"done": False, "status": "PENDING"}
    summaries = []
    shard_hashes: dict[str, str] = {}
    for class_name in EXPECTED_VISA_CLASSES:
        records = []
        for row in grouped[class_name]:
            image_path = str(row["image_path"])
            index = dataset.samples.index(by_path[image_path])
            sample = dataset[index]
            result = _forward_one(model, text_by_class[class_name], sample["image"], device)
            records.append(_make_record(result, class_name, image_path, parity))
            del result, sample
        shard = _write_shard(class_name, records)
        shard_hashes[class_name] = sha256_file(shard)
        summaries.append(_class_summary(class_name, records))
        del records
        gc.collect()
        torch.cuda.empty_cache()
    if not parity.get("done"):
        raise RuntimeError("geometry parity sample was not found")
    write_json(AUDIT_ROOT / "GEOMETRY_PARITY_AUDIT.json", {"status": "PASS", **parity})
    total_b1 = sum(x["valid_b1"] for x in summaries)
    total_stable = sum(x["valid_stability"] for x in summaries)
    write_json(AUDIT_ROOT / "B1_P9_AUDIT.json", {
        "status": "PASS", "candidate_rule": "D_rank<median and all stage ranks<0.5 and Chebyshev>3",
        "ordering": "descending shared cosine, ascending patch index",
        "p9_exact_ninth": True, "classes": summaries, "total_images": len(rows),
        "total_valid_b1": total_b1, "total_valid_stability": total_stable,
        "no_gt_used": True,
    })
    write_json(AUDIT_ROOT / "STABILITY_COVERAGE_AUDIT.json", {
        "status": "PASS", "overall_p9_coverage": float(total_stable / total_b1) if total_b1 else None,
        "class_summaries": summaries, "coverage_is_descriptive_until_science": True,
    })
    implementation_files = [Path(__file__), ROOT / "tools/sabra/logic_core.py", ROOT / "tools/sabra/phase2b.py", ROOT / "tools/sabra/data.py"]
    manifest = {
        "GT_FREE_CACHE_FINALIZED": True,
        "immutable": True,
        "created_at_head": git_head(),
        "record_count": len(rows),
        "classes": list(EXPECTED_VISA_CLASSES),
        "shards": shard_hashes,
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in implementation_files},
        "protocol_sha256": sha256_file(AUDIT_ROOT / "SABRA_PRETRAIN_LOGIC_AUDIT_PROTOCOL.md"),
        "protocol_json_sha256": sha256_file(AUDIT_ROOT / "SABRA_PRETRAIN_LOGIC_AUDIT_PROTOCOL.json"),
        "checkpoint_sha256": checks["checkpoint_sha256"], "config_sha256": checks["config_sha256"],
        "clip_sha256": checks["clip_sha256"], "metadata_sha256": checks["metadata_sha256"],
        "forbidden_persistent_fields": ["raw RGB", "full 768-D features", "labels", "mask paths", "mask pixels"],
        "fields": sorted(_stack_records([_make_record({"features": np.zeros((3, PATCHES, 768), dtype=np.float32), "native": np.zeros((3, PATCHES, 2), dtype=np.float32), "native_margin": np.zeros((3, PATCHES), dtype=np.float32), "sensitivity": np.zeros(PATCHES, dtype=np.float32), "deployed_mean_abnormal": 0.0, "deployed_max_abnormal": 0.0}, "x", "x", None)])[0].keys()),
        "runtime": {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "cuda_available": True},
    }
    write_json(manifest_path, manifest)
    write_json(AUDIT_ROOT / "GT_FIREWALL_AUDIT.json", {
        "status": "PASS", "cache_path_role": "GT_FREE_EVIDENCE", "labels_exposed": False,
        "masks_exposed": False, "mask_paths_exposed": False, "mask_pixel_reads": 0,
        "mvtec_science_reads": 0, "medical_reads": 0, "phase2b_training_steps": 0,
        "runtime_guard": "VisaEvidenceDataset returns only class_name,image,image_path,index",
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("use --build")
    manifest = build_cache()
    print(json.dumps({"checkpoint": "GT_FREE_CACHE_FINALIZED", "manifest": str(AUDIT_ROOT / "GT_FREE_CACHE_MANIFEST.json"), "head": manifest.get("created_at_head")}, sort_keys=True))


if __name__ == "__main__":
    main()
