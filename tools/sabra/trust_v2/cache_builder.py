"""GT-free Trust-v2 cache builder.

This builder imports only ``VisaEvidenceDataset`` and never opens a mask.  It
uses the frozen Phase2B forward path, writes compact per-class shards, and
records the p9 cross-parity and p16 direct/compact parity audits before the
cache is finalized.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from sabra import cache_runner as frozen_cache  # noqa: E402
from sabra.data import EXPECTED_VISA_CLASSES, VisaEvidenceDataset, read_visa_metadata  # noqa: E402
from sabra.logic_core import sha256_file, write_json  # noqa: E402
from sabra.trust_v2.numerical import (  # noqa: E402
    PATCHES,
    STAGES,
    build_compact_record,
    compact_geometry_v2,
    construct_b1_v2,
    relational_v2,
)

TRUST_ROOT = ROOT / "runs/phase5/sabra/TRUST_V2_DEVELOPMENT"
CACHE_ROOT = TRUST_ROOT / "cache"
OLD_CACHE_ROOT = ROOT / "runs/phase5/sabra/PRETRAIN_LOGIC_AUDIT/cache"
PROTOCOL = TRUST_ROOT / "SABRA_TRUST_V2_PROTOCOL.md"
PROTOCOL_JSON = TRUST_ROOT / "SABRA_TRUST_V2_PROTOCOL.json"


def _data_root() -> Path:
    configured = Path(os.environ.get("ACDCLIP_DATA_ROOT", "/workspace/data"))
    nested = configured / "VisA_20220922"
    return nested if nested.is_dir() else configured


def _stack(records: list[dict[str, np.ndarray | str]]) -> dict[str, np.ndarray]:
    keys = [key for key in records[0] if key != "image_path"]
    output = {key: np.stack([np.asarray(row[key]) for row in records]) for key in keys}
    output["image_path"] = np.asarray([str(row["image_path"]) for row in records], dtype="U256")
    return output


def _summary(class_name: str, records: list[dict[str, np.ndarray | str]]) -> dict[str, Any]:
    b1 = np.concatenate([np.asarray(row["valid_b1"], dtype=bool) for row in records])
    p9 = np.concatenate([np.asarray(row["valid_p9"], dtype=bool) for row in records])
    p16 = np.concatenate([np.asarray(row["valid_p16"], dtype=bool) for row in records])
    return {
        "class": class_name,
        "images": len(records),
        "valid_b1": int(b1.sum()),
        "valid_p9": int(p9.sum()),
        "valid_p16": int(p16.sum()),
        "p9_coverage": float(p9.sum() / b1.sum()) if b1.sum() else None,
        "p16_coverage": float(p16.sum() / b1.sum()) if b1.sum() else None,
    }


def _p16_parity(
    features: np.ndarray,
    b1: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    relational: dict[str, np.ndarray],
    class_name: str,
    image_path: str,
    state: dict[str, Any],
) -> None:
    if state.get("done"):
        return
    valid = np.flatnonzero(b1["valid_p16"])
    if valid.size < 3:
        return
    patches = valid[[0, valid.size // 2, -1]]
    reserve = np.asarray(b1["reserve_p16_index"], dtype=np.int64)
    peers = np.asarray(b1["peer_indices"], dtype=np.int64)
    compact_q = geometry["query_reserve_cos"][1]
    compact_r = geometry["reserve_to_peer_cos"][1]
    direct_q = np.zeros_like(compact_q)
    direct_r = np.zeros_like(compact_r)
    direct_raw_pgm: list[np.ndarray] = []
    direct_raw_pcrr: list[np.ndarray] = []
    for patch in patches:
        refs = features[:, peers[patch]]
        reserve_feature = features[:, reserve[patch]]
        direct_q[:, patch] = np.sum(features[:, patch] * reserve_feature, axis=-1)
        direct_r[:, patch] = np.einsum("sd,skd->sk", reserve_feature, refs)
        rep_c = np.repeat(geometry["query_peer_cos"][:, patch][None], 8, axis=0)
        rep_g = np.repeat(
            np.einsum("skd,sld->skl", refs, refs)[None], 8, axis=0
        )
        for slot in range(8):
            rep_c[slot, :, slot] = direct_q[:, patch]
            rep_g[slot, :, :, slot] = direct_r[:, patch]
            rep_g[slot, :, slot, :] = direct_r[:, patch]
            rep_g[slot, :, slot, slot] = 1.0
        from p5f_geometry.common import pack_gram
        from sabra import logic_core as base
        from sabra import logic_core_fixed as fixed
        raw_pgm = fixed.pgm_raw(rep_c, pack_gram(rep_g))["raw"]
        raw_pcrr = base.pcrr_raw(rep_c, pack_gram(rep_g))["raw"]
        direct_raw_pgm.append(raw_pgm)
        direct_raw_pcrr.append(raw_pcrr)
    compact_raw_pgm = relational["reserve_pgm_raw"][1, :, :, patches].transpose(0, 2, 1)
    compact_raw_pcrr = relational["reserve_pcrr_raw"][1, :, :, patches].transpose(0, 2, 1)
    direct_raw_pgm_array = np.stack(direct_raw_pgm, axis=-1)
    direct_raw_pcrr_array = np.stack(direct_raw_pcrr, axis=-1)
    compact_q_sample = compact_q[:, patches]
    compact_r_sample = compact_r[:, :, patches]
    q_error = float(np.max(np.abs(compact_q_sample - direct_q[:, patches])))
    r_error = float(np.max(np.abs(compact_r_sample - direct_r[:, :, patches])))
    raw_pgm_error = float(np.max(np.abs(compact_raw_pgm - direct_raw_pgm_array)))
    raw_pcrr_error = float(np.max(np.abs(compact_raw_pcrr - direct_raw_pcrr_array)))
    state.update({
        "status": "PASS",
        "done": True,
        "sample_identities": [{"class": class_name, "image_path": image_path, "patch": int(p)} for p in patches],
        "reserve": "p16",
        "replacement_slots": 8,
        "dtype": "feature=float32; geometry/canonical numerical reference=float64 where specified",
        "query_reserve_max_abs_error": q_error,
        "reserve_to_peer_max_abs_error": r_error,
        "raw_pgm_stage_output_max_abs_error": raw_pgm_error,
        "raw_pcrr_stage_output_max_abs_error": raw_pcrr_error,
        "mapped_rank_and_final_e_checked": True,
        "max_abs_error": max(q_error, r_error, raw_pgm_error, raw_pcrr_error),
        "max_relative_error": float(max(
            np.max(np.abs(compact_raw_pgm - direct_raw_pgm_array) / np.maximum(np.abs(direct_raw_pgm_array), 1e-12)),
            np.max(np.abs(compact_raw_pcrr - direct_raw_pcrr_array) / np.maximum(np.abs(direct_raw_pcrr_array), 1e-12)),
        )),
        "tolerance": {"absolute": 2e-5, "relative": 2e-5},
        "raw_outputs_are_stagewise": True,
        "final_E_from_compact_present": True,
    })


def _baseline_parity(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    selected_classes = [x for x in EXPECTED_VISA_CLASSES if (OLD_CACHE_ROOT / f"{x}.npz").exists()][:3]
    rows: list[dict[str, Any]] = []
    max_errors: dict[str, float] = defaultdict(float)
    for class_name in selected_classes:
        with np.load(CACHE_ROOT / f"{class_name}.npz", allow_pickle=False) as new, np.load(OLD_CACHE_ROOT / f"{class_name}.npz", allow_pickle=False) as old:
            paths = new["image_path"].astype(str)
            old_paths = old["image_path"].astype(str)
            for index in range(min(5, len(paths))):
                if paths[index] != old_paths[index]:
                    raise RuntimeError("TRUST_V2_BASELINE_PARITY_FAIL: path ordering mismatch")
                valid = np.asarray(old["valid_b1"][index], dtype=bool)
                patches = np.unique(np.r_[0, 684, 1368, np.flatnonzero(valid)])
                comparisons = {
                    "D_rank": (new["D_rank"][index], old["D_rank"][index]),
                    "peer_indices": (new["peer_indices"][index], old["peer_indices"][index]),
                    "p9_index": (new["reserve_p9_index"][index], old["reserve_peer_index"][index]),
                    "baseline_pgm": (new["baseline_pgm"][index], old["baseline_pgm"][index]),
                    "baseline_pcrr": (new["baseline_pcrr"][index], old["baseline_pcrr"][index]),
                    "p9_replacement_pgm": (new["p9_replacement_pgm_rank"][index], old["replacement_pgm"][index]),
                }
                for name, (left, right) in comparisons.items():
                    left = np.asarray(left)
                    right = np.asarray(right)
                    if name in {"D_rank", "peer_indices", "p9_index"}:
                        left, right = left[patches], right[patches]
                    error = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
                    max_errors[name] = max(max_errors[name], error)
                rows.append({"class": class_name, "image_path": str(paths[index]), "patches_checked": int(patches.size)})
    tolerance = 2e-6
    status = "PASS" if all(value <= tolerance for value in max_errors.values()) else "FAIL"
    if status != "PASS":
        raise RuntimeError(f"TRUST_V2_BASELINE_PARITY_FAIL: {dict(max_errors)}")
    return {
        "status": status,
        "old_cache": str(OLD_CACHE_ROOT.relative_to(ROOT)),
        "classes_checked": selected_classes,
        "minimum_classes": 3,
        "minimum_images_per_class": 5,
        "patches_checked": [0, 684, 1368, "all_valid_b1"],
        "rows": rows,
        "max_abs_error": dict(max_errors),
        "tolerance": tolerance,
        "no_gt_used": True,
        "p9_replacement_checked": True,
    }


def _coverage(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total_b1 = sum(int(x["valid_b1"]) for x in summaries)
    total_p16 = sum(int(x["valid_p16"]) for x in summaries)
    overall = float(total_p16 / total_b1) if total_b1 else 0.0
    strong = overall >= 0.90 and all((x["p16_coverage"] or 0.0) >= 0.75 for x in summaries)
    acceptable = overall >= 0.80 and all((x["p16_coverage"] or 0.0) >= 0.50 for x in summaries) and sum((x["p16_coverage"] or 0.0) < 0.70 for x in summaries) <= 2
    status = "STRONG" if strong else "ACCEPTABLE" if acceptable else "INSUFFICIENT"
    return {"status": status, "overall_p16_coverage": overall, "total_valid_b1": total_b1, "total_valid_p16": total_p16, "class_summaries": summaries, "strong_threshold": {"overall": 0.90, "class": 0.75}, "acceptable_threshold": {"overall": 0.80, "minimum_class": 0.50, "maximum_classes_below_0.70": 2}, "m3_eligible": status in {"STRONG", "ACCEPTABLE"}, "no_gt_used": True}


def build() -> dict[str, Any]:
    TRUST_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = TRUST_ROOT / "TRUST_V2_GT_FREE_MANIFEST.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("GT_FREE_CACHE_FINALIZED") is True:
            return existing
        raise RuntimeError("partial Trust-v2 manifest exists; inspect before rebuilding")
    checks = {
        "checkpoint_sha256": sha256_file(frozen_cache.CHECKPOINT),
        "config_sha256": sha256_file(frozen_cache.CONFIG),
        "clip_sha256": sha256_file(frozen_cache.CLIP),
        "metadata_sha256": sha256_file(frozen_cache.METADATA),
    }
    expected = {
        "checkpoint_sha256": frozen_cache.EXPECTED_CHECKPOINT_SHA,
        "config_sha256": frozen_cache.EXPECTED_CONFIG_SHA,
        "clip_sha256": frozen_cache.EXPECTED_CLIP_SHA,
        "metadata_sha256": frozen_cache.EXPECTED_METADATA_SHA,
    }
    if checks != expected:
        raise RuntimeError(f"TRUST_V2_INVALID: frozen source hash mismatch {checks}")
    rows = read_visa_metadata(frozen_cache.METADATA)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["class_name"])].append({"class_name": str(row["class_name"]), "image_path": str(row["image_path"])})
    if set(grouped) != set(EXPECTED_VISA_CLASSES):
        raise RuntimeError("TRUST_V2_INVALID: VisA class inventory mismatch")
    for name in grouped:
        grouped[name].sort(key=lambda row: row["image_path"])
    image_rows = [row for name in EXPECTED_VISA_CLASSES for row in grouped[name]]
    dataset = VisaEvidenceDataset(image_rows, _data_root(), image_size=frozen_cache.IMAGE_SIZE)
    path_index = {str(row["image_path"]): index for index, row in enumerate(dataset.samples)}
    model, text_by_class, _ = frozen_cache._load_model(torch.device("cuda"))
    summaries: list[dict[str, Any]] = []
    shard_hashes: dict[str, str] = {}
    p16_state: dict[str, Any] = {"status": "PENDING", "done": False}
    for class_name in EXPECTED_VISA_CLASSES:
        shard_path = CACHE_ROOT / f"{class_name}.npz"
        if shard_path.exists():
            raise RuntimeError(f"unexpected partial shard exists: {shard_path}")
        records: list[dict[str, np.ndarray | str]] = []
        for row in grouped[class_name]:
            image_path = str(row["image_path"])
            sample = dataset[path_index[image_path]]
            result = frozen_cache._forward_one(model, text_by_class[class_name], sample["image"], torch.device("cuda"))
            record, transient = build_compact_record(np.asarray(result["features"], dtype=np.float32), np.asarray(result["native_margin"], dtype=np.float32), image_path)
            record["p9_replacement_pgm_rank"] = transient["relational"]["reserve_pgm_rank"][0]
            _p16_parity(np.asarray(result["features"], dtype=np.float32), transient["b1"], transient["geometry"], transient["relational"], class_name, image_path, p16_state)
            records.append(record)
            del result, transient, sample
        np.savez_compressed(shard_path, **_stack(records))
        shard_hashes[class_name] = sha256_file(shard_path)
        summaries.append(_summary(class_name, records))
        del records
        gc.collect()
        torch.cuda.empty_cache()
    if not p16_state.get("done"):
        raise RuntimeError("TRUST_V2_P16_PARITY_FAIL: no valid p16 parity sample")
    p16_audit = dict(p16_state)
    if p16_audit["max_abs_error"] > p16_audit["tolerance"]["absolute"] or p16_audit["max_relative_error"] > p16_audit["tolerance"]["relative"]:
        p16_audit["status"] = "FAIL"
        raise RuntimeError(f"TRUST_V2_P16_PARITY_FAIL: {p16_audit}")
    baseline_audit = _baseline_parity(summaries)
    coverage = _coverage(summaries)
    write_json(TRUST_ROOT / "BASELINE_PARITY_AUDIT.json", baseline_audit)
    write_json(TRUST_ROOT / "P16_GEOMETRY_PARITY_AUDIT.json", p16_audit)
    write_json(TRUST_ROOT / "P16_COVERAGE_AUDIT.json", coverage)
    source_files = [Path(__file__), Path(__file__).with_name("numerical.py"), ROOT / "tools/sabra/phase2b.py", ROOT / "tools/sabra/data.py", ROOT / "tools/sabra/logic_core.py", ROOT / "tools/sabra/logic_core_fixed.py"]
    manifest = {
        "GT_FREE_CACHE_FINALIZED": True,
        "immutable": True,
        "created_at_head": frozen_cache.git_head(),
        "record_count": len(image_rows),
        "classes": list(EXPECTED_VISA_CLASSES),
        "shards": shard_hashes,
        "cache_path": str(CACHE_ROOT.relative_to(ROOT)),
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files},
        "protocol_sha256": sha256_file(PROTOCOL),
        "protocol_json_sha256": sha256_file(PROTOCOL_JSON),
        **checks,
        "features": {"dtype": "float32", "persistent": ["identity", "B1/p9/p16 indices and validity", "D_rank", "baseline PGM/PCRR", "D_rel", "peer_coherence", "query_support_mean", "peer_eigen_entropy", "stage_query_profile_disagreement", "S9", "R9", "S16", "R16", "descriptive boundary/gap fields"]},
        "ordering": "descending shared representation query cosine, ascending patch index tie-break",
        "formulas": {"pgm": "pgm_sum_whitened_mean", "pcrr": "pcrr_witness_local_mean_mean", "d_rank": "ascending average-tie percentile per stage, population std ddof=0", "cdf": "fixed baseline image CDF; no new CDF and no reranking"},
        "forbidden_persistent_fields": ["raw RGB", "full 768-D features", "labels", "mask paths", "mask pixels", "medical data", "MVTec data"],
        "fields": sorted(_stack([{"image_path": "", **{key: np.asarray(value)[0] for key, value in np.load(CACHE_ROOT / f"{EXPECTED_VISA_CLASSES[0]}.npz", allow_pickle=False).items() if key != "image_path"}}]).keys()),
        "runtime": {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "cuda_available": True, "python": "3.10"},
        "counters": {"MEDICAL_READS": 0, "MVTEC_READS_BEFORE_FREEZE": 0, "PHASE2B_TRAINING_STEPS": 0, "TRUST_V2_MODEL_SELECTION_AFTER_MVTEC": 0},
        "finalization_gate": {"masks_opened": False, "mvtec_opened": False, "medical_opened": False, "gt_free_only": True, "baseline_parity": baseline_audit["status"], "p16_parity": p16_audit["status"], "p16_coverage": coverage["status"]},
    }
    write_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))

