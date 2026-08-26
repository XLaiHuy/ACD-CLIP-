"""Post-freeze P30 transfer and stability diagnostics using cached tensors."""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from model.phase2b_runtime import deploy_native_logits
from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata
from tools.sabra_car.r0_direction import classify_actions
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p28_mechanism_diagnostic import _load_masks, patch_correction_from_actions
from tools.sabra_v2.p29r1_forensic import (
    forensic_utility_for_batch,
    residual_magnitude_summary,
    sign_alignment,
    vectorized_pixel_shifts,
)
from tools.sabra_v2.p30_contract import P30_PREREGISTRATION_PATH, P30_UUID, load_and_audit_p30_preregistration, p30_cache_provenance
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import atomic_write_json
from tools.sabra_v2.region_pool import pool_patch_map
from tools.sabra_v2.train_region_distill import ROOT


P29_ROOT = Path("/workspace/p29_science_v1")
P29_ADAPTER_SCHEMA = "P29_REGION_ADAPTER_CHECKPOINT_V1"
IMAGE_SIZE = 518
PATCH_COUNT = 37 * 37


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--scoring-gate", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", choices=EXPECTED_VISA_CLASSES, default=list(EXPECTED_VISA_CLASSES))
    parser.add_argument("--p30-prereg-sha", required=True)
    parser.add_argument("--p30-uuid", default=P30_UUID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _teacher_regions(native_cache: np.memmap, indices: Sequence[int], masks: np.ndarray, device: torch.device) -> np.ndarray:
    values: list[np.ndarray] = []
    for row_index, cache_index in enumerate(indices):
        native = torch.from_numpy(np.array(native_cache[int(cache_index)], copy=True)).unsqueeze(1).to(device=device, dtype=torch.float32)
        mask = torch.from_numpy(masks[row_index : row_index + 1, None].astype(np.float32, copy=False)).to(device=device)
        utility, _ = forensic_utility_for_batch(native, mask)
        correction = patch_correction_from_actions(classify_actions(utility))
        values.append(pool_patch_map(correction).detach().cpu().numpy())
    if not values:
        raise RuntimeError("empty P30 teacher inventory")
    return np.concatenate(values, axis=0)


def _adapter_regions(
    checkpoint: Path,
    cache_root: Path,
    class_name: str,
    rows: Sequence[Mapping[str, Any]],
    provenance: Any,
    device: torch.device,
) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != P29_ADAPTER_SCHEMA or payload.get("held_class") != class_name or payload.get("status") != "FOLD_TRAINING_COMPLETE":
        raise RuntimeError(f"invalid frozen P29 adapter checkpoint: {checkpoint}")
    adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    adapter.load_state_dict(payload["state_dict"], strict=True)
    adapter.eval()
    tier_a = cache_root / "tier_a" / class_name
    manifest = json.loads((tier_a / "manifest.json").read_text(encoding="utf-8"))
    indices_by_id = {sample_id: index for index, sample_id in enumerate(manifest["sample_ids"])}
    indices = [indices_by_id[f"{row['class_name']}:{row['image_path']}"] for row in rows]
    seg_cache = np.load(tier_a / "seg_features.npy", mmap_mode="r", allow_pickle=False)
    regions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), 16):
            batch_indices = indices[start : start + 16]
            seg = torch.from_numpy(np.array(seg_cache[batch_indices], copy=True)).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
            regions.append(adapter(seg).detach().cpu().numpy())
    return np.concatenate(regions, axis=1)


def _directional_cosine(teacher: np.ndarray, student: np.ndarray) -> dict[str, float | int | None]:
    teacher = np.asarray(teacher, dtype=np.float64)
    student = np.asarray(student, dtype=np.float64)
    if teacher.shape != student.shape:
        raise RuntimeError(f"directional tensor shape mismatch: {teacher.shape} != {student.shape}")
    values: list[float] = []
    zero_student = 0
    for index in range(teacher.shape[1]):
        t = teacher[:, index].reshape(-1)
        s = student[:, index].reshape(-1)
        if not np.any(t):
            continue
        t_norm = float(np.linalg.norm(t))
        s_norm = float(np.linalg.norm(s))
        if s_norm == 0.0:
            zero_student += 1
            values.append(0.0)
        else:
            values.append(float(np.dot(t, s) / (t_norm * s_norm)))
    return {
        "mean": float(np.mean(values)) if values else None,
        "valid_sample_count": len(values),
        "zero_student_sample_count": zero_student,
    }


def _numeric_mean(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float | None:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.p30_uuid != P30_UUID:
        raise RuntimeError("P30 UUID does not match the frozen preregistration")
    load_and_audit_p30_preregistration(P30_PREREGISTRATION_PATH, args.p30_prereg_sha)
    scoring_gate = json.loads(args.scoring_gate.read_text(encoding="utf-8"))
    classes = tuple(args.classes)
    if scoring_gate.get("status") != "PASS" or scoring_gate.get("prediction_count") != len(classes):
        raise RuntimeError("P30 post-freeze diagnostics require a passing prediction-freeze scoring gate")
    rows = read_visa_metadata(args.metadata)
    provenance = p30_cache_provenance(args.metadata)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    transfer_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for class_name in classes:
        fold = loco_inventory(rows, class_name)
        prediction_path = args.run_root / class_name / "predictions" / "p30_held_predictions.pt"
        p29_prediction_path = P29_ROOT / class_name / "predictions" / "p29_held_predictions.pt"
        p29_checkpoint_path = P29_ROOT / class_name / "training" / "p29_region_adapter.pt"
        p30_payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
        p29_payload = torch.load(p29_prediction_path, map_location="cpu", weights_only=True)
        if p30_payload.get("schema_version") != "P30_IMMUTABLE_HELD_PREDICTIONS_V1" or p29_payload.get("schema_version") != "P29_IMMUTABLE_HELD_PREDICTIONS_V1":
            raise RuntimeError(f"immutable prediction schema mismatch for {class_name}")
        p30_by_path = {str(record["image_path"]): record for record in p30_payload["records"]}
        p29_by_path = {str(record["image_path"]): record for record in p29_payload["records"]}
        paths = [str(row["image_path"]) for row in fold.held_rows]
        if set(p30_by_path) != set(paths) or set(p29_by_path) != set(paths):
            raise RuntimeError(f"held prediction identity mismatch for {class_name}")
        native = np.stack([p30_by_path[path]["native_abnormal_probability"].numpy() for path in paths]).astype(np.float32, copy=False)
        p30_probability = np.stack([p30_by_path[path]["p30_abnormal_probability"].numpy() for path in paths]).astype(np.float32, copy=False)
        p29_probability = np.stack([p29_by_path[path]["p29_abnormal_probability"].numpy() for path in paths]).astype(np.float32, copy=False)
        p30_regions = np.stack([p30_by_path[path]["p30_region_residual"].numpy() for path in paths]).astype(np.float32, copy=False).transpose(1, 0, 2, 3)
        masks, mask_reads = _load_masks(fold.held_rows, args.visa_root)
        tier_a = args.cache_root / "tier_a" / class_name
        manifest = json.loads((tier_a / "manifest.json").read_text(encoding="utf-8"))
        index_by_id = {sample_id: index for index, sample_id in enumerate(manifest["sample_ids"])}
        indices = [index_by_id[f"{row['class_name']}:{row['image_path']}"] for row in fold.held_rows]
        native_cache = np.load(tier_a / "native_logits.npy", mmap_mode="r", allow_pickle=False)
        teacher = _teacher_regions(native_cache, indices, masks, device)
        teacher_staged = np.broadcast_to(teacher[None, ...], p30_regions.shape)
        p29_regions = _adapter_regions(p29_checkpoint_path, args.cache_root, class_name, fold.held_rows, provenance, device)
        p29_alignment = sign_alignment(teacher_staged, p29_regions)
        p30_alignment = sign_alignment(teacher_staged, p30_regions)
        p29_direction = _directional_cosine(teacher_staged, p29_regions)
        p30_direction = _directional_cosine(teacher_staged, p30_regions)
        transfer_rows.append({
            "class": class_name,
            "sample_count": len(paths),
            "held_mask_reads": int(mask_reads),
            "p29": {"directional_cosine": p29_direction, "alignment": p29_alignment, "magnitude": residual_magnitude_summary(p29_regions)},
            "p30": {"directional_cosine": p30_direction, "alignment": p30_alignment, "magnitude": residual_magnitude_summary(p30_regions)},
            "delta": {
                "directional_cosine": (p30_direction["mean"] - p29_direction["mean"]) if p30_direction["mean"] is not None and p29_direction["mean"] is not None else None,
                "sign_agreement": p30_alignment["sign_agreement"] - p29_alignment["sign_agreement"],
            },
        })
        stability_rows.append({
            "class": class_name,
            "sample_count": len(paths),
            "held_mask_reads": int(mask_reads),
            "p29": vectorized_pixel_shifts(native, p29_probability, masks),
            "p30": vectorized_pixel_shifts(native, p30_probability, masks),
            "p30_minus_p29": vectorized_pixel_shifts(p29_probability, p30_probability, masks),
        })
    transfer_macro = {
        "p29_directional_cosine": _numeric_mean(transfer_rows, ("p29", "directional_cosine", "mean")),
        "p30_directional_cosine": _numeric_mean(transfer_rows, ("p30", "directional_cosine", "mean")),
        "p29_sign_agreement": _numeric_mean(transfer_rows, ("p29", "alignment", "sign_agreement")),
        "p30_sign_agreement": _numeric_mean(transfer_rows, ("p30", "alignment", "sign_agreement")),
        "p29_pearson": _numeric_mean(transfer_rows, ("p29", "alignment", "pearson")),
        "p30_pearson": _numeric_mean(transfer_rows, ("p30", "alignment", "pearson")),
        "p29_spearman": _numeric_mean(transfer_rows, ("p29", "alignment", "spearman")),
        "p30_spearman": _numeric_mean(transfer_rows, ("p30", "alignment", "spearman")),
        "p29_mean_abs_residual": _numeric_mean(transfer_rows, ("p29", "magnitude", "mean_abs")),
        "p30_mean_abs_residual": _numeric_mean(transfer_rows, ("p30", "magnitude", "mean_abs")),
    }
    stability_macro: dict[str, Any] = {}
    for state in ("p29", "p30", "p30_minus_p29"):
        stability_macro[state] = {}
        for stratum in ("normal", "anomaly"):
            stability_macro[state][stratum] = {
                metric: _numeric_mean(stability_rows, (state, stratum, metric))
                for metric in ("mean", "median", "q95", "q99")
            }
    transfer_result = {
        "schema_version": "P30_TRANSFER_DIAGNOSTIC_V1",
        "p30_uuid": args.p30_uuid,
        "classes": transfer_rows,
        "macro": transfer_macro,
        "held_mask_reads_post_freeze": sum(row["held_mask_reads"] for row in transfer_rows),
        "runtime_seconds": time.perf_counter() - started,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
    }
    stability_result = {
        "schema_version": "P30_STABILITY_DIAGNOSTIC_V1",
        "p30_uuid": args.p30_uuid,
        "classes": stability_rows,
        "macro": stability_macro,
        "held_mask_reads_post_freeze": sum(row["held_mask_reads"] for row in stability_rows),
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
    }
    atomic_write_json(args.run_root / "P30_TRANSFER_DIAGNOSTIC.json", transfer_result)
    atomic_write_json(args.run_root / "P30_STABILITY_DIAGNOSTIC.json", stability_result)
    return {"transfer": transfer_result, "stability": stability_result}


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
