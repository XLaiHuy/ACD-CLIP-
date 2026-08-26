"""GT-free P28R1 replay parity qualification for frozen P27 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d

from model.phase2b_runtime import deploy_native_logits
from tools.sabra.data import EXPECTED_VISA_CLASSES
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_pool import symmetric_margin_delta, upsample_region_map


IMAGE_SIZE = 518
PATCH_GRID = (37, 37)
STAGES = 3
PATCH_COUNT = 37 * 37
TOLERANCE = 0.00002
P27_REPLAY_BATCH_SIZE = 1
CLASS_NAMES = tuple(EXPECTED_VISA_CLASSES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(observed: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = observed.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
    }


def _record_paths(payload: dict[str, Any]) -> list[str]:
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("immutable prediction records are missing")
    paths = [str(record.get("image_path")) for record in records]
    if any(not path for path in paths) or len(set(paths)) != len(paths):
        raise RuntimeError("immutable prediction paths are not unique")
    return paths


def _manifest_paths(manifest: dict[str, Any]) -> list[str]:
    values = manifest.get("sample_ids")
    if not isinstance(values, list) or not values:
        raise RuntimeError("Tier-A sample_ids are missing")
    paths = []
    for value in values:
        text = str(value)
        paths.append(text.split(":", 1)[1] if ":" in text else text)
    if len(set(paths)) != len(paths):
        raise RuntimeError("Tier-A sample identities are not unique")
    return paths


def _load_class_inputs(cache_root: Path, science_root: Path, held_class: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    cache = cache_root / held_class
    prediction_path = science_root / held_class / "predictions" / "p27_held_predictions.pt"
    checkpoint_path = science_root / held_class / "training" / "p27_region_adapter.pt"
    payload = torch.load(prediction_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "P27_IMMUTABLE_HELD_PREDICTIONS_V1":
        raise RuntimeError(f"{held_class}: immutable prediction schema mismatch")
    if payload.get("held_class") != held_class or payload.get("gt_used") is not False:
        raise RuntimeError(f"{held_class}: immutable prediction provenance mismatch")
    if payload.get("mask_reads") != 0:
        raise RuntimeError(f"{held_class}: immutable payload reports mask reads")
    checkpoint_hash = sha256_file(checkpoint_path)
    if payload.get("adapter_checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(f"{held_class}: adapter checkpoint hash mismatch")
    records = payload["records"]
    prediction_paths = _record_paths(payload)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    cache_paths = _manifest_paths(manifest)
    if set(prediction_paths) != set(cache_paths):
        raise RuntimeError(f"{held_class}: sample identity set mismatch")
    cache_index = {path: index for index, path in enumerate(cache_paths)}
    indices = np.asarray([cache_index[path] for path in prediction_paths], dtype=np.int64)
    seg = np.asarray(np.load(cache / "seg_features.npy", mmap_mode="r", allow_pickle=False)[indices], dtype=np.float32)
    native = np.asarray(np.load(cache / "native_logits.npy", mmap_mode="r", allow_pickle=False)[indices], dtype=np.float32)
    expected_native = np.stack([record["native_abnormal_probability"].numpy().astype(np.float32) for record in records])
    expected_student = np.stack([record["p27_abnormal_probability"].numpy().astype(np.float32) for record in records])
    if seg.shape[0] != len(records) or native.shape != (len(records), STAGES, PATCH_COUNT, 2):
        raise RuntimeError(f"{held_class}: cache shape mismatch")
    return seg, native, expected_native, expected_student, checkpoint_hash


def _load_adapter(checkpoint_path: Path, device: torch.device) -> RegionResidualAdapter:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    adapter = RegionResidualAdapter().to(device=device, dtype=torch.float32)
    adapter.load_state_dict(payload["state_dict"], strict=True)
    adapter.eval()
    return adapter


def _replay_batch(
    seg: np.ndarray,
    native: np.ndarray,
    adapter: RegionResidualAdapter,
    device: torch.device,
    capture: bool = False,
) -> dict[str, np.ndarray]:
    seg_tensor = torch.from_numpy(seg).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
    native_tensor = torch.from_numpy(native).permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        region = adapter(seg_tensor)
        patch = upsample_region_map(region)
        corrected = symmetric_margin_delta(native_tensor, patch)
        probability, deployed_logits = deploy_native_logits(corrected, domain="Industrial")
        native_probability, native_deployed_logits = deploy_native_logits(native_tensor, domain="Industrial")
    result = {
        "native": native_probability[:, 1].cpu().numpy(),
        "student": probability[:, 1].cpu().numpy(),
    }
    if capture:
        blurred = []
        resized = []
        for stage in range(STAGES):
            stage_logits = corrected[stage].permute(0, 2, 1).reshape(corrected.shape[1], 2, *PATCH_GRID)
            blurred_stage = gaussian_blur2d(stage_logits, (7, 7), (1.0, 1.0))
            blurred.append(blurred_stage)
            resized.append(F.interpolate(blurred_stage, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True))
        result.update({
            "region": region.permute(1, 0, 2, 3).cpu().numpy(),
            "patch": patch.permute(1, 0, 2, 3).cpu().numpy(),
            "corrected": corrected.permute(1, 0, 2, 3).cpu().numpy(),
            "post_blur": torch.stack(blurred, dim=0).permute(1, 0, 2, 3, 4).cpu().numpy(),
            "resized": torch.stack(resized, dim=0).permute(1, 0, 2, 3, 4).cpu().numpy(),
            "stage_mean": deployed_logits.cpu().numpy(),
            "final": probability[:, 1].cpu().numpy(),
            "native_stage_mean": native_deployed_logits.cpu().numpy(),
            "native_final": native_probability[:, 1].cpu().numpy(),
        })
    return result


def qualify_class(cache_root: Path, science_root: Path, held_class: str, device: torch.device) -> dict[str, Any]:
    seg, native, expected_native, expected_student, checkpoint_hash = _load_class_inputs(cache_root, science_root, held_class)
    adapter = _load_adapter(science_root / held_class / "training" / "p27_region_adapter.pt", device)
    native_observed = []
    student_observed = []
    with torch.no_grad():
        for index in range(0, len(seg), P27_REPLAY_BATCH_SIZE):
            batch = _replay_batch(seg[index:index + P27_REPLAY_BATCH_SIZE], native[index:index + P27_REPLAY_BATCH_SIZE], adapter, device)
            native_observed.append(batch["native"])
            student_observed.append(batch["student"])
    native_map = np.concatenate(native_observed, axis=0)
    student_map = np.concatenate(student_observed, axis=0)
    native_stats = stats(native_map, expected_native)
    student_stats = stats(student_map, expected_student)
    return {
        "class": held_class,
        "samples": int(len(seg)),
        "checkpoint_sha256": checkpoint_hash,
        "native": native_stats,
        "student": student_stats,
        "native_pass": bool(native_stats["max_abs"] <= TOLERANCE),
        "student_pass": bool(student_stats["max_abs"] <= TOLERANCE),
        "gt_reads": 0,
        "mask_reads": 0,
    }


def _forensic_candle(cache_root: Path, science_root: Path, device: torch.device) -> dict[str, Any]:
    held_class = "candle"
    seg, native, expected_native, expected_student, _ = _load_class_inputs(cache_root, science_root, held_class)
    adapter = _load_adapter(science_root / held_class / "training" / "p27_region_adapter.pt", device)
    stages = ("region", "patch", "corrected", "post_blur", "resized", "stage_mean", "final")
    accum = {name: {"max_abs": 0.0, "sum_abs": 0.0, "sum_sq": 0.0, "count": 0} for name in stages}
    batch_one_final = []
    batch_four_final = []
    for start in range(0, len(seg), 4):
        stop = min(start + 4, len(seg))
        one = [_replay_batch(seg[index:index + 1], native[index:index + 1], adapter, device, capture=True) for index in range(start, stop)]
        four = _replay_batch(seg[start:stop], native[start:stop], adapter, device, capture=True)
        for offset, one_item in enumerate(one):
            for name in stages:
                reference = one_item[name][0] if name not in ("stage_mean",) else one_item[name][0]
                observed = four[name][offset]
                delta = observed.astype(np.float64) - reference.astype(np.float64)
                item = accum[name]
                item["max_abs"] = max(item["max_abs"], float(np.max(np.abs(delta))))
                item["sum_abs"] += float(np.abs(delta).sum())
                item["sum_sq"] += float((delta * delta).sum())
                item["count"] += int(delta.size)
            batch_one_final.append(one_item["final"][0])
            batch_four_final.append(four["final"][offset])
    for item in accum.values():
        item["mean_abs"] = item["sum_abs"] / item["count"]
        item["rmse"] = math.sqrt(item["sum_sq"] / item["count"])
        del item["sum_abs"], item["sum_sq"], item["count"]
    one_final = np.stack(batch_one_final)
    four_final = np.stack(batch_four_final)
    return {
        "held_class": held_class,
        "gt_reads": 0,
        "mask_reads": 0,
        "p27_batch_size": 1,
        "incorrect_replay_batch_size": 4,
        "first_divergence_stage": "adapter_region_residual",
        "root_cause_class": "OTHER_EXACTLY_IDENTIFIED_ENGINEERING_CAUSE",
        "root_cause": "P27 immutable evaluation used batch-size 1, while the failed P28 replay used batch-size 4; the adapter GPU path is numerically batch-size dependent.",
        "batch_one_vs_immutable_final": {
            "native": stats(_replay_batch(seg[:1], native[:1], adapter, device)["native"], expected_native[:1]),
            "student": stats(one_final, np.asarray(expected_student)),
        },
        "batch_four_vs_batch_one": accum,
        "batch_four_vs_immutable_final": stats(four_final, expected_student),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("P28R1 parity requires the CUDA runtime used by frozen P27 evaluation")
    if args.batch_size != P27_REPLAY_BATCH_SIZE:
        raise RuntimeError("P28R1 parity batch size must remain exactly 1")
    device = torch.device(args.device)
    started = time.perf_counter()
    rows = [qualify_class(args.cache_root, args.science_root, name, device) for name in CLASS_NAMES]
    forensic = _forensic_candle(args.cache_root, args.science_root, device) if args.forensic_candle else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "P28R1_PARITY_TABLE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "samples", "native_max_abs", "native_mean_abs", "native_rmse", "student_max_abs", "student_mean_abs", "student_rmse", "native_pass", "student_pass"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"class": row["class"], "samples": row["samples"], "native_max_abs": row["native"]["max_abs"], "native_mean_abs": row["native"]["mean_abs"], "native_rmse": row["native"]["rmse"], "student_max_abs": row["student"]["max_abs"], "student_mean_abs": row["student"]["mean_abs"], "student_rmse": row["student"]["rmse"], "native_pass": row["native_pass"], "student_pass": row["student_pass"]})
    summary = {
        "schema_version": "P28R1_PARITY_QUALIFICATION_V1",
        "tolerance": TOLERANCE,
        "batch_size": P27_REPLAY_BATCH_SIZE,
        "classes": rows,
        "native_pass_count": sum(row["native_pass"] for row in rows),
        "student_pass_count": sum(row["student_pass"] for row in rows),
        "all_native_pass": all(row["native_pass"] for row in rows),
        "all_student_pass": all(row["student_pass"] for row in rows),
        "gt_reads": 0,
        "mask_reads": 0,
        "new_clip_forwards": 0,
        "new_phase2b_forwards": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "P28R1_PARITY_QUALIFICATION.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if forensic is not None:
        (args.output_dir / "P28R1_PARITY_FORENSIC.json").write_text(json.dumps(forensic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("/workspace/p27r1_cache_v1/tier_a"))
    parser.add_argument("--science-root", type=Path, default=Path("/workspace/p27r1_science_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=P27_REPLAY_BATCH_SIZE)
    parser.add_argument("--forensic-candle", action="store_true")
    return parser


if __name__ == "__main__":
    run(make_parser().parse_args())
