#!/usr/bin/env python3
"""Two-phase SABRA source calibration and MVTec lambda selection.

The default paths are real, bounded-by-CLI pipelines.  ``fit-source`` first
builds and finalizes a GT-free VisA cache (images, native Phase2B tensors, and
relational evidence only), then opens VisA masks to create the audited patch
occupancy Trust targets and the detached Need oracle targets.  ``select-lambda``
uses the same one-forward-per-image MVTec cache, opens MVTec GT only after its
GT-free manifest is finalized, and evaluates the preregistered coarse/refined
lambda grids with the shared exact evaluator.

``--records-json`` and ``--curve-json`` remain explicit debug/test seams; they
never masquerade as the real default path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from dataset.info import CLASS_NAMES, dataset_domain, is_medical_dataset
from evaluation.datasets import EXPECTED_MVTEC_CLASSES, resolve_mvtec_root
from evaluation.evaluator import evaluate_records, image_score
from evaluation.metrics import selection_score
from model.phase2b_runtime import (
    compute_deployment_sensitivity,
    configure_canonical_fp32,
    deploy_with_delta,
    forward_phase2b,
    load_json_config,
    load_phase2b_checkpoint,
    sha256_file,
)
from tools.sabra.artifacts import build_freeze_payload, validate_source_calibration, write_json
from tools.sabra.correction import build_delta, correction_values, margin_scale_p90
from tools.sabra.data import CLIP_MEAN, CLIP_STD, read_visa_metadata, safe_data_path
from tools.sabra.need import fit_need, need_oracle
from tools.sabra.relational import BACKEND_VERSION, FEATURE_ORDER, NEED_ORDER, assert_gt_free_payload, build_relational_record, need_features, trust_features
from tools.sabra.trust import fit_trust, frozen_probability

PROTOCOL_VERSION = "SABRA_CANONICAL_V1"
MEDICAL_DATASETS = tuple(name for name in CLASS_NAMES if is_medical_dataset(name))
COARSE_LAMBDAS = tuple(float(value) for value in np.round(np.arange(0.0, 1.0001, 0.025), 6))
REFINEMENT_RULE = "center +/- 0.05 clamped to [0,1], step 0.005; no duplicate coarse points"
PATCH_GRID = (37, 37)
PATCHES = PATCH_GRID[0] * PATCH_GRID[1]
IMAGE_SIZE = 518
GT_FREE_FIELDS = (
    "native_logits",
    "native_margins",
    "native_pixel_probability",
    "classification_probability",
    "deployment_sensitivity",
    "E",
    "peer_coherence",
    "query_support_mean",
    "peer_eigen_entropy",
    "stage_query_profile_disagreement",
    "margin_within_image_rank",
    "robust_margin_normalization",
    "D_rank",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _resolve_wrapped_root(root: Path, first_relative: str) -> Path:
    root = root.expanduser().resolve()
    if (root / first_relative).is_file():
        return root
    wrapped = root / "VisA_20220922"
    if (wrapped / first_relative).is_file():
        return wrapped
    wrapped = root / "mvtec_anomaly_detection"
    if (wrapped / first_relative).is_file():
        return wrapped
    raise FileNotFoundError(f"dataset root does not contain {first_relative}: {root}")


def _sanitized_rows(metadata_path: Path, expected_classes: Sequence[str]) -> list[dict[str, str]]:
    raw = read_visa_metadata(metadata_path)
    rows = [{"class_name": str(row["class_name"]), "image_path": str(row["image_path"])} for row in raw]
    if not rows:
        raise ValueError(f"empty dataset metadata: {metadata_path}")
    classes = tuple(sorted({row["class_name"] for row in rows}))
    if classes != tuple(sorted(str(name) for name in expected_classes)):
        raise ValueError(f"dataset class set mismatch: {classes}")
    if len({(row["class_name"], row["image_path"]) for row in rows}) != len(rows):
        raise ValueError("dataset metadata contains duplicate image identities")
    return sorted(rows, key=lambda row: (row["class_name"], row["image_path"]))


class _ImageOnlyDataset(Dataset):
    """Deterministic image-only dataset used before any GT is opened."""

    def __init__(self, rows: Sequence[Mapping[str, str]], root: Path, image_size: int = IMAGE_SIZE) -> None:
        self.rows = [{"class_name": str(row["class_name"]), "image_path": str(row["image_path"])} for row in rows]
        self.root = root.resolve()
        self.transform = transforms.Compose([
            transforms.Resize((int(image_size), int(image_size)), InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        path = safe_data_path(self.root, row["image_path"])
        with Image.open(path) as handle:
            image = self.transform(handle.convert("RGB")).contiguous()
        return {"image": image, "class_name": row["class_name"], "image_path": row["image_path"], "index": int(index)}


def _selection_checkpoint(selection: Mapping[str, Any]) -> tuple[Path, str]:
    value = selection.get("selected_checkpoint")
    expected = selection.get("selected_checkpoint_sha256")
    if not value or not expected:
        raise ValueError("Phase2B selection must identify selected_checkpoint and selected_checkpoint_sha256")
    path = Path(str(value)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"selected checkpoint SHA256 mismatch: expected {expected}, got {actual}")
    return path.resolve(), actual


def _source_hashes(
    checkpoint: Path,
    config_path: Path,
    clip_asset: Path,
    metadata_path: Path,
) -> dict[str, str]:
    paths = {
        "checkpoint": checkpoint,
        "config": config_path,
        "clip_asset": clip_asset,
        "metadata": metadata_path,
    }
    for relative in (
        "model/phase2b_runtime.py",
        "model/phase2b_schedule.py",
        "tools/sabra/relational.py",
        "tools/sabra/trust_v2/numerical.py",
        "tools/sabra/trust_v2/fast_geometry.py",
        "tools/sabra/need.py",
    ):
        candidate = Path(relative)
        if candidate.is_file():
            paths[relative] = candidate
    return {name: _sha256_file(path) for name, path in paths.items()}


def fit_source_payload(
    records: Iterable[Mapping[str, Any]],
    selected_epoch: int,
    checkpoint_sha256: str,
    margin_values: np.ndarray,
    git_sha: str,
    source_hashes: Mapping[str, str] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    relational_backend: str = "fast",
    gt_free_manifest: Mapping[str, Any] | None = None,
    gt_target_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("VisA source calibration requires non-empty records")
    def _medical_row(row: Mapping[str, Any]) -> bool:
        name = str(row.get("class_name"))
        return name.lower() == "medical" or (name in CLASS_NAMES and is_medical_dataset(name))
    if any(_medical_row(row) for row in rows):
        raise ValueError("Medical records cannot enter VisA source calibration")
    if str(relational_backend).lower() not in {"exact", "fast"}:
        raise ValueError("relational backend must be exact or fast")
    trust = fit_trust(rows)
    need = fit_need(rows)
    provenance: dict[str, Any] = {"git_sha": str(git_sha)}
    if source_hashes is not None:
        provenance["critical_source_hashes"] = dict(source_hashes)
    phase2b: dict[str, Any] = {"selected_epoch": int(selected_epoch), "checkpoint_sha256": str(checkpoint_sha256)}
    if checkpoint_path is not None:
        phase2b["selected_checkpoint"] = str(checkpoint_path)
    relational = {
        "implementation": "tools.sabra.relational.build_relational_record",
        "backend": str(relational_backend).lower(),
        "backend_version": BACKEND_VERSION,
        "peer_count": 8,
        "feature_contract": list(FEATURE_ORDER),
    }
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "SOURCE_FITTED",
        "phase2b": phase2b,
        "relational": relational,
        "trust": {**trust, "feature_order": list(FEATURE_ORDER)},
        "need": {**need, "feature_order": list(NEED_ORDER)},
        "margin_scale": margin_scale_p90(margin_values),
        "provenance": provenance,
    }
    if gt_free_manifest is not None:
        payload["gt_free_cache"] = dict(gt_free_manifest)
    if gt_target_manifest is not None:
        payload["gt_target_manifest"] = dict(gt_target_manifest)
    validate_source_calibration(payload)
    return payload


def lambda_grid() -> np.ndarray:
    return np.asarray(COARSE_LAMBDAS, dtype=np.float64)


def refined_lambda_grid(center: float, exclude: Sequence[float] = ()) -> np.ndarray:
    lo = max(0.0, float(center) - 0.05)
    hi = min(1.0, float(center) + 0.05)
    grid = np.round(np.arange(lo, hi + 0.0001, 0.005), 6)
    excluded = {round(float(value), 6) for value in exclude}
    return np.asarray([value for value in grid if round(float(value), 6) not in excluded], dtype=np.float64)


def select_lambda(curve: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    required = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")
    seen: set[float] = set()
    for row in curve:
        if "lambda" not in row or any(name not in row for name in required):
            raise ValueError("lambda curve rows must include lambda and four exact metrics")
        value = float(row["lambda"])
        if not 0.0 <= value <= 1.0:
            raise ValueError("lambda must be in [0,1]")
        rounded = round(value, 6)
        if rounded in seen:
            raise ValueError(f"duplicate lambda curve point: {value}")
        seen.add(rounded)
        if any(row[name] is None for name in required):
            raise ValueError("lambda selection requires all four defined MVTec metrics")
        metrics = {name: float(row[name]) for name in required}
        if not all(0.0 <= metric <= 1.0 for metric in metrics.values()):
            raise ValueError("lambda curve metrics must be in [0,1]")
        rows.append(dict(row) | {"score": selection_score(metrics), "lambda": value})
    if not rows:
        raise ValueError("empty lambda curve")
    return min(rows, key=lambda row: (-float(row["score"]), float(row["lambda"])))


def _reject_medical(dataset: str) -> None:
    if is_medical_dataset(str(dataset)) or str(dataset).lower() == "medical":
        raise SystemExit("Medical is final zero-shot test data and cannot be a calibration input")


def _read_curve(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("curve", payload.get("rows"))
    if not isinstance(payload, list):
        raise ValueError("curve JSON must be a list or an object containing curve/rows")
    return [dict(row) for row in payload]


def _write_lambda_outputs(
    output_dir: Path,
    source: Mapping[str, Any],
    rows: list[dict[str, Any]],
    selected: Mapping[str, Any],
    git_sha: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["lambda", "pixel_auroc", "pixel_ap", "image_auroc", "image_ap", "score"]
    extra = [name for name in rows[0] if name not in fieldnames]
    fieldnames.extend(extra)
    with (output_dir / "lambda_selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    freeze = build_freeze_payload(
        source,
        selected_lambda=float(selected["lambda"]),
        selected_score=float(selected["score"]),
        git_sha=git_sha,
        coarse_grid=COARSE_LAMBDAS,
        refinement_rule=REFINEMENT_RULE,
    )
    write_json(output_dir / "SABRA_FREEZE.json", freeze)


def _write_gt_free_cache(
    output_dir: Path,
    dataset_name: str,
    records: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    *,
    backend: str,
) -> dict[str, Any]:
    cache_root = output_dir / "gt_free_cache"
    if cache_root.exists() and any(cache_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing GT-free cache: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        assert_gt_free_payload(record)
        forbidden = {"label", "mask", "mask_path", "pixel_labels", "image_labels", "trust_target", "need_target"}
        if forbidden.intersection(record):
            raise AssertionError("GT-bearing field reached GT-free cache")
        grouped[str(record["class_name"])].append(record)
    shards: dict[str, str] = {}
    for class_name in sorted(grouped):
        rows = grouped[class_name]
        arrays: dict[str, np.ndarray] = {}
        for field in GT_FREE_FIELDS:
            if any(field not in row for row in rows):
                raise ValueError(f"GT-free record missing {field}: {class_name}")
            arrays[field] = np.stack([np.asarray(row[field]) for row in rows], axis=0)
        arrays["image_path"] = np.asarray([str(row["image_path"]) for row in rows], dtype="U512")
        shard = cache_root / f"{class_name}.npz"
        np.savez_compressed(shard, **arrays)
        shards[class_name] = _sha256_file(shard)
    manifest = {
        "status": "PASS",
        "GT_FREE_CACHE_FINALIZED": True,
        "immutable": True,
        "dataset": str(dataset_name),
        "backend": str(backend).lower(),
        "record_count": len(records),
        "classes": sorted(grouped),
        "shards": shards,
        "source_hashes": dict(source_hashes),
        "labels_read": False,
        "mask_paths_read": False,
        "mask_pixels_read": 0,
        "scientific_metrics_observed": False,
        "medical_reads": 0,
    }
    write_json(output_dir / "GT_FREE_MANIFEST.json", manifest)
    return manifest


def _target_rows(metadata_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_visa_metadata(metadata_path)
    return {(str(row["class_name"]), str(row["image_path"])): dict(row) for row in rows}


def _mask_occupancy(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binary = np.asarray(mask, dtype=np.float32)
    if binary.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"mask must be [{IMAGE_SIZE},{IMAGE_SIZE}], got {binary.shape}")
    occupancy = binary.reshape(PATCH_GRID[0], 14, PATCH_GRID[1], 14).mean(axis=(1, 3)).astype(np.float32).reshape(-1)
    return (occupancy > 0).astype(np.int8), occupancy


def _attach_visa_targets(
    records: list[dict[str, Any]],
    data_root: Path,
    metadata_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows = _target_rows(metadata_path)
    mask_pixels = 0
    for record in records:
        row = rows[(str(record["class_name"]), str(record["image_path"]))]
        if int(row["label"]):
            mask_path = safe_data_path(data_root, str(row["mask_path"]))
            with Image.open(mask_path) as handle:
                mask = np.asarray(handle.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST), dtype=np.float32) / 255.0
            mask_pixels += int(mask.size)
        else:
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        target, occupancy = _mask_occupancy(mask)
        native = torch.from_numpy(np.asarray(record["native_logits"], dtype=np.float32)).unsqueeze(1)
        target_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        oracle = need_oracle(native, target_mask, domain="Industrial")
        record["trust_target"] = target
        record["need_target"] = np.asarray(oracle["target"], dtype=np.int8).reshape(-1)
        record["target_occupancy"] = occupancy
        record.pop("native_logits", None)
    manifest = {
        "status": "PASS",
        "GT_FREE_CACHE_FINALIZED": True,
        "target_definition": "patch occupancy > 0 from 518x518 mask in row-major 14x14 blocks",
        "need_target_definition": "need_oracle target = signed utility > 1e-8 using utils.calculate_seg_loss",
        "labels_read_after_cache": True,
        "mask_paths_read_after_cache": True,
        "mask_pixels_read": int(mask_pixels),
        "medical_reads": 0,
    }
    write_json(output_dir / "GT_TARGET_MANIFEST.json", manifest)
    return manifest


def _build_gt_free_records(
    dataset_name: str,
    rows: Sequence[Mapping[str, str]],
    data_root: Path,
    checkpoint: Path,
    config_path: Path,
    clip_asset: Path,
    device: torch.device,
    *,
    backend: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configure_canonical_fp32()
    config = load_json_config(config_path)
    model = load_phase2b_checkpoint(checkpoint, config, clip_asset, device)
    model.eval()
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["class_name"])].append(row)
    records: list[dict[str, Any]] = []
    progress = tqdm(total=len(rows), desc=f"{dataset_name} GT-free", unit="img")
    started = time.perf_counter()
    peak_reserved = 0
    try:
        for class_name in sorted(grouped):
            dataset = _ImageOnlyDataset(grouped[class_name], data_root, IMAGE_SIZE)
            loader_kwargs: dict[str, Any] = {
                "batch_size": int(batch_size),
                "shuffle": False,
                "num_workers": int(num_workers),
                "pin_memory": device.type == "cuda",
            }
            if int(num_workers) > 0:
                loader_kwargs["persistent_workers"] = True
                loader_kwargs["prefetch_factor"] = int(prefetch_factor)
            loader = DataLoader(dataset, **loader_kwargs)
            from utils import get_phase2b_global_text_features
            text = get_phase2b_global_text_features(
                model,
                dataset_name,
                [class_name],
                device,
                use_hybrid_soft_prompt=bool(config.get("use_hybrid_soft_prompt", False)),
                use_soft_prompt=bool(config.get("use_soft_prompt", False)),
            ).float()
            for batch in loader:
                images = batch["image"].to(device, non_blocking=device.type == "cuda").float()
                names = [class_name] * int(images.shape[0])
                text_batch = text.expand(-1, int(images.shape[0]), -1, -1)
                forward = forward_phase2b(
                    model,
                    images,
                    names,
                    device,
                    config,
                    domain=dataset_domain(dataset_name),
                    require_grad=False,
                    dataset_name=dataset_name,
                    precomputed_text_features=text_batch,
                )
                sensitivity = compute_deployment_sensitivity(forward.native_logits, domain=dataset_domain(dataset_name)).cpu().numpy().astype(np.float32)
                if not np.isfinite(sensitivity).all() or np.any(np.abs(sensitivity).sum(axis=1) <= 0):
                    raise FloatingPointError("GT-free deployment sensitivity is missing or all zero")
                for index, image_path in enumerate(list(batch["image_path"])):
                    record = build_relational_record(
                        forward.seg_features[:, index].detach().cpu().numpy(),
                        forward.native_margin[:, index].detach().cpu().numpy(),
                        deployment_sensitivity=sensitivity[index],
                        image_path=str(image_path),
                        backend=backend,
                    )
                    record.update({
                        "class_name": class_name,
                        "native_logits": forward.native_logits[:, index].detach().cpu().numpy().astype(np.float32),
                        "native_margins": forward.native_margin[:, index].detach().cpu().numpy().astype(np.float32),
                        "native_pixel_probability": forward.native_segmentation_probability[index].detach().cpu().numpy().astype(np.float32),
                        "classification_probability": np.float32(forward.classification_probability[index].detach().cpu()),
                    })
                    assert_gt_free_payload(record)
                    records.append(record)
                progress.update(int(images.shape[0]))
                if device.type == "cuda":
                    peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
    finally:
        progress.close()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elapsed = max(time.perf_counter() - started, 1e-9)
    return records, {
        "images": len(records),
        "samples_per_sec": float(len(records) / elapsed),
        "elapsed_seconds": float(elapsed),
        "peak_reserved_vram": peak_reserved or None,
        "device": str(device),
        "backend": str(backend).lower(),
    }


def _mvt_target_records(records: Sequence[Mapping[str, Any]], data_root: Path, metadata_path: Path) -> list[dict[str, Any]]:
    rows = _target_rows(metadata_path)
    output: list[dict[str, Any]] = []
    for record in records:
        row = rows[(str(record["class_name"]), str(record["image_path"]))]
        if int(row["label"]):
            mask_path = safe_data_path(data_root, str(row["mask_path"]))
            with Image.open(mask_path) as handle:
                mask = np.asarray(handle.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST), dtype=np.float32) / 255.0
        else:
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        output.append({**record, "pixel_labels": (mask > 0).astype(np.int8).reshape(-1), "image_labels": np.asarray([int(row["label"])], dtype=np.int8)})
    return output


def _lambda_records(records: Sequence[Mapping[str, Any]], source: Mapping[str, Any], lambda_value: float) -> list[dict[str, Any]]:
    margin_scale = float(source["margin_scale"]["value"])
    output: list[dict[str, Any]] = []
    for record in records:
        trust = frozen_probability(source["trust"], trust_features(record))
        need = frozen_probability(source["need"], need_features(record))
        correction = correction_values(lambda_value, margin_scale, trust, need)
        native = torch.from_numpy(np.asarray(record["native_logits"], dtype=np.float32)).unsqueeze(1)
        delta = build_delta(native, correction.reshape(1, -1))
        with torch.inference_mode():
            corrected, _ = deploy_with_delta(native, delta, domain="Industrial")
        native_pixels = np.asarray(record["native_pixel_probability"], dtype=np.float32)
        corrected_pixels = corrected[0, 1].cpu().numpy().astype(np.float32)
        native_cls = float(record["classification_probability"])
        output.append({
            "class_name": record["class_name"],
            "image_path": record["image_path"],
            "pixel_labels": np.asarray(record["pixel_labels"], dtype=np.int8),
            "image_labels": np.asarray(record["image_labels"], dtype=np.int8),
            "phase2b": {"pixel_scores": native_pixels.reshape(-1), "image_scores": np.asarray([image_score(native_cls, float(native_pixels.max()), "Industrial")])},
            "sabra": {"pixel_scores": corrected_pixels.reshape(-1), "image_scores": np.asarray([image_score(native_cls, float(corrected_pixels.max()), "Industrial")])},
        })
    return output


def _real_lambda_curve(
    records: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    *,
    chunk_size: int,
) -> list[dict[str, Any]]:
    coarse = lambda_grid().tolist()
    rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(coarse), max(1, int(chunk_size))), desc="SABRA lambda coarse", unit="chunk"):
        for value in coarse[start:start + max(1, int(chunk_size))]:
            metrics = evaluate_records(_lambda_records(records, source, value), method="sabra")["macro"]
            rows.append({"lambda": float(value), **metrics, "grid": "coarse"})
    coarse_selected = select_lambda(rows)
    refined = refined_lambda_grid(float(coarse_selected["lambda"]), exclude=coarse)
    for start in tqdm(range(0, len(refined), max(1, int(chunk_size))), desc="SABRA lambda refine", unit="chunk"):
        for value in refined[start:start + max(1, int(chunk_size))]:
            metrics = evaluate_records(_lambda_records(records, source, value), method="sabra")["macro"]
            rows.append({"lambda": float(value), **metrics, "grid": "refined"})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit-source", help="fit Trust/Need and margin scale on VisA only")
    fit.add_argument("--phase2b-selection", type=Path, required=True)
    fit.add_argument("--visa-root", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--dataset", default="VisA")
    fit.add_argument("--records-json", type=Path, help="debug-only precomputed records")
    fit.add_argument("--config", type=Path, default=Path("configs/phase2b_canonical_v1.json"))
    fit.add_argument("--clip-asset", type=Path)
    fit.add_argument("--metadata", type=Path, default=Path("dataset/hub/VisA.jsonl"))
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--batch-size", type=int, default=6)
    fit.add_argument("--num-workers", type=int, default=4)
    fit.add_argument("--prefetch-factor", type=int, default=2)
    fit.add_argument("--backend", choices=["exact", "fast"], default="fast")
    fit.add_argument("--git-sha", default="WORKTREE_SHA")

    select = subparsers.add_parser("select-lambda", help="select frozen correction scale on MVTec development data")
    select.add_argument("--source-calibration", type=Path, required=True)
    select.add_argument("--mvtec-root", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    select.add_argument("--curve-json", type=Path, help="debug-only precomputed curve")
    select.add_argument("--config", type=Path, default=Path("configs/phase2b_canonical_v1.json"))
    select.add_argument("--clip-asset", type=Path)
    select.add_argument("--metadata", type=Path, default=Path("dataset/hub/MVTec.jsonl"))
    select.add_argument("--device", default="cuda")
    select.add_argument("--batch-size", type=int, default=6)
    select.add_argument("--num-workers", type=int, default=4)
    select.add_argument("--prefetch-factor", type=int, default=2)
    select.add_argument("--lambda-chunk-size", type=int, default=8)
    select.add_argument("--backend", choices=["exact", "fast"], default="fast")
    select.add_argument("--git-sha", default="WORKTREE_SHA")

    args = parser.parse_args(argv)
    if args.command == "fit-source":
        _reject_medical(args.dataset)
        selection = json.loads(args.phase2b_selection.read_text(encoding="utf-8"))
        if selection.get("status") != "FROZEN":
            raise SystemExit("phase2b selection must be FROZEN")
        if args.records_json is not None:
            rows = json.loads(args.records_json.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise SystemExit("--records-json must contain a list")
            payload = fit_source_payload(
                rows,
                int(selection["selected_epoch"]),
                str(selection["selected_checkpoint_sha256"]),
                np.asarray([value for row in rows for value in row["native_margins"]], dtype=np.float64),
                args.git_sha,
            )
            write_json(args.output_dir / "sabra_source_calibration.json", payload)
            return 0
        if args.clip_asset is None or not args.clip_asset.is_file():
            raise SystemExit("real fit-source requires --clip-asset")
        checkpoint, checkpoint_sha = _selection_checkpoint(selection)
        metadata_path = args.metadata.resolve()
        rows = _sanitized_rows(metadata_path, CLASS_NAMES["VisA"])
        root = _resolve_wrapped_root(args.visa_root, rows[0]["image_path"])
        source_hashes = _source_hashes(checkpoint, args.config.resolve(), args.clip_asset.resolve(), metadata_path)
        records, runtime = _build_gt_free_records(
            "VisA", rows, root, checkpoint, args.config.resolve(), args.clip_asset.resolve(), torch.device(args.device),
            backend=args.backend, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        )
        cache_manifest = _write_gt_free_cache(args.output_dir, "VisA", records, source_hashes, backend=args.backend)
        target_manifest = _attach_visa_targets(records, root, metadata_path, args.output_dir)
        payload = fit_source_payload(
            records,
            int(selection["selected_epoch"]),
            checkpoint_sha,
            np.concatenate([np.asarray(row["native_margins"], dtype=np.float64).reshape(-1) for row in records]),
            args.git_sha,
            source_hashes,
            checkpoint_path=checkpoint,
            relational_backend=args.backend,
            gt_free_manifest=cache_manifest,
            gt_target_manifest=target_manifest,
        )
        payload["runtime"] = runtime
        write_json(args.output_dir / "sabra_source_calibration.json", payload)
        return 0

    source = json.loads(args.source_calibration.read_text(encoding="utf-8"))
    validate_source_calibration(source)
    root = resolve_mvtec_root(args.mvtec_root)
    if root is None or not root.exists():
        raise SystemExit(f"MVTec root does not exist: {args.mvtec_root}")
    if args.curve_json is not None:
        rows = _read_curve(args.curve_json)
        scored = []
        for row in rows:
            scored.append(select_lambda([row]))
        selected = select_lambda(scored)
        _write_lambda_outputs(args.output_dir, source, scored, selected, args.git_sha)
        return 0
    if args.clip_asset is None or not args.clip_asset.is_file():
        raise SystemExit("real lambda selection requires --clip-asset")
    checkpoint_value = source.get("phase2b", {}).get("selected_checkpoint")
    if not checkpoint_value:
        raise SystemExit("source calibration lacks phase2b.selected_checkpoint for real lambda selection")
    checkpoint = Path(str(checkpoint_value)).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    expected_sha = str(source["phase2b"]["checkpoint_sha256"])
    actual_sha = _sha256_file(checkpoint)
    if actual_sha != expected_sha:
        raise SystemExit(f"source calibration checkpoint SHA256 mismatch: expected={expected_sha} actual={actual_sha}")
    metadata_path = args.metadata.resolve()
    rows = _sanitized_rows(metadata_path, EXPECTED_MVTEC_CLASSES)
    resolved_root = _resolve_wrapped_root(root, rows[0]["image_path"])
    source_hashes = _source_hashes(checkpoint, args.config.resolve(), args.clip_asset.resolve(), metadata_path)
    records, runtime = _build_gt_free_records(
        "MVTec", rows, resolved_root, checkpoint, args.config.resolve(), args.clip_asset.resolve(), torch.device(args.device),
        backend=args.backend, batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
    )
    cache_manifest = _write_gt_free_cache(args.output_dir, "MVTec", records, source_hashes, backend=args.backend)
    target_records = _mvt_target_records(records, resolved_root, metadata_path)
    curve = _real_lambda_curve(target_records, source, chunk_size=args.lambda_chunk_size)
    selected = select_lambda(curve)
    _write_lambda_outputs(args.output_dir, source, curve, selected, args.git_sha)
    write_json(args.output_dir / "lambda_runtime.json", {"gt_free_manifest": cache_manifest, "runtime": runtime, "source_hashes": source_hashes, "lambda_chunk_size": int(args.lambda_chunk_size)})
    print(f"LAMBDA_SELECTED={float(selected['lambda']):.6f} SCORE={float(selected['score']):.6f}")
    return 0


def evaluate_lambda_records(records: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    """Apply the exact evaluator shared by selection and final test."""
    return evaluate_records(records, method="sabra")["macro"]


if __name__ == "__main__":
    raise SystemExit(main())
