#!/usr/bin/env python3
"""Bounded-memory exact compare evaluator for canonical final Medical data.

The scientific model and metric definitions remain in their canonical modules.
This workflow adapter changes only prediction storage and exact metric execution:
inference is persisted to a validated fixed-width cache, then float32 score ties
are grouped in bounded sorted runs and merged on disk.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the production scientific primitives and canonical dataset adapter.
from dataset.info import MEDICAL_EVAL_PATHS, dataset_domain  # noqa: E402
from evaluation.evaluator import image_score  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    binary_average_precision,
    binary_auroc,
    macro_metrics,
)
from model.phase2b_runtime import (  # noqa: E402
    forward_phase2b,
    load_json_config,
    load_phase2b_checkpoint,
    sha256_file,
)
from test import (  # noqa: E402
    _build_inference_dataset,
    _cuda_runtime_stats,
    _verify_selected_checkpoint,
)
from tools.sabra.artifacts import load_json, validate_sabra_freeze  # noqa: E402
from tools.sabra.pipeline import compare_forward  # noqa: E402


SCIENTIFIC_CODE_SHA = "4aa9b465ddeb072e9218b74982306d6324c62375"
EVALUATOR_VERSION = "canonical-medical-numpy-external-v1"
IMAGE_SIZE = 518
PIXELS_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE
ROLE = "FINAL_ZERO_SHOT"
MANIFEST_NAME = "INFERENCE_CACHE_COMPLETE.json"
IDENTITIES_NAME = "image_identities.jsonl"
RUN_DTYPE = np.dtype([("score", "<f4"), ("positive", "<u4"), ("negative", "<u4")], align=False)

CACHE_ARRAYS: dict[str, tuple[str, tuple[str, ...]]] = {
    "pixel_labels.npy": ("int8", ("images", "pixels")),
    "phase2b_pixel_scores.npy": ("float32", ("images", "pixels")),
    "sabra_pixel_scores.npy": ("float32", ("images", "pixels")),
    "image_labels.npy": ("int8", ("images",)),
    # test.py constructs these arrays from Python floats, hence float64 is the
    # representation-preserving cache dtype for the frozen image-score formula.
    "phase2b_image_scores.npy": ("float64", ("images",)),
    "sabra_image_scores.npy": ("float64", ("images",)),
}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return value if sys.platform == "darwin" else value * 1024


def _host_ram_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        return 8 * 1024**3


def resolve_memory_budget_bytes(explicit_gb: float | None = None) -> int:
    configured = explicit_gb
    if configured is None:
        raw = os.environ.get("MEDICAL_METRIC_MEMORY_GB", "").strip()
        configured = float(raw) if raw else None
    host_bytes = _host_ram_bytes()
    if configured is None:
        configured = 2.0 if host_bytes >= 16 * 1024**3 else max(0.5, host_bytes / 1024**3 / 8.0)
    if not np.isfinite(configured) or configured < 0.25:
        raise ValueError("Medical metric memory budget must be at least 0.25 GiB")
    budget = int(configured * 1024**3)
    if budget > host_bytes // 2:
        raise ValueError("Medical metric memory budget may not exceed half of host RAM")
    return budget


def _workflow_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _evaluator_sha() -> str:
    return sha256_file(Path(__file__).resolve())


def validate_medical_inputs(
    dataset: str,
    selection_path: Path,
    freeze_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, str, str]:
    """Run the canonical immutable-artifact guard before cache or inference."""
    if dataset not in tuple(MEDICAL_EVAL_PATHS):
        raise ValueError(f"dataset is not in the canonical final Medical set: {dataset}")
    if dataset_domain(dataset) != "Medical":
        raise ValueError(f"canonical dataset domain mismatch for {dataset}")
    selection = load_json(selection_path)
    if selection.get("status") != "FROZEN":
        raise ValueError("Phase2B selection must be FROZEN before Medical evaluation")
    checkpoint_path, checkpoint_sha = _verify_selected_checkpoint(selection)
    freeze = load_json(freeze_path)
    validate_sabra_freeze(freeze, checkpoint_sha256=checkpoint_sha)
    if freeze.get("provenance", {}).get("git_sha") != SCIENTIFIC_CODE_SHA:
        raise ValueError("SABRA freeze provenance SHA differs from scientific code SHA")
    if freeze.get("relational", {}).get("backend") != "fast":
        raise ValueError("Medical evaluation requires frozen fast relational backend")
    if freeze.get("medical_seen") is not False:
        raise ValueError("SABRA freeze is not Medical-clean")
    freeze_sha = sha256_file(freeze_path)
    return selection, freeze, checkpoint_path, checkpoint_sha, freeze_sha


def _shape_for(spec: tuple[str, ...], images: int, pixels: int) -> tuple[int, ...]:
    dimensions = {"images": int(images), "pixels": int(pixels)}
    return tuple(dimensions[name] for name in spec)


def _known_cache_paths(work_dir: Path) -> Iterable[Path]:
    yield work_dir / IDENTITIES_NAME
    yield work_dir / f".{IDENTITIES_NAME}.tmp"
    for name in CACHE_ARRAYS:
        yield work_dir / name
        yield work_dir / f".{name}.tmp"


def _prepare_incomplete_cache(work_dir: Path) -> None:
    marker = work_dir / MANIFEST_NAME
    if marker.exists():
        raise ValueError(f"refusing to replace an existing inference-cache marker: {marker}")
    work_dir.mkdir(parents=True, exist_ok=True)
    # Only evaluator-owned files without a completion marker are discarded.
    for path in _known_cache_paths(work_dir):
        if path.is_file() or path.is_symlink():
            path.unlink()


class InferenceCacheWriter:
    """Fixed-file, fixed-width cache writer with an atomic completion marker."""

    def __init__(
        self,
        work_dir: Path,
        *,
        dataset: str,
        data_root: Path,
        image_count: int,
        pixels_per_image: int,
        checkpoint_sha256: str,
        freeze_sha256: str,
        workflow_package_sha: str,
        evaluator_sha256: str,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        if image_count <= 0 or pixels_per_image <= 0 or image_size <= 0:
            raise ValueError("cache dimensions must be positive")
        if int(pixels_per_image) != int(image_size) * int(image_size):
            raise ValueError("cache pixels_per_image must equal image_size squared")
        self.work_dir = work_dir
        self.dataset = str(dataset)
        self.data_root = str(data_root.expanduser().resolve())
        self.image_count = int(image_count)
        self.pixels_per_image = int(pixels_per_image)
        self.image_size = int(image_size)
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.freeze_sha256 = str(freeze_sha256)
        self.workflow_package_sha = str(workflow_package_sha)
        self.evaluator_sha256 = str(evaluator_sha256)
        self.written = 0
        self.cache_write_seconds = 0.0
        self.identity_hasher = hashlib.sha256()
        self.class_ranges: list[dict[str, Any]] = []
        self._closed = False
        _prepare_incomplete_cache(work_dir)
        self._arrays: dict[str, np.memmap] = {}
        for name, (dtype, shape_spec) in CACHE_ARRAYS.items():
            temporary = work_dir / f".{name}.tmp"
            shape = _shape_for(shape_spec, self.image_count, self.pixels_per_image)
            self._arrays[name] = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
        self._identity_path = work_dir / f".{IDENTITIES_NAME}.tmp"
        self._identity_handle = self._identity_path.open("wb")

    def write_batch(
        self,
        *,
        class_names: Sequence[str],
        image_paths: Sequence[str],
        pixel_labels: np.ndarray,
        phase2b_pixel_scores: np.ndarray,
        sabra_pixel_scores: np.ndarray,
        image_labels: np.ndarray,
        phase2b_image_scores: np.ndarray,
        sabra_image_scores: np.ndarray,
    ) -> None:
        started = time.perf_counter()
        batch = len(class_names)
        if batch <= 0 or len(image_paths) != batch:
            raise ValueError("cache batch identities are inconsistent")
        stop = self.written + batch
        if stop > self.image_count:
            raise ValueError("cache writer received more images than declared")

        labels = np.asarray(pixel_labels, dtype=np.int8).reshape(batch, -1)
        native = np.asarray(phase2b_pixel_scores, dtype=np.float32).reshape(batch, -1)
        corrected = np.asarray(sabra_pixel_scores, dtype=np.float32).reshape(batch, -1)
        if labels.shape[1] != self.pixels_per_image or native.shape != labels.shape or corrected.shape != labels.shape:
            raise ValueError("prediction maps do not match the canonical 518x518 shape")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("pixel labels must be binary")
        if not np.isfinite(native).all() or not np.isfinite(corrected).all():
            raise ValueError("pixel predictions must be finite")

        image_labels_array = np.asarray(image_labels, dtype=np.int8).reshape(-1)
        native_images = np.asarray(phase2b_image_scores, dtype=np.float64).reshape(-1)
        corrected_images = np.asarray(sabra_image_scores, dtype=np.float64).reshape(-1)
        if image_labels_array.size != batch or native_images.size != batch or corrected_images.size != batch:
            raise ValueError("image prediction batch dimensions are inconsistent")
        if not np.isin(image_labels_array, (0, 1)).all():
            raise ValueError("image labels must be binary")
        if not np.isfinite(native_images).all() or not np.isfinite(corrected_images).all():
            raise ValueError("image predictions must be finite")

        target = slice(self.written, stop)
        self._arrays["pixel_labels.npy"][target] = labels
        self._arrays["phase2b_pixel_scores.npy"][target] = native
        self._arrays["sabra_pixel_scores.npy"][target] = corrected
        self._arrays["image_labels.npy"][target] = image_labels_array
        self._arrays["phase2b_image_scores.npy"][target] = native_images
        self._arrays["sabra_image_scores.npy"][target] = corrected_images

        for offset, (class_name, image_path) in enumerate(zip(class_names, image_paths)):
            index = self.written + offset
            identity = {
                "class_name": str(class_name),
                "image_path": str(image_path),
                "order": int(index),
            }
            encoded = (json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            self._identity_handle.write(encoded)
            self.identity_hasher.update(encoded)
            if not self.class_ranges or self.class_ranges[-1]["class_name"] != str(class_name):
                if any(row["class_name"] == str(class_name) for row in self.class_ranges):
                    raise ValueError("canonical cache requires each class to occupy one contiguous image range")
                self.class_ranges.append({"class_name": str(class_name), "start_image": index, "image_count": 1})
            else:
                self.class_ranges[-1]["image_count"] += 1
        self.written = stop
        self.cache_write_seconds += time.perf_counter() - started

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self._identity_handle.close()
        finally:
            self._arrays.clear()
            self._closed = True

    def complete(self) -> dict[str, Any]:
        completion_started = time.perf_counter()
        if self.written != self.image_count:
            raise ValueError(f"cache incomplete: wrote {self.written}/{self.image_count} images")
        for array in self._arrays.values():
            array.flush()
        self._identity_handle.flush()
        os.fsync(self._identity_handle.fileno())
        self._identity_handle.close()
        self._arrays.clear()
        gc.collect()

        for name in CACHE_ARRAYS:
            temporary = self.work_dir / f".{name}.tmp"
            _fsync_file(temporary)
            os.replace(temporary, self.work_dir / name)
        os.replace(self._identity_path, self.work_dir / IDENTITIES_NAME)

        cache_files: list[dict[str, Any]] = []
        for name, (dtype, shape_spec) in CACHE_ARRAYS.items():
            path = self.work_dir / name
            cache_files.append(
                {
                    "path": name,
                    "size_bytes": path.stat().st_size,
                    "dtype": dtype,
                    "shape": list(_shape_for(shape_spec, self.image_count, self.pixels_per_image)),
                }
            )
        identity_path = self.work_dir / IDENTITIES_NAME
        cache_files.append(
            {
                "path": IDENTITIES_NAME,
                "size_bytes": identity_path.stat().st_size,
                "sha256": self.identity_hasher.hexdigest(),
            }
        )
        manifest = {
            "dataset": self.dataset,
            "role": ROLE,
            "data_root": self.data_root,
            "image_count": self.image_count,
            "pixel_count": self.image_count * self.pixels_per_image,
            "image_size": self.image_size,
            "pixels_per_image": self.pixels_per_image,
            "pixel_stride": 1,
            "selected_checkpoint_sha256": self.checkpoint_sha256,
            "sabra_freeze_sha256": self.freeze_sha256,
            "scientific_code_sha": SCIENTIFIC_CODE_SHA,
            "workflow_package_sha": self.workflow_package_sha,
            "workflow_evaluator_version": EVALUATOR_VERSION,
            "workflow_evaluator_sha256": self.evaluator_sha256,
            "score_dtype": "float32",
            "image_score_dtype": "float64",
            "label_dtype": "int8",
            "image_order_sha256": self.identity_hasher.hexdigest(),
            "class_ranges": self.class_ranges,
            "cache_files": cache_files,
        }
        _atomic_json(self.work_dir / MANIFEST_NAME, manifest)
        self.cache_write_seconds += time.perf_counter() - completion_started
        self._closed = True
        return manifest


def validate_inference_cache(work_dir: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    marker = work_dir / MANIFEST_NAME
    if not marker.is_file():
        raise FileNotFoundError(marker)
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    expected_image_size = int(expected.get("image_size", IMAGE_SIZE))
    expected_pixels_per_image = int(expected.get("pixels_per_image", expected_image_size * expected_image_size))
    required_matches = {
        "dataset": expected["dataset"],
        "role": ROLE,
        "data_root": str(Path(str(expected["data_root"])).expanduser().resolve()),
        "image_size": expected_image_size,
        "pixels_per_image": expected_pixels_per_image,
        "pixel_stride": 1,
        "selected_checkpoint_sha256": expected["selected_checkpoint_sha256"],
        "sabra_freeze_sha256": expected["sabra_freeze_sha256"],
        "scientific_code_sha": SCIENTIFIC_CODE_SHA,
        "workflow_package_sha": expected["workflow_package_sha"],
        "workflow_evaluator_version": EVALUATOR_VERSION,
        "workflow_evaluator_sha256": expected["workflow_evaluator_sha256"],
        "score_dtype": "float32",
        "image_score_dtype": "float64",
        "label_dtype": "int8",
    }
    for key, value in required_matches.items():
        if manifest.get(key) != value:
            raise ValueError(f"inference cache mismatch for {key}: expected {value!r}, got {manifest.get(key)!r}")
    images = int(manifest.get("image_count", 0))
    pixels = int(manifest.get("pixels_per_image", 0))
    if images <= 0 or int(manifest.get("pixel_count", -1)) != images * pixels:
        raise ValueError("inference cache image/pixel counts are invalid")

    entries = manifest.get("cache_files")
    if not isinstance(entries, list):
        raise ValueError("inference cache file manifest is missing")
    by_name = {str(entry.get("path")): entry for entry in entries if isinstance(entry, Mapping)}
    expected_names = {*CACHE_ARRAYS, IDENTITIES_NAME}
    if set(by_name) != expected_names:
        raise ValueError("inference cache file list is incomplete or contains unexpected files")
    for name, (dtype, shape_spec) in CACHE_ARRAYS.items():
        path = work_dir / name
        entry = by_name[name]
        if not path.is_file() or path.stat().st_size != int(entry.get("size_bytes", -1)):
            raise ValueError(f"inference cache file is missing or truncated: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = _shape_for(shape_spec, images, pixels)
        if array.dtype != np.dtype(dtype) or array.shape != expected_shape:
            raise ValueError(f"inference cache array contract mismatch: {name}")
        if entry.get("dtype") != dtype or entry.get("shape") != list(expected_shape):
            raise ValueError(f"inference cache manifest array contract mismatch: {name}")

    identity_path = work_dir / IDENTITIES_NAME
    identity_entry = by_name[IDENTITIES_NAME]
    if not identity_path.is_file() or identity_path.stat().st_size != int(identity_entry.get("size_bytes", -1)):
        raise ValueError("inference cache identity file is missing or truncated")
    identity_digest = sha256_file(identity_path)
    if identity_digest != identity_entry.get("sha256") or identity_digest != manifest.get("image_order_sha256"):
        raise ValueError("inference cache identity hash mismatch")
    identity_count = 0
    with identity_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("order", -1)) != identity_count:
                raise ValueError("inference cache identity order mismatch")
            identity_count += 1
    if identity_count != images:
        raise ValueError("inference cache identity count mismatch")

    ranges = manifest.get("class_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("inference cache class ranges are missing")
    cursor = 0
    seen: set[str] = set()
    for row in ranges:
        class_name = str(row.get("class_name", ""))
        start = int(row.get("start_image", -1))
        count = int(row.get("image_count", 0))
        if not class_name or class_name in seen or start != cursor or count <= 0:
            raise ValueError("inference cache class ranges are invalid")
        seen.add(class_name)
        cursor += count
    if cursor != images:
        raise ValueError("inference cache class ranges do not cover every image")
    return manifest


def _cache_arrays(work_dir: Path) -> dict[str, np.memmap]:
    return {name: np.load(work_dir / name, mmap_mode="r", allow_pickle=False) for name in CACHE_ARRAYS}


class TempUsageTracker:
    def __init__(self) -> None:
        self.current_bytes = 0
        self.peak_bytes = 0

    def add(self, path: Path) -> None:
        self.current_bytes += path.stat().st_size
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)

    def remove(self, path: Path) -> None:
        if path.exists():
            size = path.stat().st_size
            path.unlink()
            self.current_bytes = max(0, self.current_bytes - size)


def _write_grouped_run(
    path: Path,
    scores: np.ndarray,
    labels: np.ndarray,
) -> int:
    count = int(scores.size)
    order = np.argsort(scores, kind="quicksort")
    sorted_scores = np.asarray(scores[order], dtype=np.float32)
    sorted_labels = np.asarray(labels[order], dtype=np.int8)
    del order
    changes = np.empty(count, dtype=np.bool_)
    changes[0] = True
    np.not_equal(sorted_scores[1:], sorted_scores[:-1], out=changes[1:])
    starts = np.flatnonzero(changes)
    del changes
    grouped = np.empty(starts.size, dtype=RUN_DTYPE)
    group_block = 1_000_000
    for begin in range(0, starts.size, group_block):
        finish = min(begin + group_block, starts.size)
        indices = starts[begin:finish]
        following = starts[begin + 1 : finish + 1]
        if finish == starts.size:
            following = np.concatenate((following, np.asarray([count], dtype=np.int64)))
        if finish < starts.size:
            reduction_indices = starts[begin : finish + 1]
            positives = np.add.reduceat(sorted_labels, reduction_indices, dtype=np.uint64)[:-1]
        else:
            positives = np.add.reduceat(sorted_labels, indices, dtype=np.uint64)
        totals = following - indices
        if positives.size != indices.size or totals.size != indices.size:
            raise RuntimeError("internal grouped-run boundary mismatch")
        negatives = totals.astype(np.uint64, copy=False) - positives
        if int(positives.max(initial=0)) > np.iinfo(np.uint32).max or int(negatives.max(initial=0)) > np.iinfo(np.uint32).max:
            raise OverflowError("group count exceeds uint32 cache contract")
        grouped["score"][begin:finish] = sorted_scores[indices]
        grouped["positive"][begin:finish] = positives.astype(np.uint32)
        grouped["negative"][begin:finish] = negatives.astype(np.uint32)
    with path.open("wb") as handle:
        grouped.tofile(handle)
    return int(grouped.size)


def _read_run_block(path: Path, count: int, start: int, block: int) -> np.ndarray:
    stop = min(start + block, count)
    mapped = np.memmap(path, mode="r", dtype=RUN_DTYPE, shape=(count,))
    output = np.array(mapped[start:stop], copy=True)
    del mapped
    return output


def _coalesce_group_records(records: np.ndarray) -> np.ndarray:
    if records.size == 0:
        return records
    order = np.argsort(records["score"], kind="quicksort")
    sorted_records = records[order]
    scores = sorted_records["score"]
    changes = np.empty(scores.size, dtype=np.bool_)
    changes[0] = True
    np.not_equal(scores[1:], scores[:-1], out=changes[1:])
    starts = np.flatnonzero(changes)
    output = np.empty(starts.size, dtype=RUN_DTYPE)
    output["score"] = scores[starts]
    positive = np.add.reduceat(sorted_records["positive"], starts, dtype=np.uint64)
    negative = np.add.reduceat(sorted_records["negative"], starts, dtype=np.uint64)
    if int(positive.max(initial=0)) > np.iinfo(np.uint32).max or int(negative.max(initial=0)) > np.iinfo(np.uint32).max:
        raise OverflowError("merged group count exceeds uint32 cache contract")
    output["positive"] = positive.astype(np.uint32)
    output["negative"] = negative.astype(np.uint32)
    return output


def _merge_two_runs(
    left: tuple[Path, int],
    right: tuple[Path, int],
    output_path: Path,
    *,
    block_groups: int,
) -> int:
    left_path, left_count = left
    right_path, right_count = right
    left_position = 0
    right_position = 0
    left_buffer = np.empty(0, dtype=RUN_DTYPE)
    right_buffer = np.empty(0, dtype=RUN_DTYPE)
    written = 0
    with output_path.open("wb") as output:
        while True:
            if left_buffer.size == 0 and left_position < left_count:
                left_buffer = _read_run_block(left_path, left_count, left_position, block_groups)
                left_position += left_buffer.size
            if right_buffer.size == 0 and right_position < right_count:
                right_buffer = _read_run_block(right_path, right_count, right_position, block_groups)
                right_position += right_buffer.size

            if left_buffer.size == 0:
                if right_buffer.size:
                    right_buffer.tofile(output)
                    written += right_buffer.size
                while right_position < right_count:
                    block = _read_run_block(right_path, right_count, right_position, block_groups)
                    right_position += block.size
                    block.tofile(output)
                    written += block.size
                break
            if right_buffer.size == 0:
                left_buffer.tofile(output)
                written += left_buffer.size
                while left_position < left_count:
                    block = _read_run_block(left_path, left_count, left_position, block_groups)
                    left_position += block.size
                    block.tofile(output)
                    written += block.size
                break

            if left_buffer["score"][-1] < right_buffer["score"][0]:
                left_buffer.tofile(output)
                written += left_buffer.size
                left_buffer = np.empty(0, dtype=RUN_DTYPE)
                continue
            if right_buffer["score"][-1] < left_buffer["score"][0]:
                right_buffer.tofile(output)
                written += right_buffer.size
                right_buffer = np.empty(0, dtype=RUN_DTYPE)
                continue

            cutoff = min(float(left_buffer["score"][-1]), float(right_buffer["score"][-1]))
            combined = _coalesce_group_records(np.concatenate((left_buffer, right_buffer)))
            emit_stop = int(np.searchsorted(combined["score"], np.float32(cutoff), side="right"))
            combined[:emit_stop].tofile(output)
            written += emit_stop
            left_stop = int(np.searchsorted(left_buffer["score"], np.float32(cutoff), side="right"))
            right_stop = int(np.searchsorted(right_buffer["score"], np.float32(cutoff), side="right"))
            left_buffer = np.array(left_buffer[left_stop:], copy=True)
            right_buffer = np.array(right_buffer[right_stop:], copy=True)
    return int(written)


def _scan_grouped_metrics(path: Path, count: int, block_groups: int) -> tuple[float, float, int, int]:
    records = np.memmap(path, mode="r", dtype=RUN_DTYPE, shape=(count,))
    positive_total = int(records["positive"].sum(dtype=np.uint64))
    negative_total = int(records["negative"].sum(dtype=np.uint64))
    if positive_total == 0 or negative_total == 0:
        raise ValueError("binary metric requires both positive and negative labels")

    base_rank = 0.0
    positive_rank_sum = 0.0
    for start in range(0, count, block_groups):
        block = records[start : min(start + block_groups, count)]
        positives = block["positive"].astype(np.float64)
        totals = positives + block["negative"].astype(np.float64)
        ends = base_rank + np.cumsum(totals, dtype=np.float64)
        starts = ends - totals + 1.0
        positive_rank_sum += float(np.sum(positives * ((starts + ends) / 2.0), dtype=np.float64))
        base_rank = float(ends[-1])
    positives_float = float(positive_total)
    negatives_float = float(negative_total)
    auroc = (positive_rank_sum - positives_float * (positives_float + 1.0) / 2.0) / (
        positives_float * negatives_float
    )

    true_positive = 0.0
    false_positive = 0.0
    previous_recall = 0.0
    average_precision = 0.0
    stop = count
    while stop > 0:
        start = max(0, stop - block_groups)
        block = records[start:stop][::-1]
        positives = block["positive"].astype(np.float64)
        negatives = block["negative"].astype(np.float64)
        cumulative_positive = true_positive + np.cumsum(positives, dtype=np.float64)
        cumulative_negative = false_positive + np.cumsum(negatives, dtype=np.float64)
        recalls = cumulative_positive / positives_float
        deltas = np.empty(recalls.size, dtype=np.float64)
        deltas[0] = recalls[0] - previous_recall
        deltas[1:] = recalls[1:] - recalls[:-1]
        precision = cumulative_positive / np.maximum(cumulative_positive + cumulative_negative, 1.0)
        average_precision += float(np.sum(deltas * precision, dtype=np.float64))
        true_positive = float(cumulative_positive[-1])
        false_positive = float(cumulative_negative[-1])
        previous_recall = float(recalls[-1])
        stop = start
    del records
    return float(auroc), float(average_precision), positive_total, negative_total


def external_binary_metrics(
    score_values: np.ndarray,
    label_values: np.ndarray,
    *,
    start: int,
    stop: int,
    temp_dir: Path,
    prefix: str,
    memory_budget_bytes: int,
    tracker: TempUsageTracker,
    chunk_elements: int | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """Compute canonical exact AUROC/AP with bounded NumPy external runs."""
    if stop <= start:
        raise ValueError("scores and labels must be non-empty arrays of equal shape")
    temp_dir.mkdir(parents=True, exist_ok=True)
    if chunk_elements is None:
        chunk_elements = max(65_536, int(memory_budget_bytes // 56))
    chunk_elements = max(1, int(chunk_elements))
    block_groups = max(65_536, min(2_000_000, int(memory_budget_bytes // (RUN_DTYPE.itemsize * 16))))
    runs: list[tuple[Path, int]] = []
    sort_started = time.perf_counter()
    positive_total = 0
    negative_total = 0
    run_index = 0
    for offset in range(start, stop, chunk_elements):
        finish = min(offset + chunk_elements, stop)
        scores = np.asarray(score_values[offset:finish], dtype=np.float32)
        labels = np.asarray(label_values[offset:finish], dtype=np.int8)
        if scores.shape != labels.shape or scores.size == 0:
            raise ValueError("scores and labels must be non-empty arrays of equal shape")
        if not np.isfinite(scores).all():
            raise ValueError("metric scores must be finite")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("binary labels must be 0/1")
        positives = int(labels.sum(dtype=np.int64))
        positive_total += positives
        negative_total += int(labels.size) - positives
        path = temp_dir / f"{prefix}.run{run_index:04d}.bin"
        grouped_count = _write_grouped_run(path, scores, labels)
        tracker.add(path)
        runs.append((path, grouped_count))
        run_index += 1
    if positive_total == 0 or negative_total == 0:
        raise ValueError("binary metric requires both positive and negative labels")

    level = 0
    while len(runs) > 1:
        next_runs: list[tuple[Path, int]] = []
        for pair in range(0, len(runs), 2):
            if pair + 1 == len(runs):
                next_runs.append(runs[pair])
                continue
            output = temp_dir / f"{prefix}.merge{level:02d}.{pair // 2:04d}.bin"
            merged_count = _merge_two_runs(runs[pair], runs[pair + 1], output, block_groups=block_groups)
            tracker.add(output)
            tracker.remove(runs[pair][0])
            tracker.remove(runs[pair + 1][0])
            next_runs.append((output, merged_count))
        runs = next_runs
        level += 1
    group_sort_seconds = time.perf_counter() - sort_started

    scan_started = time.perf_counter()
    final_path, final_count = runs[0]
    auroc, average_precision, scanned_positive, scanned_negative = _scan_grouped_metrics(
        final_path, final_count, block_groups
    )
    scan_seconds = time.perf_counter() - scan_started
    if scanned_positive != positive_total or scanned_negative != negative_total:
        raise RuntimeError("external grouped metric counts do not match streamed labels")
    tracker.remove(final_path)
    return auroc, average_precision, {
        "raw_values": int(stop - start),
        "initial_runs": int(run_index),
        "final_score_groups": int(final_count),
        "chunk_elements": int(chunk_elements),
        "group_sort_seconds": float(group_sort_seconds),
        "metric_scan_seconds": float(scan_seconds),
    }


def evaluate_inference_cache(
    work_dir: Path,
    manifest: Mapping[str, Any],
    *,
    memory_budget_bytes: int,
    chunk_elements: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prepare_started = time.perf_counter()
    arrays = _cache_arrays(work_dir)
    pixel_labels = arrays["pixel_labels.npy"].reshape(-1)
    metric_temp = work_dir / "metric_tmp"
    if metric_temp.exists():
        shutil.rmtree(metric_temp)
    metric_temp.mkdir(parents=True)
    tracker = TempUsageTracker()
    prepare_seconds = time.perf_counter() - prepare_started
    phase2b: dict[str, dict[str, float | None]] = {}
    sabra: dict[str, dict[str, float | None]] = {}
    sort_seconds = 0.0
    scan_seconds = 0.0
    run_metadata: list[dict[str, Any]] = []
    try:
        for class_index, row in enumerate(manifest["class_ranges"]):
            class_name = str(row["class_name"])
            image_start = int(row["start_image"])
            image_stop = image_start + int(row["image_count"])
            pixel_start = image_start * int(manifest["pixels_per_image"])
            pixel_stop = image_stop * int(manifest["pixels_per_image"])
            image_labels = np.asarray(arrays["image_labels.npy"][image_start:image_stop], dtype=np.int8)
            for method, target in (("phase2b", phase2b), ("sabra", sabra)):
                score_name = f"{method}_pixel_scores.npy"
                pixel_scores = arrays[score_name].reshape(-1)
                pixel_auroc, pixel_ap, metadata = external_binary_metrics(
                    pixel_scores,
                    pixel_labels,
                    start=pixel_start,
                    stop=pixel_stop,
                    temp_dir=metric_temp,
                    prefix=f"c{class_index:03d}.{method}",
                    memory_budget_bytes=memory_budget_bytes,
                    tracker=tracker,
                    chunk_elements=chunk_elements,
                )
                sort_seconds += float(metadata["group_sort_seconds"])
                scan_seconds += float(metadata["metric_scan_seconds"])
                run_metadata.append({"class_name": class_name, "method": method, **metadata})
                image_scores = np.asarray(
                    arrays[f"{method}_image_scores.npy"][image_start:image_stop], dtype=np.float64
                )
                target[class_name] = {
                    "pixel_auroc": pixel_auroc,
                    "pixel_ap": pixel_ap,
                    "image_auroc": binary_auroc(image_scores, image_labels, allow_undefined=True),
                    "image_ap": binary_average_precision(image_scores, image_labels, allow_undefined=True),
                }
        phase2b_macro = macro_metrics(phase2b)
        sabra_macro = macro_metrics(sabra)
        delta = {
            key: (
                None
                if phase2b_macro[key] is None or sabra_macro[key] is None
                else float(sabra_macro[key]) - float(phase2b_macro[key])
            )
            for key in phase2b_macro
        }
        result = {
            "phase2b": phase2b,
            "phase2b_macro": phase2b_macro,
            "phase2b_metrics": phase2b_macro,
            "sabra": sabra,
            "sabra_macro": sabra_macro,
            "sabra_metrics": sabra_macro,
            "delta": delta,
        }
        runtime = {
            "metric_prepare_seconds": float(prepare_seconds),
            "external_group_sort_seconds": float(sort_seconds),
            "metric_scan_seconds": float(scan_seconds),
            "temp_disk_bytes": int(tracker.peak_bytes),
            "metric_runs": run_metadata,
        }
        return result, runtime
    finally:
        arrays.clear()
        gc.collect()
        if metric_temp.exists():
            shutil.rmtree(metric_temp)


def infer_to_cache(
    *,
    dataset: str,
    data_root: Path,
    selection: Mapping[str, Any],
    freeze: Mapping[str, Any],
    checkpoint_sha256: str,
    freeze_sha256: str,
    config_path: Path,
    clip_asset: Path,
    device: torch.device,
    work_dir: Path,
    workflow_package_sha: str,
    evaluator_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    config = load_json_config(config_path)
    checkpoint_path, verified_sha = _verify_selected_checkpoint(selection)
    if verified_sha != checkpoint_sha256:
        raise ValueError("selected checkpoint changed after Medical guard")
    model = load_phase2b_checkpoint(checkpoint_path, config, clip_asset, device)
    model.eval()
    domain = dataset_domain(dataset)
    inference_dataset = _build_inference_dataset(dataset, data_root)
    writer = InferenceCacheWriter(
        work_dir,
        dataset=dataset,
        data_root=data_root,
        image_count=len(inference_dataset),
        pixels_per_image=PIXELS_PER_IMAGE,
        checkpoint_sha256=checkpoint_sha256,
        freeze_sha256=freeze_sha256,
        workflow_package_sha=workflow_package_sha,
        evaluator_sha256=evaluator_sha256,
    )
    loader = DataLoader(
        inference_dataset,
        batch_size=6,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    progress = tqdm(total=len(inference_dataset), desc=f"{dataset} compare external-cache", unit="img")
    try:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=device.type == "cuda").float()
            class_names = [str(value) for value in batch["class_name"]]
            forward = forward_phase2b(
                model,
                image,
                class_names,
                device,
                config,
                domain=domain,
                require_grad=False,
                dataset_name=dataset,
            )
            # One shared Phase2B forward feeds both native and SABRA paths.
            native_pixels = (
                forward.deployed_segmentation_probability.detach().cpu().numpy().astype(np.float32)
            )
            native_cls = forward.classification_probability.detach().cpu().numpy().astype(np.float32)
            with torch.enable_grad():
                corrected_probability = compare_forward(
                    forward, freeze, domain=domain
                )["corrected_probability"]
            corrected_pixels = corrected_probability.detach().cpu().numpy().astype(np.float32)
            masks = batch["mask"].detach().cpu().numpy().astype(np.int8)
            labels = batch["label"].detach().cpu().numpy().reshape(-1).astype(np.int8)
            paths = [str(value) for value in batch["image_path"]]
            native_image_scores = np.asarray(
                [
                    image_score(float(native_cls[index]), float(native_pixels[index].max()), domain)
                    for index in range(len(class_names))
                ],
                dtype=np.float64,
            )
            sabra_image_scores = np.asarray(
                [
                    image_score(float(native_cls[index]), float(corrected_pixels[index].max()), domain)
                    for index in range(len(class_names))
                ],
                dtype=np.float64,
            )
            writer.write_batch(
                class_names=class_names,
                image_paths=paths,
                pixel_labels=masks,
                phase2b_pixel_scores=native_pixels,
                sabra_pixel_scores=corrected_pixels,
                image_labels=labels,
                phase2b_image_scores=native_image_scores,
                sabra_image_scores=sabra_image_scores,
            )
            progress.update(len(class_names))
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = writer.written / elapsed
            remaining = max(len(inference_dataset) - writer.written, 0)
            progress.set_postfix({"img/s": f"{rate:.2f}", "eta": f"{remaining / max(rate, 1e-9):.0f}s"})
        manifest = writer.complete()
    except BaseException:
        writer.abort()
        raise
    finally:
        progress.close()
        del loader
        del inference_dataset
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    inference_seconds = time.perf_counter() - started
    runtime = _cuda_runtime_stats(device, started, int(manifest["image_count"]))
    runtime.update(
        {
            "inference_seconds": float(inference_seconds),
            "cache_write_seconds": float(writer.cache_write_seconds),
        }
    )
    return manifest, runtime


def _result_is_complete(
    output_dir: Path,
    *,
    dataset: str,
    checkpoint_sha256: str,
    freeze_sha256: str,
) -> bool:
    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "per_class_metrics.csv"
    if not metrics_path.exists() and not csv_path.exists():
        return False
    if not metrics_path.is_file() or not csv_path.is_file() or csv_path.stat().st_size == 0:
        return False
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset or payload.get("role") != ROLE:
        raise ValueError(f"existing Medical result identity mismatch: {output_dir}")
    if payload.get("phase2b_checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"existing Medical result checkpoint mismatch: {output_dir}")
    if payload.get("sabra_freeze_sha256") != freeze_sha256:
        raise ValueError(f"existing Medical result freeze mismatch: {output_dir}")
    for key in ("phase2b", "phase2b_macro", "phase2b_metrics", "sabra", "sabra_macro", "sabra_metrics", "delta", "runtime"):
        if not isinstance(payload.get(key), Mapping):
            raise ValueError(f"existing Medical result lacks {key}: {output_dir}")
    return True


def _write_outputs_atomic(output_dir: Path, result: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for class_name in sorted(result["phase2b"]):
        rows.append(
            {
                "class_name": class_name,
                **{f"phase2b_{key}": value for key, value in result["phase2b"][class_name].items()},
                **{f"sabra_{key}": value for key, value in result["sabra"][class_name].items()},
            }
        )
    if not rows:
        raise ValueError("Medical compare result contains no per-class rows")

    csv_path = output_dir / "per_class_metrics.csv"
    csv_temp = output_dir / f".{csv_path.name}.{os.getpid()}.tmp"
    with csv_temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())

    metrics_path = output_dir / "metrics.json"
    metrics_temp = output_dir / f".{metrics_path.name}.{os.getpid()}.tmp"
    with metrics_temp.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    # metrics.json is the stage marker, so commit the CSV first and marker last.
    os.replace(csv_temp, csv_path)
    os.replace(metrics_temp, metrics_path)


def run_compare(args: argparse.Namespace) -> int:
    if args.batch_size != 6 or args.num_workers != 4 or args.prefetch_factor != 2:
        raise ValueError("canonical Medical inference requires batch=6 workers=4 prefetch=2")
    if args.pin_memory is not True:
        raise ValueError("canonical Medical inference requires pin_memory=true")
    if args.pixel_stride != 1 or args.metric_mode != "exact":
        raise ValueError("canonical Medical metrics require exact mode and pixel_stride=1")
    if args.device != "cuda":
        raise ValueError("canonical final Medical inference requires CUDA")

    selection, freeze, _, checkpoint_sha, freeze_sha = validate_medical_inputs(
        args.dataset, args.phase2b_selection, args.sabra_freeze
    )
    print(f"MEDICAL_GUARD=PASS checkpoint_sha256={checkpoint_sha} backend=fast medical_seen=false")
    if args.guard_only:
        return 0
    if _result_is_complete(
        args.output_dir,
        dataset=args.dataset,
        checkpoint_sha256=checkpoint_sha,
        freeze_sha256=freeze_sha,
    ):
        print(f"MEDICAL_DATASET_STATUS=COMPLETE_REUSED dataset={args.dataset}")
        return 0

    workflow_sha = _workflow_sha()
    evaluator_sha = _evaluator_sha()
    expected_cache = {
        "dataset": args.dataset,
        "data_root": args.data_root,
        "selected_checkpoint_sha256": checkpoint_sha,
        "sabra_freeze_sha256": freeze_sha,
        "workflow_package_sha": workflow_sha,
        "workflow_evaluator_sha256": evaluator_sha,
    }
    total_started = time.perf_counter()
    cache_reused = False
    inference_runtime: dict[str, Any] = {"inference_seconds": 0.0, "cache_write_seconds": 0.0}
    marker = args.work_dir / MANIFEST_NAME
    if marker.exists():
        if not args.reuse_inference_cache and not args.metrics_only:
            raise ValueError("completed inference cache exists; pass --reuse-inference-cache")
        manifest = validate_inference_cache(args.work_dir, expected_cache)
        cache_reused = True
        print(f"INFERENCE_CACHE=VALID_REUSED dataset={args.dataset} images={manifest['image_count']}")
    else:
        if args.metrics_only:
            raise FileNotFoundError(f"metrics-only retry requires a completed inference cache: {marker}")
        manifest, inference_runtime = infer_to_cache(
            dataset=args.dataset,
            data_root=args.data_root,
            selection=selection,
            freeze=freeze,
            checkpoint_sha256=checkpoint_sha,
            freeze_sha256=freeze_sha,
            config_path=args.config,
            clip_asset=args.clip_asset,
            device=torch.device(args.device),
            work_dir=args.work_dir,
            workflow_package_sha=workflow_sha,
            evaluator_sha256=evaluator_sha,
        )
        manifest = validate_inference_cache(args.work_dir, expected_cache)
        print(f"INFERENCE_CACHE=COMPLETE dataset={args.dataset} images={manifest['image_count']}")

    memory_budget = resolve_memory_budget_bytes(args.memory_gb)
    metric_result, metric_runtime = evaluate_inference_cache(
        args.work_dir,
        manifest,
        memory_budget_bytes=memory_budget,
        chunk_elements=args.metric_chunk_elements,
    )
    cache_bytes = sum(int(row["size_bytes"]) for row in manifest["cache_files"])
    runtime = {
        "metric_mode": "exact",
        "pixel_stride": 1,
        "external_memory": True,
        "external_backend": "numpy_external",
        "external_metric_version": EVALUATOR_VERSION,
        "memory_budget_bytes": int(memory_budget),
        "inference_cache_reused": bool(cache_reused),
        "inference_cache_bytes": int(cache_bytes),
        "workflow_package_sha": workflow_sha,
        "workflow_evaluator_sha256": evaluator_sha,
        **inference_runtime,
        **metric_runtime,
        "peak_host_rss_bytes": _peak_rss_bytes(),
        "total_seconds": float(time.perf_counter() - total_started),
    }
    result = {
        **metric_result,
        "dataset": args.dataset,
        "role": ROLE,
        "phase2b_checkpoint_sha256": checkpoint_sha,
        "sabra_freeze_sha256": freeze_sha,
        "runtime": runtime,
    }
    _write_outputs_atomic(args.output_dir, result)
    print(f"MEDICAL_DATASET_STATUS=COMPLETE dataset={args.dataset} cache_reused={str(cache_reused).lower()}")
    return 0


def exact_metrics_from_arrays(
    scores: np.ndarray,
    labels: np.ndarray,
    temp_dir: Path,
    *,
    chunk_elements: int,
) -> tuple[float, float, dict[str, Any]]:
    """Bounded test adapter over the production external metric engine."""
    score_array = np.asarray(scores, dtype=np.float32).reshape(-1)
    label_array = np.asarray(labels, dtype=np.int8).reshape(-1)
    tracker = TempUsageTracker()
    result = external_binary_metrics(
        score_array,
        label_array,
        start=0,
        stop=score_array.size,
        temp_dir=temp_dir,
        prefix="fixture",
        memory_budget_bytes=256 * 1024**2,
        tracker=tracker,
        chunk_elements=chunk_elements,
    )
    result[2]["temp_disk_bytes"] = tracker.peak_bytes
    return result


def run_synthetic_stress(directory: Path, pixels: int, chunk_elements: int) -> dict[str, Any]:
    if pixels < 2 or chunk_elements < 1:
        raise ValueError("synthetic stress dimensions are invalid")
    directory.mkdir(parents=True, exist_ok=True)
    score_path = directory / "synthetic_scores.npy"
    label_path = directory / "synthetic_labels.npy"
    scores = np.lib.format.open_memmap(score_path, mode="w+", dtype="float32", shape=(pixels,))
    labels = np.lib.format.open_memmap(label_path, mode="w+", dtype="int8", shape=(pixels,))
    rng = np.random.default_rng(20260822)
    for start in range(0, pixels, chunk_elements):
        stop = min(start + chunk_elements, pixels)
        # Deliberate ties and heavy imbalance exercise grouped external runs.
        scores[start:stop] = rng.integers(0, 65_536, size=stop - start, dtype=np.uint32).astype(np.float32) / np.float32(65_535.0)
        labels[start:stop] = (rng.random(stop - start) < 0.01).astype(np.int8)
    labels[0] = 0
    labels[1] = 1
    scores.flush()
    labels.flush()
    started = time.perf_counter()
    tracker = TempUsageTracker()
    auroc, average_precision, metadata = external_binary_metrics(
        scores,
        labels,
        start=0,
        stop=pixels,
        temp_dir=directory / "metric_tmp",
        prefix="stress",
        memory_budget_bytes=256 * 1024**2,
        tracker=tracker,
        chunk_elements=chunk_elements,
    )
    elapsed = time.perf_counter() - started
    result = {
        "pixels": int(pixels),
        "chunk_size": int(chunk_elements),
        "peak_rss_bytes": _peak_rss_bytes(),
        "temp_disk_bytes": int(tracker.peak_bytes),
        "raw_disk_bytes": int(score_path.stat().st_size + label_path.stat().st_size),
        "elapsed_seconds": float(elapsed),
        "auroc": auroc,
        "ap": average_precision,
        **metadata,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--phase2b-selection", type=Path)
    parser.add_argument("--sabra-freeze", type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/phase2b_canonical_v1.json")
    parser.add_argument("--clip-asset", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metric-mode", choices=["exact"], default="exact")
    parser.add_argument("--pixel-stride", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--reuse-inference-cache", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--guard-only", action="store_true")
    parser.add_argument("--memory-gb", type=float)
    parser.add_argument("--metric-chunk-elements", type=int)
    parser.add_argument("--synthetic-stress", type=Path)
    parser.add_argument("--synthetic-pixels", type=int, default=5_000_000)
    parser.add_argument("--synthetic-chunk-elements", type=int, default=250_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.synthetic_stress is not None:
        result = run_synthetic_stress(
            args.synthetic_stress,
            pixels=args.synthetic_pixels,
            chunk_elements=args.synthetic_chunk_elements,
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    required = {
        "dataset": args.dataset,
        "data_root": args.data_root,
        "phase2b_selection": args.phase2b_selection,
        "sabra_freeze": args.sabra_freeze,
        "clip_asset": args.clip_asset,
        "output_dir": args.output_dir,
        "work_dir": args.work_dir,
    }
    missing = [name.replace("_", "-") for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing required evaluator arguments: {', '.join(missing)}")
    return run_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
