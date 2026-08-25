"""Lossless fold-local cache for invariant frozen P26 training tensors."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from tools.sabra.data import safe_data_path


CACHE_SCHEMA = "P27_EXACT_FOLD_CACHE_V1"
TENSOR_SHAPES: dict[str, tuple[int, ...]] = {
    "seg_features": (3, 1369, 768),
    "native_logits": (3, 1369, 2),
    "teacher_region": (1, 9, 9),
    "source_mask": (1, 518, 518),
}
TENSOR_DTYPE = torch.float32


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "class_name": str(row["class_name"]),
        "image_path": str(row["image_path"]),
        "label": int(row["label"]),
        "mask_path": str(row["mask_path"]) if row.get("mask_path") else None,
    }


def source_inventory_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    records = [_canonical_record(row) for row in rows]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def source_file_inventory_digest(rows: Iterable[Mapping[str, Any]], data_root: Path) -> str:
    """Hash every source image/mask byte with its role and relative path."""
    digest = hashlib.sha256()
    for row in rows:
        paths = [("image", str(row["image_path"]))]
        if row.get("mask_path"):
            paths.append(("mask", str(row["mask_path"])))
        for role, relative_path in paths:
            digest.update(f"{role}\0{relative_path}\0".encode())
            path = safe_data_path(data_root, relative_path)
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheProvenance:
    held_class: str
    source_classes: tuple[str, ...]
    source_inventory_sha256: str
    source_files_sha256: str
    p26_checkpoint_sha256: str
    clip_asset_sha256: str
    config_sha256: str
    protocol_sha256: str
    dataset_root: str


@dataclass(frozen=True)
class ValidatedRegionCache:
    root: Path
    manifest: Mapping[str, Any]


@contextlib.contextmanager
def preserve_rng_state() -> Iterator[None]:
    """Prevent deterministic cache construction from consuming training RNG."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)


class RegionCacheWriter:
    """Append fixed-shape FP32 samples and publish a complete manifest last."""

    def __init__(self, root: Path, provenance: CacheProvenance, rows: Iterable[Mapping[str, Any]]) -> None:
        self.root = root
        self.provenance = provenance
        self.records = [_canonical_record(row) for row in rows]
        if any(record["class_name"] == provenance.held_class for record in self.records):
            raise RuntimeError("held class cannot enter source cache generation")
        if tuple(dict.fromkeys(record["class_name"] for record in self.records)) != provenance.source_classes:
            raise RuntimeError("source class inventory does not match cache provenance")
        if source_inventory_digest(self.records) != provenance.source_inventory_sha256:
            raise RuntimeError("source inventory digest does not match cache provenance")
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise RuntimeError(f"cache directory must be empty: {root}")
        self._handles = {name: (root / f"{name}.bin").open("xb") for name in TENSOR_SHAPES}
        self._hashers = {name: hashlib.sha256() for name in TENSOR_SHAPES}
        self._count = 0
        self._closed = False
        self._write_manifest("INCOMPLETE")

    def _write_manifest(self, status: str) -> None:
        payload: dict[str, Any] = {
            "schema_version": CACHE_SCHEMA,
            "status": status,
            **asdict(self.provenance),
            "source_classes": list(self.provenance.source_classes),
            "record_count": len(self.records),
            "records_written": self._count,
            "records": self.records,
            "held_gt_reads": 0,
            "held_mask_reads": 0,
            "tensor_dtype": "float32",
            "tensor_shapes_per_sample": {key: list(value) for key, value in TENSOR_SHAPES.items()},
        }
        if status == "COMPLETE":
            payload["files"] = {
                name: {
                    "path": f"{name}.bin",
                    "bytes": (self.root / f"{name}.bin").stat().st_size,
                    "sha256": self._hashers[name].hexdigest(),
                }
                for name in TENSOR_SHAPES
            }
        temporary = self.root / "manifest.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.root / "manifest.json")

    def append(self, sample: Mapping[str, torch.Tensor]) -> None:
        if self._closed:
            raise RuntimeError("cannot append to finalized cache")
        if self._count >= len(self.records):
            raise RuntimeError("cache received more tensors than source records")
        for name, shape in TENSOR_SHAPES.items():
            tensor = sample.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != TENSOR_DTYPE or tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must be exact CPU float32 tensor with shape {shape}")
            if tensor.device.type != "cpu":
                raise ValueError(f"{name} must be on CPU before cache serialization")
            contiguous = tensor.detach().contiguous()
            view = memoryview(contiguous.numpy()).cast("B")
            self._handles[name].write(view)
            self._hashers[name].update(view)
        self._count += 1

    def finalize(self) -> Path:
        if self._count != len(self.records):
            raise RuntimeError(f"incomplete cache: wrote {self._count} of {len(self.records)} records")
        for handle in self._handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self._closed = True
        self._write_manifest("COMPLETE")
        return self.root / "manifest.json"

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            if not handle.closed:
                handle.close()


def validate_region_cache(
    root: Path,
    expected: CacheProvenance,
    source_rows: Iterable[Mapping[str, Any]],
    *,
    verify_checksums: bool = True,
) -> ValidatedRegionCache:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("incomplete cache: manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != CACHE_SCHEMA or manifest.get("status") != "COMPLETE":
        raise RuntimeError("incomplete cache: completion marker is absent")
    labels = {
        "held_class": "held class",
        "source_classes": "source classes",
        "source_inventory_sha256": "source inventory",
        "source_files_sha256": "source file inventory",
        "p26_checkpoint_sha256": "P26 checkpoint hash",
        "clip_asset_sha256": "CLIP asset hash",
        "config_sha256": "config hash",
        "protocol_sha256": "protocol hash",
        "dataset_root": "dataset root",
    }
    expected_values = asdict(expected)
    expected_values["source_classes"] = list(expected.source_classes)
    for field, label in labels.items():
        if manifest.get(field) != expected_values[field]:
            raise RuntimeError(f"cache {label} mismatch")
    rows = list(source_rows)
    if source_inventory_digest(rows) != expected.source_inventory_sha256:
        raise RuntimeError("cache source inventory mismatch")
    if manifest.get("records") != [_canonical_record(row) for row in rows]:
        raise RuntimeError("cache source inventory record mismatch")
    if manifest.get("held_gt_reads") != 0 or manifest.get("held_mask_reads") != 0:
        raise RuntimeError("cache LOCO held GT/mask firewall failed")
    count = len(rows)
    files = manifest.get("files", {})
    for name, shape in TENSOR_SHAPES.items():
        metadata = files.get(name, {})
        path = root / f"{name}.bin"
        expected_bytes = count * int(np.prod(shape)) * torch.tensor([], dtype=TENSOR_DTYPE).element_size()
        if not path.is_file() or path.stat().st_size != expected_bytes or metadata.get("bytes") != expected_bytes:
            raise RuntimeError(f"incomplete cache tensor file: {name}")
        if verify_checksums and _sha256(path) != metadata.get("sha256"):
            raise RuntimeError(f"cache tensor checksum mismatch: {name}")
    return ValidatedRegionCache(root=root, manifest=manifest)


class CachedRegionDataset(Dataset):
    """Memory-mapped exact cached samples; slicing does not preload a fold."""

    def __init__(self, cache: ValidatedRegionCache) -> None:
        self.cache = cache
        self.count = int(cache.manifest["record_count"])
        self._tensors = {
            name: torch.from_file(
                str(cache.root / f"{name}.bin"),
                shared=False,
                size=self.count * int(np.prod(shape)),
                dtype=TENSOR_DTYPE,
            ).reshape(self.count, *shape)
            for name, shape in TENSOR_SHAPES.items()
        }

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.cache.manifest["records"][index]
        return {
            **{name: tensor[index] for name, tensor in self._tensors.items()},
            "index": int(index),
            "class_name": record["class_name"],
            "image_path": record["image_path"],
        }
