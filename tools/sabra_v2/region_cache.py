"""Exact, directly addressable P27 caches with fail-closed provenance checks."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from tools.sabra_v2.p26_parent import (
    P26_CLIP_ASSET_SHA256,
    P26_PHASE2B_CHECKPOINT_SHA256,
    P26_RUNTIME_CONFIG_SHA256,
)


CACHE_IMPLEMENTATION_VERSION = "P27_CACHE_V1"
TIER_A_SCHEMA = "P27_TIER_A_FROZEN_FEATURES_V1"
TIER_B_SCHEMA = "P27_TIER_B_SOURCE_SUPERVISION_V1"
PARENT_EXECUTION_SHA = "1151373f2c4968268f52cdc3e538c7ebcef7b8f0"
SEGMENTATION_SHAPE = (3, 1369, 768)
NATIVE_LOGIT_SHAPE = (3, 1369, 2)
MASK_SHAPE = (1, 518, 518)
TEACHER_SHAPE = (9, 9)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def stable_sample_id(row: Mapping[str, Any]) -> str:
    class_name = str(row.get("class_name", ""))
    image_path = str(row.get("image_path", ""))
    if not class_name or not image_path:
        raise ValueError("cache records require class_name and image_path")
    return f"{class_name}:{image_path}"


@dataclass(frozen=True)
class CacheProvenance:
    scientific_execution_base_sha: str
    metadata_sha256: str
    p26_sha256: str = P26_PHASE2B_CHECKPOINT_SHA256
    clip_sha256: str = P26_CLIP_ASSET_SHA256
    config_sha256: str = P26_RUNTIME_CONFIG_SHA256
    parent_execution_sha: str = PARENT_EXECUTION_SHA
    cache_implementation_version: str = CACHE_IMPLEMENTATION_VERSION

    def as_dict(self) -> dict[str, str]:
        return {
            "cache_implementation_version": self.cache_implementation_version,
            "parent_execution_sha": self.parent_execution_sha,
            "scientific_execution_base_sha": self.scientific_execution_base_sha,
            "metadata_sha256": self.metadata_sha256,
            "p26_sha256": self.p26_sha256,
            "clip_sha256": self.clip_sha256,
            "config_sha256": self.config_sha256,
        }


def _manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"incomplete cache shard: missing {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    if payload.get("completion_status") != "COMPLETE":
        raise RuntimeError(f"incomplete cache shard: {path}")
    return payload


def _require_provenance(manifest: Mapping[str, Any], expected: CacheProvenance) -> None:
    for key, value in expected.as_dict().items():
        if manifest.get(key) != value:
            raise RuntimeError(f"cache provenance mismatch for {key}: expected {value!r}, got {manifest.get(key)!r}")


def _validate_array(path: Path, spec: Mapping[str, Any], verify_hashes: bool) -> np.memmap:
    if not path.is_file():
        raise RuntimeError(f"incomplete cache shard: missing {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != list(spec["storage_shape"]) or str(array.dtype) != spec["dtype"]:
        raise RuntimeError(f"cache tensor contract mismatch for {path}")
    if int(path.stat().st_size) != int(spec["file_bytes"]):
        raise RuntimeError(f"cache tensor file-size mismatch for {path}")
    if verify_hashes and sha256_file(path) != spec["sha256"]:
        raise RuntimeError(f"cache tensor hash mismatch for {path}")
    return array


def validate_tier_a_shard(
    shard: Path,
    class_name: str,
    sample_ids: Sequence[str],
    provenance: CacheProvenance,
    *,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    manifest = _manifest(shard)
    if manifest.get("schema") != TIER_A_SCHEMA or manifest.get("tier") != "A":
        raise RuntimeError(f"wrong Tier-A cache schema: {shard}")
    if manifest.get("class") != class_name or manifest.get("sample_ids") != list(sample_ids):
        raise RuntimeError(f"Tier-A class or deterministic sample ordering mismatch: {shard}")
    if manifest.get("sample_count") != len(sample_ids) or manifest.get("contains_gt") is not False:
        raise RuntimeError(f"invalid GT-free Tier-A manifest: {shard}")
    _require_provenance(manifest, provenance)
    expected = {
        "seg_features": (SEGMENTATION_SHAPE, "float32"),
        "native_logits": (NATIVE_LOGIT_SHAPE, "float32"),
    }
    if manifest.get("tensor_names") != list(expected):
        raise RuntimeError(f"unexpected Tier-A tensor inventory: {shard}")
    for name, (sample_shape, dtype) in expected.items():
        spec = manifest["tensors"].get(name, {})
        if spec.get("sample_shape") != list(sample_shape) or spec.get("dtype") != dtype:
            raise RuntimeError(f"Tier-A tensor specification mismatch for {name}")
        _validate_array(shard / f"{name}.npy", spec, verify_hashes)
    return manifest


def validate_tier_b_shard(
    shard: Path,
    held_class: str,
    rows: Sequence[Mapping[str, Any]],
    provenance: CacheProvenance,
    *,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    manifest = _manifest(shard)
    sample_ids = [stable_sample_id(row) for row in rows]
    if manifest.get("schema") != TIER_B_SCHEMA or manifest.get("tier") != "B":
        raise RuntimeError(f"wrong Tier-B cache schema: {shard}")
    if manifest.get("held_class") != held_class or manifest.get("sample_ids") != sample_ids:
        raise RuntimeError(f"Tier-B held fold or deterministic sample ordering mismatch: {shard}")
    if any(str(row["class_name"]) == held_class for row in rows):
        raise RuntimeError("held supervision reached requested Tier-B inventory")
    if manifest.get("source_classes") != sorted({str(row["class_name"]) for row in rows}):
        raise RuntimeError("Tier-B source inventory mismatch")
    if manifest.get("held_mask_reads") != 0 or manifest.get("sample_count") != len(rows):
        raise RuntimeError("Tier-B held supervision firewall failed")
    _require_provenance(manifest, provenance)
    expected = {"source_mask": (MASK_SHAPE, "float32"), "teacher_region": (TEACHER_SHAPE, "float32")}
    if manifest.get("tensor_names") != list(expected):
        raise RuntimeError(f"unexpected Tier-B tensor inventory: {shard}")
    for name, (sample_shape, dtype) in expected.items():
        spec = manifest["tensors"].get(name, {})
        if spec.get("sample_shape") != list(sample_shape) or spec.get("dtype") != dtype:
            raise RuntimeError(f"Tier-B tensor specification mismatch for {name}")
        _validate_array(shard / f"{name}.npy", spec, verify_hashes)
    return manifest


def _array_spec(path: Path, array: np.ndarray, sample_shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "sample_shape": list(sample_shape),
        "storage_shape": list(array.shape),
        "dtype": str(array.dtype),
        "file_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_tier_a_shard(
    root: Path,
    class_name: str,
    sample_ids: Sequence[str],
    tensors: Iterable[tuple[torch.Tensor, torch.Tensor]],
    provenance: CacheProvenance,
) -> dict[str, Any]:
    final = root / "tier_a" / class_name
    if final.exists():
        return validate_tier_a_shard(final, class_name, sample_ids, provenance, verify_hashes=True)
    temporary = final.with_name(f".{class_name}.{uuid.uuid4().hex}.incomplete")
    temporary.mkdir(parents=True)
    count = len(sample_ids)
    seg_path = temporary / "seg_features.npy"
    native_path = temporary / "native_logits.npy"
    seg_array = np.lib.format.open_memmap(seg_path, mode="w+", dtype=np.float32, shape=(count, *SEGMENTATION_SHAPE))
    native_array = np.lib.format.open_memmap(native_path, mode="w+", dtype=np.float32, shape=(count, *NATIVE_LOGIT_SHAPE))
    observed = 0
    for observed, (seg_features, native_logits) in enumerate(tensors, start=1):
        if tuple(seg_features.shape) != SEGMENTATION_SHAPE or seg_features.dtype != torch.float32:
            raise RuntimeError("seg_features cache tensor contract changed")
        if tuple(native_logits.shape) != NATIVE_LOGIT_SHAPE or native_logits.dtype != torch.float32:
            raise RuntimeError("native_logits cache tensor contract changed")
        seg_array[observed - 1] = seg_features.detach().cpu().numpy()
        native_array[observed - 1] = native_logits.detach().cpu().numpy()
    if observed != count:
        raise RuntimeError(f"incomplete Tier-A write: expected {count}, observed {observed}")
    seg_array.flush()
    native_array.flush()
    manifest: dict[str, Any] = {
        "schema": TIER_A_SCHEMA,
        "tier": "A",
        "completion_status": "COMPLETE",
        "class": class_name,
        "sample_ids": list(sample_ids),
        "sample_count": count,
        "tensor_names": ["seg_features", "native_logits"],
        "contains_gt": False,
        "contains_masks": False,
        "contains_teacher_targets": False,
        **provenance.as_dict(),
    }
    manifest["tensors"] = {
        "seg_features": _array_spec(seg_path, seg_array, SEGMENTATION_SHAPE),
        "native_logits": _array_spec(native_path, native_array, NATIVE_LOGIT_SHAPE),
    }
    atomic_write_json(temporary / "manifest.json", manifest)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final)
    return manifest


def write_tier_b_shard(
    root: Path,
    held_class: str,
    rows: Sequence[Mapping[str, Any]],
    tensors: Iterable[tuple[torch.Tensor, torch.Tensor]],
    provenance: CacheProvenance,
    *,
    source_mask_file_reads: int,
) -> dict[str, Any]:
    if any(str(row["class_name"]) == held_class for row in rows):
        raise RuntimeError("refusing to cache held supervision")
    sample_ids = [stable_sample_id(row) for row in rows]
    final = root / "tier_b" / held_class
    if final.exists():
        return validate_tier_b_shard(final, held_class, rows, provenance, verify_hashes=True)
    temporary = final.with_name(f".{held_class}.{uuid.uuid4().hex}.incomplete")
    temporary.mkdir(parents=True)
    count = len(rows)
    mask_path = temporary / "source_mask.npy"
    teacher_path = temporary / "teacher_region.npy"
    mask_array = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.float32, shape=(count, *MASK_SHAPE))
    teacher_array = np.lib.format.open_memmap(teacher_path, mode="w+", dtype=np.float32, shape=(count, *TEACHER_SHAPE))
    observed = 0
    for observed, (source_mask, teacher_region) in enumerate(tensors, start=1):
        if tuple(source_mask.shape) != MASK_SHAPE or source_mask.dtype != torch.float32:
            raise RuntimeError("source mask cache tensor contract changed")
        if tuple(teacher_region.shape) != TEACHER_SHAPE or teacher_region.dtype != torch.float32:
            raise RuntimeError("teacher cache tensor contract changed")
        mask_array[observed - 1] = source_mask.detach().cpu().numpy()
        teacher_array[observed - 1] = teacher_region.detach().cpu().numpy()
    if observed != count:
        raise RuntimeError(f"incomplete Tier-B write: expected {count}, observed {observed}")
    mask_array.flush()
    teacher_array.flush()
    manifest: dict[str, Any] = {
        "schema": TIER_B_SCHEMA,
        "tier": "B",
        "completion_status": "COMPLETE",
        "held_class": held_class,
        "source_classes": sorted({str(row["class_name"]) for row in rows}),
        "sample_ids": sample_ids,
        "sample_count": count,
        "tensor_names": ["source_mask", "teacher_region"],
        "held_mask_reads": 0,
        "source_mask_file_reads": int(source_mask_file_reads),
        **provenance.as_dict(),
    }
    manifest["tensors"] = {
        "source_mask": _array_spec(mask_path, mask_array, MASK_SHAPE),
        "teacher_region": _array_spec(teacher_path, teacher_array, TEACHER_SHAPE),
    }
    atomic_write_json(temporary / "manifest.json", manifest)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final)
    return manifest


class TierADataset(Dataset):
    """Read GT-free frozen tensors in the exact requested row order."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        root: Path,
        provenance: CacheProvenance,
        *,
        load_seg_features: bool = True,
        load_native_logits: bool = True,
    ) -> None:
        self.rows = tuple(rows)
        self.root = Path(root)
        self.provenance = provenance
        self.load_seg_features = bool(load_seg_features)
        self.load_native_logits = bool(load_native_logits)
        self.locations: list[tuple[str, int]] = []
        by_class: dict[str, list[Mapping[str, Any]]] = {}
        for row in self.rows:
            by_class.setdefault(str(row["class_name"]), []).append(row)
        indices: dict[str, dict[str, int]] = {}
        for class_name, class_rows in by_class.items():
            all_manifest = _manifest(self.root / "tier_a" / class_name)
            all_ids = list(all_manifest.get("sample_ids", []))
            validate_tier_a_shard(self.root / "tier_a" / class_name, class_name, all_ids, provenance)
            indices[class_name] = {sample_id: index for index, sample_id in enumerate(all_ids)}
            if len(indices[class_name]) != len(all_ids):
                raise RuntimeError(f"duplicate sample ID in Tier-A class {class_name}")
        for row in self.rows:
            class_name = str(row["class_name"])
            sample_id = stable_sample_id(row)
            if sample_id not in indices[class_name]:
                raise RuntimeError(f"missing Tier-A sample: {sample_id}")
            self.locations.append((class_name, indices[class_name][sample_id]))
        self._arrays: dict[str, tuple[np.memmap, np.memmap | None]] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_arrays"] = {}
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def _class_arrays(self, class_name: str) -> tuple[np.memmap, np.memmap | None]:
        if class_name not in self._arrays:
            shard = self.root / "tier_a" / class_name
            self._arrays[class_name] = (
                np.load(shard / "seg_features.npy", mmap_mode="r", allow_pickle=False),
                np.load(shard / "native_logits.npy", mmap_mode="r", allow_pickle=False)
                if self.load_native_logits
                else None,
            )
        return self._arrays[class_name]

    def __getitem__(self, index: int) -> dict[str, Any]:
        class_name, shard_index = self.locations[index]
        seg, native = self._class_arrays(class_name)
        row = self.rows[index]
        result = {
            "class_name": class_name,
            "image_path": str(row["image_path"]),
            "sample_id": stable_sample_id(row),
            "index": index,
        }
        if self.load_native_logits:
            if native is None:
                raise RuntimeError("native-logit cache was not loaded")
            result["native_logits"] = torch.from_numpy(np.array(native[shard_index], copy=True))
        if self.load_seg_features:
            result["seg_features"] = torch.from_numpy(np.array(seg[shard_index], copy=True))
        return result


class CachedSourceDataset(Dataset):
    """Join Tier A with fold-local Tier B while keeping held supervision unreachable."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        held_class: str,
        root: Path,
        provenance: CacheProvenance,
        *,
        load_source_mask: bool = True,
        load_native_logits: bool = True,
    ) -> None:
        if any(str(row["class_name"]) == held_class for row in rows):
            raise RuntimeError("held class reached cached source dataset")
        self.rows = tuple(rows)
        self.held_class = held_class
        self.root = Path(root)
        self.load_source_mask = bool(load_source_mask)
        self.tier_a = TierADataset(
            self.rows,
            self.root,
            provenance,
            load_native_logits=load_native_logits,
        )
        tier_b_manifest = _manifest(self.root / "tier_b" / held_class)
        all_ids = list(tier_b_manifest.get("sample_ids", []))
        all_rows = [
            {"class_name": sample_id.split(":", 1)[0], "image_path": sample_id.split(":", 1)[1]}
            for sample_id in all_ids
        ]
        validate_tier_b_shard(self.root / "tier_b" / held_class, held_class, all_rows, provenance)
        by_id = {sample_id: index for index, sample_id in enumerate(all_ids)}
        if len(by_id) != len(all_ids):
            raise RuntimeError("duplicate Tier-B sample ID")
        self.tier_b_locations = []
        for row in self.rows:
            sample_id = stable_sample_id(row)
            if sample_id not in by_id:
                raise RuntimeError(f"missing Tier-B source sample: {sample_id}")
            self.tier_b_locations.append(by_id[sample_id])
        self._mask: np.memmap | None = None
        self._teacher: np.memmap | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_mask"] = None
        state["_teacher"] = None
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._teacher is None or (self.load_source_mask and self._mask is None):
            shard = self.root / "tier_b" / self.held_class
            self._teacher = np.load(shard / "teacher_region.npy", mmap_mode="r", allow_pickle=False)
            if self.load_source_mask:
                self._mask = np.load(shard / "source_mask.npy", mmap_mode="r", allow_pickle=False)
        result = self.tier_a[index]
        tier_b_index = self.tier_b_locations[index]
        if self.load_source_mask:
            if self._mask is None:
                raise RuntimeError("source mask cache was not loaded")
            result["mask"] = torch.from_numpy(np.array(self._mask[tier_b_index], copy=True))
        result["teacher_region"] = torch.from_numpy(np.array(self._teacher[tier_b_index], copy=True))
        return result
