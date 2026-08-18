"""Deterministic, domain-separated VisA data paths for SABRA setup.

The evidence path exposes only image pixels and public class identity.  It
does not retain labels or mask paths, so downstream evidence construction
cannot accidentally ask the dataset for ground truth.  The evaluation path
is separate and is used only for validating the deterministic mask transform
during setup.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_SIZE = 518
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
EXPECTED_VISA_CLASSES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_visa_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"VisA metadata line {line_number} is not an object")
        row["_line_number"] = line_number
        rows.append(row)
    return rows


def safe_data_path(root: Path, relative_path: str) -> Path:
    """Resolve a relative VisA path and reject traversal or external links."""
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("dataset path must be a non-empty string")
    root = root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"dataset path escapes root: {relative_path!r}") from exc
    return candidate


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("RGB").copy()


def _open_mask(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.convert("L").copy()


def _image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def _mask_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )


class VisaEvidenceDataset(Dataset):
    """GT-free deterministic VisA image path.

    Rows are sanitized at construction time.  The object intentionally has
    no ``label``, ``mask_path``, or ``mask`` field and never opens a mask.
    """

    path_role = "GT_FREE_EVIDENCE"

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        data_root: Path,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.data_root = data_root.resolve()
        self.image_size = int(image_size)
        self.samples = [
            {
                "class_name": str(row["class_name"]),
                "image_path": str(row["image_path"]),
            }
            for row in rows
        ]
        self.image_transform = _image_transform(self.image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        image_path = safe_data_path(self.data_root, row["image_path"])
        image = self.image_transform(_open_rgb(image_path)).contiguous()
        return {
            "image": image,
            "class_name": row["class_name"],
            "image_path": row["image_path"],
            "index": int(index),
        }


class VisaEvaluationDataset(Dataset):
    """Separate deterministic image+mask path for setup-only GT validation."""

    path_role = "GT_EVALUATION_ONLY"

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        data_root: Path,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.data_root = data_root.resolve()
        self.image_size = int(image_size)
        self.samples = [dict(row) for row in rows]
        self.image_transform = _image_transform(self.image_size)
        self.mask_transform = _mask_transform(self.image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        image_path = safe_data_path(self.data_root, str(row["image_path"]))
        image = self.image_transform(_open_rgb(image_path)).contiguous()
        label = int(row["label"])
        if label:
            mask_path_value = row.get("mask_path")
            if not isinstance(mask_path_value, str) or not mask_path_value:
                raise ValueError("anomaly sample is missing mask_path")
            mask_path = safe_data_path(self.data_root, mask_path_value)
            mask = self.mask_transform(_open_mask(mask_path)).gt(0).to(torch.float32)
        else:
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        return {
            "image": image,
            "mask": mask.contiguous(),
            "label": torch.tensor(label, dtype=torch.int64),
            "class_name": str(row["class_name"]),
            "image_path": str(row["image_path"]),
            "index": int(index),
        }


def transform_contract(image_size: int = IMAGE_SIZE) -> dict[str, Any]:
    """Return the frozen transform contract for audit artifacts."""
    return {
        "image": [
            {"op": "Resize", "size": [image_size, image_size], "interpolation": "bicubic"},
            {"op": "ToTensor"},
            {"op": "Normalize", "mean": list(CLIP_MEAN), "std": list(CLIP_STD)},
        ],
        "mask": [
            {"op": "Resize", "size": [image_size, image_size], "interpolation": "nearest"},
            {"op": "ToTensor"},
            {"op": "binary_threshold", "threshold": 0},
        ],
        "stochastic_augmentation": False,
        "forbidden_ops": [
            "Gaussian noise",
            "ColorJitter",
            "random rotation",
            "affine",
            "horizontal flip",
            "vertical flip",
            "random crop",
        ],
    }
