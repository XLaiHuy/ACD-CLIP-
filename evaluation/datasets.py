"""Dataset adapters and bounded MVTec preflight.

The adapter used for preflight is also the adapter consumed by future model
evaluation.  Preflight never constructs a model or computes predictions.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

EXPECTED_MVTEC_CLASSES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
IMAGE_SIZE = 518
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class MVTecSample:
    image_path: str
    class_name: str
    label: int
    mask_path: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_mvtec_root(explicit: str | Path | None = None) -> Path | None:
    value = explicit or os.environ.get("ACDCLIP_MVTEC_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    data_root = os.environ.get("ACDCLIP_DATA_ROOT")
    if data_root:
        candidate = Path(data_root).expanduser().resolve() / "mvtec_ad"
        if candidate.exists():
            return candidate
    return None


class MVTecDatasetAdapter:
    identity = "dataset.hub.MVTec.jsonl + relative MVTec AD paths"
    role = "DEVELOPMENT"

    def __init__(self, root: str | Path, metadata_path: str | Path | None = None, image_size: int = IMAGE_SIZE):
        self.requested_root = Path(root).expanduser().resolve()
        self.root = self._resolve_wrapped_root(self.requested_root)
        self.metadata_path = Path(metadata_path or Path(__file__).resolve().parents[1] / "dataset/hub/MVTec.jsonl").resolve()
        self.image_size = int(image_size)
        self.samples = self._read_samples()
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    @staticmethod
    def _resolve_wrapped_root(root: Path) -> Path:
        wrapped = root / "mvtec_anomaly_detection"
        return wrapped if wrapped.is_dir() else root

    def _read_samples(self) -> list[MVTecSample]:
        rows: list[MVTecSample] = []
        for line_number, line in enumerate(self.metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                sample = MVTecSample(
                    image_path=str(raw["image_path"]),
                    class_name=str(raw["class_name"]),
                    label=int(raw["label"]),
                    mask_path=None if int(raw["label"]) == 0 else str(raw["mask_path"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid MVTec metadata line {line_number}") from exc
            rows.append(sample)
        return rows

    def _path(self, relative: str) -> Path:
        root = self.root.resolve()
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"MVTec path escapes root: {relative}") from exc
        return candidate

    def preflight(self, inspect_limit: int = 2) -> dict[str, Any]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"MVTec root does not exist: {self.root}")
        classes = tuple(sorted({sample.class_name for sample in self.samples}))
        if classes != tuple(sorted(EXPECTED_MVTEC_CLASSES)):
            raise ValueError(f"MVTec class set mismatch: {classes}")
        counts: dict[str, dict[str, int]] = {name: {"normal": 0, "anomaly": 0} for name in classes}
        for sample in self.samples:
            if sample.label not in (0, 1):
                raise ValueError(f"invalid MVTec label for {sample.image_path}")
            image_path = self._path(sample.image_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"missing MVTec image: {image_path}")
            if sample.label:
                if not sample.mask_path or not self._path(sample.mask_path).is_file():
                    raise FileNotFoundError(f"missing MVTec anomaly mask: {sample.mask_path}")
                counts[sample.class_name]["anomaly"] += 1
            else:
                counts[sample.class_name]["normal"] += 1
        if any(not values["normal"] or not values["anomaly"] for values in counts.values()):
            raise ValueError(f"MVTec class lacks both labels: {counts}")
        inspected: list[dict[str, Any]] = []
        for sample in self.samples[: max(0, int(inspect_limit))]:
            with Image.open(self._path(sample.image_path)) as handle:
                image = self.image_transform(handle.convert("RGB"))
            mask_shape = None
            if sample.mask_path:
                with Image.open(self._path(sample.mask_path)) as handle:
                    mask_shape = tuple(self.mask_transform(handle.convert("L")).shape)
            inspected.append({"image_path": sample.image_path, "image_shape": list(image.shape), "mask_shape": None if mask_shape is None else list(mask_shape)})
        return {
            "status": "PREFLIGHT_PASS",
            "role": self.role,
            "dataset_identity": self.identity,
            "resolved_root": str(self.root),
            "metadata_sha256": sha256_file(self.metadata_path),
            "class_names": list(classes),
            "class_counts": counts,
            "sample_count": len(self.samples),
            "transform": {"image_size": self.image_size, "image_interpolation": "bicubic", "mask_interpolation": "nearest"},
            "inspected": inspected,
            "model_inference": False,
        }

    def __iter__(self) -> Iterable[dict[str, Any]]:
        for index, sample in enumerate(self.samples):
            with Image.open(self._path(sample.image_path)) as handle:
                image = self.image_transform(handle.convert("RGB")).contiguous()
            if sample.mask_path:
                with Image.open(self._path(sample.mask_path)) as handle:
                    mask = self.mask_transform(handle.convert("L")).gt(0).to(torch.float32)
            else:
                mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            yield {"image": image, "mask": mask.contiguous(), "label": torch.tensor(sample.label), "class_name": sample.class_name, "image_path": sample.image_path, "index": index}
