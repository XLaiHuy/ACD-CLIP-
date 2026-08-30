"""Bounded, disk-backed storage for production evaluation pixels."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Iterator

import numpy as np


SPOOL_DIR_NAME = ".cir_eval_spool"
SPOOL_SCHEMA = "CIR_EVAL_SPOOL_V1"


@dataclass
class _ClassState:
    name: str
    directory: Path
    pixel_count: int = 0
    image_scores: list[float] | None = None
    image_labels: list[int] | None = None

    def __post_init__(self) -> None:
        if self.image_scores is None:
            self.image_scores = []
        if self.image_labels is None:
            self.image_labels = []

    @property
    def score_path(self) -> Path:
        return self.directory / "pixel_scores.float32"

    @property
    def label_path(self) -> Path:
        return self.directory / "pixel_labels.uint8"


@dataclass(frozen=True)
class SpoolClass:
    """A finalized class view; pixel arrays are opened only on demand."""

    class_name: str
    pixel_count: int
    score_path: Path
    label_path: Path
    image_scores: tuple[float, ...]
    image_labels: tuple[int, ...]

    @contextmanager
    def open_arrays(self) -> Iterator[tuple[np.memmap, np.memmap]]:
        if self.pixel_count <= 0:
            raise ValueError(f"class {self.class_name!r} has no pixels")
        expected_scores = self.pixel_count * np.dtype(np.float32).itemsize
        expected_labels = self.pixel_count * np.dtype(np.uint8).itemsize
        if self.score_path.stat().st_size != expected_scores:
            raise ValueError(f"invalid score spool size for class {self.class_name!r}")
        if self.label_path.stat().st_size != expected_labels:
            raise ValueError(f"invalid label spool size for class {self.class_name!r}")
        scores = np.memmap(
            self.score_path,
            dtype=np.float32,
            mode="r",
            shape=(self.pixel_count,),
        )
        labels = np.memmap(
            self.label_path,
            dtype=np.uint8,
            mode="r",
            shape=(self.pixel_count,),
        )
        try:
            yield scores, labels
        finally:
            for array in (scores, labels):
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()


class EvaluationSpool:
    """Append pixel outputs without retaining a target-sized Python object graph."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self._states: dict[str, _ClassState] = {}
        self._closed = False

    @classmethod
    def create(cls, output_dir: str | Path) -> "EvaluationSpool":
        """Create a fresh owned spool, removing only a prior stale spool."""
        output_root = Path(output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        root = output_root / SPOOL_DIR_NAME
        if root.exists() or root.is_symlink():
            if root.is_symlink() or not root.is_dir():
                raise ValueError(f"refusing to replace non-directory spool path: {root}")
            shutil.rmtree(root)
        root.mkdir()
        (root / "format.json").write_text(
            json.dumps(
                {
                    "schema": SPOOL_SCHEMA,
                    "pixel_score_dtype": "float32",
                    "pixel_label_dtype": "uint8",
                    "ownership": "temporary evaluation spool",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return cls(root)

    def _state(self, class_name: str) -> _ClassState:
        state = self._states.get(class_name)
        if state is None:
            state = _ClassState(
                name=class_name,
                directory=self.root / f"class_{len(self._states):04d}",
            )
            state.directory.mkdir()
            self._states[class_name] = state
        return state

    def append(
        self,
        class_name: str,
        pixel_scores: np.ndarray,
        pixel_labels: np.ndarray,
        image_score: float,
        image_label: int,
    ) -> None:
        """Append one image's exact model scores and binary mask."""
        if self._closed:
            raise RuntimeError("evaluation spool is closed")
        scores = np.asarray(pixel_scores).reshape(-1)
        if scores.dtype != np.float32:
            scores = np.asarray(scores, dtype=np.float32)
        scores = np.ascontiguousarray(scores)
        labels = np.asarray(pixel_labels).reshape(-1)
        if scores.shape != labels.shape or scores.size == 0:
            raise ValueError("spooled pixel scores and labels must have equal non-empty shape")
        if not np.isfinite(scores).all():
            raise ValueError("spooled pixel scores must be finite")
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("spooled pixel labels must be 0/1")
        labels = np.ascontiguousarray(np.asarray(labels, dtype=np.uint8))
        image_label = int(image_label)
        if image_label not in (0, 1):
            raise ValueError("spooled image labels must be 0/1")
        state = self._state(str(class_name))
        with state.score_path.open("ab") as handle:
            scores.tofile(handle)
        with state.label_path.open("ab") as handle:
            labels.tofile(handle)
        state.pixel_count += int(scores.size)
        state.image_scores.append(float(image_score))
        state.image_labels.append(image_label)

    def classes(self) -> tuple[SpoolClass, ...]:
        if self._closed:
            raise RuntimeError("evaluation spool is closed")
        return tuple(
            SpoolClass(
                class_name=state.name,
                pixel_count=state.pixel_count,
                score_path=state.score_path,
                label_path=state.label_path,
                image_scores=tuple(state.image_scores),
                image_labels=tuple(state.image_labels),
            )
            for state in self._states.values()
        )

    def close(self) -> None:
        self._closed = True

    def cleanup(self) -> None:
        """Remove only this spool directory; final outputs are outside it."""
        self.close()
        if self.root.exists() or self.root.is_symlink():
            if self.root.is_symlink() or not self.root.is_dir():
                raise ValueError(f"refusing to remove non-directory spool path: {self.root}")
            shutil.rmtree(self.root)
