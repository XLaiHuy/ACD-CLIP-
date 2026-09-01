"""Disk-backed exact binary metrics for full-resolution evaluation."""

from __future__ import annotations

import gc
import mmap
import shutil
from pathlib import Path

import numpy as np
import torch


METRIC_CHUNK_SIZE = 1_000_000


def _close_memmap(array: np.memmap | None) -> None:
    if array is not None:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _drop_memmap_pages(array: np.memmap | None) -> None:
    if array is None:
        return
    mapping = getattr(array, "_mmap", None)
    advise = getattr(mapping, "madvise", None)
    if callable(advise) and hasattr(mmap, "MADV_DONTNEED"):
        try:
            advise(mmap.MADV_DONTNEED)
        except (OSError, ValueError):
            pass


def _iter_packed_group_chunks(
    keys: np.ndarray,
    *,
    descending: bool,
    chunk_size: int,
):
    if chunk_size <= 0:
        raise ValueError("metric chunk_size must be positive")
    count = int(keys.size)
    pending_code = None
    pending_start = pending_positive = pending_size = None
    for traversal_start in range(0, count, chunk_size):
        if descending:
            chunk_end = count - traversal_start
            chunk_start = max(0, chunk_end - chunk_size)
            chunk_keys = np.asarray(keys[chunk_start:chunk_end])[::-1]
        else:
            chunk_start = traversal_start
            chunk_end = min(count, chunk_start + chunk_size)
            chunk_keys = np.asarray(keys[chunk_start:chunk_end])
        codes = chunk_keys >> np.uint64(1)
        labels = (chunk_keys & np.uint64(1)).astype(np.int64, copy=False)
        breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
        starts = np.empty(breaks.size + 1, dtype=np.int64)
        starts[0] = 0
        starts[1:] = breaks
        sizes = np.diff(np.append(starts, chunk_keys.size)).astype(np.int64)
        positives = np.add.reduceat(labels, starts, dtype=np.int64).astype(np.float64)
        starts = starts + traversal_start
        group_codes = codes[np.concatenate(([0], breaks))]

        if pending_code is not None:
            if group_codes[0] == pending_code:
                positives[0] += float(pending_positive)
                sizes[0] += int(pending_size)
                starts[0] = int(pending_start)
            else:
                yield (
                    np.asarray([pending_start], dtype=np.int64),
                    np.asarray([pending_positive], dtype=np.float64),
                    np.asarray([pending_size], dtype=np.int64),
                )
            pending_code = None
        if sizes.size > 1:
            yield starts[:-1], positives[:-1], sizes[:-1]
        pending_code = group_codes[-1]
        pending_start = int(starts[-1])
        pending_positive = float(positives[-1])
        pending_size = int(sizes[-1])
    if pending_code is not None:
        yield (
            np.asarray([pending_start], dtype=np.int64),
            np.asarray([pending_positive], dtype=np.float64),
            np.asarray([pending_size], dtype=np.int64),
        )


def _metrics_from_group_iterators(ascending_groups, descending_groups, *, positives: int, total: int):
    negatives = float(total - int(positives))
    negative_seen = 0.0
    pair_count = 0.0
    for _, group_positives, group_sizes in ascending_groups:
        sizes = group_sizes.astype(np.float64, copy=False)
        negatives_in_group = sizes - group_positives
        negative_before = negative_seen + np.cumsum(negatives_in_group) - negatives_in_group
        pair_count += float(np.sum(
            group_positives * (negative_before + 0.5 * negatives_in_group),
            dtype=np.float64,
        ))
        negative_seen += float(np.sum(negatives_in_group, dtype=np.float64))
    auroc = pair_count / (positives * negatives)

    true_positive = 0.0
    seen = 0.0
    previous_recall = 0.0
    average_precision = 0.0
    for _, group_positives, group_sizes in descending_groups:
        cumulative_positive = true_positive + np.cumsum(group_positives, dtype=np.float64)
        cumulative_seen = seen + np.cumsum(group_sizes, dtype=np.float64)
        recalls = cumulative_positive / positives
        previous_recalls = np.empty_like(recalls)
        previous_recalls[0] = previous_recall
        if recalls.size > 1:
            previous_recalls[1:] = recalls[:-1]
        precisions = cumulative_positive / cumulative_seen
        average_precision += float(np.sum(
            (recalls - previous_recalls) * precisions,
            dtype=np.float64,
        ))
        true_positive = float(cumulative_positive[-1])
        seen = float(cumulative_seen[-1])
        previous_recall = float(recalls[-1])
    return float(auroc), float(average_precision)


def exact_binary_metrics_from_files(
    score_path: str | Path,
    label_path: str | Path,
    pixel_count: int,
    *,
    workspace: str | Path,
    allow_undefined: bool = False,
    chunk_size: int = METRIC_CHUNK_SIZE,
) -> tuple[float | None, float | None]:
    count = int(pixel_count)
    if count <= 0:
        raise ValueError("pixel_count must be positive")
    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError("metric chunk_size must be positive")
    score_file = Path(score_path)
    label_file = Path(label_path)
    if score_file.stat().st_size != count * np.dtype(np.float32).itemsize:
        raise ValueError("invalid score spool size")
    if label_file.stat().st_size != count * np.dtype(np.uint8).itemsize:
        raise ValueError("invalid label spool size")
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    key_path = workspace_path / "pixel_order.uint64"
    scores = labels = keys = None
    try:
        scores = np.memmap(score_file, dtype=np.float32, mode="r", shape=(count,))
        labels = np.memmap(label_file, dtype=np.uint8, mode="r", shape=(count,))
        keys = np.memmap(key_path, dtype=np.uint64, mode="w+", shape=(count,))
        positives = 0
        for start in range(0, count, chunk):
            end = min(count, start + chunk)
            score_chunk = np.asarray(scores[start:end])
            label_chunk = np.asarray(labels[start:end])
            if not np.isfinite(score_chunk).all():
                raise ValueError("metric scores must be finite")
            if not np.isin(label_chunk, (0, 1)).all():
                raise ValueError("binary labels must be 0/1")
            positives += int(label_chunk.sum(dtype=np.int64))
            bits = score_chunk.view(np.uint32).copy()
            bits[score_chunk == 0] = 0
            sign = bits >> np.uint32(31)
            flip = np.where(sign != 0, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
            ordered = bits ^ flip
            keys[start:end] = (ordered.astype(np.uint64) << np.uint64(1)) | label_chunk.astype(np.uint64)
            _drop_memmap_pages(scores)
            _drop_memmap_pages(labels)
            if start // chunk % 32 == 31:
                keys.flush()
                _drop_memmap_pages(keys)
        if positives == 0 or positives == count:
            if allow_undefined:
                return None, None
            raise ValueError("binary metric requires both positive and negative labels")
        keys.flush()
        _drop_memmap_pages(scores)
        _drop_memmap_pages(labels)
        _drop_memmap_pages(keys)
        _close_memmap(scores)
        _close_memmap(labels)
        _close_memmap(keys)
        scores = labels = keys = None
        gc.collect()
        keys = np.memmap(key_path, dtype=np.uint64, mode="r+", shape=(count,))
        keys.sort(kind="quicksort")
        keys.flush()
        return _metrics_from_group_iterators(
            _iter_packed_group_chunks(keys, descending=False, chunk_size=chunk),
            _iter_packed_group_chunks(keys, descending=True, chunk_size=chunk),
            positives=positives,
            total=count,
        )
    finally:
        _close_memmap(scores)
        _close_memmap(labels)
        _close_memmap(keys)
        if key_path.exists() or key_path.is_symlink():
            key_path.unlink()


class ExactBinaryAccumulator:
    """Append float32 scores and uint8 labels without retaining pixels in RAM."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if self.root.exists() or self.root.is_symlink():
            if self.root.is_symlink() or not self.root.is_dir():
                raise ValueError(f"refusing to replace non-directory metric spool: {self.root}")
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.score_path = self.root / "pixel_scores.float32"
        self.label_path = self.root / "pixel_labels.uint8"
        self.pixel_count = 0

    def update(self, scores, labels) -> None:
        scores_np = np.ascontiguousarray(scores.detach().float().cpu().numpy().reshape(-1))
        labels_np = np.ascontiguousarray(labels.detach().to(torch.uint8).cpu().numpy().reshape(-1))
        if scores_np.shape != labels_np.shape or scores_np.size == 0:
            raise ValueError("pixel score/label shapes must match and be non-empty")
        if not np.isfinite(scores_np).all() or not np.isin(labels_np, (0, 1)).all():
            raise ValueError("pixel spool contains non-finite scores or non-binary labels")
        with self.score_path.open("ab") as handle:
            scores_np.tofile(handle)
        with self.label_path.open("ab") as handle:
            labels_np.tofile(handle)
        self.pixel_count += int(scores_np.size)

    def compute(self) -> tuple[float | None, float | None]:
        return exact_binary_metrics_from_files(
            self.score_path,
            self.label_path,
            self.pixel_count,
            workspace=self.root / "sort_workspace",
            allow_undefined=False,
        )

    def cleanup(self) -> None:
        if self.root.exists() or self.root.is_symlink():
            if self.root.is_symlink() or not self.root.is_dir():
                raise ValueError(f"refusing to remove non-directory metric spool: {self.root}")
            shutil.rmtree(self.root)
