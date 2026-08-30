"""Full-precision, tie-aware binary metrics shared by every evaluator."""
from __future__ import annotations

import gc
import mmap
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np


# The ordering is global, but score/label gathers and group statistics are
# bounded by this size. A caller may use a smaller value in parity tests.
METRIC_CHUNK_SIZE = 1_000_000


def _arrays(
    scores: Iterable[float],
    labels: Iterable[int],
    *,
    allow_undefined: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate inputs without widening a NumPy score array."""
    score_source = scores if isinstance(scores, np.ndarray) else list(scores)
    scores_array = np.asarray(score_source)
    if scores_array.dtype.kind not in "biuf":
        scores_array = np.asarray(score_source, dtype=np.float64)
    scores_array = scores_array.reshape(-1)
    label_source = labels if isinstance(labels, np.ndarray) else list(labels)
    labels_array = np.asarray(label_source, dtype=np.uint8).reshape(-1)
    if scores_array.shape != labels_array.shape or scores_array.size == 0:
        raise ValueError("scores and labels must be non-empty arrays of equal shape")
    if not np.isfinite(scores_array).all():
        raise ValueError("metric scores must be finite")
    if not np.isin(labels_array, (0, 1)).all():
        raise ValueError("binary labels must be 0/1")
    positives = int(labels_array.sum(dtype=np.int64))
    if positives == 0 or positives == labels_array.size:
        if allow_undefined:
            return None
        raise ValueError("binary metric requires both positive and negative labels")
    return scores_array, labels_array


def _descending_groups(scores: np.ndarray) -> tuple[np.ndarray, Iterator[tuple[int, int]]]:
    """Compatibility iterator for small callers.

    The production metric path uses ``_iter_group_chunks`` below, so it does
    not materialize sorted scores or visit one Python object per pixel.
    """
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]

    def groups() -> Iterator[tuple[int, int]]:
        start = 0
        while start < sorted_scores.size:
            end = start + 1
            while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
                end += 1
            yield start, end
            start = end

    return order, groups()


def _iter_group_chunks(
    scores: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    *,
    descending: bool,
    chunk_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield vectorized score-group statistics with chunk-boundary carry."""
    if chunk_size <= 0:
        raise ValueError("metric chunk_size must be positive")
    count = int(order.size)
    pending_score = None
    pending_start: int | None = None
    pending_positive: float | None = None
    pending_size: int | None = None

    for traversal_start in range(0, count, chunk_size):
        if descending:
            chunk_end = count - traversal_start
            chunk_start = max(0, chunk_end - chunk_size)
            indices = order[chunk_start:chunk_end][::-1]
        else:
            chunk_start = traversal_start
            chunk_end = min(count, chunk_start + chunk_size)
            indices = order[chunk_start:chunk_end]

        chunk_scores = np.asarray(scores[indices])
        chunk_labels = np.asarray(labels[indices])
        local_breaks = np.flatnonzero(chunk_scores[1:] != chunk_scores[:-1]) + 1
        local_starts = np.empty(local_breaks.size + 1, dtype=np.int64)
        local_starts[0] = 0
        local_starts[1:] = local_breaks
        group_sizes = np.diff(np.append(local_starts, chunk_scores.size)).astype(np.int64)
        group_positives = np.add.reduceat(
            chunk_labels, local_starts, dtype=np.int64
        ).astype(np.float64)
        group_starts = local_starts + traversal_start
        group_scores = chunk_scores[local_starts]

        if pending_score is not None:
            if group_scores[0] == pending_score:
                group_positives[0] += float(pending_positive)
                group_sizes[0] += int(pending_size)
                group_starts[0] = int(pending_start)
            else:
                yield (
                    np.asarray([pending_start], dtype=np.int64),
                    np.asarray([pending_positive], dtype=np.float64),
                    np.asarray([pending_size], dtype=np.int64),
                )
            pending_score = None

        if group_sizes.size > 1:
            yield group_starts[:-1], group_positives[:-1], group_sizes[:-1]

        pending_score = group_scores[-1]
        pending_start = int(group_starts[-1])
        pending_positive = float(group_positives[-1])
        pending_size = int(group_sizes[-1])

    if pending_score is not None:
        yield (
            np.asarray([pending_start], dtype=np.int64),
            np.asarray([pending_positive], dtype=np.float64),
            np.asarray([pending_size], dtype=np.int64),
        )


def _metrics_from_group_iterators(
    ascending_groups: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]],
    descending_groups: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    positives: float,
    total: int,
) -> tuple[float, float]:
    negatives = float(total - int(positives))
    negative_seen = 0.0
    pair_count = 0.0
    for _, group_positives, group_sizes in ascending_groups:
        group_sizes_float = group_sizes.astype(np.float64, copy=False)
        group_negatives = group_sizes_float - group_positives
        negative_before = negative_seen + np.cumsum(
            group_negatives, dtype=np.float64
        ) - group_negatives
        pair_count += float(
            np.sum(
                group_positives * (negative_before + 0.5 * group_negatives),
                dtype=np.float64,
            )
        )
        negative_seen += float(np.sum(group_negatives, dtype=np.float64))
    auroc = pair_count / (positives * negatives)

    true_positive = 0.0
    seen = 0.0
    previous_recall = 0.0
    average_precision = 0.0
    for _, group_positives, group_sizes in descending_groups:
        cumulative_positive = true_positive + np.cumsum(
            group_positives, dtype=np.float64
        )
        cumulative_seen = seen + np.cumsum(group_sizes, dtype=np.float64)
        recalls = cumulative_positive / positives
        previous_recalls = np.empty_like(recalls)
        previous_recalls[0] = previous_recall
        if recalls.size > 1:
            previous_recalls[1:] = recalls[:-1]
        precisions = cumulative_positive / cumulative_seen
        average_precision += float(
            np.sum(
                (recalls - previous_recalls) * precisions,
                dtype=np.float64,
            )
        )
        true_positive = float(cumulative_positive[-1])
        seen = float(cumulative_seen[-1])
        previous_recall = float(recalls[-1])
    return float(auroc), float(average_precision)


def _binary_metrics_from_arrays(
    scores_array: np.ndarray,
    labels_array: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[float, float]:
    """Compute exact AUROC and AP from one shared ascending ordering."""
    positives = float(labels_array.sum(dtype=np.int64))
    # Stability is unnecessary here: both metrics consume complete equal-score
    # groups, so the order of labels within a tie cannot affect either result.
    # Quicksort keeps only the shared int64 order instead of mergesort's large
    # auxiliary ordering buffer at real Brain scale.
    order = np.argsort(scores_array, kind="quicksort")
    try:
        return _metrics_from_group_iterators(
            _iter_group_chunks(
                scores_array,
                labels_array,
                order,
                descending=False,
                chunk_size=chunk_size,
            ),
            _iter_group_chunks(
                scores_array,
                labels_array,
                order,
                descending=True,
                chunk_size=chunk_size,
            ),
            positives=positives,
            total=int(labels_array.size),
        )
    finally:
        del order


def _close_memmap(array: np.memmap | None) -> None:
    if array is None:
        return
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _drop_memmap_pages(array: np.memmap | None) -> None:
    """Release already-consumed file-backed pages without changing the file."""
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
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield groups from sorted uint64(score-order, label) keys."""
    if chunk_size <= 0:
        raise ValueError("metric chunk_size must be positive")
    count = int(keys.size)
    pending_code = None
    pending_start: int | None = None
    pending_positive: float | None = None
    pending_size: int | None = None

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
        local_breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
        local_starts = np.empty(local_breaks.size + 1, dtype=np.int64)
        local_starts[0] = 0
        local_starts[1:] = local_breaks
        group_sizes = np.diff(np.append(local_starts, chunk_keys.size)).astype(np.int64)
        group_positives = np.add.reduceat(labels, local_starts, dtype=np.int64).astype(np.float64)
        group_codes = codes[local_starts]
        group_starts = local_starts + traversal_start

        if pending_code is not None:
            if group_codes[0] == pending_code:
                group_positives[0] += float(pending_positive)
                group_sizes[0] += int(pending_size)
                group_starts[0] = int(pending_start)
            else:
                yield (
                    np.asarray([pending_start], dtype=np.int64),
                    np.asarray([pending_positive], dtype=np.float64),
                    np.asarray([pending_size], dtype=np.int64),
                )
            pending_code = None

        if group_sizes.size > 1:
            yield group_starts[:-1], group_positives[:-1], group_sizes[:-1]

        pending_code = group_codes[-1]
        pending_start = int(group_starts[-1])
        pending_positive = float(group_positives[-1])
        pending_size = int(group_sizes[-1])

    if pending_code is not None:
        yield (
            np.asarray([pending_start], dtype=np.int64),
            np.asarray([pending_positive], dtype=np.float64),
            np.asarray([pending_size], dtype=np.int64),
        )


def _binary_metrics_from_packed_keys(
    keys: np.ndarray,
    *,
    positives: int,
    chunk_size: int,
) -> tuple[float, float]:
    return _metrics_from_group_iterators(
        _iter_packed_group_chunks(keys, descending=False, chunk_size=chunk_size),
        _iter_packed_group_chunks(keys, descending=True, chunk_size=chunk_size),
        positives=float(positives),
        total=int(keys.size),
    )


def binary_metrics_from_files(
    score_path: str | Path,
    label_path: str | Path,
    pixel_count: int,
    *,
    workspace: str | Path,
    allow_undefined: bool = False,
    chunk_size: int = METRIC_CHUNK_SIZE,
) -> tuple[float | None, float | None]:
    """Compute exact metrics from a float32/uint8 spool with bounded RSS.

    A packed uint64 key is sorted in place on disk. Its high bits encode the
    monotonic float32 score order and its low bit carries the binary label, so
    one shared sorted key stream serves both AUROC and AP without an index
    array, sorted copies, or ranks.
    """
    count = int(pixel_count)
    if count <= 0:
        raise ValueError("pixel_count must be positive")
    chunk = int(chunk_size)
    if chunk <= 0:
        raise ValueError("metric chunk_size must be positive")
    score_file = Path(score_path)
    label_file = Path(label_path)
    expected_score_bytes = count * np.dtype(np.float32).itemsize
    expected_label_bytes = count * np.dtype(np.uint8).itemsize
    if score_file.stat().st_size != expected_score_bytes:
        raise ValueError("invalid score spool size")
    if label_file.stat().st_size != expected_label_bytes:
        raise ValueError("invalid label spool size")
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    key_path = workspace_path / "pixel_order.uint64"
    if key_path.exists() or key_path.is_symlink():
        key_path.unlink()

    scores: np.memmap | None = None
    labels: np.memmap | None = None
    keys: np.memmap | None = None
    score_chunk = label_chunk = bits = ordered = key_chunk = None
    try:
        scores = np.memmap(score_file, dtype=np.float32, mode="r", shape=(count,))
        labels = np.memmap(label_file, dtype=np.uint8, mode="r", shape=(count,))
        keys = np.memmap(key_path, dtype=np.uint64, mode="w+", shape=(count,))
        positives = 0
        for chunk_index, start in enumerate(range(0, count, chunk), start=1):
            end = min(count, start + chunk)
            score_chunk = np.asarray(scores[start:end])
            label_chunk = np.asarray(labels[start:end])
            if not np.isfinite(score_chunk).all():
                raise ValueError("metric scores must be finite")
            if not np.isin(label_chunk, (0, 1)).all():
                raise ValueError("binary labels must be 0/1")
            positives += int(label_chunk.sum(dtype=np.int64))
            bits = score_chunk.view(np.uint32).copy()
            bits[score_chunk == 0] = 0  # make -0.0 and +0.0 one tie
            sign = bits >> np.uint32(31)
            flip = np.where(
                sign != 0,
                np.uint32(0xFFFFFFFF),
                np.uint32(0x80000000),
            )
            ordered = bits ^ flip
            key_chunk = (ordered.astype(np.uint64) << np.uint64(1)) | label_chunk.astype(np.uint64)
            keys[start:end] = key_chunk
            score_chunk = label_chunk = bits = ordered = key_chunk = None
            _drop_memmap_pages(scores)
            _drop_memmap_pages(labels)
            if chunk_index % 32 == 0:
                keys.flush()
                _drop_memmap_pages(keys)
        if positives == 0 or positives == count:
            if allow_undefined:
                return None, None
            raise ValueError("binary metric requires both positive and negative labels")
        score_chunk = label_chunk = bits = ordered = key_chunk = None
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
        return _binary_metrics_from_packed_keys(
            keys,
            positives=positives,
            chunk_size=chunk,
        )
    finally:
        score_chunk = label_chunk = bits = ordered = key_chunk = None
        _close_memmap(scores)
        _close_memmap(labels)
        _close_memmap(keys)
        if key_path.exists() or key_path.is_symlink():
            key_path.unlink()


def binary_metrics(
    scores: Iterable[float],
    labels: Iterable[int],
    *,
    allow_undefined: bool = False,
    chunk_size: int = METRIC_CHUNK_SIZE,
) -> tuple[float | None, float | None]:
    """Return exact tie-aware AUROC and AP using one score ordering."""
    arrays = _arrays(scores, labels, allow_undefined=allow_undefined)
    if arrays is None:
        return None, None
    return _binary_metrics_from_arrays(*arrays, chunk_size=int(chunk_size))


def binary_auroc(
    scores: Iterable[float], labels: Iterable[int], *, allow_undefined: bool = False
) -> float | None:
    """Exact tie-aware AUROC in raw [0,1] units."""
    return binary_metrics(scores, labels, allow_undefined=allow_undefined)[0]


def binary_average_precision(
    scores: Iterable[float], labels: Iterable[int], *, allow_undefined: bool = False
) -> float | None:
    """Exact threshold-grouped AP with ties grouped before recall increments."""
    return binary_metrics(scores, labels, allow_undefined=allow_undefined)[1]


def class_metrics(
    pixel_scores: Iterable[float],
    pixel_labels: Iterable[int],
    image_scores: Iterable[float],
    image_labels: Iterable[int],
    *,
    allow_undefined_image: bool = False,
) -> dict[str, float | None]:
    pixel_auroc, pixel_ap = binary_metrics(pixel_scores, pixel_labels)
    image_auroc, image_ap = binary_metrics(
        image_scores,
        image_labels,
        allow_undefined=allow_undefined_image,
    )
    return {
        "pixel_auroc": pixel_auroc,
        "pixel_ap": pixel_ap,
        "image_auroc": image_auroc,
        "image_ap": image_ap,
    }


def macro_metrics(per_class: Mapping[str, Mapping[str, float | None]]) -> dict[str, float | None]:
    if not per_class:
        raise ValueError("cannot compute a macro metric over no classes")
    names = ("pixel_auroc", "pixel_ap", "image_auroc", "image_ap")
    output: dict[str, float | None] = {}
    for name in names:
        values = [float(row[name]) for row in per_class.values() if row.get(name) is not None]
        if not values:
            output[name] = None
        else:
            output[name] = float(np.mean(values, dtype=np.float64))
    return output


def selection_score(metrics: Mapping[str, float | None]) -> float:
    weights = {"pixel_auroc": 0.35, "pixel_ap": 0.35, "image_auroc": 0.15, "image_ap": 0.15}
    missing = [name for name in weights if metrics.get(name) is None]
    if missing:
        raise ValueError(f"selection score requires all four metrics: {missing}")
    return float(sum(weights[name] * float(metrics[name]) for name in weights))
