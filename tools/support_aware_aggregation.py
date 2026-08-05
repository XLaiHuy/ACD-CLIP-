#!/usr/bin/env python3
"""Support-aware aggregation for six-dataset medical validation.

Rule version: support-aware-v2

Aggregation logic
-----------------
Per-dataset component scores (unchanged from summarize_phase4_results.py):

    dataset_image_score = arithmetic_mean(image_AUROC, image_AP)
        — only for datasets with valid image metrics

    dataset_pixel_score = arithmetic_mean(pixel_AUROC, pixel_AP)
        — only for datasets with valid pixel metrics

A dataset has **valid image metrics** when its val/test split contains
at least one normal image (label == 0) AND at least one anomaly image
(label == 1).  Single-class splits (e.g. anomaly-only polyp datasets)
produce image AUROC that is undefined; the test harness stores 0.0 in
that case, but 0.0 here means "unsupported", NOT a real measurement.

Image/pixel domain macros:

    image_macro = mean(dataset_image_score) over valid-image datasets
    pixel_macro = mean(dataset_pixel_score) over valid-pixel datasets

Combined selection score (same high-level balance rule as before):

    combined_score = harmonic_mean(image_macro, pixel_macro)

The harmonic mean is computed safely: if either macro is zero or both
denominators sum to zero, the result is 0.0.

This module does NOT invent zeros for unsupported metrics and does NOT
compute harmonic_mean(per-dataset image_score, per-dataset pixel_score)
when image metrics are unsupported.

Backward compatibility
----------------------
The function also returns a `legacy_combined_score` that reproduces the
old behaviour (six-dataset macro of per-dataset harmonic means, with
zeros included) for auditing purposes.

Fingerprint
-----------
`compute_config_fingerprint(config)` produces a deterministic SHA-256
hex fingerprint over relevant configuration keys.  The fingerprint is
stable when inputs are unchanged and changes when any relevant config
parameter changes.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASETS: list[str] = [
    "Brain",
    "Liver",
    "Retina",
    "Colon_clinicDB",
    "Colon_colonDB",
    "Colon_Kvasir",
]

SELECTION_RULE_VERSION = "support-aware-v2"

_METRIC_IMPL_VERSION = "summarize_phase4_results_v1"


# ---------------------------------------------------------------------------
# Low-level numeric helpers
# ---------------------------------------------------------------------------


def _arithmetic_mean(a: float, b: float) -> float:
    return (a + b) / 2.0


def harmonic_mean_safe(a: float, b: float) -> float:
    """Numerically safe harmonic mean of two non-negative floats.

    Returns 0.0 when either operand is 0 (undefined case) or when the
    denominator would be zero.
    """
    if a <= 0.0 or b <= 0.0:
        return 0.0
    denom = a + b
    if denom == 0.0:
        return 0.0
    return 2.0 * a * b / denom


# ---------------------------------------------------------------------------
# Image-metric validity
# ---------------------------------------------------------------------------


def is_image_metric_valid(normal_count: int, anomaly_count: int) -> bool:
    """Return True iff AUROC/AP are meaningful for an image-level task.

    AUROC requires at least one positive and one negative sample.
    Single-class splits make it undefined; test.py stores 0.0 which is
    NOT a real measurement.
    """
    return normal_count >= 1 and anomaly_count >= 1


def is_pixel_metric_valid(normal_count: int, anomaly_count: int) -> bool:
    """Return True iff pixel-level metrics are meaningful.

    Pixel AUROC/AP requires at least one positive pixel (anomaly image)
    and—for AUROC—at least one negative pixel (normal image with no mask).
    In practice we require at least one anomaly image; normal-only splits
    also cannot be evaluated at pixel level.
    """
    return anomaly_count >= 1


# ---------------------------------------------------------------------------
# Per-dataset row aggregation
# ---------------------------------------------------------------------------


def compute_dataset_scores(
    image_auroc: float,
    image_ap: float,
    pixel_auroc: float,
    pixel_ap: float,
    image_valid: bool,
    pixel_valid: bool,
) -> dict[str, Any]:
    """Compute per-dataset component scores without inventing zeros.

    Returns a dict with:
        image_score     float | None   (None when image_valid is False)
        pixel_score     float | None   (None when pixel_valid is False)
        image_valid     bool
        pixel_valid     bool
    """
    image_score: float | None = None
    pixel_score: float | None = None

    if image_valid:
        image_score = _arithmetic_mean(image_auroc, image_ap)
    if pixel_valid:
        pixel_score = _arithmetic_mean(pixel_auroc, pixel_ap)

    return {
        "image_AUROC": image_auroc,
        "image_AP": image_ap,
        "pixel_AUROC": pixel_auroc,
        "pixel_AP": pixel_ap,
        "image_score": image_score,
        "pixel_score": pixel_score,
        "image_valid": image_valid,
        "pixel_valid": pixel_valid,
    }


# ---------------------------------------------------------------------------
# Legacy (old) combined score for one epoch
# ---------------------------------------------------------------------------


def _legacy_harmonic_mean(left: float, right: float) -> float:
    """Reproduce the harmonic mean used by summarize_phase4_results.py."""
    return 0.0 if left + right == 0 else 2.0 * left * right / (left + right)


def compute_legacy_combined_score(
    per_dataset_rows: Sequence[Mapping[str, Any]],
) -> float:
    """Reproduce the old six-dataset macro of per-dataset harmonic means.

    This silently includes zeros for single-class datasets, which is the
    bug we are fixing.  We retain it for audit purposes.
    """
    if not per_dataset_rows:
        return 0.0
    scores = []
    for row in per_dataset_rows:
        img_s = row.get("image_score")
        pix_s = row.get("pixel_score")
        # Legacy: treat None as 0.0 (the bug)
        img_val = img_s if img_s is not None else 0.0
        pix_val = pix_s if pix_s is not None else 0.0
        scores.append(_legacy_harmonic_mean(img_val, pix_val))
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Support-aware epoch aggregation
# ---------------------------------------------------------------------------


def aggregate_epoch(
    per_dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-dataset rows for one epoch using support-aware rules.

    Parameters
    ----------
    per_dataset_rows:
        Sequence of dicts, each with keys produced by
        :func:`compute_dataset_scores` plus ``dataset``.

    Returns
    -------
    dict with:
        support_aware_image_macro    float | None
        support_aware_pixel_macro    float | None
        support_aware_combined_score float
        legacy_combined_score        float
        valid_image_datasets         list[str]
        excluded_image_datasets      list[str]
        valid_pixel_datasets         list[str]
        excluded_pixel_datasets      list[str]
        exclusion_reasons            dict[str, str]
    """
    valid_image_scores: list[float] = []
    valid_pixel_scores: list[float] = []
    valid_image_datasets: list[str] = []
    excluded_image_datasets: list[str] = []
    valid_pixel_datasets: list[str] = []
    excluded_pixel_datasets: list[str] = []
    exclusion_reasons: dict[str, str] = {}

    for row in per_dataset_rows:
        dataset = str(row.get("dataset", ""))
        img_valid = bool(row.get("image_valid", False))
        pix_valid = bool(row.get("pixel_valid", False))
        img_score = row.get("image_score")
        pix_score = row.get("pixel_score")
        reason = str(row.get("exclusion_reason", "")) if not img_valid else ""

        if img_valid and img_score is not None:
            valid_image_scores.append(float(img_score))
            valid_image_datasets.append(dataset)
        else:
            excluded_image_datasets.append(dataset)
            if not reason:
                reason = "image_metric_not_supported (single-class split)"
            exclusion_reasons[dataset] = reason

        if pix_valid and pix_score is not None:
            valid_pixel_scores.append(float(pix_score))
            valid_pixel_datasets.append(dataset)
        else:
            excluded_pixel_datasets.append(dataset)
            if dataset not in exclusion_reasons:
                exclusion_reasons[dataset] = "pixel_metric_not_supported"

    image_macro: float | None = (
        sum(valid_image_scores) / len(valid_image_scores)
        if valid_image_scores
        else None
    )
    pixel_macro: float | None = (
        sum(valid_pixel_scores) / len(valid_pixel_scores)
        if valid_pixel_scores
        else None
    )

    if image_macro is not None and pixel_macro is not None:
        support_aware_combined = harmonic_mean_safe(image_macro, pixel_macro)
    elif pixel_macro is not None:
        support_aware_combined = float(pixel_macro)
    elif image_macro is not None:
        support_aware_combined = float(image_macro)
    else:
        support_aware_combined = 0.0

    legacy_combined = compute_legacy_combined_score(per_dataset_rows)

    return {
        "support_aware_image_macro": image_macro,
        "support_aware_pixel_macro": pixel_macro,
        "support_aware_combined_score": support_aware_combined,
        "legacy_combined_score": legacy_combined,
        "valid_image_datasets": sorted(valid_image_datasets),
        "excluded_image_datasets": sorted(excluded_image_datasets),
        "valid_pixel_datasets": sorted(valid_pixel_datasets),
        "excluded_pixel_datasets": sorted(excluded_pixel_datasets),
        "exclusion_reasons": exclusion_reasons,
    }


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def compute_config_fingerprint(config: Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 fingerprint over relevant config keys.

    The fingerprint is stable when inputs are unchanged and changes when
    any relevant configuration parameter changes.  The output is a 64-char
    hex string.

    Required keys in *config* (any extras are silently ignored):
        selection_rule_version      str
        validation_split_manifest_hash  str
        dataset_list                list[str]
        valid_metric_policy         str
        checkpoint_list             list[str]   — basenames or hashes
        metric_implementation_version  str
        pixel_stride                int
        exact_pixel_metric_mode     str
        threshold_policy            str
        seed                        int
        image_scoring_rule          str
        postprocessing_config       dict | str
        temperature                 float | str
    """
    # Extract only the canonically ordered keys we care about.
    canonical_keys = [
        "selection_rule_version",
        "validation_split_manifest_hash",
        "dataset_list",
        "valid_metric_policy",
        "checkpoint_list",
        "metric_implementation_version",
        "pixel_stride",
        "exact_pixel_metric_mode",
        "threshold_policy",
        "seed",
        "image_scoring_rule",
        "postprocessing_config",
        "temperature",
    ]
    extracted: dict[str, Any] = {key: config.get(key) for key in canonical_keys}
    canonical = json.dumps(extracted, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Build full fingerprint config from protocol parameters
# ---------------------------------------------------------------------------


def build_fingerprint_config(
    *,
    validation_epochs: Sequence[int],
    manifest_root: Path | str,
    datasets: Sequence[str] = DATASETS,
    pixel_stride: int = 1,
    seed: int = 0,
    temperature: float = 1.0,
    threshold_policy: str = "none",
    exact_pixel_metric_mode: str = "external_sorted_chunks",
    image_scoring_rule: str = "arithmetic_mean(image_AUROC, image_AP)",
    postprocessing_config: str = "none",
) -> dict[str, Any]:
    """Construct the configuration dict used for fingerprinting.

    Computes a manifest hash from the JSONL files in *manifest_root* for
    the given *datasets*.  Falls back gracefully if manifests are absent.
    """
    manifest_root = Path(manifest_root)
    manifest_hash = _hash_manifests(manifest_root, datasets)
    # Checkpoint metadata: sorted list of (relative path, size bytes)
    checkpoint_list = _enumerate_checkpoints(manifest_root.parent / "train", validation_epochs)
    return {
        "selection_rule_version": SELECTION_RULE_VERSION,
        "validation_split_manifest_hash": manifest_hash,
        "dataset_list": sorted(datasets),
        "valid_metric_policy": (
            "image_valid_when_normal>=1_and_anomaly>=1; "
            "pixel_valid_when_anomaly>=1"
        ),
        "checkpoint_list": checkpoint_list,
        "metric_implementation_version": _METRIC_IMPL_VERSION,
        "pixel_stride": pixel_stride,
        "exact_pixel_metric_mode": exact_pixel_metric_mode,
        "threshold_policy": threshold_policy,
        "seed": seed,
        "image_scoring_rule": image_scoring_rule,
        "postprocessing_config": postprocessing_config,
        "temperature": temperature,
    }


def _hash_manifests(manifest_root: Path, datasets: Sequence[str]) -> str:
    """SHA-256 over the sorted content of all val JSONL manifest files."""
    h = hashlib.sha256()
    for ds in sorted(datasets):
        for suffix in ("val", "test"):
            p = manifest_root / f"{ds}_{suffix}.jsonl"
            if p.exists():
                h.update(f"{p.name}\n".encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def _enumerate_checkpoints(
    train_dir: Path, validation_epochs: Sequence[int]
) -> list[str]:
    """Return sorted list of 'adapter_N.pth:SIZE' strings for fingerprinting."""
    items = []
    for epoch in sorted(validation_epochs):
        p = train_dir / f"adapter_{epoch}.pth"
        if p.exists():
            items.append(f"adapter_{epoch}.pth:{p.stat().st_size}")
    return items


# ---------------------------------------------------------------------------
# Load per-dataset rows from existing CSV artifacts
# ---------------------------------------------------------------------------


def load_per_dataset_rows_from_csvs(
    save_path: Path,
    split: str,
    epochs: Sequence[int],
    datasets: Sequence[str] = DATASETS,
    manifest_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Load per-dataset metric rows from exact_results_*.csv files.

    Determines image/pixel metric validity from manifest label counts when
    *manifest_root* is provided.  Falls back to heuristic (image AUC == 0.0
    and image AP == 0.0) when no manifest is available.

    Returns a list of dicts, one per (dataset, epoch) combination, with
    the fields produced by :func:`compute_dataset_scores` plus ``dataset``
    and ``epoch``.
    """
    import re

    pattern = re.compile(r"^exact_results_(.+)_(val|test)_epoch_(\d+)\.csv$")
    rows: list[dict[str, Any]] = []

    # Pre-load validity from manifests
    validity: dict[str, tuple[int, int]] = {}  # dataset -> (normal, anomaly)
    if manifest_root is not None:
        manifest_root = Path(manifest_root)
        for ds in datasets:
            p = manifest_root / f"{ds}_{split}.jsonl"
            if not p.exists():
                # Try val manifest for test split as fallback
                p = manifest_root / f"{ds}_val.jsonl"
            if p.exists():
                import json as _json
                lines = p.read_text().splitlines()
                normal = sum(1 for l in lines if _json.loads(l).get("label", 0) == 0)
                anomaly = sum(1 for l in lines if _json.loads(l).get("label", 0) == 1)
                validity[ds] = (normal, anomaly)

    epoch_set = set(epochs)
    dataset_set = set(datasets)

    for path in sorted(save_path.glob(f"exact_results_*_{split}_epoch_*.csv")):
        m = pattern.match(path.name)
        if m is None:
            continue
        dataset, result_split, epoch_text = m.groups()
        epoch = int(epoch_text)
        if result_split != split or dataset not in dataset_set or epoch not in epoch_set:
            continue

        # Parse CSV
        import csv as _csv
        with open(path, newline="") as fh:
            reader = _csv.DictReader(fh)
            average = None
            for row in reader:
                if row.get("class name") == "Average":
                    average = row
                    break
        if average is None:
            raise ValueError(f"{path}: missing Average row")

        image_auroc = float(average["image AUC"])
        image_ap = float(average["image AP"])
        pixel_auroc = float(average["pixel AUC"])
        pixel_ap = float(average["pixel AP"])

        # Determine validity
        if dataset in validity:
            normal_count, anomaly_count = validity[dataset]
            img_valid = is_image_metric_valid(normal_count, anomaly_count)
            pix_valid = is_pixel_metric_valid(normal_count, anomaly_count)
            exclusion_reason = "" if img_valid else (
                f"single_class_split: normal={normal_count} anomaly={anomaly_count}"
            )
        else:
            # Fallback heuristic: if both image metrics are exactly 0.0,
            # treat as unsupported.  This is a conservative heuristic — a
            # real zero is possible but extremely unlikely.
            img_valid = not (image_auroc == 0.0 and image_ap == 0.0)
            pix_valid = True
            exclusion_reason = (
                "heuristic: image_AUC==0 and image_AP==0 (no manifest)"
                if not img_valid
                else ""
            )

        scores = compute_dataset_scores(
            image_auroc, image_ap, pixel_auroc, pixel_ap, img_valid, pix_valid
        )
        rows.append({
            "dataset": dataset,
            "epoch": epoch,
            "exclusion_reason": exclusion_reason,
            **scores,
        })

    return rows


# ---------------------------------------------------------------------------
# Full re-aggregation for all epochs
# ---------------------------------------------------------------------------


def reaggregate_all_epochs(
    save_path: Path,
    split: str,
    epochs: Sequence[int],
    manifest_root: Path | None = None,
    datasets: Sequence[str] = DATASETS,
) -> list[dict[str, Any]]:
    """Re-aggregate per-dataset rows grouped by epoch.

    Returns a list of per-epoch summary dicts with both legacy and
    support-aware scores.
    """
    rows = load_per_dataset_rows_from_csvs(
        save_path, split, epochs, datasets, manifest_root
    )

    # Group by epoch
    by_epoch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        by_epoch.setdefault(epoch, []).append(row)

    results = []
    for epoch in sorted(by_epoch):
        epoch_rows = by_epoch[epoch]
        agg = aggregate_epoch(epoch_rows)
        agg["epoch"] = epoch
        # Also include per-dataset details
        agg["per_dataset"] = sorted(
            [
                {
                    "dataset": r["dataset"],
                    "image_AUROC": r["image_AUROC"],
                    "image_AP": r["image_AP"],
                    "pixel_AUROC": r["pixel_AUROC"],
                    "pixel_AP": r["pixel_AP"],
                    "image_score": r["image_score"],
                    "pixel_score": r["pixel_score"],
                    "image_valid": r["image_valid"],
                    "pixel_valid": r["pixel_valid"],
                    "exclusion_reason": r.get("exclusion_reason", ""),
                }
                for r in epoch_rows
            ],
            key=lambda x: x["dataset"],
        )
        results.append(agg)
    return results


# ---------------------------------------------------------------------------
# Select best epoch
# ---------------------------------------------------------------------------


def select_best_epoch(
    epoch_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the best epoch using support_aware_combined_score.

    Tie-break: lower epoch wins on exact numerical ties.

    Rule: maximize support_aware_combined_score; ties go to lower epoch.
    """
    if not epoch_summaries:
        raise ValueError("epoch_summaries is empty")
    ranked = sorted(
        epoch_summaries,
        key=lambda x: (-x["support_aware_combined_score"], x["epoch"]),
    )
    return dict(ranked[0])
