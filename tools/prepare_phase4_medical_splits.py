#!/usr/bin/env python3
"""Create reproducible Phase4 medical validation/test manifests.

The model always trains on VisA.  Brain, Liver and Retina use their provider
validation directories.  The Colon datasets have one labelled pool each, so
they are split once per run, stratified by image label, with image/mask pairs
kept in the same manifest row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
COLON_DATASETS = ("Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir")
OFFICIAL_VALIDATION = {
    "Brain": ("Brain_AD", "valid", "Brain"),
    "Liver": ("Liver_AD", "valid", "Liver"),
    "Retina": ("Retina_RESC_AD", "val", "Retina"),
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stratified_split_rows(
    rows: list[dict[str, Any]], val_ratio: float, seed: int, dataset_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by image label, never separating a row's image from its mask."""
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)

    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for label, label_rows in sorted(by_label.items()):
        shuffled = list(label_rows)
        random.Random(f"phase4-medical-split:{seed}:{dataset_name}:{label}").shuffle(shuffled)
        if len(shuffled) < 2:
            val_count = 0
        else:
            val_count = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_ratio)))
        val_rows.extend(shuffled[:val_count])
        test_rows.extend(shuffled[val_count:])

    val_rows.sort(key=lambda row: row["image_path"])
    test_rows.sort(key=lambda row: row["image_path"])
    if not val_rows or not test_rows:
        raise ValueError(f"{dataset_name}: split produced an empty validation or test set")
    return val_rows, test_rows


def official_validation_rows(data_root: Path, dataset: str) -> list[dict[str, Any]]:
    subdir, split_dir, class_name = OFFICIAL_VALIDATION[dataset]
    split_root = data_root / "MedAD" / subdir / split_dir
    if not split_root.is_dir():
        raise FileNotFoundError(f"official validation directory not found: {split_root}")

    rows: list[dict[str, Any]] = []
    for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        label = 0 if class_dir.name.casefold() == "good" else 1
        image_dir = class_dir / "img"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(path for path in image_dir.rglob("*") if path.suffix.casefold() in IMAGE_SUFFIXES):
            image_rel = image_path.relative_to(split_root).as_posix()
            row: dict[str, Any] = {
                "image_path": image_rel,
                "label": label,
                "class_name": class_name,
            }
            if label:
                mask_path = class_dir / "label" / image_path.relative_to(image_dir)
                if not mask_path.is_file():
                    raise FileNotFoundError(f"missing mask paired with {image_path}: {mask_path}")
                row["mask_path"] = mask_path.relative_to(split_root).as_posix()
            rows.append(row)
    rows.sort(key=lambda row: row["image_path"])
    if not rows:
        raise ValueError(f"{dataset}: no official validation images found in {split_root}")
    return rows


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(int(row["label"]))] += 1
    return dict(sorted(counts.items()))


def prepare(output_root: Path, data_root: Path, val_ratio: float, seed: int) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "protocol": "phase4_medical_val_test_v1",
        "seed": seed,
        "colon_val_ratio": val_ratio,
        "medical_training_data": "none; VisA only",
        "datasets": {},
    }

    for dataset in OFFICIAL_VALIDATION:
        rows = official_validation_rows(data_root, dataset)
        write_jsonl(output_root / f"{dataset}_val.jsonl", rows)
        split_dir = OFFICIAL_VALIDATION[dataset][1]
        manifest["datasets"][dataset] = {
            "val_source": f"official MedAD {split_dir}",
            "val_count": len(rows),
            "val_label_counts": label_counts(rows),
            "test_source": "dataset/hub/<dataset>.jsonl + official MedAD test directory",
        }

    for dataset in COLON_DATASETS:
        source_path = REPO_ROOT / "dataset" / "hub" / f"{dataset}.jsonl"
        source_rows = read_jsonl(source_path)
        val_rows, test_rows = stratified_split_rows(source_rows, val_ratio, seed, dataset)
        write_jsonl(output_root / f"{dataset}_val.jsonl", val_rows)
        write_jsonl(output_root / f"{dataset}_test.jsonl", test_rows)
        manifest["datasets"][dataset] = {
            "val_source": "deterministic stratified image-level split",
            "source_manifest": str(source_path.relative_to(REPO_ROOT)),
            "source_sha256": sha256_file(source_path),
            "source_count": len(source_rows),
            "val_count": len(val_rows),
            "test_count": len(test_rows),
            "val_label_counts": label_counts(val_rows),
            "test_label_counts": label_counts(test_rows),
            "grouping": "image-level; source manifests expose no patient/video ID",
        }

    (output_root / "medical_protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic Phase4 medical validation/test manifests")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT.parent / "data")
    parser.add_argument("--val-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    manifest = prepare(args.output_root, args.data_root, args.val_ratio, args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
