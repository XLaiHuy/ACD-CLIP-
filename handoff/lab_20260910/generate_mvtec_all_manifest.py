#!/usr/bin/env python3
"""Generate the explicit all-supervised MVTec source manifest.

The manifest contains train/good, test/good, and every labelled test anomaly
for every category found under the supplied official MVTec-AD tree. Paths in
the JSONL are relative to the MVTec root so the repo data symlink convention
continues to work on the laboratory machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda item: item.name,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_rows(root: Path) -> tuple[list[dict], dict]:
    if not root.is_dir():
        raise SystemExit(f"MVTec root does not exist: {root}")

    categories = sorted(
        item.name
        for item in root.iterdir()
        if item.is_dir() and (item / "train" / "good").is_dir() and (item / "test").is_dir()
    )
    if not categories:
        raise SystemExit(f"No MVTec category directories found under: {root}")

    rows: list[dict] = []
    counts = Counter()
    per_class: dict[str, Counter] = defaultdict(Counter)
    missing_images: list[str] = []
    missing_masks: list[str] = []
    seen: set[str] = set()
    duplicate_paths: list[str] = []

    def add_row(row: dict, image_path: Path) -> None:
        relative = row["image_path"]
        if relative in seen:
            duplicate_paths.append(relative)
        seen.add(relative)
        if not image_path.is_file():
            missing_images.append(relative)
        rows.append(row)

    for category in categories:
        category_root = root / category
        for split_name, split_dir in (
            ("train/good", category_root / "train" / "good"),
            ("test/good", category_root / "test" / "good"),
        ):
            for image_path in image_files(split_dir):
                relative = image_path.relative_to(root).as_posix()
                add_row(
                    {
                        "class_name": category,
                        "image_path": relative,
                        "label": 0,
                        "source_split": split_name,
                    },
                    image_path,
                )
                counts[split_name] += 1
                per_class[category][split_name] += 1

        test_root = category_root / "test"
        for anomaly_dir in sorted(
            item for item in test_root.iterdir() if item.is_dir() and item.name != "good"
        ):
            for image_path in image_files(anomaly_dir):
                mask_path = category_root / "ground_truth" / anomaly_dir.name / f"{image_path.stem}_mask.png"
                relative = image_path.relative_to(root).as_posix()
                mask_relative = mask_path.relative_to(root).as_posix()
                if not mask_path.is_file():
                    missing_masks.append(mask_relative)
                add_row(
                    {
                        "anomaly_type": anomaly_dir.name,
                        "class_name": category,
                        "image_path": relative,
                        "label": 1,
                        "mask_path": mask_relative,
                        "source_split": "test/anomaly",
                    },
                    image_path,
                )
                counts["test/anomaly"] += 1
                per_class[category]["test/anomaly"] += 1

    if missing_images or missing_masks or duplicate_paths:
        raise SystemExit(
            "MVTec manifest validation failed: missing_images=%d missing_masks=%d duplicates=%d"
            % (len(missing_images), len(missing_masks), len(duplicate_paths))
        )

    rows.sort(key=lambda row: row["image_path"])
    audit = {
        "category_count": len(categories),
        "categories": categories,
        "total_images": len(rows),
        "normal_count": counts["train/good"] + counts["test/good"],
        "anomaly_count": counts["test/anomaly"],
        "train_good_count": counts["train/good"],
        "test_good_count": counts["test/good"],
        "test_anomaly_count": counts["test/anomaly"],
        "missing_file_count": 0,
        "missing_mask_count": 0,
        "duplicate_count": 0,
        "per_class": {
            category: {
                "train/good": per_class[category]["train/good"],
                "test/good": per_class[category]["test/good"],
                "test/anomaly": per_class[category]["test/anomaly"],
                "total": sum(per_class[category].values()),
            }
            for category in categories
        },
    }
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    rows, audit = build_rows(root)
    manifest_text = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(manifest_text, encoding="utf-8")
    manifest_sha = sha256_bytes(manifest_text.encode("utf-8"))
    audit["manifest_sha256"] = manifest_sha
    audit["source_root"] = str(root)

    lines = [
        "# MVTec All Supervised Manifest Audit",
        "",
        f"Source root: `{root}`",
        f"Manifest: `{args.output}`",
        "",
        "## Verification",
        "",
        f"- Total images: {audit['total_images']}",
        f"- Normal count: {audit['normal_count']}",
        f"- Anomaly count: {audit['anomaly_count']}",
        f"- Train/good count: {audit['train_good_count']}",
        f"- Test/good count: {audit['test_good_count']}",
        f"- Test/anomaly count: {audit['test_anomaly_count']}",
        f"- Missing file count: {audit['missing_file_count']}",
        f"- Missing mask count: {audit['missing_mask_count']}",
        f"- Duplicate count: {audit['duplicate_count']}",
        f"- Manifest SHA256: `{manifest_sha}`",
        "",
        "## Per-class counts",
        "",
        "| Class | Train/good | Test/good | Test/anomaly | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in audit["categories"]:
        row = audit["per_class"][category]
        lines.append(
            f"| {category} | {row['train/good']} | {row['test/good']} | "
            f"{row['test/anomaly']} | {row['total']} |"
        )
    lines.extend(
        [
            "",
            "The manifest uses official MVTec ground-truth masks for every "
            "test anomaly. Normal rows intentionally omit `mask_path`.",
            "",
        ]
    )
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print("MVTec all manifest generated")
    print(f"total_images={audit['total_images']}")
    print(f"normal_count={audit['normal_count']}")
    print(f"anomaly_count={audit['anomaly_count']}")
    print(f"manifest_sha256={manifest_sha}")


if __name__ == "__main__":
    main()
