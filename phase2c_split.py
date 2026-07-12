"""Deterministic, group-aware VisA train/validation split construction."""
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from dataset import deterministic_sample_id


MANIFEST_FIELDS = [
    "sample_id", "group_id", "image_path", "mask_path", "label",
    "class_name", "defect_type", "grouping_method",
]
EXPLICIT_GROUP_FIELDS = ("group_id", "series_id", "object_id", "instance_id", "sequence_id")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_records(source_path):
    records = []
    with open(source_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "image_path" not in row or "class_name" not in row or "label" not in row:
                raise ValueError(f"Invalid record at {source_path}:{line_number}")
            row = dict(row)
            row["label"] = int(row["label"])
            row["sample_id"] = deterministic_sample_id(row)
            parts = Path(row["image_path"]).parts
            try:
                image_pos = [part.lower() for part in parts].index("images")
                row["defect_type"] = parts[image_pos + 1] if image_pos + 1 < len(parts) - 1 else (
                    "anomaly" if row["label"] else "normal"
                )
            except ValueError:
                row["defect_type"] = "anomaly" if row["label"] else "normal"
            records.append(row)
    ids = [row["sample_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Source metadata contains duplicate image paths/sample IDs")
    return records


def _dhash(image_path):
    with Image.open(image_path) as image:
        pixels = list(image.convert("L").resize((17, 16), Image.Resampling.LANCZOS).getdata())
    value = 0
    for y in range(16):
        for x in range(16):
            value = (value << 1) | int(pixels[y * 17 + x] > pixels[y * 17 + x + 1])
    return value


def _hamming(left, right):
    return (left ^ right).bit_count()


def _path_series_id(record):
    stem = Path(record["image_path"]).stem
    base = re.sub(r"(?i)(?:[-_](?:view|camera|cam|angle|shot))[-_]?\d+$", "", stem)
    if base != stem:
        return f'{record["class_name"]}:series:{base}'
    return None


def assign_groups(records, data_root=None, phash_distance=4):
    """Assign explicit/path groups first, then near-duplicate perceptual-hash groups."""
    unresolved = []
    for row in records:
        explicit = next((row.get(key) for key in EXPLICIT_GROUP_FIELDS if row.get(key) not in (None, "")), None)
        if explicit is not None:
            row["group_id"] = f'{row["class_name"]}:metadata:{explicit}'
            row["grouping_method"] = "metadata"
            continue
        series = _path_series_id(row)
        if series:
            row["group_id"] = series
            row["grouping_method"] = "path_series"
            continue
        unresolved.append(row)

    by_stratum = defaultdict(list)
    for row in unresolved:
        by_stratum[(row["class_name"], row["label"])].append(row)

    root = Path(data_root) if data_root else None
    for rows in by_stratum.values():
        representatives = []
        for row in sorted(rows, key=lambda item: item["sample_id"]):
            image_path = root / row["image_path"] if root else None
            image_hash = None
            if image_path is not None and image_path.is_file():
                try:
                    image_hash = _dhash(image_path)
                except (OSError, ValueError):
                    image_hash = None
            match = None
            if image_hash is not None:
                for representative_hash, group_id in representatives:
                    if _hamming(image_hash, representative_hash) <= phash_distance:
                        match = group_id
                        break
            if match is None:
                match = f'phash:{row["sample_id"]}' if image_hash is not None else f'unique:{row["sample_id"]}'
                if image_hash is not None:
                    representatives.append((image_hash, match))
            row["group_id"] = match
            row["grouping_method"] = "perceptual_hash" if image_hash is not None else "unique_path_fallback"
    return records


def _stable_rank(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def split_records(records, train_ratio=0.8, seed=42):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between zero and one")
    groups = defaultdict(list)
    for row in records:
        groups[row["group_id"]].append(row)

    strata = defaultdict(list)
    for group_id, rows in groups.items():
        key = (rows[0]["class_name"], rows[0]["label"], rows[0]["defect_type"])
        strata[key].append((group_id, rows))

    val_group_ids = set()
    for key in sorted(strata):
        candidates = sorted(strata[key], key=lambda item: _stable_rank(seed, item[0]))
        total = sum(len(rows) for _, rows in candidates)
        target = total * (1.0 - train_ratio)
        chosen = 0
        for group_id, rows in candidates:
            size = len(rows)
            if abs((chosen + size) - target) < abs(chosen - target):
                val_group_ids.add(group_id)
                chosen += size
        if len(candidates) >= 2 and target > 0 and not any(gid in val_group_ids for gid, _ in candidates):
            val_group_ids.add(candidates[0][0])
        if len(candidates) >= 2 and all(gid in val_group_ids for gid, _ in candidates):
            val_group_ids.remove(candidates[-1][0])

    train = [row for row in records if row["group_id"] not in val_group_ids]
    val = [row for row in records if row["group_id"] in val_group_ids]
    key = lambda row: (row["class_name"], row["label"], row["image_path"])
    return sorted(train, key=key), sorted(val, key=key)


def _counts(records):
    by_category = defaultdict(lambda: {"total": 0, "normal": 0, "anomaly": 0})
    for row in records:
        item = by_category[row["class_name"]]
        item["total"] += 1
        item["anomaly" if row["label"] else "normal"] += 1
    return {
        "total": len(records),
        "groups": len({row["group_id"] for row in records}),
        "by_category": dict(sorted(by_category.items())),
    }


def write_manifest(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
    return rows


def validate_integrity(source_records, train, val):
    source_ids = {row["sample_id"] for row in source_records}
    train_ids = {row["sample_id"] for row in train}
    val_ids = {row["sample_id"] for row in val}
    if train_ids & val_ids:
        raise AssertionError("Train/validation sample overlap")
    if train_ids | val_ids != source_ids:
        raise AssertionError("Manifest union does not equal eligible source set")
    if {row["group_id"] for row in train} & {row["group_id"] for row in val}:
        raise AssertionError("Train/validation group overlap")
    source_coverage = {(row["class_name"], row["label"]) for row in source_records}
    feasible = {
        key for key in source_coverage
        if len({row["group_id"] for row in source_records if (row["class_name"], row["label"]) == key}) >= 2
    }
    val_coverage = {(row["class_name"], int(row["label"])) for row in val}
    missing = sorted(feasible - val_coverage)
    if missing:
        raise AssertionError(f"Validation label coverage missing for feasible strata: {missing}")
    return True


def build_split(source_path, output_dir, data_root=None, seed=42, train_ratio=0.8, phash_distance=4):
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    records = assign_groups(load_source_records(source_path), data_root, phash_distance)
    train, val = split_records(records, train_ratio, seed)
    validate_integrity(records, train, val)
    train_path = output_dir / f"visa_train_seed{seed}.csv"
    val_path = output_dir / f"visa_val_seed{seed}.csv"
    metadata_path = output_dir / f"visa_split_seed{seed}_metadata.json"
    write_manifest(train_path, train)
    write_manifest(val_path, val)
    methods = defaultdict(int)
    for row in records:
        methods[row["grouping_method"]] += 1
    metadata = {
        "version": 1,
        "seed": seed,
        "train_ratio": train_ratio,
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "grouping_priority": ["metadata", "path_series", "perceptual_hash", "unique_path_fallback"],
        "perceptual_hash": {"algorithm": "dhash256", "hamming_distance": phash_distance},
        "grouping_method_counts": dict(sorted(methods.items())),
        "source": _counts(records),
        "train": _counts(train),
        "validation": _counts(val),
        "manifests": {"train": str(train_path), "validation": str(val_path)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return train_path, val_path, metadata_path


def verify_split(source_path, train_path, val_path, metadata_path):
    source = load_source_records(source_path)
    train = read_manifest(train_path)
    val = read_manifest(val_path)
    manifest_by_id = {row["sample_id"]: row for row in train + val}
    for row in source:
        persisted = manifest_by_id.get(row["sample_id"])
        if persisted is not None:
            row["group_id"] = persisted["group_id"]
    validate_integrity(source, train, val)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if metadata["source_sha256"] != file_sha256(source_path):
        raise AssertionError("Source hash differs from split metadata")
    if metadata["train"]["total"] != len(train) or metadata["validation"]["total"] != len(val):
        raise AssertionError("Manifest counts differ from split metadata")
    return True
