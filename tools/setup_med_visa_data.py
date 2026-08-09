#!/usr/bin/env python3
"""Discover, link, and verify Med-VISA data; downloading is explicit only."""
from __future__ import annotations
import argparse, json, os, shutil, sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.info import CLASS_NAMES

CANONICAL_PARTS = tuple(map(Path, (
    "VisA_20220922", "MedAD/Brain_AD", "MedAD/Liver_AD", "MedAD/Retina_RESC_AD",
    "Colon/CVC-ClinicDB", "Colon/CVC-ColonDB", "Colon/Kvasir")))
TOP_LEVEL = ("VisA_20220922", "MedAD", "Colon")
MANIFEST_ROOTS = {
    "VisA": Path("VisA_20220922"), "Brain": Path("MedAD/Brain_AD/test"),
    "Liver": Path("MedAD/Liver_AD/test"), "Retina": Path("MedAD/Retina_RESC_AD/test"),
    "Colon_clinicDB": Path("Colon/CVC-ClinicDB"),
    "Colon_colonDB": Path("Colon/CVC-ColonDB"), "Colon_Kvasir": Path("Colon/Kvasir")}
EXPECTED_LABEL_COUNTS = {
    "Brain": {0: 640, 1: 3075},
    "Liver": {0: 833, 1: 660},
    "Retina": {0: 1041, 1: 764},
    "Colon_clinicDB": {0: 0, 1: 612},
    "Colon_colonDB": {0: 0, 1: 380},
    "Colon_Kvasir": {0: 0, 1: 1000},
}


def is_complete_root(path: Path) -> bool:
    return path.is_dir() and all((path / part).is_dir() for part in CANONICAL_PARTS)


def discover_candidate_roots(source_root: Path, max_depth: int = 5) -> list[Path]:
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    candidates = []
    for current, dirs, _files in os.walk(source_root):
        path = Path(current)
        depth = len(path.relative_to(source_root).parts)
        if is_complete_root(path):
            candidates.append(path.resolve()); dirs[:] = []
        elif depth >= max_depth:
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if d not in {"Images", "Masks", "images", "masks", "img", "gt"}]
    return sorted(set(candidates))


def select_candidate_root(source_root: Path) -> Path:
    candidates = discover_candidate_roots(source_root)
    if not candidates:
        raise FileNotFoundError(f"no complete Med-VISA tree found under {source_root}")
    if len(candidates) != 1:
        joined = "\n  - ".join(map(str, candidates))
        raise RuntimeError(f"ambiguous dataset roots; provide a narrower --source-root:\n  - {joined}")
    return candidates[0]


def _remove_force_target(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def materialize_layout(source: Path, data_root: Path, link_mode: str, force: bool = False) -> None:
    source, data_root = source.resolve(), data_root.expanduser().resolve()
    if source == data_root:
        return
    data_root.mkdir(parents=True, exist_ok=True)
    mode = "symlink" if link_mode == "auto" else link_mode
    for name in TOP_LEVEL:
        src, dst = source / name, data_root / name
        if dst.exists() or dst.is_symlink():
            if dst.resolve() == src.resolve():
                continue
            if not force:
                raise FileExistsError(f"target exists and differs: {dst}; use --force explicitly")
            _remove_force_target(dst)
        if mode == "symlink":
            dst.symlink_to(src, target_is_directory=True)
        elif mode == "copy":
            shutil.copytree(src, dst, symlinks=True)
        else:
            raise ValueError(f"unsupported link mode: {link_mode}")


def verify_manifests(data_root: Path, manifest_dir: Path | None = None) -> dict:
    data_root = data_root.resolve()
    manifest_dir = (manifest_dir or REPO_ROOT / "dataset" / "hub").resolve()
    report = {
        "data_root": str(data_root),
        "datasets": {},
        "missing_images": [],
        "missing_masks": [],
        "invalid_class_names": [],
        "count_mismatches": [],
    }
    for dataset, relative_root in MANIFEST_ROOTS.items():
        manifest = manifest_dir / f"{dataset}.jsonl"
        counts, classes = Counter(), Counter()
        missing_images, missing_masks, invalid_class_names = [], [], []
        expected_classes = set(CLASS_NAMES.get(dataset, ()))
        if not expected_classes:
            raise ValueError(f"dataset/info.py has no authoritative CLASS_NAMES entry for {dataset}")
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                label, class_name = record.get("label"), record.get("class_name")
                if not isinstance(label, int) or isinstance(label, bool) or label not in (0, 1):
                    raise ValueError(f"{manifest}:{line_number}: label must be 0 or 1")
                if class_name not in expected_classes:
                    invalid_class_names.append(
                        f"{manifest}:{line_number}: {class_name!r} not in {sorted(expected_classes)!r}"
                    )
                counts[int(label)] += 1; classes[class_name] += 1
                sample_root = data_root / relative_root
                image = sample_root / record["image_path"]
                if not image.is_file():
                    missing_images.append(str(image))
                mask_path = record.get("mask_path")
                if label == 1:
                    if not isinstance(mask_path, str) or not mask_path.strip():
                        missing_masks.append(f"{manifest}:{line_number}: anomaly mask_path missing")
                    else:
                        mask = sample_root / mask_path
                        if not mask.is_file():
                            missing_masks.append(str(mask))
        actual_counts = {0: counts.get(0, 0), 1: counts.get(1, 0)}
        expected_counts = EXPECTED_LABEL_COUNTS.get(dataset)
        count_match = expected_counts is None or actual_counts == expected_counts
        if not count_match:
            report["count_mismatches"].append({
                "dataset": dataset, "expected": expected_counts, "actual": actual_counts,
            })
        report["datasets"][dataset] = {
            "total": sum(counts.values()),
            "counts_by_label": dict(sorted(counts.items())),
            "counts_by_class": dict(sorted(classes.items())),
            "expected_counts_by_label": expected_counts,
            "count_match": count_match,
            "missing_image_count": len(missing_images),
            "missing_mask_count": len(missing_masks),
            "invalid_class_name_count": len(invalid_class_names),
        }
        report["missing_images"].extend(missing_images)
        report["missing_masks"].extend(missing_masks)
        report["invalid_class_names"].extend(invalid_class_names)
    report["missing"] = report["missing_images"] + report["missing_masks"]
    report["missing_count"] = len(report["missing"])
    report["ok"] = not any((
        report["missing"], report["invalid_class_names"], report["count_mismatches"],
    ))
    return report


def _download_with_kagglehub() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("--download requires kagglehub to be installed") from exc
    return Path(kagglehub.dataset_download("huyzzz2109/med-visa"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--link-mode", choices=["auto", "symlink", "copy"], default="auto")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download", action="store_true", help="Explicit KaggleHub portability fallback")
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.expanduser().resolve()
    if is_complete_root(data_root):
        selected = data_root
    else:
        source = args.source_root
        if source is None and args.download:
            source = _download_with_kagglehub()
        if source is None:
            print(json.dumps({"status": "DEFERRED_NO_LOCAL_DATA", "data_root": str(data_root),
                "message": "provide --source-root; no download was attempted"}, indent=2))
            return 2
        selected = select_candidate_root(source)
        materialize_layout(selected, data_root, args.link_mode, args.force)
        selected = data_root
    payload = {"status": "READY" if is_complete_root(data_root) else "INCOMPLETE",
        "source_root": str(selected), "data_root": str(data_root),
        "layout_complete": is_complete_root(data_root)}
    if args.verify:
        payload["verification"] = verify_manifests(data_root)
        if not payload["verification"]["ok"]:
            payload["status"] = "VERIFY_FAILED"
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["status"] == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
