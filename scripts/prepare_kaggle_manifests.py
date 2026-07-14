#!/usr/bin/env python3
"""Generate runtime Kaggle manifests without modifying the original manifests.

Rewrites path-like CSV columns by replacing a specified old root prefix with a
new root prefix.  Every non-path field, row order, sample IDs, labels,
grouping, and class names are preserved exactly.

Usage
-----
python scripts/prepare_kaggle_manifests.py \\
    --train-manifest splits/visa_train_seed42.csv \\
    --val-manifest   splits/visa_val_seed42.csv \\
    --old-root       /home/ai4/data/VisA \\
    --new-root       /kaggle/input/visa-dataset \\
    --output-dir     /kaggle/working/runtime_splits

Suggested Kaggle outputs
------------------------
  /kaggle/working/runtime_splits/visa_train_seed42_kaggle.csv
  /kaggle/working/runtime_splits/visa_val_seed42_kaggle.csv
"""
import argparse
import csv
import sys
from pathlib import Path


def _is_path_value(value: str, old_root: str) -> bool:
    """Return True if the cell value looks like a path under old_root.

    Detection is conservative: a cell is considered a path when it starts
    with the given old_root prefix or when it contains a slash or backslash
    and the cell is non-empty.
    """
    if not value:
        return False
    if value.startswith(old_root):
        return True
    return False


def rewrite_manifest(
    src_path: Path,
    dst_path: Path,
    old_root: str,
    new_root: str,
) -> dict:
    """Read src_path, rewrite path cells, write dst_path.

    Returns a summary dict with row_count, path_cells_found,
    existing_count, missing_count.
    """
    if not src_path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {src_path}")
    if dst_path.resolve() == src_path.resolve():
        raise ValueError(
            "Destination and source are the same file. "
            "Overwriting source manifests is not allowed."
        )

    with src_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or header-less CSV: {src_path}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    rewritten_rows = []
    path_cells_found = 0
    existing_count = 0
    missing_count = 0
    sample_paths: list[str] = []

    for row in rows:
        new_row = {}
        for field, value in row.items():
            if _is_path_value(value, old_root):
                new_value = new_root + value[len(old_root):]
                path_cells_found += 1
                resolved = Path(new_value)
                if resolved.exists():
                    existing_count += 1
                else:
                    missing_count += 1
                if len(sample_paths) < 5:
                    sample_paths.append(f"  {value!r}  ->  {new_value!r}")
                new_row[field] = new_value
            else:
                new_row[field] = value
        rewritten_rows.append(new_row)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rewritten_rows)

    return {
        "row_count": len(rows),
        "path_cells_found": path_cells_found,
        "existing_count": existing_count,
        "missing_count": missing_count,
        "sample_paths": sample_paths,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite manifest paths for a different filesystem root.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--train-manifest", required=True, help="Source train CSV")
    parser.add_argument("--val-manifest", required=True, help="Source val CSV")
    parser.add_argument(
        "--old-root", required=True,
        help="Path prefix in source manifests to replace",
    )
    parser.add_argument(
        "--new-root", required=True,
        help="Replacement path prefix in output manifests",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for generated manifests",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_src = Path(args.train_manifest)
    val_src = Path(args.val_manifest)

    # Derive output filenames by inserting '_kaggle' before the extension
    def kaggle_name(src: Path) -> Path:
        return output_dir / (src.stem + "_kaggle" + src.suffix)

    train_dst = kaggle_name(train_src)
    val_dst = kaggle_name(val_src)

    had_missing = False

    for src, dst, label in [
        (train_src, train_dst, "train"),
        (val_src, val_dst, "val"),
    ]:
        print(f"\n── {label} manifest ──────────────────────────")
        summary = rewrite_manifest(src, dst, args.old_root, args.new_root)
        print(f"  Source     : {src}")
        print(f"  Destination: {dst}")
        print(f"  Rows       : {summary['row_count']}")
        print(f"  Path cells : {summary['path_cells_found']}")
        print(f"  Existing   : {summary['existing_count']}")
        print(f"  Missing    : {summary['missing_count']}")
        if summary["sample_paths"]:
            print("  Sample path rewrites:")
            for line in summary["sample_paths"]:
                print(line)
        if summary["missing_count"] > 0:
            print(f"  WARNING: {summary['missing_count']} paths do not exist at the new root.")
            had_missing = True

    if had_missing:
        print(
            "\nERROR: Some required paths are missing at the new root. "
            "Verify that the dataset is attached and --new-root is correct.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nManifest generation complete.  Original files are unchanged.")


if __name__ == "__main__":
    main()
