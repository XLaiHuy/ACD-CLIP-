#!/usr/bin/env python3
"""Kaggle preflight check for Phase2C P_LoRA_only.

Validates the environment before launching training.  Designed to run both
on Kaggle and locally (where /kaggle/input may not exist).

Usage
-----
python scripts/kaggle_preflight.py \\
    --train-manifest /kaggle/working/runtime_splits/visa_train_seed42_kaggle.csv \\
    --val-manifest   /kaggle/working/runtime_splits/visa_val_seed42_kaggle.csv \\
    --split-metadata splits/visa_split_seed42_metadata.json \\
    --output-dir     /kaggle/working/runs/phase2c_kaggle/PL_lora_only_seed42
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


OK = "  [OK]"
WARN = "  [WARN]"
FAIL = "  [FAIL]"
SECTION = "\n──"


def _section(title: str) -> None:
    print(f"{SECTION} {title} {'─' * max(0, 50 - len(title))}")


def _check_python_version() -> None:
    _section("Python version")
    info = sys.version.replace("\n", " ")
    print(f"{OK}  {info}")
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 9):
        print(f"{WARN}  Python >= 3.9 recommended")


def _check_torch() -> None:
    _section("PyTorch and CUDA")
    try:
        import torch

        print(f"{OK}  torch {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"{'  [OK]' if cuda_avail else WARN}  CUDA available: {cuda_avail}")
        if cuda_avail:
            device_count = torch.cuda.device_count()
            print(f"{OK}  Visible GPU count: {device_count}")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    mem_str = f"  free={free / 1e9:.1f} GB / total={total / 1e9:.1f} GB"
                except Exception:
                    mem_str = "  (memory info unavailable)"
                print(f"{OK}    GPU {i}: {name}{mem_str}")
            # Note: is_bf16_supported only checks if PyTorch accepts BF16.
            # It does NOT guarantee native Tensor Core BF16 support.
            bf16_ok = torch.cuda.is_bf16_supported()
            note = (
                "(PyTorch accepts BF16 execution on this device; "
                "does NOT imply native Tensor Core BF16 support)"
            )
            print(f"{'  [OK]' if bf16_ok else WARN}  BF16 accepted by PyTorch: {bf16_ok}  {note}")
        else:
            print(f"{WARN}  No CUDA device found; BF16 training will fail")
    except ImportError:
        print(f"{FAIL}  PyTorch is not installed", file=sys.stderr)
        sys.exit(1)


def _check_git(repo_root: Path) -> None:
    _section("Repository")
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo_root, text=True
        ).strip()
        print(f"{OK}  Branch : {branch}")
        print(f"{OK}  Commit : {sha}")
    except Exception as exc:
        print(f"{WARN}  Could not read git metadata: {exc}")


def _check_source_files(repo_root: Path) -> bool:
    _section("Required source files")
    required = [
        "phase2c_train.py",
        "phase2c_pcgrad.py",
        "phase2c_pcgrad_diagnostics.py",
        "phase2c_utils.py",
        "phase2c_split.py",
        "phase2c_analyze_ac.py",
    ]
    all_ok = True
    for fname in required:
        path = repo_root / fname
        if path.is_file():
            print(f"{OK}  {fname}")
        else:
            print(f"{FAIL}  {fname}  (missing)", file=sys.stderr)
            all_ok = False
    return all_ok


def _check_manifests(train_manifest: Path, val_manifest: Path, split_metadata: Path) -> bool:
    _section("Manifests and split metadata")
    all_ok = True
    for label, path in [
        ("train manifest", train_manifest),
        ("val manifest", val_manifest),
        ("split metadata", split_metadata),
    ]:
        if path.is_file():
            print(f"{OK}  {label}: {path}")
        else:
            print(f"{FAIL}  {label} MISSING: {path}", file=sys.stderr)
            all_ok = False
    return all_ok


def _check_manifest_paths(manifest: Path, label: str, max_check: int = 100) -> bool:
    """Spot-check that referenced files exist."""
    _section(f"Referenced file existence ({label}, first {max_check} rows)")
    if not manifest.is_file():
        print(f"{FAIL}  Manifest not found: {manifest}", file=sys.stderr)
        return False
    missing = []
    checked = 0
    with manifest.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if checked >= max_check:
                break
            for field, value in row.items():
                if value and ("/" in value or "\\" in value) and not value.startswith("phash:"):
                    p = Path(value)
                    if not p.is_file():
                        missing.append(str(p))
                    checked += 1
    if missing:
        print(f"{WARN}  {len(missing)} path(s) not found (showing up to 5):")
        for m in missing[:5]:
            print(f"        {m}")
        return False
    print(f"{OK}  All {checked} checked paths exist")
    return True


def _check_output_dir(output_dir: Path) -> bool:
    _section("Output directory")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".preflight_probe"
        probe.write_text("ok")
        probe.unlink()
        print(f"{OK}  Writable: {output_dir}")
    except Exception as exc:
        print(f"{FAIL}  Cannot write to output dir {output_dir}: {exc}", file=sys.stderr)
        return False
    return True


def _check_disk_space(path: Path, min_gb: float = 20.0) -> None:
    _section("Disk space")
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / 1e9
        total_gb = usage.total / 1e9
        status = OK if free_gb >= min_gb else WARN
        print(f"{status}  Free {free_gb:.1f} GB / Total {total_gb:.1f} GB at {path}")
        if free_gb < min_gb:
            print(f"{WARN}  At least {min_gb} GB free recommended for checkpoints + logs")
    except Exception as exc:
        print(f"{WARN}  Disk check failed: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Preflight check for Phase2C P_LoRA_only on Kaggle or local."
    )
    parser.add_argument("--train-manifest", default="splits/visa_train_seed42.csv")
    parser.add_argument("--val-manifest", default="splits/visa_val_seed42.csv")
    parser.add_argument("--split-metadata", default="splits/visa_split_seed42_metadata.json")
    parser.add_argument(
        "--output-dir",
        default="runs/phase2c_bf16/PL_lora_only_seed42",
        help="Intended run output directory (will be created if absent)",
    )
    parser.add_argument(
        "--repo-root", default=".",
        help="Path to the repository root (default: current directory)",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    train_manifest = Path(args.train_manifest)
    val_manifest = Path(args.val_manifest)
    split_metadata = Path(args.split_metadata)
    output_dir = Path(args.output_dir)

    print("Phase2C P_LoRA_only — Preflight Check")
    print("=" * 60)

    _check_python_version()
    _check_torch()
    _check_git(repo_root)
    src_ok = _check_source_files(repo_root)
    mf_ok = _check_manifests(train_manifest, val_manifest, split_metadata)
    _check_manifest_paths(train_manifest, "train")
    _check_manifest_paths(val_manifest, "val")
    out_ok = _check_output_dir(output_dir)
    _check_disk_space(output_dir.parent if output_dir.parent.exists() else Path("."))

    print("\n" + "=" * 60)
    all_ok = src_ok and mf_ok and out_ok
    if all_ok:
        print("Preflight PASSED.  Proceed to dry-run.")
    else:
        print("Preflight FAILED.  Resolve the issues above before running.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
