"""Portable structural preflight; no recursive dataset discovery."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
METADATA = {"visa": ROOT / "dataset/hub/VisA.jsonl", "mvtec": ROOT / "dataset/hub/MVTec.jsonl"}

def _safe_child(root: Path, relative: str) -> Path:
    base = root.expanduser().resolve()
    child = (base / relative).resolve()
    if child != base and base not in child.parents:
        raise RuntimeError(f"path escapes configured root: {relative}")
    return child

def structural_preflight(dataset: str, configured_root: Path, *, allow_medical: bool = False) -> dict[str, Any]:
    dataset = dataset.lower()
    if dataset == "medical":
        if not allow_medical:
            raise RuntimeError("MEDICAL_SEALED: pass --allow-medical-evaluation only after final checkpoint freeze")
        root = configured_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        return {"status": "ROOT_ONLY_PASS", "dataset": "medical", "root": str(root), "recursive_discovery": False}
    if dataset not in METADATA:
        raise ValueError(f"unsupported dataset: {dataset}")
    root = configured_root.expanduser().resolve()
    if dataset == "mvtec" and not (root / "bottle").exists() and (root / "mvtec_anomaly_detection").exists():
        root = (root / "mvtec_anomaly_detection").resolve()
    metadata = METADATA[dataset]
    if not root.is_dir() or not metadata.is_file():
        raise FileNotFoundError(f"root or metadata missing: {root}, {metadata}")
    rows = []
    missing = []
    for line_number, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        for key in ("image_path", "mask_path"):
            relative = row.get(key)
            if relative and not _safe_child(root, str(relative)).is_file():
                missing.append(f"{line_number}:{key}:{relative}")
    if missing:
        raise FileNotFoundError("missing metadata references: " + ", ".join(missing[:10]))
    return {"status": "PASS", "dataset": dataset, "root": str(root), "metadata": str(metadata.relative_to(ROOT)), "metadata_rows": len(rows), "recursive_discovery": False, "checked_fields": ["image_path", "mask_path"]}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["visa", "mvtec", "medical"], required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--allow-medical-evaluation", action="store_true")
    args = parser.parse_args()
    env_name = {"visa": "VISA_ROOT", "mvtec": "MVTEC_ROOT", "medical": "MEDICAL_ROOT"}[args.dataset]
    root = args.root or (Path(os.environ[env_name]) if os.environ.get(env_name) else None)
    if root is None:
        raise SystemExit(f"{env_name} or --root is required")
    print(json.dumps(structural_preflight(args.dataset, root, allow_medical=args.allow_medical_evaluation), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
