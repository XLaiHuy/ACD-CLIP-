#!/usr/bin/env python3
"""Hash compact PA artifacts and reproducibility scripts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--script", action="append", default=[])
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve()
    repo = args.repo_root.expanduser().resolve()
    paths = [path for path in archive.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt"]
    paths.extend((repo / value).resolve() for value in args.script)
    unique = sorted({path for path in paths if path.is_file()})
    lines = []
    for path in unique:
        try:
            label = path.relative_to(repo)
        except ValueError:
            label = path
        lines.append(f"{sha256(path)}  {label}")
    output = archive / "SHA256SUMS.txt"
    output.write_text("# SHA256SUMS excludes itself to avoid a self-referential digest.\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"hashed={len(unique)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
