#!/usr/bin/env python3
"""CIR/REPORT: one-shot status inspection for a CIR run root."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any

def inspect_run(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"stage": "CIR/REPORT", "run_root": str(root), "exists": root.is_dir(), "identity": {}, "checkpoints": [], "result_files": []}
    if not root.is_dir():
        result["status"] = "NOT_STARTED"
        return result
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if path.name.endswith((".pth", ".pt")):
            result["checkpoints"].append({"path": rel, "bytes": path.stat().st_size})
        if path.name in {"metrics.json", "results.csv", "summary.csv", "run_manifest.json", "status.json"}:
            result["result_files"].append(rel)
        if path.name in {"config_resolved.json", "identity.json"}:
            try:
                result["identity"][rel] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                result["identity"][rel] = {"parse_error": True}
    result["status"] = "COMPLETE" if (root / "COMPLETE.json").is_file() else "RUNNING_OR_PARTIAL"
    return result

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/cir_rmt/CIR_DFG_RMT_V1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = json.dumps(inspect_run(args.run_root), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
