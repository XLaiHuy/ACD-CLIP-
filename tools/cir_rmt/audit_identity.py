#!/usr/bin/env python3
"""CIR/G0-IDENTITY: hard-fail repository/config/environment identity audit."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .identity import assert_g0, load_cir_config

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cir_dfg_rmt_v1.json"))
    parser.add_argument("--allow-dirty", action="store_true", help="development-only; still reports all dirty paths")
    parser.add_argument("--require-assets", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = assert_g0(allow_dirty=args.allow_dirty, config_path=args.config)
    cfg = load_cir_config(args.config)
    roots = {"visa_root": cfg.get("visa_root") or os.environ.get("VISA_ROOT") or os.environ.get("ACDCLIP_VISA_ROOT"), "mvtec_root": cfg.get("mvtec_root") or os.environ.get("MVTEC_ROOT") or os.environ.get("ACDCLIP_MVTEC_ROOT"), "medical_root": cfg.get("medical_root") or os.environ.get("MEDICAL_ROOT") or os.environ.get("ACDCLIP_DATA_ROOT")}
    result["dataset_roots"] = roots
    if args.require_assets:
        missing = [name for name, value in roots.items() if not value or not Path(str(value)).expanduser().exists()]
        if missing:
            raise RuntimeError(f"G0 dataset roots unavailable: {missing}")
    payload = json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
