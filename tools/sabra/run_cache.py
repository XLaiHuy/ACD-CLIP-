"""Entry point that binds the axis-checked core to the GT-free cache builder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from sabra import cache_runner  # noqa: E402
from sabra import logic_core_fixed  # noqa: E402

cache_runner.compute_relational_scores = logic_core_fixed.compute_relational_scores


if __name__ == "__main__":
    manifest = cache_runner.build_cache()
    print(json.dumps({
        "checkpoint": "GT_FREE_CACHE_FINALIZED",
        "manifest": str(cache_runner.AUDIT_ROOT / "GT_FREE_CACHE_MANIFEST.json"),
        "head": manifest.get("created_at_head"),
    }, sort_keys=True))
