"""Exact asset provenance checks for the immutable P26 parent."""
from __future__ import annotations

import hashlib
from pathlib import Path


P26_PHASE2B_CHECKPOINT_SHA256 = "a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34"
P26_CLIP_ASSET_SHA256 = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
P26_RUNTIME_CONFIG_SHA256 = "edf5745686e3d3d0d3b4142341569da06ad5b54025a779b78d83f74303ce67fc"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require_digest(path: Path, expected: str) -> str:
    """Return an exact digest or refuse a substituted frozen artifact."""
    observed = digest(path)
    if observed != expected:
        raise RuntimeError(f"P26 frozen asset digest mismatch for {path}: expected {expected}, got {observed}")
    return observed


def verify_p26_parent(checkpoint: Path, clip_asset: Path, config: Path) -> dict[str, str]:
    """Check every P27-consumed P26 runtime asset against P26's manifest."""
    return {
        "checkpoint": require_digest(checkpoint, P26_PHASE2B_CHECKPOINT_SHA256),
        "clip_asset": require_digest(clip_asset, P26_CLIP_ASSET_SHA256),
        "config": require_digest(config, P26_RUNTIME_CONFIG_SHA256),
    }
