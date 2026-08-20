"""Backend selection for the frozen Trust-v2 geometry evaluator.

The exact implementation remains the scientific reference. The fast backend
is selected only for post-training evaluator execution after certification.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def compact_record_builder(backend: str) -> Callable[..., tuple[dict[str, Any], dict[str, Any]]]:
    normalized = str(backend).lower()
    if normalized == "exact":
        from sabra.trust_v2.numerical import build_compact_record
        return build_compact_record
    if normalized == "fast":
        from sabra.trust_v2.fast_geometry import build_compact_record_fast
        return build_compact_record_fast
    raise ValueError(f"unknown Trust-v2 backend: {backend!r}; expected exact or fast")


def validate_backend(backend: str) -> str:
    compact_record_builder(backend)
    return str(backend).lower()
