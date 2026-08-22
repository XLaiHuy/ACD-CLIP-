#!/usr/bin/env python3
"""Run the canonical Phase2B selector with repaired candidate serialization."""
from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR_PATH = REPO_ROOT / "select_phase2b_checkpoint.py"


def _load_selector() -> Any:
    """Load the selector module from this repository, not an installed copy."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    selector = importlib.import_module("select_phase2b_checkpoint")
    loaded_path = Path(str(getattr(selector, "__file__", ""))).resolve()
    if loaded_path != SELECTOR_PATH.resolve():
        raise RuntimeError(f"canonical selector loaded from unexpected path: {loaded_path}")
    return selector


def score_candidate_rows(
    candidates: Iterable[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
    selection_score: Callable[[Mapping[str, float]], float],
) -> list[dict[str, Any]]:
    """Add canonical scores while preserving every original candidate field."""
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = {name: float(candidate[name]) for name in metric_names}
        scored.append(dict(candidate) | {"score": float(selection_score(metrics))})
    return scored


def main(argv: list[str] | None = None) -> int:
    """Delegate all evaluation and selection work to the canonical selector."""
    selector = _load_selector()
    original_output_rows = selector._output_rows

    def repaired_output_rows(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        scored = score_candidate_rows(
            candidates,
            metric_names=selector.METRIC_NAMES,
            selection_score=selector.selection_score,
        )
        return original_output_rows(scored)

    selector._output_rows = repaired_output_rows
    try:
        return int(selector.main(argv))
    finally:
        selector._output_rows = original_output_rows


if __name__ == "__main__":
    raise SystemExit(main())
