from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.sabra_v2.forensics.p35_soft_actionability import actionability_map


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_p35_preflight_is_passed_and_non_scientific() -> None:
    artifact = json.loads(
        (ROOT / "research/sabra_v2/region_distill/P35_PREFLIGHT_FALSIFICATION.json").read_text()
    )
    assert artifact["status"] == "P35_PREFLIGHT_PASS"
    assert artifact["selected_candidate"] == "SOFT_TANH_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER"
    assert artifact["gates"]["held_reads"] == 0
    assert artifact["gates"]["new_clip_forwards"] == 0
    assert artifact["gates"]["new_phase2b_forwards"] == 0
    assert artifact["gates"]["new_teacher_forwards"] == 0
    assert artifact["gates"]["cache_rebuilds"] == 0


def test_p35_maps_are_bounded_monotonic_and_zero_preserving() -> None:
    values = np.array([0.0, 0.01, 0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    mapped = actionability_map(values, "tanh")
    assert mapped[0] == 0.0
    assert np.all(np.isfinite(mapped))
    assert np.all(mapped >= 0.0)
    assert np.all(mapped <= 1.0)
    assert np.all(np.diff(mapped) >= 0.0)
    assert np.count_nonzero(mapped == 1.0) == 0


def test_p35_preflight_records_full_target_importance_semantics() -> None:
    artifact = json.loads(
        (ROOT / "research/sabra_v2/region_distill/P35_PREFLIGHT_FALSIFICATION.json").read_text()
    )
    assert artifact["source_only"]["no_target_shrinkage"] is True
    assert artifact["interpretation"]["p35_target"] == "full detached E_t for every candidate and every weight"
    assert artifact["interpretation"]["p35_zero_actionability"].startswith("zero direct importance")
