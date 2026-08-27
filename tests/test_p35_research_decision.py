from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _json(name: str) -> dict:
    return json.loads((ROOT / "research/sabra_v2/region_distill" / name).read_text())


def test_p35_decision_contains_required_inventory_and_single_selection() -> None:
    decision = _json("P35_RESEARCH_DECISION.json")
    assert decision["status"] == "P35_RESEARCH_DECISION_COMPLETE"
    assert decision["selected_next_hypothesis"] == "SOFT_ACTIONABILITY_WEIGHTED_FUNCTIONAL_TRANSFER"
    assert decision["selected_candidate"] == "B"
    assert set(decision["artifact_inventory"]) == {"P31_native", "P30R1", "P32", "P33", "P34"}
    conclusions = decision["required_forensic_conclusions"]
    assert "importance allocation" in conclusions["p33_gain_mechanism"]
    assert "not established intrinsically harmful" in conclusions["dense_support"]
    assert conclusions["complexity"]["objective_count"] == 1
    assert conclusions["complexity"]["new_tuned_hyperparameters"] == 0


def test_p35_decision_records_frozen_p34_target_magnitude_forensic() -> None:
    decision = _json("P35_RESEARCH_DECISION.json")
    target = _json("P35_SOURCE_TARGET_FORENSIC.json")
    reference = decision["source_only"]["target_preservation_forensic"]
    assert reference["full_target"] == "E_t"
    assert reference["p34_target"] == "wE_t"
    assert reference["l1_ratio_p34_to_full"] == target["retained_target_mass"]["l1_ratio_p34_to_full"]
    assert reference["rms_ratio_p34_to_full"] == target["retained_target_mass"]["rms_ratio_p34_to_full"]
    assert target["source"]["held_reads"] == 0
    assert target["source"]["new_clip_forwards"] == 0
    assert target["source"]["cache_rebuilds"] == 0


def test_p35_artifact_inventory_distinguishes_p31_native_control() -> None:
    inventory = _json("P35_RESEARCH_DECISION.json")["artifact_inventory"]
    native = inventory["P31_native"]
    assert native["native_logits"]["available"] is True
    assert native["corrected_logits"]["available"] is False
    assert native["region_residuals"]["representation"] == "implicit exact zero"
    assert native["score_effects"]["representation"] == "exact zero native-to-control effect"
