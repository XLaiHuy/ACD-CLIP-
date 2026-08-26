from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.sabra_v2.p29_contract import audit_p29_protocol
from tools.sabra_v2.p29_post_run_audit import classify_p29


def test_p29_protocol_audit_rejects_loss_or_schedule_drift() -> None:
    protocol = json.loads(Path("research/sabra_v2/region_distill/P29_PROTOCOL.json").read_text())
    assert audit_p29_protocol(protocol)["status"] == "PASS"
    protocol["frozen_method"]["loss"]["total"] = "drift"
    with pytest.raises(RuntimeError, match="loss contract"):
        audit_p29_protocol(protocol)


def test_p29_status_is_mechanically_derived_from_preregistered_supported_gate() -> None:
    supported = {"p29_macro_pAP": 0.46, "native_macro_pAP": 0.45, "p29_macro_pAUROC": 0.94, "native_macro_pAUROC": 0.93, "median_delta_pAP": 0.001, "improving_class_count": 7}
    assert classify_p29(supported, audit_pass=True) == "P29_SUPPORTED"
    supported["p29_macro_pAUROC"] = 0.92
    assert classify_p29(supported, audit_pass=True) != "P29_SUPPORTED"
