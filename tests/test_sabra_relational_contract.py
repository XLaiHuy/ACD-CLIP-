from __future__ import annotations

import numpy as np

from tools.sabra.relational import FEATURE_ORDER, NEED_ORDER, assert_gt_free_payload, build_relational_record, need_features, trust_features


def test_relational_contract_and_feature_order():
    features = np.ones((3, 1369, 768), dtype=np.float32)
    margins = np.zeros((3, 1369), dtype=np.float32)
    record = build_relational_record(features, margins)
    assert record["E"].shape == (1369,)
    assert trust_features(record).shape == (1369, len(FEATURE_ORDER))
    assert need_features(record).shape == (1369, len(NEED_ORDER))
    assert record["peer_indices"].shape == (1369, 8)


def test_relational_input_rejects_gt():
    try:
        assert_gt_free_payload({"mask": object()})
    except AssertionError:
        pass
    else:
        raise AssertionError("GT firewall did not reject mask")
