from __future__ import annotations

import numpy as np

from tools.sabra.relational import assert_gt_free_payload, build_relational_record


def test_relational_pass_has_no_gt_argument_or_access():
    record = build_relational_record(np.zeros((3, 1369, 768), dtype=np.float32), np.zeros((3, 1369), dtype=np.float32))
    assert "mask" not in record
    assert "label" not in record
    assert_gt_free_payload(record)
