from __future__ import annotations

import numpy as np
import torch

from tools.sabra.need import intervention_delta, need_oracle
from tools.sabra.relational import FEATURE_ORDER, NEED_ORDER
from tools.sabra.trust import fit_binary_predictor, frozen_probability


def test_trust_serialization_probability_parity():
    values = np.asarray([[0.0, 0.2, 0.3, 0.4, 0.5], [1.0, 0.4, 0.3, 0.2, 0.1], [0.2, 0.3, 0.4, 0.5, 0.6], [0.8, 0.7, 0.6, 0.5, 0.4]], dtype=np.float64)
    targets = np.asarray([0, 1, 0, 1], dtype=np.int8)
    artifact = fit_binary_predictor(values, targets, FEATURE_ORDER)
    left = frozen_probability(artifact, values)
    right = frozen_probability(dict(artifact), values)
    assert np.allclose(left, right, atol=1e-7, rtol=0.0)


def test_need_serialization_contract_and_gradient_firewall():
    native = torch.zeros((3, 1, 1369, 2), dtype=torch.float32)
    mask = torch.zeros((1, 1, 518, 518), dtype=torch.float32)
    result = need_oracle(native, mask)
    assert result["intervention_gradient"].shape == (1, 1369)
    assert result["target"].dtype == torch.int8
    delta = intervention_delta(torch.ones((1, 1369)), native)
    assert torch.equal(delta[..., 0], torch.zeros_like(delta[..., 0]))
    assert torch.equal(delta[0, ..., 1], delta[-1, ..., 1])
    assert native.grad is None
