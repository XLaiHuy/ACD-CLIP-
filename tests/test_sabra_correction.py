from __future__ import annotations

import numpy as np
import torch

from model.phase2b_runtime import deploy_native_logits
from tools.sabra.correction import authority, build_delta, correction_values, margin_scale_p90, validate_lambda


def test_lambda_bounds_margin_scale_and_authority():
    assert validate_lambda(0.0) == 0.0
    assert validate_lambda(1.0) == 1.0
    for value in (-1.0, 1.01):
        try:
            validate_lambda(value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid lambda accepted")
    scale = margin_scale_p90(np.asarray([-1.0, 0.0, 2.0, 4.0]))
    assert scale["implementation"] == "numpy.percentile(method=linear)"
    assert np.allclose(authority(np.asarray([0.5]), np.asarray([0.2])), np.asarray([0.1]))


def test_correction_zero_and_shared_stage_contract():
    native = torch.randn(3, 1, 1369, 2)
    correction = correction_values(0.5, 2.0, np.full((1, 1369), 0.5), np.full((1, 1369), 0.25))
    delta = build_delta(native, correction)
    assert torch.equal(delta[..., 0], torch.zeros_like(delta[..., 0]))
    assert torch.equal(delta[0, ..., 1], delta[-1, ..., 1])
    native_probability, _ = deploy_native_logits(native)
    corrected_probability, _ = deploy_native_logits(native + build_delta(native, np.zeros((1, 1369), dtype=np.float32)))
    assert torch.equal(native_probability, corrected_probability)
