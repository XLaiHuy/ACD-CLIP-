from __future__ import annotations

import inspect

from tools.sabra_v2.forensics import p30r1_causal_forensic as forensic


def test_forensic_is_offline_and_has_no_trainable_model_path() -> None:
    source = inspect.getsource(forensic)
    forbidden = (
        "torch.optim",
        "optimizer.step",
        "RegionResidualAdapter",
        "build_phase2b",
        "load_phase2b",
        "train_region_distill",
        "clip.load",
    )
    assert not [token for token in forbidden if token in source]
    assert forensic.GAMMA_VALUES == (0.0, 0.5, 1.0)


def test_forensic_declares_zero_new_execution_and_fixed_descriptors() -> None:
    assert forensic.SCORE_DIFFERENCE_THRESHOLDS == (1e-6, 1e-4, 1e-3, 1e-2, 5e-2)
    assert forensic.TOP_PIXEL_FRACTIONS == (0.001, 0.005, 0.01, 0.05)
    assert forensic.REGION_COORDINATES == 243
