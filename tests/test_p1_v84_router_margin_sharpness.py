from __future__ import annotations

import torch

from tools.audit_p1_v84a_router_margin_sharpness import _SharpnessAccumulator, _tau_usable


def _gain() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Patches 0 and 2 are selected with F1 and F3 winners; patch 1 is tied
    # and patch 3 is non-positive, so neither is eligible.
    gain = torch.tensor(
        [[[
            [0.40, 0.00, -0.10, -0.20],
            [0.20, 0.20, -0.10, -0.20],
            [0.00, -0.10, 0.40, -0.20],
            [0.00, -0.10, -0.20, -0.30],
        ]]]
    )
    return gain, torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]), torch.ones(1, 1, 4, dtype=torch.bool)


def test_sharpness_accumulator_uses_only_margin_eligible_patches_and_is_tau_invariant():
    gain, targets, valid = _gain()
    accumulator = _SharpnessAccumulator()
    accumulator.add(gain, targets, valid)
    result = accumulator.result()
    assert result["support"]["overall"]["selected_count"] == 2
    assert result["support"]["normal"]["selected_count"] == 1
    assert result["support"]["anomaly"]["selected_count"] == 1
    assert result["winner_shares"]["F1"]["count"] == 1
    assert result["winner_shares"]["F3"]["count"] == 1
    for tau in ("0.05", "0.03", "0.02"):
        assert result["tau"][tau]["q_argmax_matches_raw_winner"] is True
        assert result["tau"][tau]["regions"]["overall"]["normalized_entropy"]["count"] == 2
        assert result["tau"][tau]["regions"]["overall"]["kl_q_to_uniform"]["mean"] > 0.0


def test_tau_usable_requires_both_selected_regions_anomaly_multifactor_and_entropy_contract():
    gain, targets, valid = _gain()
    accumulator = _SharpnessAccumulator()
    accumulator.add(gain, targets, valid)
    result = accumulator.result()
    assert _tau_usable(result, "0.05") is False  # F2/F4 anomaly coverage intentionally absent.
    result["anomaly_non_f1_coverage"] = {
        "F2": {"count": 1}, "F3": {"count": 1}, "F4": {"count": 1},
    }
    result["tau"]["0.05"]["regions"]["overall"]["normalized_entropy"]["p50"] = 0.97
    result["tau"]["0.05"]["regions"]["anomaly"]["normalized_entropy"]["p50"] = 0.97
    assert _tau_usable(result, "0.05") is True
