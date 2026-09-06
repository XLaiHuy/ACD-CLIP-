import ast

import pytest
import torch

from h2_clean.stage_fusion import (
    H2_EQUAL_STAGE_FUSION_WEIGHTS,
    H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS,
    fuse_stage_logits,
    validate_stage_fusion_weights,
)


def test_frozen_weight_contract_and_no_trainable_parameters():
    assert H2_EQUAL_STAGE_FUSION_WEIGHTS == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS == (
        0.3736138197153701,
        0.3270300383596602,
        0.2993561419249697,
    )
    assert sum(H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS) == pytest.approx(1.0, abs=1e-12)
    assert tuple(validate_stage_fusion_weights(H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS)) == H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS
    assert list(torch.nn.Module().parameters()) == []


def test_equal_weights_reproduce_historical_mean_exactly():
    logits = torch.randn(3, 2, 2, 4, 4)
    assert torch.equal(
        fuse_stage_logits(logits, H2_EQUAL_STAGE_FUSION_WEIGHTS),
        torch.mean(logits, dim=0),
    )


def test_candidate_weights_apply_to_pre_softmax_logits_and_preserve_order():
    logits = torch.zeros(3, 1, 2, 1, 1)
    logits[0, 0, 1, 0, 0] = 3.0
    logits[1, 0, 1, 0, 0] = 5.0
    logits[2, 0, 1, 0, 0] = 7.0
    expected = sum(
        weight * logits[index]
        for index, weight in enumerate(H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS)
    )
    fused = fuse_stage_logits(logits, H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS)
    assert torch.allclose(fused, expected)
    assert not torch.allclose(fused.softmax(dim=1), torch.stack([
        logits[index].softmax(dim=1) for index in range(3)
    ]).mul(torch.tensor(H2_FIXED_RELIABILITY_STAGE_FUSION_WEIGHTS).view(3, 1, 1, 1, 1)).sum(0))


def test_invalid_weights_are_rejected():
    with pytest.raises(ValueError):
        validate_stage_fusion_weights((0.5, 0.5))
    with pytest.raises(ValueError):
        validate_stage_fusion_weights((0.4, 0.4, 0.3))
    with pytest.raises(ValueError):
        validate_stage_fusion_weights((-0.1, 0.5, 0.6))


def test_adapter_changes_only_final_aggregation_call():
    source = ast.parse(open("model/adapter.py", encoding="utf-8").read())
    calls = [
        node for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "fuse_stage_logits"
    ]
    assert len(calls) == 1
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "all_group_preds", "self.stage_fusion_weights"
    ]
