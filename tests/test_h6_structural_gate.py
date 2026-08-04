import torch

from model.h6.gated_early_stop import H6StructuralGateState, StructuralGateConfig


def _gate(**kwargs):
    config = StructuralGateConfig(enabled=True, **kwargs)
    return H6StructuralGateState(config)


def _diag(**overrides):
    diag = {
        "final_query_effective_rank": torch.tensor([3.0, 3.0, 3.0]),
        "final_query_top1_energy_ratio": torch.tensor([0.50, 0.50, 0.50]),
        "final_query_pairwise_cos_mean": torch.tensor([0.10, 0.10, 0.10]),
        "per_factor_logit_std_across_patches": torch.ones(3, 4) * 0.01,
        "unique_topk_pairs": torch.tensor([3, 3, 3]),
        "final_router_key_cos_max": torch.tensor(0.10),
        "final_router_key_l2_min": torch.tensor(1.0),
        "raw_concept_key_cos_max": torch.tensor(0.10),
        "dynamic_residual_cos_mean": torch.tensor(0.10),
        "stage_dynamic_text_norm_cos_mean": torch.tensor(0.10),
        "factor_identity_tangent_pair_cos_max": torch.tensor(0.10),
        "dynamic_mean_hard_cos": torch.tensor([[[0.80, 0.80]]]),
        "sparse_factor_usage": torch.ones(3, 4) * 0.25,
    }
    diag.update(overrides)
    return diag


def _eval(gate, epoch=8, diag=None, **kwargs):
    return gate.evaluate(
        epoch=epoch,
        diagnostics=_diag() if diag is None else diag,
        epoch_metrics={"task": 1.0, "total": 2.0, "orth": 0.10, "task_rolling_median": 1.0, **kwargs.pop("metrics", {})},
        teacher_diag={"teacher_entropy": torch.tensor(0.5), "teacher_informative_patch_fraction": torch.tensor(1.0), **kwargs.pop("teacher", {})},
        sparse_ratio=kwargs.pop("sparse_ratio", 1.0),
        hybrid_alpha=kwargs.pop("hybrid_alpha", 0.20),
        hybrid_alpha_max=kwargs.pop("hybrid_alpha_max", 0.20),
        router_teacher_weight=kwargs.pop("router_teacher_weight", 0.01),
        router_teacher_target=kwargs.pop("router_teacher_target", 0.01),
        dynamic_mean_anchor_weight=kwargs.pop("dynamic_mean_anchor_weight", 0.001),
        dynamic_mean_anchor_target=kwargs.pop("dynamic_mean_anchor_target", 0.001),
        query_mode=kwargs.pop("query_mode", "local_global_bypass"),
        tangent_enabled=kwargs.pop("tangent_enabled", True),
    )


def test_query_collapse_requires_persistence_and_warmup():
    collapse = _diag(
        final_query_effective_rank=torch.tensor([1.0, 1.0, 1.0]),
        final_query_top1_energy_ratio=torch.tensor([0.999, 0.999, 0.999]),
        final_query_pairwise_cos_mean=torch.tensor([1.0, 1.0, 1.0]),
        per_factor_logit_std_across_patches=torch.zeros(3, 4),
        unique_topk_pairs=torch.tensor([1, 1, 1]),
    )
    gate = _gate()
    assert _eval(gate, epoch=7, diag=collapse).hard_failure is False
    first = _eval(gate, epoch=8, diag=collapse)
    assert first.hard_failure is False
    second = _eval(gate, epoch=9, diag=collapse)
    assert second.hard_failure is True
    assert second.abort_reason == "h6_dense_patch_query_collapse"


def test_query_rank_teacher_fraction_and_task_spike_alone_do_not_abort():
    gate = _gate()
    low_rank = _diag(final_query_effective_rank=torch.tensor([1.0, 1.0, 1.0]))
    decision = _eval(gate, diag=low_rank, teacher={"teacher_informative_patch_fraction": torch.tensor(0.0)}, metrics={"task": 3.0, "task_rolling_median": 1.0})
    assert decision.hard_failure is False
    assert "teacher_informative_fraction_low" in decision.soft_warnings
    assert "task_loss_spike" in decision.soft_warnings


def test_recovery_resets_only_corresponding_counter_and_counters_are_independent():
    collapse = _diag(
        final_query_effective_rank=torch.tensor([1.0, 1.0, 1.0]),
        final_query_top1_energy_ratio=torch.tensor([0.999, 0.999, 0.999]),
        final_query_pairwise_cos_mean=torch.tensor([1.0, 1.0, 1.0]),
        per_factor_logit_std_across_patches=torch.zeros(3, 4),
        unique_topk_pairs=torch.tensor([1, 1, 1]),
    )
    gate = _gate()
    _eval(gate, diag=collapse)
    assert gate.counters["query_collapse"] == 1
    _eval(gate, diag=_diag(final_router_key_cos_max=torch.tensor(0.99)))
    assert gate.counters["query_collapse"] == 0
    assert gate.counters["key_anchor_failure"] == 1


def test_key_anchor_failure_uses_final_keys_not_raw_keys():
    gate = _gate()
    raw_only = _diag(raw_concept_key_cos_max=torch.tensor(1.0))
    assert _eval(gate, diag=raw_only).hard_failure is False
    bad_final = _diag(final_router_key_cos_max=torch.tensor(0.99))
    assert _eval(gate, diag=bad_final).hard_failure is False
    decision = _eval(gate, diag=bad_final)
    assert decision.hard_failure is True
    assert decision.abort_reason == "h6_final_router_key_anchor_failure"


def test_dynamic_factor_collapse_requires_persistence_and_alpha_activation():
    bad = _diag(
        dynamic_residual_cos_mean=torch.tensor(1.0),
        stage_dynamic_text_norm_cos_mean=torch.tensor(1.0),
        factor_identity_tangent_pair_cos_max=torch.tensor(1.0),
    )
    gate = _gate()
    assert _eval(gate, diag=bad, hybrid_alpha=0.10, metrics={"orth": 0.75}).hard_failure is False
    assert _eval(gate, diag=bad, hybrid_alpha=0.20, metrics={"orth": 0.75}).hard_failure is False
    decision = _eval(gate, diag=bad, hybrid_alpha=0.20, metrics={"orth": 0.75})
    assert decision.hard_failure is True
    assert decision.abort_reason == "h6_dynamic_factor_directional_collapse"


def test_semantic_drift_warning_and_hard_abort_thresholds():
    gate = _gate()
    warning = _eval(gate, diag=_diag(dynamic_mean_hard_cos=torch.tensor([0.69])))
    assert warning.hard_failure is False
    assert "dynamic_mean_below_trust_floor" in warning.soft_warnings
    assert _eval(gate, diag=_diag(dynamic_mean_hard_cos=torch.tensor([0.29]))).hard_failure is False
    decision = _eval(gate, diag=_diag(dynamic_mean_hard_cos=torch.tensor([0.29])))
    assert decision.hard_failure is True
    assert decision.abort_reason == "h6_catastrophic_clip_semantic_drift"


def test_sparse_gate_ignores_ratio_025_and_requires_all_levels_by_default():
    bad_sparse = _diag(
        sparse_factor_usage=torch.tensor([[0.5, 0.5, 0.0, 0.0]] * 3),
        unique_topk_pairs=torch.tensor([1, 1, 1]),
    )
    gate = _gate()
    assert _eval(gate, diag=bad_sparse, sparse_ratio=0.25).hard_failure is False
    assert _eval(gate, diag=bad_sparse, sparse_ratio=0.50).hard_failure is False
    decision = _eval(gate, diag=bad_sparse, sparse_ratio=0.50)
    assert decision.hard_failure is True
    assert decision.abort_reason == "h6_sparse_single_pair_collapse"

    one_level = _diag(
        sparse_factor_usage=torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]),
        unique_topk_pairs=torch.tensor([1, 3, 3]),
    )
    gate = _gate()
    assert _eval(gate, diag=one_level, sparse_ratio=0.50).hard_failure is False
    assert _eval(gate, diag=one_level, sparse_ratio=0.50).hard_failure is False


def test_nonfinite_metrics_abort_immediately_and_checkpoint_state_can_reset():
    gate = _gate()
    decision = _eval(gate, diag=_diag(final_router_key_cos_max=torch.tensor(float("nan"))))
    assert decision.hard_failure is True
    assert decision.abort_reason == "h6_nonfinite_structural_gate_metric"

    gate.counters["query_collapse"] = 1
    saved = gate.state_dict()["structural_gate_counters"]
    restored = H6StructuralGateState(gate.config, counters=saved)
    assert restored.counters["query_collapse"] == 1
    restored.reset()
    assert all(value == 0 for value in restored.counters.values())
