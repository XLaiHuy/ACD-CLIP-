from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from model.adapter import ACDCLIP
from model.h6.model import H6Progress1
from model.h6.semantic_bank import CoPSSemanticCore
from model.h6.utility_routing import (
    build_patch_targets,
    exploration_epsilon,
    utility_diagnostics,
    utility_factor_loss,
    utility_router_loss,
    utility_teacher,
)
from model.h6.specialization_trajectory import (
    aggregate_utility_records,
    capture_utility_record,
    teacher_sensitivity_grid,
    write_trajectory_artifacts,
)
from utils import get_structured_prompt_sentence


def test_structured_literal_layout():
    normal = get_structured_prompt_sentence("brain tissue", 0, 4)
    abnormal = get_structured_prompt_sentence("brain tissue", 1, 4)
    assert normal.split()[:6] == ["X"] * 6
    assert normal.split()[6:] == ["normal", "brain", "tissue."]
    assert abnormal.split()[6] == "abnormal"


class _Transformer:
    @staticmethod
    def get_cast_dtype():
        return torch.float32


class _Clip(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.transformer = _Transformer()
        self.token_embedding = nn.Embedding(64, dim)


class _StructuredEncoderHarness(nn.Module):
    encode_dynamic_prompt_text = ACDCLIP.encode_dynamic_prompt_text

    def __init__(self):
        super().__init__()
        self.clipmodel = _Clip()
        self.soft_prompt_ctx_len = 4
        self.shared_text_lora = nn.Linear(8, 1, bias=False)
        self.captured = None
        self.adapt_text = None

    def _encode_text_from_embeddings(self, token_ids, embeddings, adapt_text=True):
        self.captured = embeddings
        self.adapt_text = adapt_text
        return [self.shared_text_lora(embeddings).sum(dim=(1, 2), keepdim=False).unsqueeze(-1).expand(-1, 8)]


def test_state_and_class_positions_are_separate_and_use_shared_lora():
    model = _StructuredEncoderHarness()
    ids = torch.zeros(2, 12, dtype=torch.long)
    contexts = torch.randn(2, 4, 8)
    state = torch.randn(2, 8, requires_grad=True)
    class_token = torch.randn(2, 8, requires_grad=True)
    output = model.encode_dynamic_prompt_text(ids, contexts, state, class_token)
    assert model.adapt_text is True
    assert torch.equal(model.captured[:, 5], state)
    assert torch.equal(model.captured[:, 6], class_token)
    assert not torch.equal(model.captured[:, 5], model.captured[:, 6])
    output[0].sum().backward()
    assert model.shared_text_lora.weight.grad is not None
    assert state.grad is not None
    assert class_token.grad is not None


def test_semantic_core_state_is_factor_specific_and_class_uses_deterministic_mu():
    torch.manual_seed(4)
    core = CoPSSemanticCore(
        n_groups=1, num_factors=4, bank_dim=8, text_dim=8, ctx_len=4,
        vae_hidden_dim=8, vae_latent_dim=4,
    ).train()
    seg = [torch.randn(2, 4, 8)]
    cls = torch.randn(2, 8)
    ctx_normal = torch.randn(4, 8)
    ctx_abnormal = torch.randn(4, 8)
    first = core(seg, cls, ctx_normal, ctx_abnormal)
    second = core(seg, cls, ctx_normal, ctx_abnormal)
    assert first["state_tokens"].shape == (2, 4, 2, 8)
    assert first["structured_state_position"].item() == 5
    assert first["structured_class_position"].item() == 6
    assert torch.equal(first["class_token"], first["decoded_semantic"])
    assert torch.allclose(first["class_token"], second["class_token"])
    assert not torch.allclose(first["reconstruction_sample"], second["reconstruction_sample"])
    assert first["state_tokens"].std(dim=1).mean() > 0
    expected_context = torch.stack([ctx_normal, ctx_abnormal], dim=0)[None, None]
    assert torch.allclose(first["structured_contexts"], expected_context.expand_as(first["structured_contexts"]))


def test_patch_target_area_pool_and_invalid_area_exclusion():
    mask = torch.tensor([[[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0],
                           [0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]]])
    valid = torch.ones_like(mask)
    valid[:, :, :2, :2] = 0
    target, patch_valid = build_patch_targets(mask, 4, valid)
    assert torch.allclose(target, torch.tensor([[0.0, 1.0, 0.5, 1.0]]))
    assert torch.equal(patch_valid, torch.tensor([[False, True, True, True]]))


def _payload(requires_grad=True):
    base = torch.zeros(1, 1, 4)
    evidence = torch.tensor([[[[-8.0, 8.0, 0.0, 0.0],
                               [-8.0, 8.0, 0.0, 0.0],
                               [8.0, -8.0, 0.0, 0.0],
                               [8.0, -8.0, 0.0, 0.0]]]], requires_grad=requires_grad)
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    valid = torch.ones_like(target, dtype=torch.bool)
    return utility_teacher(base, evidence, target, valid, epsilon=0.15), evidence, target


def test_utility_equations_detach_and_exploration():
    payload, evidence, target = _payload()
    manual_base = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor(0.0), torch.tensor(0.0), reduction="none"
    )
    assert payload["loss_base"][0, 0, 0] == pytest.approx(manual_base.item())
    expected = (manual_base - payload["loss_per_factor"][0, 0, 0, 0]) / max(manual_base, torch.tensor(0.1))
    assert payload["gain_rel"][0, 0, 0, 0].detach().item() == pytest.approx(expected.item())
    assert payload["q_utility"].requires_grad is False
    assert payload["responsibility"].requires_grad is False
    assert torch.allclose(payload["responsibility"].sum(dim=-1), torch.ones(1, 1, 4))
    assert exploration_epsilon(1, 20) == pytest.approx(0.15)
    assert exploration_epsilon(20, 20) == pytest.approx(0.05)
    loss = utility_factor_loss(payload, target)
    loss.backward()
    assert evidence.grad is not None


def test_router_teacher_filters_uninformative_and_only_router_receives_router_gradient():
    payload, evidence, _ = _payload()
    router_logits = nn.Parameter(torch.zeros_like(payload["q_utility"]))
    probabilities = router_logits.softmax(dim=-1)
    loss = utility_router_loss(probabilities, payload)
    loss.backward()
    assert router_logits.grad is not None
    assert evidence.grad is None
    flat_payload, _, _ = _payload(requires_grad=False)
    flat_payload["informative"] = torch.zeros_like(flat_payload["informative"])
    assert utility_router_loss(probabilities.detach(), flat_payload).item() == 0.0


def test_utility_diagnostics_and_dense_routing_math():
    base = torch.zeros(1, 1, 2)
    evidence = torch.tensor([[[[-8.0, 8.0, 0.0, 0.0],
                               [-8.0, 8.0, 0.0, 0.0]]]])
    target = torch.tensor([[0.0, 1.0]])
    valid = torch.ones_like(target, dtype=torch.bool)
    payload = utility_teacher(base, evidence, target, valid, epsilon=0.0)
    router = payload["q_utility"].clone()
    diagnostics = utility_diagnostics(payload, router, target)
    required = {
        "Base", "BestSingle", "OracleMulti", "Uniform", "SoftRouted", "HardRouted",
        "G_local", "G_multi", "capture", "capture_denominator", "capture_valid",
        "base_denominator_valid", "L_base", "L_per_factor",
        "teacher_entropy", "teacher_max_probability", "informative_fraction",
        "all_harm_fraction", "winner_shares", "router_top1_agreement",
        "teacher_router_KL", "router_entropy", "router_usage",
    }
    assert required <= diagnostics.keys()
    assert diagnostics["OracleMulti"] <= diagnostics["BestSingle"]
    assert diagnostics["BestSingle"] <= diagnostics["Base"]
    assert diagnostics["Base"] != pytest.approx(1.0)
    assert diagnostics["BestSingle"] > diagnostics["OracleMulti"]
    assert diagnostics["G_local"] == pytest.approx(
        ((diagnostics["Base"] - diagnostics["OracleMulti"]) / diagnostics["Base"]).item()
    )
    assert diagnostics["G_multi"] == pytest.approx(
        ((diagnostics["BestSingle"] - diagnostics["OracleMulti"]) / diagnostics["Base"]).item()
    )
    assert diagnostics["capture_valid"].item() is True
    assert diagnostics["capture"] == pytest.approx(
        ((diagnostics["Uniform"] - diagnostics["SoftRouted"])
         / (diagnostics["Uniform"] - diagnostics["OracleMulti"])).item()
    )
    assert diagnostics["router_top1_agreement"] == pytest.approx(1.0)
    assert diagnostics["teacher_router_KL"] == pytest.approx(0.0, abs=1e-7)


def test_capture_invalid_denominator_is_explicit_and_finite():
    base = torch.zeros(1, 1, 2)
    evidence = torch.zeros(1, 1, 2, 4)
    target = torch.tensor([[0.0, 1.0]])
    valid = torch.ones_like(target, dtype=torch.bool)
    payload = utility_teacher(base, evidence, target, valid)
    diagnostics = utility_diagnostics(payload, payload["q_utility"], target)
    assert diagnostics["capture_denominator"] == pytest.approx(0.0, abs=1e-12)
    assert diagnostics["capture_valid"].item() is False
    assert diagnostics["capture"] == pytest.approx(0.0)
    assert torch.isfinite(diagnostics["capture"])


def test_rho_rejects_noncanonical_training_values():
    base = torch.zeros(1, 1, 1)
    factors = torch.zeros(1, 1, 1, 4)
    target = torch.zeros(1, 1)
    valid = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="canonical"):
        utility_teacher(base, factors, target, valid, rho=0.04)


def test_specialization_trajectory_exact_aggregation_and_gate_causes(tmp_path):
    payload, _, target = _payload(requires_grad=False)
    router = payload["q_utility"].clone()
    record = capture_utility_record(
        payload, router, target, utility_router_loss(router, payload)
    )
    aggregate = aggregate_utility_records(
        [record, record], gain_threshold=0.02, entropy_threshold=0.98
    )
    direct = utility_diagnostics(payload, router, target)
    assert aggregate["Base"] == pytest.approx(direct["Base"].item())
    assert aggregate["OracleMulti"] == pytest.approx(direct["OracleMulti"].item())
    assert aggregate["gain_threshold_pass_fraction"] > 0
    assert 0 <= aggregate["entropy_threshold_pass_fraction"] <= 1
    assert aggregate["router_supervised_patch_count"] == int(payload["informative"].sum()) * 2
    assert set(aggregate["normal_anomaly_breakdown"]) == {"normal", "anomaly"}
    assert set(aggregate["best_gain_rel"]) >= {"mean", "std", "p50", "p99"}

    structure = {
        "factor_embedding_effective_rank": 1.1,
        "factor_embedding_pairwise_cosine_mean": 0.9,
        "factor_patch_pairwise_correlation_mean": 0.8,
    }
    milestone = {
        "batch": 32, "optimizer_steps": 5, "cumulative": aggregate,
        "recent_window": aggregate, "structure": structure,
    }
    write_trajectory_artifacts(tmp_path, [milestone], {"status": "PASS"})
    assert (tmp_path / "trajectory.json").is_file()
    assert (tmp_path / "trajectory.csv").is_file()
    assert (tmp_path / "final_summary.json").is_file()


def test_teacher_sensitivity_grid_is_offline_and_complete():
    payload, evidence, target = _payload(requires_grad=True)
    router = payload["q_utility"].clone()
    record = capture_utility_record(
        payload, router, target, utility_router_loss(router, payload)
    )
    evidence_before = evidence.detach().clone()
    grid = teacher_sensitivity_grid([record], gain_threshold=0.02)
    assert len(grid) == 9
    assert {(row["tau_utility"], row["entropy_threshold"]) for row in grid} == {
        (tau, threshold)
        for tau in (0.05, 0.03, 0.02)
        for threshold in (0.98, 0.99, 0.995)
    }
    assert torch.equal(evidence.detach(), evidence_before)
    assert evidence.grad is None
