import torch
import torch.nn.functional as F
from torch import nn

from model.checkpoint_utils import build_phase4_checkpoint, load_adapter_checkpoint
from model.h6.losses import router_teacher_loss, routing_balance_loss
from model.h6.model import H6Progress1
from model.h6.router import PatchRouter
from train import get_h6_vae_beta, router_specialization_failed


class _TinyPhase4Model(nn.Module):
    def __init__(self, h6_router_soft_epochs=8, h6_sparse_transition_epochs=4):
        super().__init__()
        self.image_adapter = nn.Linear(3, 3)
        self.text_adapter = nn.Linear(3, 3)
        self.soft_prompt = nn.Module()
        self.soft_prompt.ctx = nn.Parameter(torch.randn(4, 768))
        self.h6_enabled = True
        self.h6 = H6Progress1(
            n_groups=3,
            router_soft_epochs=h6_router_soft_epochs,
            sparse_transition_epochs=h6_sparse_transition_epochs,
            load_bias_enabled=True,
            vae_class_ratio=0.25,
        )
        self.n_groups = 3
        self.dfg_mode = "attn"
        self.dfg_attn_dim = 256
        self.dfg_attn_tau = 8.0
        self.use_ss2d_dfg = True
        self.dfg_gamma_max = 0.2
        self.dfg_ss2d_fusion = "weight_residual"
        self.dfg_beta = 0.1
        self.dfg_beta_schedule = "warmup010"
        self.dfg_beta_target = 0.1
        self.dfg_weight_residual_fp32 = True
        self.soft_prompt_ctx_len = 4
        self.soft_prompt_init = "phrase"
        self.soft_prompt_init_phrase = "a photo of a"
        self.hybrid_alpha_current = 0.05
        self.hybrid_alpha_max = 0.2
        self.soft_prompt_freeze_epochs = 3


def test_straight_through_transition_schedule_and_gradients():
    torch.manual_seed(0)
    router = PatchRouter(
        n_groups=1, num_factors=4, text_dim=8, bank_dim=4, hidden_dim=6,
        top_k=2, soft_routing_epochs=8, sparse_transition_epochs=4,
    )
    tokens = torch.randn(1, 2, 5, 8)
    keys = torch.eye(4)
    expected = {1: 0.0, 8: 0.0, 9: 0.25, 10: 0.50, 11: 0.75, 12: 1.0, 20: 1.0}
    for epoch, ratio in expected.items():
        assert abs(router.sparse_ratio(epoch) - ratio) < 1e-8
    dense = router(tokens, epoch_one_based=8, concept_keys=keys)
    mixed = router(tokens, epoch_one_based=10, concept_keys=keys)
    sparse = router(tokens, epoch_one_based=12, concept_keys=keys)
    assert torch.allclose(dense["prediction_probabilities"], dense["dense_probabilities"], atol=1e-7)
    assert torch.allclose(sparse["prediction_probabilities"], sparse["sparse_probabilities"], atol=1e-7)
    assert (sparse["prediction_probabilities"] > 0).sum(dim=-1).eq(2).all()
    expected_mixed = 0.5 * mixed["dense_probabilities"] + 0.5 * mixed["sparse_probabilities"]
    assert torch.allclose(mixed["prediction_probabilities"], expected_mixed, atol=1e-7)
    loss = sparse["prediction_probabilities"].square().mean()
    loss.backward()
    assert router.query_projector[0].weight.grad is not None
    assert torch.isfinite(router.query_projector[0].weight.grad).all()


def test_router_teacher_is_detached_and_updates_dense_router_only():
    projected = torch.randn(1, 1, 4, 2, requires_grad=True)
    prototype_normal = F.normalize(torch.randn(1, 4, 2), dim=-1).requires_grad_()
    prototype_abnormal = F.normalize(torch.randn(1, 4, 2), dim=-1).requires_grad_()
    dense = F.softmax(torch.randn(1, 1, 4, 4), dim=-1).requires_grad_()
    masks = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    labels = torch.tensor([1])

    loss, diagnostics = router_teacher_loss(
        projected, prototype_normal, prototype_abnormal, dense, masks, labels, temperature=0.15
    )
    loss.backward()
    assert dense.grad is not None and torch.isfinite(dense.grad).all()
    assert projected.grad is None
    assert prototype_normal.grad is None
    assert prototype_abnormal.grad is None
    assert torch.isfinite(diagnostics["router_teacher_entropy"])


def test_load_bias_selection_only_and_checkpoint_buffers_roundtrip():
    router = PatchRouter(
        n_groups=1, num_factors=4, text_dim=8, bank_dim=4, hidden_dim=6,
        top_k=2, soft_routing_epochs=0, sparse_transition_epochs=1,
        load_bias_enabled=True, load_bias_momentum=0.0, load_bias_step=0.1, load_bias_max=0.2,
    )
    frequency = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    router.update_load_bias(frequency)
    assert router.load_bias[0, 0] > 0
    assert router.load_bias[0, 3] < 0
    assert router.load_bias.abs().max() <= 0.2
    assert router.load_bias.requires_grad is False
    assert router.ema_topk_usage.requires_grad is False

    tokens = torch.randn(1, 1, 2, 8)
    keys = torch.eye(4)
    with torch.no_grad():
        router.load_bias[:] = torch.tensor([[0.2, 0.2, -0.2, -0.2]])
    biased = router(tokens, epoch_one_based=1, concept_keys=keys)
    with torch.no_grad():
        router.load_bias.zero_()
    unbiased = router(tokens, epoch_one_based=1, concept_keys=keys)
    assert torch.allclose(biased["dense_probabilities"], unbiased["dense_probabilities"], atol=1e-7)
    assert not torch.equal(biased["topk_indices"], unbiased["topk_indices"])

    source = _TinyPhase4Model(h6_router_soft_epochs=8, h6_sparse_transition_epochs=4)
    with torch.no_grad():
        source.h6.router.load_bias.fill_(0.01)
        source.h6.router.ema_topk_usage.fill_(0.25)
    payload = build_phase4_checkpoint(
        source,
        epoch=12,
        seed=0,
        precision="fp32",
        phase2b_config={
            "n_groups": 3,
            "h6_router_failure_patience": 2,
            "h6_router_max_sparse_dead_factors": 1,
            "h6_router_min_unique_topk_pairs": 2,
            "h6_kl_zero_epochs": 8,
            "h6_kl_warmup_epochs": 4,
            "h6_kl_free_bits": 0.02,
            "beta_h6_vae_kl": 1e-5,
        },
        loss_weights={"router_teacher": 0.01, "balance": 0.001},
    )
    restored = _TinyPhase4Model(h6_router_soft_epochs=8, h6_sparse_transition_epochs=4)
    assert load_adapter_checkpoint(restored, payload) is True
    assert torch.allclose(restored.h6.router.load_bias, source.h6.router.load_bias)
    assert torch.allclose(restored.h6.router.ema_topk_usage, source.h6.router.ema_topk_usage)


def test_failure_detector_and_kl_free_bits_behavior():
    dead = torch.tensor([2, 0, 0])
    pairs_bad = torch.tensor([1, 3, 3])
    pairs_good = torch.tensor([2, 3, 3])
    assert not router_specialization_failed(0.25, dead, pairs_bad, 1, 2)
    assert router_specialization_failed(0.50, dead, pairs_good, 1, 2)
    assert router_specialization_failed(0.50, torch.zeros(3, dtype=torch.long), pairs_bad, 1, 2)
    assert not router_specialization_failed(0.50, torch.zeros(3, dtype=torch.long), pairs_good, 1, 2)

    beta = 1e-5
    assert get_h6_vae_beta(8, beta, zero_epochs=8, warmup_epochs=4) == 0.0
    assert abs(get_h6_vae_beta(9, beta, zero_epochs=8, warmup_epochs=4) - 2.5e-6) < 1e-12
    raw_low = torch.tensor(0.01, requires_grad=True)
    effective_low = torch.clamp(raw_low, min=0.02)
    effective_low.backward()
    assert raw_low.grad.item() == 0.0
    raw_high = torch.tensor(0.03, requires_grad=True)
    effective_high = torch.clamp(raw_high, min=0.02)
    effective_high.backward()
    assert raw_high.grad.item() == 1.0


def test_diagnostics_and_dense_balance_are_finite():
    router = PatchRouter(n_groups=2, num_factors=4, text_dim=8, bank_dim=4, hidden_dim=6, top_k=2)
    tokens = torch.randn(2, 2, 5, 8)
    keys = torch.eye(4)
    output = router(tokens, epoch_one_based=3, concept_keys=keys)
    diagnostics = router.diagnostics(
        output["prediction_probabilities"],
        dense_probabilities=output["dense_probabilities"],
        sparse_probabilities=output["sparse_probabilities"],
        topk_indices=output["topk_indices"],
    )
    concept = router.concept_key_diagnostics(output["concept_keys"])
    assert diagnostics["unique_topk_pairs"].shape == (2,)
    assert torch.isfinite(concept["concept_key_cos_mean"])
    assert torch.isfinite(concept["concept_key_cos_max"])
    assert torch.isfinite(routing_balance_loss(output["dense_probabilities"]))
