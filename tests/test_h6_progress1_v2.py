import torch
import torch.nn.functional as F
from torch import nn

from model.adapter import ACDCLIP
from model.h6.losses import dynamic_residual_diversity_loss, factor_aware_center_loss
from model.h6.model import H6Progress1
from model.h6.router import PatchRouter
from model.h6.semantic_bank import ClassVAE
from train import get_h6_vae_beta


class _IdentityBlock(nn.Module):
    def forward(self, x, attn_mask=None):
        return x, None


class _FakeTextTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleList([_IdentityBlock() for _ in range(12)])

    @staticmethod
    def get_cast_dtype():
        return torch.float32


class _FakeVisualTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblocks = nn.ModuleList([_IdentityBlock() for _ in range(24)])
        self.grad_checkpointing = False


class _FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 1024, kernel_size=1, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(1024))
        self.positional_embedding = nn.Parameter(torch.zeros(5, 1024))
        self.patch_dropout = nn.Identity()
        self.ln_pre = nn.LayerNorm(1024)
        self.transformer = _FakeVisualTransformer()
        self.ln_post = nn.LayerNorm(1024)
        self.proj = nn.Parameter(torch.randn(1024, 768) * 0.01)


class _FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _FakeVisual()
        self.transformer = _FakeTextTransformer()
        self.token_embedding = nn.Embedding(49408, 768)
        self.positional_embedding = nn.Parameter(torch.zeros(77, 768))
        self.attn_mask = None


def _token_ids(count=8):
    token_ids = torch.zeros(count, 77, dtype=torch.long)
    token_ids[:, -1] = 10
    return token_ids


def test_pre_fusion_text_norms_and_alpha_endpoints():
    torch.manual_seed(3)
    hard = F.normalize(torch.randn(3, 2, 768, 2), dim=2)
    dynamic = F.normalize(torch.randn(3, 2, 4, 768, 2), dim=3)

    fused_hard = H6Progress1._fuse_factor_bank(hard * 3.0, dynamic, hybrid_alpha=0.0)
    fused_dynamic = H6Progress1._fuse_factor_bank(hard, dynamic * 5.0, hybrid_alpha=1.0)

    assert torch.allclose(fused_hard, hard.unsqueeze(2).expand_as(fused_hard), atol=1e-6)
    assert torch.allclose(fused_dynamic, dynamic, atol=1e-6)
    assert torch.allclose(fused_hard.norm(dim=3), torch.ones_like(fused_hard.norm(dim=3)), atol=1e-6)
    assert torch.allclose(fused_dynamic.norm(dim=3), torch.ones_like(fused_dynamic.norm(dim=3)), atol=1e-6)


def test_router_dense_prediction_then_sparse_prediction_with_concept_keys():
    torch.manual_seed(4)
    router = PatchRouter(n_groups=2, num_factors=4, text_dim=8, bank_dim=4, hidden_dim=6, top_k=2, soft_routing_epochs=6)
    tokens = torch.randn(2, 1, 5, 8)
    concept_keys = torch.eye(4)

    dense_epoch = router(tokens, epoch_one_based=6, concept_keys=concept_keys)
    sparse_epoch = router(tokens, epoch_one_based=7, concept_keys=concept_keys)

    assert torch.allclose(dense_epoch["prediction_probabilities"], dense_epoch["dense_probabilities"], atol=1e-7)
    assert torch.allclose(sparse_epoch["prediction_probabilities"], sparse_epoch["sparse_probabilities"], atol=1e-7)
    assert (sparse_epoch["prediction_probabilities"] > 0).sum(dim=-1).eq(2).all()
    diagnostics = router.diagnostics(
        sparse_epoch["prediction_probabilities"],
        dense_probabilities=sparse_epoch["dense_probabilities"],
        sparse_probabilities=sparse_epoch["sparse_probabilities"],
        topk_indices=sparse_epoch["topk_indices"],
    )
    assert torch.allclose(
        diagnostics["selected_topk_frequency"].sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )


def test_h6_build_batch_returns_unit_dynamic_text_before_fusion():
    torch.manual_seed(4)
    model = ACDCLIP(_FakeClip(), n_groups=3, dfg_mode="mlp", h6_progress=1)
    visual = model(torch.randn(1, 3, 2, 2), return_phase4_features=True)
    model.h6.set_epoch(3)

    batch = model.h6.build_batch(model, "VisA", ["candle"], visual, hybrid_alpha=0.2)

    assert torch.allclose(batch["hard_adapted"].norm(dim=2), torch.ones(3, 1, 2), atol=1e-5)
    assert torch.allclose(batch["hard_frozen"].norm(dim=2), torch.ones(3, 1, 2), atol=1e-5)
    assert torch.allclose(batch["dynamic_text"].norm(dim=3), torch.ones(3, 1, 4, 2), atol=1e-5)
    assert torch.isfinite(batch["residual_diversity"]).all()


def test_factor_aware_center_loss_uses_dense_factor_assignment_and_detaches_router():
    projected = torch.zeros(1, 1, 4, 2, requires_grad=True)
    prototype_normal = torch.tensor([[[0.0, 0.0], [10.0, 10.0]]])
    prototype_abnormal = torch.tensor([[[10.0, 10.0], [0.0, 0.0]]])
    masks = torch.zeros(1, 1, 2, 2)
    labels = torch.tensor([0])

    good_assignment = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]], requires_grad=True)
    bad_assignment = torch.tensor([[[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]], requires_grad=True)
    good = factor_aware_center_loss(
        projected,
        prototype_normal,
        prototype_abnormal,
        good_assignment,
        masks,
        labels,
        detach_assignment=True,
    )
    bad = factor_aware_center_loss(
        projected,
        prototype_normal,
        prototype_abnormal,
        bad_assignment,
        masks,
        labels,
        detach_assignment=True,
    )
    assert good < bad

    good.backward(retain_graph=True)
    assert projected.grad is not None
    assert good_assignment.grad is None

    trainable_assignment = torch.full((1, 1, 4, 2), 0.5, requires_grad=True)
    loss = factor_aware_center_loss(
        projected.detach(),
        prototype_normal,
        prototype_abnormal,
        trainable_assignment,
        masks,
        labels,
        detach_assignment=False,
    )
    loss.backward()
    assert trainable_assignment.grad is not None


def test_h6_vae_beta_zero_then_linear_schedule():
    beta = 1e-4
    expected = [0.0, 0.0, 0.0, 0.0, 2.5e-5, 5.0e-5, 7.5e-5, 1.0e-4, 1.0e-4]
    actual = [get_h6_vae_beta(epoch, beta, zero_epochs=4) for epoch in range(1, 10)]
    for left, right in zip(actual, expected):
        assert abs(left - right) < 1e-12


def test_frozen_anchor_bypasses_trainable_text_adapter_layernorms_and_steps():
    torch.manual_seed(5)
    model = ACDCLIP(_FakeClip(), n_groups=3, dfg_mode="mlp", h6_progress=1)
    token_ids = _token_ids(4)

    before = [level.clone() for level in model.encode_frozen_anchor_text(token_ids)]
    with torch.no_grad():
        for layer_norm in model.text_adapter["layer_norms"]:
            layer_norm.weight.uniform_(2.0, 4.0)
            layer_norm.bias.uniform_(-3.0, -1.0)
    after_layernorm_mutation = model.encode_frozen_anchor_text(token_ids)
    for left, right in zip(before, after_layernorm_mutation):
        assert torch.allclose(left, right, atol=1e-6)

    optimizer = torch.optim.SGD(model.text_adapter.parameters(), lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    for parameter in model.text_adapter.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    after_optimizer_step = model.encode_frozen_anchor_text(token_ids)
    for left, right in zip(before, after_optimizer_step):
        assert torch.allclose(left, right, atol=1e-6)
        assert right.requires_grad is False


def test_dynamic_residual_diversity_is_finite_and_uses_dynamic_residuals():
    hard = F.normalize(torch.randn(1, 1, 768, 2), dim=2)
    residual_direction = F.normalize(torch.randn(1, 1, 1, 768), dim=-1)
    identical = hard.unsqueeze(2).repeat(1, 1, 4, 1, 1)
    identical[..., 1] = F.normalize(identical[..., 1] + 0.1 * residual_direction, dim=3)
    diverse = hard.unsqueeze(2).repeat(1, 1, 4, 1, 1)
    eye_dirs = F.normalize(torch.eye(4, 768).view(1, 1, 4, 768), dim=-1)
    diverse[..., 1] = F.normalize(diverse[..., 1] + 0.1 * eye_dirs, dim=3)
    zero_residual = hard.unsqueeze(2).repeat(1, 1, 4, 1, 1)

    identical_loss = dynamic_residual_diversity_loss(identical, hard)
    diverse_loss = dynamic_residual_diversity_loss(diverse, hard)
    zero_loss = dynamic_residual_diversity_loss(zero_residual, hard)

    assert torch.isfinite(zero_loss)
    assert identical_loss > diverse_loss


def test_vae_prompt_semantic_uses_decoder_mu_deterministically():
    torch.manual_seed(6)
    vae = ClassVAE(input_dim=8, hidden_dim=12, latent_dim=4)
    vae.train()
    cls = torch.randn(3, 8)

    torch.manual_seed(7)
    first = vae(cls)
    torch.manual_seed(8)
    second = vae(cls)

    assert torch.allclose(first["class_semantic"], second["class_semantic"], atol=1e-6)
    assert torch.allclose(first["class_semantic"].norm(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.allclose(first["decoded_mu"], second["decoded_mu"], atol=1e-6)
    assert not torch.allclose(first["reconstruction_sample"], second["reconstruction_sample"])
    assert torch.isfinite(first["kl"])
