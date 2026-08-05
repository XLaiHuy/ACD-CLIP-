import torch
import torch.nn.functional as F

from model.h6.paired_experts import FOFSPairedSemanticExperts


def _inputs():
    bank = F.normalize(torch.randn(2, 3, 4, 32, 2), dim=3)
    return bank, torch.randn(3, 4, 8), torch.randn(3, 4, 8)


def test_fofs_is_frozen_and_cross_expert_orthogonal():
    experts = FOFSPairedSemanticExperts(4, 32, 8, bottleneck=4, seed_offset=17)
    gram = torch.einsum("mrd,nsd->mnrs", experts.fofs_A, experts.fofs_A)
    eye = torch.eye(4)
    assert not experts.fofs_A.requires_grad
    assert torch.allclose(gram[0, 0], eye, atol=1e-5)
    assert gram[0, 1].abs().max() < 1e-5


def test_zero_expert_is_identity_and_b_gets_gradient():
    experts = FOFSPairedSemanticExperts(4, 32, 8, bottleneck=4)
    bank, normal, abnormal = _inputs()
    result = experts(bank, normal, abnormal, scale=.1)
    assert torch.allclose(result["expert_factor_bank"], bank, atol=2e-6)
    result["expert_factor_bank"].sum().backward()
    assert experts.expert_B.grad is not None


def test_paired_delta_is_tangent_and_bounded():
    experts = FOFSPairedSemanticExperts(4, 32, 8, bottleneck=4, max_relative_ratio=.1)
    experts.expert_B.data.normal_(0, .2)
    bank, normal, abnormal = _inputs()
    output = experts(bank, normal, abnormal, scale=.1)
    base = F.normalize(bank[..., 1] - bank[..., 0], dim=-1)
    tangent = (output["expert_delta_tangent"] * base).sum(-1)
    assert tangent.abs().max() < 1e-4
    assert output["expert_relative_ratio"].max() <= .100001
    assert torch.allclose(output["expert_factor_bank"][..., 0], F.normalize(bank[..., 0] - output["expert_applied_delta"], dim=-1))
