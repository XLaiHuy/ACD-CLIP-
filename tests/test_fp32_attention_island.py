import unittest

import torch
import torch.nn.functional as F

from model.transformer import ResidualAttentionBlock


def _explicit_fp32_reference(block: ResidualAttentionBlock, x: torch.Tensor):
    attention = block.attn
    q, k, v = F._in_projection_packed(
        x,
        x,
        x,
        attention.in_proj_weight,
        attention.in_proj_bias,
    )
    target_length, batch_size, embed_dim = q.shape
    num_heads = attention.num_heads
    head_dim = embed_dim // num_heads
    with torch.autocast(device_type="cuda", enabled=False):
        q = q.float().view(target_length, batch_size * num_heads, head_dim).transpose(0, 1)
        k = k.float().view(target_length, batch_size * num_heads, head_dim).transpose(0, 1)
        v = v.float().view(target_length, batch_size * num_heads, head_dim).transpose(0, 1)
        weights = torch.softmax(torch.bmm(q * (head_dim ** -0.5), k.transpose(1, 2)), dim=-1)
        output = torch.bmm(weights, v).transpose(0, 1).contiguous().view(
            target_length * batch_size, embed_dim
        )
        weights = weights.view(batch_size, num_heads, target_length, target_length).mean(dim=1)
    output = attention.out_proj(output.to(dtype=x.dtype)).view(target_length, batch_size, embed_dim)
    return output, weights


def _gradient_vector(block, x, repaired: bool):
    block.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        if repaired:
            output, weights = block.attention(x, x, x)
        else:
            output, weights = block.attn(x, x, x)
        loss = output.float().square().mean() + weights.float().square().mean()
    loss.backward()
    gradients = [x.grad.detach().float().reshape(-1)]
    gradients.extend(
        parameter.grad.detach().float().reshape(-1)
        for parameter in block.parameters()
        if parameter.grad is not None
    )
    return loss.detach(), output.detach(), weights.detach(), torch.cat(gradients)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
def test_fp32_reference_and_backward_are_finite():
    torch.manual_seed(7)
    block = ResidualAttentionBlock(32, 4).cuda().eval()
    x = torch.randn(16, 2, 32, device="cuda", dtype=torch.float16) * 0.1
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        repaired_output, repaired_weights = block.attention(x, x, x)
        reference_output, reference_weights = _explicit_fp32_reference(block, x)
    assert torch.isfinite(repaired_output).all()
    assert torch.isfinite(repaired_weights).all()
    assert torch.allclose(repaired_output, reference_output, atol=2e-3, rtol=2e-3)
    assert torch.allclose(repaired_weights, reference_weights, atol=2e-3, rtol=2e-3)

    _, _, _, gradient = _gradient_vector(block, x, repaired=True)
    assert torch.isfinite(gradient).all()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
def test_nonfailing_control_matches_original_direction():
    torch.manual_seed(11)
    block = ResidualAttentionBlock(32, 4).cuda().eval()
    x = torch.randn(16, 2, 32, device="cuda", dtype=torch.float16) * 0.1
    repaired_loss, repaired_output, repaired_weights, repaired_gradient = _gradient_vector(
        block, x, repaired=True
    )
    original_loss, original_output, original_weights, original_gradient = _gradient_vector(
        block, x, repaired=False
    )
    cosine = torch.dot(repaired_gradient, original_gradient) / (
        repaired_gradient.norm() * original_gradient.norm()
    )
    assert abs(float(repaired_loss - original_loss)) < 2e-3
    assert torch.allclose(repaired_output, original_output, atol=5e-3, rtol=5e-3)
    assert torch.allclose(repaired_weights, original_weights, atol=5e-3, rtol=5e-3)
    assert float(cosine) > 0.999
    assert 0.95 < float(repaired_gradient.norm() / original_gradient.norm()) < 1.05


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
def test_active_autocast_gate_with_fp32_input():
    torch.manual_seed(13)
    block = ResidualAttentionBlock(32, 4).cuda().eval()
    x = torch.randn(16, 2, 32, device="cuda", dtype=torch.float32) * 0.1
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        repaired_output, repaired_weights = block.attention(x, x, x)
        reference_output, reference_weights = _explicit_fp32_reference(block, x)
    assert torch.isfinite(repaired_output).all()
    assert torch.isfinite(repaired_weights).all()
    assert torch.allclose(repaired_output, reference_output, atol=2e-3, rtol=2e-3)
    assert torch.allclose(repaired_weights, reference_weights, atol=2e-3, rtol=2e-3)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
def test_known_fp16_overflow_is_repaired():
    block = ResidualAttentionBlock(32, 4).cuda().eval()
    x = torch.full((16, 2, 32), 1000.0, device="cuda", dtype=torch.float16)

    x_old = x.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        old_output, old_weights = block.attn(x_old, x_old, x_old)
        old_loss = old_output.float().square().mean()
    old_loss.backward()
    assert not (
        torch.isfinite(old_output).all()
        and torch.isfinite(old_weights).all()
        and torch.isfinite(x_old.grad).all()
    )

    x_repaired = x.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        repaired_output, repaired_weights = block.attention(x_repaired, x_repaired, x_repaired)
        repaired_loss = repaired_output.float().square().mean()
    repaired_loss.backward()
    assert torch.isfinite(repaired_output).all()
    assert torch.isfinite(repaired_weights).all()
    assert torch.isfinite(x_repaired.grad).all()


class TestFp32AttentionIsland(unittest.TestCase):
    def test_reference_and_backward(self):
        test_fp32_reference_and_backward_are_finite()

    def test_nonfailing_control(self):
        test_nonfailing_control_matches_original_direction()

    def test_active_autocast_gate(self):
        test_active_autocast_gate_with_fp32_input()

    def test_known_overflow(self):
        test_known_fp16_overflow_is_repaired()
