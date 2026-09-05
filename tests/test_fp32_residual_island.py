"""Focused tests for the approved post-adapter visual residual repair.

The exact E7 state replay is run by the source-only validation harness.  These
tests keep the local operation-level contract fast and deterministic: the
repaired post-adapter block is compared with the same operations evaluated in
an explicit FP32 reference, while a pre-adapter control block exercises the
unchanged AMP path.
"""

import unittest

import torch
import torch.nn.functional as F

from model.transformer import ResidualAttentionBlock


def _reference_post_adapter_block(block: ResidualAttentionBlock, x: torch.Tensor):
    """Explicit reference for the repaired branch's mathematical operations."""
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        q_x = block.ln_1(x)
        q, k, v = F._in_projection_packed(
            q_x,
            q_x,
            q_x,
            block.attn.in_proj_weight,
            block.attn.in_proj_bias,
        )
    target_length, batch_size, embed_dim = q.shape
    head_dim = embed_dim // block.attn.num_heads
    with torch.autocast(device_type="cuda", enabled=False):
        q = q.float().view(target_length, batch_size * block.attn.num_heads, head_dim).transpose(0, 1)
        k = k.float().view(target_length, batch_size * block.attn.num_heads, head_dim).transpose(0, 1)
        v = v.float().view(target_length, batch_size * block.attn.num_heads, head_dim).transpose(0, 1)
        weights = torch.softmax(torch.bmm(q * (head_dim ** -0.5), k.transpose(1, 2)), dim=-1)
        attention_output = torch.bmm(weights, v).transpose(0, 1).contiguous().view(
            target_length * batch_size, embed_dim
        )
        attention_output = F.linear(
            attention_output,
            block.attn.out_proj.weight.float(),
            block.attn.out_proj.bias.float(),
        ).view(target_length, batch_size, embed_dim)
        weights = weights.view(batch_size, block.attn.num_heads, target_length, target_length).mean(dim=1)

        after_attention = x.float() + block.ls_1(attention_output)
        hidden = F.linear(
            block.ln_2(after_attention).float(),
            block.mlp.c_fc.weight.float(),
            block.mlp.c_fc.bias.float(),
        )
        hidden = block.mlp.gelu(hidden)
        hidden = F.linear(hidden, block.mlp.c_proj.weight.float(), block.mlp.c_proj.bias.float())
        hidden = block.ls_2(hidden)
        output = after_attention + hidden
    return output, weights


def _run_block(block, x):
    block.zero_grad(set_to_none=True)
    value = x.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output, weights = block(value)
        loss = output.float().square().mean() + weights.float().square().mean()
    loss.backward()
    grads = [value.grad.detach().float().reshape(-1)]
    grads.extend(
        parameter.grad.detach().float().reshape(-1)
        for parameter in block.parameters()
        if parameter.grad is not None
    )
    return output.detach(), weights.detach(), torch.cat(grads), value


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
def test_post_adapter_reference_equivalence_and_gradient_propagation():
    torch.manual_seed(2026)
    block = ResidualAttentionBlock(1024, 16, idx=8).cuda().eval()
    x = (torch.randn(6, 2, 1024, device="cuda", dtype=torch.float32) * 0.05).requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        repaired_output, repaired_weights = block(x)
    reference_output, reference_weights = _reference_post_adapter_block(block, x)

    assert torch.isfinite(repaired_output).all()
    assert torch.isfinite(repaired_weights).all()
    assert repaired_output.dtype == torch.float32
    assert repaired_weights.dtype == torch.float32
    assert torch.allclose(repaired_output, reference_output, atol=3e-3, rtol=3e-3)
    assert torch.allclose(repaired_weights, reference_weights, atol=3e-3, rtol=3e-3)

    output, weights, gradient, value = _run_block(block, x)
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert torch.isfinite(gradient).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    for name in ("attn.out_proj.weight", "mlp.c_proj.weight", "mlp.c_fc.weight"):
        parameter = dict(block.named_parameters())[name]
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.dtype == torch.float32, name


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
def test_pre_adapter_control_retains_original_amp_path():
    torch.manual_seed(2027)
    block = ResidualAttentionBlock(1024, 16, idx=7).cuda().eval()
    x = torch.randn(6, 2, 1024, device="cuda", dtype=torch.float32) * 0.05
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        repaired_output, repaired_weights = block(x)
        q_x = block.ln_1(x)
        original_output, original_weights = block.attn(q_x, q_x, q_x)
        original = q_x.new_empty(q_x.shape)
        original.copy_(x + block.ls_1(original_output))
        original = original + block.ls_2(block.mlp(block.ln_2(original)))
    assert repaired_output.dtype == original.dtype
    assert repaired_weights.dtype == original_weights.dtype
    assert torch.allclose(repaired_output, original, atol=5e-3, rtol=5e-3)
    assert torch.allclose(repaired_weights, original_weights, atol=5e-3, rtol=5e-3)


class TestFp32ResidualIsland(unittest.TestCase):
    def test_post_adapter_reference_and_gradients(self):
        test_post_adapter_reference_equivalence_and_gradient_propagation()

    def test_pre_adapter_control(self):
        test_pre_adapter_control_retains_original_amp_path()


if __name__ == "__main__":
    unittest.main()
