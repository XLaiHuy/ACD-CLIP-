import torch
from torch import nn

from model.adapter import ACDCLIP


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


def test_adapter_legacy_and_phase4_visual_contracts():
    torch.manual_seed(0)
    legacy = ACDCLIP(_FakeClip(), n_groups=3, dfg_mode="mlp", h6_progress=0)
    seg_tokens, det_tokens = legacy(torch.randn(1, 3, 2, 2))
    assert len(seg_tokens) == len(det_tokens) == 3
    assert seg_tokens[0].shape == (1, 4, 768)
    assert det_tokens[0].shape == (1, 768)

    phase4 = ACDCLIP(_FakeClip(), n_groups=3, dfg_mode="mlp", h6_progress=1)
    output = phase4(torch.randn(1, 3, 2, 2), return_phase4_features=True)
    assert set(output) == {"seg_tokens", "seg_tokens_pre_l2", "det_tokens", "cls24"}
    assert output["seg_tokens_pre_l2"][0].shape == (1, 4, 768)
    assert output["cls24"].shape == (1, 768)
    assert torch.isfinite(output["cls24"]).all()
    token_ids = torch.zeros(8, 77, dtype=torch.long)
    token_ids[:, -1] = 10
    dynamic_levels = phase4.encode_dynamic_prompt_text(token_ids, torch.randn(8, 4, 768), adapt_text=False)
    assert len(dynamic_levels) == 3
    assert dynamic_levels[0].shape == (8, 768)
    phase4.h6.set_epoch(3)
    batch = phase4.h6.build_batch(phase4, "VisA", ["candle"], output, hybrid_alpha=0.05)
    assert batch["factor_bank"].shape == (3, 1, 4, 768, 2)
    assert batch["text_global"].shape == (3, 1, 768, 2)
    assert batch["h6_logits"].shape == (3, 1, 4)
    assert (batch["probabilities"] > 0).sum(dim=-1).eq(2).all()
