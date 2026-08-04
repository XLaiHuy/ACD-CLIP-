import sys
import types

import torch
import torch.nn.functional as F
from torch import nn

from model.h6.losses import factor_stage_diagnostics, teacher_candidate_diagnostics
from model.h6.router import PatchRouter


class _TinySoftPrompt(nn.Module):
    def __init__(self, text_dim=8, ctx_len=4):
        super().__init__()
        self.ctx_normal = nn.Parameter(torch.randn(ctx_len, text_dim))
        self.ctx_abnormal = nn.Parameter(torch.randn(ctx_len, text_dim))


class _TinyDynamicTextModel(nn.Module):
    def __init__(self, text_dim=8, ctx_len=4, n_groups=3):
        super().__init__()
        self.soft_prompt = _TinySoftPrompt(text_dim=text_dim, ctx_len=ctx_len)
        self.n_groups = int(n_groups)
        self.text_dim = int(text_dim)

    def encode_dynamic_prompt_text(self, token_ids, contexts, adapt_text=False):
        pooled = F.normalize(contexts.float().mean(dim=1), dim=-1)
        return [pooled + 0.01 * level for level in range(self.n_groups)]


def test_router_is_per_patch_not_mean_pool_broadcast():
    torch.manual_seed(0)
    router = PatchRouter(n_groups=1, num_factors=4, text_dim=8, bank_dim=4, hidden_dim=12, top_k=2)
    tokens = torch.zeros(1, 1, 4, 8)
    tokens[0, 0, 0, 0] = 1.0
    tokens[0, 0, 1, 1] = 1.0
    tokens[0, 0, 2, 2] = 1.0
    tokens[0, 0, 3, 3] = 1.0
    keys = F.normalize(torch.randn(4, 4), dim=-1)
    out = router(tokens, epoch_one_based=12, concept_keys=keys)
    assert out["queries"].shape == (1, 1, 4, 4)
    assert out["logits"].shape == (1, 1, 4, 4)
    assert out["dense_probabilities"].shape == (1, 1, 4, 4)
    assert out["router_softmax_dim"].item() == 3
    assert out["router_topk_dim"].item() == 3
    assert not torch.allclose(out["queries"][0, 0, 0], out["queries"][0, 0, 1])
    assert out["query_variance_across_patches"].item() > 0


def test_router_diagnostics_contains_loggable_patch_query_keys(monkeypatch):
    torch.manual_seed(2)
    from model.h6.model import H6Progress1

    def _fake_hard_embedding(base_model, dataset_name, class_name, device, adapt_text=True):
        values = torch.randn(3, 8, 2, device=device)
        return F.normalize(values, dim=1)

    fake_utils = types.ModuleType("utils")
    fake_utils.get_real_name = lambda dataset_name, class_name: class_name
    fake_utils.get_soft_prompt_sentence = lambda real_name, state, ctx_len: f"{real_name}-{state}-{ctx_len}"
    fake_utils.get_hard_phase1_single_class_text_embedding = _fake_hard_embedding
    fake_utils.get_hard_anchor_single_class_text_embedding = (
        lambda base_model, dataset_name, class_name, device: _fake_hard_embedding(
            base_model, dataset_name, class_name, device, adapt_text=False
        )
    )
    fake_tokenizer = types.ModuleType("model.tokenizer")
    fake_tokenizer.tokenize = lambda sentences: torch.zeros(len(sentences), 77, dtype=torch.long)
    monkeypatch.setitem(sys.modules, "utils", fake_utils)
    monkeypatch.setitem(sys.modules, "model.tokenizer", fake_tokenizer)

    h6 = H6Progress1(n_groups=3, num_factors=4, bank_dim=8, text_dim=8, router_dim=12)
    visual = {
        "seg_tokens_pre_l2": [torch.randn(1, 4, 8) for _ in range(3)],
        "seg_tokens": [torch.randn(1, 4, 8) for _ in range(3)],
        "cls24": torch.randn(1, 8),
    }
    batch = h6.build_batch(
        base_model=_TinyDynamicTextModel(text_dim=8, ctx_len=4),
        dataset_name="VisA",
        class_names=["candle"],
        visual_output=visual,
        hybrid_alpha=0.0,
    )
    diagnostics = batch["router_diagnostics"]
    for key in (
        "router_patch_count",
        "router_softmax_dim",
        "router_topk_dim",
        "query_pairwise_cos_mean_across_patches",
        "query_pairwise_cos_max_across_patches",
        "query_variance_across_patches",
        "query_effective_rank",
        "query_singular_value_ratio",
        "per_factor_logit_std_across_patches",
    ):
        assert key in diagnostics
        assert torch.is_tensor(diagnostics[key])


def test_teacher_candidates_report_patchwise_nonuniformity():
    projected = torch.zeros(1, 1, 4, 3)
    projected[0, 0, :, 0] = torch.tensor([1.0, 0.5, -0.5, -1.0])
    projected[0, 0, :, 1] = torch.tensor([0.0, 0.5, 0.5, 0.0])
    proto = F.normalize(torch.tensor([[[1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0],
                                       [-1.0, 0.0, 0.0],
                                       [0.0, -1.0, 0.0]]]), dim=-1)
    masks = torch.zeros(1, 1, 2, 2)
    labels = torch.tensor([0])
    diag = teacher_candidate_diagnostics(projected, proto, proto, masks, labels, temperature=0.15)
    assert diag["teacher_raw_candidate_entropy"].shape == (1,)
    assert diag["teacher_centered_candidate_probability_std_across_patches"].shape == (1,)
    assert diag["teacher_distance_candidate_unique_topk_pairs"].shape == (1,)
    assert torch.isfinite(diag["teacher_raw_candidate_entropy"]).all()
    assert diag["teacher_raw_candidate_probability_std_across_patches"].item() > 0


def test_factor_stage_diagnostics_finds_identity_loss_point():
    separated = torch.eye(4, 8)
    collapsed = torch.ones(4, 8)
    sep = factor_stage_diagnostics(separated, "sep", factor_dim=0)
    col = factor_stage_diagnostics(collapsed, "col", factor_dim=0)
    assert sep["sep_cos_max"].item() < 0.5
    assert col["col_cos_mean"].item() > 0.99
    assert sep["sep_l2_min"].item() > col["col_l2_min"].item()
