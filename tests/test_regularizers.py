import torch
from torch import nn

from train import compute_hybrid_k_regularization
import utils


class KOnlyModel:
    def __init__(self):
        self.image_adapter = nn.ModuleDict(
            {
                "vision_text_k": nn.ModuleList(
                    [nn.Linear(6, 6, bias=True), nn.Linear(6, 6, bias=True)]
                )
            }
        )


def test_k_regularizer_detaches_wk_and_reaches_soft_branch():
    torch.manual_seed(23)
    hard = torch.randn(3, 6, 2, requires_grad=True)
    soft_seed = hard.detach() + 0.3 * torch.randn(3, 6, 2)
    for alpha in (0.0, 0.05, 0.1, 0.2):
        model = KOnlyModel()
        soft = soft_seed.clone().requires_grad_(True)
        loss, stats = compute_hybrid_k_regularization(model, hard, soft, alpha)
        assert torch.isfinite(loss)
        assert loss.item() >= -1e-6
        assert stats
        loss.backward()
        assert all(parameter.grad is None for parameter in model.image_adapter.parameters())
        assert hard.grad is None
        if alpha == 0.0:
            assert soft.grad is not None
            assert torch.linalg.vector_norm(soft.grad).item() <= 1e-7
        else:
            assert soft.grad is not None
            assert torch.isfinite(soft.grad).all()
            assert torch.linalg.vector_norm(soft.grad).item() > 1e-8


class FakeTokenized:
    def to(self, device):
        return self


class FakeSoftPrompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Parameter(torch.randn(2, 6))

    def get_context(self, state_idx):
        return self.context[state_idx]


class FakePromptModel:
    def __init__(self):
        self.soft_prompt_ctx_len = 4
        self.soft_prompt = FakeSoftPrompt()

    def encode_soft_prompt_text(self, tokenized, ctx, adapt_text=False):
        return [ctx.unsqueeze(0)]


def test_kg_regularizer_gradient_reaches_soft_prompt_only(monkeypatch):
    model = FakePromptModel()
    hard = torch.randn(1, 6, 2)
    hard = torch.nn.functional.normalize(hard, dim=1)
    monkeypatch.setattr(utils, "get_real_name", lambda dataset, name: name)
    monkeypatch.setattr(utils, "get_soft_prompt_sentence", lambda *args: "stub")
    monkeypatch.setattr(utils, "tokenize", lambda sentences: FakeTokenized())
    monkeypatch.setattr(utils, "get_hard_anchor_single_class_text_embedding", lambda *args: hard)
    _, kg_loss, stats = utils.get_soft_prompt_single_class_text_embedding(
        model, "stub_dataset", "stub_class", torch.device("cpu"), return_kg=True
    )
    assert torch.isfinite(kg_loss)
    assert kg_loss.item() >= -1e-6
    assert stats["kg_loss_normal"] >= -1e-6
    kg_loss.backward()
    assert model.soft_prompt.context.grad is not None
    assert torch.isfinite(model.soft_prompt.context.grad).all()
    assert torch.linalg.vector_norm(model.soft_prompt.context.grad).item() > 1e-8
