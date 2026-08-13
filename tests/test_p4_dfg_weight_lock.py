import torch
import torch.nn.functional as F

from model.adapter import ACDCLIP
from tests.test_h6_adapter_contract import _FakeClip


def _historical_mlp(model, image, text, group_index):
    weights = model.image_adapter["vision_text_gate"][group_index](
        image.mean(dim=1, keepdim=True)
    ).squeeze(1)
    weights = F.softmax(weights.view(image.shape[0], model.n_groups, 2), dim=1)
    return (text * weights.unsqueeze(2)).sum(dim=1)


def _historical_attention(model, image, text, group_index):
    v_gap = image.mean(dim=1)
    q = model.image_adapter["vision_text_q"][group_index](v_gap)
    normal = text[..., 0]
    abnormal = text[..., 1]
    key = model.image_adapter["vision_text_k"][group_index]
    scale = model.dfg_attn_dim**0.5 * model.dfg_attn_tau
    wn = F.softmax(torch.einsum("bd,bnd->bn", q, key(normal)) / scale, dim=1)
    wa = F.softmax(torch.einsum("bd,bnd->bn", q, key(abnormal)) / scale, dim=1)
    return torch.stack(
        [
            F.normalize(torch.einsum("bn,bnd->bd", wn, normal), dim=-1),
            F.normalize(torch.einsum("bn,bnd->bd", wa, abnormal), dim=-1),
        ],
        dim=-1,
    )


def test_compute_then_apply_reconstructs_historical_mlp_dfg():
    torch.manual_seed(13)
    model = ACDCLIP(_FakeClip(), n_groups=3, dfg_mode="mlp")
    image = torch.randn(2, 4, 768)
    text = torch.randn(2, 3, 768, 2)
    weights = model.compute_dfg_weights(image, text, 1)
    reconstructed = model.apply_dfg_weights(
        text, weights["normal"], weights["abnormal"]
    )
    expected = _historical_mlp(model, image, text, 1)
    assert torch.allclose(reconstructed, expected, atol=1e-6, rtol=0.0)


def test_compute_then_apply_reconstructs_historical_attention_dfg():
    torch.manual_seed(17)
    model = ACDCLIP(
        _FakeClip(), n_groups=3, dfg_mode="attn", dfg_attn_dim=32
    )
    image = torch.randn(2, 4, 768)
    text = torch.randn(2, 3, 768, 2)
    weights = model.compute_dfg_weights(image, text, 1)
    reconstructed = model.apply_dfg_weights(
        text, weights["normal"], weights["abnormal"]
    )
    expected = _historical_attention(model, image, text, 1)
    assert torch.allclose(reconstructed, expected, atol=1e-6, rtol=0.0)


def test_locked_weights_do_not_receive_dynamic_path_gradient():
    torch.manual_seed(19)
    model = ACDCLIP(_FakeClip(), n_groups=3, dfg_mode="attn", dfg_attn_dim=32)
    image = torch.randn(2, 4, 768, requires_grad=True)
    base_text = torch.randn(2, 3, 768, 2, requires_grad=True)
    dynamic_text = torch.randn(2, 3, 768, 2, requires_grad=True)
    weights = model.compute_dfg_weights(image, base_text, 0)
    fused = model.apply_dfg_weights(
        dynamic_text, weights["normal"].detach(), weights["abnormal"].detach()
    )
    fused.sum().backward()

    assert image.grad is None
    assert base_text.grad is None
    assert dynamic_text.grad is not None
