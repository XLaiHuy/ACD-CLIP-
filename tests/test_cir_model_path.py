import copy

import torch
from torch import nn

from model.adapter import ACDCLIP


def make_minimal_model() -> ACDCLIP:
    model = ACDCLIP.__new__(ACDCLIP)
    nn.Module.__init__(model)
    model.n_groups = 2
    model.dfg_mode = "mlp"
    model.image_adapter = nn.ModuleDict({
        "vision_text_gate": nn.ModuleList([
            nn.Linear(8, 4),
            nn.Linear(8, 4),
        ])
    })
    return model


def test_model_cir_alpha_zero_preserves_output_and_gradients():
    torch.manual_seed(23)
    base = make_minimal_model()
    native_model = copy.deepcopy(base)
    zero_model = copy.deepcopy(base)
    features = torch.randn(2, 2, 4, 8)
    text = torch.randn(2, 2, 8, 2)

    native_features = features.clone().requires_grad_(True)
    native_output = native_model.vision_text_fusion_gate_seg(
        native_features,
        text,
        img_size=2,
        cir_training=False,
    )
    native_grads = torch.autograd.grad(
        native_output.sum(),
        [native_features, *native_model.image_adapter.parameters()],
    )

    zero_features = features.clone().requires_grad_(True)
    zero_output = zero_model.vision_text_fusion_gate_seg(
        zero_features,
        text,
        img_size=2,
        cir_training=True,
        cir_alpha=0.0,
    )
    zero_grads = torch.autograd.grad(
        zero_output.sum(),
        [zero_features, *zero_model.image_adapter.parameters()],
    )

    torch.testing.assert_close(zero_output, native_output, rtol=0.0, atol=0.0)
    for zero_grad, native_grad in zip(zero_grads, native_grads):
        torch.testing.assert_close(zero_grad, native_grad, rtol=0.0, atol=0.0)
    assert zero_model._last_cir_stats["enabled"] is False
