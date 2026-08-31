from __future__ import annotations

import torch

from tools.cir_rmt.parameter_anchor import ImageParameterAnchor


def test_image_parameter_anchor_zero_at_reference_and_has_image_gradients() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.LayerNorm(2))
    reference = {name: value.detach().clone() for name, value in module.named_parameters()}
    anchor = ImageParameterAnchor(reference, checkpoint_sha256="abc", epoch=14, config_sha256="cfg", device=torch.device("cpu"))
    assert float(anchor.loss(module).detach()) == 0.0
    with torch.no_grad():
        next(module.parameters()).add_(0.1)
    loss = anchor.loss(module)
    loss.backward()
    assert float(loss.detach()) > 0.0
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_image_parameter_anchor_metadata_is_train_only() -> None:
    module = torch.nn.Linear(2, 2)
    reference = {name: value.detach().clone() for name, value in module.named_parameters()}
    anchor = ImageParameterAnchor(reference, checkpoint_sha256="abc", epoch=14, config_sha256="cfg", device=torch.device("cpu"))
    metadata = anchor.metadata(1.0e-3)
    assert metadata["enabled"] is True
    assert metadata["scope"] == "image_adapter_parameters_only"
    assert metadata["train_only"] is True
    disabled = anchor.metadata(0.0)
    assert disabled["enabled"] is False
    assert disabled["lambda_image_anchor"] == 0.0
