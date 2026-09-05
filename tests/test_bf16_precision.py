"""Focused contracts for the explicit BF16 H2 precision policy."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from h2_clean.contract import (
    build_full_checkpoint,
    make_dataloader_generator,
    parent_scientific_config,
    restore_full_checkpoint,
    validate_resume_identity,
)
from h2_clean.precision import PrecisionPolicy, resolve_precision_policy


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_adapter = nn.Linear(3, 3)
        self.text_adapter = nn.Linear(3, 3)


def test_bf16_policy_has_autocast_but_no_gradscaler():
    policy = resolve_precision_policy("bf16")
    assert policy.autocast_enabled
    assert policy.autocast_dtype is torch.bfloat16
    assert not policy.gradscaler_enabled
    assert resolve_precision_policy(None, legacy_amp=True).name == "fp16"
    with pytest.raises(ValueError):
        resolve_precision_policy("fp32", legacy_amp=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16 autocast")
def test_bf16_autocast_preserves_fp32_parameter_storage():
    policy = PrecisionPolicy("bf16")
    linear = nn.Linear(8, 8, device="cuda", dtype=torch.float32)
    x = torch.randn(4, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    with policy.autocast("cuda"):
        output = linear(x)
        assert output.dtype is torch.bfloat16
        loss = output.float().square().mean()
    loss.backward()
    assert linear.weight.dtype is torch.float32
    assert linear.weight.grad.dtype is torch.float32
    assert torch.isfinite(linear.weight.grad).all()


def test_bf16_checkpoint_has_empty_scaler_state_and_rejects_stale_precision(tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    config = {
        "epoch": 15,
        "seed": 1,
        "amp": True,
        "precision": "bf16",
        "bf16_local_fp32_islands": False,
        "tf32_enabled": False,
    }
    payload = build_full_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        epoch=1,
        global_step=1,
        config=config,
        parent_config=parent_scientific_config(config),
        operational_config={},
        repo=".",
        clip_sha256="clip",
        dataset_manifest_sha256="manifest",
        dataloader_generator=make_dataloader_generator(1),
        anchor=None,
        anchor_lambda=0.0,
        seed=1,
        precision="bf16",
        tf32_enabled=False,
    )
    assert payload["scaler_state"] == {}
    assert payload["amp_enabled"] is True
    assert payload["gradscaler_enabled"] is False
    validate_resume_identity(
        payload,
        expected_scientific_config=config,
        expected_parent_config=parent_scientific_config(config),
        expected_epoch=1,
        expected_total_epoch=15,
        expected_seed=1,
        expected_clip_sha256="clip",
        expected_manifest_sha256="manifest",
    )
    restored = _TinyModel()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.9)
    checkpoint = tmp_path / "bf16.pth"
    torch.save(payload, checkpoint)
    restore_full_checkpoint(
        torch.load(checkpoint, map_location="cpu", weights_only=False),
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=None,
        dataloader_generator=make_dataloader_generator(1),
        expected_scientific_config=config,
        expected_parent_config=parent_scientific_config(config),
        expected_epoch=1,
        expected_total_epoch=15,
        expected_seed=1,
        expected_clip_sha256="clip",
        expected_manifest_sha256="manifest",
    )
    stale = dict(config, precision="fp16")
    with pytest.raises(ValueError):
        validate_resume_identity(
            payload,
            expected_scientific_config=stale,
            expected_parent_config=parent_scientific_config(stale),
        )
