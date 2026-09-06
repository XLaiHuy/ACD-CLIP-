"""Contracts for the three explicit H2 precision protocols."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from h2_clean.contract import build_full_checkpoint, make_dataloader_generator
from h2_clean.precision import (
    BF16_V1,
    HISTORICAL_MIXED_FP16_FP32_V1,
    REPAIRED_MIXED_FP16_FP32_EXPERIMENTAL,
    resolve_precision_runtime_mode,
)
from model.transformer import ResidualAttentionBlock


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_adapter = nn.Linear(3, 3)
        self.text_adapter = nn.Linear(3, 3)


@pytest.mark.parametrize(
    ("name", "precision", "scaler", "islands"),
    [
        (HISTORICAL_MIXED_FP16_FP32_V1, "fp16", True, False),
        (BF16_V1, "bf16", False, False),
        (REPAIRED_MIXED_FP16_FP32_EXPERIMENTAL, "fp16", True, True),
    ],
)
def test_named_precision_protocols_are_distinct(name, precision, scaler, islands):
    mode = resolve_precision_runtime_mode(name, None)
    assert mode.protocol_name == name
    assert mode.policy.name == precision
    assert mode.policy.gradscaler_enabled is scaler
    assert mode.later_transformer_fp32_islands is islands


def test_named_protocol_rejects_conflicting_low_level_flags():
    with pytest.raises(ValueError):
        resolve_precision_runtime_mode(HISTORICAL_MIXED_FP16_FP32_V1, "bf16")
    with pytest.raises(ValueError):
        resolve_precision_runtime_mode(HISTORICAL_MIXED_FP16_FP32_V1, None, legacy_local_fp32_islands=True)
    with pytest.raises(ValueError):
        resolve_precision_runtime_mode(BF16_V1, None, legacy_amp=True)


def test_historical_mode_uses_native_attention_and_native_mlp(monkeypatch):
    block = ResidualAttentionBlock(8, 2, idx=8)
    block.enable_fp16_numerical_islands = False
    native_calls = 0
    original_attention = block.attn.forward

    def counted_attention(*args, **kwargs):
        nonlocal native_calls
        native_calls += 1
        return original_attention(*args, **kwargs)

    monkeypatch.setattr(block.attn, "forward", counted_attention)
    monkeypatch.setattr(
        block,
        "_fp32_mlp",
        lambda _x: (_ for _ in ()).throw(AssertionError("later FP32 MLP island activated")),
    )
    output, _ = block(torch.randn(4, 2, 8))
    assert output.shape == (4, 2, 8)
    assert native_calls == 1


def test_historical_protocol_identity_is_written_to_checkpoint():
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    payload = build_full_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
        global_step=1,
        config={"epoch": 15, "seed": 0},
        parent_config={"epoch": 15, "seed": 0},
        operational_config={},
        repo=".",
        clip_sha256="clip",
        dataset_manifest_sha256="manifest",
        dataloader_generator=make_dataloader_generator(0),
        anchor=None,
        anchor_lambda=0.0,
        seed=0,
        precision="fp16",
        precision_protocol=HISTORICAL_MIXED_FP16_FP32_V1,
        later_transformer_fp32_islands=False,
        tf32_enabled=False,
    )
    assert payload["precision_protocol"] == HISTORICAL_MIXED_FP16_FP32_V1
    assert payload["precision"] == "fp16"
    assert payload["later_transformer_fp32_islands"] is False
    assert payload["amp_enabled"] is True
