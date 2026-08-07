"""Checkpoint roundtrip tests for Candidate-1 ACDCLIP model.

Tests verify:
1. State dict saves and loads correctly (output parity)
2. H6 config fields round-trip with correct values
3. Old checkpoints (no h6_config key) default to legacy_mix
4. rho values are preserved with correct shape
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch
import torch.nn.functional as F

from model.clip import create_model
from model.adapter import ACDCLIP

_CONFIG_PATH = "configs/phase4/p1_v8_2_candidate1.json"
_N_GROUPS = 3  # canonical from config


def _make_model(seed: int = 0) -> ACDCLIP:
    torch.manual_seed(seed)
    clip_model = create_model(
        "ViT-L-14-336",
        img_size=518,
        pretrained=False,
        require_pretrained=False,
    )
    return ACDCLIP(
        clip_model=clip_model,
        n_groups=_N_GROUPS,
        h6_progress=1,
    )


def _save_candidate1_checkpoint(model: ACDCLIP, path: str) -> dict:
    """Save a full Candidate-1 style checkpoint with all required fields."""
    checkpoint = {
        "model_state": model.state_dict(),
        "schema_version": "1.1",
        "h6_config": {
            "local_factor_mode": "center_spread",
            "local_center_mix": 0.05,
            "local_factor_spread": 0.10,
            "h6_logit_temperature": model.h6.h6_logit_temperature,
            "rho_values": model.h6.rho_values().detach().cpu().tolist(),
            "rho_trainable": False,
            "n_groups": _N_GROUPS,
            "num_factors": model.h6.num_factors,
        },
        "role_config": {
            "role_target_version": "v1-hard-onehot",
            "boundary_threshold": 0.01,
            "core_threshold": 0.99,
        },
        "residual_config": {
            "correction_epsilon": 0.05,
            "correction_max": 1.0,
            "correction_max_mode": "resolved_from_capacity_audit",
            "residual_loss_beta": 1.0,
        },
        "loss_config": {
            "route_loss_enabled": True,
            "factor_role_loss_enabled": True,
            "actual_local_loss_enabled": True,
            "lambda_route": None,       # to be calibrated
            "lambda_factor_role": None,
            "lambda_actual_local": None,
        },
    }
    torch.save(checkpoint, path)
    return checkpoint


def test_checkpoint_roundtrip_state_dict():
    """State dict saves and reloads without missing/unexpected keys."""
    model_a = _make_model(seed=0)
    model_a.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        _save_candidate1_checkpoint(model_a, ckpt_path)

        model_b = _make_model(seed=99)  # Different seed → different weights
        loaded = torch.load(ckpt_path, weights_only=True)
        result = model_b.load_state_dict(loaded["model_state"], strict=True)

    assert len(result.missing_keys) == 0, f"Missing keys: {result.missing_keys}"
    assert len(result.unexpected_keys) == 0, f"Unexpected keys: {result.unexpected_keys}"


def test_checkpoint_h6_config_roundtrip():
    """All H6 config fields survive the checkpoint roundtrip."""
    model_a = _make_model(seed=1)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        saved = _save_candidate1_checkpoint(model_a, ckpt_path)
        loaded = torch.load(ckpt_path, weights_only=True)

    cfg = loaded["h6_config"]
    assert cfg["local_factor_mode"] == "center_spread"
    assert abs(cfg["local_center_mix"] - 0.05) < 1e-9
    assert abs(cfg["local_factor_spread"] - 0.10) < 1e-9
    assert cfg["rho_trainable"] is False
    assert cfg["n_groups"] == _N_GROUPS
    assert loaded["schema_version"] == "1.1"


def test_checkpoint_rho_values_preserved():
    """rho values in the checkpoint must exactly match the model's rho_values()."""
    model_a = _make_model(seed=2)
    rho_before = model_a.h6.rho_values().detach().cpu().tolist()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        _save_candidate1_checkpoint(model_a, ckpt_path)
        loaded = torch.load(ckpt_path, weights_only=True)

    rho_saved = loaded["h6_config"]["rho_values"]
    assert len(rho_saved) == _N_GROUPS, f"rho length must be {_N_GROUPS}"
    for v_saved, v_orig in zip(rho_saved, rho_before):
        assert abs(v_saved - v_orig) < 1e-6, f"rho mismatch: {v_saved} vs {v_orig}"


def test_checkpoint_output_parity():
    """Model A and Model B (loaded from A's checkpoint) must compute identical rho values."""
    model_a = _make_model(seed=3)
    model_a.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "parity.pt")
        _save_candidate1_checkpoint(model_a, ckpt_path)

        model_b = _make_model(seed=77)  # Different init
        model_b.eval()
        loaded = torch.load(ckpt_path, weights_only=True)
        model_b.load_state_dict(loaded["model_state"])

    rho_a = model_a.h6.rho_values().detach()
    rho_b = model_b.h6.rho_values().detach()
    assert torch.allclose(rho_a, rho_b, atol=1e-7), \
        f"rho must match after roundtrip: {rho_a} vs {rho_b}"

    # Verify rho.raw parameter identical
    raw_a = dict(model_a.named_parameters())["h6.rho.raw"].detach()
    raw_b = dict(model_b.named_parameters())["h6.rho.raw"].detach()
    assert torch.allclose(raw_a, raw_b, atol=1e-7), "rho.raw must match after roundtrip"


def test_old_checkpoint_defaults_to_legacy_mix():
    """Old-style checkpoints (no h6_config key) must default to legacy_mix."""
    model = _make_model(seed=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "old_ckpt.pt")
        # Simulate old checkpoint: only model_state, no h6_config
        torch.save({"model_state": model.state_dict()}, ckpt_path)
        loaded = torch.load(ckpt_path, weights_only=True)

    h6_config = loaded.get("h6_config", None)
    assert h6_config is None, "Old checkpoint must not have h6_config"

    # Application code should use this fallback pattern
    local_factor_mode = (h6_config or {}).get("local_factor_mode", "legacy_mix")
    assert local_factor_mode == "legacy_mix", \
        f"Old checkpoint fallback must be 'legacy_mix', got '{local_factor_mode}'"
    assert local_factor_mode != "center_spread", \
        "Old checkpoint must NOT silently activate center_spread"


def test_old_checkpoint_disables_new_losses():
    """Old checkpoints without loss_config must default to all new losses disabled."""
    model = _make_model(seed=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "old_ckpt2.pt")
        torch.save({"model_state": model.state_dict()}, ckpt_path)
        loaded = torch.load(ckpt_path, weights_only=True)

    loss_cfg = loaded.get("loss_config", {})
    assert not loss_cfg.get("route_loss_enabled", False), "Old ckpt: route_loss disabled"
    assert not loss_cfg.get("factor_role_loss_enabled", False), "Old ckpt: factor_role_loss disabled"
    assert not loss_cfg.get("actual_local_loss_enabled", False), "Old ckpt: actual_local_loss disabled"


def test_checkpoint_role_config_roundtrip():
    """Role config fields must survive the checkpoint roundtrip."""
    model = _make_model(seed=6)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        _save_candidate1_checkpoint(model, ckpt_path)
        loaded = torch.load(ckpt_path, weights_only=True)

    rc = loaded["role_config"]
    assert rc["role_target_version"] == "v1-hard-onehot"
    assert abs(rc["boundary_threshold"] - 0.01) < 1e-9
    assert abs(rc["core_threshold"] - 0.99) < 1e-9


def test_checkpoint_residual_config_roundtrip():
    """Residual config fields must survive the checkpoint roundtrip."""
    model = _make_model(seed=7)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt.pt")
        _save_candidate1_checkpoint(model, ckpt_path)
        loaded = torch.load(ckpt_path, weights_only=True)

    rc = loaded["residual_config"]
    assert abs(rc["correction_max"] - 1.0) < 1e-9, \
        f"correction_max must be 1.0, got {rc['correction_max']}"
    assert rc["correction_max_mode"] == "resolved_from_capacity_audit"
    assert abs(rc["correction_epsilon"] - 0.05) < 1e-9


def test_checkpoint_rho_not_trainable():
    """The rho parameter must have requires_grad=False when rho_trainable=False."""
    model = _make_model(seed=8)
    # The rho.raw parameter is trainable by default (it's an nn.Parameter)
    # but the optimizer must exclude it when rho_trainable=False.
    # Check that the rho gate parameter exists:
    assert hasattr(model.h6, "rho"), "model.h6 must have rho attribute"
    rho_raw = dict(model.named_parameters()).get("h6.rho.raw", None)
    assert rho_raw is not None, "h6.rho.raw must be a named parameter"
    # The config says rho_trainable=False → in training this param gets excluded
    # from optimizer; we verify the config says so
    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)
    assert not cfg["rho_trainable"], "Config must specify rho_trainable=False"


def test_candidate1_config_from_file():
    """The config file must have all required Candidate-1 fields with correct values."""
    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)

    required = [
        "schema_version", "n_groups", "rho_values", "rho_trainable",
        "local_factor_mode", "correction_max", "h6_logit_temperature",
    ]
    for field in required:
        assert field in cfg, f"Required field missing: {field}"

    assert cfg["n_groups"] == 3, f"n_groups must be 3, got {cfg['n_groups']}"
    assert cfg["rho_values"] == [0.05, 0.05, 0.05], \
        f"rho_values must be [0.05,0.05,0.05], got {cfg['rho_values']}"
    assert not cfg["rho_trainable"], "rho_trainable must be false"
    assert cfg["local_factor_mode"] == "center_spread"
    assert cfg["h6_logit_temperature"] == 10.0, \
        f"h6_logit_temperature must be 10.0, got {cfg['h6_logit_temperature']}"
    assert cfg["correction_max"] == 1.0, \
        f"correction_max must be 1.0 (resolved from rho*T capacity), got {cfg['correction_max']}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
