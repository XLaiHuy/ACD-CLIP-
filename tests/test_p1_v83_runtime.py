from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dataset import info
from model.checkpoint_utils import build_phase4_checkpoint, validate_h6_configuration
from model.clip import resolve_openai_checkpoint
from model.h6.model import H6Progress1
from utils import (
    configure_canonical_fp32,
    make_dataloader_generator,
    metrics_eval_gpu,
    seed_worker,
)


def _tiny_h6(**kwargs):
    return H6Progress1(
        n_groups=1, num_factors=4, top_k=2, bank_dim=8, router_dim=4,
        vae_hidden_dim=8, vae_latent_dim=4, text_dim=8, ctx_len=4,
        progress_version="P1-v8.3", prediction_routing="dense", **kwargs,
    )


def test_data_root_override_has_highest_priority(tmp_path, monkeypatch):
    override = tmp_path / "portable-data"
    monkeypatch.setenv("ACDCLIP_DATA_ROOT", str(override))
    assert info.resolve_data_root() == override.resolve()


def test_all_dataset_paths_derive_from_one_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ACDCLIP_DATA_ROOT", str(tmp_path))
    root = info.resolve_data_root()
    assert root == tmp_path.resolve()
    assert all(str(path).startswith(str(info.DATA_ROOT)) for path in info.DATA_PATH.values())


def test_clip_checkpoint_override_and_clear_attempts(tmp_path, monkeypatch):
    checkpoint = tmp_path / "ViT-L-14-336px.pt"
    checkpoint.write_bytes(b"path-resolution-only")
    monkeypatch.setenv("ACDCLIP_CLIP_VITL14_336", str(checkpoint))
    assert resolve_openai_checkpoint("ViT-L-14-336") == checkpoint.resolve()
    checkpoint.unlink()
    monkeypatch.setattr(type(checkpoint), "is_file", lambda self: False)
    with pytest.raises(FileNotFoundError) as exc:
        resolve_openai_checkpoint("ViT-L-14-336")
    assert str(checkpoint) in str(exc.value)
    assert "Attempted paths" in str(exc.value)


def test_fp32_tf32_contract_is_explicit():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    configure_canonical_fp32()
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_worker_seed_and_generator_are_deterministic(monkeypatch):
    monkeypatch.setattr(torch, "initial_seed", lambda: 2**32 + 123)
    seed_worker(99)
    python_value = random.random()
    numpy_value = np.random.rand()
    random.seed(123)
    np.random.seed(123)
    assert python_value == random.random()
    assert numpy_value == np.random.rand()
    assert torch.equal(
        torch.rand(5, generator=make_dataloader_generator(7)),
        torch.rand(5, generator=make_dataloader_generator(7)),
    )


def test_fixed_rho_and_diagnostic_zero_override():
    model = _tiny_h6()
    assert model.rho.raw.requires_grad is False
    assert torch.equal(model.rho_values(), torch.full((1,), 0.05))
    partition_ids = {id(p) for params in model.parameter_partitions().values() for p in params}
    assert id(model.rho.raw) not in partition_ids
    model.eval()
    model.test_rho_override = 0.0
    assert torch.equal(model.rho_values(), torch.zeros(1))
    model.train()
    assert torch.equal(model.rho_values(), torch.full((1,), 0.05))


def test_one_class_image_metrics_report_na_but_pixel_metrics_remain():
    pixel_label = torch.tensor([[[0, 1], [0, 1]], [[0, 1], [0, 1]]])
    pixel_pred = torch.tensor([[[0.1, 0.9], [0.2, 0.8]], [[0.2, 0.8], [0.3, 0.7]]])
    result = metrics_eval_gpu(
        pixel_label, torch.zeros(2, dtype=torch.int64), pixel_pred,
        torch.tensor([0.2, 0.3]), "synthetic", "Medical",
    )
    assert result["image AUC"] == "N/A"
    assert result["image AP"] == "N/A"
    assert result["pixel AUC"] == 100.0
    assert result["pixel AP"] == 100.0


class _CheckpointModel:
    def __init__(self):
        self.h6_enabled = True
        self.h6 = _tiny_h6()
        self.image_adapter = nn.Linear(2, 2)
        self.text_adapter = nn.Linear(2, 2)
        self.soft_prompt = nn.Linear(2, 2)
        self.soft_prompt_ctx_len = 4
        self.soft_prompt_init = "phrase"
        self.soft_prompt_init_phrase = "a photo of a"
        self.hybrid_alpha_current = 0.0
        self.hybrid_alpha_max = 0.2
        self.soft_prompt_freeze_epochs = 3
        self.n_groups = 1
        self.dfg_mode = "attn"
        self.dfg_attn_dim = 4
        self.dfg_attn_tau = 4.0
        self.use_ss2d_dfg = True
        self.dfg_gamma_max = 0.2
        self.dfg_ss2d_fusion = "weight_residual"
        self.dfg_beta = 0.1
        self.dfg_beta_schedule = "fixed"
        self.dfg_beta_target = 0.1
        self.dfg_weight_residual_fp32 = True


def test_v83_checkpoint_metadata_and_geometry_roundtrip():
    model = _CheckpointModel()
    payload = build_phase4_checkpoint(
        model, epoch=2, seed=0, precision="fp32", phase2b_config={
            "h6_local_factor_mode": "center_spread",
            "h6_local_center_mix": 0.05,
            "h6_local_factor_spread": 0.10,
            "h6_tau_utility": 0.05,
        }, loss_weights={},
    )
    assert payload["checkpoint_version"] == 8
    assert payload["precision"] == "fp32"
    assert payload["h6_config"]["progress_version"] == "P1-v8.3"
    assert payload["h6_config"]["rho_fixed"] is True
    assert payload["h6_config"]["rho_trainable"] is False
    assert payload["h6_config"]["local_factor_mode"] == "center_spread"
    assert payload["h6_config"]["local_center_mix"] == pytest.approx(0.05)
    assert payload["h6_config"]["local_factor_spread"] == pytest.approx(0.10)
    validate_h6_configuration(model, payload)
