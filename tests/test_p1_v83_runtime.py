from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from dataset import info
from model.checkpoint_utils import build_phase4_checkpoint, validate_h6_configuration
from model.clip import _is_usable_checkpoint_file, resolve_openai_checkpoint
from model.h6.model import H6Progress1
from test import combine_image_score, image_auc_ap_or_none
from tools.preflight_p1_v83_final_checkpoint import validate_final_checkpoint_payload
from train import (
    grad_accum_window_size,
    h6_drift_gradient_attribution,
    p1_v83_structure_diagnostics,
    scalar_metric_value,
)
from utils import (
    configure_canonical_fp32,
    make_dataloader_generator,
    metrics_eval_gpu,
    seed_worker,
)


def _tiny_h6(n_groups=1, **kwargs):
    return H6Progress1(
        n_groups=n_groups, num_factors=4, top_k=2, bank_dim=8, router_dim=4,
        vae_hidden_dim=8, vae_latent_dim=4, text_dim=8, ctx_len=4,
        progress_version="P1-v8.3", prediction_routing="dense", **kwargs,
    )


def test_scalar_metric_value_accepts_disabled_python_float_losses():
    assert scalar_metric_value(0.0) == 0.0
    assert scalar_metric_value(torch.tensor(1.25)) == pytest.approx(1.25)
    with pytest.raises(ValueError, match="scalar"):
        scalar_metric_value(torch.ones(2))


def test_p1_v83_structure_diagnostics_detect_factor_separation():
    dynamic = torch.zeros(1, 1, 4, 2, 4)
    state = torch.zeros(1, 4, 2, 4)
    logits = torch.zeros(1, 1, 3, 4)
    for factor in range(4):
        dynamic[:, :, factor, :, factor] = 1.0
        state[:, factor, :, factor] = 1.0
        logits[:, :, :, factor] = torch.tensor([factor, factor + 1, factor + 3])
    diagnostics = p1_v83_structure_diagnostics({
        "dynamic_text": dynamic,
        "state_tokens": state,
        "factor_patch_logits": logits,
    })
    assert diagnostics["factor_embedding_effective_rank"] == pytest.approx(4.0)
    assert diagnostics["state_pairwise_l2_min"] > 0
    assert diagnostics["factor_patch_pairwise_max_difference"] > 0
    assert diagnostics["factor_patch_outputs_exactly_collapsed"] == 0


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


def test_clip_lfs_pointer_is_not_treated_as_pretrained_weight(tmp_path):
    pointer = tmp_path / "ViT-L-14-336px.pt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02\n"
        "size 934088680\n",
        encoding="utf-8",
    )
    assert _is_usable_checkpoint_file(pointer) is False


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


def test_utility_gradient_attribution_is_no_step_and_reports_ratios():
    shared = nn.Parameter(torch.tensor(2.0))
    router = nn.Parameter(torch.tensor(3.0))
    report = h6_drift_gradient_attribution(
        {
            "main_task": (shared.square(), 1.0),
            "utility_factor": (3.0 * shared, 0.1),
            "utility_router": (5.0 * router, 0.1),
        },
        {"shared_semantic": [shared], "router": [router]},
    )
    assert shared.grad is None
    assert router.grad is None
    assert report["components"]["utility_factor"]["shared_semantic"] == pytest.approx(0.3)
    assert report["components"]["utility_router"]["router"] == pytest.approx(0.5)
    assert report["components"]["utility_factor"]["raw_gradient_norms"]["shared_semantic"] == pytest.approx(3.0)
    assert report["components"]["utility_factor"]["weighted_gradient_norms"]["shared_semantic"] == pytest.approx(0.3)
    assert report["ratio_basis"] == "lambda_weighted"
    assert report["ratios"]["utility_factor_to_task_shared_grad_ratio"] == pytest.approx(0.075)
    assert report["raw_ratios"]["utility_factor_to_task_shared_grad_ratio"] == pytest.approx(0.75)
    assert report["ratios"]["utility_router_to_task_shared_grad_ratio"] == pytest.approx(0.0)


def test_gradient_attribution_weight_scaling_and_zero_weight_are_explicit():
    shared = nn.Parameter(torch.tensor(2.0))
    report = h6_drift_gradient_attribution(
        {
            "main_task": (shared.square(), 1.0),
            "weighted_aux": (3.0 * shared, 0.25),
            "disabled_aux": (7.0 * shared, 0.0),
        },
        {"shared_semantic": [shared]},
    )
    weighted = report["components"]["weighted_aux"]
    assert weighted["raw_gradient_norms"]["shared_semantic"] == pytest.approx(3.0)
    assert weighted["weighted_gradient_norms"]["shared_semantic"] == pytest.approx(0.75)
    disabled = report["components"]["disabled_aux"]
    assert disabled["differentiable"] is True
    assert disabled["active"] is False
    assert disabled["raw_gradient_norms"]["shared_semantic"] == pytest.approx(7.0)
    assert disabled["weighted_gradient_norms"]["shared_semantic"] == pytest.approx(0.0)
    assert shared.grad is None


def test_gradient_accumulation_remainder_uses_actual_window_size():
    assert [grad_accum_window_size(i, 14, 6) for i in range(1, 15)] == [
        6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6,
        2, 2,
    ]
    full = nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([full], lr=1.0)
    for batch_index in range(1, 15):
        (full / grad_accum_window_size(batch_index, 14, 6)).backward()
        if batch_index % 6 == 0 or batch_index == 14:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    assert full.item() == pytest.approx(-3.0)


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


def test_frozen_medical_image_score_and_unsupported_metrics():
    cls = torch.tensor([0.2, 0.8])
    pmax = torch.tensor([0.6, 0.4])
    assert torch.equal(combine_image_score(cls, pmax, "Medical"), torch.tensor([0.4, 0.6]))
    assert image_auc_ap_or_none(torch.ones(2, dtype=torch.int64), cls) == (None, None)


class _CheckpointModel:
    def __init__(self):
        self.h6_enabled = True
        self.h6 = _tiny_h6(n_groups=3)
        self.image_adapter = nn.Linear(2, 2)
        self.text_adapter = nn.Linear(2, 2)
        self.soft_prompt = nn.Linear(2, 2)
        self.soft_prompt_ctx_len = 4
        self.soft_prompt_init = "phrase"
        self.soft_prompt_init_phrase = "a photo of a"
        self.hybrid_alpha_current = 0.0
        self.hybrid_alpha_max = 0.2
        self.soft_prompt_freeze_epochs = 3
        self.n_groups = 3
        self.dfg_mode = "attn"
        self.dfg_attn_dim = 256
        self.dfg_attn_tau = 8.0
        self.use_ss2d_dfg = True
        self.dfg_gamma_max = 0.2
        self.dfg_ss2d_fusion = "weight_residual"
        self.dfg_beta = 0.1
        self.dfg_beta_schedule = "warmup010"
        self.dfg_beta_target = 0.1
        self.dfg_weight_residual_fp32 = True
        self.use_soft_prompt = False
        self.use_hybrid_soft_prompt = True


def test_v83_checkpoint_metadata_and_geometry_roundtrip():
    model = _CheckpointModel()
    payload = build_phase4_checkpoint(
        model, epoch=20, seed=0, precision="fp32", phase2b_config={
            "git_sha": "deadbeef",
            "tf32_enabled": False,
            "amp_enabled": False,
            "grad_checkpointing": True,
            "h6_global_text_mode": "phase2b_hybrid",
            "h6_prediction_routing": "dense",
            "img_size": 518,
            "batch_size": 1,
            "grad_accum_steps": 6,
            "h6_local_factor_mode": "center_spread",
            "h6_local_center_mix": 0.05,
            "h6_local_factor_spread": 0.10,
            "h6_tau_utility": 0.05,
            "lambda_h6_factor": 0.03,
            "lambda_h6_router": 0.10,
            "h6_utility_factor_effective_beta": 0.999,
            "h6_router_support_normalized": True,
            "h6_pcgrad_main_factor": False,
            "h6_primary_anchored_factor_surgery": True,
            "h6_collect_router_gradient_geometry": False,
        }, loss_weights={
            "balance": 0.0,
            "center": 0.0,
            "orth": 0.0,
            "functional_factor_diversity": 0.0,
            "router_teacher": 0.0,
            "cluster_loss_weight": 0.0,
        },
    )
    assert payload["checkpoint_version"] == 8
    assert payload["precision"] == "fp32"
    assert payload["h6_config"]["progress_version"] == "P1-v8.3"
    assert payload["h6_config"]["variant"] == "p1_v8_3_structured_utility_routing"
    assert payload["global_text_mode"] == "phase2b_hybrid"
    assert payload["gradient_checkpointing"] is True
    assert payload["initialization"] == "openai_clip"
    assert payload["phase2b_checkpoint_loaded"] is False
    assert payload["use_hybrid_soft_prompt"] is True
    assert payload["h6_config"]["rho_fixed"] is True
    assert payload["h6_config"]["rho_trainable"] is False
    assert payload["h6_config"]["local_factor_mode"] == "center_spread"
    assert payload["h6_config"]["local_center_mix"] == pytest.approx(0.05)
    assert payload["h6_config"]["local_factor_spread"] == pytest.approx(0.10)
    assert payload["h6_config"]["primary_anchored_factor_surgery"] is True
    assert payload["h6_config"]["pcgrad_main_factor"] is False
    validate_h6_configuration(model, payload)
    assert validate_final_checkpoint_payload(payload)["status"] == "PASS"
    legacy_symmetric = copy.deepcopy(payload)
    legacy_symmetric["h6_config"].pop("primary_anchored_factor_surgery")
    legacy_symmetric["h6_config"]["pcgrad_main_factor"] = True
    validate_h6_configuration(model, legacy_symmetric)
