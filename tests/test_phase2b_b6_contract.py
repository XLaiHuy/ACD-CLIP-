from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model import phase2b_legacy_bridge as bridge
from model import phase2b_runtime as runtime
from model.phase2b_schedule import get_dfg_beta_for_epoch, get_hybrid_alpha_for_epoch
from tests.canonical_fixtures import TinyAdapter, TinyClip, tiny_config
from tools.sabra.trust_v2 import numerical as exact
from tools.sabra.trust_v2.fast_geometry import _batched_pgm_pcrr
from train import (
    _make_optimizer,
    _sample_weighted_regularizer,
    _set_epoch_state,
    clip_trainable_gradients,
)


def _tiny_model(monkeypatch, tmp_path):
    monkeypatch.setattr(bridge, "_adapter_and_clip", lambda: (TinyAdapter, lambda *args, **kwargs: TinyClip()))
    monkeypatch.setattr(
        runtime,
        "_text_features",
        lambda model, dataset_name, class_names, device, config: torch.ones(
            (3, len(class_names), 768, 2), device=device
        ),
    )
    asset = tmp_path / "clip.pt"
    asset.write_bytes(b"fixture")
    return runtime.build_phase2b_trainable(tiny_config(), asset, torch.device("cpu"))


def test_b6_kg_regularizer_is_sample_weighted_and_deduplicated():
    names = ["candle"] * 5 + ["capsule"]
    weighted = _sample_weighted_regularizer(
        {"candle": torch.tensor(2.0), "capsule": torch.tensor(8.0)},
        names,
        torch.device("cpu"),
    )
    assert weighted.item() == pytest.approx(3.0)
    all_same = _sample_weighted_regularizer(
        {"candle": torch.tensor(2.0)}, ["candle"] * 6, torch.device("cpu")
    )
    assert all_same.item() == pytest.approx(2.0)

def test_canonical_schedules_and_batch_contract():
    assert [get_dfg_beta_for_epoch(epoch, "warmup010", 0.1, 0.1) for epoch in (1, 3, 4, 6, 7)] == [0.0, 0.0, 0.05, 0.05, 0.1]
    assert [get_hybrid_alpha_for_epoch(epoch, 0.2, 3) for epoch in (1, 3, 4, 5, 6)] == [0.0, 0.0, 0.05, 0.1, 0.2]
    config = tiny_config()
    config.update(micro_batch_size=6, grad_accum_steps=1, effective_batch_size=6)
    runtime.validate_phase2b_config(config)



def test_clip_is_frozen_and_epoch_schedule_controls_soft_prompt(monkeypatch, tmp_path):
    model = _tiny_model(monkeypatch, tmp_path)
    audit = runtime.runtime_audit(model)
    assert audit["parameter_summary"]["clip_trainable"] == 0
    assert all(not parameter.requires_grad for parameter in model.clipmodel.parameters())
    assert all(parameter.requires_grad for parameter in model.image_adapter.parameters())
    assert all(parameter.requires_grad for parameter in model.text_adapter.parameters())
    assert all(not parameter.requires_grad for parameter in model.soft_prompt.parameters())

    config = tiny_config()
    config.update(
        image_lr=1e-3,
        text_lr=5e-4,
        soft_prompt_lr=1e-4,
        dfg_beta_schedule="warmup010",
        dfg_beta_target=0.1,
    )
    optimizer = _make_optimizer(model, config)
    alpha, beta, frozen = _set_epoch_state(model, optimizer, config, epoch=1)
    assert (alpha, beta, frozen) == pytest.approx((0.0, 0.0, True))
    assert all(not parameter.requires_grad for parameter in model.soft_prompt.parameters())
    alpha, beta, frozen = _set_epoch_state(model, optimizer, config, epoch=4)
    assert (alpha, beta, frozen) == pytest.approx((0.05, 0.05, False))
    assert all(parameter.requires_grad for parameter in model.soft_prompt.parameters())


def test_gradient_clipping_and_post_backward_contract_exclude_clip(monkeypatch, tmp_path):
    model = _tiny_model(monkeypatch, tmp_path)
    optimizer = torch.optim.SGD(
        [
            {"params": list(model.image_adapter.parameters())},
            {"params": list(model.text_adapter.parameters())},
            {"params": list(model.soft_prompt.parameters())},
        ],
        lr=0.1,
    )
    (model.image_adapter.weight.sum() + model.text_adapter.weight.sum()).backward()
    bridge.assert_phase2b_gradient_contract(model, soft_prompt_trainable=False)
    norm = clip_trainable_gradients(optimizer, 1.0)
    assert torch.isfinite(norm)
    assert all(parameter.grad is None for parameter in model.clipmodel.parameters())


def test_b1_and_b6_forward_outputs_match(monkeypatch, tmp_path):
    model = _tiny_model(monkeypatch, tmp_path)
    config = tiny_config()
    device = torch.device("cpu")
    images = torch.stack([torch.zeros(3, 518, 518), torch.ones(3, 518, 518)])
    names = ["candle", "capsule"]
    batch = runtime.forward_phase2b(model, images, names, device, config, dataset_name="VisA")
    singles = [
        runtime.forward_phase2b(model, images[index:index + 1], [names[index]], device, config, dataset_name="VisA")
        for index in range(2)
    ]
    for index, single in enumerate(singles):
        assert torch.allclose(single.native_logits[:, 0], batch.native_logits[:, index], atol=0.0, rtol=0.0)
        assert torch.allclose(single.native_segmentation_probability[0], batch.native_segmentation_probability[index], atol=0.0, rtol=0.0)
        assert torch.allclose(single.classification_probability[0], batch.classification_probability[index], atol=0.0, rtol=0.0)


def test_cpu_fast_pgm_uses_authoritative_fixed_semantics():
    rng = np.random.default_rng(123)
    raw = rng.normal(size=(4, 3, 8, 8)).astype(np.float64)
    gram = np.einsum("...ik,...jk->...ij", raw, raw) + np.eye(8, dtype=np.float64)
    from p5f_geometry.common import pack_gram

    c = rng.normal(size=(4, 3, 8)).astype(np.float32)
    packed = pack_gram(gram)
    fast_pgm, fast_pcrr = _batched_pgm_pcrr(c, packed)
    exact_pgm = exact.fixed.pgm_raw(c, packed)
    exact_pcrr = exact.base.pcrr_raw(c, packed)
    for key in ("rank",):
        np.testing.assert_array_equal(fast_pgm[key], exact_pgm[key])
    for key in ("raw", "tol", "max_eigen"):
        np.testing.assert_allclose(fast_pgm[key], exact_pgm[key], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(fast_pcrr["comparison_count"], exact_pcrr["comparison_count"])
    np.testing.assert_allclose(fast_pcrr["raw"], exact_pcrr["raw"], rtol=1e-6, atol=1e-6)


def test_frozen_backend_provenance_rejects_override():
    from tests.test_sabra_freeze import _freeze
    freeze = _freeze()
    forward = SimpleNamespace(native_logits=torch.zeros(3, 1, 1369, 2), seg_features=torch.zeros(3, 1369, 16))
    from tools.sabra.pipeline import corrected_from_forward

    with pytest.raises(ValueError, match="does not match frozen backend"):
        corrected_from_forward(forward, freeze, backend="exact")
