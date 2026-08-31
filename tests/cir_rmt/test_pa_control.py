from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from scripts.cir_rmt.train_pa import compose_pa_loss
from tests.canonical_fixtures import TinyAdapter, TinyClip
from tools.cir_rmt.parameter_anchor import ImageParameterAnchor


ROOT = Path(__file__).resolve().parents[2]
PA_TRAIN = ROOT / "scripts/cir_rmt/train_pa.py"


def _train_ast() -> ast.FunctionDef:
    tree = ast.parse(PA_TRAIN.read_text(encoding="utf-8"))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "train_pa")


def _is_scheduler_step(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "scheduler"
    )


def test_pa_lambda_zero_is_native_loss_exactly() -> None:
    base = torch.tensor(1.25, requires_grad=True)
    anchor = torch.tensor(7.5, requires_grad=True)
    result = compose_pa_loss(base, anchor, 0.0)
    assert float(result.detach()) == pytest.approx(float(base.detach()))
    result.backward()
    assert base.grad is not None and float(base.grad) == pytest.approx(1.0)
    assert anchor.grad is not None and float(anchor.grad) == pytest.approx(0.0)


def test_pa_anchor_reference_is_immutable_and_not_optimizer_registered() -> None:
    image = torch.nn.Linear(3, 2)
    text = torch.nn.Linear(3, 2)
    reference = {name: value.detach().clone() for name, value in image.named_parameters()}
    anchor = ImageParameterAnchor(reference, checkpoint_sha256="anchor", epoch=14, config_sha256="cfg", device=torch.device("cpu"))
    before = {name: value.clone() for name, value in anchor.reference.items()}
    optimizer = torch.optim.Adam(list(image.parameters()) + list(text.parameters()), lr=1e-3)
    loss = anchor.loss(image)
    loss.backward()
    assert all(value.requires_grad is False for value in anchor.reference.values())
    assert all(parameter.grad is not None for parameter in image.parameters())
    assert all(parameter.grad is None for parameter in text.parameters())
    assert all(id(value) not in {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} for value in anchor.reference.values())
    assert all(torch.equal(before[name], value) for name, value in anchor.reference.items())


def test_pa_has_native_forward_and_no_cir_training_path() -> None:
    source = PA_TRAIN.read_text(encoding="utf-8")
    assert "forward_phase2b" in source
    assert "from tools.cir_rmt.runtime" not in source
    for forbidden in ("forward_cir", "peer_valid", "cir_segmentation_probability", "rmt_delta", "delta_stats"):
        assert forbidden not in source


def test_pa_reuses_canonical_training_helpers() -> None:
    source = PA_TRAIN.read_text(encoding="utf-8")
    for required in ("_build_loader", "_text_with_regularizers", "_make_optimizer", "_set_epoch_state", "clip_trainable_gradients", "grad_accum_window_size"):
        assert required in source
    assert "scheduler.step()" in source
    assert "write_torch_checkpoint_atomic" in source


def test_pa_scheduler_steps_once_after_epoch_before_checkpoint() -> None:
    function = _train_ast()
    steps = [node for node in ast.walk(function) if _is_scheduler_step(node)]
    assert len(steps) == 1
    scheduler_step = steps[0]
    checkpoint_calls = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_torch_checkpoint_atomic"
    ]
    assert checkpoint_calls
    assert scheduler_step.lineno < min(node.lineno for node in checkpoint_calls)


def test_pa_identity_and_protocol_contract_are_frozen() -> None:
    source = PA_TRAIN.read_text(encoding="utf-8")
    assert 'CONTROL_ID = "PA_PHASE2B_IMAGE_ANCHOR_V1"' in source
    assert '"training_forward": "native_phase2b"' in source
    assert '"cir_training": False' in source
    assert '"rmt_training": False' in source
    assert '"alpha_inference": None' in source
    config = json.loads((ROOT / "configs/phase2b_canonical_v1.json").read_text(encoding="utf-8"))
    assert config["precision"] == "fp32"
    assert config["candidate_epochs"] == [10, 12, 14, 16, 18, 20]
    assert config["image_lr"] == pytest.approx(1e-3)
    assert config["text_lr"] == pytest.approx(5e-4)
    assert config["soft_prompt_lr"] == pytest.approx(1e-4)
    assert config["lr_gamma"] == pytest.approx(0.9)


def test_pa_tiny_fixture_can_construct_native_model() -> None:
    # This exercises the same anchor object used by the real control without
    # loading the ViT-L asset or touching the scientific run directory.
    model = TinyAdapter(TinyClip())
    reference = {name: value.detach().clone() for name, value in model.image_adapter.named_parameters()}
    anchor = ImageParameterAnchor(reference, checkpoint_sha256="anchor", epoch=14, config_sha256="cfg", device=torch.device("cpu"))
    assert float(anchor.loss(model.image_adapter).detach()) == pytest.approx(0.0)
