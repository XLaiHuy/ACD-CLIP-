from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from scripts.cir_rmt import train_full as cir_train
from tests.canonical_fixtures import TinyAdapter, TinyClip
from tools.cir_rmt.identity import load_cir_config
from train import _make_optimizer as parent_optimizer
from train import _set_epoch_state as parent_set_epoch_state


ROOT = Path(__file__).resolve().parents[2]
TRAIN_FULL = ROOT / "scripts" / "cir_rmt" / "train_full.py"
RUNNER = ROOT / "scripts" / "cir_rmt" / "run_full_cir_v2.sh"


def _parent_config() -> dict:
    return json.loads((ROOT / "configs" / "phase2b_canonical_v1.json").read_text(encoding="utf-8"))


def _cir_config() -> dict:
    return load_cir_config(ROOT / "configs" / "cir_dfg_rmt_v2.json")


def _tiny_model() -> TinyAdapter:
    return TinyAdapter(TinyClip())


def _train_ast() -> ast.FunctionDef:
    tree = ast.parse(TRAIN_FULL.read_text(encoding="utf-8"))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "train")


def _is_scheduler_step(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "step"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "scheduler"
    )


def test_cir_scheduler_steps_once_after_epoch_and_before_checkpoint():
    function = _train_ast()
    epoch_loop = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "epoch"
    )
    scheduler_steps = [node for node in ast.walk(function) if _is_scheduler_step(node)]
    assert len(scheduler_steps) == 1
    scheduler_step = scheduler_steps[0]
    scheduler_expr = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Expr) and node.value is scheduler_step
    )
    assert scheduler_expr in epoch_loop.body

    checkpoint_calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_torch_checkpoint_atomic"
    ]
    assert checkpoint_calls
    assert scheduler_step.lineno < min(node.lineno for node in checkpoint_calls)


def test_cir_candidate_epochs_follow_parent_and_include_e10():
    parent = _parent_config()
    assert parent["candidate_epochs"] == [10, 12, 14, 16, 18, 20]
    source = TRAIN_FULL.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert 'parent_config["candidate_epochs"]' in source
    assert '"target_epochs": list(candidate_epochs)' in source
    assert "TARGET_EPOCHS" not in source
    assert '"${CANDIDATE_EPOCHS[@]}"' in runner
    assert "for epoch in 12 14 16 18 20" not in runner


def test_cir_optimizer_matches_parent_and_step_lr_trajectory():
    config = _parent_config()
    parent_model = _tiny_model()
    cir_model = _tiny_model()
    parent_opt = parent_optimizer(parent_model, config)
    cir_opt = cir_train._optimizer(cir_model, config)
    parent_scheduler = torch.optim.lr_scheduler.StepLR(parent_opt, step_size=1, gamma=0.9)
    cir_scheduler = torch.optim.lr_scheduler.StepLR(cir_opt, step_size=1, gamma=0.9)

    assert parent_opt.defaults == cir_opt.defaults
    assert parent_opt.defaults["betas"] == (0.9, 0.999)
    assert parent_opt.defaults["eps"] == pytest.approx(1e-8)
    assert parent_opt.defaults["weight_decay"] == pytest.approx(0.0)
    assert [group["name"] for group in cir_opt.param_groups] == [
        "image_adapter",
        "text_adapter",
        "soft_prompt",
    ]
    assert cir_opt.param_groups[0]["lr"] == pytest.approx(2.0 * cir_opt.param_groups[1]["lr"])
    assert cir_opt.param_groups[0]["lr"] == pytest.approx(10.0 * cir_opt.param_groups[2]["lr"])

    starts: dict[int, list[float]] = {}
    posts: dict[int, list[float]] = {}
    for epoch in range(1, 21):
        parent_set_epoch_state(parent_model, parent_opt, config, epoch)
        cir_train._set_epoch_state(cir_model, cir_opt, config, epoch)
        starts[epoch] = [float(group["lr"]) for group in cir_opt.param_groups]
        assert starts[epoch] == pytest.approx([float(group["lr"]) for group in parent_opt.param_groups])
        assert [parameter.requires_grad for parameter in parent_model.soft_prompt.parameters()] == [
            parameter.requires_grad for parameter in cir_model.soft_prompt.parameters()
        ]

        parent_opt.step()
        cir_opt.step()
        parent_scheduler.step()
        cir_scheduler.step()
        posts[epoch] = [float(group["lr"]) for group in cir_opt.param_groups]
        assert posts[epoch] == pytest.approx([float(group["lr"]) for group in parent_opt.param_groups])

    assert starts[1][0] == pytest.approx(1e-3)
    assert starts[10][0] == pytest.approx(0.001 * (0.9**9))
    assert starts[12][0] == pytest.approx(0.001 * (0.9**11))
    assert starts[14][0] == pytest.approx(0.001 * (0.9**13))
    assert starts[16][0] == pytest.approx(0.001 * (0.9**15))
    assert starts[18][0] == pytest.approx(0.001 * (0.9**17))
    assert starts[20][0] == pytest.approx(0.001 * (0.9**19))
    assert posts[20][0] == pytest.approx(0.001 * (0.9**20))
    assert starts[1][2] == pytest.approx(0.0)
    assert starts[3][2] == pytest.approx(0.0)
    assert starts[4][2] == pytest.approx(1e-4)
    assert starts[20][2] == pytest.approx(1e-4)
    assert cir_scheduler.last_epoch == 20
    assert cir_scheduler.state_dict()["_step_count"] == 21


def test_candidate_checkpoint_captures_post_scheduler_state_and_resume():
    config = _parent_config()
    cir_config = _cir_config()
    model = _tiny_model()
    optimizer = cir_train._optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    loss = sum(parameter.sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    scheduler.step()
    payload = cir_train.checkpoint_payload(
        model,
        config,
        cir_config,
        "visa",
        1,
        1,
        optimizer,
        scheduler,
        torch.Generator().manual_seed(0),
        "test-fix",
    )
    assert payload["epoch"] == 1
    assert payload["scheduler_state"]["last_epoch"] == 1
    assert payload["scheduler_state"]["_step_count"] == 2
    assert payload["optimizer_state"]["param_groups"][0]["lr"] == pytest.approx(9e-4)
    assert payload["optimizer_state"]["param_groups"][1]["lr"] == pytest.approx(4.5e-4)

    resumed_model = _tiny_model()
    resumed_optimizer = cir_train._optimizer(resumed_model, config)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.9)
    resumed_optimizer.load_state_dict(payload["optimizer_state"])
    resumed_scheduler.load_state_dict(payload["scheduler_state"])
    optimizer.step()
    scheduler.step()
    resumed_optimizer.step()
    resumed_scheduler.step()
    assert resumed_scheduler.last_epoch == scheduler.last_epoch == 2
    assert resumed_scheduler.state_dict()["_step_count"] == scheduler.state_dict()["_step_count"] == 3
    assert [group["lr"] for group in resumed_optimizer.param_groups] == pytest.approx(
        [group["lr"] for group in optimizer.param_groups]
    )
