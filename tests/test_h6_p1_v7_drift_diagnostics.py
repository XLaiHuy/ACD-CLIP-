import torch

from train import h6_drift_gradient_attribution


def test_gradient_attribution_does_not_mutate_parameters_or_grads():
    shared = torch.nn.Parameter(torch.tensor(2.0))
    expert = torch.nn.Parameter(torch.tensor(3.0))
    losses = {
        "main_task": ((shared * 2).square(), 1.0),
        "assigned_expert": ((expert * 3).square(), .25),
    }
    report = h6_drift_gradient_attribution(
        losses, {"shared_semantic": [shared], "expert_B": [expert]},
    )
    assert shared.grad is None and expert.grad is None
    assert report["components"]["main_task"]["shared_semantic"] > 0
    assert report["components"]["assigned_expert"]["expert_B"] > 0
    assert report["components"]["assigned_expert"]["shared_semantic"] == 0
