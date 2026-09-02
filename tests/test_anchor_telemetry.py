import copy

import torch
from torch import nn

from train import anchor_gradient_audit_ratio


def run_fixed_batch(start_state, audit_interval):
    model = nn.Linear(3, 1)
    model.load_state_dict(copy.deepcopy(start_state))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.tensor([[1.0, -2.0, 0.5], [0.25, 0.75, -1.5]])
    target = torch.tensor([[0.4], [-0.2]])
    reference = torch.zeros_like(model.weight)
    task_loss = (model(x) - target).square().mean()
    anchor_loss = (model.weight - reference).square().mean()
    anchor_lambda = 0.001
    loss = task_loss + anchor_lambda * anchor_loss
    ratio = None
    if audit_interval > 0:
        ratio = anchor_gradient_audit_ratio(
            loss,
            anchor_loss,
            list(model.parameters()),
            anchor_lambda,
            "cpu",
        )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return copy.deepcopy(model.state_dict()), ratio


def test_anchor_telemetry_on_off_has_identical_update():
    torch.manual_seed(77)
    initial = nn.Linear(3, 1).state_dict()
    state_off, ratio_off = run_fixed_batch(initial, 0)
    state_on, ratio_on = run_fixed_batch(initial, 1)
    assert ratio_off is None
    assert ratio_on is not None and ratio_on >= 0.0
    for key in state_off:
        torch.testing.assert_close(state_off[key], state_on[key], rtol=0.0, atol=0.0)
