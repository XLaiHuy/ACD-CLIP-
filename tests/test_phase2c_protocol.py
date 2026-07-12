import copy
import random
import unittest

import numpy as np
import torch
from torch import nn

from phase2c_utils import (
    EpochDeterministicSampler, gradient_pair_stats, image_anchor,
    normalized_config, phase2c_config, run_gradient_diagnostics,
    select_checkpoint, state_checksum,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_adapter = nn.ModuleDict({
            "lora_adapters": nn.ModuleList([nn.Linear(2, 2, bias=False)]),
            "m_i_w": nn.ModuleList([nn.Linear(2, 2, bias=False)]),
        })
        self.text_adapter = nn.Linear(2, 2, bias=False)
        self.soft_prompt = nn.Linear(2, 2, bias=False)
        self.register_buffer("running", torch.tensor([3.0]))

    def forward(self, x):
        values = self.image_adapter["lora_adapters"][0](x)
        values = values + self.image_adapter["m_i_w"][0](x)
        values = values + self.text_adapter(x) + self.soft_prompt(x)
        return values.sum(dim=1)


class Phase2CProtocolTests(unittest.TestCase):
    def test_sampler_and_condition_config(self):
        data = list(range(23))
        left = EpochDeterministicSampler(data, 42)
        right = EpochDeterministicSampler(data, 42)
        for epoch in (0, 1, 7):
            left.set_epoch(epoch)
            right.set_epoch(epoch)
            self.assertEqual(list(left), list(right))
        a = phase2c_config("A_prime", "a", 0.20)
        b = phase2c_config("B", "b", 0.15)
        c = phase2c_config("C", "c", 0.20)
        self.assertEqual(normalized_config(a), normalized_config(b))
        self.assertEqual(a["alpha_schedule"][:6], [0.0, 0.0, 0.0, 0.05, 0.10, 0.20])
        self.assertEqual(b["alpha_schedule"][:6], [0.0, 0.0, 0.0, 0.0375, 0.075, 0.15])
        self.assertEqual(c["activation_delay_epochs"], 2)
        self.assertEqual(c["soft_prompt_freeze_epochs"], 5)
        self.assertEqual(c["alpha_schedule"][:9], [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.10, 0.20, 0.20])
        self.assertNotEqual(normalized_config(a), normalized_config(c))

    def test_anchor_constraint_and_tie_break(self):
        rows = [
            {"epoch": 1, "pixel_auc": 1, "pixel_ap": 5, "image_auc": 1, "image_ap": 70},
            {"epoch": 2, "pixel_auc": 1, "pixel_ap": 6, "image_auc": 1, "image_ap": 80},
            {"epoch": 3, "pixel_auc": 1, "pixel_ap": 10, "image_auc": 1, "image_ap": 90},
            {"epoch": 4, "pixel_auc": 1, "pixel_ap": 50, "image_auc": 1, "image_ap": 86},
            {"epoch": 5, "pixel_auc": 1, "pixel_ap": 50, "image_auc": 1, "image_ap": 86},
            {"epoch": 6, "pixel_auc": 1, "pixel_ap": 99, "image_auc": 1, "image_ap": 83},
        ]
        self.assertEqual(image_anchor(rows), 85)
        selected = select_checkpoint(rows)
        self.assertEqual(selected["selected_epoch"], 4)

    def test_zero_gradient_is_na(self):
        self.assertEqual(
            gradient_pair_stats(None, None),
            {"cls_grad_norm": "NA", "seg_grad_norm": "NA", "cosine": "NA"},
        )

    def test_diagnostics_preserve_all_state(self):
        torch.manual_seed(4)
        random.seed(4)
        np.random.seed(4)
        model = ToyModel()
        model.train()
        model.text_adapter.eval()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
        modes = {name: module.training for name, module in model.named_modules()}
        params = {name: value.detach().clone() for name, value in model.named_parameters()}
        buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        optimizer_hash = state_checksum(optimizer.state_dict())
        scheduler_hash = state_checksum(scheduler.state_dict())
        cpu_rng = torch.get_rng_state().clone()
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        batch = {"x": torch.tensor([[1.0, 2.0], [3.0, 4.0]]), "y": torch.tensor([1.0, -1.0])}

        def losses(item):
            random.random()
            np.random.rand()
            torch.rand(1)
            pred = model(item["x"])
            return ((pred - item["y"]) ** 2).mean(), ((pred + item["y"]) ** 2).mean()

        rows = run_gradient_diagnostics(model, optimizer, scheduler, [batch], losses, 1)
        self.assertEqual(len(rows), 4)
        self.assertEqual(modes, {name: module.training for name, module in model.named_modules()})
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(params[name], value))
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(buffers[name], value))
        self.assertEqual(optimizer_hash, state_checksum(optimizer.state_dict()))
        self.assertEqual(scheduler_hash, state_checksum(scheduler.state_dict()))
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertEqual(python_rng, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(numpy_rng[0], after_numpy[0])
        self.assertTrue(np.array_equal(numpy_rng[1], after_numpy[1]))
        self.assertEqual(numpy_rng[2:], after_numpy[2:])


if __name__ == "__main__":
    unittest.main()
