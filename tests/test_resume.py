import random

import numpy as np
import torch
from torch import nn

from h2_clean.contract import (
    build_full_checkpoint,
    checkpoint_required_keys,
    restore_full_checkpoint,
    seed_everything,
    make_dataloader_generator,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_adapter = nn.Linear(3, 3)
        self.text_adapter = nn.Linear(3, 3)

    def forward(self, x):
        return self.image_adapter(x) + self.text_adapter(x)


def make_training_objects():
    model = ToyModel()
    optimizer = torch.optim.Adam(
        [
            {"name": "image_adapter", "params": model.image_adapter.parameters(), "lr": 1e-2},
            {"name": "text_adapter", "params": model.text_adapter.parameters(), "lr": 2e-2},
        ]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    generator = make_dataloader_generator(777)
    return model, optimizer, scheduler, scaler, generator


def run_steps(model, optimizer, scheduler, generator, count):
    batch_ids = []
    for _ in range(count):
        batch_ids.append(torch.randperm(8, generator=generator))
        x = torch.randn(8, 3)
        x = x + random.random() + float(np.random.rand())
        target = torch.randn(8, 3)
        loss = (model(x) - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return batch_ids


def assert_nested_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_full_state_resume_matches_uninterrupted(tmp_path):
    total_steps = 6
    split_steps = 3
    seed_everything(91)
    full_model, full_optimizer, full_scheduler, full_scaler, full_generator = make_training_objects()
    run_steps(full_model, full_optimizer, full_scheduler, full_generator, total_steps)
    full_next_batch = torch.randperm(12, generator=full_generator)

    seed_everything(91)
    first_model, first_optimizer, first_scheduler, first_scaler, first_generator = make_training_objects()
    run_steps(first_model, first_optimizer, first_scheduler, first_generator, split_steps)
    payload = build_full_checkpoint(
        model=first_model,
        optimizer=first_optimizer,
        scheduler=first_scheduler,
        scaler=first_scaler,
        epoch=1,
        global_step=split_steps,
        config={"toy": True, "seed": 91},
        repo=".",
        clip_sha256="clip",
        dataset_manifest_sha256="manifest",
        dataloader_generator=first_generator,
        anchor=None,
        anchor_lambda=0.0,
        seed=91,
        precision="fp32",
        tf32_enabled=False,
    )
    for key in checkpoint_required_keys():
        assert key in payload
    checkpoint_path = tmp_path / "resume.pth"
    torch.save(payload, checkpoint_path)

    resumed_model, resumed_optimizer, resumed_scheduler, resumed_scaler, resumed_generator = make_training_objects()
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch, global_step = restore_full_checkpoint(
        loaded,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        dataloader_generator=resumed_generator,
    )
    assert (epoch, global_step) == (1, split_steps)
    run_steps(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        resumed_generator,
        total_steps - split_steps,
    )
    resumed_next_batch = torch.randperm(12, generator=resumed_generator)

    assert_nested_equal(full_model.state_dict(), resumed_model.state_dict())
    assert_nested_equal(full_optimizer.state_dict(), resumed_optimizer.state_dict())
    assert_nested_equal(full_scheduler.state_dict(), resumed_scheduler.state_dict())
    assert_nested_equal(full_scaler.state_dict(), resumed_scaler.state_dict())
    assert torch.equal(full_next_batch, resumed_next_batch)
    assert full_scheduler.get_last_lr() == resumed_scheduler.get_last_lr()
