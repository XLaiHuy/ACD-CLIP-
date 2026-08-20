import random
import numpy as np
import torch
from model.checkpoint_utils import capture_rng_state, restore_rng_state, write_torch_checkpoint_atomic

def test_checkpoint_round_trip_captures_and_restores_rng_state(tmp_path) -> None:
    random.seed(11); np.random.seed(12); torch.manual_seed(13)
    generator = torch.Generator().manual_seed(14)
    checkpoint = capture_rng_state(dataloader_generator=generator)
    path = tmp_path / "checkpoint.pth"
    write_torch_checkpoint_atomic(path, {"epoch": 1, **checkpoint})
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.random(); np.random.rand(); torch.rand(())
    restore_rng_state(loaded, dataloader_generator=generator)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert actual == expected
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp"))
