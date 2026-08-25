from __future__ import annotations

import torch

from tools.sabra_v2.region_pool import (
    pool_patch_map,
    symmetric_margin_delta,
    upsample_region_map,
)


def test_pool_patch_map_preserves_constant_values_and_region_shape() -> None:
    """A wrong adaptive-pooling grid must not silently change the 9x9 contract."""
    patch = torch.full((2, 37, 37), 3.25)

    pooled = pool_patch_map(patch)

    assert pooled.shape == (2, 9, 9)
    assert torch.equal(pooled, torch.full((2, 9, 9), 3.25))


def test_pool_and_upsample_are_deterministic_for_staged_batch_maps() -> None:
    """A stochastic or reordered region transform would invalidate teacher targets."""
    patch = torch.arange(3 * 2 * 37 * 37, dtype=torch.float32).reshape(3, 2, 37, 37)

    first = upsample_region_map(pool_patch_map(patch))
    second = upsample_region_map(pool_patch_map(patch))

    assert first.shape == (3, 2, 37, 37)
    assert torch.equal(first, second)


def test_symmetric_margin_delta_changes_only_margin_without_common_offset() -> None:
    """Swapping either sign or channel would break the specified two-class residual semantics."""
    native = torch.zeros((3, 1, 1369, 2), dtype=torch.float32)
    delta = torch.ones((3, 1, 37, 37), dtype=torch.float32)

    corrected = symmetric_margin_delta(native, delta)

    assert corrected.shape == native.shape
    assert torch.allclose(corrected[..., 0], torch.full((3, 1, 1369), -0.5))
    assert torch.allclose(corrected[..., 1], torch.full((3, 1, 1369), 0.5))
    margin = corrected[..., 1] - corrected[..., 0]
    assert torch.allclose(margin, torch.ones((3, 1, 1369)))
