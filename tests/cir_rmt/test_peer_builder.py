import torch

from tools.cir_rmt.core import select_gt_free_peers


def test_peer_builder_is_valid_deterministic_and_spatially_exclusive():
    torch.manual_seed(4)
    features = torch.nn.functional.normalize(torch.randn(3, 2, 49, 5), dim=-1)
    margins = torch.zeros(3, 2, 49)
    first = select_gt_free_peers(features, margins, peer_count=8, spatial_radius=1)
    second = select_gt_free_peers(features, margins, peer_count=8, spatial_radius=1)
    assert torch.equal(first["peer_indices"], second["peer_indices"])
    assert first["valid"].all()
    assert first["peer_indices"].shape == (2, 49, 8)
    side = 7
    for batch in range(2):
        for patch in range(49):
            picked = first["peer_indices"][batch, patch].tolist()
            assert len(set(picked)) == 8
            y, x = divmod(patch, side)
            for peer in picked:
                py, px = divmod(peer, side)
                assert max(abs(y - py), abs(x - px)) > 1
                assert peer != patch
    assert not first["peer_indices"].requires_grad
