import torch

from tools.cir_rmt.core import score_optimized, score_reference


def test_reference_and_optimized_exact_score_space_match():
    torch.manual_seed(7)
    stages, batch, patches, groups, dim, classes = 3, 2, 16, 3, 11, 2
    image = torch.nn.functional.normalize(torch.randn(stages, batch, patches, dim), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(stages, batch, groups, dim, classes), dim=-2)
    weights = torch.rand(stages, batch, patches, groups, classes) + 0.1
    weights = weights / weights.sum(dim=-2, keepdim=True)
    reference = score_reference(image, text, weights)
    optimized = score_optimized(image, text, weights)
    assert reference.shape == (stages, batch, patches, classes)
    assert torch.max((reference - optimized).abs()).item() <= 1e-5
