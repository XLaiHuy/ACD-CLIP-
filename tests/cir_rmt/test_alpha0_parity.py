import torch

from tools.cir_rmt.core import cir_logits_from_native_weights, score_reference


def test_alpha_zero_cir_path_matches_native_score_path():
    torch.manual_seed(8)
    stages, batch, patches, groups, dim = 3, 2, 16, 3, 9
    image = torch.nn.functional.normalize(torch.randn(stages, batch, patches, dim), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(batch, groups, dim, 2), dim=-2)
    native = torch.rand(stages, batch, groups, 2) + 0.2
    native = native / native.sum(dim=-2, keepdim=True)
    delta = torch.randn(stages, batch, patches, groups)
    cir, native_score = cir_logits_from_native_weights(image, text, native, delta, 0.0, score_mode="reference")
    direct = score_reference(image, text, native)
    assert torch.equal(cir, native_score)
    assert torch.max((cir - direct).abs()).item() <= 1e-6
