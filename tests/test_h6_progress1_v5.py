import torch
import torch.nn.functional as F

from model.h6.losses import factor_stage_diagnostics, teacher_candidate_diagnostics
from model.h6.router import PatchRouter


def test_router_is_per_patch_not_mean_pool_broadcast():
    torch.manual_seed(0)
    router = PatchRouter(n_groups=1, num_factors=4, text_dim=8, bank_dim=4, hidden_dim=12, top_k=2)
    tokens = torch.zeros(1, 1, 4, 8)
    tokens[0, 0, 0, 0] = 1.0
    tokens[0, 0, 1, 1] = 1.0
    tokens[0, 0, 2, 2] = 1.0
    tokens[0, 0, 3, 3] = 1.0
    keys = F.normalize(torch.randn(4, 4), dim=-1)
    out = router(tokens, epoch_one_based=12, concept_keys=keys)
    assert out["queries"].shape == (1, 1, 4, 4)
    assert out["logits"].shape == (1, 1, 4, 4)
    assert out["dense_probabilities"].shape == (1, 1, 4, 4)
    assert out["router_softmax_dim"].item() == 3
    assert out["router_topk_dim"].item() == 3
    assert not torch.allclose(out["queries"][0, 0, 0], out["queries"][0, 0, 1])
    assert out["query_variance_across_patches"].item() > 0


def test_teacher_candidates_report_patchwise_nonuniformity():
    projected = torch.zeros(1, 1, 4, 3)
    projected[0, 0, :, 0] = torch.tensor([1.0, 0.5, -0.5, -1.0])
    projected[0, 0, :, 1] = torch.tensor([0.0, 0.5, 0.5, 0.0])
    proto = F.normalize(torch.tensor([[[1.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0],
                                       [-1.0, 0.0, 0.0],
                                       [0.0, -1.0, 0.0]]]), dim=-1)
    masks = torch.zeros(1, 1, 2, 2)
    labels = torch.tensor([0])
    diag = teacher_candidate_diagnostics(projected, proto, proto, masks, labels, temperature=0.15)
    assert diag["teacher_raw_candidate_entropy"].shape == (1,)
    assert diag["teacher_centered_candidate_probability_std_across_patches"].shape == (1,)
    assert diag["teacher_distance_candidate_unique_topk_pairs"].shape == (1,)
    assert torch.isfinite(diag["teacher_raw_candidate_entropy"]).all()
    assert diag["teacher_raw_candidate_probability_std_across_patches"].item() > 0


def test_factor_stage_diagnostics_finds_identity_loss_point():
    separated = torch.eye(4, 8)
    collapsed = torch.ones(4, 8)
    sep = factor_stage_diagnostics(separated, "sep", factor_dim=0)
    col = factor_stage_diagnostics(collapsed, "col", factor_dim=0)
    assert sep["sep_cos_max"].item() < 0.5
    assert col["col_cos_mean"].item() > 0.99
    assert sep["sep_l2_min"].item() > col["col_l2_min"].item()
