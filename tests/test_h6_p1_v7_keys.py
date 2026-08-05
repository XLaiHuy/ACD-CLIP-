import torch
from model.h6.router import PatchRouter


def test_raw_and_final_router_keys_coexist_without_overwrite():
    router = PatchRouter(n_groups=1, num_factors=4, text_dim=8, bank_dim=4, hidden_dim=4, top_k=2)
    raw = torch.randn(4, 4)
    output = router([torch.randn(2, 4, 8)], epoch_one_based=1, concept_keys=raw)
    assert output["raw_concept_keys"] is raw
    assert output["final_router_keys"] is not raw
    assert not torch.allclose(output["raw_concept_keys"], output["final_router_keys"])
