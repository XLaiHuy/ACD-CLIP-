import importlib.util
from pathlib import Path

import torch

from model.h6.router import PatchRouter
from model.h6.semantic_bank import CoPSSemanticCore


_info_spec = importlib.util.spec_from_file_location("phase4_dataset_info", Path(__file__).parents[1] / "dataset" / "info.py")
_info = importlib.util.module_from_spec(_info_spec)
_info_spec.loader.exec_module(_info)
PROMPTS = _info.PROMPTS


def _hard_prompt_expansion(name: str, state: int):
    key = "prompt_normal" if state == 0 else "prompt_abnormal"
    return [template.format(prompt.format(name)) for prompt in PROMPTS[key] for template in PROMPTS["prompt_templates"]]


def test_hard_prompt_integrity_and_expansion_counts():
    assert PROMPTS["prompt_normal"] == ["{}", "a {}", "the {}"]
    assert PROMPTS["prompt_abnormal"] == [
        "a damaged {}", "a broken {}", "a {} with flaw", "a {} with defect", "a {} with damage",
    ]
    assert PROMPTS["prompt_templates"] == ["{}.", "a photo of {}."]
    assert len(_hard_prompt_expansion("object", 0)) == 6
    assert len(_hard_prompt_expansion("object", 1)) == 10


def test_semantic_core_shapes_gates_and_correspondence():
    torch.manual_seed(0)
    core = CoPSSemanticCore(n_groups=3, num_factors=4, bank_dim=256, ctx_len=4)
    levels = [torch.randn(2, 16, 768) for _ in range(3)]
    output = core(levels, torch.randn(2, 768), torch.randn(4, 768), torch.randn(4, 768))
    assert output["projected_levels"].shape == (3, 2, 16, 256)
    assert output["multi_features"].shape == (2, 48, 256)
    assert output["prototype_normal"].shape == (2, 4, 256)
    assert output["prototype_abnormal"].shape == (2, 4, 256)
    assert output["dynamic_contexts"].shape == (2, 4, 2, 4, 768)
    assert output["class_semantic"].shape == (2, 768)
    assert torch.isfinite(output["dynamic_contexts"]).all()
    assert torch.allclose(output["gamma_state"], torch.tensor([0.05]), atol=1e-6)
    assert torch.allclose(output["gamma_class"], torch.tensor([0.02]), atol=1e-6)
    assert core.normal_query.weight.data_ptr() != core.abnormal_query.weight.data_ptr()
    assert core.concept_slots.shape == (4, 256)


def test_router_probabilities_and_topk_after_warmup():
    router = PatchRouter(n_groups=3, num_factors=4, top_k=2, soft_routing_epochs=2)
    tokens = torch.randn(3, 2, 16, 768)
    dense = router(tokens, epoch_one_based=1)
    assert dense["probabilities"].shape == (3, 2, 16, 4)
    assert torch.allclose(dense["probabilities"].sum(dim=-1), torch.ones(3, 2, 16), atol=1e-6)
    sparse = router(tokens, epoch_one_based=3)
    assert torch.allclose(sparse["probabilities"].sum(dim=-1), torch.ones(3, 2, 16), atol=1e-6)
    assert (sparse["probabilities"] > 0).sum(dim=-1).eq(2).all()
    assert torch.isfinite(sparse["logits"]).all()
