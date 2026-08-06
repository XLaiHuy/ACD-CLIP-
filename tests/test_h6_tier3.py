import torch
import torch.nn.functional as F
from torch import nn

from model.checkpoint_utils import load_adapter_checkpoint
from model.h6.cluster_responsibility import (
    balanced_kmeans,
    cluster_responsibility_loss,
    detached_cluster_targets,
)
from model.h6.model import H6Progress1


def _h6(cluster_enabled: bool = True) -> H6Progress1:
    return H6Progress1(
        n_groups=3,
        num_factors=4,
        top_k=2,
        bank_dim=16,
        router_dim=8,
        text_dim=32,
        ctx_len=2,
        factor_generator_specialization_enabled=True,
        cluster_responsibility_enabled=cluster_enabled,
        cluster_temperature=0.2,
        router_key_anchor_enabled=False,
    )


def _centroids() -> torch.Tensor:
    torch.manual_seed(11)
    return F.normalize(torch.randn(4, 32), dim=-1)


def test_router_patch_features_are_the_exact_dense_router_input():
    torch.manual_seed(3)
    h6 = _h6()
    raw = torch.randn(3, 2, 7, 32)
    routed = h6.router(raw, epoch_one_based=1, concept_keys=h6.semantic_core.concept_keys())
    assert torch.allclose(routed["router_input_features"], h6.router.router_input_features(raw))
    assert torch.allclose(routed["router_input_features"].norm(dim=-1), torch.ones(3, 2, 7))


def test_balanced_clustering_passes_the_five_percent_guard():
    torch.manual_seed(4)
    seeds = F.normalize(torch.randn(4, 32), dim=-1)
    patches = torch.cat([F.normalize(seed + 0.01 * torch.randn(40, 32), dim=-1) for seed in seeds])
    centroids, assignments, report = balanced_kmeans(patches, seed=17, max_initializations=3)
    assert tuple(centroids.shape) == (4, 32)
    assert tuple(assignments.shape) == (160,)
    assert report["balance_passed"]
    assert min(report["fractions"]) >= 0.05


def test_centroid_index_ties_factor_identity_semantic_slot_and_router_key():
    h6 = _h6()
    h6.bind_cluster_centroids(_centroids())
    identity = h6.cluster_identity
    core = h6.semantic_core
    assert torch.allclose(F.normalize(core.factor_id_embedding, dim=-1), identity)
    assert torch.allclose(F.normalize(core.concept_slots, dim=-1), identity)
    assert torch.allclose(core.concept_keys(), core.router_key(core.concept_slots + core.factor_id_embedding))
    assert h6.cluster_ready


def test_targets_vary_over_patch_features_and_kl_is_finite_nonzero():
    centers = _centroids()
    patches = torch.stack([centers[0], centers[1], centers[2]]).view(1, 1, 3, 32)
    targets = detached_cluster_targets(patches, centers, temperature=0.05)
    assert not torch.allclose(targets[..., 0, :], targets[..., 1, :])
    reversed_probs = targets.flip(-1).requires_grad_()
    loss, _, diag = cluster_responsibility_loss(patches, centers, reversed_probs, temperature=0.05)
    assert torch.isfinite(loss) and loss.item() > 0.0
    assert torch.isfinite(diag["cluster_target_entropy"])


def test_resp_only_backward_reaches_router_and_semantic_key_modules():
    torch.manual_seed(5)
    h6 = _h6()
    h6.bind_cluster_centroids(_centroids())
    raw = torch.randn(3, 2, 6, 32)
    routed = h6.router(raw, epoch_one_based=1, concept_keys=h6.semantic_core.concept_keys())
    loss, _, _ = cluster_responsibility_loss(
        routed["router_input_features"], h6.cluster_centroids, routed["dense_probabilities"], 0.2
    )
    loss.backward()
    assert loss.item() > 0.0
    assert h6.router.local_query_projector[0].weight.grad.norm().item() > 0.0
    assert h6.semantic_core.concept_slots.grad.norm().item() > 0.0
    assert h6.semantic_core.factor_id_embedding.grad.norm().item() > 0.0
    assert h6.semantic_core.router_key.weight.grad.norm().item() > 0.0


def test_disabling_tier3_reproduces_the_tier2_router_path():
    torch.manual_seed(6)
    tier2 = _h6(cluster_enabled=False)
    torch.manual_seed(6)
    tier3_disabled = _h6(cluster_enabled=False)
    raw = torch.randn(3, 2, 6, 32)
    key2 = tier2.semantic_core.concept_keys()
    key3 = tier3_disabled.semantic_core.concept_keys()
    out2 = tier2.router(raw, epoch_one_based=1, concept_keys=key2)["dense_probabilities"]
    out3 = tier3_disabled.router(raw, epoch_one_based=1, concept_keys=key3)["dense_probabilities"]
    assert torch.allclose(out2, out3)
    assert not tier3_disabled.cluster_ready


def test_tier3_centroids_and_metadata_round_trip_in_h6_state():
    h6 = _h6()
    h6.bind_cluster_centroids(_centroids())
    restored = _h6()
    restored.bind_cluster_centroids(h6.cluster_centroids)
    restored.load_state_dict(h6.state_dict(), strict=True)
    assert torch.allclose(restored.cluster_centroids, h6.cluster_centroids)
    assert restored.config_dict()["cluster_responsibility_enabled"]
    assert restored.config_dict()["cluster_identity_tied"]


def test_checkpoint_loader_materializes_tier3_buffers_before_validation():
    class DummyModel:
        def __init__(self):
            self.h6_enabled = True
            self.h6 = _h6()
            self.image_adapter = nn.Linear(1, 1)
            self.text_adapter = nn.Linear(1, 1)
            self.soft_prompt = nn.Linear(1, 1)

    source = DummyModel()
    source.h6.bind_cluster_centroids(_centroids())
    checkpoint = {
        "checkpoint_version": 6,
        "h6_enabled": True,
        "phase4_progress": 1,
        "h6_config": source.h6.config_dict(),
        "h6_state_dict": source.h6.state_dict(),
        "image_adapter": source.image_adapter.state_dict(),
        "text_adapter": source.text_adapter.state_dict(),
        "soft_prompt": source.soft_prompt.state_dict(),
        "epoch": 1,
    }
    restored = DummyModel()
    assert load_adapter_checkpoint(restored, checkpoint)
    assert torch.allclose(restored.h6.cluster_centroids, source.h6.cluster_centroids)
