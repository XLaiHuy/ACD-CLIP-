"""Tier-3 detached patch-cluster supervision and deterministic clustering."""
from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F


def detached_cluster_targets(patch_features: torch.Tensor, centroids: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return detached q_cluster [G,B,P,M] from normalized router patch features."""
    if temperature <= 0:
        raise ValueError("cluster temperature must be positive")
    patches = F.normalize(patch_features.float(), dim=-1)
    centers = F.normalize(centroids.float(), dim=-1)
    return F.softmax(torch.einsum("gbpd,md->gbpm", patches, centers) / temperature, dim=-1).detach()


def cluster_responsibility_loss(
    patch_features: torch.Tensor,
    centroids: torch.Tensor,
    router_probs: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """KL(q_cluster || dense_router_probs), with q detached and router differentiable."""
    q = detached_cluster_targets(patch_features, centroids, temperature)
    if q.shape != router_probs.shape:
        raise ValueError(f"target/router shape mismatch: {q.shape} vs {router_probs.shape}")
    loss = F.kl_div(router_probs.float().clamp_min(1e-8).log(), q, reduction="batchmean")
    diagnostics = {
        "cluster_target_usage": q.float().mean(dim=(0, 1, 2)).detach(),
        "cluster_router_usage": router_probs.float().mean(dim=(0, 1, 2)).detach(),
        "cluster_target_entropy": -(q.float().clamp_min(1e-8).log() * q.float()).sum(dim=-1).mean().detach(),
        "cluster_router_entropy": -(
            router_probs.float().clamp_min(1e-8).log() * router_probs.float()
        ).sum(dim=-1).mean().detach(),
        "cluster_kl": loss.detach(),
    }
    return loss, q, diagnostics


def _balanced_assignment(distances: torch.Tensor) -> torch.Tensor:
    """Assign every row to a centroid with deterministic near-equal capacities."""
    if distances.ndim != 2:
        raise ValueError("distances must be [N, M]")
    count, clusters = distances.shape
    if count < clusters:
        raise ValueError("need at least one patch per cluster")
    capacities = torch.full((clusters,), count // clusters, device=distances.device, dtype=torch.long)
    capacities[: count % clusters] += 1
    flat_order = torch.argsort(distances.reshape(-1), stable=True)
    assignment = torch.full((count,), -1, device=distances.device, dtype=torch.long)
    for flat_index in flat_order.tolist():
        patch = flat_index // clusters
        cluster = flat_index % clusters
        if assignment[patch] < 0 and capacities[cluster] > 0:
            assignment[patch] = cluster
            capacities[cluster] -= 1
    if (assignment < 0).any() or capacities.any():
        raise RuntimeError("balanced assignment did not cover every patch exactly once")
    return assignment


def _deterministic_initial_centroids(features: torch.Tensor, clusters: int, seed: int) -> torch.Tensor:
    """Deterministic farthest-point seeding without external clustering dependencies."""
    generator = torch.Generator(device=features.device)
    generator.manual_seed(int(seed))
    first = int(torch.randint(features.shape[0], (1,), generator=generator, device=features.device).item())
    selected = [first]
    min_distance = 1.0 - features @ features[first]
    for _ in range(1, clusters):
        next_index = int(torch.argmax(min_distance).item())
        selected.append(next_index)
        min_distance = torch.minimum(min_distance, 1.0 - features @ features[next_index])
    return features[torch.tensor(selected, device=features.device)]


def balanced_kmeans(
    patch_features: torch.Tensor,
    num_clusters: int = 4,
    seed: int = 0,
    max_initializations: int = 3,
    max_iterations: int = 30,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run bounded deterministic balanced k-means and return normalized centroids.

    Each initialization has an exact balanced assignment, so the report's
    minimum-cluster check is a meaningful guard rather than a best-effort
    post-hoc repair.
    """
    if patch_features.ndim != 2:
        raise ValueError("patch_features must be [N, D]")
    if num_clusters != 4:
        raise ValueError("Tier-3 is fixed to M=4 clusters")
    if not 1 <= max_initializations <= 3:
        raise ValueError("max_initializations must be in [1, 3]")
    features = F.normalize(patch_features.detach().float(), dim=-1)
    if features.shape[0] < num_clusters * 20:
        raise ValueError("need at least 20 patches per Tier-3 cluster")
    best: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None = None
    for init in range(max_initializations):
        centers = _deterministic_initial_centroids(features, num_clusters, seed + init)
        for _ in range(max_iterations):
            distances = 1.0 - features @ centers.T
            assignment = _balanced_assignment(distances)
            updated = torch.stack([
                F.normalize(features[assignment == cluster].mean(dim=0), dim=0)
                for cluster in range(num_clusters)
            ])
            if torch.allclose(updated, centers, rtol=0.0, atol=1e-7):
                centers = updated
                break
            centers = updated
        distances = 1.0 - features @ centers.T
        assignment = _balanced_assignment(distances)
        objective = distances.gather(1, assignment[:, None]).mean()
        if best is None or objective < best[0]:
            best = (objective, centers, assignment, init)
    assert best is not None
    objective, centers, assignment, selected_initialization = best
    counts = torch.bincount(assignment, minlength=num_clusters)
    fractions = counts.float() / float(assignment.numel())
    if float(fractions.min()) < 0.05:
        raise RuntimeError("rejected cluster solution: a cluster is below the 5% minimum")
    report: dict[str, Any] = {
        "algorithm": "deterministic_balanced_kmeans",
        "num_clusters": int(num_clusters),
        "seed": int(seed),
        "max_initializations": int(max_initializations),
        "selected_initialization": int(selected_initialization),
        "max_iterations": int(max_iterations),
        "objective": float(objective.cpu()),
        "counts": counts.cpu().tolist(),
        "fractions": fractions.cpu().tolist(),
        "min_fraction": float(fractions.min().cpu()),
        "balance_passed": True,
    }
    return centers.cpu(), assignment.cpu(), report


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Stable content fingerprint for patch-bank and centroid provenance."""
    values = tensor.detach().contiguous().cpu().float().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()
