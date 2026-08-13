#!/usr/bin/env python3
"""Cache-only intrinsic role/cardinality audit for P1-v8.4-A.

This intentionally never constructs the model and never performs an optimizer
step.  It consumes the persisted one-pass E3 cache and emits compact JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from model.h6.utility_routing import utility_teacher


def _json(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    return x


def _rank_summary(x: torch.Tensor) -> dict:
    """Rank/energy/correlation of scalar factor functions in x[..., M]."""
    x = x.float().reshape(-1, x.shape[-1])
    if x.shape[0] < 2:
        return {"n": int(x.shape[0]), "effective_rank": 0.0}
    # Covariance is sufficient because M is four; avoid retaining a 2.5M x M
    # NumPy copy in the report process.
    xc = x - x.mean(0, keepdim=True)
    cov = (xc.T @ xc) / max(1, x.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
    energy = eig / eig.sum().clamp_min(1e-20)
    positive = energy[energy > 0]
    erank = float(torch.exp(-(positive * positive.log()).sum()))
    std = torch.sqrt(torch.diag(cov).clamp_min(1e-20))
    corr = cov / (std[:, None] * std[None, :]).clamp_min(1e-20)
    corr.fill_diagonal_(1.0)
    upper = corr.abs()[torch.triu(torch.ones_like(corr, dtype=torch.bool), diagonal=1)]
    return {
        "n": int(x.shape[0]),
        "effective_rank": erank,
        "pca_energy": energy.tolist(),
        "pairwise_correlation": corr.tolist(),
        "max_abs_pairwise_correlation": float(upper.max()) if upper.numel() else 0.0,
    }


def _utility_summary(x: torch.Tensor) -> dict:
    x = x.float().reshape(-1, x.shape[-1])
    if x.numel() == 0:
        return {"n": 0}
    winners = x.argmax(-1)
    counts = torch.bincount(winners, minlength=x.shape[-1]).float()
    return {
        "n": int(x.shape[0]),
        "mean_gain": x.mean(0).tolist(),
        "std_gain": x.std(0).tolist(),
        "winner_share": (counts / x.shape[0]).tolist(),
        "gain_covariance": torch.cov(x.T).tolist() if x.shape[0] > 1 else torch.zeros((x.shape[-1], x.shape[-1])).tolist(),
    }


def _bootstrap_ci(values: torch.Tensor, seed: int = 841, n_boot: int = 1000) -> list[float]:
    values = values.float().flatten()
    if values.numel() == 0:
        return [float("nan"), float("nan")]
    g = torch.Generator().manual_seed(seed)
    # Image-level bootstrap: each image contributes one region-mean value.
    idx = torch.randint(values.numel(), (n_boot, values.numel()), generator=g)
    means = values[idx].mean(1)
    return [float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))]


def _mask_stats(name: str, mask: torch.Tensor, residual: torch.Tensor, gain: torch.Tensor) -> dict:
    return {
        "region": name,
        "residual_function": _rank_summary(residual[mask]),
        "factor_utility": _utility_summary(gain[mask]),
    }


def _role_stats(
    role_gain: torch.Tensor,
    patch_mask: torch.Tensor,
    image_region_gain: torch.Tensor,
    image_region_mask: torch.Tensor,
    class_names: list[str],
    class_ids: list[str],
    *,
    anomaly_roles: tuple[int, ...] = (),
) -> dict:
    """Role means, winners, leave-role-out utility and image bootstrap CIs."""
    out: dict = {"patch": {}, "image_region": {}, "per_class_image_region": {}}
    for label, mask, ig, imask in [
        ("patch", patch_mask, None, None),
        ("image_region", None, image_region_gain, image_region_mask),
    ]:
        if label == "patch":
            vals = role_gain[mask]
        else:
            vals = ig[imask]
        if vals.numel() == 0:
            out[label] = {"n": 0}
            continue
        full = vals.max(-1).values
        entry = {
            "n": int(vals.shape[0]),
            "mean_gain": vals.mean(0).tolist(),
            "winner_share": (torch.bincount(vals.argmax(-1), minlength=vals.shape[-1]).float() / vals.shape[0]).tolist(),
            "leave_role_out_unique_mean": [float((full - vals[..., j]).mean()) for j in range(vals.shape[-1])],
        }
        if label == "image_region":
            entry["leave_role_out_unique_ci95"] = [
                _bootstrap_ci(full - vals[..., j], seed=841 + j) for j in range(vals.shape[-1])
            ]
        out[label] = entry
    # Per-class role utility is image-level, not patch-count weighted.
    for cls in class_names:
        cm = torch.tensor([c == cls for c in class_ids], dtype=torch.bool)
        mask = image_region_mask & cm
        vals = image_region_gain[mask]
        if vals.numel() == 0:
            continue
        full = vals.max(-1).values
        out["per_class_image_region"][cls] = {
            "n_images": int(vals.shape[0]),
            "mean_gain": vals.mean(0).tolist(),
            "winner_share": (torch.bincount(vals.argmax(-1), minlength=vals.shape[-1]).float() / vals.shape[0]).tolist(),
            "leave_role_out_unique_mean": [float((full - vals[..., j]).mean()) for j in range(vals.shape[-1])],
            "leave_role_out_unique_ci95": [
                _bootstrap_ci(full - vals[..., j], seed=9841 + 17 * j) for j in range(vals.shape[-1])
            ],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cache_bytes = args.cache.read_bytes()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    n = len(cache["class"])
    residual = torch.stack(cache["residual"]).squeeze(2)  # [N,G,P,M]
    z0 = torch.stack(cache["z0"]).squeeze(2)  # [N,G,P]
    dense = torch.stack(cache["dense"]).squeeze(2)
    target = torch.stack([x.squeeze(0) for x in cache["target"]]).float()  # [N,P]
    valid = torch.stack([x.squeeze(0) for x in cache["valid"]]).bool()
    classes = [str(x) for x in cache["class"]]
    labels = torch.tensor(cache["label"], dtype=torch.long)
    class_names = sorted(set(classes))

    # The call is CPU-only and consumes cached tensors; this recomputes teacher
    # labels but does not replay the model or touch any parameter.
    teacher = utility_teacher(
        z0.permute(1, 0, 2),
        residual.permute(1, 0, 2, 3),
        target,
        valid,
        rho=0.05,
        router_confidence_mode="margin_rel",
        router_margin_rel_threshold=0.10,
        routed_probabilities=dense.permute(1, 0, 2, 3),
    )
    gain = teacher["gain_rel"].permute(1, 0, 2, 3).mean(1)  # [N,P,M]
    residual_mean = residual.mean(1)
    all_mask = valid
    normal_patch = valid & (target <= 0)
    anomaly_patch = valid & (target > 0)
    normal_images = labels == 0
    anomaly_images = labels == 1

    regions: dict[str, dict] = {}
    regions["all_valid"] = _mask_stats("all_valid", all_mask, residual_mean, gain)
    regions["normal_patch"] = _mask_stats("normal_patch", normal_patch, residual_mean, gain)
    regions["anomaly_patch"] = _mask_stats("anomaly_patch", anomaly_patch, residual_mean, gain)
    for cls in class_names:
        cm = torch.tensor([c == cls for c in classes], dtype=torch.bool)[:, None]
        regions[f"class:{cls}:all"] = _mask_stats(f"class:{cls}:all", valid & cm, residual_mean, gain)
        regions[f"class:{cls}:anomaly"] = _mask_stats(f"class:{cls}:anomaly", anomaly_patch & cm, residual_mean, gain)

    # Image-level region means prevent the very large normal background area
    # from determining the role decision.
    image_gain = torch.zeros((n, 2, gain.shape[-1]))
    image_gain[:, 0] = torch.where(normal_patch[..., None], gain, 0).sum(1) / normal_patch.sum(1).clamp_min(1)[:, None]
    image_gain[:, 1] = torch.where(anomaly_patch[..., None], gain, 0).sum(1) / anomaly_patch.sum(1).clamp_min(1)[:, None]
    image_region_mask = torch.stack([normal_images, anomaly_images], 1)

    # Enumerate semantic partitions but do not use k-means.  Factor 0 is kept
    # in role A to avoid duplicate complements.
    partitions = []
    for r in range(1, 4):
        for a in itertools.combinations(range(4), r):
            if 0 not in a or len(a) == 4:
                continue
            b = tuple(i for i in range(4) if i not in a)
            role_gain_patch = torch.stack([gain[..., list(a)].max(-1).values, gain[..., list(b)].max(-1).values], -1)
            role_gain_image = torch.stack([image_gain[..., 0, list(a)].max(-1).values, image_gain[..., 0, list(b)].max(-1).values], -1)
            role_gain_anom_image = torch.stack([image_gain[..., 1, list(a)].max(-1).values, image_gain[..., 1, list(b)].max(-1).values], -1)
            role_gain_image_regions = torch.stack([role_gain_image, role_gain_anom_image], 1)
            role_gain_image_mask = image_region_mask
            entry = {
                "role_A_factors": list(a),
                "role_B_factors": list(b),
                "stats": _role_stats(
                    role_gain_patch,
                    torch.stack([normal_patch, anomaly_patch], 1).any(1),
                    role_gain_image_regions.reshape(n * 2, 2),
                    role_gain_image_mask.reshape(n * 2),
                    class_names,
                    classes * 2,
                ),
            }
            # Explicit region splits are more interpretable than the combined
            # mask used by the generic helper.
            entry["normal_patch"] = _role_stats(role_gain_patch, normal_patch, role_gain_image_regions[:, 0], normal_images, class_names, classes)
            entry["anomaly_patch"] = _role_stats(role_gain_patch, anomaly_patch, role_gain_image_regions[:, 1], anomaly_images, class_names, classes)
            partitions.append(entry)

    # The semantically motivated R2 partition is {F1,F4} vs {F2,F3}.
    r2 = next(x for x in partitions if x["role_A_factors"] == [0, 3])
    r2_patch = torch.stack([gain[..., [0, 3]].max(-1).values, gain[..., [1, 2]].max(-1).values], -1)
    r2_img = torch.stack([image_gain[..., 0, [0, 3]].max(-1).values, image_gain[..., 0, [1, 2]].max(-1).values], -1)
    r2_anom_img = torch.stack([image_gain[..., 1, [0, 3]].max(-1).values, image_gain[..., 1, [1, 2]].max(-1).values], -1)

    # R3 keeps the normal/background role and tests F2/F3 as two anomaly roles.
    r3_patch = torch.stack([gain[..., [0, 3]].max(-1).values, gain[..., 1], gain[..., 2]], -1)
    r3_img = torch.stack([image_gain[..., 0, [0, 3]].max(-1).values, image_gain[..., 0, 1], image_gain[..., 0, 2]], -1)
    r3_anom_img = torch.stack([image_gain[..., 1, [0, 3]].max(-1).values, image_gain[..., 1, 1], image_gain[..., 1, 2]], -1)
    r3 = {
        "roles": {"normal": [0, 3], "anomaly_A": [1], "anomaly_B": [2]},
        "normal_patch": _role_stats(r3_patch, normal_patch, r3_img, normal_images, class_names, classes),
        "anomaly_patch": _role_stats(r3_patch, anomaly_patch, r3_anom_img, anomaly_images, class_names, classes),
    }

    # Direct anomaly-mode test: each anomaly role must retain unique utility
    # after the other anomaly role and the normal role are available.
    av = r3_anom_img[anomaly_images]
    full = av.max(-1).values
    r3["anomaly_image_unique"] = {
        "anomaly_A_F2_mean": float((full - av[:, [0, 2]].max(-1).values).mean()),
        "anomaly_A_F2_ci95": _bootstrap_ci(full - av[:, [0, 2]].max(-1).values, seed=1901),
        "anomaly_B_F3_mean": float((full - av[:, [0, 1]].max(-1).values).mean()),
        "anomaly_B_F3_ci95": _bootstrap_ci(full - av[:, [0, 1]].max(-1).values, seed=1902),
        "pairwise_anomaly_gain_corr": torch.corrcoef(av[:, 1:].T).tolist(),
    }

    out = {
        "audit": "P1_V84A_ROLE_CARDINALITY_R0_CACHE_ONLY",
        "cache": {
            "path": str(args.cache),
            "sha256": hashlib.sha256(cache_bytes).hexdigest(),
            "images": n,
            "classes": class_names,
            "forward_replayed": False,
            "optimizer_steps": 0,
        },
        "provenance": {"checkpoint": "runs/p1_v84a_gpu/factor_generator_specialization_fresh_3e_seed0/adapter_3.pth", "rho": 0.05, "teacher_source": "utility_teacher over cached base/residual/target/valid"},
        "regions": regions,
        "semantic_partitions": partitions,
        "R2": {"roles": {"normal": [0, 3], "anomaly": [1, 2]}, "evidence": r2},
        "R3": r3,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json(out), indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "images": n,
        "R2_anomaly_role_patch_winner": r2["anomaly_patch"]["patch"]["winner_share"],
        "R2_normal_role_patch_winner": r2["normal_patch"]["patch"]["winner_share"],
        "R3_F2_unique_ci95": out["R3"]["anomaly_image_unique"]["anomaly_A_F2_ci95"],
        "R3_F3_unique_ci95": out["R3"]["anomaly_image_unique"]["anomaly_B_F3_ci95"],
    }, indent=2))


if __name__ == "__main__":
    main()
