#!/usr/bin/env python3
"""Summarize A0/A1 geometry and semantic-affinity alignment from saved probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _summary(values):
    value = torch.cat(values).float()
    return {
        "count": int(value.numel()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/phase4/k1_noop/short64_seed0"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.run_dir / "K1_NOOP_A0_A1_DISTINGUISHABILITY.json"
    regions = ("all", "normal", "anomaly")
    cosine, l2 = [], []
    margin = {region: [] for region in regions}
    alpha_dynamic = {region: [] for region in regions}
    for batch_index in (1, 32, 64):
        probe = torch.load(
            args.run_dir / "k1_fixed_train_probes" / f"batch_{batch_index:03d}.pt",
            map_location="cpu",
            weights_only=False,
        )
        a0 = probe["noop_base_abnormal_semantic"].float()
        a1 = probe["noop_dynamic_abnormal_semantic"].float()
        scores = probe["noop_scores"].float()
        alpha = probe["noop_alpha"].float()
        groups, _, patches, _ = scores.shape
        side = int(patches**0.5)
        target = F.adaptive_avg_pool2d(probe["mask"].float(), (side, side)).flatten(1)
        valid = F.adaptive_avg_pool2d(probe["local_mask_valid"].float(), (side, side)).flatten(1) >= 1.0 - 1e-6
        target = target.unsqueeze(0).expand(groups, -1, -1)
        valid = valid.unsqueeze(0).expand_as(target)
        masks = {"all": valid, "normal": valid & (target < 0.5), "anomaly": valid & (target >= 0.5)}
        cosine.append(F.cosine_similarity(a0, a1, dim=-1).reshape(-1))
        l2.append((a1 - a0).norm(dim=-1).reshape(-1))
        for region, mask in masks.items():
            margin[region].append((scores[..., 1] - scores[..., 0])[mask])
            alpha_dynamic[region].append(alpha[..., 1][mask])
    report = {
        "a0_a1_geometry": {"cosine": _summary(cosine), "l2": _summary(l2)},
        "affinity_margin_score1_minus_score0": {region: _summary(values) for region, values in margin.items()},
        "alpha_dynamic": {region: _summary(values) for region, values in alpha_dynamic.items()},
        "note": "Descriptive audit only; it defines no success threshold.",
    }
    normal = report["alpha_dynamic"]["normal"]["mean"]
    anomaly = report["alpha_dynamic"]["anomaly"]["mean"]
    report["diagnosis"] = (
        "A0_A1_GEOMETRICALLY_DISTINGUISHABLE_BUT_AFFINITY_MISALIGNED"
        if report["a0_a1_geometry"]["l2"]["max"] > 0.0 and normal > anomaly
        else "A0_A1_DISTINGUISHABILITY_INCONCLUSIVE"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
