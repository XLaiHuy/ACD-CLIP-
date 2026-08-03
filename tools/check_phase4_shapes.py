#!/usr/bin/env python3
"""Dataset-free shape/finite check for the Phase4 Progress 1 semantic core."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.h6.model import H6Progress1


def main():
    torch.manual_seed(0)
    h6 = H6Progress1(n_groups=3, num_factors=4, top_k=2)
    visual = {
        "seg_tokens_pre_l2": [torch.randn(2, 16, 768) for _ in range(3)],
        "seg_tokens": [torch.nn.functional.normalize(torch.randn(2, 16, 768), dim=-1) for _ in range(3)],
        "cls24": torch.randn(2, 768),
    }
    core = h6.forward_core(visual, torch.randn(4, 768), torch.randn(4, 768))
    routing = h6.router(visual["seg_tokens"], epoch_one_based=3)
    report = {
        "projected_levels": list(core["projected_levels"].shape),
        "dynamic_contexts": list(core["dynamic_contexts"].shape),
        "router_logits": list(routing["logits"].shape),
        "topk_indices": list(routing["topk_indices"].shape),
        "probability_sum_error": float((routing["probabilities"].sum(dim=-1) - 1).abs().max()),
        "finite": bool(torch.isfinite(core["dynamic_contexts"]).all() and torch.isfinite(routing["logits"]).all()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
