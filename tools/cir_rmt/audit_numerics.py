#!/usr/bin/env python3
"""CIR/G1-MATH: deterministic synthetic numerical audit."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from .core import cir_logits_from_native_weights, midpoint_median, peer_delta_from_native_margins, score_optimized, score_reference, transport_pair
from .identity import load_cir_config, release_identity_fields

def run_audit(seed: int = 17) -> dict[str, object]:
    torch.manual_seed(int(seed))
    stages, batch, patches, groups, dim = 3, 2, 16, 3, 13
    image = torch.nn.functional.normalize(torch.randn(stages, batch, patches, dim), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(stages, batch, groups, dim, 2), dim=-2)
    native = torch.rand(stages, batch, groups, 2) + 0.1
    native = native / native.sum(dim=-2, keepdim=True)
    delta = torch.randn(stages, batch, patches).tanh()
    reference, optimized = score_reference(image, text, native), score_optimized(image, text, native)
    cir, native_score = cir_logits_from_native_weights(image, text, native, delta, 0.0, score_mode="reference")
    group_delta = delta.permute(1, 2, 0).unsqueeze(0).expand(stages, batch, patches, groups)
    normal, abnormal = transport_pair(native[..., 0].unsqueeze(2).expand(stages, batch, patches, groups), native[..., 1].unsqueeze(2).expand(stages, batch, patches, groups), group_delta, 0.5)
    features = torch.nn.functional.normalize(torch.randn(stages, batch, 49, dim), dim=-1)
    margins = torch.randn(stages, batch, 49)
    peer_delta, peer_stats = peer_delta_from_native_margins(features, margins)
    checks = {
        "stage": "CIR/G1-MATH",
        "reference_optimized_max_abs": float((reference - optimized).abs().max()),
        "alpha0_native_max_abs": float((cir - native_score).abs().max()),
        "weight_sum_normal_max_abs": float((normal.sum(-1) - 1).abs().max()),
        "weight_sum_abnormal_max_abs": float((abnormal.sum(-1) - 1).abs().max()),
        "delta_abs_max": float(peer_delta.abs().max()),
        "midpoint_median_k8": float(midpoint_median(torch.arange(8, dtype=torch.float32))),
        "peer_valid_fraction": float(peer_stats["valid"].float().mean()),
        "finite": bool(torch.isfinite(cir).all() and torch.isfinite(normal).all() and torch.isfinite(abnormal).all()),
        "real_mini_batch": "NOT_RUN_NO_ASSETS",
    }
    checks["status"] = "PASS" if checks["reference_optimized_max_abs"] <= 1e-5 and checks["alpha0_native_max_abs"] <= 1e-6 and checks["weight_sum_normal_max_abs"] <= 1e-6 and checks["weight_sum_abnormal_max_abs"] <= 1e-6 and checks["delta_abs_max"] <= 1.0 and checks["finite"] else "FAIL"
    return checks

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cir_dfg_rmt_v1.json"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = load_cir_config(args.config)
    started = time.perf_counter()
    result = run_audit(args.seed)
    result.update({"gate": "G1", "scope": "unit", "real": False, "identity": release_identity_fields(config)})
    result["elapsed_seconds"] = time.perf_counter() - started
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
