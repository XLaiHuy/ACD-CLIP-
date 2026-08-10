"""CPU-only benchmark for exact P1-v8.4 trajectory AUROC diagnostics.

This tool performs no model construction, data loading, training, or writes
unless ``--output`` is provided.  The legacy reference is deliberately used
only at the small size because it is quadratic in unique score values.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Permit direct ``python tools/...`` execution without changing installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.h6.specialization_trajectory import binary_auroc


def legacy_binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    scores = scores.flatten().float()
    labels = labels.flatten().bool()
    positives = int(labels.sum().item())
    negatives = int((~labels).sum().item())
    if not positives or not negatives:
        return None
    order = scores.argsort()
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    for value in scores.unique():
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    return float(
        ((ranks[labels].sum() - positives * (positives + 1) / 2.0)
         / float(positives * negatives)).item()
    )


def measure(function, scores: torch.Tensor, labels: torch.Tensor) -> tuple[float | None, float]:
    started = time.perf_counter()
    result = function(scores, labels)
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-size", type=int, default=4096)
    parser.add_argument("--large-size", type=int, default=250000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(0)
    small_scores = torch.randn(args.small_size, generator=generator)
    small_labels = torch.rand(args.small_size, generator=generator) > 0.5
    large_scores = torch.randn(args.large_size, generator=generator)
    large_labels = torch.rand(args.large_size, generator=generator) > 0.5
    legacy_value, legacy_seconds = measure(legacy_binary_auroc, small_scores, small_labels)
    fast_value, fast_small_seconds = measure(binary_auroc, small_scores, small_labels)
    fast_large_value, fast_large_seconds = measure(binary_auroc, large_scores, large_labels)
    if legacy_value is None or fast_value is None or abs(legacy_value - fast_value) > 1e-7:
        raise RuntimeError("exact AUROC mismatch against the legacy small reference")
    result = {
        "semantics": "exact rank AUROC with average tie ranks",
        "training_or_model_state_touched": False,
        "legacy_small": {"samples": args.small_size, "seconds": legacy_seconds, "auroc": legacy_value},
        "optimized_small": {"samples": args.small_size, "seconds": fast_small_seconds, "auroc": fast_value},
        "optimized_large": {"samples": args.large_size, "seconds": fast_large_seconds, "auroc": fast_large_value},
        "legacy_large_intentionally_not_run": True,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.output)
    print(text, end="")


if __name__ == "__main__":
    main()
