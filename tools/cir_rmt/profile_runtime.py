#!/usr/bin/env python3
"""CIR/G4-PROFILE: bounded native/reference/optimized score profiling."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from .core import score_optimized, score_reference
from .identity import load_cir_config

def _timer(fn, warmup: int, steps: int, device: torch.device) -> tuple[float, float]:
    for _ in range(max(0, warmup)):
        fn()
    values = []
    for _ in range(max(1, steps)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append(time.perf_counter() - start)
    values.sort()
    return float(sum(values) / len(values)), float(values[int(0.95 * (len(values) - 1))])

def run_profile(device: torch.device, warmup: int, steps: int) -> dict[str, object]:
    torch.manual_seed(19)
    stages, batch, patches, groups, dim = 3, 2, 1369, 3, 768
    image = torch.nn.functional.normalize(torch.randn(stages, batch, patches, dim, device=device), dim=-1)
    text = torch.nn.functional.normalize(torch.randn(stages, batch, groups, dim, 2, device=device), dim=-2)
    weights = torch.rand(stages, batch, patches, groups, 2, device=device) + 0.1
    weights = weights / weights.sum(dim=-2, keepdim=True)
    entries = {}
    for name, fn in {"native_reference": lambda: score_reference(image, text, weights), "cir_reference": lambda: score_reference(image, text, weights), "cir_optimized": lambda: score_optimized(image, text, weights)}.items():
        mean, p95 = _timer(fn, warmup=warmup, steps=steps, device=device)
        entries[name] = {"mean_seconds": mean, "p95_seconds": p95, "images_per_sec": float(batch / max(mean, 1e-9))}
    if device.type == "cuda":
        entries["vram"] = {"peak_allocated": int(torch.cuda.max_memory_allocated(device)), "peak_reserved": int(torch.cuda.max_memory_reserved(device)), "total": int(torch.cuda.get_device_properties(device).total_memory)}
    entries["reference_optimized_max_abs"] = float((score_reference(image, text, weights) - score_optimized(image, text, weights)).abs().max())
    entries["stage"] = "CIR/G4-PROFILE"
    entries["status"] = "PASS" if entries["reference_optimized_max_abs"] <= 1e-5 else "FAIL"
    return entries

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cir_dfg_rmt_v1.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    load_cir_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = run_profile(device, args.warmup, args.steps)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
