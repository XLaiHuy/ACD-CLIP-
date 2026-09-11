#!/usr/bin/env python3
"""Benchmark locked Four-View TTA candidates on one fixed dataset subset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc, average_precision

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import BaseSingleClassDataset
from dataset.info import DATA_PATH, DOMAINS
from h2_clean.contract import seed_everything, sha256_file
from h2_clean.exact_metrics import ExactBinaryAccumulator
from scripts.r3_source_tta import IMG_SIZE, build_model, forward_logits_and_map, inverse_map, transform_image
from utils import get_multiple_adapted_text_embedding

POLICY = ("identity", "hflip", "vflip", "hvflip")


def load_fixed_batches(dataset_name: str, class_name: str, batch_size: int, num_batches: int):
    dataset = BaseSingleClassDataset(
        DATA_PATH[dataset_name], f"./dataset/hub/{dataset_name}.jsonl", IMG_SIZE, class_name
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = []
    for index, batch in enumerate(loader):
        if index >= num_batches:
            break
        batches.append({key: batch[key].clone() for key in ("image", "mask", "label")})
    if not batches:
        raise ValueError("fixed benchmark subset is empty")
    return batches


def sequential_views(model, text_features, image):
    return [
        forward_logits_and_map(model, text_features, transform_image(image, name), "VisA")
        for name in POLICY
    ]


def batched_views(model, text_features, image, view_names):
    batch_size = image.shape[0]
    joined = torch.cat([transform_image(image, name) for name in view_names], dim=0)
    logits, maps = forward_logits_and_map(model, text_features, joined, "VisA")
    return [
        (logits[offset:offset + batch_size], maps[offset:offset + batch_size])
        for offset in range(0, len(view_names) * batch_size, batch_size)
    ]


def run_scenario(name, model, text_features, batches, output_dir, mode, microbatch=None):
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    outputs = {"logits": [], "maps": [], "image_scores": []}
    spool = ExactBinaryAccumulator(output_dir / ".spool" / name)
    labels_all = []
    start = time.perf_counter()
    context = torch.no_grad() if mode == "baseline" else torch.inference_mode()
    with context:
        for batch in batches:
            image = batch["image"].to(model.device, non_blocking=True)
            mask = batch["mask"].to(model.device, non_blocking=True)
            labels = batch["label"].to(model.device, non_blocking=True)
            if mode in ("baseline", "safe"):
                transformed = sequential_views(model, text_features, image)
            elif mode == "batched":
                transformed = batched_views(model, text_features, image, POLICY)
            elif mode == "microbatched":
                transformed = []
                for offset in range(0, len(POLICY), microbatch or 2):
                    transformed.extend(batched_views(
                        model, text_features, image, POLICY[offset:offset + (microbatch or 2)]
                    ))
            else:
                raise ValueError(mode)
            maps = torch.stack(
                [inverse_map(value[1], name) for name, value in zip(POLICY, transformed)], dim=0
            ).mean(dim=0)
            logits = torch.stack([value[0] for value in transformed], dim=0).mean(dim=0)
            image_scores = F.softmax(logits, dim=1)[:, 1]
            image_scores = image_scores * 0.9 + maps.flatten(start_dim=1).amax(dim=1) * 0.1
            if not torch.isfinite(maps).all() or not torch.isfinite(image_scores).all():
                raise ValueError(f"non-finite output in {name}")
            spool.update(maps, mask)
            labels_all.append(labels.detach().cpu())
            outputs["logits"].append(logits.detach().cpu())
            outputs["maps"].append(maps.detach().cpu())
            outputs["image_scores"].append(image_scores.detach().cpu())
            # Match the current sequential reference implementation exactly.
            # Optimized candidates deliberately omit this hot-loop cache flush.
            if mode == "baseline" and model.device.type == "cuda":
                torch.cuda.empty_cache()
    runtime = time.perf_counter() - start
    try:
        pixel_auc, pixel_ap = spool.compute()
    finally:
        spool.cleanup()
    labels = torch.cat(labels_all).flatten()
    scores = torch.cat(outputs["image_scores"]).flatten()
    image_auc = image_ap = 0.0
    if labels.min() != labels.max():
        image_auc = float(auroc(scores, labels, task="binary").item() * 100)
        image_ap = float(average_precision(scores, labels, task="binary").item() * 100)
    outputs = {key: torch.cat(value) for key, value in outputs.items()}
    torch.save(outputs, output_dir / f"{name}_outputs.pt")
    peak = torch.cuda.max_memory_allocated(model.device) if model.device.type == "cuda" else 0
    return {
        "runtime_seconds": runtime,
        "peak_cuda_bytes": int(peak),
        "metrics": {
            "pixel_auroc": float(pixel_auc * 100), "pixel_ap": float(pixel_ap * 100),
            "image_auroc": image_auc, "image_ap": image_ap,
        },
        "output_path": f"{name}_outputs.pt", "outputs": outputs,
    }


def compare(baseline, candidate):
    output_differences = {}
    parity = True
    for key in ("logits", "maps", "image_scores"):
        diff = (candidate["outputs"][key] - baseline["outputs"][key]).abs()
        output_differences[key] = {
            "max_abs": float(diff.max().item()), "mean_abs": float(diff.mean().item())
        }
        if not torch.allclose(candidate["outputs"][key], baseline["outputs"][key], rtol=1e-4, atol=1e-5):
            parity = False
    metric_differences = {}
    for key, value in candidate["metrics"].items():
        metric_differences[key] = float(value - baseline["metrics"][key])
        if abs(metric_differences[key]) > 1e-4:
            parity = False
    return {
        "parity": "PASS" if parity else "FAIL",
        "output_differences": output_differences,
        "metric_differences": metric_differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="VisA")
    parser.add_argument("--class-name", default="candle")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-batches", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite {args.output_dir}")
    if args.dataset != "VisA":
        raise ValueError("the fixed benchmark currently uses the VisA image-score rule")
    if sha256_file(args.checkpoint) != args.expected_sha256:
        raise ValueError("checkpoint SHA256 does not match --expected-sha256")
    args.output_dir.mkdir(parents=True)
    seed_everything(0, deterministic_algorithms=True)
    batches = load_fixed_batches(args.dataset, args.class_name, args.batch_size, args.num_batches)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {}
    scenarios = (
        ("baseline_sequential", "baseline", None),
        ("safe_inference_mode", "safe", None),
        ("batched_four_view", "batched", None),
        ("microbatched_two_plus_two", "microbatched", 2),
    )
    for name, mode, microbatch in scenarios:
        model, _ = build_model(device, args.checkpoint, args.expected_sha256)
        model.device = device
        with torch.inference_mode():
            text_features = get_multiple_adapted_text_embedding(model, args.dataset, device)[args.class_name]
            warmup = batches[0]["image"][:1].to(device)
            if mode in ("baseline", "safe"):
                sequential_views(model, text_features, warmup)
            elif mode == "batched":
                batched_views(model, text_features, warmup, POLICY)
            else:
                batched_views(model, text_features, warmup, POLICY[:2])
                batched_views(model, text_features, warmup, POLICY[2:])
        results[name] = run_scenario(name, model, text_features, batches, args.output_dir, mode, microbatch)
        del text_features, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline = results["baseline_sequential"]
    summary = {
        "protocol": "RENTAL_TTA_PARITY_BENCHMARK_V1",
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset": args.dataset, "class_name": args.class_name, "num_batches": len(batches),
        "batch_size": args.batch_size, "policy": list(POLICY),
        "tolerance": {"output_rtol": 1e-4, "output_atol": 1e-5, "metric_abs": 1e-4},
        "scenarios": {},
    }
    for name, result in results.items():
        row = {key: value for key, value in result.items() if key != "outputs"}
        row["parity"] = "REFERENCE" if name == "baseline_sequential" else compare(baseline, result)
        if name != "baseline_sequential":
            row.update(row.pop("parity"))
        summary["scenarios"][name] = row
    summary["optimized_candidates_parity"] = {
        name: summary["scenarios"][name]["parity"]
        for name in summary["scenarios"] if name != "baseline_sequential"
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
