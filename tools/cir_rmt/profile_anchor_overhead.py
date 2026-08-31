#!/usr/bin/env python3
"""Profile the selected anchor once on one real VisA training batch.

This is a measurement-only command: it performs no optimizer step and writes
only a compact timing JSON.  It is intentionally separate from the training
loop so the matched experiment has no per-step diagnostic forward pass.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from model.phase2b_runtime import build_phase2b_trainable, configure_canonical_fp32
from scripts.cir_rmt.train_full import _optimizer, _set_epoch_state, _text_with_regularizers, build_loader
from tools.cir_rmt.identity import load_cir_config
from tools.cir_rmt.parameter_anchor import load_image_parameter_anchor
from tools.cir_rmt.runtime import forward_cir
from utils import calculate_seg_loss, make_dataloader_generator


ROOT = Path(__file__).resolve().parents[2]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _step(model: Any, optimizer: torch.optim.Optimizer, batch: dict[str, Any], device: torch.device, config: dict[str, Any], anchor: Any | None) -> float:
    optimizer.zero_grad(set_to_none=True)
    image = batch["image"].to(device).float()
    masks = batch["mask"].to(device).float()
    labels = batch["label"].to(device).long()
    classes = [str(value) for value in batch["class_name"]]
    started = time.perf_counter()
    text, kg_loss, k_loss = _text_with_regularizers(model, classes, config, device)
    output = forward_cir(model, image, classes, device, config["_cir_config"], domain="Industrial", require_grad=True, dataset_name="VisA", precomputed_text_features=text)
    cls_loss = F.cross_entropy(output.classification_logits.float(), labels)
    seg_loss = calculate_seg_loss(output.cir_training_segmentation_probability.float(), masks.float())
    loss = cls_loss + seg_loss + float(config.get("lambda_kg", 0.001)) * kg_loss + float(config.get("lambda_k", 0.0)) * k_loss
    if anchor is not None:
        loss = loss + float(config["_anchor_lambda"]) * anchor.loss(model.image_adapter)
    loss.backward()
    _sync(device)
    elapsed = time.perf_counter() - started
    optimizer.zero_grad(set_to_none=True)
    del output, image, masks, labels, text
    return elapsed


def run(args: argparse.Namespace) -> None:
    cir_config = load_cir_config(Path(args.config))
    parent_path = Path(cir_config.get("parent_config_path", "configs/phase2b_canonical_v1.json"))
    if not parent_path.is_absolute():
        parent_path = ROOT / parent_path
    parent_config = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_config.update({"dataset": "VisA", "seed": 0, "epochs": 14, "micro_batch_size": 6, "batch_size": 6, "grad_accum_steps": 1, "effective_batch_size": 6, "num_workers": 0, "pin_memory": False, "persistent_workers": False, "prefetch_factor": 2})
    configure_canonical_fp32()
    device = torch.device(args.device)
    model = build_phase2b_trainable(parent_config, Path(args.clip_asset), device)
    optimizer = _optimizer(model, parent_config)
    _set_epoch_state(model, optimizer, parent_config, 10)
    anchor = load_image_parameter_anchor(Path(args.anchor_checkpoint), model, device)
    loader_args = SimpleNamespace(micro_batch_size=6, num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=2)
    _, loader = build_loader("visa", Path(args.source_root), parent_config, loader_args, make_dataloader_generator(0))
    batch = next(iter(loader))
    timing_config = dict(parent_config, _cir_config=cir_config, _anchor_lambda=float(args.anchor_lambda))
    model.train(); model.clipmodel.eval(); model.image_encoder.eval()
    for _ in range(int(args.warmup)):
        _step(model, optimizer, batch, device, timing_config, None)
        _step(model, optimizer, batch, device, timing_config, anchor)
    baseline = [_step(model, optimizer, batch, device, timing_config, None) for _ in range(int(args.repetitions))]
    anchored = [_step(model, optimizer, batch, device, timing_config, anchor) for _ in range(int(args.repetitions))]
    baseline_median = float(torch.tensor(baseline).median().item())
    anchored_median = float(torch.tensor(anchored).median().item())
    result = {
        "status": "PASS",
        "scope": "one_real_visa_training_batch_no_optimizer_step",
        "device": str(device),
        "batch_size": 6,
        "anchor_reference_resident_once": True,
        "anchor_reference_epoch": anchor.reference_epoch,
        "anchor_reference_checkpoint_sha256": anchor.reference_checkpoint_sha256,
        "anchor_lambda": float(args.anchor_lambda),
        "repetitions": int(args.repetitions),
        "baseline_step_seconds": baseline,
        "anchored_step_seconds": anchored,
        "baseline_median_seconds": baseline_median,
        "anchored_median_seconds": anchored_median,
        "anchor_overhead_seconds": anchored_median - baseline_median,
        "anchor_overhead_percent": 100.0 * (anchored_median - baseline_median) / max(baseline_median, 1.0e-12),
        "no_vectorization_change_after_profile": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--anchor-checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor-lambda", type=float, default=1.0e-3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
