"""Train exactly one P27 LOCO fold on source rows only; never evaluate held GT."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.sabra.data import EXPECTED_VISA_CLASSES, VisaEvaluationDataset, read_visa_metadata
from tools.sabra_v2.audit_region_distill import PROTOCOL_PATH, audit_protocol
from tools.sabra_v2.correction_teacher import build_source_teacher_region_target
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.region_cache import (
    CachedRegionDataset,
    CacheProvenance,
    RegionCacheWriter,
    preserve_rng_state,
    source_file_inventory_digest,
    source_inventory_digest,
    validate_region_cache,
)
from tools.sabra_v2.student_forward import forward_region_student, materialize_frozen_inputs
from tools.sabra_v2.p26_parent import verify_p26_parent


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-class", choices=EXPECTED_VISA_CLASSES, required=True)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--p26-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="fold-local exact cache (defaults to OUTPUT/cache)")
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=None, help="engineering cap; omit for the preregistered epoch schedule")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--engineering-smoke", action="store_true", help="label output as non-scientific and prohibit held prediction")
    return parser


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cache_provenance(args: argparse.Namespace, rows: tuple[dict[str, Any], ...]) -> CacheProvenance:
    config_path = ROOT / "configs/phase2b_canonical_v1.json"
    return CacheProvenance(
        held_class=args.held_class,
        source_classes=tuple(dict.fromkeys(str(row["class_name"]) for row in rows)),
        source_inventory_sha256=source_inventory_digest(rows),
        source_files_sha256=source_file_inventory_digest(rows, args.visa_root),
        p26_checkpoint_sha256=_sha256(args.p26_checkpoint),
        clip_asset_sha256=_sha256(args.clip_asset),
        config_sha256=_sha256(config_path),
        protocol_sha256=_sha256(PROTOCOL_PATH),
        dataset_root=str(args.visa_root.resolve()),
    )


def _build_exact_cache(
    args: argparse.Namespace,
    phase2b: torch.nn.Module,
    config: dict[str, Any],
    rows: tuple[dict[str, Any], ...],
    provenance: CacheProvenance,
    device: torch.device,
) -> dict[str, Any]:
    from model.phase2b_runtime import forward_phase2b

    started = time.perf_counter()
    writer = RegionCacheWriter(args.cache_dir, provenance, rows)
    loader = DataLoader(VisaEvaluationDataset(rows, args.visa_root), batch_size=1, shuffle=False, num_workers=0)
    with preserve_rng_state():
        for batch in loader:
            frozen = forward_phase2b(
                phase2b, batch["image"], batch["class_name"], device, config,
                domain="Industrial", require_grad=False,
            )
            source_mask = batch["mask"].to(device=device, dtype=torch.float32)
            teacher_region = build_source_teacher_region_target(frozen.native_logits, source_mask)
            seg_features, native_logits = materialize_frozen_inputs(frozen.seg_features, frozen.native_logits)
            writer.append({
                "seg_features": seg_features[:, 0].cpu(),
                "native_logits": native_logits[:, 0].cpu(),
                "teacher_region": teacher_region.cpu(),
                "source_mask": source_mask[0].cpu(),
            })
    manifest = writer.finalize()
    elapsed = time.perf_counter() - started
    return {
        "manifest": str(manifest),
        "build_seconds": elapsed,
        "seconds_per_sample": elapsed / len(rows),
        "bytes": sum(path.stat().st_size for path in args.cache_dir.glob("*.bin")),
    }


def _load_frozen_phase2b(checkpoint: Path, clip_asset: Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    from model.phase2b_runtime import load_json_config, load_phase2b_checkpoint

    config_path = ROOT / "configs/phase2b_canonical_v1.json"
    verify_p26_parent(checkpoint, clip_asset, config_path)
    config = load_json_config(config_path)
    phase2b = load_phase2b_checkpoint(checkpoint, config, clip_asset, device)
    phase2b.eval()
    for parameter in phase2b.parameters():
        parameter.requires_grad_(False)
    return phase2b, config


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive when provided")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    audit_protocol(protocol)
    if args.cache_dir is None:
        args.cache_dir = args.output / "cache"
    inventory = loco_inventory(read_visa_metadata(args.metadata), args.held_class)
    if args.batch_size != 1 and not args.engineering_smoke:
        raise ValueError("scientific P27 batch-size is frozen at 1")
    if args.epochs != 20 and not args.engineering_smoke:
        raise ValueError("scientific P27 epochs are frozen at 20")
    if args.learning_rate != 1e-3 and not args.engineering_smoke:
        raise ValueError("scientific P27 learning rate is frozen at 0.001")
    device = torch.device(args.device)
    provenance = _cache_provenance(args, inventory.fit_rows)
    cache_build: dict[str, Any] | None = None
    if not (args.cache_dir / "manifest.json").exists():
        phase2b, config = _load_frozen_phase2b(args.p26_checkpoint, args.clip_asset, device)
        cache_build = _build_exact_cache(args, phase2b, config, inventory.fit_rows, provenance, device)
        del phase2b
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        verify_p26_parent(args.p26_checkpoint, args.clip_asset, ROOT / "configs/phase2b_canonical_v1.json")
    validated_cache = validate_region_cache(args.cache_dir, provenance, inventory.fit_rows)
    _seed_everything(args.seed)
    adapter = RegionResidualAdapter().to(device)
    # The frozen parent is absent during cached optimization, making it
    # structurally impossible for P27 optimizer steps to update P26/CLIP.
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)
    source_loader = DataLoader(
        CachedRegionDataset(validated_cache), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    from utils import calculate_seg_loss

    steps = 0
    last_loss: float | None = None
    adapter.train()
    for _epoch in range(args.epochs):
        for batch in source_loader:
            seg_features = batch["seg_features"].permute(1, 0, 2, 3).to(device, non_blocking=True)
            native_logits = batch["native_logits"].permute(1, 0, 2, 3).to(device, non_blocking=True)
            teacher_region = batch["teacher_region"][:, 0].to(device, non_blocking=True)
            source_mask = batch["source_mask"].to(device, non_blocking=True)
            student = forward_region_student(adapter, seg_features, native_logits)
            distillation = F.smooth_l1_loss(student.region_residual, teacher_region.unsqueeze(0).expand(3, -1, -1, -1))
            localization = calculate_seg_loss(student.deployed_probability, source_mask)
            loss = distillation + localization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in adapter.parameters()):
                raise FloatingPointError("non-finite P27 adapter gradient")
            optimizer.step()
            steps += 1
            last_loss = float(loss.detach().cpu())
            if args.max_steps is not None and steps >= args.max_steps:
                break
        if args.max_steps is not None and steps >= args.max_steps:
            break
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "p27_region_adapter.pt"
    if checkpoint_path.exists():
        raise RuntimeError("immutable fold checkpoint already exists; refusing a rerun")
    payload = {
        "schema_version": "P27_REGION_ADAPTER_CHECKPOINT_V1",
        "status": "ENGINEERING_SMOKE_ONLY" if args.engineering_smoke else "FOLD_TRAINING_COMPLETE",
        "held_class": args.held_class,
        "state_dict": adapter.state_dict(),
        "steps": steps,
        "p26_checkpoint_sha256": _sha256(args.p26_checkpoint),
        "clip_asset_sha256": _sha256(args.clip_asset),
        "config_sha256": _sha256(ROOT / "configs/phase2b_canonical_v1.json"),
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "cache_manifest_sha256": _sha256(args.cache_dir / "manifest.json"),
        "seed": args.seed,
    }
    temporary_checkpoint = args.output / "p27_region_adapter.pt.tmp"
    torch.save(payload, temporary_checkpoint)
    temporary_checkpoint.replace(checkpoint_path)
    return {"checkpoint": str(checkpoint_path), "held_class": args.held_class, "fit_records": len(inventory.fit_rows), "held_records_not_read": len(inventory.held_rows), "held_gt_reads": 0, "held_mask_reads": 0, "steps": steps, "last_loss": last_loss, "status": payload["status"], "cache_build": cache_build}


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
