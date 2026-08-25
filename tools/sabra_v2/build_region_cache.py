"""Build exact P27 frozen-feature or fold-local source-supervision caches."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from PIL import Image
from torch.utils.data import DataLoader

from tools.sabra.data import (
    EXPECTED_VISA_CLASSES,
    VisaEvidenceDataset,
    _mask_transform,
    read_visa_metadata,
    safe_data_path,
)
from tools.sabra_v2.audit_region_distill import PROTOCOL_PATH, audit_protocol
from tools.sabra_v2.correction_teacher import build_source_teacher_region_target
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.p26_parent import verify_p26_parent
from tools.sabra_v2.region_cache import (
    MASK_SHAPE,
    NATIVE_LOGIT_SHAPE,
    SEGMENTATION_SHAPE,
    TEACHER_SHAPE,
    CacheProvenance,
    TierADataset,
    atomic_write_json,
    sha256_file,
    stable_sample_id,
    write_tier_a_shard,
    write_tier_b_shard,
)
from tools.sabra_v2.train_region_distill import ROOT, _load_frozen_phase2b


MINIMUM_FREE_RESERVE_BYTES = 80 * 1024**3


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("a", "b"), required=True)
    parser.add_argument("--held-class", choices=EXPECTED_VISA_CLASSES)
    parser.add_argument("--visa-root", type=Path, required=True)
    parser.add_argument("--p26-checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--execution-base-sha", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, choices=(0, 2, 4), default=0)
    parser.add_argument("--prefetch-factor", type=int, choices=(2, 4), default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--engineering-limit-per-class", type=int)
    return parser


def _provenance(args: argparse.Namespace) -> CacheProvenance:
    execution_base = str(args.execution_base_sha or _git_head())
    return CacheProvenance(
        scientific_execution_base_sha=execution_base,
        metadata_sha256=sha256_file(args.metadata),
    )


def _loader(dataset: torch.utils.data.Dataset, args: argparse.Namespace) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory),
    }
    if args.num_workers:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    return DataLoader(dataset, **kwargs)


def _projected_bytes(record_count: int) -> dict[str, int]:
    tier_a_per_sample = (
        torch.empty(SEGMENTATION_SHAPE, dtype=torch.float32).numel()
        + torch.empty(NATIVE_LOGIT_SHAPE, dtype=torch.float32).numel()
    ) * 4
    tier_b_per_exposure = (
        torch.empty(MASK_SHAPE, dtype=torch.float32).numel()
        + torch.empty(TEACHER_SHAPE, dtype=torch.float32).numel()
    ) * 4
    return {
        "tier_a_bytes": record_count * tier_a_per_sample,
        "all_fold_tier_b_bytes": record_count * 11 * tier_b_per_exposure,
        "total_bytes": record_count * tier_a_per_sample + record_count * 11 * tier_b_per_exposure,
    }


def _check_space(args: argparse.Namespace, record_count: int) -> dict[str, int]:
    projection = _projected_bytes(record_count)
    free = shutil.disk_usage(args.cache_root.parent).free
    if args.engineering_limit_per_class is None and free - projection["total_bytes"] < MINIMUM_FREE_RESERVE_BYTES:
        raise RuntimeError(
            f"cache would violate 80 GiB reserve: free={free}, projected={projection['total_bytes']}"
        )
    return {**projection, "free_bytes_before": free, "minimum_free_reserve_bytes": MINIMUM_FREE_RESERVE_BYTES}


def build_tier_a(args: argparse.Namespace, rows: list[dict[str, Any]], provenance: CacheProvenance) -> dict[str, Any]:
    if args.held_class is not None:
        raise ValueError("Tier A is class-sharded and must not receive --held-class")
    device = torch.device(args.device)
    phase2b, config = _load_frozen_phase2b(args.p26_checkpoint, args.clip_asset, device)
    from model.phase2b_runtime import forward_phase2b

    class_summaries: dict[str, Any] = {}
    started_all = time.perf_counter()
    for class_name in EXPECTED_VISA_CLASSES:
        class_rows = [row for row in rows if row["class_name"] == class_name]
        if args.engineering_limit_per_class is not None:
            class_rows = class_rows[: args.engineering_limit_per_class]
        dataset = VisaEvidenceDataset(class_rows, args.visa_root)
        loader = _loader(dataset, args)
        sample_seconds: list[float] = []

        def tensors() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            for batch in loader:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                frozen = forward_phase2b(
                    phase2b,
                    batch["image"],
                    batch["class_name"],
                    device,
                    config,
                    domain="Industrial",
                    require_grad=False,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                sample_seconds.append(time.perf_counter() - started)
                yield frozen.seg_features[:, 0], frozen.native_logits[:, 0]

        started = time.perf_counter()
        manifest = write_tier_a_shard(
            args.cache_root,
            class_name,
            [stable_sample_id(row) for row in class_rows],
            tensors(),
            provenance,
        )
        class_summaries[class_name] = {
            "samples": len(class_rows),
            "seconds": time.perf_counter() - started,
            "median_frozen_forward_seconds": statistics.median(sample_seconds) if sample_seconds else None,
            "manifest": str(args.cache_root / "tier_a" / class_name / "manifest.json"),
            "completion_status": manifest["completion_status"],
        }
    del phase2b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "tier": "A",
        "seconds": time.perf_counter() - started_all,
        "classes": class_summaries,
        "engineering_limit_per_class": args.engineering_limit_per_class,
    }


def _source_mask(row: dict[str, Any], visa_root: Path, transform: Any) -> torch.Tensor:
    if int(row["label"]) == 0:
        return torch.zeros(MASK_SHAPE, dtype=torch.float32)
    mask_path_value = row.get("mask_path")
    if not isinstance(mask_path_value, str) or not mask_path_value:
        raise RuntimeError("anomalous source row lacks a mask path")
    mask_path = safe_data_path(visa_root, mask_path_value)
    with Image.open(mask_path) as handle:
        mask_image = handle.convert("L").copy()
    return transform(mask_image).gt(0).to(torch.float32).contiguous()


def build_tier_b(args: argparse.Namespace, rows: list[dict[str, Any]], provenance: CacheProvenance) -> dict[str, Any]:
    if args.held_class is None:
        raise ValueError("Tier B requires --held-class")
    inventory = loco_inventory(rows, args.held_class)
    source_rows = list(inventory.fit_rows)
    frozen = TierADataset(source_rows, args.cache_root, provenance, load_seg_features=False)
    transform = _mask_transform(518)
    device = torch.device(args.device)
    mask_file_reads = sum(int(row["label"]) for row in source_rows)
    teacher_seconds: list[float] = []

    def tensors() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for index, row in enumerate(source_rows):
            mask = _source_mask(row, args.visa_root, transform)
            native = frozen[index]["native_logits"].unsqueeze(1).to(device, non_blocking=False)
            source_mask = mask.unsqueeze(0).to(device, non_blocking=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            teacher = build_source_teacher_region_target(native, source_mask)[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            teacher_seconds.append(time.perf_counter() - started)
            yield mask, teacher.detach().cpu()

    started = time.perf_counter()
    manifest = write_tier_b_shard(
        args.cache_root,
        args.held_class,
        source_rows,
        tensors(),
        provenance,
        source_mask_file_reads=mask_file_reads,
    )
    return {
        "tier": "B",
        "held_class": args.held_class,
        "samples": len(source_rows),
        "seconds": time.perf_counter() - started,
        "median_teacher_seconds": statistics.median(teacher_seconds) if teacher_seconds else None,
        "source_mask_file_reads": mask_file_reads,
        "held_mask_reads": manifest["held_mask_reads"],
        "manifest": str(args.cache_root / "tier_b" / args.held_class / "manifest.json"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    audit_protocol(json.loads(PROTOCOL_PATH.read_text()))
    verify_p26_parent(args.p26_checkpoint, args.clip_asset, ROOT / "configs/phase2b_canonical_v1.json")
    rows = read_visa_metadata(args.metadata)
    observed_classes = tuple(sorted({str(row["class_name"]) for row in rows}))
    if observed_classes != tuple(sorted(EXPECTED_VISA_CLASSES)):
        raise RuntimeError(f"VisA class inventory mismatch: {observed_classes}")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(args)
    projection = _check_space(args, len(rows))
    result = build_tier_a(args, rows, provenance) if args.tier == "a" else build_tier_b(args, rows, provenance)
    result.update({"provenance": provenance.as_dict(), "storage_projection": projection})
    summary_name = "tier_a_build.json" if args.tier == "a" else f"tier_b_{args.held_class}_build.json"
    atomic_write_json(args.cache_root / "summaries" / summary_name, result)
    return result


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
