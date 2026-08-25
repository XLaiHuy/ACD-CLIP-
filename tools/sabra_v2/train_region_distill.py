"""Train exactly one P27 LOCO fold on source rows only; never evaluate held GT."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tools.sabra.data import EXPECTED_VISA_CLASSES, VisaEvaluationDataset, read_visa_metadata
from tools.sabra_v2.audit_region_distill import PROTOCOL_PATH, audit_protocol
from tools.sabra_v2.correction_teacher import build_source_teacher_region_target
from tools.sabra_v2.data_protocol import loco_inventory
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.student_forward import assert_frozen_phase2b, forward_region_student
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
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=None, help="engineering cap; omit for the preregistered epoch schedule")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--engineering-smoke", action="store_true", help="label output as non-scientific and prohibit held prediction")
    return parser


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
    inventory = loco_inventory(read_visa_metadata(args.metadata), args.held_class)
    device = torch.device(args.device)
    phase2b, config = _load_frozen_phase2b(args.p26_checkpoint, args.clip_asset, device)
    adapter = RegionResidualAdapter().to(device)
    assert_frozen_phase2b(phase2b, adapter)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)
    source_loader = DataLoader(VisaEvaluationDataset(inventory.fit_rows, args.visa_root), batch_size=args.batch_size, shuffle=True, num_workers=0)
    from model.phase2b_runtime import forward_phase2b
    from utils import calculate_seg_loss

    steps = 0
    last_loss: float | None = None
    adapter.train()
    for _epoch in range(args.epochs):
        for batch in source_loader:
            frozen = forward_phase2b(phase2b, batch["image"], batch["class_name"], device, config, domain="Industrial", require_grad=False)
            source_mask = batch["mask"].to(device=device, dtype=torch.float32)
            teacher_region = build_source_teacher_region_target(frozen.native_logits, source_mask)
            student = forward_region_student(adapter, frozen.seg_features, frozen.native_logits)
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
    payload = {
        "schema_version": "P27_REGION_ADAPTER_CHECKPOINT_V1",
        "status": "ENGINEERING_SMOKE_ONLY" if args.engineering_smoke else "FOLD_TRAINING_COMPLETE",
        "held_class": args.held_class,
        "state_dict": adapter.state_dict(),
        "steps": steps,
        "p26_checkpoint_sha256": _sha256(args.p26_checkpoint),
        "clip_asset_sha256": _sha256(args.clip_asset),
    }
    torch.save(payload, checkpoint_path)
    return {"checkpoint": str(checkpoint_path), "held_class": args.held_class, "fit_records": len(inventory.fit_rows), "held_records_not_read": len(inventory.held_rows), "steps": steps, "last_loss": last_loss, "status": payload["status"]}


def main() -> None:
    print(json.dumps(run(make_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
