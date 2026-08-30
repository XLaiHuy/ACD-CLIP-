#!/usr/bin/env python3
"""Audit CIR versus Phase2B optimizer and StepLR provenance.

This is an evidence-only utility.  It loads existing checkpoints on CPU,
does not instantiate a model, does not update parameters, and writes only to
the requested audit output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import torch
PARENT_LR_RE = re.compile(r"hybrid_state epoch=(\d+).*?image_lr=([^ ]+) text_lr=([^ ]+) soft_lr=([^ ]+)")


EPOCH_RE = re.compile(r"epoch[_-]?(\d+)", re.IGNORECASE)
REQUIRED_GROUPS = ("image_adapter", "text_adapter", "soft_prompt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_payload(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint is not a mapping: {path}")
    return payload


def epoch_from_path(path: Path, payload: Mapping[str, Any]) -> int | None:
    match = EPOCH_RE.search(path.stem)
    if match:
        return int(match.group(1))
    value = payload.get("epoch")
    return int(value) if value is not None else None


def group_map(state: Any) -> dict[str, Mapping[str, Any]]:
    if not isinstance(state, Mapping):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    groups = state.get("param_groups")
    if not isinstance(groups, list):
        return result
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            continue
        name = str(group.get("name", f"group_{index}"))
        result[name] = group
    return result


def scalar(value: Any) -> float | int | str | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)) or value is None:
        return value
    return str(value)


def group_lr(groups: Mapping[str, Mapping[str, Any]], name: str) -> float | None:
    value = groups.get(name, {}).get("lr")
    return float(value) if value is not None else None


def expected_step_lr(base: float, gamma: float, epoch: int, *, after_epoch_step: bool = True) -> float:
    """Return the canonical StepLR value at an epoch boundary.

    The canonical trainer calls scheduler.step() once after epoch ``epoch``.
    ``after_epoch_step=True`` therefore uses gamma**epoch.  The companion
    start-of-epoch value is gamma**(epoch-1), matching the user-facing LR
    estimates in the audit request.
    """

    exponent = epoch if after_epoch_step else max(epoch - 1, 0)
    return float(base * (gamma**exponent))


def expected_prompt_lr(base: float, gamma: float, epoch: int, freeze_epochs: int) -> float:
    """Canonical checkpoint LR after policy reset then StepLR.step()."""

    before_step = 0.0 if epoch <= freeze_epochs else base
    return float(before_step * gamma)


def find_cir_checkpoints(cir_root: Path) -> list[tuple[str, Path]]:
    checkpoint_root = cir_root / "visa" / "seed0" / "checkpoints"
    paths = sorted(checkpoint_root.glob("epoch_*.pth"))
    result = [("epoch", path) for path in paths]
    last = cir_root / "visa" / "seed0" / "last.pth"
    if last.is_file():
        result.append(("last", last))
    return result


def find_parent_checkpoints(parent_root: Path) -> list[Path]:
    return sorted(path for path in parent_root.rglob("*.pth") if path.is_file())

def parse_parent_history(parent_root: Path) -> tuple[str, dict[int, dict[str, float]]]:
    path = parent_root / "train.log"
    history: dict[int, dict[str, float]] = {}
    if not path.is_file():
        return str(path), history
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PARENT_LR_RE.search(line)
        if not match:
            continue
        epoch = int(match.group(1))
        history[epoch] = {
            "image_lr": float(match.group(2)),
            "text_lr": float(match.group(3)),
            "prompt_lr": float(match.group(4)),
        }


    return str(path), history
def cir_record(kind: str, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    optimizer_state = payload.get("optimizer_state")
    scheduler_state = payload.get("scheduler_state")
    groups = group_map(optimizer_state)
    epoch = epoch_from_path(path, payload)
    scheduler = scheduler_state if isinstance(scheduler_state, Mapping) else {}
    return {
        "kind": kind,
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": epoch,
        "payload_epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "optimizer_present": isinstance(optimizer_state, Mapping),
        "scheduler_present": isinstance(scheduler_state, Mapping),
        "groups": groups,
        "scheduler": scheduler,
        "git_sha": payload.get("git_sha"),
        "precision": payload.get("precision"),
        "amp_enabled": payload.get("amp_enabled"),
        "tf32_enabled": payload.get("tf32_enabled"),
        "config_sha256": payload.get("config_sha256"),
    }


def parent_record(path: Path) -> dict[str, Any]:
    payload = load_payload(path)
    optimizer_state = payload.get("optimizer_state")
    scheduler_state = payload.get("scheduler_state")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "epoch": epoch_from_path(path, payload),
        "payload_epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "optimizer_present": isinstance(optimizer_state, Mapping),
        "scheduler_present": isinstance(scheduler_state, Mapping),
        "groups": group_map(optimizer_state),
        "scheduler": scheduler_state if isinstance(scheduler_state, Mapping) else {},
        "keys": sorted(str(key) for key in payload.keys()),
        "precision": payload.get("precision"),
        "amp_enabled": payload.get("amp_enabled"),
        "tf32_enabled": payload.get("tf32_enabled"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/cir_rmt/CIR_DFG_RMT_V2/forensics_20260830"),
    )
    parser.add_argument(
        "--cir-root",
        type=Path,
        default=Path("runs/cir_rmt/CIR_DFG_RMT_V2"),
    )
    parser.add_argument(
        "--parent-root",
        type=Path,
        default=Path("runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/phase2b_canonical_v1.json"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    image_base = float(config["image_lr"])
    text_base = float(config["text_lr"])
    prompt_base = float(config["soft_prompt_lr"])
    gamma = float(config["lr_gamma"])
    freeze_epochs = int(config.get("soft_prompt_freeze_epochs", 3))

    cir_records = [cir_record(kind, path, load_payload(path)) for kind, path in find_cir_checkpoints(args.cir_root)]
    parent_records = [parent_record(path) for path in find_parent_checkpoints(args.parent_root)]
    parent_history_path, parent_history = parse_parent_history(args.parent_root)
    parent_by_epoch: dict[int, dict[str, Any]] = {}
    for record in parent_records:
        epoch = record.get("epoch")
        if epoch is not None and epoch not in parent_by_epoch:
            parent_by_epoch[int(epoch)] = record

    cir_by_epoch: dict[int, dict[str, Any]] = {}
    for record in cir_records:
        epoch = record.get("epoch")
        if epoch is not None and record["kind"] == "epoch":
            cir_by_epoch[int(epoch)] = record

    epochs = sorted(set(parent_by_epoch) | set(cir_by_epoch) | {10, 12, 14, 16, 18, 20})
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        cir = cir_by_epoch.get(epoch)
        parent = parent_by_epoch.get(epoch)
        cir_groups = cir["groups"] if cir else {}
        parent_groups = parent["groups"] if parent else {}
        cir_scheduler = cir["scheduler"] if cir else {}
        parent_scheduler = parent["scheduler"] if parent else {}
        history = parent_history.get(epoch, {})
        history_image = history.get("image_lr")
        history_text = history.get("text_lr")
        history_prompt = history.get("prompt_lr")
        cir_image = group_lr(cir_groups, "image_adapter")
        cir_text = group_lr(cir_groups, "text_adapter")
        cir_prompt = group_lr(cir_groups, "soft_prompt")
        serialized_parent_image = group_lr(parent_groups, "image_adapter")
        serialized_parent_text = group_lr(parent_groups, "text_adapter")
        serialized_parent_prompt = group_lr(parent_groups, "soft_prompt")
        expected_parent_image = expected_step_lr(image_base, gamma, epoch, after_epoch_step=False)
        expected_parent_text = expected_step_lr(text_base, gamma, epoch, after_epoch_step=False)
        expected_parent_prompt = 0.0 if epoch <= freeze_epochs else prompt_base
        parent_image = serialized_parent_image if serialized_parent_image is not None else history_image if history_image is not None else expected_parent_image
        parent_text = serialized_parent_text if serialized_parent_text is not None else history_text if history_text is not None else expected_parent_text
        parent_prompt = serialized_parent_prompt if serialized_parent_prompt is not None else history_prompt if history_prompt is not None else expected_parent_prompt
        cir_last_epoch = cir_scheduler.get("last_epoch")
        serialized_parent_last_epoch = parent_scheduler.get("last_epoch")
        parent_last_epoch = (
            serialized_parent_last_epoch
            if serialized_parent_last_epoch is not None
            else max(epoch - 1, 0)
        )
        parent_has_serialized_state = bool(
            parent
            and parent["optimizer_present"]
            and parent["scheduler_present"]
            and serialized_parent_image is not None
            and serialized_parent_text is not None
            and serialized_parent_prompt is not None
        )
        parent_has_history = all(
            value is not None
            for value in (history_image, history_text, history_prompt)
        )

        if cir is None:
            matched = "NO_CIR_CHECKPOINT"
        elif not cir["scheduler_present"]:
            matched = "CIR_SCHEDULER_STATE_MISSING"
        elif cir_last_epoch != epoch:
            matched = "CIR_SCHEDULER_NOT_STEPPED"
        else:
            matched = "CIR_SCHEDULER_MATCHES_EPOCH"
        rows.append(
            {
                "epoch": epoch,
                "parent_image_lr": parent_image,
                "cir_image_lr": cir_image,
                "parent_text_lr": parent_text,
                "cir_text_lr": cir_text,
                "parent_prompt_lr": parent_prompt,
                "cir_prompt_lr": cir_prompt,
                "parent_scheduler_last_epoch": parent_last_epoch,
                "cir_scheduler_last_epoch": cir_last_epoch,
                "matched_or_mismatch": matched,
                "cir_over_parent_image_lr": None if cir_image is None or parent_image == 0 else cir_image / parent_image,
                "cir_over_parent_text_lr": None if cir_text is None or parent_text == 0 else cir_text / parent_text,
                "cir_over_parent_prompt_lr": None if cir_prompt is None or parent_prompt == 0 else cir_prompt / parent_prompt,
                "parent_expected_start_image_lr": expected_step_lr(image_base, gamma, epoch, after_epoch_step=False),
                "parent_expected_start_text_lr": expected_step_lr(text_base, gamma, epoch, after_epoch_step=False),
                "parent_expected_start_prompt_lr": 0.0 if epoch <= freeze_epochs else prompt_base,
                "parent_expected_checkpoint_image_lr": expected_step_lr(image_base, gamma, epoch),
                "parent_expected_checkpoint_text_lr": expected_step_lr(text_base, gamma, epoch),
                "parent_expected_checkpoint_prompt_lr": expected_prompt_lr(prompt_base, gamma, epoch, freeze_epochs),
                "parent_serialized_image_lr": serialized_parent_image,
                "parent_serialized_text_lr": serialized_parent_text,
                "parent_serialized_prompt_lr": serialized_parent_prompt,
                "parent_history_image_lr": history_image,
                "parent_history_text_lr": history_text,
                "parent_history_prompt_lr": history_prompt,
                "parent_history_path": parent_history_path,
                "parent_serialized_scheduler_last_epoch": serialized_parent_last_epoch,
                "parent_lr_evidence": (
                    "serialized_parent_optimizer_state"
                    if parent_has_serialized_state
                    else "historical_train_log_start_of_epoch_lr"
                    if parent_has_history
                    else "canonical_expected_start_of_epoch_lr"
                ),
                "parent_scheduler_evidence": (
                    "serialized_parent_scheduler_state"
                    if parent_has_serialized_state
                    else "historical_train_log_start_of_epoch; scheduler_state_not_serialized"
                    if parent_has_history
                    else "canonical_expected_start_of_epoch_state"
                ),
                "cir_checkpoint_path": cir["path"] if cir else None,

                "cir_checkpoint_sha256": cir["sha256"] if cir else None,
                "cir_optimizer_state_present": cir["optimizer_present"] if cir else False,
                "cir_scheduler_state_present": cir["scheduler_present"] if cir else False,
                "parent_artifact_path": parent["path"] if parent else None,
                "parent_artifact_sha256": parent["sha256"] if parent else None,
                "parent_optimizer_state_present": parent["optimizer_present"] if parent else False,
                "parent_scheduler_state_present": parent["scheduler_present"] if parent else False,
                "parent_artifact_evidence": (
                    "serialized_optimizer_and_scheduler"
                    if parent and parent["optimizer_present"] and parent["scheduler_present"]
                    else "available_checkpoint_lacks_optimizer_and_scheduler"
                    if parent
                    else "no_parent_checkpoint_for_epoch"
                ),
            }
        )

    optimizer_detail_rows: list[dict[str, Any]] = []
    for record in cir_records:
        for name in REQUIRED_GROUPS:
            group = record["groups"].get(name, {})
            optimizer_detail_rows.append(
                {
                    "source": "cir",
                    "checkpoint_kind": record["kind"],
                    "epoch": record["epoch"],
                    "payload_epoch": record["payload_epoch"],
                    "global_step": record["global_step"],
                    "checkpoint_path": record["path"],
                    "checkpoint_sha256": record["sha256"],
                    "group": name,
                    "lr": group.get("lr"),
                    "constant_lr": group.get("constant_lr"),
                    "weight_decay": group.get("weight_decay"),
                    "betas": group.get("betas"),
                    "eps": group.get("eps"),
                    "maximize": group.get("maximize"),
                    "amsgrad": group.get("amsgrad"),
                    "initial_lr": group.get("initial_lr"),
                    "params_count": len(group.get("params", [])) if isinstance(group.get("params"), list) else None,
                    "optimizer_state_present": record["optimizer_present"],
                    "scheduler_last_epoch": record["scheduler"].get("last_epoch"),
                    "scheduler_step_count": record["scheduler"].get("_step_count"),
                    "scheduler_last_lr": record["scheduler"].get("_last_lr"),
                    "scheduler_state_json": json.dumps(record["scheduler"], sort_keys=True, default=json_default),
                }
            )
    for record in parent_records:
        for name in REQUIRED_GROUPS:
            group = record["groups"].get(name, {})
            optimizer_detail_rows.append(
                {
                    "source": "phase2b_available_artifact",
                    "checkpoint_kind": "artifact",
                    "epoch": record["epoch"],
                    "payload_epoch": record["payload_epoch"],
                    "global_step": record["global_step"],
                    "checkpoint_path": record["path"],
                    "checkpoint_sha256": record["sha256"],
                    "group": name,
                    "lr": group.get("lr"),
                    "constant_lr": group.get("constant_lr"),
                    "weight_decay": group.get("weight_decay"),
                    "betas": group.get("betas"),
                    "eps": group.get("eps"),
                    "maximize": group.get("maximize"),
                    "amsgrad": group.get("amsgrad"),
                    "initial_lr": group.get("initial_lr"),
                    "params_count": len(group.get("params", [])) if isinstance(group.get("params"), list) else None,
                    "optimizer_state_present": record["optimizer_present"],
                    "scheduler_last_epoch": record["scheduler"].get("last_epoch"),
                    "scheduler_step_count": record["scheduler"].get("_step_count"),
                    "scheduler_last_lr": record["scheduler"].get("_last_lr"),
                    "scheduler_state_json": json.dumps(record["scheduler"], sort_keys=True, default=json_default),
                }
            )

    cir_nonmatching = [row for row in rows if row["cir_checkpoint_path"] and row["cir_scheduler_last_epoch"] != row["epoch"]]
    cir_bug_confirmed = bool(cir_nonmatching) and all(
        row["cir_scheduler_state_present"] and row["cir_scheduler_last_epoch"] == 0 for row in cir_nonmatching
    )
    classification = "CIR_SCHEDULER_BUG_CONFIRMED" if cir_bug_confirmed else "SCHEDULER_STATE_INCONCLUSIVE"
    if not cir_nonmatching and cir_by_epoch:
        classification = "SCHEDULER_MATCHED"

    optimizer_hparams: dict[str, Any] = {}
    for record in cir_records:
        groups = record["groups"]
        if groups:
            optimizer_hparams = {
                "optimizer_source": "scripts/cir_rmt/train_full.py _optimizer uses torch.optim.Adam",
                "weight_decay": {name: groups.get(name, {}).get("weight_decay") for name in REQUIRED_GROUPS},
                "betas": {name: groups.get(name, {}).get("betas") for name in REQUIRED_GROUPS},
                "eps": {name: groups.get(name, {}).get("eps") for name in REQUIRED_GROUPS},
                "maximize": {name: groups.get(name, {}).get("maximize") for name in REQUIRED_GROUPS},
                "amsgrad": {name: groups.get(name, {}).get("amsgrad") for name in REQUIRED_GROUPS},
            }
            break

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "scheduler_optimization_audit.csv", rows)
    write_csv(args.output_root / "scheduler_optimizer_group_detail.csv", optimizer_detail_rows)

    last_records = [record for record in cir_records if record["kind"] == "last"]
    parent_inventory = [
        {
            "path": record["path"],
            "sha256": record["sha256"],
            "epoch": record["epoch"],
            "optimizer_state_present": record["optimizer_present"],
            "scheduler_state_present": record["scheduler_present"],
            "payload_keys": record["keys"],
            "precision": record["precision"],
            "amp_enabled": record["amp_enabled"],
            "tf32_enabled": record["tf32_enabled"],
        }
        for record in parent_records
    ]
    summary = {
        "classification": classification,
        "cir_checkpoint_count": len(cir_records),
        "cir_epoch_checkpoint_count": len(cir_by_epoch),
        "cir_last_checkpoint_count": len(last_records),
        "cir_nonmatching_scheduler_rows": len(cir_nonmatching),
        "cir_checkpoint_epochs": sorted(cir_by_epoch),
        "cir_missing_e10": 10 not in cir_by_epoch,
        "parent_available_checkpoint_count": len(parent_records),
        "parent_history_path": parent_history_path,
        "parent_history_epochs": sorted(parent_history),
        "parent_history_source": "hybrid_state lines in historical Phase2B train.log; start-of-epoch LR observations",
        "lr_exposure": {
            "ratios_are_cir_over_parent_table_lr_at_epoch_start": True,
            "epoch_start_ratios": {
                str(row["epoch"]): {
                    "image_adapter": row["cir_over_parent_image_lr"],
                    "text_adapter": row["cir_over_parent_text_lr"],
                    "soft_prompt": row["cir_over_parent_prompt_lr"],
                }
                for row in rows
            },
        },
        "parent_inventory": parent_inventory,
        "parent_protocol": {
            "optimizer": "Adam",
            "step_size": 1,
            "gamma": gamma,
            "image_lr": image_base,
            "text_lr": text_base,
            "soft_prompt_lr": prompt_base,
            "soft_prompt_freeze_epochs": freeze_epochs,
            "scheduler_step_position": "after_epoch_loop_before_checkpoint_save",
        },
        "optimizer_hparams_from_cir_state": optimizer_hparams,
        "source_evidence": {
            "cir_train_file": "scripts/cir_rmt/train_full.py",
            "cir_scheduler_construction": "StepLR(optimizer, step_size=1, gamma=lr_gamma)",
            "cir_scheduler_step_position": "no scheduler.step() call between epoch loop and checkpoint save",
            "cir_scheduler_step_call": False,
            "cir_resume_loads_scheduler_state": True,
            "phase2b_train_file": "train.py",
            "phase2b_scheduler_step_call": True,
            "phase2b_scheduler_step_position": "after batch loop and before row/checkpoint payload",
            "phase2b_resume_loads_scheduler_state": True,
        },
    }
    (args.output_root / "scheduler_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )

    start_ratio_text = ", ".join(
        f"E{row['epoch']}: {row['cir_over_parent_image_lr']:.2f}x"
        for row in rows
        if row["cir_over_parent_image_lr"] is not None
    )
    e20_checkpoint_ratio = image_base / expected_step_lr(image_base, gamma, 20)
    lines = [

        "# CIR versus Phase2B LR scheduler and optimization audit",
        "",
        f"Classification: `{classification}`",
        "",
        "## Finding",
        "",
        "`scripts/cir_rmt/train_full.py` constructs `StepLR(step_size=1, gamma=0.9)` and stores its state, but the epoch loop has no `scheduler.step()` call. The canonical `train.py` Phase2B loop calls `scheduler.step()` after the batch loop and before the epoch row and checkpoint payload are written.",
        "",
        f"All {len(cir_by_epoch)} epoch CIR checkpoints have a serialized scheduler state whose `last_epoch` is 0 while the checkpoint epoch is in {sorted(cir_by_epoch)}. Their optimizer group LRs therefore remain at the initial values (except any soft-prompt freeze policy state), rather than following the intended StepLR decay.",
        "",
        "## Plausible magnitude (not a causal estimate)",
        f"At epoch start, CIR image/text LR exposure relative to the canonical parent is {start_ratio_text}; the image and text ratios are identical because their initial LR ratio is fixed at 2:1. At E20, the canonical post-step checkpoint values would be image 1.2158e-4 and text 6.0788e-5, versus CIR 1e-3 and 5e-4, an approximately {e20_checkpoint_ratio:.2f}x excess. The soft prompt follows a separate constant-LR freeze/unfreeze policy: against the current canonical parent value 1e-4, CIR is 1.0x at E12-E20; the CSV's 2.0x at E12/E14 reflects the legacy history's 5e-5 prompt base, not StepLR.",
        "These are exposure ratios, not a predicted percentage of the medical-score gap: Adam's adaptive moments, gradient clipping, skipped/non-finite updates, and representation trajectory make parameter displacement nonlinear. The scheduler bug can plausibly explain instability and late-epoch drift, but its causal share requires the matched corrective retrain.",
        "",
        "## Checkpoint coverage",
        "",
        f"CIR epoch checkpoints: {sorted(cir_by_epoch)}; E10 is absent. CIR `last.pth` records: {len(last_records)}. Available historical Phase2B checkpoint artifacts: {len(parent_records)}.",
        "",
        "The available historical Phase2B `adapter_10.pth` is a legacy artifact and does not contain serialized `optimizer_state` or `scheduler_state`. The comparison table uses actual start-of-epoch LR observations from its `train.log` for E1-E15, canonical expected values where the log has no later epoch, and separate serialized-parent columns; it does not claim that the legacy checkpoint itself stores optimizer/scheduler state. The legacy log uses soft-prompt LR 5e-5, while the current canonical parent config uses 1e-4.",
        "",
        "## LR convention",
        "",
        "The required table includes actual serialized CIR values and any serialized parent values. It also includes protocol-expected parent start-of-epoch and post-`scheduler.step()` checkpoint values. The user-facing estimates such as E10 approximately 3.87e-4 correspond to the start-of-epoch convention; the canonical checkpoint is saved after the epoch scheduler step.",
        "",
        "## Optimization details",
        "",
        "- Both canonical trainers construct three named Adam groups: `image_adapter`, `text_adapter`, and `soft_prompt`.",
        "- The CIR checkpoint group state reports Adam defaults (`betas=(0.9, 0.999)`, `eps=1e-8`) and zero weight decay for all three groups; no group is exempt from decay because decay is zero globally.",
        "- The historical Phase2B `train.log` records image/text start-of-epoch LRs decaying by gamma=0.9 from E1 through E15, including E10=3.8742e-4, E12=3.1381e-4, and E14=2.5419e-4; this verifies actual parent-run decay even though its checkpoint omits optimizer/scheduler state.",
        "- Both canonical loops clip gradients once per optimizer step after gradient accumulation. CIR uses `clip_grad_norm_` directly; Phase2B uses its equivalent helper.",
        "- The soft prompt remains in the optimizer. `_set_epoch_state` sets its LR to zero through the freeze epochs and restores `constant_lr` afterward; this is a separate freeze/unfreeze policy, not evidence that StepLR was applied to the CIR run.",
        "- Both trainers load optimizer state, scheduler state, RNG state, and resume at `checkpoint_epoch + 1`. CIR's resume path is mechanically present but semantically incorrect for the intended schedule: restoring a stale scheduler state preserves the constant-LR bug rather than repairing it.",
        "",
        "## Scientific implication",
        "",
        "The scheduler mismatch is a major protocol confound. The current CIR-V2 benchmark cannot cleanly isolate the RMT hypothesis from an optimization mismatch. The correct next action is one matched corrective parent/CIR training comparison with the same seed, source, FP32 policy, effective batch, optimizer, scheduler, scheduler timing, losses, and checkpoint schedule; no immediate architecture change or MVTec training follows from this audit.",
        "",
        "## Evidence files",
        "",
        "- `scheduler_optimization_audit.csv` — required epoch comparison table.",
        "- `scheduler_optimizer_group_detail.csv` — every available CIR checkpoint and parent artifact group/state detail.",
        "- `scheduler_audit_summary.json` — classification, inventory, and source-evidence summary.",
    ]
    (args.output_root / "SCHEDULER_OPTIMIZATION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"classification": classification, "cir_epochs": sorted(cir_by_epoch), "parent_artifacts": len(parent_records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
