"""Audited Phase2B epoch schedules and mathematically explicit batch windows."""
from __future__ import annotations

from typing import Any, Mapping


def get_dfg_beta_for_epoch(
    epoch_one_based: int,
    dfg_beta_schedule: str,
    dfg_beta_target: float,
    dfg_beta: float,
) -> float:
    """Historical Phase2B DFG schedule (epoch numbers are one-based)."""
    epoch = int(epoch_one_based)
    if epoch < 1:
        raise ValueError("epoch must be one-based and positive")
    schedule = str(dfg_beta_schedule)
    if schedule == "fixed":
        return float(dfg_beta)
    if schedule == "warmup010":
        if epoch <= 3:
            return 0.0
        if epoch <= 6:
            return min(0.05, float(dfg_beta_target))
        return float(dfg_beta_target)
    raise ValueError(f"unknown dfg_beta_schedule: {schedule}")


def get_hybrid_alpha_for_epoch(
    epoch_one_based: int,
    hybrid_alpha_max: float,
    soft_prompt_freeze_epochs: int,
) -> float:
    """Historical Phase2B hybrid schedule: freeze, then 25/50/100% ramp."""
    epoch = int(epoch_one_based)
    freeze_epochs = int(soft_prompt_freeze_epochs)
    if epoch < 1 or freeze_epochs < 0:
        raise ValueError("epoch must be positive and freeze epochs non-negative")
    if epoch <= freeze_epochs:
        return 0.0
    warm_epoch = epoch - freeze_epochs
    if warm_epoch == 1:
        return 0.25 * float(hybrid_alpha_max)
    if warm_epoch == 2:
        return 0.50 * float(hybrid_alpha_max)
    return float(hybrid_alpha_max)


def grad_accum_window_size(batch_index_one_based: int, total_batches: int, accum_steps: int) -> int:
    """Return the actual microbatch count in the current accumulation window."""
    index = int(batch_index_one_based)
    total = int(total_batches)
    accum = int(accum_steps)
    if index < 1 or total < index or accum < 1:
        raise ValueError("invalid accumulation window arguments")
    start = ((index - 1) // accum) * accum + 1
    return min(accum, total - start + 1)


def apply_soft_prompt_lr_policy(optimizer: Any, frozen: bool) -> None:
    """Keep the soft-prompt optimizer group present while freezing its updates."""
    for group in optimizer.param_groups:
        if group.get("name") == "soft_prompt":
            constant_lr = float(group.get("constant_lr", group.get("lr", 0.0)))
            group["constant_lr"] = constant_lr
            group["lr"] = 0.0 if frozen else constant_lr


def scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Extract fields that must match across a resume."""
    execution_only = {
        "num_workers", "pin_memory", "persistent_workers", "prefetch_factor",
        "micro_batch_size", "batch_size", "grad_accum_steps", "run_root",
        "device", "output_dir", "clip_asset", "visa_root", "resume",
    }
    return {str(key): value for key, value in config.items() if key not in execution_only}
