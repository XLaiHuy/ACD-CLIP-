"""Versioned, backward-compatible checkpoint helpers for Phase 4."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch


PHASE4_CHECKPOINT_VERSION = 1


def is_phase4_checkpoint(checkpoint: Mapping[str, Any]) -> bool:
    return bool(checkpoint.get("h6_enabled", False) or checkpoint.get("phase4_progress", 0))


def h6_config_from_checkpoint(checkpoint: Mapping[str, Any]) -> Dict[str, Any] | None:
    if not is_phase4_checkpoint(checkpoint):
        return None
    config = checkpoint.get("h6_config")
    if not isinstance(config, dict):
        raise ValueError("Phase 4 checkpoint is missing its h6_config metadata")
    return dict(config)


def validate_h6_configuration(model, checkpoint: Mapping[str, Any]) -> None:
    config = h6_config_from_checkpoint(checkpoint)
    if config is None:
        return
    if not getattr(model, "h6_enabled", False) or getattr(model, "h6", None) is None:
        raise ValueError("checkpoint contains H6 weights but the CLI constructed an H6-disabled model")
    expected = model.h6.config_dict()
    mismatches = {
        key: (config.get(key), expected.get(key))
        for key in expected
        if key in config and config.get(key) != expected.get(key)
    }
    if mismatches:
        formatted = ", ".join(f"{key}: checkpoint={old!r}, CLI={new!r}" for key, (old, new) in mismatches.items())
        raise ValueError(f"H6 checkpoint configuration conflicts with CLI: {formatted}")


def load_adapter_checkpoint(model, checkpoint: Mapping[str, Any]) -> bool:
    """Load either an old Phase2B payload or a new Phase4 payload.

    Returns whether H6 was restored.  Old checkpoints remain valid because the
    Phase2B adapter/text fields keep their original names.
    """
    model.image_adapter.load_state_dict(checkpoint["image_adapter"])
    model.text_adapter.load_state_dict(checkpoint["text_adapter"])
    if "soft_prompt" in checkpoint:
        model.soft_prompt.load_state_dict(checkpoint["soft_prompt"])
    if not is_phase4_checkpoint(checkpoint):
        return False
    validate_h6_configuration(model, checkpoint)
    if "h6_state_dict" not in checkpoint:
        raise ValueError("Phase 4 checkpoint is missing h6_state_dict")
    model.h6.load_state_dict(checkpoint["h6_state_dict"], strict=True)
    model.h6.set_epoch(int(checkpoint.get("router_warmup_epoch", checkpoint.get("epoch", 1))))
    return True


def build_phase4_checkpoint(
    model,
    *,
    epoch: int,
    seed: int,
    precision: str,
    phase2b_config: Mapping[str, Any],
    loss_weights: Mapping[str, float],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> Dict[str, Any]:
    if not getattr(model, "h6_enabled", False) or getattr(model, "h6", None) is None:
        raise ValueError("build_phase4_checkpoint requires an H6-enabled model")
    h6 = model.h6
    payload: Dict[str, Any] = {
        "checkpoint_version": PHASE4_CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "seed": int(seed),
        "phase4_progress": 1,
        "h6_enabled": True,
        "h6_config": h6.config_dict(),
        "h6_state_dict": h6.state_dict(),
        "image_adapter": model.image_adapter.state_dict(),
        "text_adapter": model.text_adapter.state_dict(),
        "soft_prompt": model.soft_prompt.state_dict(),
        "prompt_mode": "h6_dynamic",
        "use_soft_prompt": False,
        "use_hybrid_soft_prompt": True,
        "soft_prompt_ctx_len": model.soft_prompt_ctx_len,
        "soft_prompt_init": model.soft_prompt_init,
        "soft_prompt_init_phrase": model.soft_prompt_init_phrase,
        "hybrid_alpha_current": float(getattr(model, "hybrid_alpha_current", 0.0)),
        "hybrid_alpha_max": float(getattr(model, "hybrid_alpha_max", 0.2)),
        "soft_prompt_freeze_epochs": int(getattr(model, "soft_prompt_freeze_epochs", 3)),
        "precision": str(precision),
        "loss_weights": dict(loss_weights),
        "gate_values": {
            "gamma_state": float(h6.semantic_core.gamma_state().detach().item()),
            "gamma_class": float(h6.semantic_core.gamma_class().detach().item()),
            "rho": h6.rho_values().detach().float().cpu().tolist(),
        },
        # Epoch is the authoritative router warm-up state; the runtime counter
        # is intentionally not part of a module state_dict.
        "router_warmup_epoch": int(epoch),
        "n_groups": model.n_groups,
        "dfg_mode": model.dfg_mode,
        "dfg_attn_dim": model.dfg_attn_dim,
        "dfg_attn_tau": model.dfg_attn_tau,
        "use_ss2d_dfg": model.use_ss2d_dfg,
        "dfg_gamma_max": model.dfg_gamma_max,
        "dfg_ss2d_fusion": model.dfg_ss2d_fusion,
        "dfg_beta": model.dfg_beta,
        "dfg_beta_schedule": model.dfg_beta_schedule,
        "dfg_beta_target": model.dfg_beta_target,
        "dfg_weight_residual_fp32": model.dfg_weight_residual_fp32,
        "phase2b_config": dict(phase2b_config),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    return payload
