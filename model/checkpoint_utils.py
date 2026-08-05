"""Versioned, backward-compatible checkpoint helpers for Phase 4."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch


PHASE4_CHECKPOINT_VERSION = 7


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
    expected_version = expected.get("progress_version")
    if expected_version == "P1-v6":
        if int(checkpoint.get("checkpoint_version", 0)) != 6:
            raise ValueError(
                "P1-v6 model requires checkpoint_version=6. P1-v5 checkpoints are not "
                "silently migrated because v6 adds deterministic router/query/factor anchor buffers."
            )
    if expected_version == "P1-v7-full":
        if int(checkpoint.get("checkpoint_version", 0)) != 7 or config.get("progress_version") != "P1-v7-full":
            raise ValueError("P1-v7-full requires explicit checkpoint_version=7 and P1-v7-full metadata")
    if expected_version in {"P1-v3", "P1-v4", "P1-v5", "P1-v5-fix"} and config.get("progress_version") != expected_version:
        raise ValueError(
            f"{expected_version} model requires a {expected_version} checkpoint with explicit "
            f"progress_version metadata; got {config.get('progress_version')!r}"
        )
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
    structural_gate_config: Mapping[str, Any] | None = None,
    structural_gate_state: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> Dict[str, Any]:
    if not getattr(model, "h6_enabled", False) or getattr(model, "h6", None) is None:
        raise ValueError("build_phase4_checkpoint requires an H6-enabled model")
    h6 = model.h6
    h6_config = dict(h6.config_dict())

    def _config_value(source_key: str, config_key: str):
        value = phase2b_config.get(source_key)
        return h6_config.get(config_key) if value is None else value

    h6_config.update({
        "router_teacher_enabled": float(loss_weights.get("router_teacher", 0.0)) > 0.0,
        "router_teacher_temperature": loss_weights.get("router_teacher_temperature"),
        "router_teacher_start_epoch": loss_weights.get("router_teacher_start_epoch"),
        "router_teacher_warmup_epochs": loss_weights.get("router_teacher_warmup_epochs"),
        "router_teacher_weight": loss_weights.get("router_teacher"),
        "teacher_confidence_gate": loss_weights.get(
            "teacher_confidence_gate", h6_config.get("teacher_confidence_gate")
        ),
        "teacher_entropy_threshold": loss_weights.get("teacher_entropy_threshold"),
        "teacher_prob_std_threshold": loss_weights.get("teacher_prob_std_threshold"),
        "router_teacher_mode": loss_weights.get("router_teacher_mode", h6_config.get("router_teacher_mode")),
        "router_teacher_state_aware": True,
        "router_teacher_detached": True,
        "balance_uses_dense": True,
        "balance_weight": loss_weights.get("balance"),
        "router_failure_patience": phase2b_config.get("h6_router_failure_patience"),
        "router_max_sparse_dead_factors": phase2b_config.get("h6_router_max_sparse_dead_factors"),
        "router_min_unique_topk_pairs": phase2b_config.get("h6_router_min_unique_topk_pairs"),
        "kl_zero_epochs": phase2b_config.get("h6_kl_zero_epochs"),
        "kl_warmup_epochs": phase2b_config.get("h6_kl_warmup_epochs"),
        "beta_kl_max": phase2b_config.get("beta_h6_vae_kl", loss_weights.get("vae_kl_current")),
        "kl_free_bits": phase2b_config.get("h6_kl_free_bits"),
        "kl_reduction_mode": "sum_latent_mean_batch",
        "concept_key_diversity_weight": phase2b_config.get("lambda_h6_concept_key_diversity", 0.0),
        "concept_key_cosine_margin": phase2b_config.get("h6_concept_key_cosine_margin"),
        "concept_key_diversity_start_epoch": phase2b_config.get("h6_concept_key_diversity_start_epoch"),
        "concept_key_diversity_warmup_epochs": phase2b_config.get("h6_concept_key_diversity_warmup_epochs"),
        "factor_grad_diagnostics_enabled": phase2b_config.get("h6_factor_grad_diagnostics", False),
        "late_factor_identity_enabled": phase2b_config.get(
            "h6_late_factor_identity_enabled", h6_config.get("late_factor_identity_enabled")
        ),
        "factor_id_scale": phase2b_config.get("h6_factor_id_scale", h6_config.get("factor_id_scale")),
        "factor_id_max_ratio": phase2b_config.get(
            "h6_factor_id_max_ratio", h6_config.get("factor_id_max_ratio")
        ),
        "factor_id_direction_method": h6_config.get("factor_id_direction_method"),
        "factor_id_projection_mode": "shared_linear_bankdim_to_textdim",
        "factor_id_shared_across_states": True,
        "router_query_mode": _config_value("h6_router_query_mode", "router_query_mode"),
        "router_query_global_weight": _config_value("h6_router_query_global_weight", "router_query_global_weight"),
        "router_local_bypass_scale": _config_value("h6_router_local_bypass_scale", "router_local_bypass_scale"),
        "router_local_bypass_max_ratio": _config_value(
            "h6_router_local_bypass_max_ratio", "router_local_bypass_max_ratio"
        ),
        "router_local_projection_method": "qr_semi_orthogonal_buffer",
        "router_local_projection_seed_offset": _config_value(
            "h6_router_local_projection_seed_offset", "router_local_projection_seed_offset"
        ),
        "router_key_anchor_enabled": _config_value("h6_router_key_anchor_enabled", "router_key_anchor_enabled"),
        "router_key_anchor_method": "qr_orthonormal_rows_buffer",
        "router_key_anchor_seed_offset": _config_value(
            "h6_router_key_anchor_seed_offset", "router_key_anchor_seed_offset"
        ),
        "router_key_adaptation_initial_ratio": _config_value(
            "h6_router_key_adaptation_initial_ratio", "router_key_adaptation_initial_ratio"
        ),
        "router_key_adaptation_max_ratio": _config_value(
            "h6_router_key_adaptation_max_ratio", "router_key_adaptation_max_ratio"
        ),
        "factor_context_anchor_enabled": _config_value(
            "h6_factor_context_anchor_enabled", "factor_context_anchor_enabled"
        ),
        "factor_context_anchor_method": "qr_orthonormal_rows_buffer",
        "factor_context_anchor_seed_offset": _config_value(
            "h6_factor_context_anchor_seed_offset", "factor_context_anchor_seed_offset"
        ),
        "factor_context_adaptation_initial_ratio": _config_value(
            "h6_factor_context_adaptation_initial_ratio", "factor_context_adaptation_initial_ratio"
        ),
        "factor_context_adaptation_max_ratio": _config_value(
            "h6_factor_context_adaptation_max_ratio", "factor_context_adaptation_max_ratio"
        ),
        "factor_identity_tangent_projection_enabled": _config_value(
            "h6_factor_identity_tangent_projection_enabled", "factor_identity_tangent_projection_enabled"
        ),
        "lambda_dynamic_mean_anchor": _config_value("lambda_h6_dynamic_mean_anchor", "lambda_dynamic_mean_anchor"),
        "dynamic_mean_anchor_min_cosine": _config_value(
            "h6_dynamic_mean_anchor_min_cosine", "dynamic_mean_anchor_min_cosine"
        ),
        "dynamic_mean_anchor_start_epoch": _config_value(
            "h6_dynamic_mean_anchor_start_epoch", "dynamic_mean_anchor_start_epoch"
        ),
        "dynamic_mean_anchor_warmup_epochs": _config_value(
            "h6_dynamic_mean_anchor_warmup_epochs", "dynamic_mean_anchor_warmup_epochs"
        ),
        "router_teacher_center_detached": True,
        "router_teacher_probability_detached": True,
        "teacher_confidence_gate_enabled": loss_weights.get("teacher_confidence_gate"),
        "teacher_probability_std_threshold": loss_weights.get("teacher_prob_std_threshold"),
        "teacher_gate_scope": "patch",
    })
    # Keep the full P1-v7 schedule alongside the architecture.  These are
    # intentionally explicit rather than inferred from a launcher at test time.
    for key in (
        "lambda_h6_expert", "lambda_h6_advantage", "lambda_h6_etf",
        "lambda_h6_expert_anchor", "lambda_h6_expert_radius",
        "h6_expert_start_epoch", "h6_expert_warmup_epochs",
        "h6_advantage_start_epoch", "h6_advantage_warmup_epochs", "h6_advantage_margin",
        "h6_etf_start_epoch", "h6_etf_warmup_epochs", "lambda_h6_balance_final",
        "h6_balance_decay_epochs",
    ):
        if key in phase2b_config:
            h6_config[key] = phase2b_config[key]
    payload: Dict[str, Any] = {
        "checkpoint_version": 7 if h6_config.get("progress_version") == "P1-v7-full" else 6,
        "epoch": int(epoch),
        "seed": int(seed),
        "phase4_progress": 1,
        "h6_enabled": True,
        "h6_config": h6_config,
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
    if structural_gate_config is not None:
        payload["structural_gate_config"] = dict(structural_gate_config)
        payload["h6_config"]["structural_gate_mode"] = structural_gate_config.get("structural_gate_mode", "abort")
    if structural_gate_state is not None:
        payload["structural_gate_state"] = dict(structural_gate_state)
    return payload
