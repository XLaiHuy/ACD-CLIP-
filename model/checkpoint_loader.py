"""Checkpoint-aware model loading for post-training evaluation.

This module has one strict rule: an explicitly supplied checkpoint is the
only checkpoint that may be loaded. Historical evaluators retain their old
defaults outside this helper; lab post-training commands use this helper
with adapter_20.pth.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .adapter import ACDCLIP
from .checkpoint_utils import load_adapter_checkpoint, validate_h6_configuration
from .clip import create_model


_REQUIRED_FINAL_KEYS = (
    "epoch", "global_step", "checkpoint_version", "image_adapter",
    "text_adapter", "soft_prompt", "h6_state_dict", "optimizer_state",
    "scheduler_state", "amp_scaler_state", "python_random_state",
    "numpy_random_state", "torch_cpu_rng_state", "torch_cuda_rng_state_all",
    "dataloader_generator_state", "phase2b_config", "h6_config", "git_sha",
    "package_config_sha256", "dataset_role_contract_sha256",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(
    path: str | Path,
    *,
    expected_epoch: int | None = None,
    require_final_contract: bool = True,
) -> dict[str, Any]:
    """Read and validate identity metadata without constructing a model."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"requested checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {checkpoint_path}")
    if require_final_contract:
        missing = [key for key in _REQUIRED_FINAL_KEYS if key not in checkpoint]
        if missing:
            raise ValueError(f"checkpoint contract missing fields: {missing}")
    epoch = int(checkpoint.get("epoch", -1))
    if expected_epoch is not None and epoch != int(expected_epoch):
        raise ValueError(f"checkpoint epoch {epoch} does not equal expected epoch {expected_epoch}")
    h6_config = checkpoint.get("h6_config")
    phase2b_config = checkpoint.get("phase2b_config")
    identity = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "epoch": epoch,
        "global_step": int(checkpoint.get("global_step", -1)),
        "source_git_sha": checkpoint.get("git_sha") or (phase2b_config or {}).get("git_sha"),
        "checkpoint_version": checkpoint.get("checkpoint_version"),
        "h6_progress_version": (h6_config or {}).get("progress_version"),
        "h6_progress": checkpoint.get("phase4_progress"),
        "package_config_sha256": checkpoint.get("package_config_sha256"),
        "dataset_role_contract_sha256": checkpoint.get("dataset_role_contract_sha256"),
    }
    return identity


def _get(source: Mapping[str, Any], key: str, default: Any = None) -> Any:
    value = source.get(key, default)
    return default if value is None else value


def _construct_kwargs(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    base = checkpoint.get("phase2b_config") or {}
    h6 = checkpoint.get("h6_config") or {}
    if not isinstance(base, Mapping) or not isinstance(h6, Mapping):
        raise ValueError("checkpoint phase2b_config and h6_config must be mappings")
    return {
        "text_adapt_weight": _get(base, "text_adapt_weight", 0.15),
        "image_adapt_weight": _get(base, "image_adapt_weight", 0.15),
        "n_groups": int(_get(base, "n_groups", checkpoint.get("n_groups", 3))),
        "lora_rank": int(_get(base, "lora_rank", 16)),
        "lora_alpha": float(_get(base, "lora_alpha", 2.0)),
        "conv_lora_rank": int(_get(base, "conv_lora_rank", 8)),
        "conv_lora_alpha": float(_get(base, "conv_lora_alpha", 2.0)),
        "conv_kernel_size_list": tuple(_get(base, "conv_kernel_size_list", (3, 5))),
        "dfg_mode": str(_get(base, "dfg_mode", checkpoint.get("dfg_mode", "mlp"))),
        "dfg_attn_dim": int(_get(base, "dfg_attn_dim", checkpoint.get("dfg_attn_dim", 256))),
        "dfg_attn_tau": float(_get(base, "dfg_attn_tau", checkpoint.get("dfg_attn_tau", 4.0))),
        "use_ss2d_dfg": bool(_get(base, "use_ss2d_dfg", checkpoint.get("use_ss2d_dfg", False))),
        "dfg_gamma_max": float(_get(base, "dfg_gamma_max", checkpoint.get("dfg_gamma_max", 0.2))),
        "dfg_ss2d_fusion": str(_get(base, "dfg_ss2d_fusion", checkpoint.get("dfg_ss2d_fusion", "feature_residual"))),
        "dfg_beta": float(checkpoint.get("dfg_beta", _get(base, "dfg_beta", 0.10))),
        "dfg_beta_schedule": str(_get(base, "dfg_beta_schedule", checkpoint.get("dfg_beta_schedule", "fixed"))),
        "dfg_beta_target": float(_get(base, "dfg_beta_target", checkpoint.get("dfg_beta_target", 0.10))),
        "dfg_beta_current": float(checkpoint.get("dfg_beta", _get(base, "dfg_beta", 0.10))),
        "dfg_weight_residual_fp32": bool(checkpoint.get("dfg_weight_residual_fp32", True)),
        "use_soft_prompt": bool(checkpoint.get("use_soft_prompt", False)),
        "soft_prompt_ctx_len": int(checkpoint.get("soft_prompt_ctx_len", h6.get("ctx_len", 4))),
        "soft_prompt_init": str(checkpoint.get("soft_prompt_init", "phrase")),
        "soft_prompt_init_phrase": str(checkpoint.get("soft_prompt_init_phrase", "a photo of a")),
        "h6_progress": int(checkpoint.get("phase4_progress", h6.get("progress", 0))),
        "h6_num_factors": int(h6.get("num_factors", 4)),
        "h6_top_k": int(h6.get("top_k", 2)),
        "h6_role_topology": str(h6.get("role_topology", "flat")),
        "h6_role_teacher_scale": h6.get("role_teacher_scale"),
        "h6_intrinsic_factor_responsibility": bool(h6.get("intrinsic_factor_responsibility", False)),
        "h6_prediction_routing": str(h6.get("prediction_routing", "dense")),
        "h6_bank_dim": int(h6.get("bank_dim", 256)),
        "h6_router_dim": int(h6.get("router_dim", 128)),
        "h6_router_temperature": float(h6.get("router_temperature", 1.0)),
        "h6_router_boundary_mode": str(h6.get("router_boundary_mode", "none")),
        "h6_router_boundary_trust_scale": h6.get("router_boundary_trust_scale"),
        "h6_router_soft_epochs": int(h6.get("router_soft_epochs", 2)),
        "h6_sparse_transition_epochs": int(h6.get("sparse_transition_epochs", 1)),
        "h6_load_bias_enabled": bool(h6.get("load_bias_enabled", False)),
        "h6_load_bias_momentum": float(h6.get("load_bias_momentum", 0.9)),
        "h6_load_bias_step": float(h6.get("load_bias_step", 0.001)),
        "h6_load_bias_max": float(h6.get("load_bias_max", 0.03)),
        "h6_vae_hidden_dim": int(h6.get("vae_hidden_dim", 512)),
        "h6_vae_latent_dim": int(h6.get("vae_latent_dim", 256)),
        "h6_vae_class_ratio": float(h6.get("vae_class_ratio", 0.25)),
        "h6_slot_init_enabled": bool(h6.get("slot_init_enabled", False)),
        "h6_slot_init_scale": float(h6.get("slot_init_scale", 0.02)),
        "h6_slot_init_seed_offset": int(h6.get("slot_init_seed_offset", 6100)),
        "h6_factor_grad_diagnostics": bool(h6.get("factor_grad_diagnostics_enabled", False)),
        "h6_late_factor_identity_enabled": bool(h6.get("late_factor_identity_enabled", False)),
        "h6_factor_id_scale": float(h6.get("factor_id_scale", 0.02)),
        "h6_factor_id_max_ratio": float(h6.get("factor_id_max_ratio", 0.05)),
        "h6_factor_generator_specialization_enabled": bool(h6.get("factor_generator_specialization_enabled", False)),
        "h6_factor_head_init_scale": float(h6.get("factor_head_init_scale", 1e-3)),
        "h6_factor_local_dynamic_mix": float(h6.get("factor_local_dynamic_mix", 0.0)),
        "h6_cluster_responsibility": bool(h6.get("cluster_responsibility_enabled", False)),
        "h6_cluster_temperature": float(h6.get("cluster_temperature", 0.10)),
        "h6_router_query_mode": str(h6.get("router_query_mode", "local_global_bypass")),
        "h6_router_query_global_weight": float(h6.get("router_query_global_weight", 0.10)),
        "h6_router_local_bypass_scale": float(h6.get("router_local_bypass_scale", 0.10)),
        "h6_router_local_bypass_max_ratio": float(h6.get("router_local_bypass_max_ratio", 0.20)),
        "h6_router_local_projection_seed_offset": int(h6.get("router_local_projection_seed_offset", 7200)),
        "h6_router_key_anchor_enabled": bool(h6.get("router_key_anchor_enabled", True)),
        "h6_router_key_anchor_seed_offset": int(h6.get("router_key_anchor_seed_offset", 7300)),
        "h6_router_key_adaptation_initial_ratio": float(h6.get("router_key_adaptation_initial_ratio", 0.10)),
        "h6_router_key_adaptation_max_ratio": float(h6.get("router_key_adaptation_max_ratio", 0.25)),
        "h6_factor_context_anchor_enabled": bool(h6.get("factor_context_anchor_enabled", True)),
        "h6_factor_context_anchor_seed_offset": int(h6.get("factor_context_anchor_seed_offset", 7400)),
        "h6_factor_context_adaptation_initial_ratio": float(h6.get("factor_context_adaptation_initial_ratio", 0.10)),
        "h6_factor_context_adaptation_max_ratio": float(h6.get("factor_context_adaptation_max_ratio", 0.25)),
        "h6_factor_identity_tangent_projection_enabled": bool(h6.get("factor_identity_tangent_projection_enabled", True)),
        "lambda_h6_dynamic_mean_anchor": float(h6.get("lambda_dynamic_mean_anchor", 0.001)),
        "h6_dynamic_mean_anchor_min_cosine": float(h6.get("dynamic_mean_anchor_min_cosine", 0.70)),
        "h6_dynamic_mean_anchor_start_epoch": int(h6.get("dynamic_mean_anchor_start_epoch", 4)),
        "h6_dynamic_mean_anchor_warmup_epochs": int(h6.get("dynamic_mean_anchor_warmup_epochs", 3)),
        "h6_router_teacher_mode": str(h6.get("router_teacher_mode", "raw_cosine")),
        "h6_progress_version": str(h6.get("progress_version", "P1-v6")),
        "h6_local_factor_mode": str(h6.get("local_factor_mode", "center_spread")),
        "h6_local_center_mix": float(h6.get("local_center_mix", 0.05)),
        "h6_local_factor_spread": float(h6.get("local_factor_spread", 0.10)),
        "h6_expert_enabled": bool(h6.get("expert_enabled", False)),
        "h6_expert_bottleneck": int(h6.get("expert_bottleneck", 64)),
        "h6_expert_fofs_seed_offset": int(h6.get("expert_fofs_seed_offset", 7500)),
        "h6_expert_state_condition_scale": float(h6.get("expert_state_condition_scale", 0.25)),
        "h6_expert_scale_target": float(h6.get("expert_scale_target", 0.10)),
        "h6_expert_scale_start_epoch": int(h6.get("expert_scale_start_epoch", 1)),
        "h6_expert_scale_warmup_epochs": int(h6.get("expert_scale_warmup_epochs", 6)),
        "h6_expert_max_relative_ratio": float(h6.get("expert_max_relative_ratio", 0.10)),
        "diagnostics_mode": "light",
        "diagnostics_interval": 1,
    }


def load_checkpoint_for_evaluation(
    path: str | Path,
    device: torch.device,
    *,
    expected_epoch: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """Construct the frozen architecture and load exactly ``path``."""
    identity = checkpoint_identity(path, expected_epoch=expected_epoch, require_final_contract=True)
    checkpoint_path = Path(identity["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    kwargs = _construct_kwargs(checkpoint)
    model_name = str(
        checkpoint.get("model_name")
        or (checkpoint.get("phase2b_config") or {}).get("model_name")
        or "ViT-L-14-336"
    )
    clip_model = create_model(
        model_name=model_name,
        img_size=int(checkpoint.get("img_size", (checkpoint.get("phase2b_config") or {}).get("img_size", 518))),
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    if bool(checkpoint.get("gradient_checkpointing", False)):
        clip_model.set_grad_checkpointing(True)
    clip_model.eval()
    model = ACDCLIP(clip_model=clip_model, **kwargs).to(device)
    model.eval()
    model.h6_global_text_mode = str(checkpoint.get("global_text_mode", (checkpoint.get("h6_config") or {}).get("global_text_mode", "phase2b_hybrid")))
    model.prompt_mode = "h6_dynamic" if bool(checkpoint.get("h6_enabled", False)) else ("hybrid" if checkpoint.get("use_hybrid_soft_prompt") else ("soft" if checkpoint.get("use_soft_prompt") else "hard"))
    model.use_hybrid_soft_prompt = bool(checkpoint.get("use_hybrid_soft_prompt", False))
    model.use_soft_prompt = bool(checkpoint.get("use_soft_prompt", False))
    model.hybrid_alpha_current = float(checkpoint.get("hybrid_alpha_current", 0.20))
    restored_h6 = load_adapter_checkpoint(model, checkpoint)
    if restored_h6:
        model.prompt_mode = "h6_dynamic"
        model.use_hybrid_soft_prompt = True
        model.use_soft_prompt = False
        model.hybrid_alpha_current = float(checkpoint.get("hybrid_alpha_current", 0.20))
        model.h6.set_epoch(int(checkpoint.get("router_warmup_epoch", checkpoint.get("epoch", 1))))
    identity["loaded_checkpoint_path"] = str(checkpoint_path)
    if identity["loaded_checkpoint_path"] != identity["checkpoint_path"]:
        raise RuntimeError("explicit checkpoint path was not the loaded checkpoint")
    return model, identity, checkpoint
