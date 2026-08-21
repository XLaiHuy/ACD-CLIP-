"""Single compatibility bridge for constructing the existing Phase2B base.

The historical adapter constructor still exposes an optional legacy branch.
Canonical callers never pass those options: this bridge forces the branch off,
rejects checkpoints that contain legacy state, and exposes only the adapter
state required by Phase2B.
"""
from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Mapping

import torch


def _adapter_and_clip() -> tuple[type, Any]:
    from model.adapter import ACDCLIP
    from model.clip import create_model

    return ACDCLIP, create_model


def _constructor_kwargs(config: Mapping[str, Any], checkpoint: Mapping[str, Any] | None) -> dict[str, Any]:
    adapter_cls, _ = _adapter_and_clip()
    parameters = inspect.signature(adapter_cls.__init__).parameters
    legacy_names = {
        "h6_progress",
        "h6_num_factors",
        "h6_top_k",
        "h6_bank_dim",
        "h6_router_dim",
        "h6_router_temperature",
        "h6_router_soft_epochs",
        "h6_sparse_transition_epochs",
    }
    kwargs: dict[str, Any] = {}
    for name, parameter in parameters.items():
        if name in {"self", "clip_model", "kwargs"} or parameter.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        if name in legacy_names or name.startswith("h6_"):
            continue
        if name in config:
            kwargs[name] = config[name]
    # This is the only place where the old constructor's compatibility switch
    # is used.  No canonical caller can request the legacy branch.
    kwargs["h6_progress"] = 0
    if checkpoint is not None:
        beta = checkpoint.get("dfg_beta_current", checkpoint.get("dfg_beta", config.get("dfg_beta", 0.0)))
        kwargs["dfg_beta_current"] = float(beta)
    return kwargs


def _reject_legacy_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    forbidden = (
        "h6_state_dict",
        "h6_config",
        "h6_enabled",
        "phase4_progress",
        "phase4_config",
    )
    found = [key for key in forbidden if key in checkpoint and checkpoint.get(key) not in (None, False, 0, {})]
    if found:
        raise ValueError(
            "canonical Phase2B refuses a checkpoint containing legacy state: "
            + ", ".join(found)
        )


def assert_legacy_branch_disabled(model: Any) -> None:
    """Runtime graph assertions required by the canonical protocol."""
    if bool(getattr(model, "h6_enabled", False)):
        raise AssertionError("legacy branch is active in canonical Phase2B")
    if getattr(model, "h6", None) is not None:
        raise AssertionError("legacy module was constructed in canonical Phase2B")
    trainable_legacy = [
        name for name, parameter in model.named_parameters()
        if "h6" in name.lower() and parameter.requires_grad
    ]
    if trainable_legacy:
        raise AssertionError(f"legacy trainable parameters remain: {trainable_legacy[:5]}")


def _module_parameters(module: Any) -> list[torch.nn.Parameter]:
    if module is None or not hasattr(module, "parameters"):
        return []
    return list(module.parameters())


def assert_phase2b_trainable_contract(
    model: Any,
    *,
    soft_prompt_trainable: bool | None = None,
) -> None:
    """Assert the canonical trainable/frozen component partition."""
    assert_legacy_branch_disabled(model)
    clip_parameters = _module_parameters(getattr(model, "clipmodel", None))
    if any(parameter.requires_grad for parameter in clip_parameters):
        raise AssertionError("canonical Phase2B CLIP backbone has trainable parameters")
    for name in ("image_adapter", "text_adapter"):
        parameters = _module_parameters(getattr(model, name, None))
        if not parameters:
            raise AssertionError(f"canonical Phase2B component is missing: {name}")
        if any(not parameter.requires_grad for parameter in parameters):
            raise AssertionError(f"canonical Phase2B component is partially frozen: {name}")
    soft_parameters = _module_parameters(getattr(model, "soft_prompt", None))
    if soft_prompt_trainable is not None and any(
        parameter.requires_grad != bool(soft_prompt_trainable)
        for parameter in soft_parameters
    ):
        raise AssertionError(
            "soft-prompt requires_grad does not match the canonical epoch schedule"
        )


def assert_phase2b_gradient_contract(
    model: Any,
    *,
    soft_prompt_trainable: bool,
) -> None:
    """Audit gradients immediately after backward, before clipping/step."""
    assert_phase2b_trainable_contract(model, soft_prompt_trainable=soft_prompt_trainable)
    clip_parameters = _module_parameters(getattr(model, "clipmodel", None))
    if any(parameter.grad is not None for parameter in clip_parameters):
        raise AssertionError("frozen CLIP backbone received gradients")
    for name in ("image_adapter", "text_adapter"):
        parameters = _module_parameters(getattr(model, name, None))
        if not any(parameter.grad is not None for parameter in parameters):
            raise AssertionError(f"{name} received no gradient in canonical Phase2B backward")
    soft_parameters = _module_parameters(getattr(model, "soft_prompt", None))
    if soft_prompt_trainable:
        if not any(parameter.grad is not None for parameter in soft_parameters):
            raise AssertionError("trainable soft prompt received no gradient")
    elif any(parameter.grad is not None for parameter in soft_parameters):
        raise AssertionError("frozen soft prompt received a gradient")


def build_adapter(
    config: Mapping[str, Any],
    clip_asset: str | Path,
    device: torch.device,
    checkpoint: Mapping[str, Any] | None = None,
    trainable: bool = True,
) -> Any:
    """Construct the existing ACDCLIP adapter with the legacy path disabled."""
    adapter_cls, create_model = _adapter_and_clip()
    asset = Path(clip_asset).expanduser().resolve()
    if not asset.exists():
        raise FileNotFoundError(f"CLIP asset not found: {asset}")
    os.environ["ACDCLIP_CLIP_VITL14_336"] = str(asset)
    model_name = str(config["model_name"])
    image_size = int(config["img_size"])
    precision = str(config.get("precision", "fp32"))
    if precision != "fp32":
        raise ValueError("canonical Phase2B setup requires precision=fp32")
    clip = create_model(
        model_name,
        img_size=image_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
        precision="fp32",
    )
    if bool(config.get("grad_checkpointing", False)):
        clip.set_grad_checkpointing(True)
    model = adapter_cls(
        clip_model=clip,
        **_constructor_kwargs(config, checkpoint),
    ).to(device)
    assert_legacy_branch_disabled(model)
    if checkpoint is not None:
        _reject_legacy_checkpoint(checkpoint)
    model.use_hybrid_soft_prompt = bool(config.get("use_hybrid_soft_prompt", False))
    model.use_soft_prompt = bool(config.get("use_soft_prompt", False))
    model.hybrid_alpha_max = float(config.get("hybrid_alpha_max", 0.2))
    model.soft_prompt_freeze_epochs = int(config.get("soft_prompt_freeze_epochs", 3))
    model.prompt_mode = "hybrid" if model.use_hybrid_soft_prompt else "soft" if model.use_soft_prompt else "hard"
    model.hybrid_alpha_current = float(config.get("hybrid_alpha_current", 0.0))
    if trainable:
        model.train()
        model.clipmodel.eval()
        model.image_encoder.eval()
        if hasattr(model.clipmodel, "requires_grad_"):
            model.clipmodel.requires_grad_(False)
        if hasattr(model.image_encoder, "requires_grad_"):
            model.image_encoder.requires_grad_(False)
        model.image_adapter.requires_grad_(True)
        model.text_adapter.requires_grad_(True)
        # E1-E3 are frozen by the canonical schedule; the trainer enables the
        # soft prompt explicitly at E4+ after constructing the optimizer.
        model.soft_prompt.requires_grad_(False)
        assert_phase2b_trainable_contract(model, soft_prompt_trainable=False)
    else:
        model.eval()
        model.clipmodel.eval()
        model.requires_grad_(False)
    return model


def load_adapter_state(model: Any, checkpoint: Mapping[str, Any]) -> None:
    """Load only the original adapter/text/prompt state into a Phase2B model."""
    _reject_legacy_checkpoint(checkpoint)
    from model.checkpoint_utils import load_adapter_checkpoint

    restored_legacy = load_adapter_checkpoint(model, checkpoint)
    if restored_legacy:
        raise ValueError("checkpoint loader reported legacy state in canonical Phase2B")
    if hasattr(model, "set_dfg_beta"):
        model.set_dfg_beta(float(checkpoint.get("dfg_beta_current", checkpoint.get("dfg_beta", model.dfg_beta))))
    model.use_hybrid_soft_prompt = bool(checkpoint.get("use_hybrid_soft_prompt", False))
    model.use_soft_prompt = bool(checkpoint.get("use_soft_prompt", False))
    model.hybrid_alpha_current = float(checkpoint.get("hybrid_alpha_current", 0.0))
    model.hybrid_alpha_max = float(checkpoint.get("hybrid_alpha_max", getattr(model, "hybrid_alpha_max", 0.2)))
    model.soft_prompt_freeze_epochs = int(checkpoint.get("soft_prompt_freeze_epochs", getattr(model, "soft_prompt_freeze_epochs", 3)))
    model.prompt_mode = str(checkpoint.get("prompt_mode", getattr(model, "prompt_mode", "hybrid")))
    assert_legacy_branch_disabled(model)


def trainable_parameter_summary(model: Any) -> dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    clip_parameters = _module_parameters(getattr(model, "clipmodel", None))
    image_parameters = _module_parameters(getattr(model, "image_adapter", None))
    text_parameters = _module_parameters(getattr(model, "text_adapter", None))
    soft_parameters = _module_parameters(getattr(model, "soft_prompt", None))
    return {
        "trainable": int(trainable),
        "frozen": int(frozen),
        "clip_total": int(sum(parameter.numel() for parameter in clip_parameters)),
        "clip_trainable": int(sum(parameter.numel() for parameter in clip_parameters if parameter.requires_grad)),
        "image_adapter_trainable": int(sum(parameter.numel() for parameter in image_parameters if parameter.requires_grad)),
        "text_adapter_trainable": int(sum(parameter.numel() for parameter in text_parameters if parameter.requires_grad)),
        "soft_prompt_trainable": int(sum(parameter.numel() for parameter in soft_parameters if parameter.requires_grad)),
    }


def assert_canonical_config(config: Mapping[str, Any]) -> None:
    forbidden = [key for key in config if key.lower().startswith(("h6", "lambda_h6"))]
    if forbidden:
        raise ValueError(f"legacy fields are not allowed in canonical config: {forbidden}")


def runtime_audit(model: Any) -> dict[str, Any]:
    assert_legacy_branch_disabled(model)
    legacy_parameters = [name for name, _ in model.named_parameters() if "h6" in name.lower()]
    trainable_legacy = [
        name for name, parameter in model.named_parameters()
        if "h6" in name.lower() and parameter.requires_grad
    ]
    return {
        "legacy_module_active": bool(getattr(model, "h6_enabled", False)),
        "legacy_output_consumed": False,
        "legacy_parameter_count": len(legacy_parameters),
        "legacy_trainable_parameter_count": len(trainable_legacy),
        "parameter_summary": trainable_parameter_summary(model),
    }
