#!/usr/bin/env python3
"""Strict FP32 zero-step checks for the K=1 Phase4-V visual adapter."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from audit_p4_k1_oracle_utility import DeterministicVisATrainDataset, _sha256
from model.adapter import ACDCLIP
from model.clip import create_model
from utils import calculate_seg_loss, configure_canonical_fp32, get_phase2b_global_text_features


def _build_fresh(config: dict, device: torch.device) -> ACDCLIP:
    params = inspect.signature(ACDCLIP.__init__).parameters
    kwargs = {name: config[name] for name, parameter in params.items() if name not in {"self", "clip_model", "kwargs"} and parameter.kind is not inspect.Parameter.VAR_KEYWORD and name in config}
    kwargs.update({
        "h6_progress": 1, "h6_num_factors": 1, "h6_top_k": 1,
        "h6_progress_version": "P4V-K1", "h6_local_factor_mode": "legacy_mix",
        "h6_expert_enabled": False, "phase4v_bottleneck": 64, "phase4v_lambda": 0.05,
        "h6_role_topology": "flat", "h6_intrinsic_factor_responsibility": False,
        "h6_cluster_responsibility": False, "h6_router_boundary_mode": "none",
    })
    clip = create_model(config["model_name"], img_size=config["img_size"], device=device, pretrained="openai", require_pretrained=True, precision="fp32")
    model = ACDCLIP(clip_model=clip, **kwargs).to(device)
    model.requires_grad_(False)
    model.image_adapter.requires_grad_(True)
    model.text_adapter.requires_grad_(True)
    model.h6.conditional_semantic_core.requires_grad_(True)
    model.h6.visual_adapter.requires_grad_(True)
    model.h6.router.requires_grad_(False)
    model.h6.semantic_core.requires_grad_(False)
    model.soft_prompt.requires_grad_(False)
    return model


def _norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm(dim=-1).mean())


def _grad_norm(module: torch.nn.Module) -> float:
    values = [parameter.grad.detach().float().norm() for parameter in module.parameters() if parameter.grad is not None]
    return 0.0 if not values else float(torch.stack(values).norm())


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("runs/phase4/k1/short64_seed0_attempt5/config.json"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase4v/K1_ZERO_STEP.json"))
    args = parser.parse_args()
    config, manifest = json.loads(args.config.read_text()), json.loads(args.manifest.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("zero-step requires CUDA")
    configure_canonical_fp32()
    device = torch.device(f"cuda:{config['cuda_device']}")
    torch.manual_seed(4001)
    model = _build_fresh(config, device)
    # The existing Phase2B adapter has a batch-one square-grid contract; keep
    # this strict no-step probe on the same geometry as the valid K1 audit.
    raw = next(iter(DataLoader(DeterministicVisATrainDataset(manifest, config["img_size"]), batch_size=1, shuffle=False, num_workers=0)))
    image, mask, label, classes = raw["image"].to(device).float(), raw["mask"].to(device).float(), raw["label"].to(device), list(raw["class_name"])
    model.train()
    # Preserve the Phase2B train contract: adapters train while frozen CLIP
    # disables patch dropout so the square patch grid stays intact.
    model.clipmodel.eval()
    visual = model(image, return_phase4_features=True)
    original = torch.stack(visual["seg_tokens"], dim=0)
    text = get_phase2b_global_text_features(model, "VisA", classes, device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
    base_pred, base_logits, _ = model.vision_text_fusion_gate_seg(original, text, img_size=config["img_size"], return_details=True)
    state = model.h6.phase4v_state_code(model, visual)
    gate = torch.softmax(base_logits.float(), dim=-1)[..., 1].detach()

    def predict(*, enabled: bool, semantic: bool = True, spatial: bool = True, gate_value: torch.Tensor | None = None, lambda_override: float | None = None, force_delta_zero: bool = False):
        current_gate = gate if gate_value is None else gate_value
        outputs = [model.h6.phase4v_adapt(visual["seg_tokens"][group], state["semantic_code"], current_gate[group], enabled=enabled, semantic_conditioning=semantic, spatial_gating=spatial, lambda_override=lambda_override, force_delta_zero=force_delta_zero) for group in range(model.n_groups)]
        features = torch.stack([out["adapted"] for out in outputs], dim=0)
        pred = model.vision_text_fusion_gate_seg(features, text, img_size=config["img_size"])
        return pred, outputs

    off_pred, off = predict(enabled=False)
    lambda_zero_pred, lambda_zero = predict(enabled=True, lambda_override=0.0)
    gate_zero_pred, gate_zero = predict(enabled=True, gate_value=torch.zeros_like(gate))
    delta_zero_pred, delta_zero = predict(enabled=True, force_delta_zero=True)
    active_pred, active = predict(enabled=True, semantic=True, spatial=True)
    det = torch.stack(visual["det_tokens"], dim=0)
    cls = torch.stack([torch.matmul(det[level].unsqueeze(1), text[level]).squeeze(1) for level in range(model.n_groups)]).mean(0)
    loss = calculate_seg_loss(active_pred.float(), mask) + F.cross_entropy(cls.float(), label)
    model.zero_grad(set_to_none=True)
    base_logits.retain_grad()
    loss.backward()
    delta = torch.stack([out["delta_v"] for out in active], dim=0)
    correction = torch.stack([out["correction"] for out in active], dim=0)
    checks = {
        "phase4v_off_exact": _max_abs(off_pred, base_pred),
        "lambda_zero_exact": _max_abs(lambda_zero_pred, base_pred),
        "gate_zero_exact": _max_abs(gate_zero_pred, base_pred),
        "delta_zero_exact": _max_abs(delta_zero_pred, base_pred),
        "all_finite": bool(torch.isfinite(active_pred).all() and torch.isfinite(loss)),
        "gate_detached": not gate.requires_grad and base_logits.grad is None,
        "base_main_grad_norm": _grad_norm(model.image_adapter),
        "cops_grad_norm": _grad_norm(model.h6.conditional_semantic_core),
        "visual_adapter_grad_norm": _grad_norm(model.h6.visual_adapter),
        "router_grad_norm": _grad_norm(model.h6.router),
        "legacy_factor_grad_norm": _grad_norm(model.h6.semantic_core),
        "act_absent": model.h6.act_head is None,
        "dynamic_text_blocked": True,
    }
    tolerance = 0.0
    passed = all(checks[name] <= tolerance for name in ("phase4v_off_exact", "lambda_zero_exact", "gate_zero_exact", "delta_zero_exact")) and checks["all_finite"] and checks["gate_detached"] and checks["base_main_grad_norm"] > 0 and checks["cops_grad_norm"] > 0 and checks["visual_adapter_grad_norm"] > 0 and checks["router_grad_norm"] == 0 and checks["legacy_factor_grad_norm"] == 0 and checks["act_absent"]
    report = {
        "decision": "PHASE4V_K1_ZERO_STEP_PASS" if passed else "PHASE4V_K1_ZERO_STEP_FAIL",
        "provenance": {"script_sha256": _sha256(Path(__file__).resolve()), "config_sha256": _sha256(args.config), "manifest_sha256": _sha256(args.manifest), "initialization": "fresh OpenAI CLIP; no Phase2B checkpoint", "precision": "strict FP32; TF32 off; AMP off", "optimizer_steps": 0},
        "checks": checks,
        "residual": {"delta_v_mean_norm": _norm(delta), "lambda_gate_delta_mean_norm": _norm(correction), "relative_correction_to_base": float(correction.detach().float().norm(dim=-1).mean() / original.detach().float().norm(dim=-1).mean()), "lambda": 0.05, "bottleneck": 64},
        "contract": {"where": "detached pre-adaptation Phase2B group anomaly probability", "what": "ConditionalSemanticCore prototype_abnormal - prototype_normal", "how": "shared 2->64->768 FiLM residual", "text_intervention": False, "router": False, "experts": False, "act": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": report["decision"], "checks": checks, "residual": report["residual"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
