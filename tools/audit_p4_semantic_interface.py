#!/usr/bin/env python3
"""Strict Stage-0 interface audit on one real fresh OpenAI-CLIP TRAIN batch."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import get_text_and_image_dataset
from model.adapter import ACDCLIP
from model.clip import create_model
from model.h6.conditional_semantics import predictor_aligned_abnormal_residual
from utils import get_phase2b_global_text_features


PASS_LABEL = "PHASE4_INTERFACE_CLEANUP_PASS"
FAIL_LABEL = "PHASE4_INTERFACE_CLEANUP_FAIL"


def _historical_attention_dfg(model, image, text, group_index):
    """Verbatim algebra of the pre-refactor attention/SS2D DFG path."""
    v_gap = image.mean(dim=1)
    v_ss2d = model.image_adapter["dfg_ss2d_branches"][group_index](image)
    q_gap = model.image_adapter["vision_text_q"][group_index](v_gap)
    q_ss2d = model.image_adapter["vision_text_q"][group_index](v_ss2d)
    key = model.image_adapter["vision_text_k"][group_index]
    normal = text[..., 0]
    abnormal = text[..., 1]
    key_normal = key(normal)
    key_abnormal = key(abnormal)
    scale = model.dfg_attn_dim**0.5 * model.dfg_attn_tau

    def score(query, keys):
        return torch.einsum("bd,bnd->bn", query.float(), keys.float()) / scale

    gap_normal = F.softmax(score(q_gap, key_normal), dim=1)
    gap_abnormal = F.softmax(score(q_gap, key_abnormal), dim=1)
    ss2d_normal = F.softmax(score(q_ss2d, key_normal), dim=1)
    ss2d_abnormal = F.softmax(score(q_ss2d, key_abnormal), dim=1)
    weight_normal = (1.0 - model.dfg_beta) * gap_normal + model.dfg_beta * ss2d_normal
    weight_abnormal = (1.0 - model.dfg_beta) * gap_abnormal + model.dfg_beta * ss2d_abnormal
    fused_normal = F.normalize(
        torch.einsum("bn,bnd->bd", weight_normal, normal), dim=-1
    )
    fused_abnormal = F.normalize(
        torch.einsum("bn,bnd->bd", weight_abnormal, abnormal), dim=-1
    )
    return torch.stack([fused_normal, fused_abnormal], dim=-1)


def _grad_sum(parameters):
    return float(
        sum(
            parameter.grad.detach().float().abs().sum().item()
            for parameter in parameters
            if parameter.grad is not None
        )
    )


def _run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("strict real-batch Stage-0 audit requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    clip_model = create_model(
        "ViT-L-14-336",
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
        precision="fp32",
    )
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=3,
        dfg_mode="attn",
        dfg_attn_dim=256,
        dfg_attn_tau=8.0,
        use_ss2d_dfg=True,
        dfg_gamma_max=0.2,
        dfg_ss2d_fusion="weight_residual",
        dfg_beta=0.10,
        dfg_weight_residual_fp32=True,
        use_soft_prompt=True,
        soft_prompt_ctx_len=4,
        h6_progress=1,
        h6_num_factors=1,
        h6_top_k=1,
        h6_progress_version="P4-CSF-K1",
        h6_local_factor_mode="legacy_mix",
        h6_expert_enabled=False,
        h6_late_factor_identity_enabled=False,
        h6_factor_generator_specialization_enabled=False,
        h6_router_boundary_mode="none",
        h6_intrinsic_factor_responsibility=False,
        h6_cluster_responsibility=False,
    ).to(device)
    model.train()
    model.clipmodel.eval()
    model.requires_grad_(False)
    model.h6.conditional_semantic_core.requires_grad_(True)

    dataset = get_text_and_image_dataset("VisA", args.img_size, "train")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True
    )
    raw_batch = next(iter(loader))
    image = raw_batch["image"].to(device=device, dtype=torch.float32)
    class_names = list(raw_batch["class_name"])

    with torch.no_grad():
        visual = model(image, return_phase4_features=True)
        base_text = get_phase2b_global_text_features(
            model,
            "VisA",
            class_names,
            device,
            use_hybrid_soft_prompt=False,
            use_soft_prompt=False,
        ).float()

    base_cross = base_text.permute(1, 0, 2, 3)
    historical_text = []
    reconstructed_text = []
    historical_logits = []
    reconstructed_logits = []
    for group_index, patches in enumerate(visual["seg_tokens"]):
        historical = _historical_attention_dfg(model, patches, base_cross, group_index)
        weights = model.compute_dfg_weights(patches, base_cross, group_index)
        reconstructed = model.apply_dfg_weights(
            base_cross, weights["normal"], weights["abnormal"]
        )
        historical_text.append(historical)
        reconstructed_text.append(reconstructed)
        historical_logits.append(torch.matmul(10.0 * patches, historical))
        reconstructed_logits.append(torch.matmul(10.0 * patches, reconstructed))
    historical_text = torch.stack(historical_text)
    reconstructed_text = torch.stack(reconstructed_text)
    historical_logits = torch.stack(historical_logits)
    reconstructed_logits = torch.stack(reconstructed_logits)

    batch = model.h6.build_batch(
        model,
        "VisA",
        class_names,
        visual,
        hybrid_alpha=0.0,
        base_text_features=base_text,
    )
    batch["dynamic_text"].retain_grad()
    batch["predictor_residual_logits"].float().mean().backward()

    zero = predictor_aligned_abnormal_residual(
        batch["base_group_logits"],
        batch["base_group_logits"][..., 1],
        model.h6.rho_values(),
    )
    core = model.h6.conditional_semantic_core
    checks = {
        "phase2b_logit_max_abs_error": float(
            (historical_logits - batch["base_group_logits"]).abs().max().item()
        ),
        "dfg_text_reconstruction_max_abs_error": float(
            (historical_text - reconstructed_text).abs().max().item()
        ),
        "dfg_logit_reconstruction_max_abs_error": float(
            (historical_logits - reconstructed_logits).abs().max().item()
        ),
        "zero_residual_max_abs": float(zero["predictor_residual_logits"].abs().max().item()),
        "zero_final_base_max_abs_error": float(
            (zero["final_group_logits"] - batch["base_group_logits"]).abs().max().item()
        ),
        "normal_invariant_max_abs_error": float(batch["normal_invariant_error"].item()),
        "grad_cops_state": _grad_sum(
            list(core.prototype_attention.parameters())
            + list(core.abnormal_state_update.parameters())
            + [core.abnormal_query]
        ),
        "grad_state_to_context": _grad_sum(core.state_to_context_abnormal.parameters()),
        "grad_class_to_context": _grad_sum(core.class_to_context.parameters()),
        "grad_vae": _grad_sum(core.class_vae.parameters()),
        "grad_dynamic_text": float(batch["dynamic_text"].grad.abs().sum().item()),
        "grad_legacy_router": _grad_sum(model.h6.router.parameters()),
        "grad_legacy_factor_core": _grad_sum(model.h6.semantic_core.parameters()),
        "rho": [float(value) for value in model.h6.rho_values().detach().cpu()],
        "all_outputs_finite": bool(
            all(
                torch.isfinite(value).all().item()
                for value in (
                    historical_logits,
                    reconstructed_logits,
                    batch["dynamic_text"],
                    batch["predictor_residual_logits"],
                    batch["final_group_logits"],
                )
            )
        ),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "amp_enabled": False,
        "dtype": str(image.dtype),
    }
    passed = (
        checks["phase2b_logit_max_abs_error"] <= 1e-6
        and checks["dfg_text_reconstruction_max_abs_error"] <= 1e-6
        and checks["dfg_logit_reconstruction_max_abs_error"] <= 1e-6
        and checks["zero_residual_max_abs"] == 0.0
        and checks["zero_final_base_max_abs_error"] == 0.0
        and checks["normal_invariant_max_abs_error"] == 0.0
        and checks["grad_cops_state"] > 0.0
        and checks["grad_state_to_context"] > 0.0
        and checks["grad_class_to_context"] > 0.0
        and checks["grad_vae"] > 0.0
        and checks["grad_dynamic_text"] > 0.0
        and checks["grad_legacy_router"] == 0.0
        and checks["grad_legacy_factor_core"] == 0.0
        and max(abs(value - 0.05) for value in checks["rho"]) <= 1e-6
        and checks["all_outputs_finite"]
        and not checks["tf32_matmul"]
        and not checks["tf32_cudnn"]
        and checks["dtype"] == "torch.float32"
    )
    return {
        "decision": PASS_LABEL if passed else FAIL_LABEL,
        "fresh_initialization": "OpenAI CLIP only",
        "dataset": "VisA",
        "split": "train",
        "batch_size": 1,
        "class_names": class_names,
        "optimizer_steps": 0,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/phase4/stage0/semantic_interface_zero_step.json"),
    )
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        report = _run(args)
    except Exception as error:
        report = {
            "decision": FAIL_LABEL,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["decision"] == PASS_LABEL else 1)


if __name__ == "__main__":
    main()
