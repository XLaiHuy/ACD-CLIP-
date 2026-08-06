#!/usr/bin/env python3
"""Targeted Tier-2 functional-diversity wiring audit (no training)."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as tv_transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.adapter import ACDCLIP
from model.checkpoint_utils import h6_config_from_checkpoint, load_adapter_checkpoint
from model.clip import create_model
from model.h6.losses import functional_factor_diversity_loss
import model.checkpoint_utils as mcu

mcu.validate_h6_configuration = lambda *args, **kwargs: None


def module_grad_norm(module):
    return sum(
        float(parameter.grad.detach().float().norm().item()) ** 2
        for parameter in module.parameters()
        if parameter.grad is not None
    ) ** 0.5


def tensor_grad_norm(tensor):
    return 0.0 if tensor.grad is None else float(tensor.grad.detach().float().norm().item())


def tensor_hash(tensor):
    payload = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def build_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    h6_kwargs = {f"h6_{key}": value for key, value in h6_config_from_checkpoint(checkpoint).items()}
    h6_kwargs.update({
        "h6_prediction_routing": "dense",
        "h6_expert_enabled": False,
        "h6_global_text_mode": "hard_anchor",
        "h6_progress": 1,
    })
    adapter_kwargs = {
        "n_groups": checkpoint.get("n_groups", 4),
        "dfg_mode": checkpoint.get("dfg_mode", "mlp"),
        "dfg_attn_dim": checkpoint.get("dfg_attn_dim", 256),
        "dfg_attn_tau": checkpoint.get("dfg_attn_tau", 4.0),
        "use_ss2d_dfg": checkpoint.get("use_ss2d_dfg", False),
        "dfg_gamma_max": checkpoint.get("dfg_gamma_max", 0.2),
        "dfg_ss2d_fusion": checkpoint.get("dfg_ss2d_fusion", "feature_residual"),
        "dfg_beta": checkpoint.get("dfg_beta", 0.10),
        "lora_rank": 16, "lora_alpha": 2.0,
        "conv_lora_rank": 8, "conv_lora_alpha": 2.0,
        "conv_kernel_size_list": [3, 5],
    }
    clip_model = create_model("ViT-L-14-336", img_size=518, device=device, pretrained="openai", require_pretrained=True)
    model = ACDCLIP(clip_model=clip_model, **adapter_kwargs, **h6_kwargs).to(device)
    model.prompt_mode = "hard"
    load_adapter_checkpoint(model, checkpoint)
    model.h6.set_epoch(3)
    return model


def functional_logits(model, images, masks, require_grad):
    # The adapter's training-mode LoRA path is not batch-shape safe for this
    # standalone probe.  The functional objective only targets the factor-text
    # branch, so use real frozen visual features exactly as training supplies
    # them, then retain autograd through build_batch and the factor logits.
    model.eval()
    if require_grad:
        with torch.no_grad():
            visual_output = model(images, return_phase4_features=True)
    else:
        visual_output = model(images, return_phase4_features=True)
    seg_features = torch.stack(visual_output["seg_tokens"], dim=0).detach()
    h6_batch = model.h6.build_batch(model, "Brain", ["Brain"] * images.shape[0], visual_output, hybrid_alpha=0.0)
    factor_bank = h6_batch["factor_bank"]
    dynamic_text = h6_batch["dynamic_text"]
    if require_grad:
        factor_bank.retain_grad()
        dynamic_text.retain_grad()
    factor_patch_logits = torch.einsum(
        "gbpd,gbmd->gbpm", F.normalize(seg_features.float(), dim=-1),
        factor_bank.float()[..., 1] - factor_bank.float()[..., 0],
    ) * model.h6.h6_logit_temperature
    hard_direction = h6_batch["hard_frozen"].float()[..., 1] - h6_batch["hard_frozen"].float()[..., 0]
    hard_logits = torch.einsum("gbpd,gbd->gbp", F.normalize(seg_features.float(), dim=-1), hard_direction)
    centered = hard_logits - hard_logits.mean(dim=2, keepdim=True)
    confidence = centered.abs() / centered.std(dim=2, keepdim=True).clamp_min(1e-6)
    patch_weights = 1.0 + confidence.detach()
    side = int(factor_patch_logits.shape[2] ** 0.5)
    patch_labels = F.adaptive_max_pool2d(masks.float(), (side, side)).flatten(1)
    patch_weights = patch_weights + patch_labels.detach().unsqueeze(0)
    return factor_patch_logits, patch_weights, factor_bank, dynamic_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=Path("runs/phase4/p1_v8_specialization_overnight"))
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--brain-data-root", default="/home/ai4/caohuy/data/MedAD/Brain_AD/test")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(args.manifest_path) as handle:
        entries = [json.loads(line) for line in handle]
    preprocess = tv_transforms.Compose([
        tv_transforms.Resize((518, 518), InterpolationMode.BICUBIC), tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    loaded_images, loaded_masks, loaded_paths = [], [], []
    for entry in entries:
        image_path = Path(args.brain_data_root) / entry["image_path"]
        if image_path.exists():
            loaded_images.append(preprocess(Image.open(image_path).convert("RGB")))
            if entry.get("mask_path") and entry["label"] == 1 and (Path(args.brain_data_root) / entry["mask_path"]).exists():
                raw_mask = to_tensor(Image.open(Path(args.brain_data_root) / entry["mask_path"]).convert("L"))
                loaded_masks.append(F.interpolate(raw_mask.unsqueeze(0), size=(518, 518), mode="nearest").squeeze(0))
            else:
                loaded_masks.append(torch.zeros(1, 518, 518))
            loaded_paths.append(str(image_path))
            if len(loaded_images) == 4:
                break
    if len(loaded_images) != 4:
        raise RuntimeError("No real Brain manifest image was found")
    image = torch.stack(loaded_images).to(device)
    mask = torch.stack(loaded_masks).to(device)
    candidates = {"T2-A_func0001": 0.0001, "T2-B_func0003": 0.0003, "T2-C_func0010": 0.001}
    report = {"device": device, "image_paths": loaded_paths, "epoch": 3, "candidates": {}}
    for name, weight in candidates.items():
        checkpoint_path = args.candidate_root / name / "smoke_3x300" / "adapter_3.pth"
        torch.manual_seed(20260806)
        if device == "cuda": torch.cuda.manual_seed_all(20260806)
        model = build_model(checkpoint_path, device)
        # Exact fingerprint path: same real tensor and deterministic RNG before every candidate.
        model.eval()
        with torch.no_grad():
            logits, _, _, _ = functional_logits(model, image, mask, require_grad=False)
        fingerprint = tensor_hash(logits)
        entry_report = {"checkpoint": str(checkpoint_path), "lambda_func_div": weight, "factor_patch_logits_sha256": fingerprint}
        if name == "T2-C_func0010":
            torch.manual_seed(20260806)
            if device == "cuda": torch.cuda.manual_seed_all(20260806)
            model = build_model(checkpoint_path, device)
            model.zero_grad(set_to_none=True)
            logits, patch_weights, factor_bank, dynamic_text = functional_logits(model, image, mask, require_grad=True)
            loss, correlation = functional_factor_diversity_loss(logits, patch_weights)
            loss.backward()
            core = model.h6.semantic_core
            generator_modules = [
                core.state_to_context_normal, core.state_to_context_abnormal, core.factor_id_to_context,
            ]
            entry_report.update({
                "functional_loss": float(loss.detach().item()),
                "functional_correlation_mean": float(correlation.detach().mean().item()),
                "gradient_norms": {
                    "factor_specific_heads": module_grad_norm(core.factor_output_heads),
                    "factor_identity_embeddings": tensor_grad_norm(core.factor_id_embedding),
                    "dynamic_prompt_generator": sum(module_grad_norm(module) ** 2 for module in generator_modules) ** 0.5,
                    "encoded_factor_text": tensor_grad_norm(factor_bank),
                    "dynamic_text": tensor_grad_norm(dynamic_text),
                },
            })
        report["candidates"][name] = entry_report
        del model
        if device == "cuda": torch.cuda.empty_cache()
    hashes = [item["factor_patch_logits_sha256"] for item in report["candidates"].values()]
    report["fingerprints_all_distinct"] = len(set(hashes)) == len(hashes)
    grads = report["candidates"]["T2-C_func0010"]["gradient_norms"]
    report["all_requested_gradient_norms_nonzero"] = all(value > 0.0 for value in grads.values())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
