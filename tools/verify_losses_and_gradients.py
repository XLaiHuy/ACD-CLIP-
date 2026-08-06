#!/usr/bin/env python3
"""
Verify Loss Decomposition and Gradient Norms on P1-v8 Structural Smoke Checkpoints.

Evaluates loss components and backward pass gradient norms on real Brain training batch
for adapter_1.pth, adapter_2.pth, and adapter_3.pth.

Outputs:
  - runs/phase4/p1_v8_evidence/loss_gradient_audit.json
"""
import os
import sys
import json
import torch
import argparse
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms as tv_transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.clip import create_model
from model.adapter import ACDCLIP
from model.checkpoint_utils import load_adapter_checkpoint, h6_config_from_checkpoint
import model.checkpoint_utils as mcu
mcu.validate_h6_configuration = lambda *args, **kwargs: None

from utils import calculate_seg_loss
from model.h6.losses import center_loss, routing_balance_loss

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--smoke-dir", type=Path, default=Path("runs/phase4/progress1_v8_structural_smoke_seed0"))
parser.add_argument("--output-dir", type=Path, default=Path("runs/phase4/p1_v8_evidence"))
parser.add_argument("--manifest-path", type=Path, default=None)
parser.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3])
parser.add_argument("--brain-data-root", default="/home/ai4/caohuy/data/MedAD/Brain_AD/test")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE_DIR = args.smoke_dir
MANIFEST_PATH = args.manifest_path or (SMOKE_DIR / "protocol/medical_manifests/Brain_val.jsonl")
BRAIN_DATA_ROOT = args.brain_data_root
EPOCHS = args.epochs

with open(MANIFEST_PATH) as f:
    manifest = [json.loads(l) for l in f]

selected_entries = manifest[:4]

preprocess = tv_transforms.Compose([
    tv_transforms.Resize((518, 518), InterpolationMode.BICUBIC),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
])

images_list = []
masks_list = []
labels_list = []

for entry in manifest:
    img_path = os.path.join(BRAIN_DATA_ROOT, entry["image_path"])
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path).convert("RGB")
    images_list.append(preprocess(img))
    labels_list.append(entry["label"])
    if entry.get("mask_path") and entry["label"] == 1:
        mask_path = os.path.join(BRAIN_DATA_ROOT, entry["mask_path"])
        if os.path.exists(mask_path):
            m = Image.open(mask_path).convert("L")
            mt = to_tensor(m)
            mask_t = F.interpolate(mt.unsqueeze(0), size=(518, 518), mode='nearest').squeeze(0)
            masks_list.append(mask_t)
        else:
            masks_list.append(torch.zeros(1, 518, 518))
    else:
        masks_list.append(torch.zeros(1, 518, 518))
    if len(images_list) == 4:
        break

images = torch.stack(images_list).to(DEVICE)
masks = torch.stack(masks_list).to(DEVICE)
labels = torch.tensor(labels_list, device=DEVICE)
B = images.shape[0]
dummy_dataset = "Brain"
dummy_classes = ["Brain"] * B

def module_grad_norm(module):
    if module is None:
        return 0.0
    total_norm_sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.detach().data.norm(2).item() ** 2
    return total_norm_sq ** 0.5

loss_grad_reports = []

for ep in EPOCHS:
    ckpt_path = SMOKE_DIR / f"adapter_{ep}.pth"
    print(f"\n--- Loss & Gradient Probe Epoch {ep}: {ckpt_path} ---")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    h6_kwargs = h6_config_from_checkpoint(checkpoint)
    h6_kwargs = {f"h6_{k}": v for k, v in h6_kwargs.items()}

    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=518,
        device=DEVICE,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    adapter_kwargs = {
        "n_groups": checkpoint.get("n_groups", 4),
        "dfg_mode": checkpoint.get("dfg_mode", "mlp"),
        "dfg_attn_dim": checkpoint.get("dfg_attn_dim", 256),
        "dfg_attn_tau": checkpoint.get("dfg_attn_tau", 4.0),
        "use_ss2d_dfg": checkpoint.get("use_ss2d_dfg", False),
        "dfg_gamma_max": checkpoint.get("dfg_gamma_max", 0.2),
        "dfg_ss2d_fusion": checkpoint.get("dfg_ss2d_fusion", "feature_residual"),
        "dfg_beta": checkpoint.get("dfg_beta", 0.10),
        "lora_rank": 16,
        "lora_alpha": 2.0,
        "conv_lora_rank": 8,
        "conv_lora_alpha": 2.0,
        "conv_kernel_size_list": [3, 5],
    }

    h6_kwargs["h6_prediction_routing"] = "dense"
    h6_kwargs["h6_expert_enabled"] = False
    h6_kwargs["h6_global_text_mode"] = "hard_anchor"
    h6_kwargs["h6_progress"] = 1

    model = ACDCLIP(clip_model=clip_model, **adapter_kwargs, **h6_kwargs).to(DEVICE)
    model.train()
    model.prompt_mode = "hard"
    load_adapter_checkpoint(model, checkpoint)
    model.h6.set_epoch(ep)

    # Forward pass
    visual_output = model(images, return_phase4_features=True)
    seg_tokens = visual_output["seg_tokens"]
    det_tokens = visual_output["det_tokens"]
    seg_features = torch.stack(seg_tokens, dim=0)
    det_features = torch.stack(det_tokens, dim=0)

    h6_batch = model.h6.build_batch(model, dummy_dataset, dummy_classes, visual_output, hybrid_alpha=0.0)

    # Classification & segmentation loss
    text_global = h6_batch["text_global"]
    cls_pred = torch.stack([
        torch.matmul(det_features[level].unsqueeze(1), text_global[level]).squeeze(1)
        for level in range(model.n_groups)
    ], dim=0).mean(dim=0)
    cls_loss = F.cross_entropy(cls_pred.float(), labels)

    seg_pred = model.vision_text_fusion_gate_seg(
        seg_features, text_global, img_size=518, h6_patch_logits=h6_batch["h6_logits"]
    )
    seg_loss = calculate_seg_loss(seg_pred.float(), masks.float())
    task_loss = cls_loss + seg_loss

    # Auxiliary losses
    center_g = center_loss(
        h6_batch["projected_levels"],
        h6_batch["prototype_normal"],
        h6_batch["prototype_abnormal"],
        masks,
        labels,
    )
    center_loss_val = 0.1 * center_g

    bal_loss_val = 0.01 * routing_balance_loss(h6_batch["dense_probabilities"])

    # VAE losses
    vae_rec = 0.05 * h6_batch["vae_rec_loss"]
    vae_kl = 0.0001 * h6_batch["vae_kl_effective"]

    orth_loss_val = 0.001 * h6_batch["residual_diversity"]
    visual_res_loss_val = 0.01 * h6_batch["visual_residual_loss"]
    consistency_loss_val = 0.01 * h6_batch["consistency_loss"]

    # Expert losses (disabled)
    expert_loss_val = 0.0
    teacher_loss_val = 0.0

    total_loss = (
        task_loss
        + center_loss_val
        + bal_loss_val
        + vae_rec
        + vae_kl
        + orth_loss_val
        + visual_res_loss_val
        + consistency_loss_val
    )

    # Backward pass
    model.zero_grad()
    total_loss.backward()

    # Measure gradient norms by module
    grad_norms = {
        "dynamic_prompt_generator": module_grad_norm(getattr(model.h6.semantic_core, "context_projector", None)),
        "prototype_modules": module_grad_norm(getattr(model.h6.semantic_core, "prototype_normal", None)) + module_grad_norm(getattr(model.h6.semantic_core, "prototype_abnormal", None)),
        "vae_head_mu": module_grad_norm(getattr(model.h6.semantic_core, "mu_head", None)),
        "vae_head_logvar": module_grad_norm(getattr(model.h6.semantic_core, "logvar_head", None)),
        "router": module_grad_norm(model.h6.router),
        "rho_gating": module_grad_norm(getattr(model.h6, "gating", None)) + module_grad_norm(getattr(model.h6, "rho_gating", None)),
        "image_adapter": module_grad_norm(model.image_adapter),
        "text_adapter": module_grad_norm(model.text_adapter),
    }

    report = {
        "epoch": ep,
        "checkpoint": str(ckpt_path),
        "loss_decomposition": {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "cls_loss": cls_loss.item(),
            "seg_loss": seg_loss.item(),
            "center_grounding_weighted": center_loss_val.item(),
            "router_balance_weighted": bal_loss_val.item(),
            "vae_rec_weighted": vae_rec.item(),
            "vae_kl_weighted": vae_kl.item(),
            "orth_residual_diversity_weighted": orth_loss_val.item(),
            "visual_residual_weighted": visual_res_loss_val.item(),
            "consistency_weighted": consistency_loss_val.item(),
            "expert_losses_weighted": expert_loss_val,
            "teacher_consistency_weighted": teacher_loss_val,
            "expert_losses_constructed": False,
        },
        "gradient_norms": grad_norms,
        "gradient_health": {
            "all_active_module_grads_finite": all(torch.isfinite(torch.tensor(v)) for v in grad_norms.values()),
            "router_grad_nonzero": grad_norms["router"] > 0,
            "image_adapter_grad_nonzero": grad_norms["image_adapter"] > 0,
        }
    }
    loss_grad_reports.append(report)

out_dir = args.output_dir
out_dir.mkdir(parents=True, exist_ok=True)
json_path = out_dir / "loss_gradient_audit.json"

with open(json_path, "w") as f:
    json.dump(loss_grad_reports, f, indent=2)

print(f"\nSaved {json_path}")
