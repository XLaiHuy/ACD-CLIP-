#!/usr/bin/env python3
"""
Verify local branch functional strength on REAL Brain images.

Compares A2 (hard_anchor + dense local, experts off) vs A1 (same but rho=0, h6_logits=0)
on real images from the Brain val split to measure actual H6 local residual effect.

Outputs:
  - runs/phase4/p1_v8_evidence/local_branch_realimage_diagnostics.json
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import os
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.clip import create_model
from model.adapter import ACDCLIP
from model.checkpoint_utils import load_adapter_checkpoint, h6_config_from_checkpoint
from torchvision.transforms.functional import to_tensor

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "runs/phase4/progress1_v7_full_seed0_ready3/train/adapter_12.pth"
MANIFEST_PATH = "runs/phase4/progress1_v7_full_seed0_ready3/train/protocol/medical_manifests/Brain_val.jsonl"
BRAIN_DATA_ROOT = "/home/ai4/caohuy/data/MedAD/Brain_AD/test"
NUM_SAMPLES = 4

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
h6_kwargs = h6_config_from_checkpoint(checkpoint)
h6_kwargs = {f"h6_{k}": v for k, v in h6_kwargs.items()}

print("Loading CLIP base model (openai pretrained)...")
clip_model = create_model(
    model_name="ViT-L-14-336",
    img_size=518,
    device=device,
    pretrained="openai",
    require_pretrained=True,
)
clip_model.eval()

adapter_kwargs = {
    "dfg_mode": checkpoint.get("dfg_mode", "qkv"),
    "dfg_attn_dim": checkpoint.get("dfg_attn_dim", 256),
    "dfg_attn_tau": checkpoint.get("dfg_attn_tau", 8.0),
    "use_ss2d_dfg": checkpoint.get("use_ss2d_dfg", False),
    "dfg_gamma_max": checkpoint.get("dfg_gamma_max", 0.2),
    "dfg_ss2d_fusion": checkpoint.get("dfg_ss2d_fusion", "weight_residual"),
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

print("Building A2 model...")
model = ACDCLIP(
    clip_model=clip_model,
    **adapter_kwargs,
    **h6_kwargs
).to(device)

model.eval()
model.prompt_mode = "hard"
model.image_adapter.load_state_dict(checkpoint["image_adapter"])
model.text_adapter.load_state_dict(checkpoint["text_adapter"])
if "soft_prompt" in checkpoint:
    model.soft_prompt.load_state_dict(checkpoint["soft_prompt"])
model.h6.load_state_dict(checkpoint["h6_state_dict"], strict=False)
model.h6.set_epoch(12)

# ---------- Load real Brain images from manifest ----------
print("Loading real Brain images from manifest...")
with open(MANIFEST_PATH) as f:
    manifest = [json.loads(l) for l in f]

anomalous = [e for e in manifest if e["label"] == 1][:NUM_SAMPLES]
normal = [e for e in manifest if e["label"] == 0][:max(0, NUM_SAMPLES - len(anomalous))]
selected = anomalous + normal

# Find actual data root by trying different locations
for candidate_root in [
    "/home/ai4/caohuy/data/MedAD/Brain_AD/test",
    "/home/ai4/caohuy/ACD-CLIP-phase4/data/MedAD/Brain_AD/test",
    "/home/ai4/data/MedAD/Brain_AD/test",
]:
    if os.path.exists(os.path.join(candidate_root, selected[0]["image_path"])):
        BRAIN_DATA_ROOT = candidate_root
        print(f"Found data at: {BRAIN_DATA_ROOT}")
        break
else:
    print(f"WARNING: could not find Brain data")

from torchvision import transforms as tv_transforms
from torchvision.transforms import InterpolationMode
preprocess = tv_transforms.Compose([
    tv_transforms.Resize((518, 518), InterpolationMode.BICUBIC),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
])

images = []
masks = []
loaded_paths = []
for entry in selected:
    img_path = os.path.join(BRAIN_DATA_ROOT, entry["image_path"])
    if not os.path.exists(img_path):
        print(f"  SKIP: {img_path}")
        continue
    img = Image.open(img_path).convert("RGB")
    images.append(preprocess(img))
    if entry.get("mask_path") and entry["label"] == 1:
        mask_path = os.path.join(BRAIN_DATA_ROOT, entry["mask_path"])
        if os.path.exists(mask_path):
            m = Image.open(mask_path).convert("L")
            mt = to_tensor(m)  # [1, H, W], values in [0,1]
            mask_t = F.interpolate(mt.unsqueeze(0), size=(518, 518), mode='nearest').squeeze(0)  # [1,518,518]
            masks.append(mask_t)
        else:
            masks.append(torch.zeros(1, 518, 518))
    else:
        masks.append(torch.zeros(1, 518, 518))
    loaded_paths.append(img_path)

if len(images) == 0:
    raise RuntimeError("No real images could be loaded!")

images = torch.stack(images).to(device)
masks = torch.stack(masks).to(device)  # [B, 1, 518, 518]
print(f"Loaded {len(images)} real Brain images: {loaded_paths}")
print(f"  Anomalous (has mask): {sum(1 for m in masks if m.max() > 0)}")

dummy_dataset = "Brain"
dummy_classes = ["Brain"] * images.shape[0]

# ---------- A2 forward pass ----------
print("Running A2 forward pass (dense routing, hard_anchor, rho active)...")
with torch.no_grad():
    visual_output = model(images, return_phase4_features=True)
    seg_tokens = visual_output["seg_tokens"]
    seg_features = torch.stack(seg_tokens, dim=0)

    h6_batch_A2 = model.h6.build_batch(model, dummy_dataset, dummy_classes, visual_output, 0.0)
    rho_value = h6_batch_A2["rho"].mean().item()
    rho_per_sample = h6_batch_A2["rho"].flatten().tolist()

    seg_pred_A2 = model.vision_text_fusion_gate_seg(
        seg_features, h6_batch_A2["text_global"], test_mode=True, domain="Medical",
        h6_patch_logits=h6_batch_A2["h6_logits"]
    )

    # ---------- A1 (zero out local h6 contribution) ----------
    print("Simulating A1 (h6_logits=0, rho=0, no local residual)...")
    h6_batch_A1 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in h6_batch_A2.items()}
    h6_batch_A1["h6_logits"] = torch.zeros_like(h6_batch_A2["h6_logits"])
    if "rho" in h6_batch_A1:
        h6_batch_A1["rho"] = torch.zeros_like(h6_batch_A2["rho"])

    seg_pred_A1 = model.vision_text_fusion_gate_seg(
        seg_features, h6_batch_A1["text_global"], test_mode=True, domain="Medical",
        h6_patch_logits=h6_batch_A1["h6_logits"]
    )

# ---------- Compute differences ----------
B = images.shape[0]

# Upsample to 518x518 if needed
def upsample_preds(preds, target_size=(518, 518)):
    if preds.dim() == 3:  # [B, H, W]
        preds = preds.unsqueeze(1)  # [B,1,H,W]
    p = F.interpolate(preds.float(), size=target_size, mode='bilinear', align_corners=False)
    return p.squeeze(1)  # [B, H, W]

pred_A2_up = upsample_preds(seg_pred_A2)
pred_A1_up = upsample_preds(seg_pred_A1)
h6_effect = pred_A2_up - pred_A1_up  # signed residual
h6_effect_abs = h6_effect.abs()

inside_mask = masks.squeeze(1) > 0  # [B, H, W]
outside_mask = ~inside_mask
has_any_mask = inside_mask.any().item()

mean_abs_effect = h6_effect_abs.mean().item()
max_abs_effect = h6_effect_abs.max().item()

if has_any_mask:
    inside_effect_signed = h6_effect[inside_mask].mean().item()
    outside_effect_signed = h6_effect[outside_mask].mean().item()
    inside_effect_abs = h6_effect_abs[inside_mask].mean().item()
    outside_effect_abs = h6_effect_abs[outside_mask].mean().item()
    inside_vs_outside_ratio = inside_effect_abs / (outside_effect_abs + 1e-9)
else:
    inside_effect_signed = 0.0
    outside_effect_signed = h6_effect.mean().item()
    inside_effect_abs = 0.0
    outside_effect_abs = h6_effect_abs.mean().item()
    inside_vs_outside_ratio = None

# Effect ratio relative to base (A1) prediction magnitude
base_logit_abs = pred_A1_up.abs()
effect_ratio = h6_effect_abs / (base_logit_abs + 1e-6)
mean_effect_ratio = effect_ratio.mean().item()

# Prediction area change
pred_area_A1 = (seg_pred_A1 > 0).float().sum().item()
pred_area_A2 = (seg_pred_A2 > 0).float().sum().item()
pred_area_change = pred_area_A2 - pred_area_A1

# Per-image stats
per_image = []
for i in range(B):
    eff_i = h6_effect_abs[i]
    im_stat = {
        "image_path": loaded_paths[i] if i < len(loaded_paths) else f"image_{i}",
        "mean_abs_effect": eff_i.mean().item(),
        "max_abs_effect": eff_i.max().item(),
        "rho": rho_per_sample[i] if i < len(rho_per_sample) else rho_value,
        "has_mask": bool(inside_mask[i].any().item()),
    }
    if inside_mask[i].any():
        im_stat["inside_mask_signed_effect"] = h6_effect[i][inside_mask[i]].mean().item()
        im_stat["outside_mask_signed_effect"] = h6_effect[i][outside_mask[i]].mean().item()
    per_image.append(im_stat)

# ---------- Router diagnostics ----------
print("Computing router diagnostics on real images...")
with torch.no_grad():
    router_out = model.h6.router(visual_output["seg_tokens"], epoch_one_based=12)

probs = router_out["st_sparse_probabilities"]  # [G, B, P, M]
G_dim, B_dim, P_dim, M_dim = probs.shape

usage = probs.mean(dim=(1, 2))  # [G, M]
log_M = torch.log(torch.tensor(float(M_dim), device=device))
entropy = -(probs.clamp_min(1e-10) * probs.clamp_min(1e-10).log()).sum(dim=-1).mean(dim=(1, 2)) / log_M

concept_keys = router_out.get("concept_keys", None)
mean_cosine = None
if concept_keys is not None:
    ck = F.normalize(concept_keys.float(), dim=-1)
    if ck.dim() == 2:  # [M, D]
        cosine_sim = torch.einsum("md,nd->mn", ck, ck)
    else:  # [G, M, D]
        cosine_sim = torch.einsum("gmd,gnd->gmn", ck, ck).mean(0)
    offdiag = cosine_sim[~torch.eye(cosine_sim.shape[0], device=device, dtype=torch.bool)]
    mean_cosine = offdiag.mean().item()

pq_var = router_out.get("query_variance_across_patches", None)
pq_var_val = pq_var.mean().item() if pq_var is not None else None
factor_std = router_out.get("per_factor_logit_std_across_patches", None)
factor_std_val = factor_std.mean().item() if factor_std is not None else None

EFFECT_THRESHOLD = 1e-3
branch_is_live = mean_abs_effect > EFFECT_THRESHOLD

out = {
    "note": "real_Brain_images_from_val_manifest",
    "num_images": B,
    "images_with_gt_mask": sum(1 for im in per_image if im["has_mask"]),
    "rho_mean": rho_value,
    "rho_per_sample": rho_per_sample[:B],
    "local_h6_effect": {
        "mean_abs_effect": mean_abs_effect,
        "max_abs_effect": max_abs_effect,
        "mean_effect_ratio_to_base_logit": mean_effect_ratio,
        "branch_is_live": branch_is_live,
        "effect_threshold_used": EFFECT_THRESHOLD,
        "interpretation": (
            "LIVE: local residual has measurable effect on final prediction"
            if branch_is_live else
            "DEAD: local residual has negligible effect on final prediction"
        ),
    },
    "spatial_decomposition": {
        "has_gt_masks": bool(has_any_mask),
        "inside_mask_signed_effect": inside_effect_signed,
        "outside_mask_signed_effect": outside_effect_signed,
        "inside_mask_abs_effect": inside_effect_abs,
        "outside_mask_abs_effect": outside_effect_abs,
        "inside_vs_outside_abs_ratio": inside_vs_outside_ratio,
        "interpretation": (
            "local branch pushes anomaly regions MORE than normal regions"
            if inside_vs_outside_ratio is not None and inside_vs_outside_ratio > 1.0
            else "no preferential anomaly/normal separation from local branch"
        ),
    },
    "prediction_area": {
        "area_change_pixels": pred_area_change,
        "A1_positive_pixels": pred_area_A1,
        "A2_positive_pixels": pred_area_A2,
    },
    "router_diagnostics": {
        "G": G_dim,
        "M": M_dim,
        "per_factor_usage_per_group": usage.tolist(),
        "router_entropy_per_group": entropy.tolist(),
        "pairwise_factor_text_cosine": mean_cosine,
        "factor_patch_logit_std": factor_std_val,
        "patch_query_variation": pq_var_val,
    },
    "per_image_stats": per_image,
}

print(json.dumps(out, indent=2))

out_dir = Path("runs/phase4/p1_v8_evidence")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "local_branch_realimage_diagnostics.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {out_path}")
