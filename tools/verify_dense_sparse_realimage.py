#!/usr/bin/env python3
"""
Correct dense vs sparse routing verification on real Brain images.

Key insight: prediction_routing is a MODEL config flag, not runtime.
A2 uses prediction_routing="dense" -> prediction_probabilities = dense_probabilities
A3 uses prediction_routing="scheduled_topk" -> at epoch 12, probs = st_sparse_probabilities

This script:
1. Loads a single model (A2 config: hard_anchor + dense routing + no experts)
2. Gets the router outputs: dense_probs, sparse_probs, st_sparse_probs, topk_indices
3. Computes what H6 logits WOULD be under dense (A2) vs sparse (A3) routing
4. Compares them to verify that the routing distinction is real and measurable

Outputs:
  - runs/phase4/p1_v8_evidence/dense_sparse_realimage_diff.json
"""
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import os
from PIL import Image
from torchvision import transforms as tv_transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.clip import create_model
from model.adapter import ACDCLIP
from model.checkpoint_utils import load_adapter_checkpoint, h6_config_from_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "runs/phase4/progress1_v7_full_seed0_ready3/train/adapter_12.pth"
MANIFEST_PATH = "runs/phase4/progress1_v7_full_seed0_ready3/train/protocol/medical_manifests/Brain_val.jsonl"
BRAIN_DATA_ROOT = "/home/ai4/caohuy/data/MedAD/Brain_AD/test"
NUM_SAMPLES = 1  # Keep to 1 to fit in GPU memory

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location="cpu")
h6_kwargs = h6_config_from_checkpoint(checkpoint)
h6_kwargs = {f"h6_{k}": v for k, v in h6_kwargs.items()}

print("Loading CLIP model...")
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

# A2 config: dense prediction routing
h6_kwargs["h6_prediction_routing"] = "dense"
h6_kwargs["h6_expert_enabled"] = False
h6_kwargs["h6_global_text_mode"] = "hard_anchor"
h6_kwargs["h6_progress"] = 1

print("Building model (A2 config: hard_anchor + dense routing + no experts)...")
model = ACDCLIP(clip_model=clip_model, **adapter_kwargs, **h6_kwargs).to(device)
model.eval()
model.prompt_mode = "hard"
model.image_adapter.load_state_dict(checkpoint["image_adapter"])
model.text_adapter.load_state_dict(checkpoint["text_adapter"])
if "soft_prompt" in checkpoint:
    model.soft_prompt.load_state_dict(checkpoint["soft_prompt"])
model.h6.load_state_dict(checkpoint["h6_state_dict"], strict=False)
model.h6.set_epoch(12)

# ---------- Load real images ----------
print("Loading real Brain images...")
with open(MANIFEST_PATH) as f:
    manifest = [json.loads(l) for l in f]

anomalous = [e for e in manifest if e["label"] == 1][:NUM_SAMPLES]
preprocess = tv_transforms.Compose([
    tv_transforms.Resize((518, 518), InterpolationMode.BICUBIC),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
])

images = []
masks = []
loaded_paths = []
for entry in anomalous:
    img_path = os.path.join(BRAIN_DATA_ROOT, entry["image_path"])
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path).convert("RGB")
    images.append(preprocess(img))
    if entry.get("mask_path"):
        mask_path = os.path.join(BRAIN_DATA_ROOT, entry["mask_path"])
        if os.path.exists(mask_path):
            m = Image.open(mask_path).convert("L")
            mt = to_tensor(m)
            mask_t = F.interpolate(mt.unsqueeze(0), size=(518, 518), mode='nearest').squeeze(0)
            masks.append(mask_t)
        else:
            masks.append(torch.zeros(1, 518, 518))
    else:
        masks.append(torch.zeros(1, 518, 518))
    loaded_paths.append(img_path)

assert len(images) > 0, "No images loaded!"
images = torch.stack(images).to(device)
masks = torch.stack(masks).to(device)
B = images.shape[0]
print(f"Loaded {B} real images: {loaded_paths}")

dummy_dataset = "Brain"
dummy_classes = ["Brain"] * B

# ---------- Get router outputs ----------
print("Extracting visual features and router outputs...")
with torch.no_grad():
    visual_output = model(images, return_phase4_features=True)
    seg_tokens = visual_output["seg_tokens"]
    seg_features = torch.stack(seg_tokens, dim=0)

    # Get ALL router probability distributions at epoch 12
    router_out = model.h6.router(seg_tokens, epoch_one_based=12)

    dense_probs = router_out["dense_probabilities"]        # [G, B, P, M] — softmax over all M
    sparse_probs = router_out["sparse_probabilities"]      # [G, B, P, M] — softmax over top-K only
    st_sparse_probs = router_out["st_sparse_probabilities"] # STE version
    prediction_probs = router_out["prediction_probabilities"]  # what the model actually uses
    topk_indices = router_out["topk_indices"]              # [G, B, P, K]

    G, _, P, M = dense_probs.shape
    K = topk_indices.shape[-1]

    print(f"Router shapes: dense_probs={dense_probs.shape}, topk_indices={topk_indices.shape}")
    print(f"sparse_ratio at epoch 12: {router_out['sparse_ratio'].item():.3f}")
    print(f"prediction_routing config: {model.h6.router.prediction_routing}")
    print(f"Dense probs sample [G=0, B=0, P=0]: {dense_probs[0,0,0].tolist()}")
    print(f"Sparse probs sample [G=0, B=0, P=0]: {sparse_probs[0,0,0].tolist()}")
    print(f"Top-K indices [G=0, B=0, P=0]: {topk_indices[0,0,0].tolist()}")

    # Factor bank for computing h6_logits under dense vs sparse
    h6_batch = model.h6.build_batch(
        model, dummy_dataset, dummy_classes, visual_output, 0.0, update_load_bias=False
    )
    active_factor_bank = h6_batch["active_factor_bank"]

    # Compute H6 logits under A2 (dense prediction_probs)
    local_text_dense = model.h6.router.local_text(dense_probs, active_factor_bank)
    patches_stacked = torch.stack(seg_tokens, dim=0)  # [G, B, P, D]
    # Normalize patches
    patches_norm = F.normalize(patches_stacked.float(), dim=-1)
    # h6_logit uses per-patch logit across all groups
    # We need to compute h6_logit with each version of local_text

    # local_text: [G, B, P, D_bank]
    # h6_logit: calls self.h6_logit(patches, local_text) which does:
    #   dot(patch, abnormal_local_text) - dot(patch, normal_local_text)
    h6_logits_dense = model.h6.h6_logit(patches_norm, local_text_dense)

    # Compute H6 logits under A3 (st_sparse prediction_probs)
    local_text_sparse = model.h6.router.local_text(st_sparse_probs, active_factor_bank)
    h6_logits_sparse = model.h6.h6_logit(patches_norm, local_text_sparse)

    # Compare
    h6_diff_max = (h6_logits_dense - h6_logits_sparse).abs().max().item()
    h6_diff_mean = (h6_logits_dense - h6_logits_sparse).abs().mean().item()
    h6_cos = F.cosine_similarity(
        h6_logits_dense.reshape(1, -1), h6_logits_sparse.reshape(1, -1), dim=-1
    ).item()

    print(f"\nH6 logits: dense range=[{h6_logits_dense.min().item():.4f}, {h6_logits_dense.max().item():.4f}]")
    print(f"H6 logits: sparse range=[{h6_logits_sparse.min().item():.4f}, {h6_logits_sparse.max().item():.4f}]")
    print(f"H6 logits diff: max={h6_diff_max:.6f}, mean={h6_diff_mean:.6f}, cos={h6_cos:.6f}")

    # Final seg logits comparison
    seg_pred_dense = model.vision_text_fusion_gate_seg(
        seg_features, h6_batch["text_global"], test_mode=True, domain="Medical",
        h6_patch_logits=h6_logits_dense
    )
    seg_pred_sparse = model.vision_text_fusion_gate_seg(
        seg_features, h6_batch["text_global"], test_mode=True, domain="Medical",
        h6_patch_logits=h6_logits_sparse
    )

    final_diff_max = (seg_pred_dense - seg_pred_sparse).abs().max().item()
    final_diff_mean = (seg_pred_dense - seg_pred_sparse).abs().mean().item()
    final_cos = F.cosine_similarity(
        seg_pred_dense.reshape(1, -1).float(), seg_pred_sparse.reshape(1, -1).float(), dim=-1
    ).item()

    # Spatial decomposition
    inside_mask = masks.squeeze(1) > 0
    outside_mask = ~inside_mask

    def upsample(t, size=(518, 518)):
        if t.dim() == 3:
            t = t.unsqueeze(1)
        return F.interpolate(t.float(), size=size, mode='bilinear', align_corners=False).squeeze(1)

    diff_up = upsample((seg_pred_dense - seg_pred_sparse).abs())
    has_mask = inside_mask.any().item()
    inside_diff = diff_up[inside_mask].mean().item() if has_mask else 0.0
    outside_diff = diff_up[outside_mask].mean().item() if outside_mask.any() else 0.0

    # Prob distribution metrics
    prob_diff_max = (dense_probs - sparse_probs).abs().max().item()
    prob_diff_mean = (dense_probs - sparse_probs).abs().mean().item()

    # Usage statistics
    dense_usage_per_group = [dense_probs[g].mean(dim=(0, 1)).tolist() for g in range(G)]
    sparse_usage_per_group = [sparse_probs[g].mean(dim=(0, 1)).tolist() for g in range(G)]

    # Is dense/sparse the same in terms of top-k selection?
    dense_topk_check, _ = torch.topk(dense_probs, k=K, dim=-1)
    # The top-k indices should be the same as what router computed (they use same logits)
    topk_same_as_dense_topk = True  # By construction (same logits -> same top-K)

    DISTINCTNESS_THRESHOLD = 1e-3
    routing_produces_distinct_h6 = h6_diff_max > DISTINCTNESS_THRESHOLD

result = {
    "note": "single_model_analysis_A2_dense_vs_A3_sparse_prediction_probs",
    "num_images": B,
    "model_prediction_routing_config": model.h6.router.prediction_routing,
    "sparse_ratio_at_epoch_12": router_out["sparse_ratio"].item(),
    "routing_epoch": 12,
    "architecture_note": (
        "prediction_routing='dense' means prediction_probs=dense_probs always. "
        "A3 uses prediction_routing='scheduled_topk' which at epoch 12 gives prediction_probs=st_sparse_probs. "
        "We compare what h6_logits WOULD be under each probability distribution."
    ),
    "routing_probabilities": {
        "shape_GBPM": [G, B, P, M],
        "max_prob_diff_dense_vs_sparse": prob_diff_max,
        "mean_prob_diff_dense_vs_sparse": prob_diff_mean,
        "dense_factor_usage_per_group": dense_usage_per_group,
        "sparse_factor_usage_per_group": sparse_usage_per_group,
        "dense_sample_G0_B0_P0": dense_probs[0, 0, 0].tolist(),
        "sparse_sample_G0_B0_P0": sparse_probs[0, 0, 0].tolist(),
        "topk_indices_sample_G0_B0_P0": topk_indices[0, 0, 0].tolist(),
    },
    "h6_logits": {
        "shape": list(h6_logits_dense.shape),
        "max_abs_diff_dense_vs_sparse_h6_logits": h6_diff_max,
        "mean_abs_diff_dense_vs_sparse_h6_logits": h6_diff_mean,
        "cosine_similarity": h6_cos,
        "routing_produces_distinct_h6_logits": routing_produces_distinct_h6,
        "interpretation": (
            "Dense and sparse routing produce DIFFERENT h6_logits -> routing distinction is real"
            if routing_produces_distinct_h6 else
            "Dense and sparse routing produce IDENTICAL h6_logits -> investigate further"
        ),
    },
    "final_seg_logits": {
        "max_abs_diff": final_diff_max,
        "mean_abs_diff": final_diff_mean,
        "cosine_similarity": final_cos,
    },
    "spatial_decomposition": {
        "has_gt_masks": bool(has_mask),
        "inside_mask_mean_abs_diff": inside_diff,
        "outside_mask_mean_abs_diff": outside_diff,
        "inside_vs_outside_ratio": inside_diff / (outside_diff + 1e-9) if outside_diff > 0 else None,
    },
}

print("\nResult:")
print(json.dumps(result, indent=2))

out_dir = Path("runs/phase4/p1_v8_evidence")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "dense_sparse_realimage_diff.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {out_path}")
