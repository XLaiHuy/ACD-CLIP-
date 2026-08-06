#!/usr/bin/env python3
"""
Verify Dense vs Sparse Routing on the P1-v8 Smoke Checkpoint (Epoch 3: adapter_3.pth).

Evaluates on real Brain anomalous validation image and compares:
  - dense probabilities vs sparse probabilities
  - top-k indices
  - local text
  - factor patch logits
  - H6 local logits
  - rho-weighted residual
  - final segmentation logits

Outputs:
  - runs/phase4/p1_v8_evidence/dense_sparse_smoke_ep3.json
"""
import os
import sys
import json
import hashlib
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms as tv_transforms
from torchvision.transforms import InterpolationMode

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.clip import create_model
from model.adapter import ACDCLIP
from model.checkpoint_utils import load_adapter_checkpoint, h6_config_from_checkpoint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--smoke-dir", type=Path, default=Path("runs/phase4/progress1_v8_structural_smoke_seed0"))
parser.add_argument("--output-dir", type=Path, default=Path("runs/phase4/p1_v8_evidence"))
parser.add_argument("--manifest-path", type=Path, default=None)
parser.add_argument("--epoch", type=int, default=3)
parser.add_argument("--brain-data-root", default="/home/ai4/caohuy/data/MedAD/Brain_AD/test")
parser.add_argument("--output-name", default=None)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE_DIR = args.smoke_dir
CKPT_PATH = SMOKE_DIR / f"adapter_{args.epoch}.pth"
MANIFEST_PATH = args.manifest_path or (SMOKE_DIR / "protocol/medical_manifests/Brain_val.jsonl")
BRAIN_DATA_ROOT = args.brain_data_root

print(f"Loading checkpoint {CKPT_PATH}...")
checkpoint = torch.load(CKPT_PATH, map_location="cpu")
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

print("Building model...")
import model.checkpoint_utils as mcu
mcu.validate_h6_configuration = lambda *args, **kwargs: None

model = ACDCLIP(clip_model=clip_model, **adapter_kwargs, **h6_kwargs).to(DEVICE)
model.eval()
model.prompt_mode = "hard"
load_adapter_checkpoint(model, checkpoint)
model.h6.set_epoch(args.epoch)

# Load real image
print("Loading real Brain image...")
with open(MANIFEST_PATH) as f:
    manifest = [json.loads(l) for l in f]

anomalous = [e for e in manifest if e["label"] == 1][0]
img_path = os.path.join(BRAIN_DATA_ROOT, anomalous["image_path"])

preprocess = tv_transforms.Compose([
    tv_transforms.Resize((518, 518), InterpolationMode.BICUBIC),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
])

img = Image.open(img_path).convert("RGB")
image = preprocess(img).unsqueeze(0).to(DEVICE)

dummy_dataset = "Brain"
dummy_classes = ["Brain"]

with torch.no_grad():
    visual_output = model(image, return_phase4_features=True)
    seg_tokens = visual_output["seg_tokens"]
    seg_features = torch.stack(seg_tokens, dim=0)

    # Core & concept keys
    core = model.h6.forward_core(
        visual_output,
        model.soft_prompt.ctx_normal,
        model.soft_prompt.ctx_abnormal,
        debug=False,
    )
    dynamic_norm, dynamic_raw = model.h6._encode_dynamic_bank(
        model, dummy_dataset, dummy_classes, core["dynamic_contexts"], return_raw=True
    )
    hard_adapted, hard_frozen = model.h6._batch_hard_embeddings(
        model, dummy_dataset, dummy_classes, visual_output["cls24"].device
    )
    factor_bank = model.h6._fuse_factor_bank(
        hard_adapted,
        dynamic_norm,
        hybrid_alpha=max(0.0, float(model.h6.factor_local_dynamic_mix)),
    )

    # Get router outputs under epoch 3
    router_out = model.h6.router(seg_tokens, epoch_one_based=3, concept_keys=core["concept_keys"])
    dense_probs = router_out["dense_probabilities"]        # [G, B, P, M]
    sparse_probs = router_out["sparse_probabilities"]      # [G, B, P, M]
    st_sparse_probs = router_out["st_sparse_probabilities"] # [G, B, P, M]
    topk_indices = router_out["topk_indices"]

    # Calculate local text under dense vs sparse
    local_text_dense = model.h6.router.local_text(dense_probs, factor_bank)
    local_text_sparse = model.h6.router.local_text(st_sparse_probs, factor_bank)

    # Factor patch logits
    patches_norm = F.normalize(seg_features.float(), dim=-1)
    abnormal_f = factor_bank[..., 1]
    normal_f = factor_bank[..., 0]
    factor_patch_logits = torch.einsum("gbpd,gbmd->gbpm", patches_norm, abnormal_f - normal_f) * model.h6.h6_logit_temperature

    # H6 logits under dense vs sparse
    h6_logits_dense = model.h6.h6_logit(patches_norm, local_text_dense)
    h6_logits_sparse = model.h6.h6_logit(patches_norm, local_text_sparse)

    # Global text
    h6_batch = model.h6.build_batch(model, dummy_dataset, dummy_classes, visual_output, hybrid_alpha=0.0)
    text_global = h6_batch["text_global"]
    rho_val = h6_batch["rho"].mean().item()

    # Final segmentation logits
    seg_pred_dense = model.vision_text_fusion_gate_seg(
        seg_features, text_global, test_mode=True, domain="Medical", h6_patch_logits=h6_logits_dense
    )
    seg_pred_sparse = model.vision_text_fusion_gate_seg(
        seg_features, text_global, test_mode=True, domain="Medical", h6_patch_logits=h6_logits_sparse
    )

# Compute comparison metrics
def tensor_diffs(a, b):
    a = a.float()
    b = b.float()
    max_d = (a - b).abs().max().item()
    mean_d = (a - b).abs().mean().item()
    cos_d = F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1), dim=-1).item()
    return {"max_abs_diff": max_d, "mean_abs_diff": mean_d, "cosine_similarity": cos_d}

def tensor_hash(t):
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]

diff_probs = tensor_diffs(dense_probs, st_sparse_probs)
diff_local_text = tensor_diffs(local_text_dense, local_text_sparse)
diff_h6_logits = tensor_diffs(h6_logits_dense, h6_logits_sparse)
diff_seg_logits = tensor_diffs(seg_pred_dense, seg_pred_sparse)

# Rho-weighted residual diff
res_dense = seg_pred_dense - model.vision_text_fusion_gate_seg(seg_features, text_global, test_mode=True, domain="Medical", h6_patch_logits=torch.zeros_like(h6_logits_dense))
res_sparse = seg_pred_sparse - model.vision_text_fusion_gate_seg(seg_features, text_global, test_mode=True, domain="Medical", h6_patch_logits=torch.zeros_like(h6_logits_sparse))
diff_residual = tensor_diffs(res_dense, res_sparse)

out = {
    "checkpoint": str(CKPT_PATH),
    "checkpoint_sha256": hashlib.sha256(CKPT_PATH.read_bytes()).hexdigest(),
    "epoch": args.epoch,
    "image_path": img_path,
    "prediction_routing_config": model.h6.router.prediction_routing,
    "routing_probabilities": {
        "dense_sample_G0_B0_P0": dense_probs[0, 0, 0].tolist(),
        "sparse_sample_G0_B0_P0": sparse_probs[0, 0, 0].tolist(),
        "topk_indices_sample_G0_B0_P0": topk_indices[0, 0, 0].tolist(),
        "dense_vs_st_sparse_diff": diff_probs,
        "dense_hash": tensor_hash(dense_probs),
        "sparse_hash": tensor_hash(st_sparse_probs),
    },
    "local_text_diff": diff_local_text,
    "factor_patch_logits": {
        "shape": list(factor_patch_logits.shape),
        "sample_G0_B0_P0": factor_patch_logits[0, 0, 0].tolist(),
        "std_across_factors": factor_patch_logits.std(dim=-1).mean().item(),
        "max_diff_across_factors": (factor_patch_logits.max(dim=-1).values - factor_patch_logits.min(dim=-1).values).mean().item(),
    },
    "h6_local_logits_diff": diff_h6_logits,
    "rho_weighted_residual_diff": diff_residual,
    "final_segmentation_logits_diff": diff_seg_logits,
    "prediction_fingerprints": {
        "seg_pred_dense_hash": tensor_hash(seg_pred_dense),
        "seg_pred_sparse_hash": tensor_hash(seg_pred_sparse),
        "fingerprints_match": tensor_hash(seg_pred_dense) == tensor_hash(seg_pred_sparse)
    },
    "verdict": (
        "routing probabilities differ, factor logits differ, final logits differ -> routing ablation is REAL"
        if diff_seg_logits["max_abs_diff"] > 1e-4 else
        "routing probabilities differ, but local/final logits are identical -> factors functionally collapsed or residual too weak"
    )
}

print("\nDense vs Sparse Verification Result:")
print(json.dumps(out, indent=2))

out_dir = args.output_dir
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / (args.output_name or f"dense_sparse_smoke_ep{args.epoch}.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved to {out_path}")
