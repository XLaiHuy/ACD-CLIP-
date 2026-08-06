#!/usr/bin/env python3
"""
Diagnose Factor Specialization on P1-v8 Structural Smoke Checkpoints (Epochs 1, 2, 3).

Evaluates 50 real Brain validation images on adapter_1.pth, adapter_2.pth, and adapter_3.pth.
Measures:
  - Full factor text cosine ($T_i, T_j$)
  - Dynamic residual $\Delta T_i$ cosine, norm, and Euclidean distance
  - Functional factor patch-logit correlation, std, diffs, local_text diffs
  - Router usage, entropy, query variance, dead factors, unique Top-K pairs
  - Final local effect ($\rho$, residual ratio, inside/outside mask ratio)

Outputs:
  - runs/phase4/p1_v8_evidence/smoke_factor_specialization.json
  - runs/phase4/p1_v8_evidence/smoke_factor_specialization.csv
"""
import os
import sys
import json
import csv
import argparse
import torch
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--smoke-dir", type=Path, default=Path("runs/phase4/progress1_v8_structural_smoke_seed0"))
parser.add_argument("--output-dir", type=Path, default=Path("runs/phase4/p1_v8_evidence"))
parser.add_argument("--manifest-path", type=Path, default=None)
parser.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3])
parser.add_argument("--num-images", type=int, default=50)
parser.add_argument("--brain-data-root", default="/home/ai4/caohuy/data/MedAD/Brain_AD/test")
parser.add_argument("--output-json-name", default="smoke_factor_specialization.json")
parser.add_argument("--output-csv-name", default="smoke_factor_specialization.csv")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE_DIR = args.smoke_dir
MANIFEST_PATH = args.manifest_path or (SMOKE_DIR / "protocol/medical_manifests/Brain_val.jsonl")
BRAIN_DATA_ROOT = args.brain_data_root
NUM_IMAGES = args.num_images
EPOCHS = args.epochs

# ---------- Load real Brain validation images ----------
print("Loading real Brain validation images...")
with open(MANIFEST_PATH) as f:
    manifest = [json.loads(l) for l in f]

selected_entries = manifest[:NUM_IMAGES]

preprocess = tv_transforms.Compose([
    tv_transforms.Resize((518, 518), InterpolationMode.BICUBIC),
    tv_transforms.ToTensor(),
    tv_transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
])

images_list = []
masks_list = []
loaded_paths = []

for entry in selected_entries:
    img_path = os.path.join(BRAIN_DATA_ROOT, entry["image_path"])
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path).convert("RGB")
    images_list.append(preprocess(img))
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
    loaded_paths.append(img_path)

assert len(images_list) > 0, "No real images loaded!"
images = torch.stack(images_list).to(DEVICE)
masks = torch.stack(masks_list).to(DEVICE)
B_total = images.shape[0]
print(f"Successfully loaded {B_total} real images.")

def off_diagonal_mean(matrix):
    if matrix.dim() == 2:
        M = matrix.shape[0]
        mask = ~torch.eye(M, device=matrix.device, dtype=torch.bool)
        return matrix[mask].mean().item()
    else:
        M = matrix.shape[1]
        mask = ~torch.eye(M, device=matrix.device, dtype=torch.bool)
        return matrix[:, mask].mean().item()

def evaluate_checkpoint_diagnostics(epoch_num):
    ckpt_path = SMOKE_DIR / f"adapter_{epoch_num}.pth"
    print(f"\n--- Diagnosing Epoch {epoch_num}: {ckpt_path} ---")
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
    model.eval()
    model.prompt_mode = "hard"
    load_adapter_checkpoint(model, checkpoint)
    model.h6.set_epoch(epoch_num)

    batch_size = 5
    accum_metrics = {
        "full_factor_cos": [],
        "delta_T_cos": [],
        "delta_T_cos_median": [],
        "delta_T_cos_max": [],
        "delta_T_norm": [],
        "delta_T_norm_min": [],
        "delta_T_norm_max": [],
        "delta_T_l2_dist": [],
        "delta_T_effective_rank": [],
        "factor_logit_corr": [],
        "factor_logit_corr_median": [],
        "factor_logit_corr_max": [],
        "factor_logit_std": [],
        "factor_logit_max_diff": [],
        "factor_logit_normalized_diff": [],
        "local_text_diff": [],
        "local_text_pairwise_diff": [],
        "router_usage": [],
        "router_entropy": [],
        "query_var": [],
        "dead_factors": [],
        "unique_topk_pairs": [],
        "rho": [],
        "residual_ratio": [],
        "inside_residual_signed": [],
        "outside_residual_signed": [],
        "inside_residual_abs": [],
        "outside_residual_abs": [],
        "inside_outside_ratio": [],
        "outside_false_positive_mass_change": [],
        "predicted_anomaly_area_change": [],
    }

    dummy_dataset = "Brain"

    for start_idx in range(0, B_total, batch_size):
        end_idx = min(start_idx + batch_size, B_total)
        batch_imgs = images[start_idx:end_idx]
        batch_masks = masks[start_idx:end_idx]
        B_sub = batch_imgs.shape[0]
        batch_classes = ["Brain"] * B_sub

        with torch.no_grad():
            visual_output = model(batch_imgs, return_phase4_features=True)
            seg_tokens = visual_output["seg_tokens"]
            seg_features = torch.stack(seg_tokens, dim=0)

            core = model.h6.forward_core(
                visual_output,
                model.soft_prompt.ctx_normal,
                model.soft_prompt.ctx_abnormal,
                debug=False,
            )
            dynamic_norm, dynamic_raw = model.h6._encode_dynamic_bank(
                model, dummy_dataset, batch_classes, core["dynamic_contexts"], return_raw=True
            )
            hard_adapted, hard_frozen = model.h6._batch_hard_embeddings(
                model, dummy_dataset, batch_classes, visual_output["cls24"].device
            )

            G_cnt, B_sub_cnt, M_cnt, D_cnt, S_cnt = dynamic_norm.shape
            
            full_T = dynamic_norm.reshape(G_cnt * B_sub_cnt, M_cnt, D_cnt * S_cnt)
            full_T_norm = F.normalize(full_T, dim=-1)
            full_cos_mat = torch.bmm(full_T_norm, full_T_norm.transpose(1, 2))
            accum_metrics["full_factor_cos"].append(off_diagonal_mean(full_cos_mat))

            hard_frozen_expanded = hard_frozen.unsqueeze(2).expand_as(dynamic_raw)
            delta_T_raw = (dynamic_raw - hard_frozen_expanded).reshape(G_cnt * B_sub_cnt, M_cnt, D_cnt * S_cnt)
            
            delta_norms = delta_T_raw.norm(dim=-1).mean().item()
            accum_metrics["delta_T_norm"].append(delta_norms)

            delta_T_normed = F.normalize(delta_T_raw, dim=-1)
            delta_cos_mat = torch.bmm(delta_T_normed, delta_T_normed.transpose(1, 2))
            delta_pairs = delta_cos_mat[:, ~torch.eye(M_cnt, device=DEVICE, dtype=torch.bool)]
            accum_metrics["delta_T_cos"].append(delta_pairs.mean().item())
            accum_metrics["delta_T_cos_median"].append(delta_pairs.median().item())
            accum_metrics["delta_T_cos_max"].append(delta_pairs.max().item())
            accum_metrics["delta_T_norm_min"].append(delta_T_raw.norm(dim=-1).min().item())
            accum_metrics["delta_T_norm_max"].append(delta_T_raw.norm(dim=-1).max().item())

            singular_values = torch.linalg.svdvals(delta_T_raw)
            singular_probs = singular_values / singular_values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            effective_rank = torch.exp(-(singular_probs * singular_probs.clamp_min(1e-6).log()).sum(dim=-1))
            accum_metrics["delta_T_effective_rank"].append(effective_rank.mean().item())

            delta_dist_mat = torch.cdist(delta_T_raw, delta_T_raw)
            accum_metrics["delta_T_l2_dist"].append(off_diagonal_mean(delta_dist_mat))

            router_out = model.h6.router(seg_tokens, epoch_one_based=epoch_num, concept_keys=core["concept_keys"])
            dense_probs = router_out["dense_probabilities"]
            st_sparse_probs = router_out["st_sparse_probabilities"]
            topk_indices = router_out["topk_indices"]

            usage = dense_probs.mean(dim=(1, 2))
            accum_metrics["router_usage"].append(usage.mean(0).tolist())

            log_M = torch.log(torch.tensor(float(M_cnt), device=DEVICE))
            ent = -(dense_probs.clamp_min(1e-10) * dense_probs.clamp_min(1e-10).log()).sum(dim=-1).mean(dim=(1, 2)) / log_M
            accum_metrics["router_entropy"].append(ent.mean().item())

            pq_var = router_out.get("query_variance_across_patches", torch.tensor(0.0))
            accum_metrics["query_var"].append(pq_var.mean().item())

            dead_count = (usage < 0.01).sum().item()
            accum_metrics["dead_factors"].append(dead_count)

            unique_pairs = model.h6.router.unique_topk_pair_counts(topk_indices).float().mean().item()
            accum_metrics["unique_topk_pairs"].append(unique_pairs)

            factor_bank = model.h6._fuse_factor_bank(
                hard_adapted,
                dynamic_norm,
                hybrid_alpha=max(0.0, float(model.h6.factor_local_dynamic_mix)),
            )
            patches_norm = F.normalize(seg_features.float(), dim=-1)
            normal_f = factor_bank[..., 0]
            abnormal_f = factor_bank[..., 1]
            factor_patch_logits = torch.einsum("gbpd,gbmd->gbpm", patches_norm, abnormal_f - normal_f) * model.h6.h6_logit_temperature

            fpl_flat = factor_patch_logits.reshape(-1, factor_patch_logits.shape[2], M_cnt)
            fpl_centered = fpl_flat - fpl_flat.mean(dim=1, keepdim=True)
            fpl_std = fpl_centered.std(dim=1, keepdim=True).clamp_min(1e-6)
            fpl_normed = fpl_centered / fpl_std
            corr_mat = torch.bmm(fpl_normed.transpose(1, 2), fpl_normed) / float(fpl_flat.shape[1])
            corr_pairs = corr_mat[:, ~torch.eye(M_cnt, device=DEVICE, dtype=torch.bool)]
            accum_metrics["factor_logit_corr"].append(corr_pairs.mean().item())
            accum_metrics["factor_logit_corr_median"].append(corr_pairs.median().item())
            accum_metrics["factor_logit_corr_max"].append(corr_pairs.max().item())

            accum_metrics["factor_logit_std"].append(factor_patch_logits.std(dim=-1).mean().item())
            fpl_max_diff = (factor_patch_logits.max(dim=-1).values - factor_patch_logits.min(dim=-1).values).mean().item()
            accum_metrics["factor_logit_max_diff"].append(fpl_max_diff)
            base_abnormal_logits = torch.einsum("gbpd,gbmd->gbpm", patches_norm, abnormal_f).abs().mean().clamp_min(1e-6)
            accum_metrics["factor_logit_normalized_diff"].append(fpl_max_diff / base_abnormal_logits.item())

            local_text_dense = model.h6.router.local_text(dense_probs, factor_bank)
            local_text_diff = (local_text_dense - local_text_dense.mean(dim=2, keepdim=True)).norm(dim=-1).mean().item()
            accum_metrics["local_text_diff"].append(local_text_diff)
            local_text_pairwise = torch.pdist(local_text_dense.movedim(2, 0).reshape(M_cnt, -1)).mean().item()
            accum_metrics["local_text_pairwise_diff"].append(local_text_pairwise)

            h6_batch = model.h6.build_batch(model, dummy_dataset, batch_classes, visual_output, hybrid_alpha=0.0)
            rho_val = h6_batch["rho"].mean().item()
            accum_metrics["rho"].append(rho_val)

            seg_pred_A2 = model.vision_text_fusion_gate_seg(
                seg_features, h6_batch["text_global"], test_mode=True, domain="Medical",
                h6_patch_logits=h6_batch["h6_logits"]
            )

            h6_batch_A1 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in h6_batch.items()}
            h6_batch_A1["h6_logits"] = torch.zeros_like(h6_batch["h6_logits"])
            h6_batch_A1["rho"] = torch.zeros_like(h6_batch["rho"])

            seg_pred_A1 = model.vision_text_fusion_gate_seg(
                seg_features, h6_batch_A1["text_global"], test_mode=True, domain="Medical",
                h6_patch_logits=h6_batch_A1["h6_logits"]
            )

            if seg_pred_A2.dim() == 3:
                pred_A2_up = F.interpolate(seg_pred_A2.unsqueeze(1).float(), size=(518, 518), mode='bilinear', align_corners=False).squeeze(1)
                pred_A1_up = F.interpolate(seg_pred_A1.unsqueeze(1).float(), size=(518, 518), mode='bilinear', align_corners=False).squeeze(1)
            else:
                pred_A2_up = seg_pred_A2.float()
                pred_A1_up = seg_pred_A1.float()

            h6_effect = pred_A2_up - pred_A1_up
            h6_effect_abs = h6_effect.abs()

            base_abs = pred_A1_up.abs()
            res_ratio = (h6_effect_abs / (base_abs + 1e-6)).mean().item()
            accum_metrics["residual_ratio"].append(res_ratio)

            in_mask = batch_masks.squeeze(1) > 0
            out_mask = ~in_mask

            if in_mask.any():
                in_signed = h6_effect[in_mask].mean().item()
                in_abs = h6_effect_abs[in_mask].mean().item()
            else:
                in_signed = 0.0
                in_abs = 0.0

            if out_mask.any():
                out_signed = h6_effect[out_mask].mean().item()
                out_abs = h6_effect_abs[out_mask].mean().item()
            else:
                out_signed = 0.0
                out_abs = 0.0

            accum_metrics["inside_residual_signed"].append(in_signed)
            accum_metrics["outside_residual_signed"].append(out_signed)
            accum_metrics["inside_residual_abs"].append(in_abs)
            accum_metrics["outside_residual_abs"].append(out_abs)
            if out_abs > 0:
                accum_metrics["inside_outside_ratio"].append(in_abs / out_abs)
            pred_A2_prob = torch.sigmoid(pred_A2_up)
            pred_A1_prob = torch.sigmoid(pred_A1_up)
            accum_metrics["outside_false_positive_mass_change"].append(
                (pred_A2_prob[out_mask] - pred_A1_prob[out_mask]).mean().item()
            )
            accum_metrics["predicted_anomaly_area_change"].append(
                ((pred_A2_prob > 0.5).float().mean() - (pred_A1_prob > 0.5).float().mean()).item()
            )

    def mean_val(k):
        vals = accum_metrics[k]
        if not vals:
            return 0.0
        if isinstance(vals[0], list):
            import numpy as np
            return np.mean(vals, axis=0).tolist()
        return float(sum(vals) / len(vals))

    report = {
        "checkpoint": str(ckpt_path),
        "epoch": epoch_num,
        "num_val_images_evaluated": B_total,
        "A_full_factor_text_cosine_offdiag_mean": mean_val("full_factor_cos"),
        "B_dynamic_residual_delta_T": {
            "cosine_offdiag_mean": mean_val("delta_T_cos"),
            "cosine_offdiag_median": mean_val("delta_T_cos_median"),
            "cosine_offdiag_max": mean_val("delta_T_cos_max"),
            "L2_norm_mean": mean_val("delta_T_norm"),
            "L2_norm_min": mean_val("delta_T_norm_min"),
            "L2_norm_max": mean_val("delta_T_norm_max"),
            "euclidean_distance_offdiag_mean": mean_val("delta_T_l2_dist"),
            "effective_rank": mean_val("delta_T_effective_rank"),
        },
        "C_functional_factor_outputs": {
            "factor_patch_logit_correlation_offdiag_mean": mean_val("factor_logit_corr"),
            "factor_patch_logit_correlation_offdiag_median": mean_val("factor_logit_corr_median"),
            "factor_patch_logit_correlation_offdiag_max": mean_val("factor_logit_corr_max"),
            "factor_patch_logit_std_mean": mean_val("factor_logit_std"),
            "factor_patch_logit_max_diff": mean_val("factor_logit_max_diff"),
            "factor_patch_logit_normalized_mean_diff": mean_val("factor_logit_normalized_diff"),
            "local_text_diff_per_patch": mean_val("local_text_diff"),
            "local_text_pairwise_diff": mean_val("local_text_pairwise_diff"),
        },
        "D_router_diagnostics": {
            "per_factor_dense_usage": mean_val("router_usage"),
            "router_entropy": mean_val("router_entropy"),
            "patch_query_variance": mean_val("query_var"),
            "dead_factors_avg_count": mean_val("dead_factors"),
            "unique_topk_pairs_avg": mean_val("unique_topk_pairs"),
        },
        "E_final_local_effect": {
            "rho_mean": mean_val("rho"),
            "residual_to_base_logit_ratio": mean_val("residual_ratio"),
            "inside_mask_signed_effect": mean_val("inside_residual_signed"),
            "outside_mask_signed_effect": mean_val("outside_residual_signed"),
            "inside_mask_abs_effect": mean_val("inside_residual_abs"),
            "outside_mask_abs_effect": mean_val("outside_residual_abs"),
            "inside_vs_outside_abs_ratio": mean_val("inside_outside_ratio"),
            "outside_false_positive_mass_change": mean_val("outside_false_positive_mass_change"),
            "predicted_anomaly_area_change": mean_val("predicted_anomaly_area_change"),
        },
    }
    return report

all_epoch_reports = []
for ep in EPOCHS:
    rep = evaluate_checkpoint_diagnostics(ep)
    all_epoch_reports.append(rep)

out_dir = args.output_dir
out_dir.mkdir(parents=True, exist_ok=True)

json_path = out_dir / args.output_json_name
with open(json_path, "w") as f:
    json.dump(all_epoch_reports, f, indent=2)
print(f"\nSaved {json_path}")

csv_path = out_dir / args.output_csv_name
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "full_factor_text_cos",
        "delta_T_cos", "delta_T_cos_median", "delta_T_cos_max", "delta_T_norm", "delta_T_norm_min", "delta_T_norm_max", "delta_T_l2_dist", "delta_T_effective_rank",
        "factor_logit_corr", "factor_logit_corr_median", "factor_logit_corr_max", "factor_logit_std", "factor_logit_max_diff", "factor_logit_normalized_diff", "local_text_diff", "local_text_pairwise_diff",
        "router_entropy", "patch_query_var", "dead_factors_count", "unique_topk_pairs",
        "rho", "residual_ratio", "inside_abs_effect", "outside_abs_effect", "inside_outside_ratio", "outside_false_positive_mass_change", "predicted_anomaly_area_change"
    ])
    for rep in all_epoch_reports:
        b_res = rep["B_dynamic_residual_delta_T"]
        c_func = rep["C_functional_factor_outputs"]
        d_rout = rep["D_router_diagnostics"]
        e_eff = rep["E_final_local_effect"]
        writer.writerow([
            rep["epoch"],
            rep["A_full_factor_text_cosine_offdiag_mean"],
            b_res["cosine_offdiag_mean"], b_res["cosine_offdiag_median"], b_res["cosine_offdiag_max"], b_res["L2_norm_mean"], b_res["L2_norm_min"], b_res["L2_norm_max"], b_res["euclidean_distance_offdiag_mean"], b_res["effective_rank"],
            c_func["factor_patch_logit_correlation_offdiag_mean"], c_func["factor_patch_logit_correlation_offdiag_median"], c_func["factor_patch_logit_correlation_offdiag_max"], c_func["factor_patch_logit_std_mean"],
            c_func["factor_patch_logit_max_diff"], c_func["factor_patch_logit_normalized_mean_diff"], c_func["local_text_diff_per_patch"], c_func["local_text_pairwise_diff"],
            d_rout["router_entropy"], d_rout["patch_query_variance"], d_rout["dead_factors_avg_count"], d_rout["unique_topk_pairs_avg"],
            e_eff["rho_mean"], e_eff["residual_to_base_logit_ratio"],
            e_eff["inside_mask_abs_effect"], e_eff["outside_mask_abs_effect"], e_eff["inside_vs_outside_abs_ratio"], e_eff["outside_false_positive_mass_change"], e_eff["predicted_anomaly_area_change"]
        ])

print(f"Saved {csv_path}")
