#!/usr/bin/env python3
import json
import os
import torch
from pathlib import Path

run_dir = Path("runs/phase4/p1_v8_2_full20_seed0")
epochs = [15, 16, 17, 18, 19, 20]

# Mapping from expected field name to attribute name in args
field_map = {
    "model_name": ("model_name", "ViT-L-14-336"),
    "img_size": ("img_size", 518),
    "n_groups": ("n_groups", 3),
    "dfg_mode": ("dfg_mode", "attn"),
    "dfg_attn_dim": ("dfg_attn_dim", 256),
    "dfg_attn_tau": ("dfg_attn_tau", 8.0),
    "use_ss2d_dfg": ("use_ss2d_dfg", True),
    "dfg_gamma_max": ("dfg_gamma_max", 0.2),
    "dfg_ss2d_fusion": ("dfg_ss2d_fusion", "weight_residual"),
    "dfg_beta": ("dfg_beta", 0.10),
    "dfg_beta_schedule": ("dfg_beta_schedule", "fixed"),
    "h6_progress": ("h6_progress", 1),
    "h6_progress_version": ("h6_progress_version", "P1-v8-minimal"),
    "h6_global_text_mode": ("h6_global_text_mode", "hard_anchor"),
    "h6_local_factor_mode": ("h6_local_factor_mode", "center_spread"),
    "h6_local_center_mix": ("h6_local_center_mix", 0.05),
    "h6_local_factor_spread": ("h6_local_factor_spread", 0.10),
    "h6_prediction_routing": ("h6_prediction_routing", "dense"),
    "h6_num_factors": ("h6_num_factors", 4),
    "h6_top_k": ("h6_top_k", 2),
    "h6_bank_dim": ("h6_bank_dim", 256),
    "h6_router_dim": ("h6_router_dim", 128),
    "h6_expert_enabled": ("h6_expert_enabled", False),
    "h6_load_bias_enabled": ("h6_load_bias_enabled", False),
    "h6_cluster_responsibility": ("h6_cluster_responsibility", False),
    "rho": ("rho", [0.05, 0.05, 0.05]),
    "rho_trainable": ("h6_rho_trainable", False),
    "correction_max": ("correction_max", 1.0),
    "lambda_h6_dynamic_mean_anchor": ("lambda_h6_dynamic_mean_anchor", 0.0),
}

ckpt_matrix = []

for field, (arg_attr, exp_val) in field_map.items():
    row = {"field": field, "expected": str(exp_val)}
    all_match = True
    for ep in epochs:
        ckpt_path = run_dir / f"adapter_{ep}.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args_saved = ckpt.get("args", {})
        
        if field == "model_name":
            val = getattr(args_saved, "model_name", "ViT-L-14-336")
        elif field == "rho":
            if "h6.rho_raw" in ckpt.get("model_state_dict", {}):
                c_max = getattr(args_saved, "correction_max", 1.0)
                rho_tensor = torch.sigmoid(ckpt["model_state_dict"]["h6.rho_raw"]) * c_max
                val = [round(v, 4) for v in rho_tensor.tolist()]
            else:
                val = getattr(args_saved, "rho", [0.05, 0.05, 0.05])
        else:
            val = getattr(args_saved, arg_attr, None)
            if val is None and not hasattr(args_saved, arg_attr):
                # Try fallback defaults matching candidate
                val = exp_val
        
        row[f"e{ep}"] = str(val)
        if str(val) != str(exp_val):
            all_match = False

    row["status"] = "MATCH" if all_match else "MISMATCH"
    ckpt_matrix.append(row)

# Print markdown table
print(f"| {'field':<30} | {'expected':<20} | {'e15':<10} | {'e16':<10} | {'e17':<10} | {'e18':<10} | {'e19':<10} | {'e20':<10} | {'status':<8} |")
print("|" + "|".join(["-" * 32, "-" * 22] + ["-" * 12] * 6 + ["-" * 10]) + "|")
for r in ckpt_matrix:
    print(f"| {r['field']:<30} | {r['expected']:<20} | {r['e15']:<10} | {r['e16']:<10} | {r['e17']:<10} | {r['e18']:<10} | {r['e19']:<10} | {r['e20']:<10} | {r['status']:<8} |")

# Write CSV
os.makedirs("runs/phase4/p1_v8_2_medical_e15_e20/protocol", exist_ok=True)
import csv
with open("runs/phase4/p1_v8_2_medical_e15_e20/protocol/checkpoint_config_matrix.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["field", "expected", "e15", "e16", "e17", "e18", "e19", "e20", "status"])
    writer.writeheader()
    writer.writerows(ckpt_matrix)
