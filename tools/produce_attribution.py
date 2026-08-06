import json
import csv
from pathlib import Path

out_dir = Path("runs/phase4/p1_v8_evidence")
out_dir.mkdir(parents=True, exist_ok=True)

modes = {
    "B0": {
        "dir": "runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch/B0_-_Phase2B_baseline",
        "global_text_mode": "phase2b_hybrid",
        "prediction_routing": "N/A",
        "expert_mode": "disabled",
        "rho_override": "N/A",
    },
    "A0": {
        "dir": "runs/phase4/progress1_v7_full_seed0_ready3/train/A0_-_legacy_P1-v7",
        "global_text_mode": "dynamic_legacy",
        "prediction_routing": "scheduled_topk",
        "expert_mode": "enabled",
        "rho_override": "N/A",
    },
    "A1": {
        "dir": "runs/phase4/progress1_v7_full_seed0_ready3/train/A1_-_hard-anchor_baseline_inside_P1_checkpoint",
        "global_text_mode": "hard_anchor",
        "prediction_routing": "N/A",
        "expert_mode": "disabled",
        "rho_override": "0.0",
    },
    "A2": {
        "dir": "runs/phase4/progress1_v7_full_seed0_ready3/train/A2_-_target_P1-v8",
        "global_text_mode": "hard_anchor",
        "prediction_routing": "dense",
        "expert_mode": "disabled",
        "rho_override": "N/A",
    },
    "A3": {
        "dir": "runs/phase4/progress1_v7_full_seed0_ready3/train/A3_-_sparse_comparison",
        "global_text_mode": "hard_anchor",
        "prediction_routing": "scheduled_topk",
        "expert_mode": "disabled",
        "rho_override": "N/A",
    },
    "A4": {
        "dir": "runs/phase4/progress1_v7_full_seed0_ready3/train/A4_-_optional_safety_comparison",
        "global_text_mode": "phase2b_hybrid",
        "prediction_routing": "dense",
        "expert_mode": "disabled",
        "rho_override": "N/A",
    }
}

macro_rows = []
dataset_rows = []
config_dict = {}

for mode_name, meta in modes.items():
    json_path = Path(meta["dir"]) / "selection_support_aware_v2.json"
    if not json_path.exists():
        print(f"Missing {json_path}")
        continue
    with open(json_path) as f:
        data = json.load(f)
    
    epoch_data = data["macro_by_epoch"][0]
    
    # Macro
    macro_row = {
        "Mode": mode_name,
        "Pixel AUROC": None,
        "Pixel AP": None,
        "Image AUROC": None,
        "Image AP": None,
        "Combined": data["support_aware_combined_score"]
    }
    
    # Config
    config_dict[mode_name] = {
        "checkpoint_path": data["selected_checkpoint"],
        "manifest_path": data["fingerprint_config"]["validation_split_manifest_hash"],
        "global_text_mode": meta["global_text_mode"],
        "prediction_routing": meta["prediction_routing"],
        "expert_mode": meta["expert_mode"],
        "rho_override": meta["rho_override"],
        "output_directory": meta["dir"]
    }

    # Aggregate pixel/image macro properly
    img_auroc_sum = 0
    img_ap_sum = 0
    img_count = 0
    pix_auroc_sum = 0
    pix_ap_sum = 0
    pix_count = 0
    
    for ds_row in epoch_data["per_dataset"]:
        ds = ds_row["dataset"]
        # Per dataset
        dataset_rows.append({
            "Mode": mode_name,
            "Dataset": ds,
            "Pixel AUROC": ds_row["pixel_AUROC"],
            "Pixel AP": ds_row["pixel_AP"],
            "Image AUROC": ds_row["image_AUROC"] if ds_row["image_valid"] else "N/A",
            "Image AP": ds_row["image_AP"] if ds_row["image_valid"] else "N/A"
        })
        
        if ds_row["image_valid"]:
            img_auroc_sum += ds_row["image_AUROC"]
            img_ap_sum += ds_row["image_AP"]
            img_count += 1
        if ds_row["pixel_valid"]:
            pix_auroc_sum += ds_row["pixel_AUROC"]
            pix_ap_sum += ds_row["pixel_AP"]
            pix_count += 1
            
    macro_row["Pixel AUROC"] = pix_auroc_sum / pix_count if pix_count else None
    macro_row["Pixel AP"] = pix_ap_sum / pix_count if pix_count else None
    macro_row["Image AUROC"] = img_auroc_sum / img_count if img_count else None
    macro_row["Image AP"] = img_ap_sum / img_count if img_count else None
    
    macro_rows.append(macro_row)

# Save Macro CSV
with open(out_dir / "attribution_macro.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Mode", "Pixel AUROC", "Pixel AP", "Image AUROC", "Image AP", "Combined"])
    writer.writeheader()
    writer.writerows(macro_rows)

# Save Dataset CSV
with open(out_dir / "attribution_by_dataset.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Mode", "Dataset", "Pixel AUROC", "Pixel AP", "Image AUROC", "Image AP"])
    writer.writeheader()
    writer.writerows(dataset_rows)

# Save Config JSON
with open(out_dir / "attribution_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

print("Saved Step 2 outputs")
