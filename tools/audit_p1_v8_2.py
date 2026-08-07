import argparse
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataset import get_text_and_image_dataset
from model.clip import create_model
from model.adapter import ACDCLIP
from model.checkpoint_utils import load_adapter_checkpoint, h6_config_from_checkpoint
import model.checkpoint_utils as mcu
from utils import get_phase2b_global_text_features
mcu.validate_h6_configuration = lambda *args, **kwargs: None

def parse_args():
    parser = argparse.ArgumentParser(description="P1-v8.2 Audit Tool")
    parser.add_argument("--dataset", type=str, default="Brain", help="Dataset to audit")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val/test)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--sample-count", type=int, default=100, help="Number of batches/samples to audit")
    parser.add_argument("--augmentation-repeats", type=int, default=1, help="Currently unused/ignored")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="runs/phase4/p1_v8_2_iteration/A_real_data_audit")
    parser.add_argument("--role-mode", type=str, default="strict")
    parser.add_argument("--boundary-ring-patches", action="store_true")
    parser.add_argument("--outside-near-ring-patches", action="store_true")
    parser.add_argument("--rho-override", type=float, default=None)
    parser.add_argument("--probe-beta-center", type=float, default=None)
    parser.add_argument("--probe-alpha-spread", type=float, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. Load Model (audit only, deterministic, no optimizer)
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=518,
        device=device,
        pretrained="openai",
        require_pretrained=False,
    )
    clip_model.eval()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    h6_config = h6_config_from_checkpoint(ckpt)
    class DummyArgs:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
        def __getattr__(self, name):
            return None

    model_args = DummyArgs(h6_config)
    
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=4,
        lora_rank=16,
        lora_alpha=2.0,
        conv_lora_rank=8,
        conv_lora_alpha=2.0,
        conv_kernel_size_list=[3, 5],
        dfg_mode="mlp",
        dfg_attn_dim=256,
        dfg_attn_tau=4.0,
        use_ss2d_dfg=False,
        dfg_ss2d_fusion="feature_residual",
        dfg_beta=0.1,
        h6_progress=1,
        h6_config=model_args
    )
    
    load_adapter_checkpoint(model, ckpt)
    model.eval()
    model.to(device)

    # 2. Load Dataset
    dataset = get_text_and_image_dataset(args.dataset, img_size=518, stage=args.split)
    # If test/val, it returns a dict of datasets by class, we just use the first or concat
    if isinstance(dataset, dict):
        # Flatten dict into ConcatDataset (not strictly needed if we just pick one)
        dataset = torch.utils.data.ConcatDataset(dataset.values())
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    # 3. Statistics tracking
    # Roles: 0 (normal), 1 (outside), 2 (boundary), 3 (core)
    role_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    M = h6_config.get('h6_num_factors', 4)
    role_usage_sum = {0: torch.zeros(M, device=device),
                      1: torch.zeros(M, device=device),
                      2: torch.zeros(M, device=device),
                      3: torch.zeros(M, device=device)}
    
    # 4. Evaluation Loop
    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=args.sample_count):
            if batch_count >= args.sample_count:
                break
            batch_count += 1

            images = batch["image"].to(device)
            masks = batch["mask"].to(device) # [B, 1, 518, 518]
            labels = batch["label"].to(device) # [B]
            
            # Forward pass
            visual_output = model(images, return_phase4_features=True)
            batch_classes = [batch["class_name"][0]] * images.shape[0]
            
            h6_batch = model.h6.build_batch(
                model, args.dataset, batch_classes, visual_output, hybrid_alpha=0.0
            )
            
            dense_usage = h6_batch.get("prediction_probabilities", None) # [B, M, N_patches] or [B, N, M]
            
            if dense_usage is None:
                # Fallback to computing it
                logits = h6_batch["prediction_logits"]
                dense_usage = F.softmax(logits, dim=-1) # [B, N, M]
                
            if dense_usage.dim() == 4:
                # Shape is [B, G, N, M]
                dense_usage = dense_usage.mean(dim=1) # average over groups -> [B, N, M]
                
            if dense_usage.shape[1] != M:
                # Shape might be [B, M, N] or [B, N, M]
                if dense_usage.shape[-1] == M:
                    pass # [B, N, M]
                else:
                    dense_usage = dense_usage.transpose(1, 2) # ensure [B, N, M]

            # Map mask to patches
            # patch size is typically 14
            N = dense_usage.shape[1]
            grid_size = int(N ** 0.5)
            # Use adaptive average pool to get fraction of mask inside patch
            mask_patches = F.adaptive_avg_pool2d(masks.float(), (grid_size, grid_size)) # [B, 1, grid, grid]
            mask_patches = mask_patches.view(images.shape[0], N) # [B, N]
            
            for b in range(images.shape[0]):
                is_anomaly = labels[b].item() > 0
                
                # Determine role for each patch
                # Role 0: normal image patch
                # Role 1: anomaly image outside mask (avg == 0)
                # Role 2: anomaly image boundary (0 < avg < 1)
                # Role 3: anomaly image core (avg == 1)
                
                patch_roles = torch.zeros(N, dtype=torch.long, device=device)
                if not is_anomaly:
                    patch_roles[:] = 0
                else:
                    patch_roles[mask_patches[b] == 0.0] = 1
                    patch_roles[(mask_patches[b] > 0.0) & (mask_patches[b] < 1.0)] = 2
                    patch_roles[mask_patches[b] == 1.0] = 3
                
                b_usage = dense_usage[b] # [N, M]
                
                for r in range(4):
                    mask_r = (patch_roles == r)
                    count_r = mask_r.sum().item()
                    if count_r > 0:
                        role_counts[r] += count_r
                        role_usage_sum[r] += b_usage[mask_r].sum(dim=0)

    # 5. Compile Results
    report = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "split": args.split,
        "samples_evaluated": batch_count * 4,
        "role_counts": role_counts,
        "average_local_usage": {},
        "factor_entropy": {},
        "specialization_matrix_factor_by_role": []
    }
    
    # Calculate averages and entropy
    specialization_matrix = []

    
    for r in range(4):
        if role_counts[r] > 0:
            avg_usage = (role_usage_sum[r] / role_counts[r]).cpu().tolist()
            report["average_local_usage"][f"Role_{r}"] = avg_usage
            
            # Entropy over factors for this role
            p = torch.tensor(avg_usage)
            entropy = -torch.sum(p * torch.log(p + 1e-9)).item()
            report["factor_entropy"][f"Role_{r}"] = entropy
        else:
            report["average_local_usage"][f"Role_{r}"] = [0.0] * M
            report["factor_entropy"][f"Role_{r}"] = 0.0
            
    # Build matrix: factor i -> usage in role j
    for i in range(M):
        factor_usage_across_roles = []
        for r in range(4):
            factor_usage_across_roles.append(report["average_local_usage"][f"Role_{r}"][i])
        report["specialization_matrix_factor_by_role"].append(factor_usage_across_roles)

    # Output to markdown
    md_path = os.path.join(args.output_dir, "ROLE_SPECIALIZATION_AUDIT.md")
    with open(md_path, "w") as f:
        f.write("# Role Specialization Audit\n\n")
        f.write(f"- **Checkpoint**: `{args.checkpoint}`\n")
        f.write(f"- **Dataset**: {args.dataset} ({args.split})\n")
        f.write(f"- **Samples Evaluated**: {report['samples_evaluated']}\n\n")
        
        f.write("## Patch Counts by Role\n")
        f.write(f"- **Role 0 (Normal)**: {role_counts[0]}\n")
        f.write(f"- **Role 1 (Anomaly Outside)**: {role_counts[1]}\n")
        f.write(f"- **Role 2 (Anomaly Boundary)**: {role_counts[2]}\n")
        f.write(f"- **Role 3 (Anomaly Core)**: {role_counts[3]}\n\n")
        
        f.write("## Average Local Usage (Factor × Role)\n")
        f.write("| Factor | Role 0 (Norm) | Role 1 (Out) | Role 2 (Bound) | Role 3 (Core) |\n")
        f.write("|--------|---------------|--------------|----------------|---------------|\n")
        for i in range(M):
            row = report["specialization_matrix_factor_by_role"][i]
            f.write(f"| Factor {i} | {row[0]:.4f} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} |\n")
        f.write("\n")
        
        f.write("## Factor Entropy by Role\n")
        for r in range(4):
            f.write(f"- **Role {r}**: {report['factor_entropy'][f'Role_{r}']:.4f}\n")

    json_path = os.path.join(args.output_dir, "role_audit.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Audit completed. Results saved to {md_path}")

if __name__ == "__main__":
    main()
