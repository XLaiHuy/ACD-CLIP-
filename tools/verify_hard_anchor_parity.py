#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from dataset import get_text_and_image_dataset, DOMAINS
from model.clip import create_model
from model.adapter import ACDCLIP
from model.checkpoint_utils import load_adapter_checkpoint
from utils import get_phase2b_global_text_features, get_multiple_adapted_single_class_text_embedding

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "runs/phase4/p1_v8_2_full20_seed0/adapter_20.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 1. Instantiate model via test.py pathway
    clip_model = create_model("ViT-L-14-336", img_size=518, device=device, pretrained="openai", require_pretrained=True)
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=3,
        dfg_mode="attn",
        dfg_attn_dim=256,
        dfg_attn_tau=8.0,
        use_ss2d_dfg=True,
        dfg_gamma_max=0.2,
        dfg_ss2d_fusion="weight_residual",
        dfg_beta_current=0.10,
        dfg_beta_schedule="fixed",
        h6_progress=1,
        h6_progress_version="P1-v8-minimal",
        h6_num_factors=4,
        h6_top_k=2,
        h6_bank_dim=256,
        h6_router_dim=128,
        local_factor_mode="center_spread",
        local_center_mix=0.05,
        local_factor_spread=0.10,
        h6_prediction_routing="dense",
        h6_expert_enabled=False,
        h6_load_bias_enabled=False,
        h6_cluster_responsibility=False,
        lambda_h6_dynamic_mean_anchor=0.0,
    ).to(device)
    load_adapter_checkpoint(model, ckpt)
    model.eval()
    model.hybrid_alpha_current = 0.0
    model.h6_global_text_mode = "hard_anchor"

    # Load 1 sample from Brain dataset
    datasets = get_text_and_image_dataset("Brain", img_size=518, stage="test")
    single_ds = datasets["Brain"]
    loader = torch.utils.data.DataLoader(single_ds, batch_size=1, shuffle=False)
    batch_data = next(iter(loader))

    image = batch_data["image"].to(device)
    class_names = list(batch_data["class_name"])

    # Pipeline A: test.py current pathway
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            visual_output = model(image, return_phase4_features=True)
            h6_batch = model.h6.build_batch(
                model, "Brain", class_names, visual_output, model.hybrid_alpha_current
            )
            seg_features = torch.stack(visual_output["seg_tokens"], dim=0)
            det_features = torch.stack(visual_output["det_tokens"], dim=0)
            
            h6_mode = getattr(model, "h6_global_text_mode", "phase2b_hybrid")
            is_hybrid = h6_mode == "phase2b_hybrid" and getattr(model, "use_hybrid_soft_prompt", False)
            text_A = get_phase2b_global_text_features(
                model, "Brain", class_names, device,
                use_hybrid_soft_prompt=is_hybrid,
                use_soft_prompt=getattr(model, "use_soft_prompt", False) if is_hybrid else False,
            ).to(dtype=det_features.dtype)

            cls_preds_A = [
                torch.matmul(det_features[i].unsqueeze(1), text_A[i]).squeeze(1)
                for i in range(det_features.shape[0])
            ]
            cls_preds_A = torch.stack(cls_preds_A, dim=0).mean(dim=0)
            pred_image_A = F.softmax(cls_preds_A, dim=1)[:, 1]

            seg_pred_A = model.vision_text_fusion_gate_seg(
                seg_features, text_A, test_mode=True, domain=DOMAINS["Brain"],
                h6_patch_logits=h6_batch["h6_logits"],
            )

    # Pipeline B: Explicit hard-anchor calculation
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text_B_single = get_multiple_adapted_single_class_text_embedding(model, "Brain", "Brain", device)
            text_B = text_B_single.unsqueeze(1).to(dtype=det_features.dtype) # [n_groups, 1, 768, 2]

            cls_preds_B = [
                torch.matmul(det_features[i].unsqueeze(1), text_B[i]).squeeze(1)
                for i in range(det_features.shape[0])
            ]
            cls_preds_B = torch.stack(cls_preds_B, dim=0).mean(dim=0)
            pred_image_B = F.softmax(cls_preds_B, dim=1)[:, 1]

            seg_pred_B = model.vision_text_fusion_gate_seg(
                seg_features, text_B, test_mode=True, domain=DOMAINS["Brain"],
                h6_patch_logits=h6_batch["h6_logits"],
            )

    diff_text = (text_A - text_B).abs().max().item()
    diff_image = (pred_image_A - pred_image_B).abs().max().item()
    diff_seg = (seg_pred_A - seg_pred_B).abs().max().item()

    print(f"Text Max Abs Diff:  {diff_text:.8e}")
    print(f"Image Max Abs Diff: {diff_image:.8e}")
    print(f"Seg Max Abs Diff:   {diff_seg:.8e}")

    assert diff_text <= 1e-6, f"Text mismatch: {diff_text}"
    assert diff_image <= 1e-6, f"Image mismatch: {diff_image}"
    assert diff_seg <= 1e-6, f"Seg mismatch: {diff_seg}"
    print("[SUCCESS] Hard-Anchor / Hybrid-Soft-Prompt Parity Verified (max_abs_diff <= 1e-6)")

if __name__ == "__main__":
    main()
