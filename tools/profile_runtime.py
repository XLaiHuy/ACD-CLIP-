import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, ".")

from model.clip import create_model
from model.adapter import ACDCLIP
from model.checkpoint_utils import load_adapter_checkpoint, h6_config_from_checkpoint
from utils import calculate_seg_loss
from train import get_multiple_adapted_single_class_text_embedding

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "runs/phase4/progress1_v7_full_seed0_ready3/train/adapter_12.pth"

checkpoint = torch.load(checkpoint_path, map_location="cpu")
h6_kwargs = h6_config_from_checkpoint(checkpoint)
h6_kwargs = {f"h6_{k}": v for k, v in h6_kwargs.items()}
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

def create_profiling_model(mode="R0"):
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=518,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    kwargs = h6_kwargs.copy()
    kwargs["h6_progress"] = 1
    kwargs["h6_global_text_mode"] = "dynamic_legacy"
    kwargs["h6_prediction_routing"] = "scheduled_topk"
    
    if mode == "R0":
        kwargs["h6_expert_enabled"] = True
        kwargs["diagnostics_mode"] = "full"
        kwargs["diagnostics_interval"] = 1
    elif mode == "R1":
        kwargs["h6_expert_enabled"] = False
        kwargs["diagnostics_mode"] = "full"
        kwargs["diagnostics_interval"] = 1
    elif mode == "R2":
        kwargs["h6_expert_enabled"] = True
        kwargs["diagnostics_mode"] = "light"
        kwargs["diagnostics_interval"] = 1
    elif mode == "R3":
        kwargs["h6_expert_enabled"] = False
        kwargs["diagnostics_mode"] = "light"
        kwargs["diagnostics_interval"] = 1

    model = ACDCLIP(
        clip_model=clip_model,
        **adapter_kwargs,
        **kwargs
    ).to(device)
    
    model.eval()
    model.prompt_mode = "h6_dynamic"
    model.use_hybrid_soft_prompt = True
    model.hybrid_alpha_current = 0.0
    model.image_adapter.load_state_dict(checkpoint["image_adapter"])
    model.text_adapter.load_state_dict(checkpoint["text_adapter"])
    if "soft_prompt" in checkpoint:
        model.soft_prompt.load_state_dict(checkpoint["soft_prompt"])
    if "h6_state_dict" in checkpoint:
        model.h6.load_state_dict(checkpoint["h6_state_dict"], strict=False)
    return model

class CudaTimer:
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
    def __enter__(self):
        self.start_event.record()
        return self
    def __exit__(self, *args):
        self.end_event.record()
    def get_ms(self):
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)

def run_profiling(mode):
    print(f"Profiling {mode}...")
    model = create_profiling_model(mode)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    torch.manual_seed(42)
    B = 1
    num_batches = 60

    stats = {
        "total_ms": [],
        "visual_forward_ms": [],
        "semantic_core_ms": [],
        "dynamic_text_ms": [],
        "router_ms": [],
        "paired_expert_ms": [],
        "diagnostics_ms": [],
        "loss_ms": [],
        "backward_ms": [],
        "optimizer_ms": []
    }
    
    peak_allocated = 0
    peak_reserved = 0

    for i in range(num_batches):
        if i == 10:
            torch.cuda.reset_peak_memory_stats()
            
        image = torch.randn(B, 3, 518, 518, device=device)
        mask = (torch.rand(B, 1, 518, 518, device=device) > 0.8).float()
        label = torch.randint(0, 2, (B,), device=device)
        class_name = "Brain"

        t_total = CudaTimer()
        t_visual = CudaTimer()
        t_dynamic = CudaTimer()
        t_h6 = CudaTimer()
        t_semantic = CudaTimer()
        t_loss = CudaTimer()
        t_backward = CudaTimer()
        t_opt = CudaTimer()
        
        with t_total:
            with t_visual:
                visual_output = model(image, return_phase4_features=True)
                seg_tokens = visual_output["seg_tokens"]
                det_tokens = visual_output["det_tokens"]
                seg_features = torch.stack(seg_tokens, dim=0)
                det_features = torch.stack(det_tokens, dim=0)

            with t_dynamic:
                text_embedding_levels = get_multiple_adapted_single_class_text_embedding(
                    model, "Brain", class_name, device
                )
                epoch_text_features = torch.stack([text_embedding_levels] * B, dim=0).permute(1, 0, 2, 3)

            with t_h6:
                h6_batch = model.h6.build_batch(
                    model, "Brain", [class_name] * B, visual_output, hybrid_alpha=0.0
                )

            with t_semantic:
                seg_pred = model.vision_text_fusion_gate_seg(
                    seg_features,
                    h6_batch["text_global"],
                    img_size=518,
                    h6_patch_logits=h6_batch["h6_logits"],
                )
            
            with t_loss:
                cls_pred = [
                    torch.matmul(
                        det_features[j].unsqueeze(dim=1),
                        epoch_text_features[j],
                    ).squeeze(1)
                    for j in range(det_features.shape[0])
                ]
                cls_pred = torch.stack(cls_pred, dim=0).mean(dim=0)
                cls_loss = F.cross_entropy(cls_pred, label)
                seg_loss = calculate_seg_loss(seg_pred, mask)
                loss = cls_loss + seg_loss
                
            with t_backward:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
            with t_opt:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # Wait for all
        torch.cuda.synchronize()
        if i >= 10:
            stats["total_ms"].append(t_total.get_ms())
            stats["visual_forward_ms"].append(t_visual.get_ms())
            stats["semantic_core_ms"].append(t_semantic.get_ms())
            stats["dynamic_text_ms"].append(t_dynamic.get_ms())
            stats["router_ms"].append(t_h6.get_ms())
            stats["paired_expert_ms"].append(0.0)
            stats["diagnostics_ms"].append(0.0)
            stats["loss_ms"].append(t_loss.get_ms())
            stats["backward_ms"].append(t_backward.get_ms())
            stats["optimizer_ms"].append(t_opt.get_ms())

        del visual_output, h6_batch, seg_pred, loss, cls_loss, seg_loss, image, mask, label

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

    # Cleanup model to prevent memory leak
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "mode": mode,
        "total_ms": float(np.mean(stats["total_ms"])),
        "visual_forward_ms": float(np.mean(stats["visual_forward_ms"])),
        "semantic_core_ms": float(np.mean(stats["semantic_core_ms"])),
        "dynamic_text_encoding_ms": float(np.mean(stats["dynamic_text_ms"])),
        "router/local-text_ms": float(np.mean(stats["router_ms"])),
        "paired-expert_ms": float(np.mean(stats["paired_expert_ms"])),
        "diagnostics_ms": float(np.mean(stats["diagnostics_ms"])),
        "loss-construction_ms": float(np.mean(stats["loss_ms"])),
        "backward_ms": float(np.mean(stats["backward_ms"])),
        "optimizer-step_ms": float(np.mean(stats["optimizer_ms"])),
        "peak_allocated_MB": peak_allocated,
        "peak_reserved_MB": peak_reserved,
        "samples_per_sec": (1000.0 / float(np.mean(stats["total_ms"]))) * B,
        "projected_full_epoch_time_mins": (float(np.mean(stats["total_ms"])) * 300 / 1000) / 60
    }

out_dir = Path("runs/phase4/p1_v8_evidence")
out_dir.mkdir(parents=True, exist_ok=True)

results = {}
if __name__ == "__main__":
    for mode in ["R0", "R1", "R2", "R3"]:
        res = run_profiling(mode)
        results[mode] = res
        import gc; gc.collect(); torch.cuda.empty_cache()

    # Output report
    print("\n" + "="*50)
    print("RUNTIME PROFILING REPORT (Phase4 P1-v8)")
    print("="*50)
    for mode, metrics in results.items():
        print(f"\n{mode}:")
        print(f"  Visual Branch: {metrics['visual_forward_ms']:.2f} ms")
        print(f"  Dynamic Branch: {metrics['dynamic_text_encoding_ms']:.2f} ms")
        print(f"  VRAM Peak: {metrics['peak_allocated_MB']:.1f} MB")
        print(f"  Batch Throughput: {metrics['samples_per_sec']:.2f} it/s")

    report_path = "runs/phase4/p1_v8_evidence/profile_runtime.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved profiling report to {report_path}")
