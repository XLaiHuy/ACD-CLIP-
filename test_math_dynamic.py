import torch
import torch.nn as nn
from model.clip import create_model
from model.adapter import ACDCLIP

def test_dynamic_conv_math():
    print("=== Starting Dynamic Conv Math & Gradient Validation ===")
    
    # 1. Initialize device (CPU for testing)
    device = torch.device("cpu")
    
    # 2. Create a mock clip model with very small resolution to prevent memory exhaustion
    print("Creating mock CLIP model structure...")
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=112,
        device=device,
        pretrained=None,
        require_pretrained=False,
        force_image_size=112,
    )
    clip_model.eval()
    
    # 3. Instantiate ACDCLIP with use_dynamic_conv=True
    print("Instantiating ACDCLIP with use_dynamic_conv=True...")
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=4,
        image_adapt_weight=0.2,
        conv_lora_rank=8,
        conv_lora_alpha=2.0,
        conv_kernel_size_list=[3, 5],
        text_adapt_weight=0.2,
        lora_rank=16,
        lora_alpha=2.0,
        use_dynamic_conv=True,
    ).to(device)
    
    # Set trainable parameters as in training
    model.requires_grad_(False)
    model.text_adapter.requires_grad_(True)
    model.image_adapter.requires_grad_(True)
    
    # 4. Perform Optimizer parameter grouping check
    print("Checking optimizer parameter groups split...")
    image_dw_params = []
    image_other_params = []
    for name, param in model.image_adapter.named_parameters():
        if not param.requires_grad:
            continue
        if "depthwise" in name or "bias" in name or "bn" in name:
            image_dw_params.append(param)
        else:
            image_other_params.append(param)
            
    print(f"Total image adapter trainable parameters: {len(image_dw_params) + len(image_other_params)}")
    print(f"Excluded from weight decay (depthwise/bias/bn): {len(image_dw_params)}")
    print(f"Subject to weight decay: {len(image_other_params)}")
    
    assert len(image_dw_params) > 0, "No depthwise/bias/bn parameters detected!"
    assert len(image_other_params) > 0, "No standard image adapter parameters detected!"
    
    # 5. Forward Pass Check
    print("Running forward pass with dummy input...")
    dummy_img = torch.randn(1, 3, 112, 112, device=device) # Batch size of 1, 112x112 image
    seg_tokens, det_tokens = model(dummy_img)
    
    print(f"Number of groups: {len(seg_tokens)}")
    print(f"Seg tokens shape: {[t.shape for t in seg_tokens]}")
    print(f"Det tokens shape: {[t.shape for t in det_tokens]}")
    
    # Verify outputs match expected shapes
    # For ViT-L-14-336 at 112x112, seq_len = (112/14)^2 = 8^2 = 64 tokens
    for t_seg, t_det in zip(seg_tokens, det_tokens):
        assert t_seg.shape == (1, 64, 768), f"Expected seg token shape (1, 64, 768), got {t_seg.shape}"
        assert t_det.shape == (1, 768), f"Expected det token shape (1, 768), got {t_det.shape}"
        
    # 6. Backward Pass & Gradient Verification
    print("Testing backward pass and gradient computation...")
    
    # Segment tokens and text features are fused to generate predictions
    vision_tokens = torch.stack(seg_tokens, dim=0) # [4, 1, 64, 768]
    text_features = torch.randn(4, 1, 768, 2, device=device) # [n_groups, bs, 768, 2]
    
    seg_pred = model.vision_text_fusion_gate_seg(vision_tokens, text_features)
    
    # We sum all outputs to compute a unified loss, including detection token projection
    loss = seg_pred.sum() + sum(t.sum() for t in det_tokens)
    loss.backward()
    
    # Check that gradients are successfully backpropagated through dynamic conv weights
    grad_ok = True
    for name, param in model.image_adapter.named_parameters():
        if param.requires_grad:
            is_dynamic_adapter = "lora_adapters" in name
            if param.grad is None:
                print(f"Warning: Param {name} has no gradient!")
                if is_dynamic_adapter:
                    grad_ok = False
            elif torch.all(param.grad == 0):
                print(f"Warning: Param {name} gradient is all zeros!")
                if is_dynamic_adapter:
                    grad_ok = False
                
    if grad_ok:
        print("SUCCESS: All dynamic adapter parameters have valid, non-zero gradients!")
    else:
        print("FAILURE: Some dynamic adapter parameters did not compute gradients properly.")
        raise AssertionError("Gradient flow failed.")
        
    print("=== Dynamic Conv Math & Gradient Validation Passed Successfully! ===")

if __name__ == "__main__":
    test_dynamic_conv_math()
