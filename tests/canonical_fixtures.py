from __future__ import annotations

import torch
from torch import nn

from model.phase2b_runtime import IMAGE_SIZE, PATCH_COUNT, STAGES


class TinyClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Identity()
        self.backbone_parameter = nn.Parameter(torch.ones(1))

    def set_grad_checkpointing(self, enabled: bool):
        del enabled

    def eval(self):
        return self


class TinyAdapter(nn.Module):
    def __init__(self, clip_model, n_groups=3, dfg_beta=0.1, h6_progress=0, **kwargs):
        super().__init__()
        del kwargs
        if n_groups != STAGES or h6_progress != 0:
            raise ValueError("fixture only models the canonical three-stage disabled path")
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.n_groups = n_groups
        self.dfg_beta = float(dfg_beta)
        self.h6_enabled = False
        self.h6 = None
        self.image_adapter = nn.Linear(1, 1, bias=False)
        self.text_adapter = nn.Linear(1, 1, bias=False)
        self.soft_prompt = nn.Linear(1, 1, bias=False)
        self.hybrid_alpha_current = 0.0
        self.use_hybrid_soft_prompt = True
        self.use_soft_prompt = False
        self.prompt_mode = "hybrid"

    def set_dfg_beta(self, value: float):
        self.dfg_beta = float(value)

    def forward(self, image, return_phase4_features=False):
        batch = image.shape[0]
        base = image.mean(dim=(1, 2, 3)).view(batch, 1, 1)
        weight = self.image_adapter.weight.reshape(1, 1, 1)
        seg = (base + weight).expand(batch, PATCH_COUNT, 768)
        det = seg.mean(dim=1)
        if return_phase4_features:
            return {"seg_tokens": [seg + (index * 0.01) for index in range(STAGES)], "det_tokens": [det + (index * 0.01) for index in range(STAGES)]}
        return [seg + (index * 0.01) for index in range(STAGES)], [det + (index * 0.01) for index in range(STAGES)]

    def vision_text_fusion_gate_seg(self, vision_tokens, text_features, img_size=IMAGE_SIZE, test_mode=False, domain="Industrial", return_details=False, **kwargs):
        del domain, kwargs
        native = torch.einsum("sbpd,sbdc->sbpc", vision_tokens, text_features)
        margin = native[..., 1] - native[..., 0]
        logits = native.permute(1, 0, 3, 2).reshape(vision_tokens.shape[1], STAGES, 2, 37, 37).mean(dim=1)
        logits = torch.nn.functional.interpolate(logits, size=(img_size, img_size), mode="bilinear", align_corners=True)
        probability = torch.softmax(logits, dim=1)
        if test_mode:
            probability = probability[:, 1]
        if return_details:
            return probability, native, margin
        return probability


def tiny_config() -> dict:
    return {
        "protocol_version": "PHASE2B_CANONICAL_V1",
        "model_name": "tiny",
        "img_size": IMAGE_SIZE,
        "precision": "fp32",
        "n_groups": STAGES,
        "dfg_mode": "attn",
        "dfg_attn_dim": 8,
        "dfg_attn_tau": 4.0,
        "dfg_beta": 0.1,
        "use_hybrid_soft_prompt": True,
        "use_soft_prompt": False,
        "soft_prompt_ctx_len": 4,
        "soft_prompt_trainable": True,
        "image_adapt_weight": 0.2,
        "text_adapt_weight": 0.2,
        "lora_rank": 2,
        "lora_alpha": 2.0,
        "conv_lora_rank": 2,
        "conv_lora_alpha": 2.0,
        "conv_kernel_size_list": [3, 5],
        "dfg_ss2d_fusion": "feature_residual",
        "dfg_gamma_max": 0.2,
        "dfg_beta_schedule": "fixed",
        "dfg_beta_target": 0.1,
        "dfg_weight_residual_fp32": True,
        "grad_checkpointing": False,
    }
