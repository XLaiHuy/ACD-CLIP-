import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from torch import nn

import sys
import os
import math

# Try to mock triton if not installed, to prevent VMamba's import-time jit decorator errors
try:
    import triton
except ImportError:
    from unittest.mock import MagicMock
    triton_mock = MagicMock()
    triton_mock.jit = lambda f: f
    triton_mock.__path__ = []
    sys.modules['triton'] = triton_mock
    sys.modules['triton.language'] = MagicMock()
    sys.modules['triton.backends'] = MagicMock()
    sys.modules['triton.backends.compiler'] = MagicMock()
    sys.modules['triton.compiler'] = MagicMock()
    sys.modules['triton.compiler.compiler'] = MagicMock()

# Dynamically import SS2D from VMamba repo
possible_paths = [
    "C:/Users/HUY/Documents/ACD-CLIP++/VMamba",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../VMamba")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../VMamba")),
    "/content/VMamba",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../VMamba"))
]
for p in possible_paths:
    if os.path.exists(p):
        sys.path.append(p)
        break

import vmamba
# Ensure Triton is disabled if we had to mock it
if 'triton' in sys.modules and isinstance(sys.modules['triton'], MagicMock):
    vmamba.WITH_TRITON = False

from vmamba import SS2D
from .adapter_modules import TextDoraAdapter, MLPAdapter, ConvLoraAdapter


class MambaDFGBlock(nn.Module):
    def __init__(self, c_in=768, d_k=256, n_groups=3):
        super().__init__()
        self.n_groups = n_groups
        self.mamba_ln1 = nn.LayerNorm(c_in)
        self.mamba_linear1 = nn.Linear(c_in, 2 * c_in)
        self.mamba_silu = nn.SiLU()
        self.mamba_ss2d = SS2D(d_model=2*c_in, channel_first=False, forward_type="v0")
        self.mamba_linear2 = nn.Linear(2 * c_in, c_in)
        self.mamba_ln2 = nn.LayerNorm(c_in)
        
        # Learnable blending weight gamma (init 0)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Cross-Attention projections
        self.query_proj = nn.Linear(c_in, d_k)
        self.key_proj = nn.Linear(c_in, d_k)
        
    def forward(self, img_feat, T_norm, T_abnorm, tau):
        # img_feat: [bs, patch_num, c_in]
        # T_norm: [bs, n_groups, c_in]
        # T_abnorm: [bs, n_groups, c_in]
        # tau: scalar parameter
        
        bs, patch_num, c_in = img_feat.shape
        H = int(math.sqrt(patch_num))
        
        # 1. Mamba Dual-Pooling
        v_avg = img_feat.mean(dim=1) # [bs, c_in]
        
        # v_mamba branch
        V_i_2D = img_feat.view(bs, H, H, c_in) # [bs, H, H, c_in]
        x_mamba = self.mamba_ln1(V_i_2D)
        x_mamba = self.mamba_linear1(x_mamba)
        x_mamba = self.mamba_silu(x_mamba)
        x_mamba = self.mamba_ss2d(x_mamba) # [bs, H, H, 2*c_in]
        x_mamba = self.mamba_linear2(x_mamba)
        x_mamba = self.mamba_ln2(x_mamba)
        v_mamba = x_mamba.mean(dim=[1, 2]) # GAP -> [bs, c_in]
        
        v_global = v_avg + self.gamma * v_mamba # [bs, c_in]
        
        # 2. Cross-Attention
        Q = self.query_proj(v_global).unsqueeze(1) # [bs, 1, d_k]
        
        K_norm = self.key_proj(T_norm) # [bs, n_groups, d_k]
        K_abnorm = self.key_proj(T_abnorm) # [bs, n_groups, d_k]
        
        # Attention normal
        scores_norm = torch.bmm(Q, K_norm.transpose(1, 2)) / (K_norm.shape[-1] ** 0.5) # [bs, 1, n_groups]
        attn_norm = F.softmax(scores_norm, dim=-1)
        T_normal_fused = torch.bmm(attn_norm, T_norm).squeeze(1) # [bs, c_in]
        
        # Attention abnormal
        scores_abnorm = torch.bmm(Q, K_abnorm.transpose(1, 2)) / (K_abnorm.shape[-1] ** 0.5) # [bs, 1, n_groups]
        attn_abnorm = F.softmax(scores_abnorm, dim=-1)
        T_abnormal_fused = torch.bmm(attn_abnorm, T_abnorm).squeeze(1) # [bs, c_in]
        
        # 3. Cosine Similarity Map
        img_feat_norm = F.normalize(img_feat, p=2, dim=-1) # [bs, patch_num, c_in]
        T_normal_fused_norm = F.normalize(T_normal_fused, p=2, dim=-1).unsqueeze(1) # [bs, 1, c_in]
        T_abnormal_fused_norm = F.normalize(T_abnormal_fused, p=2, dim=-1).unsqueeze(1) # [bs, 1, c_in]
        
        sim_n = torch.sum(img_feat_norm * T_normal_fused_norm, dim=-1) / tau  # [bs, patch_num]
        sim_a = torch.sum(img_feat_norm * T_abnormal_fused_norm, dim=-1) / tau  # [bs, patch_num]
        
        seg_logits = torch.stack([sim_n, sim_a], dim=1) # [bs, 2, patch_num]
        seg_logits = seg_logits.view(bs, 2, H, H) # [bs, 2, H, H]
        
        return seg_logits



class AddWeight(nn.Module):
    def __init__(
            self,
            image_adapt_weight,
            is_text=False,
            change_x=True,
    ):
        super().__init__()
        d_model = 768 if is_text else 1024
        self.i_w = nn.Parameter(torch.ones(1, 1, d_model) * image_adapt_weight)
        self.change_x = change_x

    def forward(self, x, adapt_out):
        if not self.change_x:
            return x + self.i_w * adapt_out
        return (1 - self.i_w) * x + self.i_w * adapt_out


class ACDCLIP(nn.Module):
    def __init__(
            self,
            clip_model,
            text_adapt_weight: float = 0.15,
            image_adapt_weight: float = 0.15,
            n_groups: int = 3,
            lora_rank: int = 16,
            lora_alpha: float = 2,
            conv_lora_rank: int = 8,
            conv_lora_alpha: float = 2,
            conv_kernel_size_list=(3, 5),
            **kwargs,
    ):
        super().__init__()
        assert n_groups in [2, 3, 4, 6], "n_groups must be one of [2, 3, 4, 6]"
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        text_step = 12 // n_groups
        image_step = 24 // n_groups
        self.image_levels = [t for t in range(image_step, 24 + 1, image_step)]
        self.text_levels = [t for t in range(text_step, 12 + 1, text_step)]
        self.n_groups = n_groups
        self.image_adapt_weight = image_adapt_weight
        self.text_adapt_weight = text_adapt_weight

        image_adapt_weights = nn.ModuleList(
            [AddWeight(image_adapt_weight) for _ in range(n_groups)]
        )
        image_lora_adapters = nn.ModuleList(
            [
                ConvLoraAdapter(1024, 1024, lora_rank, lora_alpha, conv_lora_rank, conv_lora_alpha,
                                conv_kernel_size_list)
                for _ in
                range(n_groups)
            ]
        )
        seg_proj = nn.ModuleList(
            [MLPAdapter(1024, 768, 256) for _ in range(n_groups)]
        )
        det_proj = nn.ModuleList(
            MLPAdapter(1024, 768, 256) for _ in range(n_groups)
        )
        seg_image_layer_norms = nn.ModuleList(
            [nn.LayerNorm(768) for _ in range(n_groups)]
        )
        det_image_layer_norms = nn.ModuleList(
            [nn.LayerNorm(768) for _ in range(n_groups)]
        )
        # 动态路由门：Mamba Dual-Pooling + Cross-Attention DFG
        vision_text_gate = nn.ModuleList([
            MambaDFGBlock(c_in=768, d_k=256, n_groups=n_groups) for _ in range(n_groups)
        ])
        self.image_adapter = nn.ModuleDict(
            {
                "m_i_w": image_adapt_weights,
                "lora_adapters": image_lora_adapters,
                "seg_layer_norms": seg_image_layer_norms,
                "seg_proj": seg_proj,
                "det_layer_norms": det_image_layer_norms,
                "det_proj": det_proj,
                "vision_text_gate": vision_text_gate,
            }
        )
        # Register learnable temperature tau to image_adapter so it is saved/loaded in state_dict
        self.image_adapter.register_parameter('tau', nn.Parameter(torch.tensor(0.07)))

        text_adapt_weights = nn.ModuleList(
            [AddWeight(text_adapt_weight, is_text=True) for _ in range(n_groups)]
        )
        text_lora_adapters = nn.ModuleList(
            [TextDoraAdapter(768, 768, r=lora_rank, alpha=lora_alpha) for _ in range(n_groups)]
        )
        text_layer_norms = nn.ModuleList(
            [nn.LayerNorm(768) for _ in range(n_groups)]
        )
        self.text_adapter = nn.ModuleDict(
            {
                "m_t_w": text_adapt_weights,
                "layer_norms": text_layer_norms,
                "lora_adapters": text_lora_adapters,
            }
        )

    def forward_original(self, x, modality="visual"):
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [
                self.clipmodel.visual._global_pool(t)[1] for t in patch_features
            ]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")

    def forward(self, x):
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        group_outs = []
        for i in range(24):
            x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            # [1370, bs, 1024]
            index = -1
            for j in range(self.n_groups):
                if i + 1 == self.image_levels[j]:
                    index = j
                    break
            if index != -1:
                t = x[1:, :, :]
                adapt_out = self.image_adapter["lora_adapters"][index](t)
                adapt_out = (
                        adapt_out
                        * t.norm(dim=-1, keepdim=True)
                        / adapt_out.norm(dim=-1, keepdim=True)
                )
                # [1369, bs, 1024]
                group_out = self.image_adapter["m_i_w"][index](t, adapt_out)
                # group_out = t * (1 - self.image_adapt_weight) + adapt_out * self.image_adapt_weight
                group_outs.append(group_out)
                x = torch.cat(
                    [
                        x[0, :, :].unsqueeze(0),
                        group_out
                    ],
                    dim=0,
                )

        # [batch_size, seq_len, dim]
        group_tokens = [t.permute(1, 0, 2) for t in group_outs]  # [bs, 1369, 1024]

        # 1: Segmentation Tokens
        seg_tokens_proj = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(group_tokens)
        ]  # 投影到分割空间
        seg_tokens_norm = [
            self.image_adapter["seg_layer_norms"][i](t) for i, t in enumerate(seg_tokens_proj)
        ]  # 层归一化
        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens_norm]  # L2归一化

        # 2: Detection Tokens
        det_tokens_proj = [
            self.image_adapter["det_proj"][i](t) for i, t in enumerate(group_tokens)
        ]
        det_tokens_norm = [
            self.image_adapter["det_layer_norms"][i](t) for i, t in enumerate(det_tokens_proj)
        ]
        det_tokens = [F.normalize(t, dim=-1).mean(1) for t in det_tokens_norm]  # L2归一化 + 全局平均池化

        return seg_tokens, det_tokens

    def vision_text_fusion_gate_seg(
            self,
            vision_tokens: torch.Tensor,
            text_features: torch.Tensor,
            img_size: int = 518,
            test_mode: bool = False,
            domain: str = "Industrial",
    ):
        """
        Fuse vision and text features using a gating mechanism.
        :param vision_tokens: vision tokens from the image encoder. [n_groups, bs, patch_num, 768]
        :param text_features: text features from the text encoder. [n_groups, bs, 768, 2]
        :return: fused seg features.
        """
        B, patch_size, _ = vision_tokens.shape[1:]
        H = int(np.sqrt(patch_size))
        group_seg_preds = []
        for i in range(self.n_groups):
            img_feat = vision_tokens[i]  # [bs, patch_num, 768]
            
            # T_norm and T_abnorm from text_features
            # text_features has shape [n_groups, bs, 768, 2]
            T_norm = text_features[:, :, :, 0].permute(1, 0, 2)  # [bs, n_groups, 768]
            T_abnorm = text_features[:, :, :, 1].permute(1, 0, 2)  # [bs, n_groups, 768]
            
            # Forward through MambaDFGBlock
            seg_logits = self.image_adapter["vision_text_gate"][i](
                img_feat, T_norm, T_abnorm, self.image_adapter.tau
            )
            group_seg_preds.append(seg_logits)
            
        if test_mode:
            sigma = 1 if domain == "Industrial" else 1.5
            kernel_size = 7 if domain == "Industrial" else 9
            group_seg_preds = [
                gaussian_blur2d(
                    seg_pred,
                    (kernel_size, kernel_size),
                    (sigma, sigma)
                ) for seg_pred in group_seg_preds
            ]
        group_seg_preds = [
            F.interpolate(
                seg_pred,
                size=img_size,
                mode="bilinear",
                align_corners=True
            ).unsqueeze(0) for seg_pred in group_seg_preds
            # [1, bs, 2, img_size, img_size]
        ]  # [1, bs, 2, img_size, img_size] * n_groups
        all_group_preds = torch.cat(group_seg_preds, dim=0)  # [n_groups, bs, 2, img_size, img_size]
        final_seg_pred = torch.mean(all_group_preds, dim=0)  # [bs, 2, img_size, img_size]
        final_seg_pred = F.softmax(final_seg_pred, dim=1)  # [bs, 2, img_size, img_size]
        if test_mode:
            # [bs, img_size, img_size]
            final_seg_pred = final_seg_pred[:, 1, :, :]
            # final_seg_pred = (final_seg_pred[:, 1, :, :] + 1 - final_seg_pred[:, 0, :, :]) / 2
        return final_seg_pred

    def encode_text(self, text, adapt_text=True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(
            cast_dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        out_features = []
        for i in range(12):
            x, attn = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )
            index = -1
            for j in range(self.n_groups):
                if i + 1 == self.text_levels[j]:
                    index = j
                    break
            if index != -1:
                adapt_out = self.text_adapter["lora_adapters"][index](x)
                adapt_out = (
                        adapt_out
                        * x.norm(dim=-1, keepdim=True)
                        / adapt_out.norm(dim=-1, keepdim=True)
                )
                x = self.text_adapter["m_t_w"][index](x, adapt_out)
                # x = x * (1 - self.text_adapt_weight) + adapt_out * self.text_adapt_weight
                out_features.append(x)

        indices = text.argmax(dim=-1)
        out_features = [t.permute(1, 0, 2) for t in out_features]
        out_features = [self.text_adapter["layer_norms"][i](t) for i, t in enumerate(out_features)]
        out_features = [t[torch.arange(t.shape[0]), indices] for t in out_features]
        return out_features
