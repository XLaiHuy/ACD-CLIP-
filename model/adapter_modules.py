import math

import torch
import torch.nn.functional as F
from torch import nn


class MLPAdapter(nn.Module):
    def __init__(self, c_in, c_out=768, hidden_size=512):
        super(MLPAdapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, c_out)
        )

    def forward(self, x):
        # x shape: [H * W, bs, c_in]
        x = self.fc(x)
        return x


class TextLoraAdapter(nn.Module):
    def __init__(self, c_in, c_out=768, r=16, alpha=2.0):
        super(TextLoraAdapter, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.r = r
        self.scale = alpha / r ** 0.5  # LoRA的缩放系数

        self.lora_A = nn.Parameter(torch.randn(c_in, r))
        self.lora_B = nn.Parameter(torch.randn(r, c_out))

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A)  # 使用Kaiming初始化A
        nn.init.normal_(self.lora_B, mean=0, std=0.02)  # 正态分布初始化B

    def forward(self, x):
        # x shape: [H * W, bs, c_in]
        lora_output = x @ self.lora_A @ self.lora_B * self.scale  # [H * W, bs, c_out]
        return lora_output


def _make_activation(name: str):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _init_conv_norm(module: nn.Module):
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv2d):
            nn.init.kaiming_uniform_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.BatchNorm2d):
            nn.init.ones_(submodule.weight)
            nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.Linear):
            nn.init.xavier_uniform_(submodule.weight)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
            self,
            c_in,
            c_out,
            kernel_size,
            use_bn=True,
            activation="silu",
            zero_init=False,
    ):
        super().__init__()
        self.dw = nn.Conv2d(
            c_in, c_in, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, groups=c_in, bias=False
        )
        self.norm = nn.BatchNorm2d(c_in) if use_bn else nn.Identity()
        self.act = _make_activation(activation)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=True)
        _init_conv_norm(self)
        if zero_init:
            nn.init.zeros_(self.pw.weight)
            nn.init.zeros_(self.pw.bias)

    def forward(self, x):
        return self.pw(self.act(self.norm(self.dw(x))))


class DynamicDepthwiseConv2d(nn.Module):
    def __init__(
            self,
            channels,
            kernel_size,
            num_experts=2,
            temperature=10.0,
            gate_hidden_ratio=0.25,
            activation="silu",
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("dynamic depthwise num_experts must be >= 1")
        if temperature <= 0:
            raise ValueError("dynamic depthwise temperature must be positive")
        hidden = max(1, int(channels * gate_hidden_ratio))
        self.num_experts = int(num_experts)
        self.temperature = float(temperature)
        self.experts = nn.ModuleList([
            nn.Conv2d(
                channels, channels, kernel_size=kernel_size, stride=1,
                padding=kernel_size // 2, groups=channels, bias=False
            )
            for _ in range(self.num_experts)
        ])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            _make_activation(activation),
            nn.Linear(hidden, self.num_experts),
        )
        self._last_pi = None
        _init_conv_norm(self)

    def forward(self, x):
        logits = self.gate(x)
        pi = F.softmax(logits.float() / self.temperature, dim=-1).to(dtype=x.dtype)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        weights = pi.transpose(0, 1).view(self.num_experts, x.shape[0], 1, 1, 1)
        out = (weights * expert_outputs).sum(dim=0)
        self._last_pi = pi.detach()
        return out


class DynamicDepthwiseSeparableConv2d(nn.Module):
    def __init__(
            self,
            c_in,
            c_out,
            kernel_size,
            num_experts=2,
            temperature=10.0,
            gate_hidden_ratio=0.25,
            use_bn=True,
            activation="silu",
            zero_init=False,
    ):
        super().__init__()
        self.dynamic_dw = DynamicDepthwiseConv2d(
            c_in,
            kernel_size=kernel_size,
            num_experts=num_experts,
            temperature=temperature,
            gate_hidden_ratio=gate_hidden_ratio,
            activation=activation,
        )
        self.norm = nn.BatchNorm2d(c_in) if use_bn else nn.Identity()
        self.act = _make_activation(activation)
        self.pw = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=True)
        _init_conv_norm(self.norm)
        _init_conv_norm(self.pw)
        if zero_init:
            nn.init.zeros_(self.pw.weight)
            nn.init.zeros_(self.pw.bias)

    def forward(self, x):
        return self.pw(self.act(self.norm(self.dynamic_dw(x))))

    @property
    def last_pi(self):
        return self.dynamic_dw._last_pi


class ConvLoraBlock(nn.Module):
    def __init__(
            self,
            c_in,
            c_out=768,
            lora_rank=16,
            lora_alpha=2.0,
            conv_lora_rank=8,
            conv_lora_alpha=2.0,
            conv_kernel_size=3,
            convlora_variant="standard",
            dynamic_dw_num_experts=2,
            dynamic_dw_temperature=10.0,
            dynamic_dw_gate_hidden_ratio=0.25,
            dynamic_dw_use_bn=True,
            dynamic_dw_activation="silu",
            dynamic_dw_zero_init=False,
    ):
        super(ConvLoraBlock, self).__init__()
        if convlora_variant not in ["standard", "depthwise_separable", "dynamic_depthwise_expert"]:
            raise ValueError(f"Unknown convlora_variant: {convlora_variant}")
        if dynamic_dw_activation not in ["relu", "silu"]:
            raise ValueError(f"Unknown dynamic_dw_activation: {dynamic_dw_activation}")
        self.convlora_variant = convlora_variant
        self.conv_kernel_size = conv_kernel_size
        # 缩放
        self.lora_scale = lora_alpha / lora_rank ** 0.5
        self.conv_lora_scale = conv_lora_alpha / conv_lora_rank

        # downsample
        self.lora_A = nn.Parameter(torch.randn(c_in, lora_rank))
        if convlora_variant == "standard":
            self.conv_lora_A = nn.Conv2d(lora_rank, conv_lora_rank, kernel_size=conv_kernel_size, stride=1,
                                         padding=conv_kernel_size // 2, bias=False)
        elif convlora_variant == "depthwise_separable":
            self.conv_lora_A = DepthwiseSeparableConv2d(
                lora_rank, conv_lora_rank, conv_kernel_size,
                use_bn=dynamic_dw_use_bn,
                activation=dynamic_dw_activation,
                zero_init=dynamic_dw_zero_init,
            )
        else:
            self.conv_lora_A = DynamicDepthwiseSeparableConv2d(
                lora_rank, conv_lora_rank, conv_kernel_size,
                num_experts=dynamic_dw_num_experts,
                temperature=dynamic_dw_temperature,
                gate_hidden_ratio=dynamic_dw_gate_hidden_ratio,
                use_bn=dynamic_dw_use_bn,
                activation=dynamic_dw_activation,
                zero_init=dynamic_dw_zero_init,
            )
        # upsample
        if convlora_variant == "standard":
            self.conv_lora_B = nn.Conv2d(conv_lora_rank, lora_rank, kernel_size=conv_kernel_size, stride=1,
                                         padding=conv_kernel_size // 2, bias=False)
        elif convlora_variant == "depthwise_separable":
            self.conv_lora_B = DepthwiseSeparableConv2d(
                conv_lora_rank, lora_rank, conv_kernel_size,
                use_bn=dynamic_dw_use_bn,
                activation=dynamic_dw_activation,
                zero_init=dynamic_dw_zero_init,
            )
        else:
            self.conv_lora_B = DynamicDepthwiseSeparableConv2d(
                conv_lora_rank, lora_rank, conv_kernel_size,
                num_experts=dynamic_dw_num_experts,
                temperature=dynamic_dw_temperature,
                gate_hidden_ratio=dynamic_dw_gate_hidden_ratio,
                use_bn=dynamic_dw_use_bn,
                activation=dynamic_dw_activation,
                zero_init=dynamic_dw_zero_init,
            )
        self.lora_B = nn.Parameter(torch.randn(lora_rank, c_out))

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.normal_(self.lora_B, mean=0, std=0.02)
        if self.convlora_variant == "standard":
            nn.init.kaiming_uniform_(self.conv_lora_A.weight)
            nn.init.kaiming_uniform_(self.conv_lora_B.weight)

    def forward(self, x):
        # x shape: [H * W, bs, c_in]
        patch_size, B = int(math.sqrt(x.shape[0])), x.shape[1]  # 假设输入是正方形的
        # Downsample
        down_lora_output = x @ self.lora_A  # [H * W, bs, lora_rank]
        down_lora_output = down_lora_output.permute(1, 2, 0).view(B, -1, patch_size,
                                                                  patch_size)  # [bs, lora_rank, H, W]
        up_lora_input = self.conv_lora_A(down_lora_output)  # [bs, conv_lora_rank, H, W]
        # Upsample
        up_lora_output = self.conv_lora_B(up_lora_input) * self.conv_lora_scale  # [bs, lora_rank, H, W]
        up_lora_output = up_lora_output.view(B, -1, patch_size * patch_size).permute(2, 0, 1)  # [H * W, bs, lora_rank]
        up_lora_output = up_lora_output @ self.lora_B * self.lora_scale  # [H * W, bs, c_out]
        return up_lora_output

    def get_dynamic_pis(self):
        pis = []
        for module in [self.conv_lora_A, self.conv_lora_B]:
            if isinstance(module, DynamicDepthwiseSeparableConv2d) and module.last_pi is not None:
                pis.append(module.last_pi)
        return pis


class ConvLoraAdapter(nn.Module):
    def __init__(
            self,
            c_in,
            c_out=768,
            lora_rank=16,
            lora_alpha=2.0,
            conv_lora_rank=8,
            conv_lora_alpha=2.0,
            conv_kernel_size_list=(3, 5),
            convlora_variant="standard",
            dynamic_dw_num_experts=2,
            dynamic_dw_temperature=10.0,
            dynamic_dw_gate_hidden_ratio=0.25,
            dynamic_dw_use_bn=True,
            dynamic_dw_activation="silu",
            dynamic_dw_zero_init=False,
    ):
        super(ConvLoraAdapter, self).__init__()
        kernel_size_list = conv_kernel_size_list
        self.kernel_size_list = list(kernel_size_list)
        self.convlora_variant = convlora_variant
        self.dynamic_dw_num_experts = int(dynamic_dw_num_experts)
        self.dynamic_dw_temperature = float(dynamic_dw_temperature)
        self._last_stats = {}
        self.conv_lora_blocks = nn.ModuleList([
            ConvLoraBlock(
                c_in=c_in,
                c_out=c_out,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                conv_lora_rank=conv_lora_rank,
                conv_lora_alpha=conv_lora_alpha,
                conv_kernel_size=kernel_size,
                convlora_variant=convlora_variant,
                dynamic_dw_num_experts=dynamic_dw_num_experts,
                dynamic_dw_temperature=dynamic_dw_temperature,
                dynamic_dw_gate_hidden_ratio=dynamic_dw_gate_hidden_ratio,
                dynamic_dw_use_bn=dynamic_dw_use_bn,
                dynamic_dw_activation=dynamic_dw_activation,
                dynamic_dw_zero_init=dynamic_dw_zero_init,
            ) for kernel_size in kernel_size_list
        ])
        self.fusion_conv = nn.Conv2d(len(kernel_size_list) * c_out, c_out, kernel_size=1, stride=1, padding=0,
                                     bias=False)

    def forward(self, x):
        # x [H * W, bs, c_in] [1369, 4, 1024]
        patch_size, B = int(math.sqrt(x.shape[0])), x.shape[1]
        outputs = [block(x).permute(1, 2, 0) for block in
                   self.conv_lora_blocks]  # 每个block输出 [H * W, bs, c_out] -> [bs, c_out, H * W]
        outputs = [out.view(B, -1, patch_size, patch_size) for out in outputs]  # [bs, c_out, H, W]
        branch_stats = {}
        for idx, out in enumerate(outputs):
            kernel = self.kernel_size_list[idx]
            branch_stats[f"branch{kernel}_norm"] = out.detach().float().norm().item()
        outputs = torch.cat(outputs, dim=1)
        # 特征融合
        outputs = self.fusion_conv(outputs)  # [bs, c_out, H, W]
        out_detached = outputs.detach().float()
        self._last_stats = {
            "convlora_variant": self.convlora_variant,
            "adapter_output_norm": out_detached.norm().item(),
            "adapter_output_absmax": out_detached.abs().max().item(),
            "adapter_output_finite": bool(torch.isfinite(out_detached).all().item()),
            **branch_stats,
        }
        if self.convlora_variant == "dynamic_depthwise_expert":
            pis = []
            for block in self.conv_lora_blocks:
                pis.extend(block.get_dynamic_pis())
            if pis:
                pi = torch.cat(pis, dim=0).float()
                entropy = -(pi * pi.clamp_min(1e-8).log()).sum(dim=-1)
                self._last_stats.update({
                    "dynamic_dw_num_experts": self.dynamic_dw_num_experts,
                    "dynamic_dw_temperature": self.dynamic_dw_temperature,
                    "pi_entropy_mean": entropy.mean().item(),
                    "pi_collapsed": bool((pi.max(dim=-1).values > 0.95).float().mean().item() > 0.5),
                })
                for expert_idx in range(pi.shape[-1]):
                    values = pi[:, expert_idx]
                    self._last_stats[f"pi_expert{expert_idx}_mean"] = values.mean().item()
                    self._last_stats[f"pi_expert{expert_idx}_min"] = values.min().item()
                    self._last_stats[f"pi_expert{expert_idx}_max"] = values.max().item()
        outputs = outputs.view(B, -1, patch_size * patch_size).permute(2, 0, 1)  # [H * W, bs, c_out]
        return outputs


class FourDirectionSS2D(nn.Module):
    """Dependency-free 4-direction spatial scan used by Phase 1B DFG.

    This is intentionally lightweight: it lets each patch aggregate context from
    left/right/top/bottom scans without pulling the VMamba CUDA dependency into
    the clean Phase 1 ablation repo.
    """

    def __init__(self, dim=768):
        super().__init__()
        self.direction_logits = nn.Parameter(torch.zeros(4))
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight)

    @staticmethod
    def _forward_average(x, dim):
        denom = torch.arange(1, x.shape[dim] + 1, device=x.device, dtype=x.dtype)
        shape = [1] * x.ndim
        shape[dim] = x.shape[dim]
        return x.cumsum(dim=dim) / denom.view(*shape)

    def forward(self, x):
        # x: [B, H, W, C]
        lr = self._forward_average(x, dim=2)
        rl = torch.flip(self._forward_average(torch.flip(x, dims=[2]), dim=2), dims=[2])
        tb = self._forward_average(x, dim=1)
        bt = torch.flip(self._forward_average(torch.flip(x, dims=[1]), dim=1), dims=[1])
        weights = F.softmax(self.direction_logits, dim=0).to(dtype=x.dtype)
        scanned = weights[0] * lr + weights[1] * rl + weights[2] * tb + weights[3] * bt
        return self.out_proj(scanned)


class DFGSS2DResidualBranch(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.ss2d = FourDirectionSS2D(dim)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, L, C], L must be a square grid.
        bsz, patch_num, channels = x.shape
        grid = int(math.sqrt(patch_num))
        if grid * grid != patch_num:
            raise ValueError(f"SS2D branch expects square patch tokens, got L={patch_num}")
        x_2d = x.view(bsz, grid, grid, channels)
        x_2d = self.pre_norm(x_2d)
        x_2d = self.act(self.in_proj(x_2d))
        x_2d = self.ss2d(x_2d)
        x_2d = self.post_norm(x_2d)
        return x_2d.mean(dim=(1, 2))


class ASPPImageFeatureAdapter(nn.Module):
    def __init__(self, c_in, c_hidden=256):
        super(ASPPImageFeatureAdapter, self).__init__()
        # 输入降维
        self.fc = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),
        )

        # 多尺度特征提取
        self.aspp1 = nn.Sequential(
            nn.Conv2d(c_hidden, c_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, stride=1, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),
        )

        # 全局特征提取分支
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(c_hidden, c_hidden, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(inplace=True),
        )

        # 特征拼接后的通道整合
        self.concat_conv = nn.Sequential(
            nn.Conv2d(c_hidden * 4, c_in, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x [H * W, bs, c_in]
        HW, B, C = x.shape
        x = x.permute(1, 2, 0)  # shape: [bs, c_in, H * W]
        H = int(math.sqrt(HW))
        x = x.view(B, C, H, H)

        # 输入降维
        x = self.fc(x)  # shape: [bs, c_out, H, W]
        # 多尺度特征提取
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)

        # 全局特征提取
        global_feat = self.global_avg_pool(x)  # shape: [bs, c_out, 1, 1]
        global_feat = self.global_conv(global_feat)  # shape: [bs, c_out, 1, 1]
        # 上采样到原始输入大小
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 特征拼接
        concat = torch.cat([aspp1, aspp2, aspp3, global_feat], dim=1)  # shape: [bs, c_out * 4, H, W]

        # 通道整合
        out = self.concat_conv(concat)  # shape: [bs, c_out, H, W]

        out = out.view(B, C, -1)  # shape: [bs, c_out, H * W]
        out = out.permute(2, 0, 1)  # shape: [H * W, bs, c_out]
        return out


if __name__ == '__main__':
    conv_lora_adapter = ConvLoraAdapter(c_in=1024, c_out=1024, lora_rank=16, lora_alpha=2.0, conv_lora_rank=8,
                                        conv_lora_alpha=2.0)
    x = torch.randn(1369, 4, 1024)  # [H * W, bs, c_in]
    print(x[:, 0, :].min())
    output = conv_lora_adapter(x)
    print(output[:, 0, :].min())
    print(output.shape)  # 应该是 [1369, 4, 768]
