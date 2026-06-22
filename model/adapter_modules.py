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


class TextDoraAdapter(nn.Module):
    def __init__(self, c_in, c_out=768, r=16, alpha=2.0):
        super(TextDoraAdapter, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.r = r
        self.scale = alpha / (r ** 0.5)

        # Base weight (frozen)
        self.weight = nn.Parameter(
            torch.eye(c_in, c_out) if c_in == c_out else torch.randn(c_in, c_out) * 0.02,
            requires_grad=False
        )

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(c_in, r))
        self.lora_B = nn.Parameter(torch.zeros(r, c_out))  # Initialize with zeros

        # Learnable magnitude parameter m
        self.m = nn.Parameter(self.weight.norm(p=2, dim=0, keepdim=True))

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A)
        # lora_B is already initialized to 0

    def forward(self, x):
        # x shape: [H * W, bs, c_in]
        delta_W = (self.lora_A @ self.lora_B) * self.scale  # [c_in, c_out]
        W_full = self.weight + delta_W                      # [c_in, c_out]
        
        # Column-wise L2 norm normalization
        W_norm = W_full / W_full.norm(p=2, dim=0, keepdim=True)
        W_dora = self.m * W_norm                           # [c_in, c_out]
        
        return x @ W_dora                                  # [H * W, bs, c_out]



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
    ):
        super(ConvLoraBlock, self).__init__()
        # 缩放
        self.lora_scale = lora_alpha / lora_rank ** 0.5
        self.conv_lora_scale = conv_lora_alpha / conv_lora_rank

        # downsample
        self.lora_A = nn.Parameter(torch.randn(c_in, lora_rank))
        self.conv_lora_A = nn.Conv2d(lora_rank, conv_lora_rank, kernel_size=conv_kernel_size, stride=1,
                                     padding=conv_kernel_size // 2, bias=False)
        # upsample
        self.conv_lora_B = nn.Conv2d(conv_lora_rank, lora_rank, kernel_size=conv_kernel_size, stride=1,
                                     padding=conv_kernel_size // 2, bias=False)
        self.lora_B = nn.Parameter(torch.randn(lora_rank, c_out))

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.normal_(self.lora_B, mean=0, std=0.02)
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


class ConvLoraAdapter(nn.Module):
    def __init__(
            self,
            c_in,
            c_out=768,
            lora_rank=16,
            lora_alpha=2.0,
            conv_lora_rank=8,
            conv_lora_alpha=2.0,
            conv_kernel_size_list=(3, 5)
    ):
        super(ConvLoraAdapter, self).__init__()
        kernel_size_list = conv_kernel_size_list
        self.conv_lora_blocks = nn.ModuleList([
            ConvLoraBlock(
                c_in=c_in,
                c_out=c_out,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                conv_lora_rank=conv_lora_rank,
                conv_lora_alpha=conv_lora_alpha,
                conv_kernel_size=kernel_size
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
        outputs = torch.cat(outputs, dim=1)
        # 特征融合
        outputs = self.fusion_conv(outputs)  # [bs, c_out, H, W]
        outputs = outputs.view(B, -1, patch_size * patch_size).permute(2, 0, 1)  # [H * W, bs, c_out]
        return outputs


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


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, dim=1)


class DynamicDepthwiseConv2d(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, K=4, temperature=30.0):
        super(DynamicDepthwiseConv2d, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.K = K
        self.temperature = temperature
        
        self.attention = attention2d(channels, 0.25, K, temperature)
        
        # K base filters: shape [K, channels, 1, kernel_size, kernel_size]
        self.weight = nn.Parameter(
            torch.randn(K, channels, 1, kernel_size, kernel_size), 
            requires_grad=True
        )
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(K, channels), requires_grad=True)
        else:
            self.bias_param = None
            
        self.init_weights()

    def init_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        # x shape: [B, C, H, W]
        batch_size, channels, height, width = x.size()
        
        # Get attention weights: [B, K]
        softmax_attention = self.attention(x)
        
        outputs = []
        for i in range(batch_size):
            # Combine base kernels for sample i
            # self.weight has shape: [K, C, 1, k, k]
            # softmax_attention[i] has shape: [K]
            w_i = (softmax_attention[i].view(self.K, 1, 1, 1, 1) * self.weight).sum(dim=0)
            
            if self.bias_param is not None:
                b_i = (softmax_attention[i].view(self.K, 1) * self.bias_param).sum(dim=0)
            else:
                b_i = None
                
            # Perform depthwise convolution for sample i (groups = channels)
            out_i = F.conv2d(
                x[i:i+1],
                weight=w_i,
                bias=b_i,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=channels
            )
            outputs.append(out_i)
            
        return torch.cat(outputs, dim=0)


class DynamicDepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, K=4, temperature=30.0):
        super(DynamicDepthwiseSeparableConv2d, self).__init__()
        # 1. Dynamic depthwise conv
        self.depthwise = DynamicDepthwiseConv2d(
            channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            K=K,
            temperature=temperature
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_act = nn.SiLU(inplace=True)

        # 2. Pointwise 1x1 static conv
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.pw_act = nn.SiLU(inplace=True)

    def forward(self, x):
        # DW -> BN -> SiLU -> PW -> BN -> SiLU
        x = self.dw_act(self.dw_bn(self.depthwise(x)))
        x = self.pw_act(self.pw_bn(self.pointwise(x)))
        return x


class DynamicDepthwiseConvLoraBlock(nn.Module):
    def __init__(
            self,
            c_in,
            c_out=768,
            lora_rank=16,
            lora_alpha=2.0,
            conv_lora_rank=8,
            conv_lora_alpha=2.0,
            conv_kernel_size=3,
    ):
        super(DynamicDepthwiseConvLoraBlock, self).__init__()
        self.lora_scale = lora_alpha / lora_rank ** 0.5
        self.conv_lora_scale = conv_lora_alpha / conv_lora_rank

        self.lora_A = nn.Parameter(torch.randn(c_in, lora_rank))
        self.conv_lora_A = DynamicDepthwiseSeparableConv2d(
            in_channels=lora_rank, 
            out_channels=conv_lora_rank, 
            kernel_size=conv_kernel_size, 
            padding=conv_kernel_size // 2
        )
        self.conv_lora_B = DynamicDepthwiseSeparableConv2d(
            in_channels=conv_lora_rank, 
            out_channels=lora_rank, 
            kernel_size=conv_kernel_size, 
            padding=conv_kernel_size // 2
        )
        self.lora_B = nn.Parameter(torch.randn(lora_rank, c_out))

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A)
        nn.init.normal_(self.lora_B, mean=0, std=0.02)

    def forward(self, x):
        patch_size, B = int(math.sqrt(x.shape[0])), x.shape[1]
        
        down_lora_output = x @ self.lora_A  # [H * W, bs, lora_rank]
        down_lora_output = down_lora_output.permute(1, 2, 0).view(B, -1, patch_size, patch_size)  # [bs, lora_rank, H, W]
        up_lora_input = self.conv_lora_A(down_lora_output)  # [bs, conv_lora_rank, H, W]
        
        up_lora_output = self.conv_lora_B(up_lora_input) * self.conv_lora_scale  # [bs, lora_rank, H, W]
        up_lora_output = up_lora_output.view(B, -1, patch_size * patch_size).permute(2, 0, 1)  # [H * W, bs, lora_rank]
        up_lora_output = up_lora_output @ self.lora_B * self.lora_scale  # [H * W, bs, c_out]
        return up_lora_output


class DynamicDepthwiseConvLoraAdapter(nn.Module):
    def __init__(
            self,
            c_in,
            c_out=768,
            lora_rank=16,
            lora_alpha=2.0,
            conv_lora_rank=8,
            conv_lora_alpha=2.0,
            conv_kernel_size_list=(3, 5)
    ):
        super(DynamicDepthwiseConvLoraAdapter, self).__init__()
        kernel_size_list = conv_kernel_size_list
        self.conv_lora_blocks = nn.ModuleList([
            DynamicDepthwiseConvLoraBlock(
                c_in=c_in,
                c_out=c_out,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                conv_lora_rank=conv_lora_rank,
                conv_lora_alpha=conv_lora_alpha,
                conv_kernel_size=kernel_size
            ) for kernel_size in kernel_size_list
        ])
        self.fusion_conv = nn.Conv2d(len(kernel_size_list) * c_out, c_out, kernel_size=1, stride=1, padding=0,
                                     bias=False)

    def forward(self, x):
        patch_size, B = int(math.sqrt(x.shape[0])), x.shape[1]
        outputs = [block(x).permute(1, 2, 0) for block in self.conv_lora_blocks]
        outputs = [out.view(B, -1, patch_size, patch_size) for out in outputs]
        outputs = torch.cat(outputs, dim=1)
        outputs = self.fusion_conv(outputs)
        outputs = outputs.view(B, -1, patch_size * patch_size).permute(2, 0, 1)
        return outputs


if __name__ == '__main__':
    conv_lora_adapter = ConvLoraAdapter(c_in=1024, c_out=1024, lora_rank=16, lora_alpha=2.0, conv_lora_rank=8,
                                        conv_lora_alpha=2.0)
    x = torch.randn(1369, 4, 1024)  # [H * W, bs, c_in]
    print("ConvLoraAdapter input min:", x[:, 0, :].min().item())
    output = conv_lora_adapter(x)
    print("ConvLoraAdapter output min:", output[:, 0, :].min().item())
    print("ConvLoraAdapter output shape:", output.shape)  # 应该是 [1369, 4, 768]

    dynamic_adapter = DynamicDepthwiseConvLoraAdapter(c_in=1024, c_out=1024, lora_rank=16, lora_alpha=2.0, conv_lora_rank=8,
                                                      conv_lora_alpha=2.0)
    output_dyn = dynamic_adapter(x)
    print("DynamicDepthwiseConvLoraAdapter output shape:", output_dyn.shape)

