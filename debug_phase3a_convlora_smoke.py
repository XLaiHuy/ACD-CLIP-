import torch

from model.adapter_modules import ConvLoraAdapter, DynamicDepthwiseSeparableConv2d


def check_variant(variant: str):
    torch.manual_seed(0)
    adapter = ConvLoraAdapter(
        c_in=1024,
        c_out=1024,
        lora_rank=16,
        lora_alpha=2.0,
        conv_lora_rank=8,
        conv_lora_alpha=2.0,
        conv_kernel_size_list=(3, 5),
        convlora_variant=variant,
        dynamic_dw_num_experts=2,
        dynamic_dw_temperature=10.0,
        dynamic_dw_gate_hidden_ratio=0.25,
        dynamic_dw_use_bn=True,
        dynamic_dw_activation="silu",
    )
    x = torch.randn(1369, 2, 1024)
    with torch.no_grad():
        y = adapter(x)
    assert y.shape == x.shape, f"{variant}: expected {x.shape}, got {y.shape}"
    assert torch.isfinite(y).all(), f"{variant}: output has non-finite values"
    if variant == "dynamic_depthwise_expert":
        pis = []
        for block in adapter.conv_lora_blocks:
            for module in [block.conv_lora_A, block.conv_lora_B]:
                assert isinstance(module, DynamicDepthwiseSeparableConv2d)
                pi = module.last_pi
                assert pi is not None, f"{variant}: missing dynamic pi"
                assert pi.shape == (2, 2), f"{variant}: expected pi [2, 2], got {pi.shape}"
                assert torch.allclose(pi.sum(dim=-1), torch.ones(2), atol=1e-5), (
                    f"{variant}: pi rows do not sum to 1"
                )
                assert torch.isfinite(pi).all(), f"{variant}: pi has non-finite values"
                pis.append(pi)
        print(f"{variant}: ok shape={tuple(y.shape)} pi_mean={torch.cat(pis).mean(dim=0).tolist()}")
    else:
        print(f"{variant}: ok shape={tuple(y.shape)}")


def main():
    for variant in ["standard", "depthwise_separable", "dynamic_depthwise_expert"]:
        check_variant(variant)


if __name__ == "__main__":
    main()
