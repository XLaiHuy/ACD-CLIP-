# Phase3A Dynamic Depthwise Conv-LoRA Context

Branch: `phase3a_dynamic_dw_convlora`

Base context:
- Built on the Phase2B hybrid code branch.
- Goal: test whether replacing Conv-LoRA spatial convs with a lightweight dynamic depthwise expert improves medical zero-shot transfer.
- This run is train-from-scratch on VisA, not fine-tuning from a Phase1/Phase2 checkpoint.

## Architecture Change

Phase3A keeps the existing Conv-LoRA macro-structure:

```text
3x3 Conv-LoRA branch
5x5 Conv-LoRA branch
concat branches
fixed 1x1 fusion conv
```

Only the internal convolution inside each Conv-LoRA encoder/decoder is changed when:

```text
--convlora_variant dynamic_depthwise_expert
```

Dynamic part:

```text
input x
-> gate(x) produces expert attention pi
-> depthwise expert outputs DWConv_m(x)
-> weighted sum sum_m pi_m * DWConv_m(x)
-> BN/activation
-> fixed pointwise 1x1 Conv2d
```

Important:
- Dynamic is applied only to the depthwise convolution.
- Pointwise 1x1 convolution remains a normal fixed Conv2d.
- This is not full dense dynamic convolution and not 3x3-vs-5x5 branch gating.
- External dynamic/depthwise repos were used as conceptual references only; no code was imported or copied.

## Run

Run directory:

```text
runs/phase3a/phase3a_dyn_dwexpert_convlora_k35_e2_t30_h0125_alpha02_kreg2e3_train20_bf16
```

Script:

```text
run_phase3a_dyn_dwexpert_convlora_k35_e2_alpha02_kreg2e3_train20_bf16.sh
```

Main settings:

```text
dataset=VisA
epoch=20
test_epochs=7..20
n_groups=3
dfg_mode=attn
dfg_attn_tau=8.0
use_ss2d_dfg=True
dfg_ss2d_fusion=weight_residual
dfg_beta_schedule=warmup010
dfg_beta_target=0.10
text_adapt_weight=0.2

prompt_mode=hybrid
hybrid_alpha_max=0.2
lambda_kg=1e-2
lambda_k=2e-3
soft_prompt_lr=5e-5

convlora_variant=dynamic_depthwise_expert
dynamic_dw_num_experts=2
dynamic_dw_temperature=30.0
dynamic_dw_gate_hidden_ratio=0.125
dynamic_dw_use_bn=True
dynamic_dw_activation=silu
dynamic_dw_zero_init=False

amp=True
amp_dtype=bfloat16
batch_size=6
num_workers=6
grad_checkpointing=True
grad_clip_norm=1.0
```

Test protocol:

```text
6 medical datasets
batch_size=8
num_workers=6
pixel_stride=4
metric_thresholds=none
epochs=7..20
```

## Result Summary

Training stability:

```text
epoch 20 non_finite_loss=0
epoch 20 non_finite_grad=0
```

Dynamic gate behavior at epoch 20:

```text
image_stage1_pi_entropy_mean=0.6932
image_stage2_pi_entropy_mean=0.6927
image_stage3_pi_entropy_mean=0.6922
pi_collapsed=False
```

For 2 experts, max entropy is about 0.693, so gates stayed almost uniform. The run was numerically stable but the dynamic expert did not learn a clearly useful expert selection.

Best 6-medical mean pixel AP is epoch 14:

```text
mean pixel AUC/AP = 88.94 / 32.46
mean image AUC/AP = 71.85 / 72.17
```

Epoch 14 per-dataset metrics:

```text
ColonDB     pixel AUC/AP = 81.50 / 25.29
ClinicDB    pixel AUC/AP = 84.00 / 42.05
Kvasir      pixel AUC/AP = 84.30 / 51.81
BrainMRI    pixel AUC/AP = 94.03 / 27.84, image AUC/AP = 76.11 / 92.77
Liver CT    pixel AUC/AP = 95.83 / 4.70,  image AUC/AP = 60.72 / 47.91
Retina OCT  pixel AUC/AP = 94.01 / 43.07, image AUC/AP = 78.72 / 75.84
```

Best Brain AP is epoch 7:

```text
Brain pixel AUC/AP = 94.51 / 31.96
Brain image AUC/AP = 75.03 / 92.31
```

Comparison to selected previous local bests:

```text
Mainbase e10:      mean pixel AUC/AP = 90.25 / 38.21, mean image AUC/AP = 73.80 / 74.36
Phase1 best e9:    mean pixel AUC/AP = 90.76 / 39.82, mean image AUC/AP = 73.80 / 75.06
Phase2B best e10:  mean pixel AUC/AP = 90.98 / 40.35, mean image AUC/AP = 71.65 / 70.71
Phase3A best e14:  mean pixel AUC/AP = 88.94 / 32.46, mean image AUC/AP = 71.85 / 72.17
```

Conclusion:
- Phase3A dynamic depthwise expert is a negative ablation for medical zero-shot transfer.
- The implementation is stable and finite under bf16, but it significantly hurts pixel AP versus Phase1 and Phase2B.
- Do not use this Phase3A run as the paper best.

## Artifacts Committed

The commit includes:
- architecture/source changes
- train/test script
- smoke test
- `train.log`
- `test.log`
- `parsed_results.csv`
- `key_ap_summary.csv`

Checkpoints are intentionally not committed in this context commit.
