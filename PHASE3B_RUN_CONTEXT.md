# Phase3B Run Context

Current branch:

```text
phase3b-stage-routing-consistency
```

## Implemented code state

This branch starts from the Phase2B Hybrid hard-soft prompt source and adds Phase3B stage routing consistency regularization.

Core Phase2B path remains:

```text
T_hard = hard prompt ensemble + text LoRA/adapters
T_soft = soft prompt path without text LoRA
T_main = normalize((1 - alpha) * T_hard + alpha * T_soft, dim=1)

L_kg = mean(1 - cosine(T_soft, T_hard.detach()))
L_k  = K-space consistency using detached W_K parameters
```

Phase3B adds an optional stage routing loss:

```text
L_stage = consistency between final DFG routing weights from T_main and T_hard
```

Important implementation details:

```text
- Stage routing compares the final routing weights actually used after DFG fusion.
- For weight_residual fusion, final weights are:
  (1 - beta) * weights_gap + beta * weights_ss2d
- detach_qk=True detaches both W_Q/W_K parameters and the projected inputs for this auxiliary loss.
- detach_visual=True detaches visual features for this auxiliary loss.
- stage_consistency_update_soft_only=True computes T_main for stage loss with T_hard detached,
  so this auxiliary term regularizes the soft contribution instead of pulling the hard branch.
- Setting lambda_stage=0 or stage_consistency_loss=none keeps Phase2B behavior.
```

New train flags:

```text
--lambda_stage
--stage_consistency_loss {none,js,js_margin}
--stage_consistency_margin
--stage_consistency_update_soft_only
--stage_consistency_detach_visual
--stage_consistency_detach_qk
```

Smoke coverage:

```text
debug_phase3b_stagecons_smoke.py
```

The smoke test checks:

```text
- disabled stage loss returns zero loss/stats
- enabled stage loss is finite
- stage routing weights sum to 1
- with detach flags, soft text receives gradients
- with detach flags, visual features and W_Q/W_K do not receive stage-loss gradients
```

## Completed Phase3B run kept for reporting

Run directory:

```text
runs/phase3b/phase3b_stagecons_alpha02_kreg2e3_lstage5e4_m002_train15
```

Committed report artifacts:

```text
train.log
test.log
parsed_results.csv
key_ap_summary.csv
```

Settings:

```text
hybrid_alpha_max = 0.2
lambda_kg = 1e-2
lambda_k = 2e-3
lambda_stage = 5e-4
stage_consistency_loss = js_margin
stage_consistency_margin = 0.02
stage_consistency_update_soft_only = True
stage_consistency_detach_visual = True
stage_consistency_detach_qk = True
soft_prompt_lr = 5e-5
train_epoch = 15
test_epochs = 7 8 9 10 11 12 13 14 15
test datasets = Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir
pixel_stride = 4
metric_thresholds = none
```

Best by 6-medical pixel AP:

```text
epoch 7
mean pixel AUC/AP = 90.27 / 37.93
mean image AUC/AP = 73.26 / 72.23
```

Per-dataset epoch 7 pixel-level AUC/AP:

```text
ColonDB    = 84.02 / 32.44
ClinicDB   = 87.95 / 50.06
Kvasir     = 86.77 / 54.69
BrainMRI   = 94.95 / 45.63
Liver CT   = 96.07 / 5.24
Retina OCT = 91.88 / 39.51
```

Per-dataset epoch 7 image-level AUC/AP:

```text
BrainMRI   = 81.53 / 94.74
Liver CT   = 64.48 / 52.11
Retina OCT = 73.78 / 69.83
```

Interpretation:

```text
Phase3B stage consistency with js_margin=0.02 did not beat Phase2B or Phase1.
The margin made the auxiliary term effectively inactive:
mean_stage_loss = 0.0
weighted_stage_loss = 0.0
stage_active_fraction = 0.0
mean_stage_js was only about 0.0002-0.0004

This run is useful as a negative ablation:
the implementation is stable, but the chosen margin is too high for the observed stage-routing JS scale.
```

## Current run intentionally excluded from this commit

Do not use or commit logs/checkpoints from:

```text
runs/phase3b/phase3b_stagecons_alpha02_kreg2e3_lstage1e2_js_train15
```

Reason:

```text
This is the newer/current JS no-margin run requested after the previous Phase3B attempt.
Its train/test logs should not be committed until the run is complete and reviewed.
```
