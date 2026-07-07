# Phase2B Run Context

Current branch:

```text
phase2b_kgsoftprompt_ctx4_fromscratch
```

## Implemented code state

This branch implements Phase2B Hybrid hard-soft prompt with optional K-space regularization.

Main code commit:

```text
e039669 Add Phase2B hybrid K-space regularization
```

K-reg behavior:

```text
T_hard = hard prompt ensemble + text LoRA/adapters
T_soft = soft prompt path without text LoRA
T_main = normalize((1 - alpha) * T_hard + alpha * T_soft, dim=1)

L_kg = mean(1 - cosine(T_soft, T_hard.detach()))
L_k  = mean per-stage K-space distance, using detached W_K parameters

loss = loss_main + lambda_kg * L_kg + lambda_k * L_k
```

K-reg uses detached `vision_text_k` weights so `L_k` regularizes the soft contribution without directly updating W_K.
The normal Phase1/Hybrid loss path still trains image/text adapters normally.

## Completed runs kept for reporting

### Phase2B Hybrid alpha0.2, no K-reg

Run directory:

```text
runs/phase2b/phase2b_hybrid_alpha02_lkg1e2_lr5e5_train20_test6medical7to20_fromscratch
```

Best by 6-medical pixel AP:

```text
epoch 9
mean pixel AUC/AP = 89.87 / 39.54
mean image AUC/AP = 72.70 / 74.13
```

Compared with Phase1 best e9:

```text
Phase1 best mean pixel AUC/AP = 90.76 / 39.82
Phase1 best mean image AUC/AP = 73.80 / 75.06
```

Interpretation:

```text
Near miss. Improves ColonDB/ClinicDB/Kvasir, but hurts BrainMRI and Retina OCT.
```

### Phase2B Hybrid alpha0.1 + K-reg 5e-3

Run directory:

```text
runs/phase2b/phase2b_hybrid_alpha01_kreg5e3_lkg1e2_lr5e5_train20_test6medical7to15_fromscratch
```

Committed report artifacts:

```text
train.log
test.log
parsed_results.csv
key_ap_summary.csv
```

Best observed result:

```text
best 6-medical pixel mean: epoch 7 = 90.52 / 37.81
best Brain AP: epoch 8 = 41.83
```

Interpretation:

```text
Stable but too conservative overall. K-space stayed close to hard path, but the run did not recover Brain/Retina enough and did not beat Phase1 best.
```

### Phase2B Hybrid alpha0.2 + K-reg 2e-3

Script:

```text
run_phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_fromscratch_train_then_test.sh
```

Default settings:

```text
hybrid_alpha_max = 0.2
lambda_kg = 1e-2
lambda_k = 2e-3
soft_prompt_lr = 5e-5
soft_prompt_freeze_epochs = 3
grad_clip_norm = 1.0
train_epoch = 15
test_epochs = 7 8 9 10 11 12 13 14 15
test datasets = Brain Liver Retina Colon_clinicDB Colon_colonDB Colon_Kvasir
pixel_stride = 4
metric_thresholds = none
```

Rationale:

```text
alpha0.2 had the strongest mean AP among Phase2B runs but drifted too much for Brain/Retina.
alpha0.1 + K-reg 5e-3 was stable but too conservative.
lambda_k=2e-3 is the last rescue setting: enough K-space regularization to reduce drift, but lighter than 5e-3 so the soft prompt can still help Colon/Kvasir.
```

Gate:

```text
PASS if mean pixel AP > 39.82 or Brain AP >= 44 with mean AP close to Phase1.
FAIL if Brain AP stays < 44 and mean pixel AP does not beat 39.82 through epochs 7-15.
```

Run command:

```bash
cd /home/ai4/caohuy/ACD-CLIP-base-new-phase1
conda activate torchhuy
bash run_phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_fromscratch_train_then_test.sh
```

Expected output directory:

```text
runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch
```

Committed report artifacts:

```text
train.log
test.log
parsed_results.csv
key_ap_summary.csv
```

Best by 6-medical pixel AP:

```text
epoch 10
mean pixel AUC/AP = 90.98 / 40.35
mean image AUC/AP = 71.65 / 70.71
```

Phase1 best reference:

```text
epoch 9
mean pixel AUC/AP = 90.76 / 39.82
mean image AUC/AP = 73.80 / 75.06
```

Per-dataset epoch 10 pixel-level AUC/AP:

```text
ColonDB    = 83.88 / 35.70
ClinicDB   = 89.06 / 57.46
Kvasir     = 88.40 / 63.45
BrainMRI   = 95.24 / 38.28
Liver CT   = 97.07 / 6.81
Retina OCT = 92.20 / 40.39
```

Per-dataset epoch 10 image-level AUC/AP:

```text
BrainMRI   = 79.87 / 93.57
Liver CT   = 59.24 / 47.86
Retina OCT = 75.84 / 70.69
```

Interpretation:

```text
Best Phase2B candidate for 6-medical pixel-level mean so far.
It beats Phase1 best in mean pixel AUC/AP by +0.22 / +0.53.
However, it hurts image-level mean and still underperforms Phase1 best on Brain pixel AP.
Use this run as the Phase2B pixel-level result, while Phase1 best remains the more balanced pixel+image model.
```
